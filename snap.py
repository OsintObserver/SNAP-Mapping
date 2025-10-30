from __future__ import annotations
import os
import io
import sys
import zipfile
import time
import math
import json
import shutil
import warnings
from typing import Tuple, Dict
import pandas as pd
import geopandas as gpd
import requests
from tqdm import tqdm
import us

# ----------------------------
# Config
# ----------------------------
OUTDIR = os.path.join(os.path.expanduser("~"), "Documents", "SNAP", "snap_output")
DATADIR = os.path.join(os.path.expanduser("~"), "Documents", "SNAP", "snap_data_cache")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)

# Source URLs (stable Census / OMB endpoints & files)
CENSUS_API_BASE = "https://api.census.gov/data/2024/acs/acs1/subject"

# Subject table S2201 vars (estimate+MOE of % households receiving SNAP)
ACS_VARS = ["NAME", "S2201_C04_001E", "S2201_C04_001M", "S2201_C02_001E"]

# OMB List 2 (Principal Cities) — 2023 delineations (XLSX)
OMB_PCITIES_XLSX = "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2023/delineation-files/list2_2023.xlsx"

# Gazetteer: 2024 place gazetteer (national)
GAZETTEER_PLACES_TXT = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_place_national.zip"

# Cartographic Boundary: Places (nationwide) 1:500k — 2024
PLACES_CB_500K = "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_place_500k.zip"

# ----------------------------
# Helpers
# ----------------------------

def download(url: str, dest_path: str, desc: str = "") -> str:
    if os.path.exists(dest_path):
        return dest_path
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=desc or os.path.basename(dest_path))
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        bar.close()
    return dest_path

def read_principal_cities_metropolitan(path_xls: str) -> pd.DataFrame:
    # The file layout has “List 2. Principal Cities of Metropolitan and Micropolitan Statistical Areas”
    # We’ll parse and extract: CBSA Code, CBSA Title, LSAD Type (Metropolitan/Micropolitan), Principal City, State
    # Different years sometimes tweak column names; handle flexibly.
    # The sheet has two introductory rows before the real header row.
    # Read with header=2 (zero-based) so the third row becomes the DataFrame columns.
    try:
        df = pd.read_excel(path_xls, header=2, engine="openpyxl")
    except Exception:
        # fallback to a generic read if header=2 fails
        df = pd.read_excel(path_xls, engine="openpyxl")

    # Normalize columns map: lowercase -> original
    cols = {str(c).strip().lower(): c for c in df.columns}

    # Helper to find a column by trying multiple candidate substrings
    def find_col(candidates):
        for cand in candidates:
            cand = cand.lower()
            for k, v in cols.items():
                if cand in k:
                    return v
        return None

    # Expected column candidates (covers common variants)
    cbsa_code_col = find_col(["cbsa code", "cbsa"])
    cbsa_title_col = find_col(["cbsa title", "cbsa name", "cbsa"])
    area_type_col = find_col(["metropolitan/micropolitan", "metropolitan/micropolitan statistical area", "metropolitan"])
    princ_city_col = find_col(["principal city name", "principal city", "principal city name"])
    # The sheet uses FIPS codes for state and place; capture those columns
    state_col = find_col(["fips state code", "fips state", "state fips", "state code", "state"])
    place_col = find_col(["fips place code", "fips place", "place code", "place fips", "place"])

    missing = [name for name, val in [
        ("CBSA Code", cbsa_code_col), ("CBSA Title", cbsa_title_col),
        ("Area Type", area_type_col), ("Principal City", princ_city_col),
        ("State", state_col)
    ] if val is None]
    if missing:
        available = ", ".join(list(cols.keys())[:50])
        raise KeyError(f"Required columns not found in OMB List 2: {missing}. Available columns (sample): {available}")

    df = df[[cbsa_code_col, cbsa_title_col, area_type_col, princ_city_col, state_col]].copy()
    df.columns = ["CBSA_CODE", "CBSA_TITLE", "AREA_TYPE", "CITY", "STATE"]
    # Filter to Metropolitan Statistical Area principal cities
    df = df[df["AREA_TYPE"].str.contains("Metropolitan", case=False, na=False)].copy()
    # Deduplicate (a city may be principal for multiple CBSAs—keep all rows; we’ll de-dup by place later)
    # Standardize for matching
    # Reusable normalization for city names (strip common suffixes, punctuation, parentheses)
    def _normalize_city(name: str) -> str:
        if pd.isna(name):
            return ""
        s = str(name)
        # remove parenthetical content: "Name (balance)" -> "Name"
        s = pd.Series([s]).str.replace(r"\s*\(.*\)", "", regex=True).iloc[0]
        # common suffixes
        s = pd.Series([s]).str.replace(r"\b(city|cdp|town|village|borough|municipality|municipal|municipio)\b", "", regex=True, flags=0).iloc[0]
        # normalize punctuation and whitespace
        s = pd.Series([s]).str.replace(r"[.,',\"]", "", regex=True).iloc[0]
        s = " ".join(s.split()).strip().lower()
        return s

    df["CITY_NORM"] = df["CITY"].apply(_normalize_city)
    # Convert FIPS state code or existing values to USPS abbreviation robustly
    def _to_state_abbr(val):
        s = str(val).strip()
        if s == "" or s.lower() in ("nan", "none", "na"):
            return pd.NA
        # numeric FIPS code (handle floats like '48.0' or numeric types)
        # Map of territory FIPS codes to postal
        territory_fips = {
            "72": "PR",  # Puerto Rico
            "78": "VI",  # U.S. Virgin Islands
            "66": "GU",  # Guam
            "60": "AS",  # American Samoa
            "69": "MP",  # Northern Mariana Islands
        }
        try:
            if isinstance(val, (int, float)) or str(val).replace('.', '', 1).isdigit():
                f = str(int(float(s))).zfill(2)
                if f in territory_fips:
                    return territory_fips[f]
                # us.states doesn't provide by_fips in all versions; build a small map
                fips_map = {st.fips: st for st in us.states.STATES}
                st = fips_map.get(f)
                return st.abbr if st else pd.NA
        except Exception:
            pass
        # already a 2-letter postal code
        if len(s) == 2 and s.isalpha():
            return s.upper()
        # try extracting digits then mapping
        digits = "".join(ch for ch in s if ch.isdigit())
        if digits:
            f = digits.zfill(2)
            fips_map = {st.fips: st for st in us.states.STATES}
            st = fips_map.get(f)
            if st:
                return st.abbr
        # try state lookup by name
        st = us.states.lookup(s)
        if st:
            return st.abbr
        # fallback: uppercase raw value (may match gazetteer USPS if already abbreviation)
        return s.upper()

    df["STATE_ABBR"] = df["STATE"].apply(_to_state_abbr)
    return df

def read_gazetteer_places(path_txt: str) -> pd.DataFrame:
    # Gazetteer is pipe-delimited, with headers in the first line
    g = pd.read_csv(path_txt, sep="\t", dtype=str, low_memory=False)
    # Expected columns include: "GEOID","NAME","USPS","ALAND","AWATER", etc.
    # Normalize
    g.columns = [c.strip().upper() for c in g.columns]
    # Build normalized name and state
    # Reuse normalization logic for gazetteer names
    def _normalize_city_series(s: pd.Series) -> pd.Series:
        # remove parenthetical content
        s = s.str.replace(r"\s*\(.*\)", "", regex=True)
        s = s.str.replace(r"\b(city|cdp|town|village|borough|municipality|municipal|municipio)\b", "", regex=True)
        s = s.str.replace(r"[.,',\"]", "", regex=True)
        s = s.str.replace(r"\s+", " ", regex=True)
        return s.str.strip().str.lower()

    g["CITY_NORM"] = _normalize_city_series(g["NAME"].astype(str))
    g["STATE_ABBR"] = g["USPS"].str.strip()
    # GEOID is the 7-digit place GEOID (SS + PPPPP)
    return g[["GEOID", "NAME", "USPS", "CITY_NORM", "STATE_ABBR"]].drop_duplicates()

def derive_state_from_cbsa(title: str) -> str | None:
    """Derive a likely state USPS code from a CBSA title.

    Examples:
    - "Washington-Arlington-Alexandria, DC-VA-MD-WV" -> 'DC'
    - returns None when not derivable
    """
    if not title or pd.isna(title):
        return None
    parts = [p.strip() for p in str(title).split(',') if p.strip()]
    if len(parts) > 1:
        candidate = parts[-1]
    else:
        candidate = parts[0]
    for token in candidate.replace(';', ',').replace('/', '-').split('-'):
        tok = token.strip().upper()
        # Accept DC explicitly (us.states.lookup may not return DC by token)
        if tok == 'DC':
            return 'DC'
        if len(tok) == 2 and (us.states.lookup(tok)):
            return tok
    return None

def left_join_city_to_gazetteer(pcities: pd.DataFrame, gaz: pd.DataFrame) -> pd.DataFrame:
    # Primary join on (CITY_NORM, STATE_ABBR)
    m = pcities.merge(gaz, on=["CITY_NORM", "STATE_ABBR"], how="left", suffixes=("", "_GAZ"))
    # Flag unmatched
    m["MATCHED"] = ~m["GEOID"].isna()
    # Attempt fuzzy fallbacks for unmatched rows
    unmatched = m[m["GEOID"].isna()].copy()
    if len(unmatched):
        # Track updates to apply in bulk
        updates = []

        # 1) Try matching on CITY_NORM only when there's a single candidate in gazetteer
        for idx, row in unmatched.iterrows():
            city = row["CITY_NORM"]
            state = row.get("STATE_ABBR")
            if pd.isna(city) or city == "":
                continue
            candidates = gaz[gaz["CITY_NORM"] == city]
            if len(candidates) == 1:
                updates.append({
                    "idx": idx,
                    "GEOID": candidates.iloc[0]["GEOID"],
                    "NAME": candidates.iloc[0]["NAME"],
                    "USPS": candidates.iloc[0]["USPS"],
                    "MATCHED": True
                })
                continue
            # 2) startswith / contains heuristics
            cand2 = gaz[gaz["CITY_NORM"].str.startswith(city) | gaz["CITY_NORM"].str.contains(city)]
            if len(cand2) == 1:
                updates.append({
                    "idx": idx,
                    "GEOID": cand2.iloc[0]["GEOID"],
                    "NAME": cand2.iloc[0]["NAME"],
                    "USPS": cand2.iloc[0]["USPS"],
                    "MATCHED": True
                })
                continue
            # 3) try matching on first token (e.g., "Newark" from "Newark Township")
            first = city.split()[0]
            if first:
                cand3 = gaz[gaz["CITY_NORM"].str.split().str[0] == first]
                # prefer candidates in same state if available
                if pd.notna(state) and state in cand3["STATE_ABBR"].values:
                    cand_same = cand3[cand3["STATE_ABBR"] == state]
                    if len(cand_same) >= 1:
                        chosen = cand_same.iloc[0]
                        updates.append({
                            "idx": idx,
                            "GEOID": chosen["GEOID"],
                            "NAME": chosen["NAME"],
                            "USPS": chosen["USPS"],
                            "MATCHED": True
                        })
                        continue
                if len(cand3) == 1:
                    chosen = cand3.iloc[0]
                    updates.append({
                        "idx": idx,
                        "GEOID": chosen["GEOID"],
                        "NAME": chosen["NAME"],
                        "USPS": chosen["USPS"],
                        "MATCHED": True
                    })
                    continue

        # Apply updates in bulk
        for update in updates:
            idx = update.pop("idx")
            for col, value in update.items():
                m.at[idx, col] = value
    return m

def fetch_acs_place_s2201() -> pd.DataFrame:
    # Pull all places nationwide then filter to principal cities by GEOID
    params = {
        "get": ",".join(ACS_VARS),
        "for": "place:*",
        "in": "state:*"
    }
    # Use a session with retries and a User-Agent for polite requests
    session = requests.Session()
    try:
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        session.mount('https://', adapter)
    except Exception:
        # adapters may not be available in this environment; continue without mounting
        pass
    headers = {"User-Agent": "snap-principal-cities/1.0 (+https://github.com/)"}  
    r = session.get(CENSUS_API_BASE, params=params, timeout=60, headers=headers)
    r.raise_for_status()
    js = r.json()
    header = js[0]
    rows = js[1:]

    # Normalize header case and map to actual names
    headers = [str(h) for h in header]
    hmap = {h.lower(): h for h in headers}
    needed = ["name", "s2201_c04_001e", "s2201_c04_001m", "state", "place"]
    if not all(n in hmap for n in needed):
        print("DEBUG: ACS headers:", headers)
        print("DEBUG: ACS sample row:", js[1] if len(js) > 1 else None)
        missing = [n for n in needed if n not in hmap]
        raise KeyError(f"Unexpected ACS response header; missing columns: {missing}")

    acs = pd.DataFrame(rows, columns=headers)
    # Build 7-digit place GEOID = state (2) + place (5)
    acs["GEOID"] = acs[hmap["state"]].astype(str).str.zfill(2) + acs[hmap["place"]].astype(str).str.zfill(5)
    # Convert numeric fields
    for c in [hmap["s2201_c04_001e"], hmap["s2201_c04_001m"]]:
        acs[c] = pd.to_numeric(acs[c], errors="coerce")
    # Return canonical columns (rename to expected names)
    acs = acs.rename(columns={
        hmap["name"]: "NAME",
        hmap["s2201_c04_001e"]: "S2201_C04_001E",
        hmap["s2201_c04_001m"]: "S2201_C04_001M",
    })
    # Diagnostic: show received ACS columns and a sample
    print("\n-- ACS columns (sample) --")
    print(list(acs.columns)[:30])
    print("ACS sample row:", acs.head(2).to_dict(orient='records'))
    return acs[["GEOID", "NAME", "S2201_C04_001E", "S2201_C04_001M"]]

def download_and_unzip(url: str, dest_dir: str) -> str:
    zpath = os.path.join(DATADIR, os.path.basename(url))
    download(url, zpath, desc=os.path.basename(url))
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(dest_dir)
    return dest_dir

def load_places_geometry() -> gpd.GeoDataFrame:
    geodir = os.path.join(DATADIR, "cb_2024_us_place_500k")
    if not os.path.exists(geodir):
        os.makedirs(geodir, exist_ok=True)
        download_and_unzip(PLACES_CB_500K, geodir)
    # Find the .shp
    shp = None
    for fn in os.listdir(geodir):
        if fn.endswith(".shp"):
            shp = os.path.join(geodir, fn)
            break
    if shp is None:
        raise FileNotFoundError("Places shapefile not found after unzip.")
    # Use pyogrio driver if available via GeoPandas
    gdf = gpd.read_file(shp)
    # Ensure GEOID column exists and is string
    if "GEOID" not in gdf.columns:
        raise ValueError("Expected GEOID in places shapefile.")
    gdf["GEOID"] = gdf["GEOID"].astype(str)
    return gdf[["GEOID", "NAME", "STATEFP", "geometry"]]

def save_kml_if_possible(gdf: gpd.GeoDataFrame, kml_path: str) -> bool:
    try:
        # Convert columns to numeric and replace placeholder values with NaN
        gdf['S2201_C04_001E'] = pd.to_numeric(gdf['S2201_C04_001E'], errors='coerce').replace(-888888888, pd.NA)

        # Debug: Log the column names to verify presence of required columns
        print(f"DEBUG: GeoDataFrame columns: {gdf.columns.tolist()}")

        # Filter out rows where S2201_C04_001E is NaN
        gdf = gdf[gdf['S2201_C04_001E'].notna()]

        # Debug: Log the number of rows after filtering
        print(f"DEBUG: Number of rows in GeoDataFrame after filtering: {len(gdf)}")

        # Debug: Log a sample of the GeoDataFrame after filtering
        print(f"DEBUG: Sample of GeoDataFrame after filtering:\n{gdf.head(5)}")

        if len(gdf) == 0:
            print("No valid data to write KML. Skipping.")
            return False

        # Create the KML document structure manually for more control
        kml_header = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Principal Cities SNAP Data</name>
  <Style id="poly_style">
    <LineStyle>
      <color>ff0000ff</color>
      <width>2</width>
    </LineStyle>
    <PolyStyle>
      <fill>0</fill>
    </PolyStyle>
  </Style>
  <Style id="balloon_style">
    <BalloonStyle>
      <text>$[description]</text>
    </BalloonStyle>
  </Style>
'''
        kml_footer = '''</Document>
</kml>'''

        def format_placemark(row, is_point=False):
            # Debug: Log the row being processed
            print(f"DEBUG: Processing row: {row}")

            # Format the SNAP percentage for display
            snap_pct = row.get('S2201_C04_001E', '')
            snap_text = f" ({snap_pct}% households receiving SNAP)" if snap_pct else " (No SNAP data)"

            # For points, create a name that includes the SNAP percentage
            name = f"{row.get('CITY', '')}{snap_text}"

            # Format description
            description = f'''
<div style="font-family: Arial, sans-serif;">
  <p><strong>SNAP Participation:</strong> {snap_pct}% households receiving SNAP</p>
  <p><strong>Location:</strong> {row.get('CITY', '')}, {row.get('STATE_ABBR', '')}</p>
  <p><strong>Metropolitan Area:</strong> {row.get('CBSA_TITLE', '')}</p>
  <p><strong>Census GEOID:</strong> {row.get('GEOID', '')}</p>
</div>'''

            # Get the geometry as KML coordinates
            if is_point:
                x, y = row['geometry'].x, row['geometry'].y
                coords = f"{x},{y},0"
                geom_xml = f"<Point><coordinates>{coords}</coordinates></Point>"
            else:
                # Handle polygons and multi-polygons
                if row['geometry'].geom_type == 'MultiPolygon':
                    geom_xml = "<MultiGeometry>" + "".join(
                        f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{' '.join(f'{x},{y},0' for x, y in poly.exterior.coords)}</coordinates></LinearRing></outerBoundaryIs></Polygon>"
                        for poly in row['geometry'].geoms
                    ) + "</MultiGeometry>"
                else:
                    geom_xml = f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{' '.join(f'{x},{y},0' for x, y in row['geometry'].exterior.coords)}</coordinates></LinearRing></outerBoundaryIs></Polygon>"

            # Create the placemark XML
            style_id = "pin_style" if is_point else "poly_style"
            return f'''
  <Placemark>
    <name>{name}</name>
    <styleUrl>#{style_id}</styleUrl>
    <description><![CDATA[{description}]]></description>
    {geom_xml}
  </Placemark>'''

        # Create points from centroids - project to a suitable projection for centroid calculation
        gdf_points = gdf.copy()
        gdf_points_proj = gdf_points.to_crs('EPSG:3857')
        gdf_points_proj['geometry'] = gdf_points_proj['geometry'].centroid
        gdf_points = gdf_points_proj.to_crs('EPSG:4326')

        # Generate KML content
        placemarks = []

        # Add polygon placemarks
        for _, row in gdf.iterrows():
            placemarks.append(format_placemark(row, is_point=False))

        # Add point placemarks
        for _, row in gdf_points.iterrows():
            placemarks.append(format_placemark(row, is_point=True))

        # Combine all parts and write the file
        kml_content = kml_header + ''.join(placemarks) + kml_footer

        with open(kml_path, 'w', encoding='utf-8') as f:
            f.write(kml_content)

        # Debug: Confirm file writing
        print(f"DEBUG: KML file successfully written to {kml_path}")

        return True

    except Exception as e:
        # Debug: Log the exception
        print(f"DEBUG: Exception occurred during KML generation: {e}")
        warnings.warn(f"KML export skipped (driver not available): {e}")
        return False

# ----------------------------
# Main pipeline
# ----------------------------

def main():
    print("== Step 1: Download source lists (OMB principal cities, Gazetteer) ==")
    # OMB principal cities (List 2)
    omb_path = os.path.join(DATADIR, os.path.basename(OMB_PCITIES_XLSX))
    download(OMB_PCITIES_XLSX, omb_path, desc="OMB List2 Principal Cities (2023)")
    pcities = read_principal_cities_metropolitan(omb_path)

    # If any principal city rows lack STATE_ABBR, try deriving from the CBSA title
    if "STATE_ABBR" in pcities.columns:
        missing_st = pcities[pcities["STATE_ABBR"].isna() | (pcities["STATE_ABBR"] == "")]
        if len(missing_st):
            pcities.loc[pcities["STATE_ABBR"].isna() | (pcities["STATE_ABBR"] == ""), "STATE_ABBR"] = (
                pcities.loc[pcities["STATE_ABBR"].isna() | (pcities["STATE_ABBR"] == ""), "CBSA_TITLE"]
                .apply(lambda x: derive_state_from_cbsa(x))
            )

    # Diagnostic: show a sample of parsed OMB principal cities
    print("\n-- OMB principal cities (sample) --")
    try:
        print(pcities[["CBSA_CODE", "CITY", "STATE", "CITY_NORM", "STATE_ABBR"]].head(12).to_string())
    except Exception:
        print(pcities.head(12).to_string())

    # Gazetteer places
    gaz_path = os.path.join(DATADIR, os.path.basename(GAZETTEER_PLACES_TXT))
    download(GAZETTEER_PLACES_TXT, gaz_path, desc="2024 Gazetteer Places")
    gaz = read_gazetteer_places(gaz_path)

    # Diagnostic: show a sample of the gazetteer
    print("\n-- Gazetteer places (sample) --")
    print(gaz.head(12).to_string())

    print("== Step 2: Match principal cities to Census PLACE GEOIDs ==")
    matched = left_join_city_to_gazetteer(pcities, gaz)

    # Diagnostic: inspect matched/unmatched
    print("\n-- Matched principal cities (sample, BEFORE fallback) --")
    print(matched.head(12).to_string())

    # Handle remaining unmatched by a soft fallback: try raw city name (without normalization) + state
    # (Helps with edge naming like "New York (Manhattan)" etc. Usually the first pass is enough.)
    remaining = matched[matched["GEOID"].isna()].copy()
    if len(remaining) > 0:
        fallback = gaz.copy()
        fallback["CITY_UPPER"] = fallback["NAME"].str.upper()
        rem2 = remaining.copy()
        rem2["CITY_UPPER"] = rem2["CITY"].str.upper()
        rem2 = rem2.merge(
            fallback[["GEOID", "CITY_UPPER", "STATE_ABBR"]],
            on=["CITY_UPPER", "STATE_ABBR"],
            how="left",
            suffixes=("", "_F")
        )
        matched.loc[matched["GEOID"].isna(), "GEOID"] = rem2["GEOID"]
        matched["MATCHED"] = ~matched["GEOID"].isna()

    # Deduplicate by GEOID (a place can be principal for multiple CBSAs)
    matched_geoids = (
        matched[matched["MATCHED"]]
        .drop_duplicates(subset=["GEOID"])
        .copy()
    )

    print(f"Matched principal cities to GEOIDs: {len(matched_geoids)}; Unmatched: {matched['GEOID'].isna().sum()}")

    # Write unmatched principal cities (for inspection)
    unmatched_list = matched[matched["GEOID"].isna()][["CBSA_CODE", "CBSA_TITLE", "CITY", "STATE_ABBR"]].copy()
    # Try to fill missing STATE_ABBR from the CBSA title when possible (e.g., "..., DC-VA-MD-WV" -> DC)
    def _derive_state_from_cbsa(title: str) -> str | None:
        if not title or pd.isna(title):
            return None
        # look for a trailing token like 'DC-VA-MD-WV' or a comma-separated state list
        parts = [p.strip() for p in str(title).split(',') if p.strip()]
        if len(parts) > 1:
            candidate = parts[-1]
        else:
            candidate = parts[0]
        # split on hyphen and try first token that maps to a state abbrev
        for token in candidate.replace(';', ',').replace('/', '-').split('-'):
            tok = token.strip().upper()
            if len(tok) == 2 and (us.states.lookup(tok) or tok == 'DC'):
                return tok
        return None

    if len(unmatched_list):
        # fill where possible
        for idx, row in unmatched_list.iterrows():
            if pd.isna(row["STATE_ABBR"]) or row["STATE_ABBR"] == "":
                derived = _derive_state_from_cbsa(row["CBSA_TITLE"])
                if derived:
                    unmatched_list.loc[idx, "STATE_ABBR"] = derived
        unmatched_path = os.path.join(OUTDIR, "unmatched_principal_cities.csv")
        unmatched_list.to_csv(unmatched_path, index=False)
        print(f"Wrote unmatched principal cities (candidates) to {unmatched_path}")

    print("== Step 3: Pull ACS 1-year 2024 S2201 for all places and subset to principal cities ==")
    acs = fetch_acs_place_s2201()

    pcs = matched_geoids.merge(acs, on="GEOID", how="left")
    # If the merge created NAME_x/NAME_y, normalize to NAME preferring ACS value
    if "NAME" not in pcs.columns:
        if "NAME_y" in pcs.columns:
            pcs["NAME"] = pcs["NAME_y"]
        elif "NAME_x" in pcs.columns:
            pcs["NAME"] = pcs["NAME_x"]
    # Compute relative MOE (% of estimate) for quick data quality flag
    pcs["REL_MOE_PCT"] = (pcs["S2201_C04_001M"] / pcs["S2201_C04_001E"] * 100).replace([math.inf, -math.inf], pd.NA)

    # Basic select/rename for CSV
    out_cols = [
        "GEOID",
        "CITY", "STATE_ABBR", "CBSA_CODE", "CBSA_TITLE",
        "S2201_C04_001E", "S2201_C04_001M", "REL_MOE_PCT", "NAME"
    ]
    pcs_out = pcs[out_cols].sort_values(["STATE_ABBR", "CITY"]).reset_index(drop=True)
    # Ensure ACS numeric columns are numeric (coerce non-numeric to NaN)
    for c in ["S2201_C04_001E", "S2201_C04_001M", "REL_MOE_PCT"]:
        if c in pcs_out.columns:
            pcs_out[c] = pd.to_numeric(pcs_out[c], errors="coerce")

    # Create two CSVs: a metrics-only export (rows with S2201 present) and a full export for inspection
    pcs_metrics = pcs_out[pcs_out["S2201_C04_001E"].notna()].copy()
    csv_metrics_path = os.path.join(OUTDIR, "snap_principal_cities_2024_metrics.csv")
    pcs_metrics.to_csv(csv_metrics_path, index=False)
    print(f"Wrote metrics-only CSV (S2201 present): {csv_metrics_path} (rows: {len(pcs_metrics)})")

    csv_all_path = os.path.join(OUTDIR, "snap_principal_cities_2024_all.csv")
    pcs_out.to_csv(csv_all_path, index=False)
    print(f"Wrote full CSV (includes rows missing ACS): {csv_all_path} (rows: {len(pcs_out)})")

    print("== Step 4: Join geometry and export GeoJSON (+ KML if supported) ==")
    g_places = load_places_geometry()
    g = g_places.merge(pcs_out, on="GEOID", how="inner")
    # CRS: GeoJSON expects EPSG:4326
    if g.crs is None:
        g = g.set_crs(4326)
    else:
        g = g.to_crs(4326)

    # Convert S2201 columns to strings for Geo export to avoid OGR/pyogrio numeric parsing warnings
    for c in ["S2201_C04_001E", "S2201_C04_001M", "REL_MOE_PCT"]:
        if c in g.columns:
            # represent NaN as empty string so fields are consistently string-typed
            g[c] = g[c].apply(lambda v: "" if pd.isna(v) else str(v))

    geojson_path = os.path.join(OUTDIR, "snap_principal_cities_2024.geojson")
    g.to_file(geojson_path, driver="GeoJSON")
    print(f"Wrote {geojson_path}")

    # Optional KML (some environments won’t have KML driver)
    kml_path = os.path.join(OUTDIR, "snap_principal_cities_2024.kml")
    if save_kml_if_possible(g, kml_path):
        print(f"Wrote {kml_path}")
    else:
        print("KML export skipped (driver not available).")

    # Quick summary
    unmatched_cities = pcs[pcs["S2201_C04_001E"].isna()][["CITY", "STATE_ABBR", "GEOID"]]
    if len(unmatched_cities):
        warn_path = os.path.join(OUTDIR, "unmatched_or_missing_acs.csv")
        unmatched_cities.to_csv(warn_path, index=False)
        print(f"Note: {len(unmatched_cities)} principal cities missing ACS values; wrote {warn_path}")

    print("== Done. Files are in ./output ==")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)

def save_kml_if_possible(gdf: gpd.GeoDataFrame, kml_path: str) -> bool:
    try:
        # Ensure GEOID column exists and is string
        if "GEOID" not in gdf.columns:
            raise ValueError("Expected GEOID in GeoDataFrame.")

        # Filter out rows where S2201_C04_001E is NaN
        gdf = gdf[gdf['S2201_C04_001E'].notna()]

        if len(gdf) == 0:
            print("No valid data to write KML. Skipping.")
            return False

        # Create the KML document structure manually for more control
        kml_header = '''<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<kml xmlns=\"http://www.opengis.net/kml/2.2\">
<Document>
  <name>Principal Cities SNAP Data</name>
  <Style id="poly_style">
    <LineStyle>
      <color>ff0000ff</color>
      <width>2</width>
    </LineStyle>
    <PolyStyle>
      <fill>0</fill>
    </PolyStyle>
  </Style>
  <Style id="balloon_style">
    <BalloonStyle>
      <text>$[description]</text>
    </BalloonStyle>
  </Style>
'''

        kml_footer = '''</Document>
</kml>'''

        def format_placemark(row, is_point=False):
            # Format the SNAP percentage for display
            snap_pct = row.get('S2201_C04_001E', '')
            snap_text = f" ({snap_pct}% households receiving SNAP)" if snap_pct else " (No SNAP data)"

            # For points, create a name that includes the SNAP percentage
            name = f"{row.get('CITY', '')}{snap_text}"

            # Format description
            description = f'''
<div style="font-family: Arial, sans-serif;">
  <p><strong>SNAP Participation:</strong> {snap_pct}% households receiving SNAP</p>
  <p><strong>Location:</strong> {row.get('CITY', '')}, {row.get('STATE_ABBR', '')}</p>
  <p><strong>Metropolitan Area:</strong> {row.get('CBSA_TITLE', '')}</p>
  <p><strong>Census GEOID:</strong> {row.get('GEOID', '')}</p>
</div>'''

            # Get the geometry as KML coordinates
            if is_point:
                x, y = row['geometry'].x, row['geometry'].y
                coords = f"{x},{y},0"
                geom_xml = f"<Point><coordinates>{coords}</coordinates></Point>"
            else:
                # Handle polygons and multi-polygons
                if row['geometry'].geom_type == 'MultiPolygon':
                    geom_xml = "<MultiGeometry>" + "".join(
                        f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{' '.join(f'{x},{y},0' for x, y in poly.exterior.coords)}</coordinates></LinearRing></outerBoundaryIs></Polygon>"
                        for poly in row['geometry'].geoms
                    ) + "</MultiGeometry>"
                else:
                    geom_xml = f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{' '.join(f'{x},{y},0' for x, y in row['geometry'].exterior.coords)}</coordinates></LinearRing></outerBoundaryIs></Polygon>"

            # Create the placemark XML
            style_id = "pin_style" if is_point else "poly_style"
            return f'''
  <Placemark>
    <name>{name}</name>
    <styleUrl>#{style_id}</styleUrl>
    <description><![CDATA[{description}]]></description>
    {geom_xml}
  </Placemark>'''

        # Create points from centroids - project to a suitable projection for centroid calculation
        gdf_points = gdf.copy()
        gdf_points_proj = gdf_points.to_crs('EPSG:3857')
        gdf_points_proj['geometry'] = gdf_points_proj['geometry'].centroid
        gdf_points = gdf_points_proj.to_crs('EPSG:4326')

        # Generate KML content
        placemarks = []

        # Add polygon placemarks
        for _, row in gdf.iterrows():
            placemarks.append(format_placemark(row, is_point=False))

        # Add point placemarks
        for _, row in gdf_points.iterrows():
            placemarks.append(format_placemark(row, is_point=True))

        # Combine all parts and write the file
        kml_content = kml_header + ''.join(placemarks) + kml_footer

        with open(kml_path, 'w', encoding='utf-8') as f:
            f.write(kml_content)

        print(f"KML file successfully written to {kml_path}")
        return True

    except Exception as e:
        print(f"Exception occurred during KML generation: {e}")
        warnings.warn(f"KML export skipped (driver not available): {e}")
        return False
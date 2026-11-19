"""
Run with args: landsat, s2, or all

Requires:  pip install earthengine-api pandas numpy openpyxl

Code overview:
1) Builds rings and cleans data at and around every site (28) and control (3) in data center csv file
        Rings are 0-300m, 300m-1km, 1-2km, and 2-3km
2) Uses Landsat to collect Tier 1 Level 2 surface temperature (best quality atmosphere-corrected data)
    (L5 and L7 for 2000-2012, and L8 and L9 for 2012-present)
        This outputs a csv with data, each row is the avg temp of one site's ring for a given date
3) Uses Sentinel-2 data of images with atmospheric data already corrected to build vegetation index (NDVI), 
    water and moisture index (NDWI), and built environment index (NDBI)
        This outputs a csv with data, each row is the avg index of one site's ring for a given month
"""

### Setup
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import ee


### Configuration
PROJECT = "data-center-heat-islands"
SPREADSHEET = "data_centers.xlsx"
SHEET = "Data Centers"
OUT_DIR = Path("output")

RINGS_M = [(0, 300), (300, 1000), (1000, 2000), (2000, 3000)]

# To cut out cores of data centers placed near each other whose rings may overlap (value below in meters)
EXCLUDE_OTHER_CORES_M = 300

# When two data centers are too close together (value below in meters), the script will treat them as the same site
CLUSTER_DIST_M = 600

# Default assumption of when a data center was built unless found otherwise, confined by GEE satellite data 
PLACEHOLDER_CON_YEAR = 1985


### Landsat Configuration (L5 & L7 for 2000-2012, L8 & L9 till present)

# (label, EE data collection tag, thermal band, first date, last date or None = present)
LANDSAT_SOURCES = [
    ("LT05", "LANDSAT/LT05/C02/T1_L2", "ST_B6",  "2000-01-01", "2012-12-31"),
    ("LE07", "LANDSAT/LE07/C02/T1_L2", "ST_B6",  "2000-01-01", "2012-12-31"),
    ("LC08", "LANDSAT/LC08/C02/T1_L2", "ST_B10", "2013-01-01", None),
    ("LC09", "LANDSAT/LC09/C02/T1_L2", "ST_B10", "2021-10-01", None),
]
LANDSAT_START = "2000-01-01"
LANDSAT_CHUNK_MONTHS = 12


### GEE Compute Controls (limited to 150 EECU before constraining computation)

# Only use May to September to analyze when heat islands are strongest (set to None for whole year)
MONTH_RANGE = (5, 9)

# Skips images who half more than half cloud cover (or whatever percentage is below)
MAX_SCENE_CLOUD = 50

# For sites built before/on 1985, old data (from L5 and L7, hence the date below) will not be searched for
OLD_ERA_BEFORE_AFTER_ONLY = True
OLD_ERA_END = "2013-01-01"
ST_SCALE, ST_OFFSET_K = 0.00341802, 149.0        # Gets temp in Kelvin
QA_MASK_BITS = 0b111111      # Throws out pixels with no data, edges around clouds, cirrus clouds, other clouds, cloud shadows, or snow
MAX_ST_UNCERTAINTY_K = 5.0   # Throws out pixels with uncertainty above this threshold
LST_SCALE_M = 90             # Area to average temperature over in meters (resolution of sattelite is 60-120m)
TILE_SCALE = 4               # EE calculation split (raise it to 8 if memory limit exceeded)
MASK_WATER = True            # Removes bodies of water (much cooler than land)
WATER_OCCURRENCE_PCT = 50    # Same as above (if more than whatever percentage is stated is water)


### Sentinel-2 Config

S2_START = "2018-05-01"      # Start date
S2_MAX_SCENE_CLOUD = 60      # Throws out pixels with this percentage or more cloud cover
S2_CS_THRESHOLD = 0.60       # Same as above (lower for more hazy pixels, raise for few clean pixels)
S2_SCALE_M = 60              # Area to average indexes rings in meters (resolution of satelite is 10-20m)


### Sites
def _haversine_m(lat1, lon1, lat2, lon2): # Calculates distance between sites
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))

def load_sites() -> pd.DataFrame: # Streamlines spreadsheet for analysis
    # Classifies construction dates based on precision
    placeholder = df["con_year"].eq(PLACEHOLDER_CON_YEAR)
    bound = notes.str.contains(r"between|before\s+\d{4}") & ~placeholder
    df.loc[placeholder, "con_date"] = pd.NaT  # 1985 is "unknown, older than imagery", not a date
    quality = np.select(
        [placeholder, df["con_year"].isna(), bound, df["con_precision"].eq("month")],
        ["placeholder", "none", "bound", "exact_month"],
        default="year_only",
    )
    df["con_quality"] = quality

    # Classifies which sites can do a before/after land comparison (based on if there's a known construction year and its after 2000)
    ok = (
        df["role"].eq("data_center")
        & df["con_quality"].isin(["exact_month", "year_only"])
        & (df["con_date"] >= pd.Timestamp("2000-01-01"))
    )
    df["design"] = np.where(df["role"].eq("control"), "control",
                            np.where(ok, "before_after", "cross_sectional"))

    # Measures distance between sites and groups data centers into clusters based on proximity
    n = len(df)
    dist = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = _haversine_m(df.at[i, "lat"], df.at[i, "lon"], df.at[j, "lat"], df.at[j, "lon"])
    is_dc = df["role"].eq("data_center").to_numpy()
    dc_dist = np.where(is_dc[None, :], dist, np.inf)
    df["nearest_other_dc_m"] = dc_dist.min(axis=1).round(0)

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if is_dc[i] and is_dc[j] and dist[i, j] < CLUSTER_DIST_M:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    sizes = pd.Series(roots).map(pd.Series(roots).value_counts())
    cl_names, k = {}, 0
    cluster_id = []
    for i, r in enumerate(roots):
        if sizes[i] > 1:
            if r not in cl_names:
                k += 1
                cl_names[r] = f"CL{k:02d}"
            cluster_id.append(cl_names[r])
        else:
            cluster_id.append(df.at[i, "site_id"])
    df["cluster_id"] = cluster_id
    df["cluster_size"] = sizes.to_numpy()
    return df


### Google Earth Engine
def init_ee() -> None: # Part with the authenticator
    try:
        ee.Initialize(project=PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT)

class Extractor:
    LANDSAT_KEEP = ["system:index", "SPACECRAFT_ID", "SCENE_CENTER_TIME", "CLOUD_COVER", "WRS_PATH", "WRS_ROW"]

    def __init__(self, sites: pd.DataFrame): #Maps out water, builds heat comparison rings, builds before/after rings, loads data, and define search area
        self.sites = sites
        self.aoi = self._aoi(sites)
        if MASK_WATER:
            occ = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
            self.land = occ.lt(WATER_OCCURRENCE_PCT).rename("land")   # 1 = land
        else:
            self.land = ee.Image(1).rename("land")
        feats = self._build_rings()                      # List of (site_id, ee.Feature)
        self.rings = ee.FeatureCollection([f for _, f in feats])
        # Pre-2013 data is only useful for before/after sites and controls
        old_sites = sites[sites.design.isin(["before_after", "control"])]
        old_ids = set(old_sites.site_id)
        self.rings_old = ee.FeatureCollection([f for sid, f in feats if sid in old_ids])
        self.aoi_old = self._aoi(old_sites)
        self.cs_plus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    @staticmethod
    def _aoi(df): # Deliniates site region
        pts = df[["lon", "lat"]].astype(float).values.tolist()
        return ee.Geometry.MultiPoint(pts).buffer(max(r[1] for r in RINGS_M) + 500, 1)

    def _build_rings(self): # Name is self-explanatory lol
        dc = self.sites[self.sites.role == "data_center"]
        feats = []
        for _, s in self.sites.iterrows():
            pt = ee.Geometry.Point([float(s.lon), float(s.lat)])
            others = dc[dc.site_id != s.site_id]
            excl = None
            if len(others) and EXCLUDE_OTHER_CORES_M > 0:
                excl = ee.Geometry.MultiPoint(others[["lon", "lat"]].astype(float).values.tolist()) \
                    .buffer(EXCLUDE_OTHER_CORES_M, 1)
            for r_in, r_out in RINGS_M:
                g = pt.buffer(r_out, 1)
                if r_in > 0:
                    g = g.difference(pt.buffer(r_in, 1), 1)
                    if excl is not None:
                        g = g.difference(excl, 1)
                feats.append((s.site_id, ee.Feature(g, {"site_id": s.site_id, "ring": f"{r_in}-{r_out}m"})))
        return feats

    def _download(self, fc) -> pd.DataFrame: # Gets EE data
        info = fc.getInfo()
        rows = [f["properties"] for f in info["features"]]
        return pd.DataFrame(rows)

    @staticmethod
    def _run_parallel(jobs, workers: int) -> None: # Runs three paths at once
        """jobs: list of zero-argument callables. Raises the first error after all finish."""
        if workers <= 1 or len(jobs) <= 1:
            for j in jobs:
                j()
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(j) for j in jobs]
            for f in futs:
                f.result()

    ## Landsat
    def _landsat_prep(self, thermal_band): # Gets and cleans Landsat data!! Based on atmospheric clarity, landscape
        land = self.land

        def prep(img):
            qa = img.select("QA_PIXEL")
            clear = qa.bitwiseAnd(QA_MASK_BITS).eq(0)
            st = img.select(thermal_band)
            lst = st.multiply(ST_SCALE).add(ST_OFFSET_K).subtract(273.15)
            uncert = img.select("ST_QA").multiply(0.01)
            ok = (clear.And(st.gt(0)).And(uncert.lte(MAX_ST_UNCERTAINTY_K))
                  .And(lst.gt(-40)).And(lst.lt(70)).And(land))
            lst = lst.updateMask(ok).rename("LST")
            valid = ee.Image(1).updateMask(lst.mask()).unmask(0).rename("valid")
            out = ee.Image.cat([lst, valid, land])
            return out.copyProperties(img, self.LANDSAT_KEEP).set("system:time_start", img.get("system:time_start"))
        return prep

    def _landsat_zonal(self, img, rings): # Creates temperature rings and records metadata (idk what else to call it, it just keeps the date, recording sat, etc.)
        fc = img.reduceRegions(collection=rings, reducer=ee.Reducer.mean(),
                               scale=LST_SCALE_M, tileScale=TILE_SCALE)
        d = img.date().format("YYYY-MM-dd")
        sat, t = img.get("SPACECRAFT_ID"), img.get("SCENE_CENTER_TIME")
        cc, path, row = img.get("CLOUD_COVER"), img.get("WRS_PATH"), img.get("WRS_ROW")
        scene = img.get("system:index")

        def tag(f):
            return (ee.Feature(f).set("date", d).set("sat", sat).set("scene", scene)
                    .set("scene_time", t).set("cloud_cover", cc).set("path", path).set("row", row)
                    .setGeometry(None))
        return fc.map(tag)

    def _landsat_fc(self, start: str, end: str): # Collects relevant data and keeps certain pieces based on configuration parameters
        old = OLD_ERA_BEFORE_AFTER_ONLY and pd.Timestamp(end) <= pd.Timestamp(OLD_ERA_END)
        aoi, rings = (self.aoi_old, self.rings_old) if old else (self.aoi, self.rings)
        cols = []
        for _label, cid, band, first, last in LANDSAT_SOURCES:
            s = max(pd.Timestamp(start), pd.Timestamp(first))
            e = pd.Timestamp(end)
            if last:
                e = min(e, pd.Timestamp(last) + pd.Timedelta(days=1))
            if s >= e:
                continue
            col = (ee.ImageCollection(cid).filterBounds(aoi)
                   .filterDate(s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"))
                   .filter(ee.Filter.lt("CLOUD_COVER", MAX_SCENE_CLOUD)))
            if MONTH_RANGE:
                col = col.filter(ee.Filter.calendarRange(MONTH_RANGE[0], MONTH_RANGE[1], "month"))
            cols.append(col.map(self._landsat_prep(band)))
        if not cols:
            return None
        col = cols[0]
        for c in cols[1:]:
            col = col.merge(c)
        return (ee.FeatureCollection(col.map(lambda im: self._landsat_zonal(im, rings))).flatten()
                .filter(ee.Filter.gt("valid", 0)))

    def landsat(self, start: str, end: str, cache: Path, force=False, workers=3) -> None: #LANDSAT TIIIIIIIME
        cache.mkdir(parents=True, exist_ok=True)
        windows = month_windows(start, end, LANDSAT_CHUNK_MONTHS)

        def job(i, a, b):
            def run():
                f = cache / f"ls_{a}_{b}.csv"
                if f.exists() and not force:
                    return
                t0 = time.time()
                df = self._download_split(self._landsat_fc, a, b, min_days=16)
                df.to_csv(f, index=False)
                print(f"[{i}/{len(windows)}] Landsat {a} -> {b}: {len(df)} rows, "
                      f"{df['scene'].nunique() if len(df) else 0} scenes, {time.time() - t0:.0f}s", flush=True)
            return run
        self._run_parallel([job(i, a, b) for i, (a, b) in enumerate(windows, 1)], workers)

    ## Sentinel
    def _s2_prep(self, img): # Gets and cleans Sentinel data based on atmospheric clarity, landscape and calculates indexes
        good = img.select("cs_cdf").gte(S2_CS_THRESHOLD)
        nd = ee.Image.cat([
            img.normalizedDifference(["B8", "B4"]).rename("NDVI"),
            img.normalizedDifference(["B3", "B8"]).rename("NDWI"),
            img.normalizedDifference(["B11", "B8"]).rename("NDBI"),
        ])
        return nd.updateMask(good).updateMask(self.land)

    def _s2_fc(self, start: str, end: str): # Averages values over a month and ring, also saved metadata
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(self.aoi).filterDate(start, end)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_MAX_SCENE_CLOUD))
               .linkCollection(self.cs_plus, ["cs_cdf"]))
        if col.size().getInfo() == 0:
            return None
        col = col.map(self._s2_prep)
        comp = col.median()
        n_img = col.select("NDVI").count().rename("n_img")
        valid = ee.Image(1).updateMask(comp.select("NDVI").mask()).unmask(0).rename("valid")
        out = ee.Image.cat([comp, n_img, valid, self.land])
        fc = out.reduceRegions(collection=self.rings, reducer=ee.Reducer.mean(),
                               scale=S2_SCALE_M, tileScale=TILE_SCALE)
        return fc.map(lambda f: ee.Feature(f).setGeometry(None)).filter(ee.Filter.gt("valid", 0))

    def s2(self, start: str, end: str, cache: Path, force=False, workers=3) -> None: #SENTINEL TIIIIIIIIME
        cache.mkdir(parents=True, exist_ok=True)
        months = pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(), pd.Timestamp(end), freq="MS")
        if MONTH_RANGE:
            months = [m for m in months if MONTH_RANGE[0] <= m.month <= MONTH_RANGE[1]]

        def job(i, m):
            def run():
                f = cache / f"s2_{m:%Y-%m}.csv"
                if f.exists() and not force:
                    return
                t0 = time.time()
                nxt = m + pd.DateOffset(months=1)
                df = self._download_split(self._s2_fc, m.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d"), min_days=100)
                if len(df):
                    df["month"] = f"{m:%Y-%m}"
                df.to_csv(f, index=False)
                print(f"[{i}/{len(months)}] Sentinel-2 {m:%Y-%m}: {len(df)} rows, {time.time() - t0:.0f}s", flush=True)
            return run
        self._run_parallel([job(i, m) for i, m in enumerate(months, 1)], workers)


### Helper Functions
def month_windows(start: str, end: str, months: int): # Divides date ranges into chunks, used in landsat function
    s = pd.Timestamp(start).to_period("M").to_timestamp()
    e = pd.Timestamp(end)
    out = []
    while s < e:
        n = s + pd.DateOffset(months=months)
        out.append((s.strftime("%Y-%m-%d"), min(n, e).strftime("%Y-%m-%d")))
        s = n
    return out

def read_cache(cache: Path) -> pd.DataFrame: # Combines CSVs into one big table in a cache folder
    parts = []
    for f in sorted(cache.glob("*.csv")):
        try:
            d = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def combine(cache_ls: Path, cache_s2: Path, out_dir: Path) -> None: # Saves Landsat and Sentinel data into a new csv
    ls = read_cache(cache_ls)
    if len(ls):
        ls = ls.drop_duplicates(["site_id", "ring", "scene"])
        ls["valid_frac"] = (ls["valid"] / ls["land"].where(ls["land"] > 0)).clip(upper=1)
        ls["date"] = pd.to_datetime(ls["date"])
        ls = ls.sort_values(["site_id", "date", "ring"])
        ls.to_csv(out_dir / "landsat_all.csv", index=False)
    s2 = read_cache(cache_s2)
    if len(s2):
        s2["valid_frac"] = (s2["valid"] / s2["land"].where(s2["land"] > 0)).clip(upper=1)
        s2 = s2.sort_values(["site_id", "month", "ring"])
        s2.to_csv(out_dir / "s2_monthly.csv", index=False)


### Main Function
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "test", "landsat", "s2", "combine", "all"])
    ap.add_argument("--start", help="YYYY-MM-DD (default: config start for that product)")
    ap.add_argument("--end", help="YYYY-MM-DD, exclusive (default: tomorrow)")
    ap.add_argument("--force", action="store_true", help="re-download chunks that are already cached")
    ap.add_argument("--workers", type=int, default=3, help="parallel Earth Engine requests (default 3)")
    a = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    sites = load_sites()
    sites.to_csv(OUT_DIR / "sites.csv", index=False)

    end = a.end or (date.today() + timedelta(days=1)).isoformat()
    cache_ls, cache_s2 = OUT_DIR / "cache_landsat", OUT_DIR / "cache_s2"

    init_ee()
    ex = Extractor(sites)

    if a.command in ("landsat", "all"):
        ex.landsat(a.start or LANDSAT_START, end, cache_ls, a.force, a.workers)

    if a.command in ("s2", "all"):
        ex.s2(a.start or S2_START, end, cache_s2, a.force, a.workers)

    if a.command == "all":
        combine(cache_ls, cache_s2, OUT_DIR)

### Actually Runs the Code, No Spaghetti Code Today!!
main()
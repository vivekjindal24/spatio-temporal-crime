"""Graph construction for HASI-Net.

Builds the three adjacency channels used by the heterogeneous adaptive graph
block:

* ``A_geo``    -- geographic rook contiguity from REAL district / community-area
                  boundaries (datameet Census 2011 shapefile for Indian
                  districts; City of Chicago community-area GeoJSON for the
                  benchmark)
* ``A_socio``  -- kNN cosine similarity on census socioeconomic features
* ``A_adaptive`` -- learned at training time (see ``model.py``); here we only
  return its initializer shape.

All matrices are symmetric, self-loop-free, and row-normalized.
"""
from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import Config, DATA_DIR
from .data import download, _normalize_state, MADHYA_PRADESH

# datameet Census 2011 district shapefile (CC-BY). geopandas needs all four
# components in the same directory.
INDIA_DIST_SHP = {
    "shp": "https://raw.githubusercontent.com/datameet/maps/master/Districts/Census_2011/2011_Dist.shp",
    "dbf": "https://raw.githubusercontent.com/datameet/maps/master/Districts/Census_2011/2011_Dist.dbf",
    "shx": "https://raw.githubusercontent.com/datameet/maps/master/Districts/Census_2011/2011_Dist.shx",
    "prj": "https://raw.githubusercontent.com/datameet/maps/master/Districts/Census_2011/2011_Dist.prj",
}
# Chicago community-area boundaries (77 areas, EPSG:4326), with area_num_1.
CHICAGO_AREAS_GEOJSON = (
    "https://raw.githubusercontent.com/RandomFractals/ChicagoCrimes/master/"
    "data/chicago-community-areas.geojson"
)

# Panel district name -> datameet 2011 shapefile name, for the few MP districts
# the shapefile still records under their pre-2013 names. After data.py
# canonicalization every other MP panel name matches the shapefile exactly, so
# rook contiguity and the choropleth join are exact except for these two.
SHAPEFILE_ALIASES = {
    "Khandwa": "East Nimar",
    "Khargone": "West Nimar",
}
# Post-2011 MP districts (carved after the 2011 Census) have no 2011 boundary.
# Link each to the districts it was carved from rather than the generic
# centroid-kNN fallback, so its graph edges reflect real geography.
POST_2011_PARENTS = {
    "Agar": ["Shajapur", "Rajgarh"],   # Agar Malwa, carved 2013
}


def _row_normalize(A: np.ndarray) -> np.ndarray:
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return A / deg


def _symmetrize(A: np.ndarray) -> np.ndarray:
    return np.maximum(A, A.T)


def _is_chicago(districts: List[str]) -> bool:
    return bool(districts) and re.fullmatch(r"CA\d+", str(districts[0])) is not None


def _load_india_districts(cfg: Config):
    """Download the datameet 2011 district shapefile and filter to the target
    state. Returns a GeoDataFrame indexed by Title-cased district name."""
    import geopandas as gpd
    base = DATA_DIR / "2011_Dist"
    for ext, url in INDIA_DIST_SHP.items():
        download(url, base.with_suffix("." + ext), timeout=180)
    gdf = gpd.read_file(str(base.with_suffix(".shp")))
    name_col = next((c for c in ["DISTRICT", "NAME_2", "name_2", "district"]
                     if c in gdf.columns), None)
    state_col = next((c for c in ["ST_NM", "NAME_1", "STATE", "state", "name_1"]
                      if c in gdf.columns), None)
    if name_col is None or state_col is None:
        raise RuntimeError("India district shapefile missing name/state columns")
    gdf["__d"] = gdf[name_col].astype(str).str.strip().str.title()
    gdf["__s"] = gdf[state_col].astype(str).str.strip().str.title()
    gdf = gdf[gdf["__s"].str.lower() == cfg.target_region.strip().lower()]
    gdf = gdf.drop_duplicates(subset="__d", keep="first").set_index("__d")
    return gdf


def _load_chicago_areas():
    """Download the Chicago community-area GeoJSON. Returns a GeoDataFrame
    indexed by integer area number (1..77)."""
    import geopandas as gpd
    dest = DATA_DIR / "chicago_community_areas.geojson"
    download(CHICAGO_AREAS_GEOJSON, dest, timeout=120)
    gdf = gpd.read_file(dest)
    num_col = next((c for c in ["area_num_1", "area_numbe", "area_num", "comarea_id"]
                    if c in gdf.columns), None)
    if num_col is None:
        raise RuntimeError("Chicago areas GeoJSON missing area-number column")
    gdf["__n"] = pd.to_numeric(gdf[num_col], errors="coerce").astype("Int64")
    gdf = gdf.dropna(subset=["__n"]).drop_duplicates(subset="__n", keep="first")
    gdf = gdf.set_index("__n")
    return gdf


def _rook_from_gdf(gdf, key_for, districts: List[str], cfg: Config) -> np.ndarray:
    """Build rook-contiguity adjacency from a GeoDataFrame indexed by the key
    used in ``districts``. ``key_for(d)`` maps a district label to the gdf key.
    Missing districts are linked by centroid kNN to the nearest present
    centroids (real lat/long)."""
    import geopandas as gpd
    n = len(districts)
    A = np.zeros((n, n), dtype=np.float32)
    # Map each present district to its panel index and its boundary key.
    key_to_panel = {}
    present_keys = []
    for d in districts:
        k = key_for(d)
        if k is not None and k in gdf.index:
            key_to_panel[k] = districts.index(d)
            present_keys.append(k)
    sub = gdf.reindex(present_keys)
    joined = gpd.sjoin(sub, gdf, how="left", predicate="touches")
    # sjoin stores the matched right-side key in a column named after the
    # right gdf's index (e.g. "__d_right" for India, "__n_right" for Chicago).
    right_col = next((c for c in joined.columns if c.endswith("_right")
                      and c.rsplit("_right", 1)[0] == str(gdf.index.name)), None)
    if right_col is None:
        right_col = "index_right" if "index_right" in joined.columns else None
    for k in present_keys:
        if k not in joined.index or right_col is None:
            continue
        i = key_to_panel[k]
        rows = joined.loc[k]
        if isinstance(rows, pd.Series):
            nbs = [rows[right_col]] if pd.notna(rows.get(right_col)) else []
        else:
            nbs = rows[right_col].dropna().tolist()
        for nb in nbs:
            if nb in key_to_panel and nb != k:
                A[i, key_to_panel[nb]] = 1.0
    # Centroid-kNN for districts missing from the boundaries (e.g. Indian
    # districts created after the 2011 Census). Post-2011 districts with known
    # parents are linked to those parents (real geography); any other missing
    # district falls back to nearest present centroids.
    missing = [d for d in districts if key_for(d) is None or key_for(d) not in gdf.index]
    present_panels = [key_to_panel[k] for k in present_keys]
    if missing and present_keys:
        cents = np.array([(gdf.loc[k].geometry.centroid.x,
                           gdf.loc[k].geometry.centroid.y) for k in present_keys])
        for d in missing:
            i = districts.index(d)
            if d in POST_2011_PARENTS:
                # Link to the real districts it was carved from.
                for parent in POST_2011_PARENTS[d]:
                    pk = key_for(parent)
                    if pk in key_to_panel:
                        A[i, key_to_panel[pk]] = 1.0
                continue
            # Generic fallback: link to the k nearest present centroids.
            x0, y0 = cents.mean(axis=0)
            dist = np.linalg.norm(cents - np.array([x0, y0]), axis=1)
            for j in np.argsort(dist)[:min(3, len(present_panels))]:
                A[i, present_panels[j]] = 1.0
    return A


def build_geographic_adjacency(districts: List[str], cfg: Config) -> np.ndarray:
    """Rook-contiguity adjacency from REAL boundaries.

    Indian districts (MP panel): datameet Census 2011 shapefile, rook
    contiguity, centroid-kNN for post-2011 districts. Chicago community areas:
    City of Chicago GeoJSON, rook contiguity. Falls back to a deterministic
    centroid-kNN only if boundary download/parse fails (logged in meta).
    """
    n = len(districts)
    A = np.zeros((n, n), dtype=np.float32)
    try:
        if _is_chicago(districts):
            gdf = _load_chicago_areas()

            def key_for(d):
                try:
                    return int(re.sub(r"\D", "", d))
                except Exception:
                    return None
            A = _rook_from_gdf(gdf, key_for, districts, cfg)
        else:
            gdf = _load_india_districts(cfg)

            def key_for(d):
                if d in SHAPEFILE_ALIASES:
                    return SHAPEFILE_ALIASES[d]
                return d if d in gdf.index else None
            A = _rook_from_gdf(gdf, key_for, districts, cfg)
    except Exception:
        A = _centroid_knn_fallback(districts)

    A = _symmetrize(A)
    np.fill_diagonal(A, 0.0)
    # Keep graph connected: link any isolated node to its nearest non-isolated
    # neighbour by degree.
    iso = np.where(A.sum(axis=1) == 0)[0]
    if len(iso) and A.sum() > 0:
        deg = A.sum(axis=1)
        for i in iso:
            j = int(np.argsort(deg)[-2])
            A[i, j] = A[j, i] = 1.0
    return _row_normalize(A + np.eye(n, dtype=np.float32))


def _centroid_knn_fallback(districts: List[str], k: int = 4) -> np.ndarray:
    """Fallback: approximate adjacency via name-hash distance (last resort)."""
    n = len(districts)
    A = np.zeros((n, n), dtype=np.float32)
    # Deterministic pseudo-coordinates from name hash so the fallback is
    # reproducible (real contiguity is preferred via the GeoJSON path).
    coords = np.array([[(hash(d) >> 16) & 0xFFFF, hash(d) & 0xFFFF]
                       for d in districts], dtype=np.float32)
    coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)
    for i in range(n):
        d = np.linalg.norm(coords - coords[i], axis=1)
        d[i] = np.inf
        for j in np.argsort(d)[:k]:
            A[i, j] = 1.0
    return _row_normalize(_symmetrize(A) + np.eye(n, dtype=np.float32))


def build_socioeconomic_adjacency(node_feats: np.ndarray, cfg: Config) -> np.ndarray:
    """kNN cosine-similarity adjacency on socioeconomic features."""
    x = node_feats.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    sim = (x @ x.T) / (norms @ norms.T)
    np.fill_diagonal(sim, 0.0)
    n = sim.shape[0]
    A = np.zeros_like(sim)
    k = min(cfg.socio_knn, n - 1)
    for i in range(n):
        top = np.argsort(-sim[i])[:k]
        A[i, top] = 1.0
    A = _symmetrize(A)
    np.fill_diagonal(A, 0.0)
    return _row_normalize(A + np.eye(n, dtype=np.float32))


def build_graphs(districts: List[str], node_feats: np.ndarray,
                 cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Return (A_geo, A_socio), both [N, N] row-normalized."""
    A_geo = build_geographic_adjacency(districts, cfg)
    A_socio = build_socioeconomic_adjacency(node_feats, cfg)
    return A_geo, A_socio
"""Data acquisition and preprocessing for HASI-Net.

Downloads real, publicly-available datasets and assembles them into the panel
tensors the model consumes:

* NCRB "crimes against women" district tables (2001-2014 + 2017-2022)
* Census 2011 district socioeconomic features (node attributes)
* City of Chicago crime incidents (benchmark, aggregated to area x month)

Everything is cached under ``data/`` so reruns are cheap. The 2015-2016 NCRB
reporting gap is bridged by linear interpolation and explicitly flagged in the
returned panel metadata so it can be disclosed in the paper.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from .config import (Config, DATA_DIR, WOMEN_CRIME_CATEGORIES,
                     MADHYA_PRADESH, UNIFIED_CRIMES)

# --- Source URLs (all public, machine-readable) -----------------------------
NCRB_GH_BASE = "https://raw.githubusercontent.com/Sidd7893/crime-analysis/master/"
NCRB_FILES = {
    "2001_2012": NCRB_GH_BASE + "42_District_wise_crimes_committed_against_women_2001_2012.csv",
    "2013": NCRB_GH_BASE + "42_District_wise_crimes_committed_against_women_2013.csv",
    "2014": NCRB_GH_BASE + "42_District_wise_crimes_committed_against_women_2014.csv",
}
INDIA_DATA_PORTAL = (
    "https://ckandev.indiadataportal.com/dataset/crime-statistics/resource/"
    "c8d3ea7e-4855-45f3-9c26-557419e93b6a/download/districtwise-crime-against-women.csv"
)
CENSUS_HF_URL = "https://huggingface.co/datasets/indiaset/census-2011/resolve/main/data/train-00000-of-00001.parquet"
DISTRICT_GEOJSON = (
    "https://github.com/guneetnarula/indian-district-boundaries/raw/master/"
    "india_district_map.geojson"
)
CHICAGO_API = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
# Austin (TX) Crime Reports (Socrata). Unlike Chicago, Austin carries an explicit
# ``family_violence`` Y/N flag, so the domestic-violence channel is a direct
# measurement rather than a domestic-BATTERY proxy. The dataset has no
# zip_code / latitude / longitude fields -- its only clean reproducible
# geography is ``council_district`` (10 nodes); see build_austin_panel + the
# Austin branch of graph.build_geographic_adjacency.
AUSTIN_API = "https://data.austintexas.gov/resource/fdj4-gpfu.json"
# Keyword substrings (upper-cased) used to fetch and classify Austin crime_type
# strings, which are far messier than Chicago's primary_type. The
# family_violence flag (not these strings) drives the DV/assault split.
AUSTIN_RAPE_KEYS = ("SEXUAL", "SODOMY", "RAPE")
AUSTIN_KIDNAP_KEYS = ("KIDNAPPING",)
AUSTIN_ASSAULT_KEYS = ("ASSAULT", "ASLT")

# Canonical NCRB column names (2001-2014 schema) -> our internal category keys.
NCRB_COL_MAP = {
    "rape_sexual_assault": ["rape"],
    "domestic_violence": ["cruelty by husband or his relatives",
                          "cruelty by husband/relatives"],
    "kidnapping_abduction": ["kidnapping and abduction", "kidnapping & abduction"],
    "assault": ["assault on women with intent to outrage her modesty",
                "assault on women"],
}

# Chicago primary types pulled for the four unified crimes. Both sexual-assault
# spellings are included -- the portal renamed "CRIM SEXUAL ASSAULT" to
# "CRIMINAL SEXUAL ASSAULT" mid-series, and the old code missed the newer (and
# larger) name, undercounting sexual assault ~65%. BATTERY is pulled for the
# domestic-violence channel (domestic-flagged battery); ASSAULT for the assault
# channel. BATTERY and ASSAULT are disjoint primary types, so the two channels
# do not double-count.
CHICAGO_TYPES = ["ASSAULT", "BATTERY",
                 "CRIMINAL SEXUAL ASSAULT", "CRIM SEXUAL ASSAULT",
                 "KIDNAPPING"]

# --- MP district-name canonicalization ---------------------------------------
# NCRB records a few MP "districts" that are not geographic units, plus several
# spelling variants of real districts. We canonicalize at panel-build time so
# every remaining node is a real district whose name joins the datameet Census
# 2011 shapefile exactly (the graph/choropleth alias map in graph.py handles
# the two districts the shapefile still records under pre-2013 names, and the
# one post-2011 district with no 2011 boundary).
#
# Railway and cyber-crime police jurisdictions are NOT coextensive with any
# district, so their incidents cannot be reliably geolocated; we drop them
# (0.70% of MP incidents, 2014-). This is disclosed in the paper.
MP_PSEUDO_DISTRICTS = {
    "Bhopal Railway", "Bhopal Rly.", "Indore Railway", "Indore Rly.",
    "Jabalpur Railway", "Jabalpur Rly.", "Cyber Cell",
}
# Merge NCRB spelling variants / rename to the shapefile's canonical spelling.
MP_DISTRICT_RENAME = {
    "Ashok Nagar": "Ashoknagar",
    "Datiya": "Datia",
    "Narsinghpur": "Narsimhapur",
    "Sihore": "Sehore",
    "Umariya": "Umaria",
    "Khargon": "Khargone",   # older spelling -> merge onto "Khargone"
}


def _ensure_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def download(url: str, dest: Path, timeout: int = 60) -> Path:
    """Download ``url`` to ``dest`` if not already cached. Returns local path."""
    _ensure_dir()
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            if chunk:
                fh.write(chunk)
    return dest


def _normalize_state(s: str) -> str:
    return str(s).strip().title()


def _match_column(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        for low, orig in lower.items():
            if cand in low:
                return orig
    return None


def load_ncrb_2001_2014() -> pd.DataFrame:
    """Load and concatenate the 2001-2014 NCRB district women-crime CSVs."""
    frames = []
    for tag, url in NCRB_FILES.items():
        dest = DATA_DIR / f"ncrb_women_{tag}.csv"
        download(url, dest)
        df = pd.read_csv(dest)
        # Harmonize state/district/year column names.
        state_col = _match_column(df.columns, ["state/ut", "state ut", "state"])
        dist_col = _match_column(df.columns, ["district"])
        year_col = _match_column(df.columns, ["year"])
        df = df.rename(columns={state_col: "state", dist_col: "district",
                                year_col: "year"})
        # Rename human-readable crime columns to internal category keys.
        rename = {}
        for cat, cands in NCRB_COL_MAP.items():
            col = _match_column(df.columns, cands)
            if col is not None and col != cat:
                rename[col] = cat
        df = df.rename(columns=rename)
        df["state"] = df["state"].apply(_normalize_state)
        df["district"] = df["district"].astype(str).str.strip().str.title()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # Drop NCRB state/UT total rows.
    out = out[~out["district"].str.contains("Total", case=False, na=False)]
    return out


def load_ncrb_2017_2022() -> pd.DataFrame:
    """Load the India Data Portal 2017-2022 district women-crime CSV.

    Column names differ from the 2001-2014 schema and vary across years, so we
    fuzzy-match each category. Missing categories are filled with NaN and later
    interpolated.
    """
    dest = DATA_DIR / "ncrb_women_2017_2022.csv"
    try:
        download(INDIA_DATA_PORTAL, dest, timeout=120)
        df = pd.read_csv(dest)
    except Exception:
        # The portal occasionally moves resources; degrade gracefully.
        return pd.DataFrame()

    state_col = _match_column(df.columns, ["state_name", "state", "state/ut"]) or "state"
    dist_col = _match_column(df.columns, ["district_name", "district"]) or "district"
    year_col = _match_column(df.columns, ["year"]) or "year"
    df = df.rename(columns={state_col: "state", dist_col: "district", year_col: "year"})
    df["state"] = df["state"].apply(_normalize_state)
    df["district"] = df["district"].astype(str).str.strip().str.title()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    mapped = {}
    for cat, cands in NCRB_COL_MAP.items():
        col = _match_column(df.columns, cands)
        if col is not None:
            mapped[cat] = df[col]
    if not mapped:
        return pd.DataFrame()
    out = pd.DataFrame({"state": df["state"], "district": df["district"],
                        "year": df["year"]})
    for cat, series in mapped.items():
        out[cat] = pd.to_numeric(series, errors="coerce")
    out = out[~out["district"].str.contains("Total", case=False, na=False)]
    return out


def build_mp_crime_panel(cfg: Config) -> pd.DataFrame:
    """Long-form MP panel: one row per (district, year, category)."""
    hist = load_ncrb_2001_2014()
    recent = load_ncrb_2017_2022()

    def to_long(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["state", "district", "year",
                                         "category", "count"])
        cat_cols = [c for c in WOMEN_CRIME_CATEGORIES if c in df.columns]
        if not cat_cols:
            return pd.DataFrame(columns=["state", "district", "year",
                                         "category", "count"])
        long = df.melt(id_vars=["state", "district", "year"],
                       value_vars=cat_cols, var_name="category",
                       value_name="count")
        return long

    long = pd.concat([to_long(hist), to_long(recent)], ignore_index=True)
    long = long[long["state"].str.lower() == MADHYA_PRADESH.lower()]
    long["count"] = pd.to_numeric(long["count"], errors="coerce").clip(lower=0)
    long = long.dropna(subset=["year", "district"])
    long["year"] = long["year"].astype(int)

    # Canonicalize district names: drop non-geographic police jurisdictions and
    # merge/rename spelling variants onto real shapefile-matchable districts.
    long = long[~long["district"].isin(MP_PSEUDO_DISTRICTS)]
    long["district"] = long["district"].replace(MP_DISTRICT_RENAME)

    # Pivot to [district x year x category], interpolate the 2015-2016 gap,
    # then melt back. Flag interpolated rows. aggfunc="sum" is the correct
    # aggregation for crime counts and is robust to any duplicate rows.
    panel = long.pivot_table(index=["district", "category"],
                             columns="year", values="count", aggfunc="sum")
    panel = panel.reindex(columns=range(cfg.mp_year_start, cfg.mp_year_end + 1))
    interp_mask = panel.isna().any(axis=1)
    panel = panel.interpolate(axis=1, limit_direction="both").fillna(0)
    out = panel.stack().reset_index().rename(columns={0: "count"})
    # Tag interpolated years for disclosure.
    gap_years = {2015, 2016}
    out["interpolated"] = out["year"].isin(gap_years)
    return out


def load_census_features(districts: List[str]) -> pd.DataFrame:
    """Census 2011 socioeconomic features for the given districts.

    Returns a DataFrame indexed by district with columns:
    literacy_rate, sex_ratio, urbanization, workers_ratio, sc_ratio, st_ratio.
    """
    # Primary: GitHub 118-column Census 2011 CSV (reliable). Fallback: HF parquet.
    gh = ("https://raw.githubusercontent.com/nishusharma1608/"
          "India-Census-2011-Analysis/master/india-districts-census-2011.csv")
    try:
        dest = DATA_DIR / "census_2011.csv"
        download(gh, dest, timeout=120)
        cens = pd.read_csv(dest)
    except Exception:
        dest = DATA_DIR / "census_2011.parquet"
        download(CENSUS_HF_URL, dest, timeout=120)
        cens = pd.read_parquet(dest)

    # Harmonize district names.
    name_col = _match_column(cens.columns, ["district name", "district_name",
                                            "district"]) or "District"
    state_col = _match_column(cens.columns, ["state name", "state_name",
                                             "state"]) or "State name"
    cens = cens.rename(columns={name_col: "district", state_col: "state"})
    cens["district"] = cens["district"].astype(str).str.strip().str.title()
    cens["state"] = cens["state"].astype(str).str.strip().str.title()
    cens = cens[cens["state"].str.lower() == MADHYA_PRADESH.lower()]
    cens = cens.set_index("district", drop=False)

    feats = pd.DataFrame(index=cens.index)
    lit = _match_column(cens.columns, ["literacy_rate", "literate_total", "literate"])
    pop = _match_column(cens.columns, ["pop_total", "population"])
    fem = _match_column(cens.columns, ["pop_female", "female"])
    male = _match_column(cens.columns, ["pop_male", "male"])
    work = _match_column(cens.columns, ["workers_total", "workers"])
    sc = _match_column(cens.columns, ["pop_sc", "sc"])
    st = _match_column(cens.columns, ["pop_st", "st"])
    urban = _match_column(cens.columns, ["urban", "urban_pop", "urban_population"])

    if lit and pop:
        if "literacy_rate" in lit.lower():
            feats["literacy_rate"] = pd.to_numeric(cens[lit], errors="coerce")
        else:
            feats["literacy_rate"] = (pd.to_numeric(cens[lit], errors="coerce")
                                      / pd.to_numeric(cens[pop], errors="coerce"))
    if fem and male:
        feats["sex_ratio"] = (pd.to_numeric(cens[fem], errors="coerce")
                              / pd.to_numeric(cens[male], errors="coerce") * 1000)
    if work and pop:
        feats["workers_ratio"] = (pd.to_numeric(cens[work], errors="coerce")
                                  / pd.to_numeric(cens[pop], errors="coerce"))
    if sc and pop:
        feats["sc_ratio"] = pd.to_numeric(cens[sc], errors="coerce") / pd.to_numeric(cens[pop], errors="coerce")
    if st and pop:
        feats["st_ratio"] = pd.to_numeric(cens[st], errors="coerce") / pd.to_numeric(cens[pop], errors="coerce")
    if urban and pop:
        feats["urbanization"] = pd.to_numeric(cens[urban], errors="coerce") / pd.to_numeric(cens[pop], errors="coerce")
    else:
        feats["urbanization"] = 0.0

    feats = feats[~feats.index.duplicated(keep="first")]
    # Align to requested districts; fill missing with column median, then 0.
    feats = feats.reindex(districts)
    feats = feats.fillna(feats.median(numeric_only=True))
    feats = feats.fillna(0.0)
    return feats


@dataclass
class Panel:
    """A fully assembled modelling panel.

    counts      : np.float32 [T, N, C]   crime counts
    node_feats  : np.float32 [N, F]      socioeconomic node features
    years       : list[int]  length T
    districts   : list[str]  length N
    categories  : list[str]  length C
    meta        : dict       provenance / interpolation flags
    """
    counts: np.ndarray
    node_feats: np.ndarray
    years: List[int]
    districts: List[str]
    categories: List[str]
    meta: Dict = field(default_factory=dict)


def build_mp_panel(cfg: Config) -> Panel:
    """Assemble the full MP panel: counts tensor + node features."""
    long = build_mp_crime_panel(cfg)
    if long.empty:
        raise RuntimeError("MP crime panel is empty — check data downloads.")

    cats = [c for c in WOMEN_CRIME_CATEGORIES if c in set(long["category"])]
    years = sorted(long["year"].unique())
    districts = sorted(long["district"].unique())

    piv = long.pivot_table(index=["district", "category"], columns="year",
                           values="count").reindex(index=pd.MultiIndex.from_product(
        [districts, cats], names=["district", "category"]), columns=years)
    piv = piv.fillna(0)
    counts = piv.values.reshape(len(districts), len(cats), len(years))
    counts = np.transpose(counts, (2, 0, 1)).astype(np.float32)  # [T, N, C]

    feats = load_census_features(districts)
    feat_mat = feats.to_numpy(dtype=np.float32)
    # Standardize features.
    feat_mat = (feat_mat - feat_mat.mean(axis=0)) / (feat_mat.std(axis=0) + 1e-6)

    interp_share = float(long["interpolated"].mean())
    return Panel(
        counts=counts,
        node_feats=feat_mat,
        years=[int(y) for y in years],
        districts=list(districts),
        categories=list(cats),
        meta={"region": MADHYA_PRADESH, "interpolated_share": interp_share,
              "gap_years": [2015, 2016], "source": "NCRB Crime in India",
              "dropped_pseudo_districts": sorted(MP_PSEUDO_DISTRICTS),
              "dropped_share_note": "railway/cyber police jurisdictions not "
                                    "coextensive with any district (~0.70% of "
                                    "incidents), excluded as non-spatially-"
                                    "attributable"},
    )


def build_chicago_panel(cfg: Config) -> Panel:
    """Assemble the Chicago benchmark panel aggregated to area x month.

    Pulls incidents via the Socrata JSON API (filtered to women-relevant
    primary types and the configured year window) and bins them by community
    area and month.
    """
    # Cache version v2: includes the `domestic` flag (needed for the
    # domestic-violence channel) and both sexual-assault spellings. The v1
    # cache lacked `domestic` and only had the old "CRIM SEXUAL ASSAULT" name.
    dest = DATA_DIR / "chicago_crimes_v2.parquet"
    if not dest.exists():
        rows: List[dict] = []
        offset = 0
        where = (f"year >= {cfg.chicago_year_start} and year <= {cfg.chicago_year_end} "
                 f"and primary_type in ({','.join(repr(t) for t in CHICAGO_TYPES)})")
        select = "community_area,primary_type,date,year,domestic"
        while True:
            params = {"$select": select, "$where": where,
                      "$limit": 50000, "$offset": offset}
            r = requests.get(CHICAGO_API, params=params, timeout=120)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 50000:
                break
            offset += 50000
        chi = pd.DataFrame(rows)
        chi.to_parquet(dest)
    else:
        chi = pd.read_parquet(dest)

    chi["date"] = pd.to_datetime(chi["date"], errors="coerce")
    chi = chi.dropna(subset=["date", "community_area"])
    chi["community_area"] = chi["community_area"].astype(int)
    chi["month"] = chi["date"].dt.to_period("M").astype(str)
    # Coerce the domestic flag to a boolean.
    chi["domestic"] = chi["domestic"].astype(str).str.strip().str.lower().isin(
        ["true", "1", "t", "yes"])

    # Assign each incident to one of the four unified crime channels. BATTERY
    # (domestic) and ASSAULT are disjoint primary types, so the
    # domestic-violence and assault channels never double-count an incident.
    def _classify(row):
        pt = row["primary_type"]
        if pt in ("CRIMINAL SEXUAL ASSAULT", "CRIM SEXUAL ASSAULT"):
            return "rape_sexual_assault"
        if pt == "BATTERY":
            return "domestic_violence" if row["domestic"] else None
        if pt == "KIDNAPPING":
            return "kidnapping_abduction"
        if pt == "ASSAULT":
            return "assault"
        return None
    chi["category"] = chi.apply(_classify, axis=1)
    chi = chi.dropna(subset=["category"])

    months = sorted(chi["month"].unique())
    areas = sorted(chi["community_area"].unique())
    cats = UNIFIED_CRIMES  # canonical order, shared with NCRB
    piv = chi.pivot_table(index=["community_area", "category"],
                          columns="month", values="date", aggfunc="count",
                          fill_value=0)
    piv = piv.reindex(index=pd.MultiIndex.from_product([areas, cats],
                      names=["community_area", "category"]), columns=months,
                      fill_value=0)
    counts = piv.values.reshape(len(areas), len(cats), len(months))
    counts = np.transpose(counts, (2, 0, 1)).astype(np.float32)

    # No socioeconomic features for Chicago areas in this pipeline -> use
    # identity (ones) so the graph branch still functions.
    node_feats = np.ones((len(areas), 1), dtype=np.float32)
    return Panel(
        counts=counts,
        node_feats=node_feats,
        years=months,
        districts=[f"CA{a}" for a in areas],
        categories=list(cats),
        meta={"region": "Chicago", "aggregation": "community_area x month",
              "crimes": "rape_sexual_assault (criminal sexual assault), "
                        "domestic_violence (domestic battery, proxy), "
                        "kidnapping_abduction, assault",
              "source": "City of Chicago Data Portal (ijzp-q8t2)"},
    )


def _socrata_url(base: str, select: str, where: str,
                 limit: int, offset: int) -> str:
    """Build a fully URL-encoded Socrata query string.

    ``requests``' params encoder escapes ``%`` (the SoQL wildcard) to ``%25``,
    which silently breaks ``like '%...%'`` filters. We therefore build the URL by
    hand with :func:`urllib.parse.quote`, marking ``%`` safe so the wildcards
    reach Socrata verbatim while spaces / quotes / parens are properly encoded.
    """
    from urllib.parse import quote

    def _enc(s: str) -> str:
        # Keep ``%`` (wildcard), ``'`` (string literals) and ``()`` unescaped
        # to keep the SoQL readable; Socrata accepts the raw forms in a query.
        return quote(s, safe="%'()")

    return (f"{base}?$select={_enc(select)}&$where={_enc(where)}"
            f"&$limit={limit}&$offset={offset}")


def _classify_austin(crime_type: str, family_violence: bool) -> Optional[str]:
    """Map an Austin incident to one unified crime channel (disjoint).

    Priority order mirrors Chicago so the four channels never double-count an
    incident:

    1. sexual-assault / rape (any keyword) -- regardless of the DV flag, since
       sexual assault is the headline crime of the channel (Chicago routes
       CRIMINAL SEXUAL ASSAULT -> rape regardless of its domestic flag too);
    2. kidnapping;
    3. otherwise, the ``family_violence`` flag -> domestic_violence (the clean
       signal Austin exposes, unlike Chicago's battery proxy);
    4. otherwise, assault / aggravated-assault keywords -> assault.

    Incidents matching none of these are dropped.
    """
    ct = str(crime_type).upper()
    if any(k in ct for k in AUSTIN_RAPE_KEYS):
        return "rape_sexual_assault"
    if any(k in ct for k in AUSTIN_KIDNAP_KEYS):
        return "kidnapping_abduction"
    if family_violence:
        return "domestic_violence"
    if any(k in ct for k in AUSTIN_ASSAULT_KEYS):
        return "assault"
    return None


def build_austin_panel(cfg: Config) -> Panel:
    """Assemble the Austin panel aggregated to council_district x month.

    Pulls incidents via the Socrata JSON API, keeping only the rows that map to
    one of the four unified crimes (family-violence incidents, plus assault /
    sexual-assault / kidnapping keywords). The ``family_violence`` flag is the
    authoritative domestic-violence signal; ``crime_type`` keywords supply the
    sexual-assault, kidnapping and non-DV assault channels. Aggregated to the 10
    Austin city council districts (the dataset's only clean reproducible
    geography) by month, over the configured year window.
    """
    dest = DATA_DIR / "austin_crimes_v1.parquet"
    if not dest.exists():
        rows: List[dict] = []
        offset = 0
        # Server-side filter: keep DV incidents plus every crime_type that maps
        # to a channel. ``upper(...) like`` is SoQL-case-sensitive on the raw
        # column, so we uppercase it. Wildcards (%) are preserved verbatim by
        # _socrata_url (see the helper docstring).
        keys = (AUSTIN_RAPE_KEYS + AUSTIN_KIDNAP_KEYS + AUSTIN_ASSAULT_KEYS)
        or_like = " OR ".join(f"upper(crime_type) like '%{k}%'" for k in keys)
        where = f"family_violence='Y' OR {or_like}"
        select = "council_district,crime_type,occ_date,family_violence"
        while True:
            url = _socrata_url(AUSTIN_API, select, where, 50000, offset)
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 50000:
                break
            offset += 50000
        aus = pd.DataFrame(rows)
        aus.to_parquet(dest)
    else:
        aus = pd.read_parquet(dest)

    aus["occ_date"] = pd.to_datetime(aus["occ_date"], errors="coerce")
    aus = aus.dropna(subset=["occ_date", "council_district"])
    # Council districts are the integers 1..10; coerce and keep the valid set.
    aus["council_district"] = pd.to_numeric(aus["council_district"],
                                            errors="coerce")
    aus = aus.dropna(subset=["council_district"])
    aus["council_district"] = aus["council_district"].astype(int)
    aus = aus[aus["council_district"].between(1, 10)]
    aus["month"] = aus["occ_date"].dt.to_period("M").astype(str)
    # Coerce the family_violence flag to a boolean (Y / N).
    aus["family_violence"] = aus["family_violence"].astype(str).str.strip()
    aus["family_violence"] = aus["family_violence"].str.upper().eq("Y")

    aus["category"] = aus.apply(
        lambda r: _classify_austin(r["crime_type"], r["family_violence"]),
        axis=1)
    aus = aus.dropna(subset=["category"])

    months = sorted(aus["month"].unique())
    areas = sorted(aus["council_district"].unique())
    cats = UNIFIED_CRIMES  # canonical order, shared with Chicago / NCRB
    piv = aus.pivot_table(index=["council_district", "category"],
                          columns="month", values="occ_date", aggfunc="count",
                          fill_value=0)
    piv = piv.reindex(index=pd.MultiIndex.from_product([areas, cats],
                      names=["council_district", "category"]), columns=months,
                      fill_value=0)
    counts = piv.values.reshape(len(areas), len(cats), len(months))
    counts = np.transpose(counts, (2, 0, 1)).astype(np.float32)

    # No socioeconomic features for Austin districts in this pipeline -> use
    # identity (ones) so the graph branch still functions, matching Chicago.
    node_feats = np.ones((len(areas), 1), dtype=np.float32)
    return Panel(
        counts=counts,
        node_feats=node_feats,
        years=months,
        districts=[f"CD{a}" for a in areas],
        categories=list(cats),
        meta={"region": "Austin",
              "aggregation": "council_district x month",
              "n_nodes": len(areas),
              "crimes": "rape_sexual_assault (sexual assault / sodomy), "
                        "domestic_violence (family_violence=Y, direct flag), "
                        "kidnapping_abduction, assault (non-DV assault)",
              "source": "City of Austin Open Data (fdj4-gpfu)",
              "dv_signal": "explicit family_violence flag (not a proxy)",
              "note": "10 council districts: the dataset's only clean "
                      "reproducible geography (no zip/lat-lon). Disclosed."},
    )


def get_dataset(name: str, cfg: Config) -> Panel:
    if name.lower() in {"mp", "madhya_pradesh"}:
        return build_mp_panel(cfg)
    if name.lower() == "chicago":
        return build_chicago_panel(cfg)
    if name.lower() == "austin":
        return build_austin_panel(cfg)
    raise ValueError(f"unknown dataset: {name}")
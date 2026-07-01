"""Figure generation for HASI-Net results.

Every function writes a PNG (300 dpi) into ``results/`` and returns the path.
These are the figures cited in the papers -- generated from real notebook runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import RESULTS_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})

PALETTE = {
    "HASI-Net": "#c0392b", "InformerOnly": "#2980b9", "STGCN": "#27ae60",
    "LSTM": "#8e44ad", "HA": "#7f8c8d",
}


def plot_training_curves(history: Dict, title: str, out: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train"], label="train loss", color="#2c3e50")
    ax.plot(history["val"], label="val MAE", color=PALETTE["HASI-Net"])
    ax.set_xlabel("epoch"); ax.set_ylabel("loss / MAE")
    ax.set_title(title); ax.legend()
    fig.tight_layout(); p = RESULTS_DIR / out; fig.savefig(p); plt.close(fig)
    return p


def plot_pso_convergence(hist: Dict, out: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist["iter"], hist["gbest"], "-o", label="global best",
            color=PALETTE["HASI-Net"])
    ax.plot(hist["iter"], hist["mean"], "--s", label="swarm mean",
            color="#2980b9")
    ax.set_xlabel("PSO iteration"); ax.set_ylabel("validation MAE")
    ax.set_title("Adaptive PSO convergence"); ax.legend()
    fig.tight_layout(); p = RESULTS_DIR / out; fig.savefig(p); plt.close(fig)
    return p


def plot_pred_vs_actual(pred: np.ndarray, true: np.ndarray, years: List[int],
                        district: str, category: str, out: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(years, true, "-o", label="actual", color="#34495e")
    ax.plot(years, pred, "-s", label="HASI-Net forecast",
            color=PALETTE["HASI-Net"])
    ax.set_xlabel("year"); ax.set_ylabel("reported cases")
    ax.set_title(f"{district} — {category.replace('_', ' ')}")
    ax.legend()
    fig.tight_layout(); p = RESULTS_DIR / out; fig.savefig(p); plt.close(fig)
    return p


def plot_model_comparison(df, metric: str, out: str) -> Path:
    """df: rows = model names, must contain a `metric` column."""
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(df.index)
    vals = df[metric].values
    colors = [PALETTE.get(n, "#34495e") for n in names]
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(f"Model comparison — {metric}")
    fig.tight_layout(); p = RESULTS_DIR / out; fig.savefig(p); plt.close(fig)
    return p


def plot_ablation(df, out: str) -> Path:
    """df: index = variant, column = MAE."""
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(df.index)
    vals = df["MAE"].values
    bars = ax.barh(names, vals, color="#c0392b")
    for b, v in zip(bars, vals):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.3f}",
                va="center", fontsize=9)
    ax.set_xlabel("MAE (lower is better)")
    ax.set_title("Ablation — component contributions")
    fig.tight_layout(); p = RESULTS_DIR / out; fig.savefig(p); plt.close(fig)
    return p


def plot_district_risk_heatmap(risk: np.ndarray, districts: List[str],
                               out: str) -> Path:
    """risk: [N] aggregated risk scores for the test horizon."""
    fig, ax = plt.subplots(figsize=(8, max(4, len(districts) * 0.18)))
    order = np.argsort(-risk)
    ordered = [districts[i] for i in order]
    vals = risk[order]
    ax.barh(ordered[::-1], vals[::-1], color="#c0392b")
    ax.set_xlabel("forecasted risk score (sum of predicted counts)")
    ax.set_title("District-level women-centric crime risk ranking")
    fig.tight_layout(); p = RESULTS_DIR / out; fig.savefig(p); plt.close(fig)
    return p


def plot_channel_weights(w: np.ndarray, out: str) -> Path:
    """w: length-3 softmax weights [geo, socio, adaptive]."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Geographic", "Socioeconomic", "Adaptive"], w,
           color=["#27ae60", "#2980b9", "#c0392b"])
    ax.set_ylabel("learned fusion weight")
    ax.set_title("Heterogeneous graph channel weights")
    fig.tight_layout(); p = RESULTS_DIR / out; fig.savefig(p); plt.close(fig)
    return p


def plot_choropleth(risk: np.ndarray, districts: List[str], out: str) -> Path:
    """Choropleth of MP district risk using the real datameet 2011 shapefile.

    Panel district names are canonicalized in data.py to match the shapefile
    exactly, except for the two districts the shapefile still records under
    pre-2013 names (Khandwa->East Nimar, Khargone->West Nimar); those are
    bridged via graph.SHAPEFILE_ALIASES. Agar (a post-2011 district with no
    2011 boundary) has no polygon and is simply absent from the map.
    """
    import geopandas as gpd
    from .config import DATA_DIR
    from .graph import SHAPEFILE_ALIASES
    shp = DATA_DIR / "2011_Dist.shp"
    if not shp.exists():
        return None
    gdf = gpd.read_file(str(shp))
    name_col = next((c for c in ["DISTRICT", "NAME_2", "name_2", "district"]
                     if c in gdf.columns), None)
    state_col = next((c for c in ["ST_NM", "NAME_1", "STATE", "state", "name_1"]
                      if c in gdf.columns), None)
    if name_col is None or state_col is None:
        return None
    gdf["__d"] = gdf[name_col].astype(str).str.strip().str.title()
    gdf = gdf[gdf[state_col].astype(str).str.strip().str.title()
              == "Madhya Pradesh"]
    gdf = gdf.drop_duplicates(subset="__d", keep="first").set_index("__d")
    # Map each shapefile polygon to a panel risk via the alias bridge.
    key_to_risk = {SHAPEFILE_ALIASES.get(d, d): r for d, r in zip(districts, risk)}
    gdf["risk"] = gdf.index.map(key_to_risk).astype(float)
    fig, ax = plt.subplots(figsize=(7, 7))
    gdf.plot(column="risk", cmap="OrRd", legend=True, ax=ax,
             edgecolor="white", linewidth=0.4, missing_kwds={"color": "#eeeeee"})
    ax.set_axis_off()
    ax.set_title("Forecasted women-centric crime risk — Madhya Pradesh")
    fig.tight_layout(); p = RESULTS_DIR / out; fig.savefig(p); plt.close(fig)
    return p
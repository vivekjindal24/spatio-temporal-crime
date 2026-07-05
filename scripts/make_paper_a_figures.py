"""Regenerate the Paper-1 (benchmark / architecture) comparison figures from
the final persisted result files.

Paper A's headline numbers are the 8-model x 5-seed x 3-horizon Chicago
benchmark and the component ablation, reported in the paper tables.  The
preliminary 3-seed artefacts that currently live in ``results/`` (named
``*_p1_prelim_*``) DO NOT match the final 5-seed tables and MUST NOT be used
for the published figures.  This script therefore reads only the FINAL
artefacts, which are produced on Colab and persisted to Drive
(``HASI_RESULTS_DIR``):

  metrics_p1_chicago_meanstd.csv      8-model MAE/RMSE/... at H=3 (mean,std)
  metrics_p1_chicago_h6_meanstd.csv   same, H=6
  metrics_p1_chicago_h12_meanstd.csv  same, H=12
  ablation_p1_chicago_meanstd.csv     channel-removal + ZINB-loss ablation
  hotspot_p1hot_chicago_meanstd.csv   POD/FAR/CSI/bias/Hit@k per model
  p1_multihorizon_stats.json          Friedman chi2/p per horizon

Expected CSV schema (mirrors the Paper-2 artefacts): a two-row header where
row 1 names the metric and row 2 is ``mean``/``std``, so duplicated columns
appear as ``MAE``, ``MAE.1`` etc.  The index column is the model / variant
name.  All numbers are taken verbatim from these files; nothing is fabricated
or hard-coded.  If a required file is missing the script prints a clear
message and exits non-zero so stale preliminary figures are never silently
substituted.

Figures produced (into ``paper_a/figures/``):
  comparison_MAE_chicago.png   per-model MAE at H=3/6/12 (matches tab:multihorizon)
  comparison_CSI_chicago.png    per-node CSI per model (matches tab:hotspot)
  ablation_chicago.png          variant MAE bars (matches tab:ablation)

The architecture, channel-weight, PSO-convergence and training-curve figures
are produced directly by the training/notebook artefacts and are not
regenerated here.

Usage (on Colab, after the final P1 runs are persisted on Drive):
  python scripts/make_paper_a_figures.py
  python scripts/make_paper_a_figures.py --out paper_a/figures
  python scripts/make_paper_a_figures.py --results $HASI_RESULTS_DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "paper_a" / "figures"
DEFAULT_RESULTS = Path(os.environ.get("HASI_RESULTS_DIR", REPO / "results"))

# Model order matches the paper tables (HA first as the persistence baseline).
MODELS = ["HA", "LSTM", "ST-GCN", "GraphWaveNet", "DCRNN", "MTGNN",
          "InformerOnly", "HASI-Net"]
HORIZONS = ["3", "6", "12"]

# Palette: persistence baseline distinct from the deep models.
C_HA = "#6c6c6c"      # historical average (persistence)
C_DEEP = "#1f5fa0"    # deep models
C_HASI = "#c04546"    # HASI-Net (ours)
C_ABL = "#2a8c4a"     # ablation bars

STYLE = {"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
         "figure.dpi": 150, "savefig.dpi": 150, "axes.axisbelow": True}


def _setup() -> None:
    plt.rcParams.update(STYLE)


def _color(model: str) -> str:
    if model == "HA":
        return C_HA
    if model == "HASI-Net":
        return C_HASI
    return C_DEEP


def _require(results: Path, name: str) -> Path:
    p = results / name
    if not p.exists():
        print(f"[make_paper_a_figures] MISSING final artefact: {p}\n"
              f"  The final 5-seed P1 results live on Drive (HASI_RESULTS_DIR)."
              f"  Reconnect the Colab notebook and re-run the P1 drivers, or"
              f"  copy {name} into the results dir. Preliminary 3-seed files"
              f" deliberately NOT used (they contradict the final tables).",
              file=sys.stderr)
        sys.exit(1)
    return p


def _load_meanstd(path: Path) -> pd.DataFrame:
    """Read a two-row-header mean/std CSV; returns DataFrame indexed by model.

    Duplicated metric columns appear as ``MAE``, ``MAE.1`` (mean, std).
    """
    df = pd.read_csv(path, index_col=0, header=[0, 1])
    df = df.drop(index=["NaN", "condition"], errors="ignore")
    # Flatten the MultiIndex columns to "METRIC" (mean) / "METRIC.std".
    flat = {}
    for col in df.columns:
        metric, stat = col
        flat[metric if stat == "mean" else f"{metric}.std"] = df[col]
    return pd.DataFrame(flat)


def _mae_at_horizon(results: Path, h: str) -> pd.DataFrame:
    fname = ("metrics_p1_chicago_meanstd.csv" if h == "3"
             else f"metrics_p1_chicago_h{h}_meanstd.csv")
    return _load_meanstd(_require(results, fname))


# --------------------------------------------------------------------------- #
# Multi-horizon MAE (matches tab:multihorizon)                                 #
# --------------------------------------------------------------------------- #
def multihorizon_figure(results: Path, out: Path) -> None:
    frames = {h: _mae_at_horizon(results, h) for h in HORIZONS}
    # Restrict / order to the paper's model set (tolerate missing labels).
    def mae(h: str, model: str) -> tuple[float, float]:
        df = frames[h]
        if model not in df.index:
            return (np.nan, 0.0)
        return (float(df.loc[model, "MAE"]),
                float(df.loc[model, "MAE.std"]))
    x = np.arange(len(MODELS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, h in enumerate(HORIZONS):
        means = [mae(h, m)[0] for m in MODELS]
        stds = [mae(h, m)[1] for m in MODELS]
        offset = (i - 1) * w
        ax.bar(x + offset, means, w, yerr=stds, capsize=3,
               label=f"$H={h}$", edgecolor="k", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("-", "\n") for m in MODELS], fontsize=8)
    ax.set_ylabel("Test MAE (lower better)")
    ax.set_title("Chicago multi-horizon MAE — 8 models, 5 seeds")
    ax.legend(fontsize=8, title="Horizon")
    # Annotate HA (persistence) as the reference.
    ha_idx = MODELS.index("HA")
    ax.axvspan(ha_idx - 0.5, ha_idx + 0.5, color="grey", alpha=0.08, zorder=0)
    fig.tight_layout()
    fig.savefig(out / "comparison_MAE_chicago.png")
    plt.close(fig)
    print("  wrote comparison_MAE_chicago.png")


# --------------------------------------------------------------------------- #
# Per-node CSI (matches tab:hotspot)                                          #
# --------------------------------------------------------------------------- #
def hotspot_figure(results: Path, out: Path) -> None:
    df = _load_meanstd(_require(results, "hotspot_p1hot_chicago_meanstd.csv"))
    present = [m for m in MODELS if m in df.index]
    csi = [float(df.loc[m, "CSI"]) for m in present]
    csi_s = [float(df.loc[m, "CSI.std"]) for m in present]
    x = np.arange(len(present))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x, csi, yerr=csi_s, capsize=3,
           color=[_color(m) for m in present], edgecolor="k", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("-", "\n") for m in present], fontsize=8)
    ax.set_ylabel("Per-node CSI (higher better)")
    ax.set_title("Leak-free per-node spike CSI — Chicago $H=3$, 5 seeds")
    for xi, v in zip(x, csi):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "comparison_CSI_chicago.png")
    plt.close(fig)
    print("  wrote comparison_CSI_chicago.png")


# --------------------------------------------------------------------------- #
# Component ablation (matches tab:ablation)                                   #
# --------------------------------------------------------------------------- #
def ablation_figure(results: Path, out: Path) -> None:
    df = _load_meanstd(_require(results, "ablation_p1_chicago_meanstd.csv"))
    order = ["Full HASI-Net", "no-adaptive-graph", "no-socio",
             "no-spatial", "ZINB-loss"]
    present = [v for v in order if v in df.index]
    mae = [float(df.loc[v, "MAE"]) for v in present]
    mae_s = [float(df.loc[v, "MAE.std"]) for v in present]
    x = np.arange(len(present))
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    colors = [C_HASI if v == "Full HASI-Net" else C_ABL for v in present]
    ax.bar(x, mae, yerr=mae_s, capsize=3, color=colors,
           edgecolor="k", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("-", "\n") for v in present], fontsize=8)
    ax.set_ylabel("Test MAE (lower better)")
    ax.set_title("Component ablation — Chicago $H=3$, 5 seeds")
    for xi, v in zip(x, mae):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ablation_chicago.png")
    plt.close(fig)
    print("  wrote ablation_chicago.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                    help="dir with the final persisted P1 CSV/JSON files")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output figures dir (paper_a/figures)")
    args = ap.parse_args()
    _setup()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Generating Paper-1 figures from {args.results} -> {args.out}")
    multihorizon_figure(args.results, args.out)
    hotspot_figure(args.results, args.out)
    ablation_figure(args.results, args.out)
    # Optionally echo the Friedman stats if present (no figure, for the log).
    stats = args.results / "p1_multihorizon_stats.json"
    if stats.exists():
        with open(stats) as fh:
            d = json.load(fh)
        print(f"  Friedman stats present: {json.dumps(d)}")
    else:
        print("  (p1_multihorizon_stats.json not found — Friedman numbers in "
              "the paper are taken from the final persisted JSON on Drive.)")
    print("Paper-1 figures complete.")


if __name__ == "__main__":
    main()
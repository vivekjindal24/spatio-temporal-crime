"""Regenerate every Paper-2 (calibration) figure from persisted result files.

Paper B's figures are all derived from the re-runnable Colab CUDA result files
persisted under ``HASI_RESULTS_DIR`` (Drive). This script reads those CSV/JSON
artefacts and writes the six paper figures into ``paper_b/figures/`` so the LaTeX
compiles with real, non-fabricated figures. Re-running it top-to-bottom
regenerates every figure in Paper B identically.

Figures produced:
  condcal_chicago.png   conditional coverage by crime-level bucket (THE headline)
  condcal_austin.png     same, out-of-distribution second city
  probbase_chicago.png   coverage@80 + CRPS for the 3 probabilistic conditions
  probbase_austin.png    same, Austin
  transfer_austin.png    Chicago->Austin transfer: MAE + coverage (scratch vs transfer)
  robustness_mp.png      MP missing-data robustness: MAE + coverage vs mask fraction

Usage (on Colab, after the P2 runs are persisted on Drive):
  python scripts/make_paper_b_figures.py
  python scripts/make_paper_b_figures.py --out paper_b/figures
  python scripts/make_paper_b_figures.py --results $HASI_RESULTS_DIR
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "paper_b" / "figures"
DEFAULT_RESULTS = Path(os.environ.get("HASI_RESULTS_DIR", REPO / "results"))

# Palette (consistent across figures).
C_CAL = "#1f5fa0"   # calibrated (ours)
C_CF = "#c04546"    # split conformal
C_QR = "#2a8c4a"    # quantile regression
C_PT = "#6c6c6c"    # point / scratch
C_TF = "#1f5fa0"    # transfer
C_SC = "#888888"    # scratch

STYLE = {"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
         "figure.dpi": 150, "savefig.dpi": 150, "axes.axisbelow": True}


def _setup():
    plt.rcParams.update(STYLE)


def _condcal(results: Path, city: str) -> dict:
    with open(results / f"summary_p2cc_{city}_condcal.json") as fh:
        return json.load(fh)["conditions"]


def _load4(results: Path, city: str, tag: str) -> pd.DataFrame:
    df = pd.read_csv(results / f"probbase_{city}_{tag}_4way_meanstd.csv",
                     index_col=0)
    return df.drop(index=["NaN", "condition"], errors="ignore")


# --------------------------------------------------------------------------- #
# Conditional coverage by bucket (the headline)                                #
# --------------------------------------------------------------------------- #
def condcal_figure(results: Path, city: str, out: Path) -> None:
    d = _condcal(results, city)
    cal, cf = d["calibrated"], d["conformal"]
    buckets = ["low", "med", "high", "top_decile"]
    labels = ["Low", "Medium", "High", "Top-decile"]
    cal_v = [cal[b + "_mean"] for b in buckets]
    cf_v = [cf[b + "_mean"] for b in buckets]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    b1 = ax.bar(x - w / 2, cal_v, w, label="Calibrated (HASI-Net)", color=C_CAL)
    b2 = ax.bar(x + w / 2, cf_v, w, label="Split conformal", color=C_CF)
    ax.axhline(0.80, ls="--", c="k", lw=1, alpha=0.7, label="Nominal 80%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Conditional coverage by crime-level bucket "
                 f"— {city.capitalize()}")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                    f"{r.get_height():.2f}", ha="center", va="bottom",
                    fontsize=8)
    ax.legend(fontsize=8, loc="lower left", ncol=3)
    fig.tight_layout()
    fname = f"condcal_{city}.png"
    fig.savefig(out / fname)
    plt.close(fig)
    print(f"  wrote {fname}")


# --------------------------------------------------------------------------- #
# Four-way probabilistic baselines (coverage@80 + CRPS)                        #
# --------------------------------------------------------------------------- #
def probbase_figure(results: Path, city: str, tag: str, out: Path) -> None:
    df = _load4(results, city, tag)
    conds = ["calibrated", "conformal", "quantreg"]
    labels = ["Calibrated\n(HASI-Net)", "Split\nconformal",
              "Quantile\nregression"]
    cov = [float(df.loc[c, "coverage80"]) for c in conds]
    crps = [float(df.loc[c, "CRPS"]) for c in conds]
    cols = [C_CAL, C_CF, C_QR]
    x = np.arange(len(conds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    bars = ax.bar(x - w / 2, cov, w, color=cols, label="Coverage@80")
    ax.axhline(0.80, ls="--", c="k", lw=1, alpha=0.7)
    ax.set_ylabel("Coverage@80")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.bar(x + w / 2, crps, w, color=cols, alpha=0.45, hatch="//",
            label="CRPS")
    ax2.set_ylabel("CRPS (lower better)")
    title = f"Probabilistic baselines — {city.capitalize()} (5 seeds)"
    ax.set_title(title)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    fig.tight_layout()
    fname = f"probbase_{city}.png"
    fig.savefig(out / fname)
    plt.close(fig)
    print(f"  wrote {fname}")


# --------------------------------------------------------------------------- #
# Cross-region transfer (Chicago -> Austin)                                   #
# --------------------------------------------------------------------------- #
def transfer_figure(results: Path, out: Path) -> None:
    df = pd.read_csv(results / "transfer_p2chi_aus_meanstd.csv", index_col=0)
    df = df.drop(index=["NaN", "condition"], errors="ignore")
    g = lambda c, k: float(df.loc[c, k])  # noqa: E731
    m_s, m_t = g("scratch", "MAE"), g("transfer", "MAE")
    ms_s, ms_t = g("scratch", "MAE.1"), g("transfer", "MAE.1")
    c_s, c_t = g("scratch", "coverage80"), g("transfer", "coverage80")
    cs_s, cs_t = g("scratch", "coverage80.1"), g("transfer", "coverage80.1")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    x0, x1 = 0, 1
    axA.bar(x0, m_s, color=C_SC, label="Scratch")
    axA.bar(x1, m_t, color=C_TF, label="Transfer")
    axA.errorbar([x0, x1], [m_s, m_t], yerr=[ms_s, ms_t], fmt="none",
                 c="k", capsize=4)
    axA.set_xticks([x0, x1])
    axA.set_xticklabels(["Scratch", "Transfer"])
    axA.set_ylabel("MAE (lower better)")
    axA.set_title("Point accuracy")
    axA.text(x0, m_s + ms_s + 0.06, f"{m_s:.2f}", ha="center", fontsize=8)
    axA.text(x1, m_t + ms_t + 0.06, f"{m_t:.2f}", ha="center", fontsize=8)
    axA.legend(fontsize=8)
    axB.bar(x0, c_s, color=C_SC)
    axB.bar(x1, c_t, color=C_TF)
    axB.errorbar([x0, x1], [c_s, c_t], yerr=[cs_s, cs_t], fmt="none",
                 c="k", capsize=4)
    axB.axhline(0.80, ls="--", c="k", lw=1, alpha=0.7)
    axB.set_ylim(0, 1.0)
    axB.set_xticks([x0, x1])
    axB.set_xticklabels(["Scratch", "Transfer"])
    axB.set_ylabel("Coverage@80")
    axB.set_title("Marginal calibration")
    axB.text(x0, c_s + cs_s + 0.02, f"{c_s:.2f}", ha="center", fontsize=8)
    axB.text(x1, c_t + cs_t + 0.02, f"{c_t:.2f}", ha="center", fontsize=8)
    fig.suptitle("Cross-region transfer: Chicago $\\to$ Austin", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "transfer_austin.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote transfer_austin.png")


# --------------------------------------------------------------------------- #
# Missing-data robustness (MP)                                                #
# --------------------------------------------------------------------------- #
def robustness_figure(results: Path, out: Path) -> None:
    df = pd.read_csv(results / "robust_mp_p2_meanstd.csv", index_col=0)
    df["MAE_num"] = pd.to_numeric(df["MAE"], errors="coerce")
    df = df[df["MAE_num"].notna()].copy()
    df["mask_num"] = pd.to_numeric(df.index, errors="coerce")
    df = df.sort_values("mask_num")
    masks = df["mask_num"].tolist()
    mae = df["MAE_num"].tolist()
    cov = pd.to_numeric(df["coverage80"], errors="coerce").tolist()
    fig, axL = plt.subplots(figsize=(6.2, 3.4))
    axR = axL.twinx()
    axR.grid(False)
    axL.plot(masks, mae, "-o", color=C_TF, lw=2, label="MAE")
    axL.set_xlabel("Missing-data fraction (random mask)")
    axL.set_ylabel("MAE", color=C_TF)
    axR.plot(masks, cov, "-s", color=C_CF, lw=2, label="Coverage@80")
    axR.axhline(0.80, ls="--", c="k", lw=1, alpha=0.7)
    axR.set_ylabel("Coverage@80", color=C_CF)
    axR.set_ylim(0, 1)
    axL.set_title("Missing-data robustness — Madhya Pradesh panel")
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out / "robustness_mp.png")
    plt.close(fig)
    print("  wrote robustness_mp.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                    help="dir with the persisted P2 CSV/JSON result files")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output figures dir (paper_b/figures)")
    args = ap.parse_args()
    _setup()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Generating Paper-2 figures from {args.results} -> {args.out}")
    condcal_figure(args.results, "chicago", args.out)
    condcal_figure(args.results, "austin", args.out)
    probbase_figure(args.results, "chicago", "p2", args.out)
    probbase_figure(args.results, "austin", "p2austin", args.out)
    transfer_figure(args.results, args.out)
    robustness_figure(args.results, args.out)
    print("Paper-2 figures complete.")


if __name__ == "__main__":
    main()
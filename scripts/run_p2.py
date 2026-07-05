"""Paper 2 experiment driver.

Runs, all resume-safe (re-run to continue after a Colab disconnect):
  1. Cross-region transfer: pretrain calibrated HASI-Net on Chicago, transfer to
     MP (resolution-agnostic weights) vs from-scratch, 5 seeds, L2-SP.
  2. Calibrated-vs-point head on MP and Chicago (5 seeds each).
  3. Missing-data robustness on MP (mask 0/10/25/50%, 5 seeds).
  4. Diebold-Mariano pairwise tests on Chicago (HASI-Net vs GraphWaveNet/HA).
  5. Friedman + Nemenyi and bootstrap CIs over the saved P1 per-seed tables
     (reads results/metrics_p1_chicago_h*_perseed.csv if present).

Usage:
  python scripts/run_p2.py
  python scripts/run_p2.py --force            # rerun every seed
  python scripts/run_p2.py --seeds 42 1 2 3 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from hasi_net.config import Config, MADHYA_PRADESH, RESULTS_DIR
from hasi_net.transfer import run_transfer_vs_scratch
from hasi_net.p2_experiments import (run_calibrated_vs_point,
                                     run_missing_data_robustness,
                                     run_dm_comparison)
from hasi_net.stats import friedman_nemenyi, bootstrap_ci

SEEDS = [42, 1, 2, 3, 4]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    seeds = args.seeds

    # --- Configs ---------------------------------------------------------
    cfg_chi = Config(target_region=MADHYA_PRADESH, use_chicago_benchmark=True,
                     chicago_year_start=2015, chicago_year_end=2024,
                     device="cuda", lookback=12, horizon=3,
                     epochs=80, batch_size=64, lr=1e-3, hidden_dim=64,
                     loss_type="nb", calibrated_head=True, pso_enabled=False)
    cfg_mp = Config(target_region=MADHYA_PRADESH, device="cuda",
                    lookback=4, horizon=2, epochs=100, batch_size=16,
                    lr=5e-4, hidden_dim=64, loss_type="nb",
                    calibrated_head=True, pso_enabled=False,
                    patience=15)
    # Austin: monthly + multi-crime like Chicago (apples-to-apples transfer),
    # 10 council districts, clean family_violence DV flag. Same temporal window
    # as Chicago (lookback 12 / horizon 3) so the Chicago-pretrained temporal
    # encoder transfers with no re-windowing.
    cfg_aus = Config(target_region=MADHYA_PRADESH, use_chicago_benchmark=True,
                     chicago_year_start=2015, chicago_year_end=2024,
                     device="cuda", lookback=12, horizon=3,
                     epochs=80, batch_size=64, lr=1e-3, hidden_dim=64,
                     loss_type="nb", calibrated_head=True, pso_enabled=False)

    print("=== 1. Cross-region transfer (Chicago -> MP) ===")
    run_transfer_vs_scratch(cfg_chi, cfg_mp, seeds=seeds, lam=1e-3,
                            tag="p2", force=args.force, verbose=True)

    print("\n=== 1b. Cross-region transfer (Chicago -> Austin) ===")
    # Reuses the Chicago pretrain cached by step 1 (pretrain_tag="p2") -- no
    # duplicate pretraining. Apples-to-apples monthly transfer; the strong
    # transfer case vs the data-scarce Chicago->MP.
    run_transfer_vs_scratch(cfg_chi, cfg_aus, seeds=seeds, lam=1e-3,
                            tag="p2chi_aus", pretrain_tag="p2",
                            source="chicago", target="austin",
                            force=args.force, verbose=True)

    print("\n=== 2a. Calibrated vs point head (MP) ===")
    run_calibrated_vs_point("mp", cfg_mp, seeds=seeds, tag="p2",
                            force=args.force, verbose=True)

    print("\n=== 2b. Calibrated vs point head (Chicago) ===")
    run_calibrated_vs_point("chicago", cfg_chi, seeds=seeds, tag="p2",
                            force=args.force, verbose=True)

    print("\n=== 2c. Calibrated vs point head (Austin) ===")
    run_calibrated_vs_point("austin", cfg_aus, seeds=seeds, tag="p2austin",
                            force=args.force, verbose=True)

    print("\n=== 3. Missing-data robustness (MP) ===")
    run_missing_data_robustness("mp", cfg_mp, seeds=seeds,
                                fractions=[0.0, 0.1, 0.25, 0.5], tag="p2",
                                force=args.force, verbose=True)

    print("\n=== 4. Diebold-Mariano pairwise (Chicago h=3) ===")
    run_dm_comparison("chicago", cfg_chi.override(calibrated_head=False,
                                                  loss_type="log1p_mse"),
                      seeds=seeds,
                      pairs=[("HASI-Net", "GraphWaveNet"),
                             ("HASI-Net", "HA"),
                             ("GraphWaveNet", "HA")],
                      tag="p1", h_step=3, force=args.force, verbose=True)

    print("\n=== 5. Friedman + Nemenyi and bootstrap CIs over P1 tables ===")
    _stats_over_p1(seeds)

    print("\nPaper 2 run complete.")


def _stats_over_p1(seeds):
    """Friedman/Nemenyi across models x horizons and bootstrap CIs on MAE.
    Reads the P1 per-seed CSVs if present; skips gracefully if not."""
    models = ["HA", "LSTM", "STGCN", "GraphWaveNet", "DCRNN", "MTGNN",
              "InformerOnly", "HASI-Net"]
    blocks = []   # rows = [MAE per model] for each (horizon, seed) block
    used = []
    for h in (3, 6, 12):
        f = RESULTS_DIR / f"metrics_p1_chicago_h{h}_perseed.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        for seed in seeds:
            row = []
            for m in models:
                v = df[(df.model == m) & (df.seed == seed)]["MAE"].values
                if len(v):
                    row.append(float(v[0]))
                else:
                    row.append(np.nan)
            if not np.isnan(row).any():
                blocks.append(row)
                used.append((h, seed))
    if len(blocks) < 2:
        print("  (P1 per-seed tables not found -- run scripts/run_p1_multiseed.py first)")
        return
    perf = np.array(blocks)
    fr = friedman_nemenyi(perf, alpha=0.1)
    print(f"  Friedman: chi2={fr['chi2']:.3f} p={fr['p']:.4g} "
          f"(n_blocks={fr['n_blocks']}, k={fr['k']}, CD={fr['cd']:.3f})")
    ranks = sorted(zip(models, fr["mean_ranks"]), key=lambda x: x[1])
    for m, r in ranks:
        print(f"    {m:14s} mean rank {r:.2f}")
    # Bootstrap 95% CI on each model's MAE.
    print("  Bootstrap 95% CI on MAE (across blocks):")
    for j, m in enumerate(models):
        m_, lo, hi = bootstrap_ci(perf[:, j], confidence=0.95, seed=0)
        print(f"    {m:14s} {m_:.4f} [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
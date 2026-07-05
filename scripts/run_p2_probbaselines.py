"""Paper-2 probabilistic baselines (conformal + quantile regression) on Chicago
and Austin, on the SAME HASI-Net backbone as the calibrated head. Produces a
4-way table (calibrated | point | conformal | quantreg) per city so the
calibration contribution is judged against real probabilistic competitors, not
only the internal point head. Run on the Colab GPU.

  * conformal  -- split conformal around the point head (per-crime residual
                  scores on the val split; marginal 80% coverage by
                  construction; sharpness/CRPS are the real comparison). MAE
                  reproduces the cached `point` condition (consistency check).
  * quantreg   -- same persistence-carry quantile head, PURE pinball loss
                  (no NB/ZINB reg, no sparsity gate). Isolates the calibrated
                  multi-objective loss. Point forecast = the median quantile.

Both merge with the existing calvspt per-seed CSV (tag "p2" for Chicago,
"p2austin" for Austin) into probbase_{ds}_{tag}_4way_meanstd.csv.

Usage:
  python scripts/run_p2_probbaselines.py
  python scripts/run_p2_probbaselines.py --datasets chicago --seeds 42 1 2 3 4
  python scripts/run_p2_probbaselines.py --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hasi_net.config import Config, MADHYA_PRADESH
from hasi_net.prob_baselines import run_prob_baselines

SEEDS = [42, 1, 2, 3, 4]


def _cfg():
    # Chicago-like monthly 4-crime config (calibrated_head=True is the base; the
    # conformal condition derives the point variant internally). Matches the
    # existing calvspt configs so the conformal MAE reproduces `point` exactly.
    return Config(target_region=MADHYA_PRADESH, use_chicago_benchmark=True,
                  chicago_year_start=2015, chicago_year_end=2024,
                  device="cuda", lookback=12, horizon=3, epochs=80,
                  batch_size=64, lr=1e-3, hidden_dim=64, loss_type="nb",
                  calibrated_head=True, pso_enabled=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--datasets", nargs="*", default=["chicago", "austin"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    seeds = args.seeds

    if "chicago" in args.datasets:
        print("=== Chicago: conformal + quantreg ===", flush=True)
        run_prob_baselines("chicago", _cfg(), seeds, tag="p2",
                           calvspt_tag="p2", force=args.force, verbose=True)
    if "austin" in args.datasets:
        print("\n=== Austin: conformal + quantreg ===", flush=True)
        run_prob_baselines("austin", _cfg(), seeds, tag="p2austin",
                           calvspt_tag="p2austin", force=args.force, verbose=True)

    print("\nPaper-2 prob-baselines run complete.", flush=True)


if __name__ == "__main__":
    main()
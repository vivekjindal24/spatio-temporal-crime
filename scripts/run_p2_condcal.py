"""Paper-2 CONDITIONAL calibration: calibrated head vs split conformal, per
count-bucket / per-crime / per-node. Tests whether the calibrated head's
heteroscedastic intervals give more UNIFORM (conditional) coverage than
conformal's homoscedastic per-crime width -- the axis where the calibrated head
can genuinely beat conformal, which is sharper on the (marginal) aggregate.

Freshly trains both heads per seed (persisted to .pt for reproducibility, since
CUDA retrains are not bit-identical) so the calibrated-vs-conformal conditional
comparison is internally consistent. Run on the Colab GPU.

Usage:
  python scripts/run_p2_condcal.py
  python scripts/run_p2_condcal.py --datasets chicago --seeds 42 1 2 3 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hasi_net.config import Config, MADHYA_PRADESH
from hasi_net.prob_baselines import run_conditional_calibration

SEEDS = [42, 1, 2, 3, 4]


def _cfg():
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
    for ds in args.datasets:
        print(f"=== {ds}: conditional calibration (calibrated vs conformal) ===",
              flush=True)
        run_conditional_calibration(ds, _cfg(), args.seeds, tag="p2cc",
                                    force=args.force, verbose=True)
    print("\nPaper-2 conditional-calibration run complete.", flush=True)


if __name__ == "__main__":
    main()
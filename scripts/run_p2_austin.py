"""Paper-2 Austin-only driver (run on Colab GPU).

Runs just the two new Austin sections of the P2 program, so the cached
Chicago/MP results on Drive are not touched:

  1b. Cross-region transfer Chicago -> Austin (apples-to-apples monthly,
      4-crime, lookback 12 / horizon 3). REUSES the cached Chicago pretrain
      (hasi_net_p2_pretrain.pt, written by the original P2 run) via
      pretrain_tag="p2" -- no duplicate pretraining. Writes
      transfer_p2chi_aus_* + summary_p2chi_aus_transfer.json.
  2c. Calibrated-vs-point head on Austin (5 seeds). Writes
      calvspt_austin_p2austin_* + summary_p2austin_calvspt.json.

All outputs go to HASI_RESULTS_DIR (Drive) so they survive Colab disconnects.
Resume-safe: re-run continues from per-seed checkpoints.

Usage:
  python scripts/run_p2_austin.py
  python scripts/run_p2_austin.py --seeds 42 1 2 3 4 --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hasi_net.config import Config, MADHYA_PRADESH
from hasi_net.transfer import run_transfer_vs_scratch
from hasi_net.p2_experiments import run_calibrated_vs_point

SEEDS = [42, 1, 2, 3, 4]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    seeds = args.seeds

    # Chicago (source) and Austin (target) share the monthly, 4-crime,
    # lookback-12 / horizon-3 structure -- the resolution-agnostic temporal +
    # graph weights transfer with no re-windowing.
    cfg_chi = Config(target_region=MADHYA_PRADESH, use_chicago_benchmark=True,
                     chicago_year_start=2015, chicago_year_end=2024,
                     device="cuda", lookback=12, horizon=3,
                     epochs=80, batch_size=64, lr=1e-3, hidden_dim=64,
                     loss_type="nb", calibrated_head=True, pso_enabled=False)
    cfg_aus = Config(target_region=MADHYA_PRADESH, use_chicago_benchmark=True,
                     chicago_year_start=2015, chicago_year_end=2024,
                     device="cuda", lookback=12, horizon=3,
                     epochs=80, batch_size=64, lr=1e-3, hidden_dim=64,
                     loss_type="nb", calibrated_head=True, pso_enabled=False)

    print("=== 1b. Cross-region transfer (Chicago -> Austin) ===", flush=True)
    run_transfer_vs_scratch(cfg_chi, cfg_aus, seeds=seeds, lam=1e-3,
                            tag="p2chi_aus", pretrain_tag="p2",
                            source="chicago", target="austin",
                            force=args.force, verbose=True)

    print("\n=== 2c. Calibrated vs point head (Austin) ===", flush=True)
    run_calibrated_vs_point("austin", cfg_aus, seeds=seeds, tag="p2austin",
                            force=args.force, verbose=True)

    print("\nPaper-2 Austin run complete.", flush=True)


if __name__ == "__main__":
    main()
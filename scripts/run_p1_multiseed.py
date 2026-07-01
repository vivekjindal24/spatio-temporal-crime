"""Paper 1 multi-seed experiment driver.

Runs, on the Chicago benchmark:
  * 8 models (HA, LSTM, STGCN, GraphWaveNet, DCRNN, MTGNN, InformerOnly,
    HASI-Net) across 5 seeds, with PSO hyperparameter search run once and
    frozen for all seeds;
  * the component ablation across the same 5 seeds.

Outputs (written to results/):
  metrics_p1_chicago_meanstd.csv      model x {metric}_mean/_std
  metrics_p1_chicago_perseed.csv      long-form per-seed metrics
  ablation_p1_chicago_meanstd.csv     variant x {metric}_mean/_std
  summary_p1_chicago_multiseed.json   full summary + PSO config
  multiseed_chicago_p1_chicago_seed{S}.json   per-seed checkpoints (resume)
  ablation_chicago_p1_chicago_seed{S}.json    per-seed ablation checkpoints
  pso_convergence_p1_chicago.png, channel_weights_p1_chicago.png

Resume: just re-run -- completed seeds are skipped. Use --force to rerun all.

Usage:
  python scripts/run_p1_multiseed.py            # Chicago, 5 seeds, PSO on
  python scripts/run_p1_multiseed.py --no-pso   # skip PSO (use default cfg)
  python scripts/run_p1_multiseed.py --force    # rerun every seed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hasi_net.config import Config, MADHYA_PRADESH
from hasi_net.multiseed import run_multiseed, run_ablation_multiseed

SEEDS = [42, 1, 2, 3, 4]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--horizons", type=int, nargs="*", default=[3, 6, 12],
                    help="forecast horizons (months) to evaluate; all reported")
    ap.add_argument("--no-pso", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ablation-epochs", type=int, default=40)
    args = ap.parse_args()

    for h in args.horizons:
        cfg = Config(
            target_region=MADHYA_PRADESH,
            use_chicago_benchmark=True,
            chicago_year_start=2015, chicago_year_end=2024,
            device="cuda",
            lookback=12, horizon=h,
            epochs=80, batch_size=64, lr=1e-3,
            hidden_dim=64, n_graph_layers=2, n_attn_heads=4,
            loss_type="log1p_mse",
            pso_enabled=not args.no_pso,
        )
        tag = f"p1_chicago_h{h}"
        print(f"\n===== horizon={h} months (tag={tag}) =====")
        print("P1 config:", cfg.to_dict())
        run_multiseed("chicago", cfg, seeds=args.seeds, tag=tag,
                      pso=not args.no_pso, force=args.force, verbose=True)
        print(f"Wrote metrics_{tag}_meanstd.csv / _perseed.csv")
        # Ablation only at the primary horizon to bound compute.
        if h == args.horizons[0]:
            run_ablation_multiseed("chicago", cfg, seeds=args.seeds, tag=tag,
                                   epochs=args.ablation_epochs,
                                   force=args.force, verbose=True)
            print(f"Wrote ablation_{tag}_meanstd.csv")
    print("\nPaper 1 multi-seed run complete (horizons:", args.horizons, ").")


if __name__ == "__main__":
    main()
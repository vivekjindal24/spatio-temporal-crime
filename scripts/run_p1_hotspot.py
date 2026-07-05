"""Paper-1 hotspot reframe driver.

Runs the 8-model P1 benchmark (HA, LSTM, STGCN, GraphWaveNet, DCRNN, MTGNN,
InformerOnly, HASI-Net) at the primary horizon (h=3) on Chicago and evaluates it
on the operationally relevant axis -- high-crime EVENT verification (per-node
90th-pct threshold, leak-free from training) and top-k hotspot Hit@k -- where
the HA persistence carry is structurally disadvantaged. Predictions are
persisted to .npz so the offline hotspot metrics are reproducible.

PSO parity: by default the frozen PSO config from the cached P1 multi-seed
benchmark (``summary_{bench_tag}_multiseed.json``) is reused, so the hotspot
models are the SAME trained models as the published point-MAE table. Pass
``--no-pso-cache`` to use the default config instead.

Outputs (results/):
  preds_chicago_p1hot_seed{S}_{model}.npz       per-seed structured preds
  hotspot_p1hot_chicago_seed{S}.json            per-seed hotspot metrics
  hotspot_p1hot_chicago_perseed.csv             long-form per-seed
  hotspot_p1hot_chicago_meanstd.csv            model x {metric}_mean/_std
  summary_p1hot_chicago.json                    full summary + config

Usage:
  python scripts/run_p1_hotspot.py
  python scripts/run_p1_hotspot.py --seeds 42 1 2 3 4
  python scripts/run_p1_hotspot.py --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hasi_net.config import Config, MADHYA_PRADESH, RESULTS_DIR
from hasi_net.hotspot import run_hotspot

SEEDS = [42, 1, 2, 3, 4]
BENCH_TAG = "p1_prelim"          # cached P1 multi-seed benchmark (PSO config)


def _load_pso_cfg(bench_tag: str, cfg: Config):
    """Reuse the frozen PSO config from the cached P1 benchmark for parity."""
    path = RESULTS_DIR / f"summary_{bench_tag}_multiseed.json"
    if not path.exists():
        print(f"[hotspot] no cached PSO config at {path}; using default cfg",
              flush=True)
        return None
    d = json.loads(path.read_text())
    pso = d.get("pso_config")
    if not pso:
        return None
    return cfg.override(**pso)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--datasets", nargs="*", default=["chicago"])
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--bench-tag", default=BENCH_TAG,
                    help="cached P1 benchmark tag whose PSO config to reuse")
    ap.add_argument("--no-pso-cache", action="store_true",
                    help="ignore the cached PSO config; use default cfg")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for ds in args.datasets:
        cfg = Config(target_region=MADHYA_PRADESH, use_chicago_benchmark=True,
                     chicago_year_start=2015, chicago_year_end=2024,
                     device="cuda", lookback=12, horizon=args.horizon,
                     epochs=80, batch_size=64, lr=1e-3, hidden_dim=64,
                     n_graph_layers=2, n_attn_heads=4, loss_type="log1p_mse",
                     pso_enabled=False)
        pso_cfg = None if args.no_pso_cache else _load_pso_cfg(args.bench_tag, cfg)
        if pso_cfg is not None:
            print(f"[hotspot] reusing PSO config from tag={args.bench_tag}",
                  flush=True)
        tag = "p1hot"
        print(f"\n===== {ds}: hotspot reframe (tag={tag}, h={args.horizon}) =====",
              flush=True)
        run_hotspot(ds, cfg, seeds=args.seeds, tag=tag, pso_cfg=pso_cfg,
                    force=args.force, verbose=True)
    print("\nPaper-1 hotspot reframe complete.", flush=True)


if __name__ == "__main__":
    main()
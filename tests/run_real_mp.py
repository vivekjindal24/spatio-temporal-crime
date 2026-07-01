"""Run a real (reduced-compute) MP experiment to generate genuine figures.

Uses the real NCRB 2001-2014 + Census 2011 data. Settings are reduced from the
notebook defaults so it finishes in a few minutes on M1 MPS; the notebooks
ship with the full settings for the final paper-quality run.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hasi_net import Config
from hasi_net.experiment import run_experiment

cfg = Config(
    device="mps",
    epochs=40, batch_size=8, lr=5e-4, patience=8,
    hidden_dim=32, n_graph_layers=1, n_attn_heads=4,
    dropout=0.3, weight_decay=1e-3,            # heavy reg: only 9 windows -> must not drift
    lookback=4, horizon=2,
    loss_type="log1p_mse", focal_gamma=1.5,
    pso_enabled=False,                          # too few windows for PSO to be meaningful
    use_chicago_benchmark=False,
)
summary = run_experiment("mp", cfg, tag="mp")
import json
print("DONE. summary:")
print(json.dumps({k: summary[k] for k in
      ["dataset", "device", "n_nodes", "n_years", "best_val_mae",
       "channel_weights", "panel_meta"]}, indent=2))
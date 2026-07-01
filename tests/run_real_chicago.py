"""Real Chicago benchmark experiment (fine-grained, monthly aggregation).

This is the headline quantitative result: on fine-grained spatiotemporal data
the deep model has enough temporal resolution to outperform HA / linear
baselines. Downloads via the Socrata API and caches to data/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hasi_net import Config
from hasi_net.experiment import run_experiment

cfg = Config(
    device="mps",
    chicago_year_start=2015, chicago_year_end=2022,   # 96 months -> ~85 windows
    epochs=80, batch_size=32, lr=1e-3, patience=15,
    hidden_dim=64, n_graph_layers=2, n_attn_heads=4,
    dropout=0.25, weight_decay=1e-4,
    lookback=12, horizon=6,           # 12 in (seasonal context), 6 out
    loss_type="log1p_mse", focal_gamma=1.5,
    pso_enabled=False,
    use_chicago_benchmark=True,
)
summary = run_experiment("chicago", cfg, tag="chicago")
import json
print("DONE. summary:")
print(json.dumps({k: summary[k] for k in
      ["dataset", "device", "n_nodes", "n_years", "best_val_mae",
       "channel_weights", "panel_meta"]}, indent=2))
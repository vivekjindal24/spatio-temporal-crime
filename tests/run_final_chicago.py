"""Final Chicago benchmark run WITH adaptive PSO enabled (the intended
methodology). Produces the headline result table + pso_convergence figure for
the papers. Modest PSO budget so it finishes on M1 MPS in reasonable time.
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
    pso_enabled=True, pso_iters=4, pso_particles=5, pso_swarms=2,
    use_chicago_benchmark=True,
)
summary = run_experiment("chicago", cfg, tag="chicago")
import json
print("DONE. summary:")
print(json.dumps({k: summary[k] for k in
      ["dataset", "device", "n_nodes", "n_years", "best_val_mae",
       "channel_weights", "panel_meta"]}, indent=2))
print("PSO-selected config:")
print(json.dumps(summary["config"], indent=2))
"""Generate the HASI-Net deployment notebooks (M1 + Colab) as valid .ipynb.

Run:  python3 notebooks/build_notebooks.py
Emits notebooks/hasi_net_m1.ipynb and notebooks/hasi_net_colab.ipynb.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True),
            "outputs": [], "execution_count": None}


COMMON_TITLE = '''# HASI-Net: Heterogeneous Adaptive Spatio-temporal Informer Network
## Forecasting women-centric crimes on aggregated district data — Madhya Pradesh

**Complete deployment notebook.** End-to-end pipeline:

1. Download real public datasets (NCRB district women-crime tables, Census 2011
   socioeconomic features, MP district boundaries, Chicago benchmark).
2. Build the heterogeneous graph (geographic + socioeconomic + learnable adaptive).
3. PSO-tune hyperparameters (adaptive inertia, multi-swarm).
4. Train HASI-Net + baselines (HA, LSTM, ST-GCN, Informer-only).
5. Evaluate (MAE, RMSE, RMSLE, WAPE, R^2, CSI) and generate publication figures.

This notebook is the **reproducible companion** to the paper. Re-running it
top-to-bottom regenerates every result table and figure cited there.

> Architecture and novel contributions are in `src/hasi_net/`. The notebook is
> a thin orchestration layer over that package.
'''

ENV_MD_M1 = '''## 0. Environment — Apple MacBook M1

On Apple Silicon we use the **MPS** (Metal Performance Shaders) backend built
into PyTorch. Notes:

* First MPS kernel compile takes ~30s; subsequent runs are fast.
* Batch sizes are kept modest (16) to fit unified memory comfortably.
* `float32` throughout — MPS fp16 paths are still uneven across ops.
* If MPS is unavailable the code falls back to CPU automatically.
'''

ENV_MD_COLAB = '''## 0. Environment — Google Colab (T4/V100 GPU)

* Runtime type: **GPU** (Colab Free gives a T4; Pro gives V100/A100).
* Results/models are persisted to Google Drive so they survive session resets.
* The repo's `src/hasi_net/` package is put on the path automatically; if you
  are running without the repo, a setup cell clones it.
'''

INSTALL_M1 = '''# Install dependencies (M1). torch >= 2.0 ships MPS support on Apple Silicon.
import subprocess, sys
def pip(pkg): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
for p in ["torch", "numpy", "pandas", "requests", "matplotlib",
          "pyarrow", "geopandas", "shapely"]:
    try:
        __import__(p)
    except Exception:
        pip(p)
print("deps ready")
'''

INSTALL_COLAB = '''# Install dependencies (Colab GPU runtime).
import subprocess, sys
def pip(pkg): subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
for p in ["torch", "numpy", "pandas", "requests", "matplotlib",
          "pyarrow", "geopandas", "shapely"]:
    try:
        __import__(p)
    except Exception:
        pip(p)
print("deps ready")
'''

SETUP_M1 = '''# Path + seed + device.
import sys, pathlib, torch, numpy as np
REPO = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(REPO / "src"))

from hasi_net import Config, select_device, set_seed, viz
from hasi_net.experiment import run_experiment
viz.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = select_device("mps")
set_seed(42)
print("Working dir:", REPO)
print("Device:", DEVICE)
'''

SETUP_COLAB = '''# Mount Drive for persistence and put the package on the path.
import sys, pathlib, torch, numpy as np, os
from google.colab import drive
drive.mount("/content/drive")

# If the repo is in Drive, point REPO there; otherwise clone it.
REPO = pathlib.Path("/content/spatio-temporal-crime")
if not REPO.exists():
    REPO = pathlib.Path("/content/drive/MyDrive/spatio-temporal-crime")
if not (REPO / "src" / "hasi_net").exists():
    import subprocess
    subprocess.check_call(["git", "clone", "--depth", "1",
        "https://github.com/<your-user>/spatio-temporal-crime", str(REPO)])
sys.path.insert(0, str(REPO / "src"))

from hasi_net import Config, select_device, set_seed, viz
from hasi_net.experiment import run_experiment
viz.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = select_device("cuda")
set_seed(42)
print("Repo:", REPO)
print("Device:", DEVICE)
'''

CONFIG_M1 = '''# M1-tuned configuration.
cfg = Config(
    device="mps",
    epochs=80, batch_size=16, lr=1e-3,
    hidden_dim=64, n_graph_layers=2, n_attn_heads=4,
    lookback=4, horizon=2,
    loss_type="log1p_mse", focal_gamma=1.5,
    pso_enabled=True, pso_swarms=3, pso_particles=8, pso_iters=6,
    use_chicago_benchmark=True,
)
print("Config:"); print(cfg.to_dict())
'''

CONFIG_COLAB = '''# Colab-tuned configuration (more compute -> larger search + epochs).
cfg = Config(
    device="cuda",
    epochs=120, batch_size=64, lr=1e-3,
    hidden_dim=96, n_graph_layers=2, n_attn_heads=4,
    lookback=4, horizon=2,
    loss_type="log1p_mse", focal_gamma=1.5,
    pso_enabled=True, pso_swarms=4, pso_particles=12, pso_iters=8,
    use_chicago_benchmark=True,
)
print("Config:"); print(cfg.to_dict())
'''

DATA_PREVIEW = '''## 1. Data acquisition & preview
Datasets are downloaded (cached under `data/`) on first run. The MP panel uses
the contiguous, fully-real NCRB 2001-2014 district women-crime block (7 IPC
categories) and merges Census 2011 socioeconomic node features. The 2015-2016
and 2017-2022 NCRB releases are not reliably machine-readable, so they are
excluded rather than forward-filled.
'''

DATA_CODE = '''from hasi_net.data import build_mp_panel, build_graphs
panel = build_mp_panel(cfg)
print("Counts tensor [T, N, C]:", panel.counts.shape)
print("Districts (N):", len(panel.districts), "— e.g.", panel.districts[:5])
print("Years (T):", panel.years[0], "->", panel.years[-1], f"({len(panel.years)} steps)")
print("Categories (C):", panel.categories)
print("Socioeconomic features [N, F]:", panel.node_feats.shape)
print("Source:", panel.meta["source"])
'''

GRAPH_MD = '''## 2. Heterogeneous graph
* `A_geo` — rook contiguity from the MP district boundary GeoJSON.
* `A_socio` — kNN cosine similarity on census features.
* `A_adaptive` — learned at training time (inside the model).
'''

GRAPH_CODE = '''import numpy as np
A_geo, A_socio = build_graphs(panel.districts, panel.node_feats, cfg)
print("A_geo density:", round(A_geo.mean(), 4), " A_socio density:", round(A_socio.mean(), 4))
print("A_geo edges:", int((A_geo > 0).sum() // 2),
      " A_socio edges:", int((A_socio > 0).sum() // 2))
'''

RUN_MP_MD = '''## 3. Full experiment — Madhya Pradesh (target region)
`run_experiment` does: assemble panel + graphs -> adaptive PSO -> train
HASI-Net -> train baselines -> evaluate on the held-out test years -> write
figures + `results/summary_mp_<tag>.json`.
'''

RUN_MP_CODE = '''summary_mp = run_experiment("mp", cfg, tag="mp")
'''

DISPLAY_MP = '''import pandas as pd, json
from IPython.display import Image, display
metrics = pd.read_csv(REPO / "results" / "metrics_mp.csv", index_col=0)
print("Test-set metrics (MP):"); print(metrics.round(4))

for fig in ["training_curves_mp.png", "pso_convergence_mp.png",
            "channel_weights_mp.png", "comparison_MAE_mp.png",
            "comparison_CSI_mp.png", "district_risk_mp.png",
            "pred_vs_actual_mp.png", "choropleth_mp.png"]:
    p = REPO / "results" / fig
    if p.exists():
        print("\\n---", fig, "---"); display(Image(filename=str(p)))
'''

ABLATION_MD = '''## 3b. Component ablation
Isolates the contribution of each HASI-Net component (adaptive graph,
socioeconomic channel, spatial block, count-aware loss) by training short
variants with one component removed at a time.
'''

ABLATION_CODE = '''from hasi_net.experiment import run_ablation
abl = run_ablation("mp", cfg, tag="mp", epochs=40, verbose=False)
print(abl.round(4))
from IPython.display import Image, display
p = REPO / "results" / "ablation_mp.png"
if p.exists(): display(Image(filename=str(p)))
'''

CHICAGO_MD = '''## 4. Benchmark — Chicago (fine-grained, exact lat/long)
Validates that HASI-Net also works on *precise* spatial data (objective 5 in the
proposal). Chicago incidents are pulled via the Socrata API and aggregated to a
community-area x month grid. We use a shorter window to keep the download
manageable.
'''

CHICAGO_CODE = '''cfg_chi = cfg.override(chicago_year_start=2018, chicago_year_end=2022,
                         epochs=60, pso_iters=4, pso_particles=6, pso_swarms=2)
summary_chi = run_experiment("chicago", cfg_chi, tag="chicago")
'''

RESULTS_MD = '''## 5. Results summary & reproducibility
All artifacts are under `results/`: model weights (`hasi_net_<tag>.pt`),
metrics CSV, and a `summary_<tag>.json` capturing config + metrics + provenance.
The figures above are exactly those referenced in the paper.
'''

RESULTS_CODE = '''import json, pathlib
res = pathlib.Path(REPO / "results")
print("Artifacts:"); [print(" ", p.name, f"{p.stat().st_size//1024} KB") for p in sorted(res.glob("*"))]
print("\\nMP summary:"); print(json.dumps({k: summary_mp[k] for k in
      ["dataset","device","n_nodes","n_years","panel_meta","channel_weights","best_val_mae"]}, indent=2))
'''

NEXT_MD = '''## 6. Next steps
* Inspect `results/summary_mp.json` for the PSO-selected hyperparameters.
* The two research papers (methods + applied MP) are generated from these
  real outputs in `paper_a/` and `paper_b/`.
* To rerun with a different seed/horizon, edit `cfg` in the config cell and
  re-execute from section 3.
'''


def build(platform: str) -> dict:
    if platform == "m1":
        env, install, setup, config = ENV_MD_M1, INSTALL_M1, SETUP_M1, CONFIG_M1
    else:
        env, install, setup, config = ENV_MD_COLAB, INSTALL_COLAB, SETUP_COLAB, CONFIG_COLAB
    cells = [
        md(COMMON_TITLE),
        md(env),
        code(install),
        code(setup),
        code(config),
        md(DATA_PREVIEW),
        code(DATA_CODE),
        md(GRAPH_MD),
        code(GRAPH_CODE),
        md(RUN_MP_MD),
        code(RUN_MP_CODE),
        code(DISPLAY_MP),
        md(ABLATION_MD),
        code(ABLATION_CODE),
        md(CHICAGO_MD),
        code(CHICAGO_CODE),
        md(RESULTS_MD),
        code(RESULTS_CODE),
        md(NEXT_MD),
    ]
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9"}},
        "nbformat": 4, "nbformat_minor": 5}
    return nb


for plat, fname in [("m1", "hasi_net_m1.ipynb"), ("colab", "hasi_net_colab.ipynb")]:
    path = OUT / fname
    path.write_text(json.dumps(build(plat), indent=1))
    print("wrote", path, path.stat().st_size, "bytes")
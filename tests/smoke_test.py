"""Smoke test: exercise HASI-Net end-to-end on synthetic data.

Bypasses all network downloads so we can verify the model, losses, training
loop, PSO, baselines and figure generation actually run. Run:
    python3 tests/smoke_test.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from hasi_net import Config
from hasi_net.train import (build_model, make_splits, train_one, evaluate,
                            WindowDataset, select_device, set_seed)
from hasi_net.baselines import build_baseline, BASELINES
from hasi_net.pso import run_pso, short_fitness
from hasi_net import viz
from hasi_net.experiment import assemble_panel, run_experiment


def make_synthetic_panel(n_nodes=12, T=20, C=3, F=4, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(3.0, size=(T, n_nodes, C)).astype(np.float32)
    # inject a trend so the model has something to learn
    counts += np.arange(T)[:, None, None].astype(np.float32) * 0.2
    node_feats = rng.standard_normal((n_nodes, F)).astype(np.float32)
    A_geo = (rng.random((n_nodes, n_nodes)) > 0.7).astype(np.float32)
    A_geo = np.maximum(A_geo, A_geo.T); np.fill_diagonal(A_geo, 1.0)
    A_geo = A_geo / A_geo.sum(1, keepdims=True)
    A_socio = (rng.random((n_nodes, n_nodes)) > 0.7).astype(np.float32)
    A_socio = np.maximum(A_socio, A_socio.T); np.fill_diagonal(A_socio, 1.0)
    A_socio = A_socio / A_socio.sum(1, keepdims=True)
    years = list(range(2001, 2001 + T))
    districts = [f"D{i}" for i in range(n_nodes)]
    cats = [f"cat{i}" for i in range(C)]

    class P:  # mimic hasi_net.data.Panel
        pass
    p = P()
    p.counts = counts; p.node_feats = node_feats
    p.years = years; p.districts = districts; p.categories = cats
    p.meta = {"region": "synthetic", "source": "smoke test"}
    return p


def main():
    set_seed(42)
    device = select_device("auto")
    print("device:", device)
    cfg = Config(epochs=4, batch_size=4, pso_iters=2, pso_particles=3,
                 pso_swarms=1, pso_enabled=True, hidden_dim=16, n_attn_heads=2,
                 n_graph_layers=1, lookback=4, horizon=2)
    panel_obj = make_synthetic_panel()
    panel = assemble_panel(panel_obj, cfg)
    print("panel counts:", panel["counts"].shape)

    # Build + train HASI-Net directly.
    tr, va, te = make_splits(panel["counts"].shape[0], cfg.lookback, cfg.horizon)
    trl = DataLoader(WindowDataset(panel["counts"], cfg.lookback, cfg.horizon, tr), batch_size=4, shuffle=True)
    val = DataLoader(WindowDataset(panel["counts"], cfg.lookback, cfg.horizon, va), batch_size=4)
    tel = DataLoader(WindowDataset(panel["counts"], cfg.lookback, cfg.horizon, te), batch_size=4)
    model = build_model(cfg, panel, device)
    res = train_one(model, cfg, trl, val, device, verbose=True)
    m = evaluate(model, tel, device, cfg.horizon)
    print("HASI-Net test metrics:", m.as_dict())
    viz.plot_training_curves(res["history"], "smoke train", "smoke_training.png")

    # Baselines.
    for name in BASELINES:
        bm = build_baseline(name, cfg, panel, device)
        if name != "HA":
            train_one(bm, cfg, trl, val, device, verbose=False)
        bm_m = evaluate(bm, tel, device, cfg.horizon)
        print(f"  baseline {name}: {bm_m.as_dict()}")

    # PSO.
    fit = short_fitness(panel, cfg, device, epochs=2)
    best, hist = run_pso(fit, cfg, verbose=True)
    print("PSO best cfg:", {k: best.to_dict()[k] for k in
          ["hidden_dim", "n_graph_layers", "n_attn_heads", "dropout", "lr"]})
    viz.plot_pso_convergence(hist, "smoke_pso.png")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
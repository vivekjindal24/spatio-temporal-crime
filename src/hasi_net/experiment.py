"""End-to-end experiment runner used by both notebooks.

``run_experiment`` downloads data, builds the panel + graphs, (optionally)
PSO-tunes the model, trains HASI-Net and all baselines, evaluates on the test
split, and writes figures + a results JSON to ``results/``. It returns a dict
summary so the notebook can display tables inline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config, RESULTS_DIR
from .data import get_dataset
from .graph import build_graphs
from .train import (build_model, make_splits, train_one, evaluate,
                    WindowDataset, select_device, set_seed)
from .baselines import build_baseline, BASELINES
from .pso import run_pso, short_fitness
from . import viz


def assemble_panel(panel, cfg: Config) -> Dict:
    A_geo, A_socio = build_graphs(panel.districts, panel.node_feats, cfg)
    return {
        "counts": panel.counts,
        "node_feats": panel.node_feats,
        "a_geo": A_geo,
        "a_socio": A_socio,
        "years": panel.years,
        "districts": panel.districts,
        "categories": panel.categories,
        "meta": panel.meta,
    }


def _loaders(panel, cfg, split):
    T = panel["counts"].shape[0]
    tr, va, te = make_splits(T, cfg.lookback, cfg.horizon)
    return (DataLoader(WindowDataset(panel["counts"], cfg.lookback, cfg.horizon, tr),
                       batch_size=cfg.batch_size, shuffle=True),
            DataLoader(WindowDataset(panel["counts"], cfg.lookback, cfg.horizon, va),
                       batch_size=cfg.batch_size),
            DataLoader(WindowDataset(panel["counts"], cfg.lookback, cfg.horizon, te),
                       batch_size=cfg.batch_size))


def run_experiment(dataset: str, cfg: Config, pso: Optional[bool] = None,
                   tag: str = "", verbose: bool = True) -> Dict:
    set_seed(cfg.seed)
    device = select_device(cfg.device)
    if verbose:
        print(f"Device: {device}")

    panel_obj = get_dataset(dataset, cfg)
    if verbose:
        print(f"Panel {dataset}: T={panel_obj.counts.shape[0]} "
              f"N={panel_obj.counts.shape[1]} C={panel_obj.counts.shape[2]}")
        print(f"  meta: {panel_obj.meta}")
    panel = assemble_panel(panel_obj, cfg)

    pso = cfg.pso_enabled if pso is None else pso
    used_cfg = cfg
    pso_hist = None
    if pso:
        if verbose:
            print("Running adaptive PSO search...")
        fit = short_fitness(panel, cfg, device, epochs=15)
        used_cfg, pso_hist = run_pso(fit, cfg, verbose=verbose)
        if verbose:
            print(f"PSO selected: {used_cfg.to_dict()}")
        viz.plot_pso_convergence(pso_hist, f"pso_convergence_{tag}.png")

    # --- HASI-Net -----------------------------------------------------------
    tr, va, te = _loaders(panel, used_cfg, "all")
    if verbose:
        print("Training HASI-Net...")
    model = build_model(used_cfg, panel, device)
    res = train_one(model, used_cfg, tr, va, device, verbose=verbose)
    hasi_metrics = evaluate(model, te, device, used_cfg.horizon)
    viz.plot_training_curves(res["history"], f"HASI-Net training ({tag})",
                             f"training_curves_{tag}.png")
    try:
        weights = model.spatial.adj.channel_weights().cpu().numpy()
        viz.plot_channel_weights(weights, f"channel_weights_{tag}.png")
    except Exception:
        weights = None

    # --- Baselines ----------------------------------------------------------
    rows = {"HASI-Net": hasi_metrics.as_dict()}
    for name in BASELINES:
        if verbose:
            print(f"Training baseline: {name}")
        bcfg = used_cfg.override(pso_enabled=False)
        bm = build_baseline(name, bcfg, panel, device)
        if name != "HA":
            bres = train_one(bm, bcfg, tr, va, device, verbose=False)
        bm_metrics = evaluate(bm, te, device, used_cfg.horizon)
        rows[name] = bm_metrics.as_dict()
    metrics_df = pd.DataFrame(rows).T
    if verbose:
        print(metrics_df.round(4))
    metrics_df.to_csv(RESULTS_DIR / f"metrics_{tag}.csv")

    # --- Forecast + risk figures -------------------------------------------
    last_x = torch.from_numpy(panel["counts"][-used_cfg.lookback:]).float().unsqueeze(0).to(device)
    pred_mean = model.predict_mean(last_x, used_cfg.horizon).cpu().numpy()[0]
    # Aggregated per-district risk across horizon + categories.
    risk = pred_mean.sum(axis=(0, 2))
    viz.plot_district_risk_heatmap(risk, panel["districts"],
                                   f"district_risk_{tag}.png")
    try:
        viz.plot_choropleth(risk, panel["districts"], f"choropleth_{tag}.png")
    except Exception:
        pass
    # One prediction-vs-actual example for the top-risk district.
    top = int(np.argmax(risk))
    true_tail = panel["counts"][-used_cfg.horizon:, top, :].sum(axis=1)
    pred_tail = pred_mean[:, top, :].sum(axis=1)
    viz.plot_pred_vs_actual(pred_tail, true_tail,
                            panel["years"][-used_cfg.horizon:],
                            panel["districts"][top], "all_categories",
                            f"pred_vs_actual_{tag}.png")

    # --- Model comparison + ablation figures --------------------------------
    viz.plot_model_comparison(metrics_df, "MAE", f"comparison_MAE_{tag}.png")
    viz.plot_model_comparison(metrics_df, "CSI", f"comparison_CSI_{tag}.png")

    # Save model + summary.
    torch.save(model.state_dict(), RESULTS_DIR / f"hasi_net_{tag}.pt")
    summary = {
        "dataset": dataset, "tag": tag, "device": str(device),
        "config": used_cfg.to_dict(),
        "metrics": {k: v for k, v in metrics_df.to_dict(orient="index").items()},
        "best_val_mae": res["best_val_mae"],
        "panel_meta": panel["meta"],
        "n_nodes": int(panel["counts"].shape[1]),
        "n_years": int(panel["counts"].shape[0]),
        "channel_weights": weights.tolist() if weights is not None else None,
    }
    (RESULTS_DIR / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_ablation(dataset: str, cfg: Config, tag: str = "ablation",
                 epochs: int = 40, verbose: bool = False) -> "pd.DataFrame":
    """Component ablation. Trains short variants and returns a MAE/CSI table.

    Variants:
      Full              -- HASI-Net as configured
      no-adaptive-graph -- fixed geo+socio adjacency only
      no-socio          -- socioeconomic channel replaced by identity
      no-spatial        -- both adjacency channels replaced by identity
      ZINB-loss         -- swap log1p_mse for zero-inflated NB
    """
    import torch as _t
    from . import viz

    set_seed(cfg.seed)
    device = select_device(cfg.device)
    panel_obj = get_dataset(dataset, cfg)
    base_panel = assemble_panel(panel_obj, cfg)
    tr, va, te = _loaders(base_panel, cfg, "all")
    n = base_panel["a_geo"].shape[0]

    variants = {
        "Full": cfg,
        "no-adaptive-graph": cfg.override(adaptive_graph=False),
        "no-socio": None,            # handled by swapping a_socio -> identity
        "no-spatial": None,          # identity adjacency + no adaptive
        "ZINB-loss": cfg.override(loss_type="zinb"),
    }

    rows = {}
    for name, vcfg in variants.items():
        panel = base_panel
        c = (vcfg or cfg).override(epochs=epochs, pso_enabled=False)
        if name == "no-socio":
            panel = dict(base_panel)
            panel["a_socio"] = np.eye(n, dtype=np.float32)
        elif name == "no-spatial":
            panel = dict(base_panel)
            panel["a_geo"] = np.eye(n, dtype=np.float32)
            panel["a_socio"] = np.eye(n, dtype=np.float32)
            c = c.override(adaptive_graph=False)
        model = build_model(c, panel, device)
        train_one(model, c, tr, va, device, verbose=verbose)
        m = evaluate(model, te, device, c.horizon)
        rows[name] = {"MAE": m.mae, "RMSE": m.rmse, "CSI": m.csi, "R2": m.r2}
        if verbose:
            print(f"  ablation {name}: {rows[name]}")
    df = pd.DataFrame(rows).T
    df.to_csv(RESULTS_DIR / f"ablation_{tag}.csv")
    viz.plot_ablation(df[["MAE"]], f"ablation_{tag}.png")
    return df
"""Multi-seed experiment runner for Paper 1.

Runs a configured model set across multiple random seeds, aggregates mean +/- std
(per-seed table + aggregated table), and checkpoints after every seed so a
Colab session interruption can be resumed by simply re-running -- completed
seeds are skipped unless ``force=True``.

Methodology notes (disclosed in the paper):
  * PSO hyperparameter search is run ONCE (on the first seed) and the resulting
    config is frozen for all seeds. Multi-seed variance therefore reflects
    training stochasticity, not HP-search variance.
  * All baselines share HASI-Net's persistence-residual head, count loss and
    evaluation path, so differences are attributable to the spatio-temporal
    representation rather than the decoding head.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

METRIC_COLS = ["MAE", "RMSE", "RMSLE", "WAPE", "R2", "CSI"]


def _assemble(panel_obj, cfg: Config) -> Dict:
    A_geo, A_socio = build_graphs(panel_obj.districts, panel_obj.node_feats, cfg)
    return {"counts": panel_obj.counts, "node_feats": panel_obj.node_feats,
            "a_geo": A_geo, "a_socio": A_socio, "years": panel_obj.years,
            "districts": panel_obj.districts, "categories": panel_obj.categories,
            "meta": panel_obj.meta}


def _loaders(panel: Dict, cfg: Config):
    T = panel["counts"].shape[0]
    tr, va, te = make_splits(T, cfg.lookback, cfg.horizon)
    def dl(idx, shuffle=False):
        return DataLoader(WindowDataset(panel["counts"], cfg.lookback,
                                        cfg.horizon, idx),
                          batch_size=cfg.batch_size, shuffle=shuffle)
    return dl(tr, True), dl(va), dl(te), (tr, va, te)


def _seed_path(dataset: str, tag: str, seed: int) -> Path:
    return RESULTS_DIR / f"multiseed_{dataset}_{tag}_seed{seed}.json"


def _run_one_seed(dataset: str, cfg: Config, seed: int, models: List[str],
                  device: torch.device, panel: Dict, pso_cfg: Optional[Config],
                  verbose: bool = True) -> Dict:
    """Train + evaluate every model for a single seed. Returns a metrics dict."""
    set_seed(seed)
    tr, va, te, _ = _loaders(panel, cfg)
    rows: Dict[str, Dict[str, float]] = {}
    hasi_weights = None
    for name in models:
        if verbose:
            print(f"  [seed {seed}] {name} ...", flush=True)
        if name == "HASI-Net":
            model = build_model(pso_cfg or cfg, panel, device)
            train_one(model, pso_cfg or cfg, tr, va, device, verbose=False)
            try:
                hasi_weights = model.spatial.adj.channel_weights().cpu().numpy()
            except Exception:
                hasi_weights = None
        else:
            bc = (pso_cfg or cfg).override(pso_enabled=False)
            model = build_baseline(name, bc, panel, device)
            if name != "HA":
                train_one(model, bc, tr, va, device, verbose=False)
        m = evaluate(model, te, device, (pso_cfg or cfg).horizon)
        rows[name] = m.as_dict()
        if verbose:
            print(f"    {name}: MAE={m.mae:.4f} RMSE={m.rmse:.4f} "
                  f"R2={m.r2:.4f} CSI={m.csi:.4f}", flush=True)
    return {"seed": seed, "metrics": rows,
            "channel_weights": hasi_weights.tolist() if hasi_weights is not None else None}


def run_multiseed(dataset: str, cfg: Config, seeds: List[int],
                  models: Optional[List[str]] = None, tag: str = "p1",
                  pso: bool = True, force: bool = False,
                  verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run ``models`` across ``seeds`` and return (mean_std_df, per_seed_df).

    PSO is run once on the first seed (if ``pso``) and frozen for all seeds.
    Per-seed results are checkpointed to results/multiseed_{ds}_{tag}_seed{s}.json
    and reused on rerun unless ``force``.
    """
    device = select_device(cfg.device)
    if verbose:
        print(f"Device: {device} | dataset={dataset} tag={tag} seeds={seeds}")
    models = models or (BASELINES + ["HASI-Net"])

    panel_obj = get_dataset(dataset, cfg)
    panel = _assemble(panel_obj, cfg)
    if verbose:
        print(f"Panel: T={panel['counts'].shape[0]} "
              f"N={panel['counts'].shape[1]} C={panel['counts'].shape[2]}")

    # PSO once, on the first seed, frozen for all seeds.
    pso_cfg = None
    if pso:
        from .pso import run_pso, short_fitness
        pso_cache = RESULTS_DIR / f"multiseed_{dataset}_{tag}_pso.json"
        if pso_cache.exists() and not force:
            pso_cfg = cfg.override(**json.loads(pso_cache.read_text())["config"])
            if verbose:
                print("PSO: loaded cached config", pso_cfg.to_dict())
        else:
            set_seed(seeds[0])
            fit = short_fitness(panel, cfg, device, epochs=15)
            pso_cfg, pso_hist = run_pso(fit, cfg, verbose=verbose)
            viz.plot_pso_convergence(pso_hist, f"pso_convergence_{tag}.png")
            pso_cache.write_text(json.dumps(
                {"config": pso_cfg.to_dict(),
                 "history": pso_hist}, indent=2))
            if verbose:
                print("PSO selected:", pso_cfg.to_dict())

    per_seed: List[Dict] = []
    for seed in seeds:
        path = _seed_path(dataset, tag, seed)
        if path.exists() and not force:
            if verbose:
                print(f"[seed {seed}] cached -> {path.name}")
            per_seed.append(json.loads(path.read_text()))
            continue
        res = _run_one_seed(dataset, cfg, seed, models, device, panel,
                            pso_cfg, verbose=verbose)
        path.write_text(json.dumps(res, indent=2))
        per_seed.append(res)

    # Aggregate mean +/- std.
    rows_meanstd, rows_perseed = {}, []
    for name in models:
        mat = np.array([[r["metrics"][name][c] for c in METRIC_COLS]
                        for r in per_seed if name in r["metrics"]])
        if mat.size == 0:
            continue
        rows_meanstd[name] = {f"{c}_mean": float(mat[:, i].mean())
                              for i, c in enumerate(METRIC_COLS)}
        rows_meanstd[name].update({f"{c}_std": float(mat[:, i].std())
                                   for i, c in enumerate(METRIC_COLS)})
        rows_meanstd[name]["n_seeds"] = int(mat.shape[0])
        for r in per_seed:
            if name in r["metrics"]:
                rows_perseed.append({"model": name, "seed": r["seed"],
                                     **r["metrics"][name]})
    meanstd_df = pd.DataFrame(rows_meanstd).T
    perseed_df = pd.DataFrame(rows_perseed)
    meanstd_df.to_csv(RESULTS_DIR / f"metrics_{tag}_meanstd.csv")
    perseed_df.to_csv(RESULTS_DIR / f"metrics_{tag}_perseed.csv", index=False)

    # Representative-seed figures (first seed's HASI-Net run).
    rep = next((r for r in per_seed if r.get("channel_weights")), None)
    if rep is not None and rep["channel_weights"] is not None:
        viz.plot_channel_weights(np.array(rep["channel_weights"]),
                                 f"channel_weights_{tag}.png")
    summary = {"dataset": dataset, "tag": tag, "seeds": seeds,
               "models": models, "panel_meta": panel["meta"],
               "n_nodes": int(panel["counts"].shape[1]),
               "n_t": int(panel["counts"].shape[0]),
               "pso_config": (pso_cfg or cfg).to_dict(),
               "mean_std": meanstd_df.round(4).to_dict(orient="index")}
    (RESULTS_DIR / f"summary_{tag}_multiseed.json").write_text(
        json.dumps(summary, indent=2))
    if verbose:
        print("\n=== mean +/- std ===")
        with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
            print(meanstd_df.round(4))
    return meanstd_df, perseed_df


# --------------------------------------------------------------------------- #
# Ablation (multi-seed)                                                        #
# --------------------------------------------------------------------------- #
ABLATION_VARIANTS = ["Full", "no-adaptive-graph", "no-socio", "no-spatial",
                     "ZINB-loss"]


def _ablation_variant(name: str, cfg: Config, panel: Dict) -> Tuple[Config, Dict]:
    c = cfg.override(pso_enabled=False)
    p = panel
    n = panel["a_geo"].shape[0]
    if name == "Full":
        return c, p
    if name == "no-adaptive-graph":
        return c.override(adaptive_graph=False), p
    if name == "no-socio":
        p = dict(panel); p["a_socio"] = np.eye(n, dtype=np.float32)
        return c, p
    if name == "no-spatial":
        p = dict(panel); p["a_geo"] = np.eye(n, dtype=np.float32)
        p["a_socio"] = np.eye(n, dtype=np.float32)
        return c.override(adaptive_graph=False), p
    if name == "ZINB-loss":
        return c.override(loss_type="zinb"), p
    raise ValueError(name)


def run_ablation_multiseed(dataset: str, cfg: Config, seeds: List[int],
                           tag: str = "p1", epochs: int = 40,
                           force: bool = False,
                           verbose: bool = True) -> pd.DataFrame:
    """Component ablation across seeds -> mean +/- std table."""
    device = select_device(cfg.device)
    panel_obj = get_dataset(dataset, cfg)
    panel = _assemble(panel_obj, cfg)

    accum: Dict[str, List[Dict[str, float]]] = {v: [] for v in ABLATION_VARIANTS}
    for seed in seeds:
        path = RESULTS_DIR / f"ablation_{dataset}_{tag}_seed{seed}.json"
        if path.exists() and not force:
            row = json.loads(path.read_text())
        else:
            set_seed(seed)
            tr, va, te, _ = _loaders(panel, cfg)
            row = {}
            for vname in ABLATION_VARIANTS:
                vc, vp = _ablation_variant(vname, cfg, panel)
                vc = vc.override(epochs=epochs)
                model = build_model(vc, vp, device)
                train_one(model, vc, tr, va, device, verbose=False)
                m = evaluate(model, te, device, vc.horizon)
                row[vname] = m.as_dict()
                if verbose:
                    print(f"  [abl seed {seed}] {vname}: MAE={m.mae:.4f}", flush=True)
            path.write_text(json.dumps(row, indent=2))
        for vname, met in row.items():
            accum[vname].append(met)

    rows = {}
    for vname, ms in accum.items():
        mat = np.array([[m[c] for c in METRIC_COLS] for m in ms])
        rows[vname] = {f"{c}_mean": float(mat[:, i].mean())
                       for i, c in enumerate(METRIC_COLS)}
        rows[vname].update({f"{c}_std": float(mat[:, i].std())
                            for i, c in enumerate(METRIC_COLS)})
    df = pd.DataFrame(rows).T
    df.to_csv(RESULTS_DIR / f"ablation_{tag}_meanstd.csv")
    if verbose:
        print("\n=== ablation mean +/- std ===")
        print(df.round(4))
    return df
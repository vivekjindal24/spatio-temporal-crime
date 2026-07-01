"""Paper 2 robustness experiments.

Three experiment families (all on MP unless noted), each multi-seed with
per-seed checkpoints:

* ``run_calibrated_vs_point`` -- validates the sparsity-aware calibrated head:
  does it match the point head's MAE while adding calibrated uncertainty
  (CRPS / coverage / sharpness)? Reported honestly even if calibrated MAE is
  slightly worse.
* ``run_missing_data_robustness`` -- masks a fraction of training cells and
  measures graceful degradation; the calibrated head's predictive intervals
  should widen rather than collapse.
* transfer-vs-scratch lives in ``transfer.py``.

Statistical tests (Diebold-Mariano, bootstrap CI, Friedman+Nemenyi) are applied
to the saved per-seed tables in the driver, not here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config, RESULTS_DIR
from .data import get_dataset
from .graph import build_graphs
from .train import (build_model, make_splits, train_one, WindowDataset,
                    select_device, set_seed, evaluate)
from .transfer import _assemble, _loaders, _train_zero_fraction, \
    evaluate_calibrated, METRIC_COLS, CAL_COLS, _json_default
from .baselines import build_baseline
from .stats import diebold_mariano, per_window_errors


def _masked_counts(counts: np.ndarray, frac: float, rng) -> np.ndarray:
    """Set a random ``frac`` of cells to NaN-equivalent (0) and return a copy.
    A mask of the dropped cells is returned alongside for disclosure."""
    out = counts.copy()
    mask = rng.random(out.shape) < frac
    out[mask] = 0.0
    return out, mask


def run_calibrated_vs_point(dataset: str, cfg_cal: Config, seeds: List[int],
                            tag: str = "p2", force: bool = False,
                            verbose: bool = True) -> pd.DataFrame:
    """Compare calibrated_head=True vs False on the same data/seeds.

    ``cfg_cal`` is the calibrated config; the point variant is derived by
    flipping calibrated_head=False (and loss_type to log1p_mse for a fair
    deterministic-objective comparison).
    """
    device = select_device(cfg_cal.device)
    cfg_pt = cfg_cal.override(calibrated_head=False, loss_type="log1p_mse")
    panel = _assemble(get_dataset(dataset, cfg_cal), cfg_cal)
    zf = _train_zero_fraction(panel, cfg_cal)
    rows = []
    for seed in seeds:
        for cond, c in (("calibrated", cfg_cal), ("point", cfg_pt)):
            path = RESULTS_DIR / f"calvspt_{dataset}_{tag}_seed{seed}_{cond}.json"
            if path.exists() and not force:
                rows.append(json.loads(path.read_text()))
                if verbose:
                    print(f"[{dataset} {seed}/{cond}] cached")
                continue
            set_seed(seed)
            tr, va, te = _loaders(panel, c)
            model = build_model(c, panel, device)
            if c.calibrated_head:
                model.set_gate_from_sparsity(zf)
            train_one(model, c, tr, va, device, verbose=False)
            point, cal, per_crime = evaluate_calibrated(model, te, device, c.horizon)
            row = {"dataset": dataset, "seed": seed, "condition": cond,
                   **point.as_dict(), **cal, "per_crime_MAE": per_crime}
            path.write_text(json.dumps(row, indent=2, default=_json_default))
            rows.append(row)
            if verbose:
                print(f"[{dataset} {seed}/{cond}] MAE={point.mae:.4f} "
                      f"CRPS={cal['CRPS']} cov80={cal['coverage80']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"calvspt_{dataset}_{tag}_perseed.csv", index=False)
    agg = df.groupby("condition")[METRIC_COLS + CAL_COLS].agg(["mean", "std"])
    agg.to_csv(RESULTS_DIR / f"calvspt_{dataset}_{tag}_meanstd.csv")
    if verbose:
        print("\n=== calibrated vs point (mean) ===")
        print(agg.round(4))
    return df


def run_missing_data_robustness(dataset: str, cfg: Config, seeds: List[int],
                                fractions: List[float], tag: str = "p2",
                                force: bool = False,
                                verbose: bool = True) -> pd.DataFrame:
    """Mask ``frac`` of training cells (set to 0) and re-evaluate the
    calibrated model. Tests graceful degradation + interval widening."""
    device = select_device(cfg.device)
    base_panel = _assemble(get_dataset(dataset, cfg), cfg)
    zf = _train_zero_fraction(base_panel, cfg)
    rows = []
    for frac in fractions:
        for seed in seeds:
            path = (RESULTS_DIR /
                    f"robust_{dataset}_{tag}_f{int(frac*100)}_seed{seed}.json")
            if path.exists() and not force:
                rows.append(json.loads(path.read_text()))
                if verbose:
                    print(f"[{dataset} f={frac} seed={seed}] cached")
                continue
            rng = np.random.default_rng(seed * 1000 + int(frac * 100))
            counts, mask = _masked_counts(base_panel["counts"], frac, rng)
            panel = dict(base_panel)
            panel["counts"] = counts.astype(np.float32)
            set_seed(seed)
            tr, va, te = _loaders(panel, cfg)
            model = build_model(cfg, panel, device)
            model.set_gate_from_sparsity(zf)
            train_one(model, cfg, tr, va, device, verbose=False)
            point, cal, per_crime = evaluate_calibrated(model, te, device, cfg.horizon)
            row = {"dataset": dataset, "seed": seed, "mask_frac": frac,
                   "masked_cells": int(mask.sum()), "mask_total": int(mask.size),
                   **point.as_dict(), **cal}
            path.write_text(json.dumps(row, indent=2, default=_json_default))
            rows.append(row)
            if verbose:
                print(f"[{dataset} f={frac} seed={seed}] MAE={point.mae:.4f} "
                      f"cov80={cal['coverage80']} sharp={cal['sharpness80']}",
                      flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"robust_{dataset}_{tag}_perseed.csv", index=False)
    agg = df.groupby("mask_frac")[METRIC_COLS + CAL_COLS].agg(["mean", "std"])
    agg.to_csv(RESULTS_DIR / f"robust_{dataset}_{tag}_meanstd.csv")
    if verbose:
        print("\n=== missing-data robustness (mean) ===")
        print(agg.round(4))
    return df


def _per_window_losses(model, loader, device, horizon) -> np.ndarray:
    """Per-forecast-window absolute errors (length = #test windows)."""
    import numpy as np
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            mu = model.predict_mean(x.to(device), horizon).cpu().numpy()
            preds.append(mu); trues.append(y.numpy())
    p = np.concatenate(preds, 0)
    t = np.concatenate(trues, 0)
    return per_window_errors(p, t, kind="abs")


def run_dm_comparison(dataset: str, cfg: Config, seeds: List[int],
                      pairs: List[tuple], tag: str = "p2", h_step: int = 1,
                      force: bool = False, verbose: bool = True
                      ) -> pd.DataFrame:
    """Diebold-Mariano pairwise tests. ``pairs`` is a list of (modelA, modelB)
    name tuples; DM positive => B is better. Trains both models per seed,
    captures per-window absolute errors, runs DM (h-step corrected), and
    aggregates DM statistics + p-values across seeds."""
    device = select_device(cfg.device)
    panel = _assemble(get_dataset(dataset, cfg), cfg)
    rows = []
    for seed in seeds:
        tr, va, te = _loaders(panel, cfg)
        cache = {}
        # Train each unique model in the pairs once per seed.
        for name in sorted({n for pr in pairs for n in pr}):
            ckpt = RESULTS_DIR / f"dm_{dataset}_{tag}_seed{seed}_{name}.json"
            if ckpt.exists() and not force:
                cache[name] = np.array(json.loads(ckpt.read_text())["err"])
                continue
            set_seed(seed)
            model = (build_model(cfg, panel, device) if name == "HASI-Net"
                     else build_baseline(name, cfg, panel, device))
            if name != "HA":
                train_one(model, cfg, tr, va, device, verbose=False)
            err = _per_window_losses(model, te, device, cfg.horizon)
            cache[name] = err
            ckpt.write_text(json.dumps({"err": err.tolist()}))
        for a, b in pairs:
            dm, p = diebold_mariano(cache[a], cache[b], h=h_step)
            rows.append({"dataset": dataset, "seed": seed, "modelA": a,
                         "modelB": b, "DM": dm, "p": p,
                         "meanA": float(cache[a].mean()),
                         "meanB": float(cache[b].mean())})
            if verbose:
                print(f"  DM[{a} vs {b}] seed{seed}: DM={dm:+.3f} p={p:.4g} "
                      f"(B better if DM>0)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"dm_{dataset}_{tag}_perseed.csv", index=False)
    agg = df.groupby(["modelA", "modelB"])[["DM", "p"]].agg(["mean", "std"])
    agg.to_csv(RESULTS_DIR / f"dm_{dataset}_{tag}_meanstd.csv")
    if verbose:
        print("\n=== DM (mean across seeds) ===")
        print(agg.round(4))
    return df
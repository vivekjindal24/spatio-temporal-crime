"""Paper-1 hotspot reframe: event-verification + top-k hotspot metrics.

The point-MAE benchmark (``run_multiseed``) showed the historical-average (HA)
persistence baseline is brutally strong on smooth monthly crime panels -- the
deep models, HASI-Net included, were mid-pack on point MAE. That metric rewards
getting the *level* right, which persistence does by carrying the lookback mean.

Policing does not act on the level; it acts on *where and when crime spikes*.
This module evaluates the same 8-model P1 benchmark on the operationally
relevant axis -- high-crime *events* and *hotspots* -- where a flat carry is
structurally disadvantaged: it lags rising spikes (misses) and false-alarms
falling ones. The metrics are standard forecast-verification quantities:

* **Event verification (per node x crime)** -- an *event* is a test cell whose
  count meets/exceeds that (node, crime)'s 90th-percentile threshold, computed
  from the TRAINING period only (leak-free). Predicted event = pred >= same
  threshold. From the 2x2 contingency: POD (hit rate), FAR (false-alarm ratio),
  CSI, and frequency bias. A per-node threshold makes a *transient spike*
  (relative to that location's own history) the event -- precisely what a
  lookback mean cannot anticipate.
* **Top-k hotspot Hit@k** -- at each forecast step, rank nodes by predicted
  crime (summed over the 4 women-related crimes) and by actual crime; Hit@k is
  the overlap fraction of the top-k predicted and top-k actual node sets. This
  measures *ranking* quality, which point MAE does not.

A global-pooled 90th-pct threshold (event = high in absolute terms, i.e. one of
the chronically high-crime nodes) is reported alongside for contrast: that
definition favours persistence, because the chronically high nodes are exactly
the ones the carry reproduces. Reporting both -- not cherry-picking -- is the
honest framing: the per-node (transient-spike) threshold is the operationally
meaningful one for anticipating *new* high-crime events.

Everything runs on re-runnable Colab CUDA training; per-seed predictions are
persisted to .npz so the offline hotspot metrics are reproducible without
retraining.
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
from .train import (build_model, make_splits, train_one, WindowDataset,
                    select_device, set_seed)
from .baselines import build_baseline, BASELINES
from .multiseed import _assemble, _loaders, METRIC_COLS

# Event threshold quantile (per node x crime, from training counts only).
EVENT_Q = 0.9
# Top-k node counts for hotspot Hit@k (Chicago has 77 community areas).
TOP_K = (5, 10)


# --------------------------------------------------------------------------- #
# Prediction collection (preserves [nWin, horizon, N, C] structure)            #
# --------------------------------------------------------------------------- #
def collect_preds(model, loader: DataLoader, device: torch.device,
                  horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pred, true) as [nWin, horizon, N, C] count-space arrays.

    Mirrors ``evaluate`` but keeps the spatio-temporal structure needed for
    per-node / per-crime hotspot metrics instead of flattening to 1D.
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            mu = model.predict_mean(x.to(device), horizon).cpu().numpy()
            preds.append(mu)
            trues.append(y.numpy())
    pred = np.concatenate(preds, axis=0)           # [nWin, H, N, C]
    true = np.concatenate(trues, axis=0)
    pred = np.clip(pred, 0.0, None)
    return pred, true


# --------------------------------------------------------------------------- #
# Leak-free event thresholds                                                  #
# --------------------------------------------------------------------------- #
def train_time_end(train_idx: List[int], lookback: int, horizon: int) -> int:
    """Last time index any training window *target* touches -- counts up to and
    including this month are seen during training, so they are the leak-free
    source for per-node event thresholds."""
    return max(train_idx) + lookback + horizon - 1


def per_node_thresholds(counts: np.ndarray, train_end: int,
                        q: float = EVENT_Q) -> np.ndarray:
    """Per-(node, crime) q-quantile of training-period counts -> [N, C].

    Training period = months [0, train_end] inclusive. A test cell is an event
    iff its count >= this location's own historical q-quantile, so an event is a
    spike *relative to that node*, not an absolute high-crime node.
    """
    train_counts = counts[:train_end + 1]                 # [t<=train_end, N, C]
    return np.quantile(train_counts, q, axis=0)           # [N, C]


def global_threshold(counts: np.ndarray, train_end: int,
                     q: float = EVENT_Q) -> float:
    """Pooled q-quantile of all training cells (absolute high-crime event)."""
    return float(np.quantile(counts[:train_end + 1], q))


# --------------------------------------------------------------------------- #
# Event verification (2x2 contingency -> POD / FAR / CSI / Bias)               #
# --------------------------------------------------------------------------- #
def event_contingency(pred: np.ndarray, true: np.ndarray,
                      thresh: np.ndarray) -> Dict[str, float]:
    """2x2 contingency for ``event = count >= thresh``.

    ``pred``/``true`` are [..., N, C]; ``thresh`` is [N, C] (broadcast) or a
    scalar. Returns POD, FAR, CSI, bias, and the event base rate.
    """
    p_event = pred >= thresh
    t_event = true >= thresh
    hits = float((p_event & t_event).sum())
    fa = float((p_event & ~t_event).sum())          # false alarms
    misses = float((~p_event & t_event).sum())
    corr_neg = float((~p_event & ~t_event).sum())
    n_event = hits + misses
    n_pred_event = hits + fa
    pod = hits / n_event if n_event > 0 else float("nan")
    far = fa / n_pred_event if n_pred_event > 0 else float("nan")
    csi = hits / (hits + misses + fa) if (hits + misses + fa) > 0 else 0.0
    bias = (n_pred_event / n_event) if n_event > 0 else float("nan")
    rate = float(t_event.mean())                    # observed event frequency
    return {"POD": pod, "FAR": far, "CSI": csi, "bias": bias,
            "event_rate": rate, "n_events": n_event,
            "hits": hits, "false_alarms": fa, "misses": misses,
            "correct_negatives": corr_neg}


# --------------------------------------------------------------------------- #
# Top-k hotspot Hit@k                                                          #
# --------------------------------------------------------------------------- #
def hit_at_k(pred: np.ndarray, true: np.ndarray, k: int) -> float:
    """Mean Hit@k over forecast windows x horizon steps.

    At each (window, horizon step) we collapse crime -> a per-node risk vector
    (sum over the 4 women-related crimes), rank nodes by predicted and by actual
    risk, and take |top-k predicted ∩ top-k actual| / k. k is capped at N-1.
    """
    # pred/true: [nWin, H, N, C] -> per-node risk [nWin, H, N]
    pr = pred.sum(axis=-1)
    tr = true.sum(axis=-1)
    n = pr.shape[-1]
    k = min(k, n - 1)
    nW, H, _ = pr.shape
    hits = 0.0
    total = 0
    for w in range(nW):
        for h in range(H):
            p_top = set(np.argsort(-pr[w, h])[:k].tolist())
            t_top = set(np.argsort(-tr[w, h])[:k].tolist())
            hits += len(p_top & t_top) / k
            total += 1
    return hits / total if total > 0 else 0.0


# --------------------------------------------------------------------------- #
# Full hotspot metric bundle                                                  #
# --------------------------------------------------------------------------- #
def hotspot_metrics(pred: np.ndarray, true: np.ndarray, counts: np.ndarray,
                    train_idx: List[int], lookback: int, horizon: int,
                    ks: Tuple[int, ...] = TOP_K,
                    q: float = EVENT_Q) -> Dict[str, float]:
    """Compute the full hotspot metric set for one (seed, model).

    * per-node event verification (transient-spike threshold) -- PRIMARY
    * global-pooled event verification (absolute high-crime threshold) -- contrast
    * top-k Hit@k
    """
    tend = train_time_end(train_idx, lookback, horizon)
    node_q = per_node_thresholds(counts, tend, q)          # [N, C]
    glob_q = global_threshold(counts, tend, q)             # scalar
    # Per-node contingency: broadcast node_q [N,C] over [nWin,H,N,C].
    per_node = event_contingency(pred, true, node_q)
    glob = event_contingency(pred, true, glob_q)
    out: Dict[str, float] = {}
    for k_, v in per_node.items():
        out[f"node_{k_}"] = float(v) if v == v else float("nan")
    for k_, v in glob.items():
        out[f"global_{k_}"] = float(v) if v == v else float("nan")
    for k in ks:
        out[f"Hit@{k}"] = float(hit_at_k(pred, true, k))
    return out


# --------------------------------------------------------------------------- #
# Per-seed runner                                                              #
# --------------------------------------------------------------------------- #
def _seed_path(dataset: str, tag: str, seed: int) -> Path:
    return RESULTS_DIR / f"hotspot_{dataset}_{tag}_seed{seed}.json"


def _run_one_seed(dataset: str, cfg: Config, seed: int, models: List[str],
                  device: torch.device, panel: Dict,
                  pso_cfg: Optional[Config], counts: np.ndarray,
                  train_idx: List[int], tag: str, force: bool,
                  verbose: bool = True) -> Dict:
    """Train + collect structured preds + compute hotspot metrics, one seed.

    Models are built exactly as in ``multiseed._run_one_seed`` (HASI-Net uses
    the PSO config; baselines share its persistence-residual head), so the
    hotspot evaluation is on the same trained models as the point-MAE
    benchmark. Predictions are persisted to .npz for reproducibility.
    """
    set_seed(seed)
    tr, va, te, _ = _loaders(panel, cfg)
    rows: Dict[str, Dict[str, float]] = {}
    for name in models:
        pred_path = RESULTS_DIR / f"preds_{dataset}_{tag}_seed{seed}_{name}.npz"
        if pred_path.exists() and not force:
            npz = np.load(pred_path)
            pred, true = npz["pred"], npz["true"]
            if verbose:
                print(f"  [seed {seed}] {name}: preds cached", flush=True)
        else:
            if verbose:
                print(f"  [seed {seed}] {name} ...", flush=True)
            if name == "HASI-Net":
                model = build_model(pso_cfg or cfg, panel, device)
                train_one(model, pso_cfg or cfg, tr, va, device, verbose=False)
            else:
                bc = (pso_cfg or cfg).override(pso_enabled=False)
                model = build_baseline(name, bc, panel, device)
                if name != "HA":
                    train_one(model, bc, tr, va, device, verbose=False)
            pred, true = collect_preds(model, te, device, cfg.horizon)
            np.savez_compressed(pred_path, pred=pred, true=true)
        # Consistency: standard point Metrics recomputed from the same
        # collected preds (so cached-pred and freshly-trained paths agree, and
        # the hotspot numbers can be read alongside the point-MAE context).
        m = _metrics_from(pred, true)
        hs = hotspot_metrics(pred, true, counts, train_idx, cfg.lookback,
                              cfg.horizon, ks=TOP_K, q=EVENT_Q)
        rows[name] = {**m, **hs}
        if verbose:
            print(f"    {name}: node_CSI={hs['node_CSI']:.3f} "
                  f"node_POD={hs['node_POD']:.3f} node_FAR={hs['node_FAR']:.3f} "
                  f"Hit@5={hs['Hit@5']:.3f}", flush=True)
    return {"seed": seed, "metrics": rows}


def _metrics_from(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """Recompute the standard point Metrics from collected preds (consistency
    check vs the cached multiseed aggregate)."""
    p = pred.reshape(-1)
    t = true.reshape(-1)
    eps = 1e-6
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    rmsle = float(np.sqrt(np.mean((np.log1p(p) - np.log1p(t)) ** 2)))
    wape = float(np.sum(np.abs(p - t)) / (float(np.sum(np.abs(t))) + eps) * 100.0)
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2)) + eps
    r2 = 1.0 - ss_res / ss_tot
    thr = float(np.quantile(t, 0.9))
    tp = float(((p >= thr) & (t >= thr)).sum())
    fp = float(((p >= thr) & ~(t >= thr)).sum())
    fn = float((~(p >= thr) & (t >= thr)).sum())
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "RMSLE": rmsle, "WAPE": wape,
            "R2": r2, "CSI": csi}


def run_hotspot(dataset: str, cfg: Config, seeds: List[int],
                tag: str = "p1hot", pso_cfg: Optional[Config] = None,
                models: Optional[List[str]] = None, force: bool = False,
                verbose: bool = True) -> pd.DataFrame:
    """Run the 8-model hotspot reframe across ``seeds``.

    ``pso_cfg``: frozen PSO config to reuse (keeps parity with the published
    point-MAE benchmark). If None, cfg is used as-is (PSO per ``cfg.pso_enabled``).

    Writes per-seed JSON, a per-seed CSV, a mean+/-std CSV, and a summary JSON.
    Returns the mean+/-std DataFrame.
    """
    device = select_device(cfg.device)
    if verbose:
        print(f"Device: {device} | dataset={dataset} tag={tag} seeds={seeds}",
              flush=True)
    models = models or (BASELINES + ["HASI-Net"])
    panel_obj = get_dataset(dataset, cfg)
    panel = _assemble(panel_obj, cfg)
    counts = panel["counts"]                       # [T, N, C]
    T = counts.shape[0]
    tr_idx, _, _ = make_splits(T, cfg.lookback, cfg.horizon)
    if verbose:
        print(f"Panel: T={T} N={counts.shape[1]} C={counts.shape[2]} "
              f"train_windows={len(tr_idx)}", flush=True)

    HOT_COLS = ["node_POD", "node_FAR", "node_CSI", "node_bias",
                "node_event_rate", "global_POD", "global_FAR", "global_CSI",
                "Hit@5", "Hit@10"]
    per_seed: List[Dict] = []
    for seed in seeds:
        path = _seed_path(dataset, tag, seed)
        if path.exists() and not force:
            if verbose:
                print(f"[seed {seed}] cached -> {path.name}", flush=True)
            per_seed.append(json.loads(path.read_text()))
            continue
        res = _run_one_seed(dataset, cfg, seed, models, device, panel,
                            pso_cfg, counts, tr_idx, tag, force, verbose=verbose)
        path.write_text(json.dumps(res, indent=2))
        per_seed.append(res)

    # Aggregate mean +/- std over the hotspot metrics (+ point-MAE for context).
    cols = METRIC_COLS + HOT_COLS
    rows_meanstd, rows_perseed = {}, []
    for name in models:
        mat = []
        for r in per_seed:
            if name in r["metrics"]:
                mat.append([r["metrics"][name].get(c, float("nan")) for c in cols])
        mat = np.array(mat, dtype=float)
        if mat.size == 0:
            continue
        rows_meanstd[name] = {}
        for i, c in enumerate(cols):
            col = mat[:, i]
            col = col[~np.isnan(col)]
            rows_meanstd[name][f"{c}_mean"] = float(col.mean()) if col.size else float("nan")
            rows_meanstd[name][f"{c}_std"] = float(col.std()) if col.size else float("nan")
        rows_meanstd[name]["n_seeds"] = int(mat.shape[0])
        for r in per_seed:
            if name in r["metrics"]:
                rows_perseed.append({"model": name, "seed": r["seed"],
                                     **r["metrics"][name]})
    meanstd_df = pd.DataFrame(rows_meanstd).T
    perseed_df = pd.DataFrame(rows_perseed)
    meanstd_df.to_csv(RESULTS_DIR / f"hotspot_{tag}_{dataset}_meanstd.csv")
    perseed_df.to_csv(RESULTS_DIR / f"hotspot_{tag}_{dataset}_perseed.csv",
                      index=False)

    summary = {"dataset": dataset, "tag": tag, "seeds": seeds,
               "models": models, "event_q": EVENT_Q, "top_k": list(TOP_K),
               "panel_meta": panel["meta"],
               "n_nodes": int(counts.shape[1]), "n_t": int(T),
               "lookback": cfg.lookback, "horizon": cfg.horizon,
               "pso_config": (pso_cfg or cfg).to_dict(),
               "mean_std": meanstd_df.round(4).to_dict(orient="index")}
    (RESULTS_DIR / f"summary_{tag}_{dataset}.json").write_text(
        json.dumps(summary, indent=2))
    if verbose:
        print("\n=== hotspot mean +/- std ===")
        with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
            print(meanstd_df[cols].round(4))
    return meanstd_df
"""Cross-region transfer for Paper 2.

Two-region design:
  * Source: Chicago (monthly, 77 community areas, lookback 12 / horizon 3) --
    data-rich, fine-grained, exact geography.
  * Target: Madhya Pradesh (yearly, 51 districts, lookback 4 / horizon 2) --
    coarse, short, the policy-relevant region.

The two panels share the four-crime output dimensionality (config.UNIFIED_CRIMES),
so the *resolution-agnostic* parts of HASI-Net -- the temporal encoder (Informer
+ TCN + series decomposition) and the graph-convolution layer weights -- can be
pretrained on Chicago and reused on MP. The node-specific parts (adaptive
adjacency node embeddings, which depend on N) and the I/O-specific parts (input
projection, which depends on node-feature dimension; the count head, which
depends on horizon) are re-initialised for MP and re-learned. A region-adaptation
regulariser (L2-SP: weight decay toward the pretrained values) constrains the
transferred weights to stay near their source values during MP fine-tuning,
preventing catastrophic forgetting of the Chicago-learned temporal dynamics.

This module also provides the calibrated evaluation path (point metrics +
CRPS / coverage / sharpness) used by the sparsity-aware head.
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
                    select_device, set_seed, Metrics)
from .losses import count_loss
from . import viz

METRIC_COLS = ["MAE", "RMSE", "RMSLE", "WAPE", "R2", "CSI"]
CAL_COLS = ["CRPS", "pinball", "coverage80", "sharpness80"]


def _json_default(o):
    """JSON fallback: serialise numpy scalars/arrays as native Python types.
    Calibration/evaluate paths return numpy float32 even after float() on some
    arrays, so every checkpoint dump routes through this."""
    import numpy as np
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serialisable")


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
    return dl(tr, True), dl(va), dl(te)


def _train_zero_fraction(panel: Dict, cfg: Config) -> List[float]:
    T = panel["counts"].shape[0]
    tr, _, _ = make_splits(T, cfg.lookback, cfg.horizon)
    # Use the training windows' target cells.
    counts = panel["counts"]
    C = counts.shape[2]
    zf = []
    for j in range(C):
        cells = []
        for t in tr:
            y = counts[t + cfg.lookback:t + cfg.lookback + cfg.horizon, :, j]
            cells.append(y.ravel())
        cells = np.concatenate(cells)
        zf.append(float((cells == 0).mean()))
    return zf


# Resolution-agnostic parameter prefixes that are transferred across regions:
# the temporal encoder (Informer + TCN + series decomposition), the graph-conv
# layer weights (hidden->hidden, independent of N), and the 3-channel adjacency
# fusion weights alpha (geo/socio/adaptive balance, independent of N). Node-
# specific (adjacency node embeddings, node features) and I/O-specific (input
# projection, count head) parameters are re-initialised for the target region.
TRANSFER_PREFIXES = ("temporal.", "spatial.layers.", "spatial.adj.alpha")


def transfer_load(target_model: torch.nn.Module,
                  pretrained_state: Dict[str, torch.Tensor]
                  ) -> Tuple[List[str], List[str]]:
    """Copy resolution-agnostic, shape-matched parameters from
    ``pretrained_state`` into ``target_model`` (strict=False).

    Returns (transferred_keys, reinit_keys). Only parameters whose names start
    with a prefix in ``TRANSFER_PREFIXES`` AND whose shapes match are copied;
    everything else (node-specific, I/O-specific) is re-initialised for the
    target region.
    """
    target_sd = target_model.state_dict()
    filtered = {}
    for k, v in pretrained_state.items():
        if not k.startswith(TRANSFER_PREFIXES):
            continue
        if k in target_sd and target_sd[k].shape == v.shape:
            filtered[k] = v
    target_model.load_state_dict(filtered, strict=False)
    transferred = sorted(filtered.keys())
    reinit = sorted(set(target_sd.keys()) - set(filtered.keys()))
    return transferred, reinit


def fine_tune_l2sp(model: torch.nn.Module, cfg: Config,
                   train_loader: DataLoader, val_loader: DataLoader,
                   device: torch.device, targets: Dict[str, torch.Tensor],
                   lam: float, verbose: bool = False) -> Dict:
    """Fine-tune with an L2-SP region-adaptation regulariser:
    loss += lam * sum_transferred (p - p_pretrained)^2.

    ``targets`` maps transferred parameter names to their frozen pretrained
    values (detached, on device). Transferred parameters are optimised at a
    10x lower learning rate than the reinitialised (region-specific)
    parameters -- standard transfer-learning practice that prevents
    catastrophic forgetting of the pretrained temporal/graph weights on the
    very small target panel (MP has only ~7 training windows)."""
    transferred_names = set(targets.keys())
    transferred_p, other_p = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (transferred_p if name in transferred_names else other_p).append(p)
    groups = []
    if transferred_p:
        groups.append({"params": transferred_p, "lr": cfg.lr * 0.1})
    if other_p:
        groups.append({"params": other_p, "lr": cfg.lr})
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                       factor=0.5, patience=5)
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(cfg.epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x, cfg.horizon)
            loss = count_loss(out, y, cfg)
            if lam > 0 and targets:
                reg = 0.0
                for name, p0 in targets.items():
                    p = dict(model.named_parameters()).get(name)
                    if p is not None:
                        reg = reg + ((p - p0) ** 2).sum()
                loss = loss + lam * reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
        train_loss = total / max(n, 1)
        val = _evaluate_point(model, val_loader, device, cfg.horizon)
        sched.step(val.mae)
        improved = val.mae < best_val - 1e-4
        if improved:
            best_val = val.mae
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                if verbose:
                    print(f"  ft early stop at epoch {epoch} "
                          f"(best val MAE {best_val:.4f})", flush=True)
                break
        if verbose:
            print(f"  ft epoch {epoch:3d} | train {train_loss:.4f} | "
                  f"val MAE {val.mae:.4f} | RMSE {val.rmse:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_mae": best_val}


def _evaluate_point(model, loader, device, horizon) -> Metrics:
    from .train import evaluate
    return evaluate(model, loader, device, horizon)


def evaluate_calibrated(model, loader, device, horizon):
    """Point metrics + calibration metrics (CRPS, coverage, sharpness, pinball).

    Returns (Metrics, dict-of-calibration, per-crime MAE list).
    """
    from .train import _csi
    from .calibrated import calibration_metrics
    model.eval()
    preds, trues, qs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mu = model.predict_mean(x, horizon).cpu().numpy()
            preds.append(mu)
            trues.append(y.numpy())
            if model.calibrated:
                q = model.predict_quantiles(x, horizon).cpu().numpy()
                qs.append(q)
    pred = np.clip(np.concatenate(preds, 0).reshape(-1), 0, None)
    true = np.concatenate(trues, 0).reshape(-1)
    eps = 1e-6
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    rmsle = float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2)))
    wape = float(np.sum(np.abs(pred - true)) / (np.sum(np.abs(true)) + eps) * 100)
    r2 = 1.0 - float(np.sum((true - pred) ** 2)) / (np.sum((true - true.mean()) ** 2) + eps)
    csi = _csi(pred, true)
    point = Metrics(mae, rmse, rmsle, wape, r2, csi)
    cal = {"CRPS": None, "pinball": None, "coverage80": None, "sharpness80": None}
    if qs:
        q = np.concatenate(qs, 0)  # [B,H,N,C,|Q|]
        # Align true counts to the quantile tensor: [B,H,N,C] -> [B*H*N, C, 1].
        true_c = np.concatenate(trues, 0).reshape(-1, q.shape[3], 1)  # [.,C,1]
        q_c = q.reshape(-1, q.shape[3], q.shape[4])                   # [.,C,|Q|]
        cal_t = torch.from_numpy(q_c); cal_y = torch.from_numpy(true_c[..., 0])
        cm = calibration_metrics(cal_t, cal_y)
        cal = {"CRPS": cm.crps, "pinball": cm.pinball,
               "coverage80": cm.coverage80, "sharpness80": cm.sharpness80}
    # per-crime MAE
    C = np.concatenate(trues, 0).shape[-1]
    p2 = np.concatenate(preds, 0).reshape(-1, C)
    t2 = np.concatenate(trues, 0).reshape(-1, C)
    per_crime = [float(np.mean(np.abs(p2[:, j] - t2[:, j]))) for j in range(C)]
    return point, cal, per_crime


def run_transfer_vs_scratch(cfg_chi: Config, cfg_mp: Config, seeds: List[int],
                            lam: float = 1e-3, tag: str = "p2",
                            force: bool = False, verbose: bool = True
                            ) -> pd.DataFrame:
    """Pretrain once on Chicago, then compare transfer-vs-scratch on MP across
    seeds. Returns a per-seed/condition table; writes mean+/-std CSV + JSON."""
    device = select_device(cfg_mp.device)
    chi_panel = _assemble(get_dataset("chicago", cfg_chi), cfg_chi)
    mp_panel = _assemble(get_dataset("mp", cfg_mp), cfg_mp)
    zf_mp = _train_zero_fraction(mp_panel, cfg_mp)

    # --- Pretrain once on Chicago (seed 42) ---------------------------------
    pre_path = RESULTS_DIR / f"transfer_{tag}_pretrain.json"
    pretrained_state = None
    if pre_path.exists() and not force and (RESULTS_DIR / f"hasi_net_{tag}_pretrain.pt").exists():
        if verbose:
            print("Pretrain: cached")
        pretrained_state = torch.load(RESULTS_DIR / f"hasi_net_{tag}_pretrain.pt",
                                      map_location=device)
    else:
        if verbose:
            print("Pretraining on Chicago ...")
        set_seed(42)
        tr, va, _ = _loaders(chi_panel, cfg_chi)
        pm = build_model(cfg_chi, chi_panel, device)
        pm.set_gate_from_sparsity(_train_zero_fraction(chi_panel, cfg_chi))
        train_one(pm, cfg_chi, tr, va, device, verbose=verbose)
        pretrained_state = {k: v.detach() for k, v in pm.state_dict().items()}
        torch.save(pm.state_dict(), RESULTS_DIR / f"hasi_net_{tag}_pretrain.pt")
        pre_path.write_text(json.dumps({"seed": 42, "done": True}))

    # --- Per seed: transfer vs scratch on MP --------------------------------
    rows = []
    for seed in seeds:
        for cond in ("transfer", "scratch"):
            path = RESULTS_DIR / f"transfer_{tag}_seed{seed}_{cond}.json"
            if path.exists() and not force:
                rows.append(json.loads(path.read_text()))
                if verbose:
                    print(f"[seed {seed}/{cond}] cached")
                continue
            set_seed(seed)
            tr, va, te = _loaders(mp_panel, cfg_mp)
            model = build_model(cfg_mp, mp_panel, device)
            model.set_gate_from_sparsity(zf_mp)
            transferred = []
            if cond == "transfer":
                transferred, _ = transfer_load(model, pretrained_state)
                targets = {k: pretrained_state[k].to(device)
                           for k in transferred
                           if k in dict(model.named_parameters())}
                fine_tune_l2sp(model, cfg_mp, tr, va, device, targets, lam,
                               verbose=verbose)
            else:
                train_one(model, cfg_mp, tr, va, device, verbose=verbose)
            point, cal, per_crime = evaluate_calibrated(model, te, device,
                                                        cfg_mp.horizon)
            row = {"seed": seed, "condition": cond,
                   **point.as_dict(), **cal,
                   "transferred_params": len(transferred),
                   "per_crime_MAE": per_crime}
            path.write_text(json.dumps(row, indent=2, default=_json_default))
            rows.append(row)
            if verbose:
                print(f"[seed {seed}/{cond}] MAE={point.mae:.4f} "
                      f"CRPS={cal['CRPS']} cov80={cal['coverage80']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"transfer_{tag}_perseed.csv", index=False)
    agg = df.groupby("condition")[METRIC_COLS + CAL_COLS].agg(["mean", "std"])
    agg.to_csv(RESULTS_DIR / f"transfer_{tag}_meanstd.csv")
    # Flatten the multi-index columns ("MAE","mean") -> "MAE_mean" so the
    # summary JSON has only string keys (json.dumps rejects tuple keys).
    agg_flat = agg.round(4).copy()
    agg_flat.columns = ["_".join(map(str, c)) for c in agg_flat.columns]
    summary = {"tag": tag, "seeds": seeds, "lam": lam,
               "mp_zero_fraction": zf_mp,
               "categories": mp_panel["categories"],
               "mean_std": agg_flat.to_dict(orient="index")}
    (RESULTS_DIR / f"summary_{tag}_transfer.json").write_text(
        json.dumps(summary, indent=2, default=_json_default))
    if verbose:
        print("\n=== transfer vs scratch (mean) ===")
        print(agg.round(4))
    return df
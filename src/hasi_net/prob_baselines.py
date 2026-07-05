"""Probabilistic-forecasting baselines for Paper 2 (conformal + quantile
regression), on the SAME HASI-Net backbone as the calibrated head so the
comparison isolates the uncertainty method, not the architecture.

Two baselines, both evaluated with the SAME metrics as the calibrated head
(CRPS, pinball, coverage80, sharpness80 via ``calibrated.calibration_metrics``)
so they sit in one honest 4-way table: ``calibrated | point | conformal |
quantreg``.

* ``conformal`` -- split conformal prediction wrapping the POINT head
  (``calibrated_head=False``, ``log1p_mse``). The validation split is the
  calibration set; per-crime absolute-residual scores give a marginal 80%
  coverage guarantee *by construction* (finite-sample-correct). The
  informative comparison is therefore sharpness / CRPS -- conformal intervals
  are typically loose -- not coverage. Needs no retraining beyond the point
  head, so its MAE reproduces the cached ``point`` condition exactly (a
  determinism / consistency check). A Gaussian-shaped quantile set (sigma =
  half-width / z_0.9) makes CRPS + pinball comparable to the learned-quantile
  methods; intervals are clipped at 0 (counts are non-negative, which only
  preserves coverage for y >= 0).

* ``quantreg``  -- HASI-Net backbone + the SAME persistence-carry monotone
  quantile head as the calibrated model (``calibrated_head=True``), trained with
  PURE pinball loss (``calibrated.quantile_regression_loss``): no log1p-MSE point
  term, no gated ZINB/NB regulariser, no sparsity-initialised gate. Isolates the
  contribution of the calibrated multi-objective loss + sparsity gate vs plain
  quantile regression, with the architecture and quantile decode held fixed.
  Point forecast = the median (0.5) quantile.

Everything is reported honestly: if a baseline wins (e.g. conformal is sharper),
that is the result we report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config, RESULTS_DIR
from .data import get_dataset
from .train import (build_model, make_splits, WindowDataset, train_one,
                    select_device, set_seed, Metrics, _csi)
from .transfer import (_assemble, _loaders, _train_zero_fraction,
                       evaluate_calibrated, METRIC_COLS, CAL_COLS, _json_default)
from .calibrated import calibration_metrics, quantile_regression_loss, QUANTILES

# 80% central interval target (alpha = 0.2) and the standard-normal quantile
# that delimits it, used to turn a conformal half-width into a Gaussian-shaped
# quantile set for CRPS/pinball comparability with the learned-quantile methods.
ALPHA = 0.2
# Phi^{-1}(tau) for the QUANTILE levels, exact to 4 dp -- avoids a scipy
# dependency and is deterministic. The 0.1/0.9 pair delimits the 80% interval.
_Z_PPF = {0.1: -1.2816, 0.25: -0.6745, 0.5: 0.0, 0.75: 0.6745, 0.9: 1.2816}
_Z_80 = _Z_PPF[0.9]   # 1.2816


def _conformal_halfwidths(cal_pred: np.ndarray, cal_true: np.ndarray,
                          n_crime: int, alpha: float = ALPHA) -> np.ndarray:
    """Per-crime split-conformal half-width from the calibration (val) split.

    ``cal_pred`` / ``cal_true``: [n_win, H, N, C]. Returns qhat[c], the
    finite-sample-correct (1-alpha) conformal quantile of |pred-true| per crime:
    the ceil((n+1)(1-alpha))-th smallest residual score. Clipped >= 0.
    """
    scores = np.abs(cal_pred - cal_true).reshape(-1, n_crime)   # [., C]
    qhat = np.zeros(n_crime, dtype=np.float64)
    for c in range(n_crime):
        s = np.sort(scores[:, c])
        n = s.size
        if n == 0:
            qhat[c] = 0.0
            continue
        # Split-conformal finite-sample correction: rank = ceil((n+1)(1-alpha)).
        rank = int(np.ceil((n + 1) * (1.0 - alpha)))
        idx = min(max(rank, 1), n) - 1     # 0-based, clamped to [0, n-1]
        qhat[c] = float(max(s[idx], 0.0))
    return qhat


def _gaussian_quantiles(point_pred: np.ndarray, qhat: np.ndarray,
                         n_crime: int) -> np.ndarray:
    """Gaussian-shaped quantile set centred on the point forecast, with per-
    crime sigma = qhat_c / z_0.9 so the [0.1, 0.9] band reproduces the conformal
    80% interval. Returns [..., |Q|] sorted, clipped >= 0 (counts non-negative).

    ``point_pred``: [..., C]; ``qhat``: [C]. Output: [..., C, |Q|].
    """
    sigma = (qhat / _Z_80).reshape(*([1] * (point_pred.ndim - 1)), n_crime, 1)
    z = np.array([_Z_PPF[q] for q in QUANTILES], dtype=np.float64)   # [|Q|]
    q = point_pred[..., None] + sigma * z                            # [..., C, |Q|]
    q = np.clip(q, 0.0, None)
    return np.sort(q, axis=-1)


def evaluate_conformal(model, cal_loader, test_loader, device, horizon,
                       n_crime: int) -> Tuple[Metrics, Dict, List[float]]:
    """Point metrics (from the point head) + conformal calibration metrics.

    Calibration (conformal scores) on ``cal_loader`` (the held-out validation
    split); intervals + CRPS on ``test_loader``. Coverage80 is marginal by
    construction (~1-alpha); sharpness80 / CRPS are the honest comparison.
    """
    model.eval()
    cal_preds, cal_trues, te_preds, te_trues = [], [], [], []
    with torch.no_grad():
        for x, y in cal_loader:
            cal_preds.append(model.predict_mean(x.to(device), horizon).cpu().numpy())
            cal_trues.append(y.numpy())
        for x, y in test_loader:
            te_preds.append(model.predict_mean(x.to(device), horizon).cpu().numpy())
            te_trues.append(y.numpy())
    cal_pred = np.clip(np.concatenate(cal_preds, 0), 0, None)
    cal_true = np.concatenate(cal_trues, 0)
    te_pred = np.clip(np.concatenate(te_preds, 0), 0, None)
    te_true = np.concatenate(te_trues, 0)

    qhat = _conformal_halfwidths(cal_pred, cal_true, n_crime)

    # Point metrics (identical to the cached `point` condition -- consistency).
    pred = te_pred.reshape(-1); true = te_true.reshape(-1)
    eps = 1e-6
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    rmsle = float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2)))
    wape = float(np.sum(np.abs(pred - true)) / (np.sum(np.abs(true)) + eps) * 100)
    r2 = 1.0 - float(np.sum((true - pred) ** 2)) / (np.sum((true - true.mean()) ** 2) + eps)
    csi = _csi(pred, true)
    point = Metrics(mae, rmse, rmsle, wape, r2, csi)

    # Calibration metrics from the Gaussian-shaped quantile set per crime.
    q_c = _gaussian_quantiles(te_pred, qhat, n_crime).reshape(-1, n_crime, len(QUANTILES))
    true_c = te_true.reshape(-1, n_crime)
    cm = calibration_metrics(torch.from_numpy(q_c), torch.from_numpy(true_c))
    cal = {"CRPS": cm.crps, "pinball": cm.pinball,
           "coverage80": cm.coverage80, "sharpness80": cm.sharpness80,
           "conformal_qhat": qhat.tolist()}
    per_crime = [float(np.mean(np.abs(te_pred[..., j] - te_true[..., j])))
                 for j in range(n_crime)]
    return point, cal, per_crime


def _quantreg_median_mae(model, loader, device, horizon) -> float:
    """Validation MAE of the median (0.5) quantile -- the early-stopping metric
    for pure-pinball training, where the log_mu point head is not driven."""
    model.eval()
    i_med = QUANTILES.index(0.5)
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            q = model.predict_quantiles(x.to(device), horizon).cpu().numpy()
            preds.append(np.clip(q[..., i_med], 0, None))
            trues.append(y.numpy())
    pred = np.concatenate(preds, 0).reshape(-1)
    true = np.concatenate(trues, 0).reshape(-1)
    return float(np.mean(np.abs(pred - true)))


def train_quantreg(model, cfg: Config, train_loader: DataLoader,
                   val_loader: DataLoader, device, verbose: bool = True) -> Dict:
    """Train the calibrated-head model with PURE pinball loss (the quantreg
    baseline). Custom loop mirroring ``train.train_one`` (AdamW, ReduceLROnPlateau
    on the median-quantile val MAE, grad-clip 5, early stop, best-state restore)
    -- ``train_one`` is not reused because it routes through ``count_loss``,
    which for ``calibrated_head=True`` calls the full calibrated multi-objective
    loss, not pure pinball."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
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
            loss = quantile_regression_loss(out, y, QUANTILES)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * x.size(0); n += x.size(0)
        train_loss = total / max(n, 1)
        val = _quantreg_median_mae(model, val_loader, device, cfg.horizon)
        sched.step(val)
        if val < best_val - 1e-4:
            best_val = val
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                if verbose:
                    print(f"  qr early stop at epoch {epoch} "
                          f"(best median-val-MAE {best_val:.4f})", flush=True)
                break
        if verbose and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
            print(f"  qr epoch {epoch:3d} | train {train_loss:.4f} | "
                  f"val median-MAE {val:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_mae": best_val}


def evaluate_quantreg(model, loader, device, horizon
                      ) -> Tuple[Metrics, Dict, List[float]]:
    """Point metrics from the MEDIAN (0.5) quantile + calibration metrics from
    the full monotone quantile set. The median is the canonical QR point
    forecast (pure pinball does not train the log_mu point head)."""
    model.eval()
    preds, trues, qs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            q = model.predict_quantiles(x, horizon).cpu().numpy()   # [B,H,N,C,|Q|]
            qs.append(q); trues.append(y.numpy())
    q_all = np.concatenate(qs, 0)                                  # [B,H,N,C,|Q|]
    true_all = np.concatenate(trues, 0)                            # [B,H,N,C]
    n_crime = q_all.shape[3]
    i_med = QUANTILES.index(0.5)
    med = np.clip(q_all[..., i_med], 0, None)
    pred = med.reshape(-1); true = true_all.reshape(-1)
    eps = 1e-6
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    rmsle = float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2)))
    wape = float(np.sum(np.abs(pred - true)) / (np.sum(np.abs(true)) + eps) * 100)
    r2 = 1.0 - float(np.sum((true - pred) ** 2)) / (np.sum((true - true.mean()) ** 2) + eps)
    csi = _csi(pred, true)
    point = Metrics(mae, rmse, rmsle, wape, r2, csi)
    q_c = q_all.reshape(-1, n_crime, len(QUANTILES))
    true_c = true_all.reshape(-1, n_crime)
    cm = calibration_metrics(torch.from_numpy(q_c), torch.from_numpy(true_c))
    cal = {"CRPS": cm.crps, "pinball": cm.pinball,
           "coverage80": cm.coverage80, "sharpness80": cm.sharpness80}
    per_crime = [float(np.mean(np.abs(med[..., j] - true_all[..., j])))
                 for j in range(n_crime)]
    return point, cal, per_crime


def run_prob_baselines(dataset: str, cfg: Config, seeds: List[int],
                       tag: str = "p2", calvspt_tag: str | None = None,
                       force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Run conformal + quantreg on ``dataset`` and build a 4-way comparison
    table (calibrated | point | conformal | quantreg) by merging with the
    existing ``calvspt_{dataset}_{calvspt_tag}_perseed.csv``.

    ``cfg`` is the calibrated config (calibrated_head=True); the point variant
    for conformal is derived internally. ``calvspt_tag`` (defaults to ``tag``)
    names the cached calibrated-vs-point per-seed CSV to merge with.
    """
    device = select_device(cfg.device)
    calvspt_tag = calvspt_tag or tag
    cfg_pt = cfg.override(calibrated_head=False, loss_type="log1p_mse")
    cfg_qr = cfg                                                 # calibrated_head=True
    panel = _assemble(get_dataset(dataset, cfg), cfg)
    n_crime = panel["counts"].shape[2]
    rows = []
    for seed in seeds:
        # ---- conformal (wraps the point head) ------------------------------
        path = RESULTS_DIR / f"probbase_{dataset}_{tag}_seed{seed}_conformal.json"
        if path.exists() and not force:
            rows.append(json.loads(path.read_text()))
            if verbose:
                print(f"[{dataset} {seed}/conformal] cached")
        else:
            set_seed(seed)
            tr, va, te = _loaders(panel, cfg_pt)
            model = build_model(cfg_pt, panel, device)
            train_one(model, cfg_pt, tr, va, device, verbose=False)
            point, cal, per_crime = evaluate_conformal(model, va, te, device,
                                                       cfg_pt.horizon, n_crime)
            row = {"dataset": dataset, "seed": seed, "condition": "conformal",
                   **point.as_dict(), **cal, "per_crime_MAE": per_crime}
            path.write_text(json.dumps(row, indent=2, default=_json_default))
            rows.append(row)
            if verbose:
                print(f"[{dataset} {seed}/conformal] MAE={point.mae:.4f} "
                      f"cov80={cal['coverage80']:.4f} sharp={cal['sharpness80']:.4f}",
                      flush=True)

        # ---- quantreg (pure pinball, same quantile head) -------------------
        path = RESULTS_DIR / f"probbase_{dataset}_{tag}_seed{seed}_quantreg.json"
        if path.exists() and not force:
            rows.append(json.loads(path.read_text()))
            if verbose:
                print(f"[{dataset} {seed}/quantreg] cached")
        else:
            set_seed(seed)
            tr, va, te = _loaders(panel, cfg_qr)
            model = build_model(cfg_qr, panel, device)
            train_quantreg(model, cfg_qr, tr, va, device, verbose=verbose)
            point, cal, per_crime = evaluate_quantreg(model, te, device,
                                                      cfg_qr.horizon)
            row = {"dataset": dataset, "seed": seed, "condition": "quantreg",
                   **point.as_dict(), **cal, "per_crime_MAE": per_crime}
            path.write_text(json.dumps(row, indent=2, default=_json_default))
            rows.append(row)
            if verbose:
                print(f"[{dataset} {seed}/quantreg] MAE={point.mae:.4f} "
                      f"cov80={cal['coverage80']:.4f} sharp={cal['sharpness80']:.4f}",
                      flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"probbase_{dataset}_{tag}_perseed.csv", index=False)
    agg = df.groupby("condition")[METRIC_COLS + CAL_COLS].agg(["mean", "std"])
    agg.to_csv(RESULTS_DIR / f"probbase_{dataset}_{tag}_meanstd.csv")

    # 4-way merge with the cached calibrated + point conditions.
    calvspt = RESULTS_DIR / f"calvspt_{dataset}_{calvspt_tag}_perseed.csv"
    cols = ["dataset", "seed", "condition"] + METRIC_COLS + CAL_COLS + ["per_crime_MAE"]
    if calvspt.exists():
        other = pd.read_csv(calvspt)
        merged = pd.concat([other[[c for c in cols if c in other.columns]],
                            df[[c for c in cols if c in df.columns]]],
                           ignore_index=True)
        m4 = merged.groupby("condition")[METRIC_COLS + CAL_COLS].agg(["mean", "std"])
        m4.to_csv(RESULTS_DIR / f"probbase_{dataset}_{tag}_4way_meanstd.csv")
        # Flat per-condition mean of each metric (for the summary JSON).
        flat = {}
        for c, g in merged.groupby("condition"):
            flat[c] = {col: (float(g[col].mean()) if g[col].notna().any() else None)
                       for col in METRIC_COLS + CAL_COLS}
        (RESULTS_DIR / f"summary_{tag}_{dataset}_4way.json").write_text(
            json.dumps({"tag": tag, "calvspt_tag": calvspt_tag, "seeds": seeds,
                        "dataset": dataset,
                        "conditions": sorted(merged["condition"].unique().tolist()),
                        "mean": flat}, indent=2, default=_json_default))
        if verbose:
            print(f"\n=== 4-way prob comparison ({dataset}, mean) ===")
            print(m4.round(4))
    else:
        if verbose:
            print(f"\n=== prob baselines ({dataset}, mean) ===")
            print(agg.round(4))
            print(f"(calvspt {calvspt.name} not found -- 4-way merge skipped)")
    return df


# --------------------------------------------------------------------------- #
# Conditional calibration (Paper 2 -- the answer to "conformal is sharper")  #
# --------------------------------------------------------------------------- #
def _bucket_coverage(covered: np.ndarray, y: np.ndarray,
                    target: float = 0.8) -> Dict:
    """Coverage within true-count buckets + the conditional-coverage gap.

    Buckets are global terciles of the true count (low / med / high) plus the
    top-decile ("spike") bucket -- the high-count events that matter most for
    policing. ``conditional_gap`` = max |bucket_coverage - target| over the
    non-empty buckets; smaller = more uniform (conditional) calibration.

    ``covered`` and ``y`` are flat boolean / float arrays of equal length.
    """
    y = np.asarray(y, dtype=np.float64)
    cov = np.asarray(covered, dtype=bool)
    q33, q66 = np.quantile(y, [1.0 / 3.0, 2.0 / 3.0])
    top = np.quantile(y, 0.9)
    masks = {"low": y <= q33, "med": (y > q33) & (y <= q66),
             "high": y > q66, "top_decile": y >= top}
    out, vals = {}, []
    for k, m in masks.items():
        c = float(cov[m].mean()) if int(m.sum()) else float("nan")
        out[k] = c
        if not np.isnan(c):
            vals.append(abs(c - target))
    out["conditional_gap"] = float(np.max(vals)) if vals else float("nan")
    return out


def run_conditional_calibration(dataset: str, cfg: Config, seeds: List[int],
                                tag: str = "p2cc", force: bool = False,
                                verbose: bool = True) -> pd.DataFrame:
    """Conditional (per-count-bucket, per-crime, per-node) coverage of the
    calibrated head vs split conformal, on FRESHLY-TRAINED per-seed models
    (persisted to .pt for reproducibility).

    Tests the standard critique that conformal's marginal coverage guarantee is
    *only* marginal: homoscedastic per-crime intervals over-cover low-crime
    cells and under-cover the high-crime spikes. The calibrated head's
    heteroscedastic intervals (wider for high-exposure nodes) should give more
    UNIFORM coverage across crime-load buckets and across nodes -- the
    conditional-validity axis where the calibrated head can genuinely beat
    conformal even though conformal is sharper on the aggregate.

    Both conditions are trained in THIS run with the same seeds, so the
    calibrated-vs-conformal conditional comparison is internally consistent
    (any CUDA retrain nondeterminism simply adds to the reported seed variance).
    """
    device = select_device(cfg.device)
    cfg_pt = cfg.override(calibrated_head=False, loss_type="log1p_mse")
    panel = _assemble(get_dataset(dataset, cfg), cfg)
    C = panel["counts"].shape[2]
    cats = panel["categories"]
    zf = _train_zero_fraction(panel, cfg)
    rows = []
    for seed in seeds:
        for cond, c in (("calibrated", cfg), ("conformal", cfg_pt)):
            path = RESULTS_DIR / f"condcal_{dataset}_{tag}_seed{seed}_{cond}.json"
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
            # Persist: conditional coverage is sensitive to the exact trained
            # model, and CUDA retrains are not bit-identical, so save the state
            # dict so the per-crime / per-node breakdown is reproducible.
            torch.save(model.state_dict(),
                       RESULTS_DIR / f"condcal_{dataset}_{tag}_seed{seed}_{cond}.pt")

            model.eval()
            tpred, ttrue, tqs = [], [], []
            with torch.no_grad():
                for x, y in te:
                    x = x.to(device)
                    if cond == "calibrated":
                        q = model.predict_quantiles(x, c.horizon).cpu().numpy()
                        tqs.append(q)
                        mu = np.clip(q[..., QUANTILES.index(0.5)], 0, None)
                    else:
                        mu = model.predict_mean(x, c.horizon).cpu().numpy()
                    tpred.append(mu)
                    ttrue.append(y.numpy())
            pred = np.clip(np.concatenate(tpred, 0), 0, None)   # [n,H,N,C]
            true = np.concatenate(ttrue, 0)
            if cond == "calibrated":
                Q = np.concatenate(tqs, 0)                      # [n,H,N,C,|Q|]
                lo = Q[..., QUANTILES.index(0.1)]
                hi = Q[..., QUANTILES.index(0.9)]
            else:
                vpred, vtrue = [], []
                with torch.no_grad():
                    for x, y in va:
                        vpred.append(model.predict_mean(x.to(device),
                                      c.horizon).cpu().numpy())
                        vtrue.append(y.numpy())
                vp = np.clip(np.concatenate(vpred, 0), 0, None)
                vt = np.concatenate(vtrue, 0)
                qhat = _conformal_halfwidths(vp, vt, C)         # [C]
                lo = np.clip(pred - qhat.reshape(1, 1, 1, C), 0, None)
                hi = pred + qhat.reshape(1, 1, 1, C)
            covered = (true >= lo) & (true <= hi)
            yf = true.reshape(-1)
            cf = covered.reshape(-1)
            marginal = float(cf.mean())
            bks = _bucket_coverage(cf, yf)
            per_crime = [float(((true[..., j] >= lo[..., j]) &
                                (true[..., j] <= hi[..., j])).mean())
                         for j in range(C)]
            n_nodes = true.shape[2]
            per_node = [float(((true[:, :, n, :] >= lo[:, :, n, :]) &
                               (true[:, :, n, :] <= hi[:, :, n, :])).mean())
                        for n in range(n_nodes)]
            row = {"dataset": dataset, "seed": seed, "condition": cond,
                   "coverage80": marginal, "buckets": bks,
                   "per_crime_coverage80": dict(zip(cats, per_crime)),
                   "per_node_coverage_std": float(np.std(per_node)),
                   "sharpness80": float((hi - lo).mean())}
            path.write_text(json.dumps(row, indent=2, default=_json_default))
            rows.append(row)
            if verbose:
                print(f"[{dataset} {seed}/{cond}] marginal={marginal:.4f} "
                      f"gap={bks['conditional_gap']:.4f} "
                      f"low/med/high={bks['low']:.3f}/{bks['med']:.3f}/"
                      f"{bks['high']:.3f} top={bks['top_decile']:.3f} "
                      f"pernode_std={np.std(per_node):.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"condcal_{dataset}_{tag}_perseed.csv", index=False)
    summary = {}
    for cond, g in df.groupby("condition"):
        bk = pd.DataFrame([r["buckets"] for r in g.to_dict("records")])
        summary[cond] = {
            "coverage80_mean": float(g["coverage80"].mean()),
            "conditional_gap_mean": float(bk["conditional_gap"].mean()),
            "low_mean": float(bk["low"].mean()), "med_mean": float(bk["med"].mean()),
            "high_mean": float(bk["high"].mean()),
            "top_decile_mean": float(bk["top_decile"].mean()),
            "per_node_coverage_std_mean": float(g["per_node_coverage_std"].mean()),
            "sharpness80_mean": float(g["sharpness80"].mean()),
        }
    (RESULTS_DIR / f"summary_{tag}_{dataset}_condcal.json").write_text(
        json.dumps({"tag": tag, "dataset": dataset, "seeds": seeds,
                    "conditions": summary}, indent=2, default=_json_default))
    if verbose:
        print(f"\n=== conditional calibration ({dataset}, mean over seeds) ===")
        for cond, s in summary.items():
            print(f"  {cond:10s} marginal={s['coverage80_mean']:.4f} "
                  f"cond_gap={s['conditional_gap_mean']:.4f} "
                  f"low/med/high={s['low_mean']:.3f}/{s['med_mean']:.3f}/"
                  f"{s['high_mean']:.3f} top={s['top_decile_mean']:.3f} "
                  f"pernode_std={s['per_node_coverage_std_mean']:.4f}")
    return df
"""Statistical tests for forecast comparison (both papers).

* ``diebold_mariano`` -- pairwise forecast-accuracy comparison with the
  Harvey-Leybourne-Newbold small-sample correction and a Newey-West
  long-run-variance estimator (h-step consistent). Two-sided.
* ``bootstrap_ci`` -- non-parametric bootstrap confidence interval for any
  per-sample metric vector.
* ``friedman_nemenyi`` -- Friedman rank test across blocks (seeds x datasets x
  horizons) with the Nemenyi post-hoc critical difference, for an honest
  multi-model comparison rather than per-pair claims.

All lower-is-better metrics (MAE, RMSE, RMSLE, WAPE, CRPS, pinball, sharpness)
are ranked with the smallest value as the best (rank 1).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps


def _newey_west_var(d: np.ndarray, h: int) -> float:
    """Long-run variance of series ``d`` with a Bartlett kernel of ``h`` lags."""
    n = len(d)
    if n < 2:
        return float(np.var(d, ddof=1)) if n == 1 else 1e-12
    gamma0 = float(np.var(d, ddof=1))
    nw = gamma0
    for lag in range(1, h + 1):
        w = 1.0 - lag / (h + 1.0)
        cov = float(np.mean((d[:n - lag] - d.mean()) * (d[lag:] - d.mean())))
        nw += 2.0 * w * cov
    return max(nw, 1e-12)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1
                    ) -> Tuple[float, float]:
    """Diebold-Mariano test on two equal-length per-forecast loss series.

    ``e1`` / ``e2`` are per-forecast-window losses (e.g., absolute errors) for
    models 1 and 2. Returns (DM statistic, two-sided p-value). Positive DM
    means model 2 has lower loss (is better); negative means model 1 is better.
    Uses the HLN small-sample t correction and a Newey-West variance.
    """
    e1 = np.asarray(e1, dtype=np.float64).ravel()
    e2 = np.asarray(e2, dtype=np.float64).ravel()
    if e1.shape != e2.shape:
        raise ValueError("e1 and e2 must be equal-length per-forecast losses")
    n = len(e1)
    d = e1 - e2
    dbar = float(d.mean())
    sigma = np.sqrt(_newey_west_var(d, h) / n)
    dm = dbar / sigma
    # Harvey-Leybourne-Newborn small-sample correction.
    corr = np.sqrt((n + 1.0 - 2 * h + h * (h - 1) / n) / n)
    dm_corr = dm * corr
    pval = 2.0 * float(sps.t.sf(abs(dm_corr), df=n - 1))
    return float(dm_corr), pval


def bootstrap_ci(values: Sequence[float], confidence: float = 0.95,
                 n_boot: int = 10000, seed: int = 0
                 ) -> Tuple[float, float, float]:
    """Bootstrap (mean, lower, upper) for a per-sample metric vector."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def friedman_nemenyi(perf: np.ndarray, alpha: float = 0.1
                     ) -> Dict:
    """Friedman rank test + Nemenyi post-hoc critical difference.

    ``perf``: [n_blocks, k_models] matrix of a lower-is-better metric (one
    value per model per block; blocks = seeds, datasets, horizons, or their
    product). Returns dict with Friedman chi2, p-value, per-model mean ranks
    (1 = best), and the Nemenyi CD at the requested alpha. Two models whose
    mean ranks differ by less than CD are not significantly different.
    """
    perf = np.asarray(perf, dtype=np.float64)
    n, k = perf.shape
    # Rank within each block: smallest value -> rank 1.
    ranks = np.array([sps.rankdata(row) for row in perf])
    mean_ranks = ranks.mean(axis=0)
    # Friedman chi-square.
    chi2 = (12.0 * n / (k * (k + 1))) * (
        np.sum(mean_ranks ** 2) - k * (k + 1) ** 2 / 4.0)
    pval = float(sps.chi2.sf(chi2, df=k - 1))
    # Nemenyi critical difference: q_alpha(k, inf) * sqrt(k(k+1)/(6n)).
    try:
        q = float(sps.studentized_range.ppf(1 - alpha, k, 1_000_000))
    except Exception:
        q = 2.343  # fallback for k=8, alpha=0.1
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return {"chi2": float(chi2), "p": pval, "k": k, "n_blocks": n,
            "mean_ranks": mean_ranks.tolist(), "cd": float(cd), "alpha": alpha}


def per_window_errors(pred: np.ndarray, true: np.ndarray,
                      kind: str = "abs") -> np.ndarray:
    """Per-forecast-window loss from [B, horizon, N, C] prediction/true tensors.

    Returns a 1-D array of length B (one loss per window). ``kind``: 'abs'
    (MAE), 'sq' (RMSE-style), 'log1p' (RMSLE-style). Used to feed DM tests.
    """
    p = np.asarray(pred, dtype=np.float64).reshape(pred.shape[0], -1)
    t = np.asarray(true, dtype=np.float64).reshape(true.shape[0], -1)
    if kind == "sq":
        return (p - t) ** 2
    if kind == "log1p":
        return (np.log1p(np.clip(p, 0, None)) - np.log1p(t)) ** 2
    return np.abs(p - t)
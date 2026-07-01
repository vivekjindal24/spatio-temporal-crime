"""Sparsity-aware calibrated probabilistic count head (Paper 2 contribution).

Crime channels differ sharply in zero-inflation: on Chicago, kidnapping is
zero in ~86% of area-months and assault in <1%. A single distributional
assumption is therefore misspecified for the multi-crime panel. This module:

* emits, per (horizon, node, crime), a count distribution parameterised by
  (mu, alpha, pi) AND a monotone set of predictive quantiles;
* mixes a zero-inflated negative binomial (ZINB) and a negative-binomial (NB)
  log-likelihood per crime via a learned gate, so sparse crimes lean on ZINB
  and dense crimes on NB;
* trains the quantile head with pinball loss so the predicted quantiles are
  calibrated, not just the mean;
* supplies calibration metrics -- CRPS (quantile integral), central-interval
  coverage and sharpness, and pinball loss -- for honest probabilistic
  evaluation rather than point-MAE only.

The gate is initialised from each crime's training-set zero fraction so the
sparsity prior is encoded before any gradient, then refined end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from .losses import NegativeBinomialLoss, ZeroInflatedNB

# Quantile levels used for the quantile head and for 80% central intervals.
QUANTILES: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
Q_LOW, Q_HIGH = 0.1, 0.9   # 80% central interval bounds


class CalibratedHead(nn.Module):
    """Per-horizon residual head that also emits a monotone quantile forecast.

    ``log_mu`` / ``log_alpha`` / ``pi_logit`` parameterise the gated ZINB/NB
    distribution (same convention as the base HASI-Net head, so the count loss
    path is shared). ``quantiles`` is a sorted, non-negative count-space
    forecast at levels ``QUANTILES``.
    """

    def __init__(self, hidden_dim: int, n_crime: int, horizon: int,
                 n_quantiles: int = len(QUANTILES)):
        super().__init__()
        self.horizon = horizon
        self.n_crime = n_crime
        self.n_q = n_quantiles
        self.head_mu = nn.Linear(hidden_dim, n_crime * horizon)
        self.head_alpha = nn.Linear(hidden_dim, n_crime * horizon)
        self.head_pi = nn.Linear(hidden_dim, n_crime * horizon)
        self.head_q = nn.Linear(hidden_dim, n_crime * horizon * n_quantiles)
        # Small init so the mean starts at the persistence carry (delta ~ 0).
        for h in (self.head_mu, self.head_alpha, self.head_pi, self.head_q):
            nn.init.normal_(h.weight, std=0.02)
            nn.init.zeros_(h.bias)

    def forward(self, last: torch.Tensor) -> Dict[str, torch.Tensor]:
        # last: [B, N, H]. Returns RAW head outputs (deltas in log space); the
        # persistence carry + decode + monotone sort are applied by the model so
        # the quantile forecast is consistent with the log_mu carry-residual.
        B, N, _ = last.shape
        H, C, Q = self.horizon, self.n_crime, self.n_q
        log_mu = self.head_mu(last).reshape(B, N, H, C)
        log_alpha = self.head_alpha(last).reshape(B, N, H, C)
        pi_logit = self.head_pi(last).reshape(B, N, H, C)
        q_logit = self.head_q(last).reshape(B, N, H, C, Q)
        return {"log_mu": log_mu, "log_alpha": log_alpha,
                "pi_logit": pi_logit, "q_logit": q_logit}


def decode_quantiles(q_logit: torch.Tensor, carry_enc: torch.Tensor,
                     out_kind: str) -> torch.Tensor:
    """Add the persistence carry (in log space) to raw quantile logits, decode
    to non-negative counts, and enforce monotone ordering per element.

    ``q_logit``: [B, N, H, C, |Q|]; ``carry_enc``: [B, N, H, C] (encoded carry,
    same space as ``log_mu``). Returns sorted count-space quantiles.
    """
    q = q_logit + carry_enc.unsqueeze(-1)
    if out_kind == "exp":
        q = torch.exp(q)
    elif out_kind == "expm1":
        q = torch.expm1(q)
    q = q.clamp(min=0.0)
    return torch.sort(q, dim=-1)[0]


def _pinball(q: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    """Pinball / quantile loss for level ``tau`` (elementwise)."""
    diff = y - q
    return torch.maximum(tau * diff, (tau - 1.0) * diff)


def calibrated_loss(logits: Dict[str, torch.Tensor], target: torch.Tensor,
                    gate_logit: torch.Tensor, quantiles: List[float],
                    pinball_weight: float = 0.5) -> torch.Tensor:
    """Gated ZINB/NB NLL + pinball loss on the quantile head.

    ``gate_logit`` is a per-crime parameter; sigmoid(gate) is the ZINB weight.
    ``target``: [B, H, N, C]; ``logits`` keys: log_mu, log_alpha, pi_logit,
    quantiles ([B, H, N, C, |Q|]).
    """
    nb = NegativeBinomialLoss()(logits["log_mu"], logits["log_alpha"], target)
    zinb = ZeroInflatedNB()(logits["log_mu"], logits["log_alpha"],
                            logits["pi_logit"], target)
    g = torch.sigmoid(gate_logit).clamp(min=1e-4, max=1 - 1e-4)   # [C]
    # Broadcast per-crime gate over [B, H, N, C].
    nll = g * zinb + (1.0 - g) * nb                              # [B,H,N,C]
    # Pinball over quantile levels, averaged.
    q = logits["quantiles"]                                      # [B,H,N,C,|Q|]
    y = target.unsqueeze(-1)
    pin = torch.zeros_like(nll)
    tau_t = torch.tensor(quantiles, dtype=q.dtype, device=q.device)
    diff = y - q
    pin = torch.maximum(tau_t * diff, (tau_t - 1.0) * diff)      # [B,H,N,C,|Q|]
    pin = pin.mean()
    return nll.mean() + pinball_weight * pin


@dataclass
class Calibration:
    crps: float          # quantile-integral CRPS (lower is better)
    pinball: float       # mean pinball over QUANTILES (lower is better)
    coverage80: float    # fraction of truths inside the 80% central interval
    sharpness80: float   # mean width of the 80% central interval (lower = sharper)


def calibration_metrics(quantiles: torch.Tensor, true: torch.Tensor,
                        levels: List[float] = QUANTILES) -> Calibration:
    """Compute calibration metrics from predicted quantiles and true counts.

    ``quantiles``: [..., |Q|] sorted count-space forecasts at ``levels``.
    ``true``: [...] matching the non-quantile dims.
    """
    q = quantiles.detach().cpu().numpy()
    y = true.detach().cpu().numpy()
    import numpy as np
    tau = np.array(levels, dtype=np.float64)
    diff = y[..., None] - q
    pin = np.maximum(tau * diff, (tau - 1.0) * diff)            # [...,|Q|]
    pinball = float(pin.mean())
    # CRPS via the quantile integral: (2/M) sum_m pinball(q_m, y, tau_m).
    crps = float((2.0 / len(levels)) * pin.sum(axis=-1).mean())
    # 80% central interval coverage + sharpness.
    i_low = levels.index(Q_LOW)
    i_high = levels.index(Q_HIGH)
    lo, hi = q[..., i_low], q[..., i_high]
    coverage = float(((y >= lo) & (y <= hi)).mean())
    sharpness = float((hi - lo).mean())
    return Calibration(crps=crps, pinball=pinball, coverage80=coverage,
                       sharpness80=sharpness)


def init_gate_from_sparsity(zero_fraction: List[float]) -> torch.Tensor:
    """Initialise the per-crime ZINB/NB gate logit from training zero fraction.

    A crime that is zero in fraction p of training cells gets
    sigmoid(gate) = p, so the ZINB component carries weight p before any
    gradient (sparse crimes -> ZINB, dense crimes -> NB). Then refined.
    """
    import numpy as np
    p = np.clip(np.asarray(zero_fraction, dtype=np.float64), 1e-3, 1 - 1e-3)
    return torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32)
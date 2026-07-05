"""Count-aware losses for HASI-Net.

Crime counts are non-negative, right-skewed, zero-inflated and imbalanced.
A plain MSE objective under-weights the high-count hotspot districts that
matter most for policing. We therefore expose:

* ``PoissonLoss``        -- Poisson negative log-likelihood
* ``NegativeBinomialLoss`` -- NB2 parameterised by (mu, alpha)
* ``ZeroInflatedNB``     -- mixture of a point mass at 0 and NB (handles the
  many districts with zero reported cases in a given year)
* ``FocalWrapper``       -- multiplicative focal-style reweighting
  (1 - p_t)^gamma that up-weights hard, high-count examples.

All implemented in PyTorch so they live on the same device as the model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _log1p_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.exp(-x.abs())) + torch.clamp(x, min=0)


class PoissonLoss(nn.Module):
    def forward(self, log_mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu = torch.exp(log_mu)
        loss = mu - target * log_mu
        return loss.mean()


class NegativeBinomialLoss(nn.Module):
    """NB2: Var = mu + alpha * mu^2. ``log_alpha`` is predicted per-step.

    The dispersion is clamped to a sane range (alpha in [1e-2, 1e1]) so the
    likelihood cannot drift into a numerically degenerate regime (tiny alpha
    makes r=1/alpha explode, blowing up lgamma terms and the gradient). This is
    NB numerical stabilisation, not a results-affecting constraint: any real
    crime-count dispersion lies well inside this band.
    """

    def forward(self, log_mu: torch.Tensor, log_alpha: torch.Tensor,
                target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        mu = torch.exp(log_mu).clamp(max=1e6)
        alpha = torch.exp(log_alpha.clamp(-4.6, 2.3))
        r = 1.0 / alpha
        nll = (torch.lgamma(target + r) - torch.lgamma(r)
               - torch.lgamma(target + 1)
               + r * torch.log(r + mu + eps)
               + target * torch.log((mu + eps) / (r + mu + eps)))
        return nll.mean()


class ZeroInflatedNB(nn.Module):
    """Zero-inflated negative binomial.

    p(y=0) = pi + (1-pi) * NB(0)
    p(y=k) = (1-pi) * NB(k)            for k > 0
    where ``pi`` (zero-inflation logit) is predicted alongside (mu, alpha).
    """

    def forward(self, log_mu: torch.Tensor, log_alpha: torch.Tensor,
                pi_logit: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
        mu = torch.exp(log_mu).clamp(max=1e6)
        alpha = torch.exp(log_alpha.clamp(-4.6, 2.3))
        r = 1.0 / alpha
        pi = torch.sigmoid(pi_logit).clamp(min=eps, max=1 - eps)

        # NB(k; r, mu): p = r/(r+mu), NB(k) = C(k+r-1,k) p^r (1-p)^k
        log_r = torch.log(r + eps)
        log_rmu = torch.log(r + mu + eps)
        log_mu_e = torch.log(mu + eps)
        # log NB(0) = r*(log r - log(r+mu))
        log_nb_zero = r * (log_r - log_rmu)
        # log NB(k) for k>=0
        log_nb_k = (torch.lgamma(target + r) - torch.lgamma(r)
                    - torch.lgamma(target + 1)
                    + r * (log_r - log_rmu)
                    + target * (log_mu_e - log_rmu))

        is_zero = (target == 0).float()
        log_prob_zero = torch.log(pi + (1 - pi) * torch.exp(log_nb_zero) + eps)
        log_prob_pos = torch.log(1 - pi + eps) + log_nb_k
        nll = -(is_zero * log_prob_zero + (1 - is_zero) * log_prob_pos)
        return nll.mean()


class FocalWrapper(nn.Module):
    """Apply (1 - p_t)^gamma reweighting to any per-element NLL tensor."""

    def __init__(self, gamma: float = 1.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, per_element_loss: torch.Tensor,
                p_t: torch.Tensor) -> torch.Tensor:
        weight = (1.0 - p_t.clamp(min=1e-4, max=1.0)).pow(self.gamma)
        return (per_element_loss * weight).mean()


def out_kind(loss_type: str) -> str:
    """How ``log_mu`` should be decoded into a count prediction."""
    lt = loss_type.lower()
    if lt in ("zinb", "nb", "poisson"):
        return "exp"       # log_mu = log(mean)  -> mean = exp(log_mu)
    if lt.startswith("log1p"):
        return "expm1"     # log_mu = log1p(count) -> count = expm1(log_mu)
    return "raw"           # log_mu is the prediction directly (normalized space)


def decode_pred(log_mu: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "exp":
        return torch.exp(log_mu)
    if kind == "expm1":
        return torch.expm1(log_mu).clamp(min=0.0)
    return log_mu


def count_loss(logits: dict, target: torch.Tensor, cfg) -> torch.Tensor:
    """Dispatch on cfg.loss_type. ``logits`` has keys among
    log_mu, log_alpha, pi_logit depending on loss_type."""
    if getattr(cfg, "calibrated_head", False):
        from .calibrated import calibrated_loss, QUANTILES
        levels = logits.get("quantile_levels", QUANTILES)
        return calibrated_loss(logits, target, logits["gate_logit"], levels,
                               pinball_weight=getattr(cfg, "pinball_weight", 1.0))
    lt = cfg.loss_type.lower()
    if lt == "poisson":
        return PoissonLoss()(logits["log_mu"], target)
    if lt == "nb":
        return NegativeBinomialLoss()(logits["log_mu"], logits["log_alpha"], target)
    if lt == "zinb":
        base = ZeroInflatedNB()(logits["log_mu"], logits["log_alpha"],
                                logits["pi_logit"], target)
        if cfg.focal_gamma > 0:
            p_t = torch.sigmoid(logits["pi_logit"])
            weight = (1.0 - p_t.clamp(min=1e-4, max=0.999)).pow(cfg.focal_gamma)
            return base * (1.0 + weight.mean())
        return base
    if lt.startswith("log1p"):
        # MSE in log1p space -- robust to count scale and zero-inflation.
        target_t = torch.log1p(target)
        return ((logits["log_mu"] - target_t) ** 2).mean()
    # plain mse (normalized space)
    return ((logits["log_mu"] - target) ** 2).mean()
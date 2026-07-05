"""HASI-Net: Heterogeneous Adaptive Spatio-temporal Informer network.

Architecture (the paper's contribution):

1. **Heterogeneous adaptive graph block.** Three adjacency channels are fused:
   - ``A_geo``    geographic rook contiguity (fixed prior)
   - ``A_socio``  kNN cosine similarity on census features (fixed prior)
   - ``A_adaptive`` a learnable latent adjacency  softmax(ReLU(E E^T))
   Fusion weights ``alpha`` (softmax over 3 channels) are learned end-to-end,
   so the model can down-weight a noisy prior when the latent graph is more
   informative -- the key adaptation for coarse, aggregated district data.

2. **Multi-scale temporal block.** A ProbSparse (Informer) encoder captures
   long-horizon dependencies while a dilated temporal convolutional branch
   captures local short-term dynamics. A series-decomposition split into
   trend + seasonal is applied first, because yearly district counts have a
   strong trend and weak seasonality. The two branches are fused by a learned
   gate -- this is why the model works on the short (~20-step) MP series where
   plain Informer has little advantage.

3. **Count-aware head.** Three prediction heads emit log_mu, log_alpha and a
   zero-inflation logit so the count-aware loss (``losses.py``) can be applied.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


# --------------------------------------------------------------------------- #
# Graph block                                                                 #
# --------------------------------------------------------------------------- #
class AdaptiveAdjacency(nn.Module):
    """Fused heterogeneous adjacency: alpha_geo*A_geo + alpha_socio*A_socio +
    alpha_adapt*A_adaptive, with learnable alpha and learnable A_adaptive.

    When ``adaptive=False`` the latent channel is disabled (ablation).
    """

    def __init__(self, n_nodes: int, embed_dim: int = 10,
                 a_geo: Optional[torch.Tensor] = None,
                 a_socio: Optional[torch.Tensor] = None,
                 adaptive: bool = True):
        super().__init__()
        self.n = n_nodes
        self.adaptive = adaptive
        self.register_buffer("a_geo", a_geo if a_geo is not None
                             else torch.eye(n_nodes))
        self.register_buffer("a_socio", a_socio if a_socio is not None
                             else torch.eye(n_nodes))
        if adaptive:
            self.src = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.1)
            self.dst = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.1)
            self.alpha = nn.Parameter(torch.zeros(3))
        else:
            self.alpha = nn.Parameter(torch.zeros(2))

    def forward(self) -> torch.Tensor:
        w = F.softmax(self.alpha, dim=0)
        if self.adaptive:
            adapt = F.softmax(F.relu(self.src @ self.dst.t()), dim=-1)
            return w[0] * self.a_geo + w[1] * self.a_socio + w[2] * adapt
        return w[0] * self.a_geo + w[1] * self.a_socio

    def channel_weights(self) -> torch.Tensor:
        return F.softmax(self.alpha, dim=0).detach()


class GraphConvLayer(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, H]; A: [N, N]
        h = torch.einsum("mn,btnh->btmh", A, x)
        h = self.drop(F.gelu(self.lin(h)))
        return self.norm(h + x)


class SpatialBlock(nn.Module):
    def __init__(self, cfg: Config, a_geo: torch.Tensor, a_socio: torch.Tensor):
        super().__init__()
        self.adj = AdaptiveAdjacency(n_nodes=a_geo.shape[0],
                                     a_geo=a_geo, a_socio=a_socio,
                                     adaptive=cfg.adaptive_graph)
        self.layers = nn.ModuleList(
            [GraphConvLayer(cfg.hidden_dim, cfg.dropout)
             for _ in range(cfg.n_graph_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.adj()
        for layer in self.layers:
            x = layer(A, x)
        return x


# --------------------------------------------------------------------------- #
# Temporal block                                                               #
# --------------------------------------------------------------------------- #
class SeriesDecomposition(nn.Module):
    """Moving-average trend extractor (DLinear-style)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.k = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
                                padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor):  # [B, T, H]
        # Pad so the moving-average output length equals the input length T.
        # Total padding = k-1, split left/right (edge-replicated).
        total = self.k - 1
        left, right = total // 2, total - total // 2
        front = x[:, :1, :].repeat(1, left, 1)
        end = x[:, -1:, :].repeat(1, right, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        trend = self.avg(x_pad.transpose(1, 2)).transpose(1, 2)
        return trend, x - trend


class ProbSparseAttention(nn.Module):
    """Informer ProbSparse self-attention with a safe full-attention fallback
    for short sequences (where sampling offers no benefit)."""

    def __init__(self, d_model: int, n_heads: int, factor: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d = d_model // n_heads
        self.factor = factor
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _prob_Q(self, Q: torch.Tensor, K: torch.Tensor, sample_k: int, n_top: int):
        # Q: [B, h, Lq, d]; K: [B, h, Lk, d]
        B, H, Lk, _ = K.shape
        idx = torch.randint(0, Lk, (sample_k,), device=K.device)
        K_sample = K[:, :, idx, :]                       # [B,h,sample_k,d]
        Q_K = torch.einsum("bhqd,bhkd->bhqk", Q, K_sample)  # [B,h,Lq,sample_k]
        m = Q_K.max(-1)[0] - Q_K.mean(-1)                # sparsity metric
        top = m.topk(n_top, sorted=False)[1]             # [B,h,n_top]
        return top

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # each [B,h,L,d]

        if L <= 32 or self.factor <= 0:
            attn = torch.einsum("bhqd,bhkd->bhqk", Q, K) / (self.d ** 0.5)
            attn = F.softmax(attn, dim=-1)
        else:
            sample_k = min(L, max(1, L // self.factor))
            n_top = min(L, max(1, L // self.factor))
            top = self._prob_Q(Q, K, sample_k, n_top)
            Q_top = torch.gather(Q, 2,
                                 top.unsqueeze(-1).expand(-1, -1, -1, self.d))
            attn = torch.einsum("bhqd,bhkd->bhqk", Q_top, K) / (self.d ** 0.5)
            attn = F.softmax(attn, dim=-1)
            ctx = torch.einsum("bhqk,bhkd->bhqd", attn, V)
            out = torch.zeros_like(Q)
            out.scatter_(2, top.unsqueeze(-1).expand(-1, -1, -1, self.d), ctx)
            attn_out = out
            return self.drop(self.out(
                attn_out.transpose(1, 2).reshape(B, L, self.h * self.d)))

        ctx = torch.einsum("bhqk,bhkd->bhqd", attn, V)
        return self.drop(self.out(ctx.transpose(1, 2).reshape(B, L, self.h * self.d)))


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, factor: int, dropout: float):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4),
                                nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_model * 4, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class TCNBlock(nn.Module):
    """Dilated causal 1D conv branch for local temporal patterns."""

    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.pad = nn.ConstantPad1d(((kernel_size - 1), 0), 0.0)
        self.conv = nn.Conv1d(channels, channels, kernel_size)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, H]
        h = self.conv(self.pad(x.transpose(1, 2))).transpose(1, 2)
        return self.norm(self.drop(F.gelu(h)) + x)


class TemporalBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size=max(3, cfg.lookback // 2 + 1))
        self.informer = nn.ModuleList(
            [InformerEncoderLayer(cfg.hidden_dim, cfg.n_attn_heads,
                                  cfg.informer_factor, cfg.dropout)
             for _ in range(2)])
        self.tcn = TCNBlock(cfg.hidden_dim, cfg.tcn_kernel_size, cfg.dropout)
        self.gate = nn.Parameter(torch.zeros(1))  # 0 => sigmoid 0.5 even mix

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, H]
        trend, seasonal = self.decomp(x)
        s = seasonal
        for layer in self.informer:
            s = layer(s)
        local = self.tcn(seasonal)
        g = torch.sigmoid(self.gate)
        return trend + g * s + (1 - g) * local


# --------------------------------------------------------------------------- #
# Full model                                                                   #
# --------------------------------------------------------------------------- #
class HASINet(nn.Module):
    def __init__(self, cfg: Config, n_nodes: int, n_crime_types: int,
                 node_feat_dim: int, a_geo: torch.Tensor, a_socio: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        from .losses import out_kind
        self.out_kind = out_kind(cfg.loss_type)
        # Safety: attention requires hidden_dim % n_attn_heads == 0.
        if cfg.hidden_dim % cfg.n_attn_heads != 0:
            cfg = cfg.override(hidden_dim=max(
                cfg.n_attn_heads,
                (cfg.hidden_dim // cfg.n_attn_heads) * cfg.n_attn_heads))
        self.input_proj = nn.Linear(n_crime_types + node_feat_dim, cfg.hidden_dim)
        self.spatial = SpatialBlock(cfg, a_geo, a_socio)
        self.temporal = TemporalBlock(cfg)
        self.n_crime = n_crime_types
        self.horizon = cfg.horizon
        self.calibrated = bool(cfg.calibrated_head)
        # When the calibrated head is enabled the model operates in log space
        # (out_kind 'exp') so the NB/ZINB likelihoods are well posed.
        if self.calibrated:
            self.out_kind = "exp"
        # Per-horizon-step residual heads: the model predicts a separate delta
        # for each forecast step (not one flat delta broadcast across the
        # horizon), so it can learn a trend/seasonal *shape* on top of the
        # persistence carry -- the structural advantage over flat HA.
        if self.calibrated:
            from .calibrated import CalibratedHead
            self.cal_head = CalibratedHead(cfg.hidden_dim, n_crime_types,
                                           cfg.horizon)
            # Per-crime ZINB/NB gate; initialised from sparsity later.
            self.gate_logit = nn.Parameter(torch.zeros(n_crime_types))
        else:
            self.head_mu = nn.Linear(cfg.hidden_dim, n_crime_types * cfg.horizon)
            self.head_alpha = nn.Linear(cfg.hidden_dim, n_crime_types * cfg.horizon)
            self.head_pi = nn.Linear(cfg.hidden_dim, n_crime_types * cfg.horizon)
            # Small-init the residual head so the model starts near the HA
            # persistence baseline (delta ~ 0) but can learn non-trivial deltas.
            nn.init.normal_(self.head_mu.weight, std=0.02)
            nn.init.zeros_(self.head_mu.bias)
        self._node_feats = nn.Parameter(torch.zeros(n_nodes, node_feat_dim),
                                        requires_grad=False)

    def set_node_features(self, feats: torch.Tensor):
        with torch.no_grad():
            self._node_feats.copy_(feats)

    def set_gate_from_sparsity(self, zero_fraction):
        """Initialise the per-crime ZINB/NB gate from the training zero
        fraction (sparse crimes -> ZINB, dense -> NB). Paper 2 only."""
        if not self.calibrated:
            return
        from .calibrated import init_gate_from_sparsity
        with torch.no_grad():
            self.gate_logit.copy_((init_gate_from_sparsity(zero_fraction)
                                   ).to(self.gate_logit.device))

    def forward(self, x: torch.Tensor, horizon: int) -> Dict[str, torch.Tensor]:
        # x: [B, T, N, C]
        B, T, N, C = x.shape
        H = self.horizon
        nf = self._node_feats.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        h = self.input_proj(torch.cat([x, nf], dim=-1))  # [B,T,N,H]
        h = self.spatial(h)
        # Temporal over T per node.
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        h = self.temporal(h)
        last = h.reshape(B, N, T, -1)[:, :, -1, :]       # [B, N, H]

        # Persistence-residual decoding: the head predicts a per-step delta on
        # top of the lookback-mean carry-forward (HA-style persistence). With a
        # freshly initialised (small-weight) head the delta is ~0, so the model
        # starts at the HA baseline and training only improves from there. The
        # per-step delta (vs one flat delta) lets the model learn a trend /
        # seasonal shape across the horizon -- the structural advantage over
        # flat HA persistence.
        carry = x.mean(dim=1)                            # [B, N, C]
        if self.out_kind == "exp":
            carry_enc = torch.log(carry.clamp(min=1e-3))
        elif self.out_kind == "expm1":
            carry_enc = torch.log1p(carry.clamp(min=0.0))
        else:
            carry_enc = carry
        carry_enc = carry_enc.unsqueeze(2).expand(-1, -1, H, -1)  # [B,N,H,C]
        # Raw count-space carry (>= 0, can be 0) for the quantile decode: lets
        # the lower quantile reach 0 so zero-inflated crimes cover y=0. The
        # log_mu / NB path still uses the log-encoded carry_enc above.
        carry_raw = carry.clamp(min=0.0).unsqueeze(2).expand(-1, -1, H, -1)

        if self.calibrated:
            from .calibrated import decode_quantiles, QUANTILES
            hd = self.cal_head(last)                       # raw deltas [B,N,H,C[Q]]
            log_mu = (carry_enc + hd["log_mu"]).permute(0, 2, 1, 3)
            alpha = hd["log_alpha"].permute(0, 2, 1, 3)
            pi = hd["pi_logit"].permute(0, 2, 1, 3)
            q = decode_quantiles(hd["q_logit"], carry_raw, self.out_kind)
            q = q.permute(0, 2, 1, 3, 4)                   # [B,H,N,C,|Q|]
            return {"log_mu": log_mu, "log_alpha": alpha, "pi_logit": pi,
                    "quantiles": q, "gate_logit": self.gate_logit,
                    "quantile_levels": QUANTILES}

        delta_mu = self.head_mu(last).reshape(B, N, H, C)        # [B,N,H,C]
        log_mu = carry_enc + delta_mu
        alpha = self.head_alpha(last).reshape(B, N, H, C)
        pi = self.head_pi(last).reshape(B, N, H, C)
        # -> [B, horizon, N, C]
        log_mu = log_mu.permute(0, 2, 1, 3)
        alpha = alpha.permute(0, 2, 1, 3)
        pi = pi.permute(0, 2, 1, 3)
        return {"log_mu": log_mu, "log_alpha": alpha, "pi_logit": pi}

    @torch.no_grad()
    def predict_mean(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        from .losses import decode_pred
        out = self.forward(x, horizon)
        return decode_pred(out["log_mu"], self.out_kind)

    @torch.no_grad()
    def predict_quantiles(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        """Count-space monotone quantile forecast [B, horizon, N, C, |Q|].
        Paper 2 calibrated head only."""
        if not self.calibrated:
            raise RuntimeError("predict_quantiles requires calibrated_head=True")
        out = self.forward(x, horizon)
        return out["quantiles"]
        out = self.forward(x, horizon)
        return decode_pred(out["log_mu"], self.out_kind)
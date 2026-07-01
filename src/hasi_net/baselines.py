"""Baseline models for comparison.

Every baseline shares HASI-Net's head convention: ``forward`` returns a dict
with ``log_mu`` (the raw head output, decoded by ``out_kind``), and
``predict_mean`` decodes it into a count prediction. This keeps the loss and
evaluation paths identical across models so comparisons are fair.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .losses import out_kind, decode_pred
from .model import SpatialBlock, TemporalBlock


class _GatedDilatedConv(nn.Module):
    """Gated dilated causal 1-D convolution (Graph WaveNet / MTGNN temporal
    block). Left-padded so the convolution is causal; gated tanh activation."""

    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 dropout: float):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size,
                              dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, H]
        h = F.pad(x.transpose(1, 2), (self.pad, 0), value=0.0)  # causal left-pad
        h = self.conv(h).transpose(1, 2)                       # [B, T, 2H]
        a, g = h.chunk(2, dim=-1)
        h = a * torch.tanh(g)
        return self.norm(self.drop(h) + x)


class _DiffusionConv(nn.Module):
    """Diffusion convolution (DCRNN): forward + backward random-walk steps over
    a predefined graph, K=2 (one step each direction)."""

    def __init__(self, in_h: int, out_h: int, a_geo: torch.Tensor):
        super().__init__()
        self.register_buffer("p_fwd", a_geo)
        col = a_geo.t().sum(dim=1, keepdim=True).clamp(min=1e-6)
        self.register_buffer("p_bwd", a_geo.t() / col)
        self.W = nn.Linear(in_h * 3, out_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., N, H]
        h0 = x
        h1 = torch.einsum("mn,...nh->...mh", self.p_fwd, x)
        h2 = torch.einsum("mn,...nh->...mh", self.p_bwd, x)
        return self.W(torch.cat([h0, h1, h2], dim=-1))


def _pad_heads(log_mu: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"log_mu": log_mu, "log_alpha": torch.zeros_like(log_mu),
            "pi_logit": torch.full_like(log_mu, -4.0)}


class CountModel(nn.Module):
    """Shared out_kind dispatch + node-feature storage."""

    def __init__(self, cfg: Config, n_nodes: int, nf_dim: int):
        super().__init__()
        self.out_kind = out_kind(cfg.loss_type)
        self._node_feats = nn.Parameter(torch.zeros(n_nodes, nf_dim),
                                        requires_grad=False)

    def set_node_features(self, feats: torch.Tensor):
        with torch.no_grad():
            self._node_feats.copy_(feats)

    def _carry_enc(self, x: torch.Tensor) -> torch.Tensor:
        """Encoded lookback-mean carry, shared by every deep model for a fair
        comparison. HA is exactly this carry with delta=0; deep models predict
        a learned delta on top, so the comparison isolates the architecture."""
        carry = x.mean(dim=1)                       # [B, N, C]
        if self.out_kind == "exp":
            return torch.log(carry.clamp(min=1e-3))
        if self.out_kind == "expm1":
            return torch.log1p(carry.clamp(min=0.0))
        return carry

    @torch.no_grad()
    def predict_mean(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        return decode_pred(self.forward(x, horizon)["log_mu"], self.out_kind)


class HistoricalAverage(CountModel):
    """Forecast = mean count over the lookback window, per node/category."""

    def __init__(self, cfg: Config, n_nodes: int, nf_dim: int):
        super().__init__(cfg, n_nodes, nf_dim)

    def forward(self, x: torch.Tensor, horizon: int) -> Dict[str, torch.Tensor]:
        mean = x.mean(dim=1)                              # [B, N, C]
        mu = mean.unsqueeze(2).expand(-1, -1, horizon, -1).permute(0, 2, 1, 3)
        # Encode the mean into the model's output space.
        if self.out_kind == "exp":
            log_mu = torch.log(mu.clamp(min=1e-3))
        elif self.out_kind == "expm1":
            log_mu = torch.log1p(mu.clamp(min=0.0))
        else:
            log_mu = mu
        return _pad_heads(log_mu)


class LSTMBaseline(CountModel):
    def __init__(self, cfg: Config, n_nodes: int, n_crime: int, nf_dim: int):
        super().__init__(cfg, n_nodes, nf_dim)
        self.proj = nn.Linear(n_crime + nf_dim, cfg.hidden_dim)
        self.lstm = nn.LSTM(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.head = nn.Linear(cfg.hidden_dim, n_crime)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, horizon: int) -> Dict[str, torch.Tensor]:
        B, T, N, C = x.shape
        nf = self._node_feats.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        h = self.proj(torch.cat([x, nf], -1)).reshape(B * N, T, -1)
        out, _ = self.lstm(h)
        last = out[:, -1, :].reshape(B, N, -1)
        delta = self.head(last)                       # [B, N, C] residual on carry
        log_mu = self._carry_enc(x) + delta
        log_mu = log_mu.unsqueeze(2).expand(-1, -1, horizon, -1).permute(0, 2, 1, 3)
        return _pad_heads(log_mu)


class STGCNBaseline(CountModel):
    """Plain ST-GCN: fixed geographic graph + temporal convolution. No adaptive
    adjacency, no Informer, no socioeconomic channel."""

    def __init__(self, cfg: Config, n_nodes: int, n_crime: int, nf_dim: int,
                 a_geo: torch.Tensor):
        super().__init__(cfg, n_nodes, nf_dim)
        self.cfg = cfg
        self.input_proj = nn.Linear(n_crime + nf_dim, cfg.hidden_dim)
        self.spatial = SpatialBlock(cfg, a_geo, torch.eye(n_nodes))
        self.temporal = nn.Sequential(
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, 3, padding=1),
            nn.GELU(), nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, 3, padding=1))
        self.head = nn.Linear(cfg.hidden_dim, n_crime)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, horizon: int) -> Dict[str, torch.Tensor]:
        B, T, N, C = x.shape
        nf = self._node_feats.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        h = self.input_proj(torch.cat([x, nf], -1))
        h = self.spatial(h).permute(0, 2, 1, 3).reshape(B * N, T, -1)
        h = self.temporal(h.transpose(1, 2)).transpose(1, 2)
        last = h[:, -1, :].reshape(B, N, -1)
        delta = self.head(last)                       # [B, N, C] residual on carry
        log_mu = self._carry_enc(x) + delta
        log_mu = log_mu.unsqueeze(2).expand(-1, -1, horizon, -1).permute(0, 2, 1, 3)
        return _pad_heads(log_mu)


class InformerOnlyBaseline(CountModel):
    """Temporal Informer block only (no graph). Tests whether the spatial
    branch is responsible for gains on aggregated data."""

    def __init__(self, cfg: Config, n_nodes: int, n_crime: int, nf_dim: int):
        super().__init__(cfg, n_nodes, nf_dim)
        self.cfg = cfg
        self.input_proj = nn.Linear(n_crime + nf_dim, cfg.hidden_dim)
        self.temporal = TemporalBlock(cfg)
        self.head = nn.Linear(cfg.hidden_dim, n_crime)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, horizon: int) -> Dict[str, torch.Tensor]:
        B, T, N, C = x.shape
        nf = self._node_feats.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        h = self.input_proj(torch.cat([x, nf], -1))
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        h = self.temporal(h)
        last = h.reshape(B, N, T, -1)[:, :, -1, :]
        delta = self.head(last)                       # [B, N, C] residual on carry
        log_mu = self._carry_enc(x) + delta
        log_mu = log_mu.unsqueeze(2).expand(-1, -1, horizon, -1).permute(0, 2, 1, 3)
        return _pad_heads(log_mu)


class GraphWaveNetBaseline(CountModel):
    """Graph WaveNet: stacked gated dilated causal temporal convolutions over a
    fixed predefined graph (A_geo) fused with a self-adaptive learned adjacency
    softmax(ReLU(E1 E2^T)) -- the paper's adaptive adjacency component. Shares
    the persistence-residual head for a fair, head-controlled comparison."""

    def __init__(self, cfg: Config, n_nodes: int, n_crime: int, nf_dim: int,
                 a_geo: torch.Tensor):
        super().__init__(cfg, n_nodes, nf_dim)
        self.cfg = cfg
        H = cfg.hidden_dim
        self.input_proj = nn.Linear(n_crime + nf_dim, H)
        self.register_buffer("a_geo", a_geo)
        self.src = nn.Parameter(torch.randn(n_nodes, 10) * 0.1)
        self.dst = nn.Parameter(torch.randn(n_nodes, 10) * 0.1)
        self.lin_g = nn.Linear(H, H)
        self.tcn = nn.ModuleList(
            [_GatedDilatedConv(H, cfg.tcn_kernel_size, d, cfg.dropout)
             for d in (1, 2)])
        self.head = nn.Linear(H, n_crime)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, horizon: int) -> Dict[str, torch.Tensor]:
        B, T, N, C = x.shape
        nf = self._node_feats.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        h = self.input_proj(torch.cat([x, nf], -1))            # [B,T,N,H]
        adapt = F.softmax(F.relu(self.src @ self.dst.t()), dim=-1)
        A = self.a_geo + adapt                                 # predefined + adaptive
        g = F.gelu(self.lin_g(torch.einsum("mn,btnh->btmh", A, h)))
        g = g.permute(0, 2, 1, 3).reshape(B * N, T, -1)        # per-node temporal
        for layer in self.tcn:
            g = layer(g)
        last = g[:, -1, :].reshape(B, N, -1)
        delta = self.head(last)
        log_mu = self._carry_enc(x) + delta
        log_mu = log_mu.unsqueeze(2).expand(-1, -1, horizon, -1).permute(0, 2, 1, 3)
        return _pad_heads(log_mu)


class DCRNNBaseline(CountModel):
    """DCRNN: diffusion-convolutional GRU encoder over the predefined graph
    (A_geo). One GRU layer with diffusion-conv input/hidden updates; the
    persistence-residual head maps the final state to a count delta."""

    def __init__(self, cfg: Config, n_nodes: int, n_crime: int, nf_dim: int,
                 a_geo: torch.Tensor):
        super().__init__(cfg, n_nodes, nf_dim)
        self.cfg = cfg
        H = cfg.hidden_dim
        self.input_proj = nn.Linear(n_crime + nf_dim, H)
        self.dc_in = _DiffusionConv(H, H, a_geo)
        self.dc_hid = _DiffusionConv(H, H, a_geo)
        self.gates = nn.Linear(2 * H, 3 * H)
        self.head = nn.Linear(H, n_crime)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, horizon: int) -> Dict[str, torch.Tensor]:
        B, T, N, C = x.shape
        H = self.cfg.hidden_dim
        nf = self._node_feats.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        h_in = self.input_proj(torch.cat([x, nf], -1))         # [B,T,N,H]
        h = torch.zeros(B, N, H, device=x.device)
        for t in range(T):
            xt = self.dc_in(h_in[:, t])
            ht = self.dc_hid(h)
            r, u, c = self.gates(torch.cat([xt, ht], -1)).chunk(3, dim=-1)
            r = torch.sigmoid(r)
            u = torch.sigmoid(u)
            c = torch.tanh(r * ht + xt)
            h = u * h + (1 - u) * c
        delta = self.head(h)
        log_mu = self._carry_enc(x) + delta
        log_mu = log_mu.unsqueeze(2).expand(-1, -1, horizon, -1).permute(0, 2, 1, 3)
        return _pad_heads(log_mu)


class MTGNNBaseline(CountModel):
    """MTGNN: a graph learned from data (adaptive adjacency only, no fixed
    prior) feeding stacked 1-D gated dilated convolutions with adaptive layer
    aggregation (learned weights over residual layers). Persistence-residual
    head for a fair, head-controlled comparison."""

    def __init__(self, cfg: Config, n_nodes: int, n_crime: int, nf_dim: int):
        super().__init__(cfg, n_nodes, nf_dim)
        self.cfg = cfg
        H = cfg.hidden_dim
        self.input_proj = nn.Linear(n_crime + nf_dim, H)
        self.src = nn.Parameter(torch.randn(n_nodes, 10) * 0.1)
        self.dst = nn.Parameter(torch.randn(n_nodes, 10) * 0.1)
        self.lin_g = nn.Linear(H, H)
        self.dilations = (1, 2, 4)
        self.tcn = nn.ModuleList(
            [_GatedDilatedConv(H, cfg.tcn_kernel_size, d, cfg.dropout)
             for d in self.dilations])
        self.agg = nn.Parameter(torch.zeros(len(self.dilations) + 1))
        self.head = nn.Linear(H, n_crime)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, horizon: int) -> Dict[str, torch.Tensor]:
        B, T, N, C = x.shape
        nf = self._node_feats.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        h = self.input_proj(torch.cat([x, nf], -1))
        A = F.softmax(F.relu(self.src @ self.dst.t()), dim=-1)  # learned graph only
        g = F.gelu(self.lin_g(torch.einsum("mn,btnh->btmh", A, h)))
        g = g.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        w = F.softmax(self.agg, dim=0)
        out = w[0] * g
        for i, layer in enumerate(self.tcn):
            g = layer(g)
            out = out + w[i + 1] * g
        last = g[:, -1, :].reshape(B, N, -1)
        delta = self.head(last)
        log_mu = self._carry_enc(x) + delta
        log_mu = log_mu.unsqueeze(2).expand(-1, -1, horizon, -1).permute(0, 2, 1, 3)
        return _pad_heads(log_mu)


def build_baseline(name: str, cfg: Config, panel, device: torch.device):
    import torch as _t
    A_geo = _t.from_numpy(panel["a_geo"]).float().to(device)
    n_nodes = panel["counts"].shape[1]
    n_crime = panel["counts"].shape[2]
    nf_dim = panel["node_feats"].shape[1]
    nf = _t.from_numpy(panel["node_feats"]).float().to(device)

    if name == "HA":
        m = HistoricalAverage(cfg, n_nodes, nf_dim).to(device)
    elif name == "LSTM":
        m = LSTMBaseline(cfg, n_nodes, n_crime, nf_dim).to(device)
    elif name == "STGCN":
        m = STGCNBaseline(cfg, n_nodes, n_crime, nf_dim, A_geo).to(device)
    elif name == "GraphWaveNet":
        m = GraphWaveNetBaseline(cfg, n_nodes, n_crime, nf_dim, A_geo).to(device)
    elif name == "DCRNN":
        m = DCRNNBaseline(cfg, n_nodes, n_crime, nf_dim, A_geo).to(device)
    elif name == "MTGNN":
        m = MTGNNBaseline(cfg, n_nodes, n_crime, nf_dim).to(device)
    elif name == "InformerOnly":
        m = InformerOnlyBaseline(cfg, n_nodes, n_crime, nf_dim).to(device)
    else:
        raise ValueError(f"unknown baseline {name}")
    m.set_node_features(nf)
    return m


BASELINES = ["HA", "LSTM", "STGCN", "GraphWaveNet", "DCRNN", "MTGNN",
             "InformerOnly"]
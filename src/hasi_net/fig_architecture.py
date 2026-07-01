"""Generate the HASI-Net architecture diagram (a real implementation figure).

Produces ``results/architecture.png`` -- a labelled block diagram of the full
pipeline: input panel -> heterogeneous adaptive graph block -> multi-scale
temporal block (series decomposition + Informer/TCN gate) -> persistence-
residual count-aware head. Cited in both papers.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from .config import RESULTS_DIR

C_INPUT = "#34495e"
C_GRAPH = "#27ae60"
C_TEMP = "#2980b9"
C_HEAD = "#c0392b"
C_LOSS = "#8e44ad"
C_BOX = "#ecf0f1"


def _box(ax, x, y, w, h, text, color, fc=C_BOX, fs=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.6, edgecolor=color, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color="#1a1a1a", wrap=True)


def _arrow(ax, x1, y1, x2, y2, color="#2c3e50"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.5, color=color))


def plot_architecture(out: str = "architecture.png") -> Path:
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.2); ax.set_axis_off()

    ax.text(6, 7.0, "HASI-Net: Heterogeneous Adaptive Spatio-temporal Informer Network",
            ha="center", fontsize=13, fontweight="bold")

    # Input
    _box(ax, 0.3, 5.5, 2.2, 0.9, "Input panel\n[T, N, C] counts\n+ [N, F] census feats",
         C_INPUT, fs=9, weight="bold")

    # Heterogeneous adaptive graph block
    _box(ax, 3.0, 5.7, 3.6, 0.55, "A_geo  (rook contiguity)", C_GRAPH, fs=9)
    _box(ax, 3.0, 5.05, 3.6, 0.55, "A_socio (kNN cosine on census)", C_GRAPH, fs=9)
    _box(ax, 3.0, 4.4, 3.6, 0.55, "A_adapt = softmax(ReLU(E E^T))", C_GRAPH, fs=9)
    _box(ax, 3.0, 3.75, 3.6, 0.5, "fusion:  alpha_geo A_geo + alpha_socio A_socio + alpha_adapt A_adapt",
         C_GRAPH, fs=8)
    ax.text(4.8, 3.55, "learnable softmax(alpha)  +  learnable E",
            ha="center", fontsize=8, style="italic", color=C_GRAPH)
    ax.text(4.8, 6.55, "Heterogeneous Adaptive Graph Block",
            ha="center", fontsize=10, fontweight="bold", color=C_GRAPH)

    # Spatial graph conv
    _box(ax, 3.4, 2.6, 2.8, 0.7, "Graph Conv layers\n(SpatialBlock)", C_GRAPH, fs=9)

    # Multi-scale temporal block
    _box(ax, 6.8, 5.4, 4.9, 0.7, "Series decomposition  ->  trend + seasonal",
         C_TEMP, fs=9)
    _box(ax, 6.8, 4.5, 2.3, 0.7, "Informer\nProbSparse attn", C_TEMP, fs=9)
    _box(ax, 9.4, 4.5, 2.3, 0.7, "Dilated TCN\n(local)", C_TEMP, fs=9)
    _box(ax, 6.8, 3.6, 4.9, 0.55, "learned gate:  trend + g*Informer + (1-g)*TCN",
         C_TEMP, fs=8)
    ax.text(9.25, 6.45, "Multi-scale Temporal Block", ha="center",
            fontsize=10, fontweight="bold", color=C_TEMP)

    # Persistence-residual head
    _box(ax, 6.8, 2.5, 4.9, 0.7,
         "Persistence-residual head:\nlog_mu = enc(carry) + delta_mu ,  carry = lookback mean",
         C_HEAD, fs=8.5)
    _box(ax, 6.8, 1.7, 4.9, 0.55, "log_alpha  (NB dispersion)  +  pi_logit  (zero-inflation)",
         C_HEAD, fs=8.5)
    ax.text(9.25, 2.2, "Count-aware head", ha="center", fontsize=9,
            fontweight="bold", color=C_HEAD)

    # Loss
    _box(ax, 4.6, 0.5, 3.0, 0.7,
         "Count-aware loss\nlog1p-MSE / ZINB + focal", C_LOSS, fs=9, weight="bold")

    # PSO optimizer (side)
    _box(ax, 0.3, 2.6, 2.6, 1.0,
         "Adaptive-inertia\nmulti-swarm PSO\n(hidden, layers, heads,\ndropout, lr)",
         "#d35400", fs=8.5)

    # Arrows
    _arrow(ax, 2.5, 5.95, 3.0, 5.95)             # input -> graph block
    _arrow(ax, 4.8, 3.75, 4.8, 3.3)              # graph block -> spatial conv
    _arrow(ax, 6.2, 2.95, 6.8, 4.0)              # spatial -> temporal (residual path)
    _arrow(ax, 9.25, 3.6, 9.25, 3.2)             # temporal -> head
    _arrow(ax, 9.25, 2.5, 7.6, 1.2)              # head -> loss
    _arrow(ax, 2.9, 3.1, 3.4, 2.95, "#d35400")   # PSO -> spatial (tunes)
    ax.text(3.1, 3.35, "tunes", fontsize=7.5, color="#d35400", style="italic")

    fig.tight_layout()
    p = RESULTS_DIR / out
    fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    return p


if __name__ == "__main__":
    p = plot_architecture()
    print("wrote", p)
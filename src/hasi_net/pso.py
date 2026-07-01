"""Adaptive multi-swarm Particle Swarm Optimization for HASI-Net.

Searches five hyperparameters: hidden_dim, n_graph_layers, n_attn_heads,
dropout, learning rate. Each particle encodes a ``Config`` override; its
fitness is the validation MAE of a short training run. We use linearly
decreasing inertia weight (adaptive PSO) and a small number of particles so
the search is affordable on an M1 GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from .config import Config

# (name, low, high, is_int)  -- the search space.
SPACE: List[Tuple[str, float, float, bool]] = [
    ("hidden_dim", 32.0, 128.0, True),
    ("n_graph_layers", 1.0, 3.0, True),
    ("n_attn_heads", 2.0, 8.0, True),
    ("dropout", 0.05, 0.40, False),
    ("lr", 1e-4, 5e-3, False),
]


@dataclass
class Particle:
    pos: np.ndarray
    vel: np.ndarray
    best_pos: np.ndarray
    best_fit: float = float("inf")


def _decode(pos: np.ndarray) -> Dict:
    out = {}
    for (name, lo, hi, is_int), v in zip(SPACE, pos):
        v = float(np.clip(v, lo, hi))
        out[name] = int(round(v)) if is_int else v
    # Keep hidden_dim divisible by n_attn_heads (attention head constraint).
    nh = out["n_attn_heads"]
    out["hidden_dim"] = max(nh, (out["hidden_dim"] // nh) * nh)
    return out


def _random_pos(rng: np.random.Generator) -> np.ndarray:
    return np.array([
        rng.uniform(lo, hi) for (_, lo, hi, _) in SPACE
    ], dtype=np.float32)


def init_swarm(n_particles: int, seed: int) -> List[Particle]:
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(n_particles):
        p = _random_pos(rng)
        parts.append(Particle(pos=p, vel=rng.normal(0, 0.05, size=len(SPACE)).astype(np.float32),
                              best_pos=p.copy()))
    return parts


def run_pso(fitness: Callable[[Config], float], cfg: Config,
            verbose: bool = True) -> Tuple[Config, Dict]:
    """Run adaptive PSO. ``fitness(cfg) -> val MAE`` (lower is better).

    Returns the best config and a history dict for plotting convergence.
    """
    rng = np.random.default_rng(cfg.seed)
    total_particles = cfg.pso_particles * cfg.pso_swarms
    particles = init_swarm(total_particles, cfg.seed)
    gbest_pos = particles[0].pos.copy()
    gbest_fit = float("inf")
    history = {"iter": [], "gbest": [], "mean": []}

    w_max, w_min = 0.9, 0.4
    c1, c2 = 1.8, 1.8

    for it in range(cfg.pso_iters):
        w = w_max - (w_max - w_min) * it / max(1, cfg.pso_iters - 1)
        fits = []
        for p in particles:
            cfg_i = cfg.override(**_decode(p.pos))
            fit = fitness(cfg_i)
            fits.append(fit)
            if fit < p.best_fit:
                p.best_fit = fit
                p.best_pos = p.pos.copy()
            if fit < gbest_fit:
                gbest_fit = fit
                gbest_pos = p.pos.copy()

        # Velocity + position update (vectorized).
        for p in particles:
            r1, r2 = rng.random(size=len(SPACE)).astype(np.float32), \
                rng.random(size=len(SPACE)).astype(np.float32)
            p.vel = (w * p.vel
                     + c1 * r1 * (p.best_pos - p.pos)
                     + c2 * r2 * (gbest_pos - p.pos))
            p.pos = p.pos + p.vel
        history["iter"].append(it)
        history["gbest"].append(gbest_fit)
        history["mean"].append(float(np.mean(fits)))
        if verbose:
            print(f"  PSO iter {it}: gbest MAE={gbest_fit:.4f}  "
                  f"mean={np.mean(fits):.4f}  w={w:.3f}")

    best_cfg = cfg.override(**_decode(gbest_pos))
    return best_cfg, history


def short_fitness(panel, cfg_eval: Config, device, epochs: int = 15):
    """Build a fitness closure that trains a reduced-epoch HASI-Net and returns
    validation MAE. Imported lazily to avoid a circular import at module load."""
    from .train import (build_model, make_splits, train_one, evaluate,
                        WindowDataset)
    from torch.utils.data import DataLoader

    def fit(cfg: Config) -> float:
        T = panel["counts"].shape[0]
        tr, va, _ = make_splits(T, cfg.lookback, cfg.horizon)
        if not tr or not va:
            return float("inf")
        model = build_model(cfg, panel, device)
        tr_loader = DataLoader(WindowDataset(panel["counts"], cfg.lookback,
                                              cfg.horizon, tr),
                               batch_size=cfg.batch_size, shuffle=True)
        va_loader = DataLoader(WindowDataset(panel["counts"], cfg.lookback,
                                             cfg.horizon, va),
                               batch_size=cfg.batch_size)
        cfg_short = cfg.override(epochs=epochs, patience=20)
        train_one(model, cfg_short, tr_loader, va_loader, device, verbose=False)
        m = evaluate(model, va_loader, device, cfg.horizon)
        return m.mae

    return fit
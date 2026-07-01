"""Central configuration for HASI-Net.

All paths and tunable defaults live here so the notebooks and modules never
hardcode values. Tweak this file (or override keys via ``Config.override``) to
run experiments at different scales.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

# Resolve paths relative to the repo root (parent of src/).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
# RESULTS_DIR defaults to <repo>/results but can be redirected (e.g. to a
# mounted Google Drive path on Colab) via the HASI_RESULTS_DIR env var, so
# experiment outputs survive Colab disconnects when the repo is cloned to the
# ephemeral /content filesystem.
import os as _os
RESULTS_DIR = Path(_os.environ.get("HASI_RESULTS_DIR", REPO_ROOT / "results"))

# --- Unified crime vocabulary (used by BOTH datasets) ------------------------
# The two papers address four women-centric crimes chosen for cross-dataset
# availability and volume (see the data-availability scan in the project notes):
#   rape_sexual_assault  -- NCRB "rape" | Chicago "CRIMINAL/CRIM SEXUAL ASSAULT"
#   domestic_violence    -- NCRB "cruelty by husband/relatives"
#                          | Chicago domestic-flagged BATTERY (proxy; no victim-sex field)
#   kidnapping_abduction -- NCRB "kidnapping & abduction" | Chicago "KIDNAPPING"
#   assault              -- NCRB "assault on women w/ intent to outrage modesty"
#                          | Chicago "ASSAULT" (generic; weakest semantic match)
# Both panels emit counts[..., C] in THIS canonical order so the cross-region
# transfer head (Paper 2) sees matching category dimensionality.
UNIFIED_CRIMES = [
    "rape_sexual_assault",
    "domestic_violence",
    "kidnapping_abduction",
    "assault",
]

# NCRB native categories that map onto the unified vocabulary above. The
# India-only heads (dowry deaths, insult to modesty, importation of girls) are
# deliberately excluded: they have no Chicago analog and "importation of girls"
# is degenerate (67 total, 97% zeros).
WOMEN_CRIME_CATEGORIES = UNIFIED_CRIMES

MADHYA_PRADESH = "Madhya Pradesh"


@dataclass
class Config:
    # --- Data ----------------------------------------------------------------
    target_region: str = MADHYA_PRADESH
    use_chicago_benchmark: bool = True
    # Year window for the MP panel. We use the contiguous, fully-real NCRB
    # 2001-2014 district block (downloadable as clean CSVs). The 2015-2016
    # reports and the 2017-2022 India Data Portal release are not reliably
    # machine-readable, so we do NOT interpolate forward-filled pseudo-years.
    mp_year_start: int = 2001
    mp_year_end: int = 2014
    # Chicago aggregation: community-area x month grid.
    chicago_year_start: int = 2010
    chicago_year_end: int = 2023

    # --- Graph ---------------------------------------------------------------
    socio_knn: int = 5              # k-nearest socioeconomic-similarity edges
    adaptive_graph: bool = True     # learnable latent adjacency component

    # --- Model (HASI-Net) ----------------------------------------------------
    hidden_dim: int = 64
    n_graph_layers: int = 2
    n_attn_heads: int = 4
    informer_factor: int = 5        # ProbSparse sampling factor
    tcn_channels: int = 32
    tcn_kernel_size: int = 3
    dropout: float = 0.15
    fusion_alpha_init: float = 0.34  # init weight for each of 3 adjacency types

    # --- Temporal windowing --------------------------------------------------
    lookback: int = 4               # input length (in MP: 4 years)
    horizon: int = 2                # forecast length (in MP: 2 years)

    # --- Count-aware loss ----------------------------------------------------
    # "log1p_mse" is the robust default (MSE in log1p space; handles the huge
    # count-scale variance and zero-inflation of district crime data).
    # "zinb" (zero-inflated negative binomial + focal) is available as an
    # ablation variant and is the count-aware contribution tested on Chicago.
    loss_type: str = "log1p_mse"     # "log1p_mse" | "zinb" | "nb" | "poisson" | "mse"
    focal_gamma: float = 1.5
    # --- Calibrated probabilistic head (Paper 2) -----------------------------
    # When True, HASI-Net uses the sparsity-aware gated ZINB/NB + quantile head
    # (calibrated.py) and is evaluated with CRPS / coverage / sharpness. The
    # per-crime ZINB/NB gate is initialised from the training zero fraction.
    calibrated_head: bool = False
    pinball_weight: float = 0.5     # weight of the quantile pinball term

    # --- Training ------------------------------------------------------------
    epochs: int = 80
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 12
    device: str = "auto"             # "auto" picks MPS/CUDA/CPU

    # --- PSO -----------------------------------------------------------------
    pso_enabled: bool = True
    pso_swarms: int = 3
    pso_particles: int = 8
    pso_iters: int = 6
    pso_search_dim: int = 5          # hidden_dim, n_graph_layers, heads, dropout, lr

    # --- Reproducibility -----------------------------------------------------
    seed: int = 42

    def override(self, **kwargs: Any) -> "Config":
        """Return a new Config with selected fields replaced (immutable update)."""
        merged = {**asdict(self), **kwargs}
        return Config(**merged)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = Config()
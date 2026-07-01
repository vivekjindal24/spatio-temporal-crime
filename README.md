# HASI-Net — Heterogeneous Adaptive Spatio-temporal Informer Network

Forecasting **women-centric crimes** on **aggregated district data** for the
**Madhya Pradesh** region, with a Chicago fine-grained benchmark.

This repository is the reproducible companion to two research papers
(`paper_a/`, `paper_b/`). It implements a novel hybrid spatiotemporal deep
model adapted to *coarse, district-level* crime counts — the regime in which
Indian NCRB data is published, and which existing Informer/ST-GCN models
(designed for precise lat/long incident data) do not address well.

## The model

**HASI-Net** fuses three ideas, each motivated by a limitation of prior work
for aggregated data:

1. **Heterogeneous adaptive graph.** Three adjacency channels are combined with
   learnable softmax weights:
   - `A_geo` — geographic rook contiguity (fixed prior, from district boundaries)
   - `A_socio` — kNN cosine similarity on Census 2011 socioeconomic features
   - `A_adaptive` — a learnable latent adjacency `softmax(ReLU(E Eᵀ))`
   The model can down-weight a noisy prior when the latent graph is more
   informative — the key adaptation for coarse data.

2. **Multi-scale temporal block.** A ProbSparse (Informer) encoder for
   long-horizon dependencies + a dilated temporal convolutional branch for
   local dynamics, applied after trend/seasonal decomposition. A learned gate
   fuses them. This is why the model works on the short (~14-step) yearly MP
   series where plain Informer has little advantage.

3. **Count-aware head + loss.** Three heads emit `log μ`, `log α` and a
   zero-inflation logit, trained with a **zero-inflated negative-binomial**
   likelihood and focal reweighting — appropriate for non-negative,
   right-skewed, zero-inflated, imbalanced crime counts.

Hyperparameters are tuned by **adaptive-inertia multi-swarm PSO**.

## Repository layout

```
src/hasi_net/        # the model package (data, graph, model, losses, pso, train, baselines, viz, experiment)
notebooks/           # hasi_net_m1.ipynb (Apple Silicon) and hasi_net_colab.ipynb (Colab GPU)
tests/               # smoke_test.py (synthetic) + run_real_mp.py (real-data figure run)
data/                # downloaded + cached datasets (gitignored)
results/             # figures, metrics CSV, model weights, summary JSON
paper_a/ paper_b/    # the two research papers (LaTeX + figures)
```

## Datasets (all public, downloaded automatically)

| Dataset | Source | Use |
|---|---|---|
| NCRB crimes-against-women, district-level, 2001–2014 | [Sidd7893/crime-analysis](https://github.com/Sidd7893/crime-analysis) (originally NCRB *Crime in India*) | MP target panel: 7 IPC categories × 62 districts × 14 years |
| Census 2011 district socioeconomic features | [India-Census-2011-Analysis](https://github.com/nishusharma1608/India-Census-2011-Analysis) | node features (literacy, sex ratio, workforce, SC/ST) |
| India district boundaries | [guneetnarula/indian-district-boundaries](https://github.com/guneetnarula/indian-district-boundaries) (MIT) | `A_geo` rook contiguity + choropleth |
| Chicago crimes 2001–present | [City of Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/explore/) | fine-grained benchmark (community-area × month) |

**Data honesty note.** The 2015–2016 NCRB reports and the 2017–2022 India Data
Portal release are not reliably available as clean machine-readable CSVs, so
the MP panel uses the contiguous, fully-real **2001–2014** block rather than
forward-filling pseudo-years. This is disclosed in the paper.

## Running

### Apple MacBook M1
```bash
pip install -r requirements.txt
jupyter lab notebooks/hasi_net_m1.ipynb
```
The notebook uses the **MPS** backend and falls back to CPU if unavailable.

### Google Colab
Open `notebooks/hasi_net_colab.ipynb` in Colab with a **GPU** runtime. It
mounts Drive for persistence and installs deps in the first cell.

### Quick verification (no downloads, ~1 min)
```bash
python tests/smoke_test.py      # synthetic end-to-end run
```

## Reproducibility
All randomness is seeded (`Config.seed`). Real data downloads are cached under
`data/`. Every run writes `results/summary_<tag>.json` capturing the exact
config, metrics, panel provenance and learned graph-channel weights, plus
`results/metrics_<tag>.csv` and the figures cited in the papers.

## Citation
See `paper_a/` and `paper_b/` for the manuscript LaTeX and bibliography.
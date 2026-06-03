# doppa-analytics

doppa-analytics is the **analysis and visualization stage** of the doppa benchmarking system. It reads the raw
benchmark results that [doppa](https://github.com/kartAI/doppa) writes to Azure Blob Storage, runs the
statistical tests behind the thesis's research questions, and generates the figures and LaTeX tables used in
the master's thesis (TBA4925, NTNU Geomatics) comparing cloud-native versus traditional geospatial
technologies on Azure.

The benchmarking framework deliberately persists every iteration's *raw* sample rather than point averages, so
all uncertainty quantification — bootstrapped confidence intervals, Wilcoxon rank-sum tests, Vargha–Delaney
Â12 effect sizes — is computed here, downstream, without re-running any benchmark.

## The doppa ecosystem

The thesis system spans three repositories that depend on each other in a pipeline:

```
doppa-data-contribution  ──►  doppa  ──►  doppa-analytics
   publish datasets          benchmarks     (this repo)
   to blob storage           run engines    read results, build
                             write metrics   thesis figures/tables
```

| Repository | Role | Link |
|------------|------|------|
| **doppa-data-contribution** | Downloads OSM, FKB, Microsoft Buildings, and municipality boundaries, normalizes them to a shared GeoParquet schema, and publishes them to blob storage | [kartAI/doppa-data-contribution](https://github.com/kartAI/doppa-data-contribution) |
| **doppa** | Reproducible benchmarking framework. Runs the spatial-query benchmarks (DuckDB, PostGIS, Shapefile, Sedona) on Azure and writes per-iteration samples and cost rows back to blob storage | [kartAI/doppa](https://github.com/kartAI/doppa) |
| **doppa-analytics** | Reads the benchmark result Parquet, runs the statistical tests, and generates the thesis figures and LaTeX tables *(this repo)* | — |

The figures and tables produced here are copied into the LaTeX thesis repository (`tba4925-masters-thesis`),
which is the final consumer of this repo's output.

## Research questions

The analysis is organized around three research questions:

- **RQ1 — Single-node.** Cloud-native (DuckDB + GeoParquet) versus traditional (PostGIS, GeoPandas +
  Shapefile) performance on point-in-polygon, kNN, and bbox queries.
- **RQ2 — Distributed scaling.** Apache Sedona on Databricks + GeoParquet versus single-node engines on the
  national-scale spatial join, across cluster sizes and join strategies.
- **RQ3 — Consistency.** Whether performance rankings hold across spatial query patterns and dataset sizes.

Four benchmark configurations are compared throughout: DuckDB + GeoParquet, PostGIS, GeoPandas + Shapefile
(local), and Sedona on Databricks.

## Table of contents

- [Repository layout](#repository-layout)
- [Notebooks](#notebooks)
- [Inputs and outputs](#inputs-and-outputs)
- [Style authority](#style-authority)
- [Setup](#setup)
- [Regenerating figures and tables](#regenerating-figures-and-tables)

## Repository layout

```
src/
  analysis/        Data loading, validation, statistical tests, table builders
                   (thesis_compute.py drives the thesis figures + tables)
  plotting/
    style.py            Single source of truth for colors, palette, fonts, rcParams
    thesis_figures.py   Thesis result figures (the PNGs used in the thesis)
    charts.py           Older exploratory per-cell charts (colors derived from style.py)
  persistence/     DuckDB query helpers for reading the benchmark Parquet
notebooks/         Analysis notebooks (run analysis + plotting)
figures/           Generated chart PNGs (output of notebook runs)
tables/            Generated LaTeX table fragments
```

## Notebooks

Run top to bottom; each loads the benchmark results, runs its analysis, and writes figures/tables.

| Notebook | Purpose |
|----------|---------|
| `00-figure-style-gallery` | Palette / font reference gallery |
| `01-measurement-quality` | Warm-up, convergence, coverage, dispersion of the raw samples |
| `02-rq1-single-node` | RQ1 single-node query performance |
| `03-rq2-distributed` | RQ2 distributed join and scaling |
| `04-rq3-synthesis` | RQ3 synthesis (consistency of winners) |
| `05-appendix-cell-grid` | Appendix per-cell grid |

## Inputs and outputs

**Input** — the `benchmarks` (and metadata) blob containers written by [doppa](https://github.com/kartAI/doppa):
per-iteration samples and per-benchmark cost rows, hive-partitioned GeoParquet. The DuckDB helpers in
`src/persistence/` read these distributions directly.

**Output** — 300 dpi PNG figures in `figures/` and LaTeX table fragments in `tables/`, which are then copied
into the thesis repository's `Chapters/<n>-<slug>/sub-chapters/figures/` directories.

## Style authority

All matplotlib styling lives **only** in `src/plotting/style.py` (with a static mirror in
`src/plotting/thesis.mplstyle`). Colors are never hardcoded as hex literals in `charts.py` or `.tex` files —
any color or font change goes through `style.py`, after which the notebooks are re-run to regenerate figures.
`style.py` also defines the conceptual-diagram language (roles, node styles, ribbons) so generated diagrams
match the hand-drawn Lucidchart figures. See `CLAUDE.md` for the full palette, semantic mappings, and font
stack.

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/kartAI/doppa-analytics.git
cd doppa-analytics

python -m venv .venv                         # Create virtual environment
source .venv/bin/activate                    # Activate venv (Linux/macOS)
# .\.venv\Scripts\Activate.ps1               # Activate venv (Windows PowerShell)
pip install -r requirements.txt              # Install dependencies (matplotlib 3.10.x, scipy, pandas, ...)
```

Reading benchmark results from blob storage requires credentials for the same storage account doppa writes to;
provide them via a `.env` file in the project root.

## Regenerating figures and tables

1. Activate the venv: `source .venv/bin/activate`
2. Run the `notebooks/01`–`05` notebooks (Jupyter, or `jupyter execute <notebook>.ipynb`).
3. Charts are written to `figures/`; LaTeX fragments to `tables/`.
4. Copy the relevant PNGs and fragments into the thesis figure/table directories.

After any color or font edit in `src/plotting/style.py`, re-run the notebooks so the regenerated figures pick
up the change.

# CLAUDE.md

## Project overview

This repository provides the analysis and visualization pipeline for a master's thesis
(TBA4925, NTNU Geomatics) benchmarking cloud-native versus traditional geospatial
technologies on Microsoft Azure. The thesis addresses three research questions: (RQ1)
single-node cloud-native (DuckDB + GeoParquet) versus traditional (PostGIS, GeoPandas +
Shapefile) performance; (RQ2) distributed scaling (Apache Sedona on Databricks +
GeoParquet) versus single-node engines; (RQ3) consistency of performance rankings across
spatial query patterns and dataset sizes. Four benchmark configurations are compared:
DuckDB + GeoParquet, PostGIS, GeoPandas + Shapefile (local), and Sedona on Databricks.

## Repository layout

```
src/
  analysis/        Data loading, validation, statistical tests, table builders
  plotting/
    style.py       Single source of truth for colors, palette, fonts, rcParams
    charts.py      All matplotlib chart functions (colors derived from style.py)
  persistence/     DuckDB query helpers
figures/           Generated chart PNGs (output of notebook runs)
tables/            Generated LaTeX table fragments
rq1-analysis.ipynb RQ1 notebook (runs analysis + plotting)
rq2-analysis.ipynb RQ2 notebook (runs analysis + plotting)
requirements.txt   Python dependencies (matplotlib 3.10.x, scipy, pandas, etc.)
```

The LaTeX thesis lives in a sibling repository (`tba4925-masters-thesis/`):
- `Chapters/<n>-<slug>/sub-chapters/figures/` — figure PNG destinations
- `packages.sty` — thesis palette definitions and listings style
- `main.tex`, `bibliography.bib`

## Style authority

- Matplotlib styling lives **only** in `src/plotting/style.py`.
- LaTeX colors live **only** in the thesis palette block in `packages.sty`.
- Colors are **never** hardcoded as hex literals in `charts.py` or `.tex` files.

Any color or font change goes through `style.py`. After edits, re-run the notebooks to
regenerate figures.

## Color palette

| Name           | Hex       | Role                                          |
|----------------|-----------|-----------------------------------------------|
| `thesisteal`   | `#4F8985` | Cloud-native primary (DuckDB + GeoParquet)    |
| `thesiscoral`  | `#B7704D` | Traditional primary (PostGIS)                 |
| `thesissteel`  | `#5B8AAC` | Distributed / Sedona / Databricks broadcast   |
| `thesisamber`  | `#C9A36A` | File-database / Shapefile path (local)        |
| `thesisviolet` | `#8E7BA3` | Categorical extra                             |
| `thesissage`   | `#6E8F76` | Categorical extra / muted "good"              |
| `thesisslate`  | `#2C3E50` | Dark neutral (axis text, headings)            |
| `thesisgray`   | `#7F7F7F` | Mid neutral (secondary lines, default strat.) |
| `thesislight`  | `#E5E5E5` | Light neutral (gridlines, faint fills)        |
| `thesispale`   | `#F5F5F5` | Lightest neutral (backgrounds)                |
| `thesisbrick`  | `#A85A4A` | Muted failure / error semantic                |

Rules:
- Cool tones (teal, steel) = cloud-native. Warm tones (coral, amber) = traditional.
- Full-strength hex for lines, bars, and markers.
- 20-40% tints (`tint(color, 0.2)` to `tint(color, 0.4)`) for area/region fills.

## Semantic mappings

**Config colors** (`config_colors`):
- `duckdb` → `thesisteal`
- `postgis` → `thesiscoral`
- `local` → `thesisamber`
- `databricks-{strategy}-{n}-nodes` → lightness ramp per strategy:
  - `broadcast` → tints/shades of `thesissteel`
  - `partitioned` → tints/shades of `thesissage`
  - `default` → tints/shades of `thesisgray`
  - Lighter = fewer nodes, darker = more nodes (generated via `lightness_ramp`).

**Size colors** (`size_colors`): sequential ramp of `thesissteel` —
small (light tint) → medium (mid tint) → large (full).

**Strategy colors** (`strategy_colors`):
broadcast → `thesissteel`, partitioned → `thesissage`, default → `thesisgray`.

**Status colors** (`status_colors`):
success → `thesissage`, warning → `thesisamber`, failure → `thesisbrick`.

**Cost-category colors** (`cost_category_colors`):
compute → `thesissteel`, storage → `thesissage`, network → `thesisamber`,
operations → `thesisviolet`.

**Coverage heatmap**: colormap from `thesisbrick` (bad) → `thesisamber` (mid) → `thesissage`
(good); `set_bad` = `thesislight`.

**Annotation boxes** (missing/failed data): text `thesisbrick`, face light `thesisamber`
tint, edge `thesiscoral`.

## Font

All matplotlib figures use sans-serif. The resolved font stack in `apply_rcparams`:

```python
["Source Sans 3", "Inter", "Liberation Sans", "Arial", "DejaVu Sans"]
```

In the current environment only **DejaVu Sans** is installed, so matplotlib resolves to
`DejaVuSans.ttf`. This is a clean, neutral sans-serif that matches the Lucidchart
conceptual figures.

`mathtext.fontset` is set to `"dejavusans"` so math-mode labels (`$A_{12}$`,
`$\chi^2_F$`, `$p$`) render in matching sans-serif, not Computer Modern serif.

**Rejected alternative**: matching the LaTeX body font (Latin Modern / Computer Modern
serif). Rejected because (1) the conceptual figures already use sans-serif, (2) uniformity
across all figure types wins, and (3) serif tick labels are denser at small print sizes.

## Figure conventions

**Result charts**: generated by matplotlib in `src/plotting/charts.py`, saved as 300 dpi
PNG via `_savefig`, placed into `figures/` (analysis repo) or copied to
`Chapters/<n>-<slug>/sub-chapters/figures/` (thesis repo).

**Conceptual figures**: redrawn by hand in Lucidchart, exported as PNG. Captioned
`Adapted from \parencite{key}` when based on a published source.

**LaTeX placement rules**:
- `[H]` placement (never float).
- `\subfloat` from `subfig` package (not `subcaption`).
- `\includegraphics[valign=t, ...]` in multi-panel stacks.
- Two-argument `\caption[short]{long}`.
- `\label{fig:<slug>}` with slug: lowercase, hyphenated, max 4 words.
- Cross-reference: `Figure \ref{fig:<slug>}`.

## Writing and citation conventions

- Concise US academic English.
- `\parencite{}` / `\textcite{}`, never `\cite{}`.
- BibTeX key format: `lastnameYYYY_short_snake_case_title`.
- Wikipedia is not an acceptable source; prefer standards bodies and CNG guide subpages.
- Raster formats are out of scope (vector only in the thesis domain).

## Acronyms

Uses the `glossaries` package. `\acrfull` on first body use, `\acrshort` thereafter.
Definitions in `Chapters/00-preliminary/sub-chapters/03-abbreviations`. Tracked
cumulatively in document order.

## Regenerating figures and building

1. Activate the venv: `source .venv/bin/activate`
2. Run `rq1-analysis.ipynb` and `rq2-analysis.ipynb` (Jupyter or `jupyter execute`).
3. Charts are written to `figures/`.
4. Copy relevant PNGs to thesis figure directories.
5. Thesis build: `latexmk -pdf main.tex` (uses `biber` + biblatex `style=apa`,
   `sorting=nyt`).

## Do / Don't

**Do**:
- Edit `src/plotting/style.py` for any color or font change.
- Run plotting notebooks after style edits to regenerate figures.
- Keep US spelling throughout.
- Use `PALETTE` and `tint`/`shade` helpers for any derived color.

**Don't**:
- Hardcode hex color values in `charts.py` or `.tex` files.
- Reintroduce Material Design colors (`#2196F3`, `#4CAF50`, `#FF9800`, etc.).
- Change analysis logic, statistics, or data loading when only styling is requested.
- Cite Wikipedia.
- Add raster-format content to the thesis.
- Use `\cite{}`; always use `\parencite{}` or `\textcite{}`.

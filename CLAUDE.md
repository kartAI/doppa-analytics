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
                   (thesis_compute.py drives the thesis figures + tables)
  plotting/
    style.py            Single source of truth for colors, palette, fonts, rcParams
    thesis_figures.py   Thesis result figures (the 06-*/09-* PNGs used in the thesis)
    charts.py           Older exploratory per-cell charts (colors derived from style.py)
  persistence/     DuckDB query helpers
notebooks/         Analysis notebooks (run analysis + plotting):
  00-figure-style-gallery   Palette/font reference gallery
  01-measurement-quality    Warm-up, convergence, coverage, dispersion
  02-rq1-single-node        RQ1 single-node query performance
  03-rq2-distributed        RQ2 distributed join & scaling
  04-rq3-synthesis          RQ3 synthesis (consistency of winners)
  05-appendix-cell-grid     Appendix per-cell grid
figures/           Generated chart PNGs (output of notebook runs)
tables/            Generated LaTeX table fragments
requirements.txt   Python dependencies (matplotlib 3.10.x, scipy, pandas, etc.)
```

The LaTeX thesis lives in a sibling repository (`tba4925-masters-thesis/`):
- `Chapters/<n>-<slug>/sub-chapters/figures/` — figure PNG destinations
- `packages.sty` — thesis palette definitions and listings style
- `main.tex`, `bibliography.bib`

## Style authority

- Matplotlib styling lives **only** in `src/plotting/style.py` (with a static mirror in
  `src/plotting/thesis.mplstyle` for non-notebook `plt.style.use`).
- LaTeX colors live **only** in the thesis palette block in `packages.sty`.
- Colors are **never** hardcoded as hex literals in `charts.py` or `.tex` files.

Any color or font change goes through `style.py`. After edits, re-run the notebooks to
regenerate figures.

`style.py` also provides the **conceptual-diagram language** used by the hand-drawn
Lucidchart figures (Chapters 02/04/05), so generated diagram-style figures match:
`ROLE_COLORS` + `role_style(role)`/`fill_style(color)` (≈25%-strength tinted fill, same-hue
border, slate text → a `NodeStyle`), `rounded_node`, `ribbon` (Sankey band), `flow_arrow`,
`two_line_title` (bold slate + gray subtitle) and `bottom_legend` (rounded swatches). Roles:
`infrastructure` (gray) · `cloudnative` (teal) · `distributed` (steel) · `traditional`
(coral) · `decision` (amber) · `positive` (sage) · `negative` (brick).

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

All matplotlib figures use sans-serif. The resolved font stack (`SANS_STACK` in
`apply_rcparams`):

```python
["Source Sans 3", "Open Sans", "Inter", "Lato", "Liberation Sans", "Arial", "DejaVu Sans"]
```

The thesis **body** is Computer Modern (no font package in `packages.sty`); figures
deliberately use a humanist sans to match the Lucidchart conceptual diagrams, whose face is
**Lato**. `Source Sans 3` is the intended figure face. In the current environment it is not
installed, but **Open Sans** (a near-twin of Lato) **is**, so figures now resolve to
**Open Sans** — a markedly closer match to the conceptual diagrams than the former DejaVu
Sans fallback. `register_fonts()` also picks up any `src/plotting/fonts/*.ttf` so a fresh
machine can pin the intended face by dropping the TTF in. Check the live resolution with
`style.resolved_sans()`. A monospace face (`MONO_STACK`, resolves to **JetBrains Mono**) is
set for code identifiers in diagram-style figures.

`mathtext.fontset` is set to `"dejavusans"` so math-mode labels (`$A_{12}$`,
`$\chi^2_F$`, `$p$`) render in matching sans-serif, not Computer Modern serif.

**Rejected alternative**: matching the LaTeX body font (Latin Modern / Computer Modern
serif). Rejected because (1) the conceptual figures already use sans-serif, (2) uniformity
across all figure types wins, and (3) serif tick labels are denser at small print sizes.

## Figure conventions

**Result charts**: generated by matplotlib in `src/plotting/thesis_figures.py`, saved as
300 dpi PNG via `_save`, placed into `figures/` (analysis repo) or copied to
`Chapters/<n>-<slug>/sub-chapters/figures/` (thesis repo). (`charts.py` holds older
exploratory per-cell charts not used in the thesis.)

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
2. Run the `notebooks/01-05` notebooks (Jupyter or `jupyter execute`).
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

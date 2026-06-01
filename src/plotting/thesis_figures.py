"""Publication composite figures for the thesis results + appendix chapters.

These are distinct from the exploratory per-cell charts in ``charts.py``: each
function here renders one *thesis float* (a grid, multi-panel, or synthesis
figure) and writes it to an explicit output path at 300 dpi. All colors come
from :class:`~src.plotting.style.StyleConfig` / ``PALETTE`` (CLAUDE.md rule:
never hardcode hex). Computation reuses ``src.analysis.stats``.

Method conventions honored here:
  * elapsed_time     -> minimum estimator (one-sided contamination)
  * network bytes    -> median estimator
  * uncertainty      -> bootstrap CI (percentile); figure whiskers cap the
                        resample input at ``_CI_CAP`` for tractability while the
                        point estimate uses the full sample. Reported table
                        CIs use the full sample / harness values.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.stats import bootstrap as _scipy_bootstrap, gaussian_kde, spearmanr

from src.analysis.loading import extract_strategy, extract_worker_count
from src.analysis.stats import classify_a12

from .style import (
    LW_BORDER,
    LW_CONNECTOR,
    LW_HAIRLINE,
    LW_MARKER_EDGE,
    LW_SERIES,
    PALETTE,
    StyleConfig,
    lightness_ramp,
    shade,
    tint,
)


def _floor_nonneg(ax, axis: str = "y") -> None:
    """Clamp a *linear* position axis so it never dips below 0.

    For position-based plots (scatter / line / box / ECDF) on a non-negative
    metric: keep matplotlib's data-hugging upper bound but never let the visible
    range extend below zero. A no-op on log axes (which cannot include 0) and
    when the data already sits above 0. Per the figure-convention rule, this does
    NOT *force* a 0 baseline — that is reserved for bar charts via ``bottom=0``.
    """
    get_scale = ax.get_yscale if axis == "y" else ax.get_xscale
    if get_scale() == "log":
        return
    lo, hi = (ax.get_ylim() if axis == "y" else ax.get_xlim())
    if lo < 0:
        if axis == "y":
            ax.set_ylim(bottom=0.0)
        else:
            ax.set_xlim(left=0.0)

# ── shared helpers ─────────────────────────────────────────────────────────

_RNG = np.random.default_rng(12345)
_CI_CAP = 2000  # subsample cap for bootstrap whiskers (point estimate uses all)

_COST_TERMS = ["compute_cost", "storage_cost", "operations_cost", "network_cost"]
_COST_LABELS = {
    "compute_cost": "Compute",
    "storage_cost": "Storage",
    "operations_cost": "Operations",
    "network_cost": "Egress",
}


def _save(fig: plt.Figure, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    # Also emit a vector PDF for print (the thesis is printed as a book; vector
    # stays sharp at any resolution). Same geometry as the PNG; matplotlib's PDF
    # backend is vector by default, so axes/text/lines stay vector. dpi only
    # affects any embedded raster layers (none here). Font embedding is TrueType
    # (pdf.fonttype=42) via StyleConfig.apply_rcparams.
    fig.savefig(
        out_path.with_suffix(".pdf"),
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.show()
    plt.close(fig)
    return out_path


def _save_summary(df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    csv = out_path.with_name(out_path.stem + "-summary.csv")
    df.to_csv(csv, index=False)
    return csv


def estimate_ci(
    values, kind: str = "median", n_resamples: int = 10_000, cap: int = _CI_CAP
) -> tuple[float, float, float]:
    """Return ``(point, ci_low, ci_high)``.

    ``point`` is the exact estimator on the full sample (``min`` for time,
    ``median`` for bytes/other). The bootstrap CI is computed on a capped
    subsample with the percentile method (robust for the boundary ``min``
    statistic and bounded in memory).
    """
    v = np.asarray(list(values), dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    stat = np.min if kind == "min" else np.median
    point = float(stat(v))
    if len(v) < 2 or np.ptp(v) == 0:
        return point, point, point
    s = v if len(v) <= cap else _RNG.choice(v, cap, replace=False)
    res = _scipy_bootstrap(
        (s,), stat, n_resamples=n_resamples, confidence_level=0.95, method="percentile"
    )
    return point, float(res.confidence_interval.low), float(res.confidence_interval.high)


def _estimator_for(metric: str) -> str:
    return "min" if metric == "elapsed_time" else "median"


def system_of(cfg: str) -> str:
    """Collapse a configuration string to its system family."""
    if cfg.startswith("databricks"):
        return "sedona"
    return cfg


SYSTEM_LABEL = {
    "duckdb": "DuckDB + GeoParquet",
    "postgis": "PostGIS",
    "local": "GeoPandas + Shapefile",
    "sedona": "Apache Sedona",
}
SYSTEM_ORDER = ["duckdb", "postgis", "local", "sedona"]


def _system_color(style: StyleConfig, system: str) -> str:
    if system == "sedona":
        return style.strategy_colors["broadcast"]  # thesissteel
    return style.color(system)


def _name_color(style: StyleConfig, name: str) -> str:
    """Resolve a color for either a system family or a raw config string."""
    if name in SYSTEM_LABEL:
        return _system_color(style, name)
    return style.color(name)


def _name_label(style: StyleConfig, name: str) -> str:
    if name in SYSTEM_LABEL:
        return SYSTEM_LABEL[name]
    return style.label(name)


def _bytes_fmt() -> FuncFormatter:
    def f(x, _):
        ax = abs(x)
        if ax >= 1e9:
            return f"{x / 1e9:.0f}G"
        if ax >= 1e6:
            return f"{x / 1e6:.0f}M"
        if ax >= 1e3:
            return f"{x / 1e3:.0f}k"
        return f"{x:.0f}"

    return FuncFormatter(f)


def _tier_order(style: StyleConfig, tiers) -> list[str]:
    return sorted(set(tiers), key=lambda t: style.size_order.get(t, 99))


# ════════════════════════════════════════════════════════════════════════════
# 01 — measurement quality
# ════════════════════════════════════════════════════════════════════════════


def fig_warmup_distribution(
    successful: pd.DataFrame, query_id: str, style: StyleConfig, out_path
) -> Path:
    """Per-iteration elapsed-time distribution for one representative query.

    Shows the one-sided (right) contamination tail that motivates the minimum
    estimator. The minimum is marked; the long right tail is the contamination.
    """
    d = successful[successful["query_id"] == query_id]
    vals = d["elapsed_time"].dropna().values
    cfg = d["configuration"].iloc[0]
    color = style.color(cfg)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    lo, hi = np.percentile(vals, [0.1, 99.5])
    ax.hist(
        vals[(vals >= lo) & (vals <= hi)],
        bins=60,
        color=tint(color, 0.25),
        edgecolor=shade(color, 0.2),
        linewidth=LW_HAIRLINE,
    )
    vmin, vmean, vmed = float(np.min(vals)), float(np.mean(vals)), float(np.median(vals))
    for x, lab, c, ls in [
        (vmin, f"min = {vmin:.3g}\\,s", PALETTE["thesisslate"], "-"),
        (vmed, f"median = {vmed:.3g}", PALETTE["thesissage"], "--"),
        (vmean, f"mean = {vmean:.3g}", PALETTE["thesisbrick"], ":"),
    ]:
        ax.axvline(x, color=c, linestyle=ls, linewidth=LW_CONNECTOR,
                   label=lab.replace("\\,", " "))
    ax.set_xlabel("Per-iteration elapsed time (s)")
    ax.set_ylabel("Count")
    _floor_nonneg(ax, "x")  # elapsed-time axis is non-negative
    ax.set_title(f"{style.workload_label(d['workload_type'].iloc[0])} — {style.label(cfg)}")
    ax.legend(fontsize=8)
    _save_summary(
        pd.DataFrame([{"query_id": query_id, "n": len(vals), "min": vmin,
                       "median": vmed, "mean": vmean,
                       "p95": float(np.percentile(vals, 95)),
                       "max": float(np.max(vals))}]),
        out_path,
    )
    return _save(fig, out_path)


def fig_warmup_decay(
    successful: pd.DataFrame, query_id: str, style: StyleConfig, out_path,
    pass_run: int | None = None, head: int = 60
) -> Path:
    """Warmup-to-steady-state decay: elapsed vs iteration index within a pass."""
    d = successful[successful["query_id"] == query_id].copy()
    if pass_run is None:
        pass_run = int(d["benchmark_run"].min())
    d = d[d["benchmark_run"] == pass_run].sort_values("local_iteration")
    cfg = d["configuration"].iloc[0]
    color = style.color(cfg)
    y = d["elapsed_time"].values[:head]
    x = np.arange(1, len(y) + 1)
    steady = float(np.min(d["elapsed_time"].values))

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(x, y, marker="o", markersize=3, linewidth=0.9, color=color,
            markeredgewidth=LW_MARKER_EDGE, label="Per-iteration time")
    ax.axhline(steady, color=PALETTE["thesisslate"], linestyle="--", linewidth=LW_CONNECTOR,
               label=f"steady-state min = {steady:.3g} s")
    ax.set_xlabel(f"Iteration index within pass {pass_run}")
    ax.set_ylabel("Elapsed time (s)")
    ax.set_title(f"{style.workload_label(d['workload_type'].iloc[0])} — {style.label(cfg)}")
    _floor_nonneg(ax)  # elapsed time is non-negative; hug data above 0
    ax.legend(fontsize=8)
    _save_summary(
        pd.DataFrame({"iteration": x, "elapsed_time": y}), out_path
    )
    return _save(fig, out_path)


def fig_convergence(
    successful: pd.DataFrame, query_id: str, style: StyleConfig, out_path,
    pass_run: int | None = None, target: float = 0.05
) -> Path:
    """Running bootstrap CI half-width (relative) vs iteration, with 5% target.

    The estimator is the minimum (elapsed-time convention); the half-width is
    of a percentile bootstrap CI recomputed on the first k iterations.
    """
    d = successful[successful["query_id"] == query_id].copy()
    if pass_run is None:
        pass_run = int(d["benchmark_run"].min())
    d = d[d["benchmark_run"] == pass_run].sort_values("local_iteration")
    vals = d["elapsed_time"].dropna().values
    cfg = d["configuration"].iloc[0]
    color = style.color(cfg)

    ks, rel = [], []
    n = len(vals)
    step = 1 if n <= 200 else max(1, n // 200)
    for k in range(5, n + 1, step):
        _, lo, hi = estimate_ci(vals[:k], kind="min", n_resamples=2000, cap=1500)
        point = float(np.min(vals[:k]))
        ks.append(k)
        rel.append((hi - lo) / 2 / point if point > 0 else np.nan)

    stop_k = next((k for k, r in zip(ks, rel) if r <= target), None)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(ks, rel, color=color, linewidth=LW_SERIES)
    ax.axhline(target, color=PALETTE["thesisbrick"], linestyle="--", linewidth=LW_CONNECTOR,
               label=f"{target:.0%} target")
    if stop_k is not None:
        ax.axvline(stop_k, color=PALETTE["thesisslate"], linestyle=":", linewidth=LW_CONNECTOR,
                   label=f"stop @ n = {stop_k}")
        ax.scatter([stop_k], [target], color=PALETTE["thesisslate"], zorder=5, s=25)
    ax.set_xlabel("Iterations accumulated")
    ax.set_ylabel("Relative CI half-width")
    ax.set_ylim(0, min(0.5, max(rel) * 1.1 if rel else 0.5))
    ax.set_title(f"Sequential-stopping convergence — {style.label(cfg)}")
    ax.legend(fontsize=8)
    _save_summary(pd.DataFrame({"n": ks, "rel_ci_halfwidth": rel}), out_path)
    return _save(fig, out_path)


def fig_estimator_illustration(
    successful: pd.DataFrame, query_id: str, style: StyleConfig, out_path
) -> Path:
    """Minimum-vs-mean estimator illustration on one representative query.

    An empirical CDF of the per-iteration elapsed time on a log axis, with the
    minimum, median and mean marked and the minimum-to-mean span shaded as the
    one-sided contamination. The ECDF was chosen over the earlier jitter strip
    (which read as a dense blob) and over a second histogram (the warm-up float
    already shows one for the same cell): the curve climbs almost vertically off
    a hard left wall and then crawls across a long, shallow right tail, so the
    skew and the gap between the minimum and the mean are both directly legible,
    and it reuses the ECDF idiom already used for the latency figure.
    """
    d = successful[successful["query_id"] == query_id]
    vals = d["elapsed_time"].dropna().values
    vals = vals[vals > 0]
    cfg = d["configuration"].iloc[0]
    color = style.color(cfg)

    vmin, vmean, vmed = float(np.min(vals)), float(np.mean(vals)), float(np.median(vals))
    vs = np.sort(vals)
    y = np.arange(1, len(vs) + 1) / len(vs)
    contamination = (vmean - vmin) / vmin if vmin > 0 else np.nan
    frac_above_min = float(np.mean(vals > vmin * 1.01))  # share past the floor
    frac_below_mean = float(np.mean(vals <= vmean))      # ECDF height at the mean

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    # shade the minimum-to-mean span: this is the one-sided contamination
    ax.axvspan(vmin, vmean, color=tint(PALETTE["thesisbrick"], 0.85), zorder=0)
    ax.step(vs, y, where="post", color=color, linewidth=LW_SERIES, zorder=3,
            label="empirical CDF")
    for x, lab, c, ls in [
        (vmin, f"minimum = {vmin:.4g} s", PALETTE["thesisslate"], "-"),
        (vmed, f"median = {vmed:.4g} s", PALETTE["thesissage"], "--"),
        (vmean, f"mean = {vmean:.4g} s", PALETTE["thesisbrick"], ":"),
    ]:
        ax.axvline(x, color=c, linestyle=ls, linewidth=LW_CONNECTOR, label=lab)
    # numbers on the figure so the contamination gap is readable at a glance
    ax.annotate(
        f"mean is {100 * contamination:.0f}% above the minimum",
        xy=(np.sqrt(vmin * vmean), 0.5), ha="center", va="center", fontsize=8,
        fontstyle="italic", color=shade(PALETTE["thesisbrick"], 0.2),
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Per-iteration elapsed time (s, log)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(
        f"Estimator choice — {style.workload_label(d['workload_type'].iloc[0])}, "
        f"{style.label(cfg)}"
    )
    ax.legend(fontsize=8, loc="lower right")
    _save_summary(
        pd.DataFrame([{"query_id": query_id, "n": int(len(vals)), "min": vmin,
                       "median": vmed, "mean": vmean, "mean_over_min": vmean / vmin,
                       "contamination_pct": 100 * contamination,
                       "frac_above_min": frac_above_min,
                       "frac_below_mean": frac_below_mean,
                       "p95": float(np.percentile(vals, 95)),
                       "max": float(np.max(vals))}]),
        out_path,
    )
    return _save(fig, out_path)


def fig_cv_dispersion(
    successful: pd.DataFrame, style: StyleConfig, out_path,
    metric: str = "elapsed_time", band: float = 0.10
) -> Path:
    """Coefficient of variation per (config) across workload x tier cells.

    One marker per cell, grouped by configuration, with a 0.10 reference band.
    """
    rows = []
    for (wt, ds, cfg), g in successful.groupby(["workload_type", "dataset_size", "configuration"]):
        v = g[metric].dropna().values
        if len(v) >= 2 and np.mean(v) > 0:
            rows.append({"workload_type": wt, "dataset_size": ds, "configuration": cfg,
                         "cv": float(np.std(v, ddof=1) / np.mean(v))})
    cv = pd.DataFrame(rows)
    configs = [c for c in sorted(cv["configuration"].unique(),
                                 key=lambda c: (system_of(c), c))]

    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(configs) + 2), 4.2))
    for i, cfg in enumerate(configs):
        sub = cv[cv["configuration"] == cfg]
        xs = _RNG.uniform(-0.18, 0.18, len(sub)) + i
        ax.scatter(xs, sub["cv"], s=34, color=style.color(cfg),
                   edgecolor="white", linewidth=LW_MARKER_EDGE, zorder=3)
        ax.scatter([i], [sub["cv"].median()], marker="_", s=520,
                   color=shade(style.color(cfg), 0.25), zorder=4, linewidth=LW_SERIES)
    ax.axhspan(0, band, color=tint(PALETTE["thesissage"], 0.55), zorder=0)
    ax.axhline(band, color=PALETTE["thesisbrick"], linestyle="--", linewidth=LW_CONNECTOR,
               label=f"CV = {band:.2f}")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([style.label(c) for c in configs], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Coefficient of variation")
    ax.set_title(f"Run-to-run dispersion — {style.metric_label(metric)}")
    _floor_nonneg(ax)  # CV is non-negative
    ax.legend(fontsize=8)
    _save_summary(cv, out_path)
    return _save(fig, out_path)


def fig_time_of_day_stability(
    successful: pd.DataFrame, style: StyleConfig, out_path, *, scatter_cap: int = 4000
) -> Path:
    """Relative runtime versus start time-of-day — a noisy-neighbour check.

    The 30 passes started at different wall-clock times, so each cell's timed
    iterations are spread across the 24-hour clock. The x-axis is the recorded
    start time-of-day (UTC, 0--24 h); the y-axis is each execution's elapsed time
    divided by the median elapsed time of its own cell (``query_id``), so cells
    spanning orders of magnitude are comparable on one axis centred on 1.0 (the
    reference line). Points are coloured by system family and each system gets a
    linear trend line (statsmodels/LOWESS is unavailable in this environment, so
    an ordinary-least-squares line is used). A flat trend, and the rank-based
    Spearman correlation reported alongside, is the evidence that no time-of-day
    or noisy-neighbour effect contaminates the comparison; a slope is a finding.
    """
    d = successful[["started_at", "elapsed_time", "query_id", "configuration"]].copy()
    d = d.dropna(subset=["started_at", "elapsed_time"])
    d = d[d["elapsed_time"] > 0]
    ts = pd.to_datetime(d["started_at"], utc=True, errors="coerce")
    d = d[ts.notna()].copy()
    ts = ts[ts.notna()]
    d["hod"] = ts.dt.hour.values + ts.dt.minute.values / 60.0 + ts.dt.second.values / 3600.0
    d["cell_med"] = d.groupby("query_id")["elapsed_time"].transform("median")
    d = d[d["cell_med"] > 0]
    d["rel"] = d["elapsed_time"] / d["cell_med"]
    d["system"] = d["configuration"].map(system_of)

    systems = [s for s in SYSTEM_ORDER if s in set(d["system"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs_line = np.linspace(0, 24, 50)
    summary = []
    rho_all, p_all = spearmanr(d["hod"].values, d["rel"].values)
    for sysname in systems:
        g = d[d["system"] == sysname]
        color = _system_color(style, sysname)
        plot_g = g if len(g) <= scatter_cap else g.sample(scatter_cap, random_state=12345)
        ax.scatter(plot_g["hod"], plot_g["rel"], s=7, alpha=0.18, color=color,
                   edgecolor="none", zorder=2)
        rho, pval = spearmanr(g["hod"].values, g["rel"].values)
        slope, intercept = np.polyfit(g["hod"].values, g["rel"].values, 1)
        ax.plot(xs_line, slope * xs_line + intercept, color=shade(color, 0.15),
                linewidth=LW_SERIES, zorder=4,
                label=f"{SYSTEM_LABEL.get(sysname, sysname)} (ρ = {rho:+.2f})")
        summary.append({"system": sysname, "n": int(len(g)), "spearman_rho": float(rho),
                        "spearman_p": float(pval), "ols_slope_per_hour": float(slope)})
    ax.axhline(1.0, color=PALETTE["thesisslate"], linestyle="--", linewidth=LW_CONNECTOR,
               zorder=3, label="cell median (1.0)")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    # relative runtime is right-skewed; clip the view to the central mass so the
    # flat trend is legible (trend lines and Spearman use every point).
    ax.set_ylim(0, float(np.percentile(d["rel"].values, 98)))
    ax.set_xlabel("Start time-of-day (UTC, h)")
    ax.set_ylabel("Relative runtime (elapsed / cell median)")
    ax.set_title(f"Runtime versus time-of-day (overall Spearman ρ = {rho_all:+.2f})")
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    summary.append({"system": "ALL", "n": int(len(d)), "spearman_rho": float(rho_all),
                    "spearman_p": float(p_all), "ols_slope_per_hour": np.nan})
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


# ════════════════════════════════════════════════════════════════════════════
# 02 — RQ1 single-node
# ════════════════════════════════════════════════════════════════════════════


def _kde_violin(ax, center, values, color, *, log, width=0.34):
    """Draw a right-side half-violin from a KDE of *values*.

    The KDE is fit in log space when the axis is logarithmic (so the density
    is symmetric on screen and never extends below zero) and mapped back to
    data coordinates. Returns ``True`` if a violin was drawn, ``False`` if the
    data were too few or too degenerate for a stable KDE (caller falls back to
    the box + strip alone).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
    if len(v) < 5:
        return False
    base = np.log10(v) if log else v
    if np.ptp(base) <= 0:
        return False
    if len(base) > 4000:  # a few thousand points fix the shape; keep the KDE fast
        base = _RNG.choice(base, 4000, replace=False)
    try:
        kde = gaussian_kde(base)
    except Exception:
        return False
    grid = np.linspace(base.min(), base.max(), 200)
    dens = kde(grid)
    if not np.all(np.isfinite(dens)) or dens.max() <= 0:
        return False
    dens = dens / dens.max() * width
    ys = np.power(10.0, grid) if log else grid
    ax.fill_betweenx(ys, center, center + dens, facecolor=tint(color, 0.5),
                     edgecolor=color, linewidth=LW_BORDER, alpha=0.9, zorder=2)
    return True


def _raincloud_one(ax, center, values, color, *, kind, log, strip_cap=180):
    """One configuration's raincloud at ``x = center``.

    Half-violin (distribution shape) + a narrow box (quartiles, p5/p95 whiskers)
    + a jittered, subsampled strip of the raw samples, with the methodology
    estimator (``min`` for time, ``median`` for bytes) and its bootstrap CI
    overlaid in slate so the point estimate reads over the cloud. Returns a
    summary dict, or ``None`` when there are no usable samples.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    point, lo, hi = estimate_ci(v, kind=kind)

    _kde_violin(ax, center, v, color, log=log)

    # narrow box: Q1–Q3 with median bar and p5/p95 whiskers (positive values
    # only on a log axis). Drawn just left of the violin so the two don't overlap.
    vbox = v[v > 0] if log else v
    bx = center - 0.02
    if len(vbox) >= 2 and np.ptp(vbox) > 0:
        q1, med, q3 = np.percentile(vbox, [25, 50, 75])
        p5, p95 = np.percentile(vbox, [5, 95])
        bw = 0.11
        ax.plot([bx, bx], [p5, q1], color=shade(color, 0.2), linewidth=LW_BORDER, zorder=3)
        ax.plot([bx, bx], [q3, p95], color=shade(color, 0.2), linewidth=LW_BORDER, zorder=3)
        ax.add_patch(plt.Rectangle((bx - bw, q1), 2 * bw, q3 - q1, facecolor="white",
                                   edgecolor=shade(color, 0.2), linewidth=LW_BORDER, zorder=3))
        ax.plot([bx - bw, bx + bw], [med, med], color=shade(color, 0.35),
                linewidth=LW_CONNECTOR, zorder=4)

    # jittered raw strip, subsampled, to the left of the box
    s = v if len(v) <= strip_cap else _RNG.choice(v, strip_cap, replace=False)
    if log:
        s = s[s > 0]
    xs = center - 0.26 + _RNG.uniform(-0.06, 0.06, len(s))
    ax.scatter(xs, s, s=3, color=color, alpha=0.3, edgecolor="none", zorder=2)

    # estimator + bootstrap CI (diamond = min, circle = median)
    ax.errorbar(center, point, yerr=[[max(point - lo, 0)], [max(hi - point, 0)]],
                fmt="D" if kind == "min" else "o", ms=4.5,
                color=PALETTE["thesisslate"], ecolor=PALETTE["thesisslate"],
                elinewidth=LW_CONNECTOR, capsize=2.5, markeredgecolor="white",
                markeredgewidth=LW_MARKER_EDGE, zorder=6)
    return {"estimator": kind, "value": point, "ci_low": lo, "ci_high": hi,
            "n": int(len(v))}


def _rq1_dist_grid(successful, workloads, configs, tiers, style, *, metric, kind,
                   title, ylabel, out_path, byte_axis=False, annotate_local_net=False):
    """Workload x tier grid of per-config rainclouds (violin + box + strip).

    Replaces the earlier bar + whisker grid: the body now shows the full
    distribution shape per configuration, with the methodology estimator and its
    bootstrap CI marked on each, supporting the estimator justification. Absent
    cells (e.g. kNN-large, Shapefile beyond the small tier) are left as
    did-not-run-by-design; the local-filesystem path's ~0 network transfer is
    annotated rather than plotted on the log axis.
    """
    tiers = _tier_order(style, tiers)
    nrows, ncols = len(workloads), len(tiers)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols + 1, 3.0 * nrows + 0.8),
                             squeeze=False)
    summary = []
    for r, wt in enumerate(workloads):
        for c, ds in enumerate(tiers):
            ax = axes[r][c]
            cell = successful[(successful["workload_type"] == wt)
                              & (successful["dataset_size"] == ds)]
            present = [cf for cf in configs if cf in cell["configuration"].unique()]
            xpos, labels = [], []
            for i, cf in enumerate(configs):
                if cf not in present:
                    continue
                v = cell[cell["configuration"] == cf][metric].dropna().values
                if len(v) == 0:
                    continue
                point = float(np.min(v)) if kind == "min" else float(np.median(v))
                # the local path reads from disk: ~0 network, unplottable on a
                # log axis. Mark it so the cell is not misread as missing.
                if annotate_local_net and cf == "local" and point <= 0:
                    ax.annotate("local read\n(≈0 network)", xy=(i, 0.04),
                                xycoords=("data", "axes fraction"), ha="center",
                                va="bottom", fontsize=6.5, fontstyle="italic",
                                color=PALETTE["thesisbrick"])
                    xpos.append(i)
                    labels.append(style.label(cf))
                    summary.append({"workload_type": wt, "dataset_size": ds,
                                    "configuration": cf, "metric": metric,
                                    "estimator": kind, "value": 0.0, "ci_low": 0.0,
                                    "ci_high": 0.0, "n": int(len(v))})
                    continue
                res = _raincloud_one(ax, i, v, style.color(cf), kind=kind, log=True)
                if res is None:
                    continue
                xpos.append(i)
                labels.append(style.label(cf))
                summary.append({"workload_type": wt, "dataset_size": ds,
                                "configuration": cf, "metric": metric, **res})
            ax.set_yscale("log")
            if byte_axis:
                ax.yaxis.set_major_formatter(_bytes_fmt())
            ax.set_xlim(-0.6, len(configs) - 0.4)
            ax.set_xticks(xpos)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8.5)
            if r == 0:
                ax.set_title(ds.capitalize(), fontsize=10, fontweight="bold")
            if c == ncols - 1:
                ax.annotate(style.workload_label(wt), xy=(1.02, 0.5),
                            xycoords="axes fraction", rotation=270, va="center",
                            ha="left", fontsize=8.5, color=PALETTE["thesisslate"])
            if not present:
                ax.text(0.5, 0.5, "did not run\nby design", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, fontstyle="italic",
                        color=PALETTE["thesisbrick"])
                ax.set_xticks([])

    est_word = "minimum" if kind == "min" else "median"
    est_handle = plt.Line2D([], [], marker="D" if kind == "min" else "o", linestyle="",
                            color=PALETTE["thesisslate"], markeredgecolor="white",
                            markersize=6, label=f"{est_word} estimator (95% bootstrap CI)")
    cloud_handle = plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                              color=tint(PALETTE["thesisgray"], 0.4),
                              markeredgecolor=PALETTE["thesisgray"],
                              label="per-iteration distribution (violin · box · points)")
    fig.legend(handles=[est_handle, cloud_handle], loc="lower center", ncol=2,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


def fig_rq1_time_grid(successful, workloads, configs, tiers, style, out_path):
    """Wall-clock time distribution per config (workload x tier), min estimator + CI."""
    return _rq1_dist_grid(
        successful, workloads, configs, tiers, style,
        metric="elapsed_time", kind="min",
        title="Wall-clock time per single-machine configuration (minimum estimator)",
        ylabel="Elapsed time (s, log)", out_path=out_path,
    )


def fig_rq1_bytes_grid(successful, workloads, configs, tiers, style, out_path):
    """Network bytes received distribution per config (workload x tier), median + CI."""
    return _rq1_dist_grid(
        successful, workloads, configs, tiers, style,
        metric="network_bytes_received", kind="median",
        title="Network bytes received per single-machine configuration (median)",
        ylabel="Bytes received (log)", out_path=out_path, byte_axis=True,
        annotate_local_net=True,
    )


def fig_rq1_bytes_vs_time(successful, workloads, configs, tiers, style, out_path):
    """Bytes received vs wall-clock time scatter (one marker per cell)."""
    tiers = _tier_order(style, tiers)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    marker = {"small": "o", "medium": "s", "large": "^"}
    summary = []
    for wt in workloads:
        for ds in tiers:
            cell = successful[(successful["workload_type"] == wt)
                              & (successful["dataset_size"] == ds)]
            for cf in configs:
                v = cell[cell["configuration"] == cf]
                t = v["elapsed_time"].dropna().values
                b = v["network_bytes_received"].dropna().values
                if len(t) == 0 or len(b) == 0:
                    continue
                tt, bb = float(np.min(t)), float(np.median(b))
                ax.scatter(tt, max(bb, 1), s=60, color=style.color(cf),
                           marker=marker.get(ds, "o"), edgecolor="white",
                           linewidth=LW_MARKER_EDGE, zorder=3)
                summary.append({"workload_type": wt, "dataset_size": ds,
                                "configuration": cf, "min_time": tt, "median_bytes": bb})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(_bytes_fmt())
    ax.set_xlabel("Wall-clock time (s, log)")
    ax.set_ylabel("Bytes received (log)")
    cfg_handles = [plt.Line2D([], [], marker="o", linestyle="", color=style.color(c),
                              label=style.label(c)) for c in configs
                   if c in successful["configuration"].unique()]
    size_handles = [plt.Line2D([], [], marker=marker[s], linestyle="", color=PALETTE["thesisgray"],
                               label=s.capitalize()) for s in tiers]
    leg1 = ax.legend(handles=cfg_handles, fontsize=7.5, loc="upper left", title="Engine")
    ax.add_artist(leg1)
    ax.legend(handles=size_handles, fontsize=7.5, loc="lower right", title="Tier")
    ax.set_title("Bytes transferred versus wall-clock time")
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


def fig_rq1_operational_cost(cost_summary, workloads, configs, tiers, style, out_path):
    """Four-term stacked operational cost per config, faceted by workload pattern."""
    tiers = _tier_order(style, tiers)
    fig, axes = plt.subplots(1, len(workloads), figsize=(4.0 * len(workloads) + 0.5, 4.2),
                             squeeze=False)
    axes = axes[0]
    summary = []
    cs = cost_summary.reset_index()
    for ax, wt in zip(axes, workloads):
        sub = cs[cs["workload_type"] == wt]
        bars = []
        for cf in configs:
            for ds in tiers:
                row = sub[(sub["configuration"] == cf) & (sub["dataset_size"] == ds)]
                if len(row):
                    bars.append((cf, ds, row.iloc[0]))
        x = np.arange(len(bars))
        bottom = np.zeros(len(bars))
        for term in _COST_TERMS:
            vals = np.array([float(r[term]) for _, _, r in bars])
            ax.bar(x, vals, bottom=bottom, width=0.7,
                   color=style.cost_category_colors[term], edgecolor="white",
                   linewidth=LW_HAIRLINE, label=_COST_LABELS[term])
            bottom += vals
        for xi, (cf, ds, r) in enumerate(bars):
            summary.append({"workload_type": wt, "configuration": cf, "dataset_size": ds,
                            **{t: float(r[t]) for t in _COST_TERMS},
                            "total_cost": float(r["total_cost"])})
        ax.set_xticks(x)
        ax.set_xticklabels([f"{style.label(cf)}\n{ds[:3]}" for cf, ds, _ in bars],
                           rotation=35, ha="right", fontsize=7)
        ax.set_title(style.workload_label(wt), fontsize=9.5)
        ax.set_ylabel("Cost (USD)" if ax is axes[0] else "")
        ax.set_ylim(bottom=0)  # stacked bars baseline at 0
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Operational cost per single-machine configuration", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


def fig_cpu_decomposition(successful, workloads, configs, tiers, style, out_path):
    """User vs system CPU seconds (median) per config, workload x tier.

    A log CPU-time axis is required (PostGIS sits ~3 decades below DuckDB), and a
    bar has no valid baseline on a log scale, so this is drawn as a dumbbell: per
    cell and configuration a filled marker is the median *user* CPU time and an
    open marker the median *system* CPU time, joined by a thin stem. Lower is
    less CPU; the user/system split that the old stacked bar carried is preserved
    by the two markers without resting either segment on an arbitrary floor.
    """
    tiers = _tier_order(style, tiers)
    cells = [(wt, ds) for wt in workloads for ds in tiers]
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(cells) * len(configs)), 4.4))
    summary = []
    width = 0.8 / max(len(configs), 1)
    x = np.arange(len(cells))
    for j, cf in enumerate(configs):
        off = (j - len(configs) / 2 + 0.5) * width
        base = style.color(cf)
        for ci, (wt, ds) in enumerate(cells):
            v = successful[(successful["workload_type"] == wt)
                           & (successful["dataset_size"] == ds)
                           & (successful["configuration"] == cf)]
            u = v["cpu_time_user_seconds"].dropna().values
            s = v["cpu_time_system_seconds"].dropna().values
            if not (len(u) or len(s)):
                continue
            uu = float(np.median(u)) if len(u) else np.nan
            ss = float(np.median(s)) if len(s) else np.nan
            xp = ci + off
            pts = [p for p in (uu, ss) if p is not None and not np.isnan(p) and p > 0]
            if len(pts) == 2:  # stem joining the user/system pair
                ax.plot([xp, xp], [min(pts), max(pts)], color=shade(base, 0.1),
                        linewidth=LW_CONNECTOR, zorder=2)
            if uu and not np.isnan(uu) and uu > 0:
                ax.scatter([xp], [uu], s=34, color=base, marker="o", zorder=4,
                           edgecolor="white", linewidth=LW_MARKER_EDGE)
            if ss and not np.isnan(ss) and ss > 0:
                ax.scatter([xp], [ss], s=30, facecolor="white", marker="o", zorder=4,
                           edgecolor=base, linewidth=LW_BORDER)
            summary.append({"workload_type": wt, "dataset_size": ds, "configuration": cf,
                            "cpu_user_s": uu, "cpu_system_s": ss})
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{style.workload_label(wt)[:10]}\n{ds}" for wt, ds in cells],
                       rotation=0, fontsize=7)
    ax.set_ylabel("CPU time (s, log)")
    # legend: config colors + user/system marker convention
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=style.color(c),
                          markeredgecolor="white", markeredgewidth=LW_MARKER_EDGE)
               for c in configs]
    handles += [plt.Line2D([], [], marker="o", linestyle="", color=PALETTE["thesisgray"],
                           markeredgecolor="white"),
                plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                           markeredgecolor=PALETTE["thesisgray"], markeredgewidth=LW_BORDER)]
    labels = [style.label(c) for c in configs] + ["user (filled)", "system (open)"]
    ax.legend(handles, labels, fontsize=7, ncol=2)
    ax.set_title("CPU-time decomposition (user vs system, median)")
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


def fig_rq1_latency_ecdf(successful, workload, tier, configs, style, out_path):
    """Empirical CDF of per-iteration elapsed time for one headline workload x tier.

    Tail-detail companion to the time grid: one step ECDF per configuration on a
    log time axis, with the p50/p95/p99 quantiles marked on each curve so the
    spread between the median and the upper tail is directly readable across
    configurations.
    """
    cell = successful[(successful["workload_type"] == workload)
                      & (successful["dataset_size"] == tier)]
    levels = [0.50, 0.95, 0.99]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    summary = []
    for cf in configs:
        v = cell[cell["configuration"] == cf]["elapsed_time"].dropna().values
        v = v[v > 0]
        if len(v) == 0:
            continue
        vs = np.sort(v)
        y = np.arange(1, len(vs) + 1) / len(vs)
        color = style.color(cf)
        ax.step(vs, y, where="post", color=color, linewidth=LW_SERIES,
                label=style.label(cf), zorder=3)
        qs = np.percentile(v, [100 * lv for lv in levels])
        ax.scatter(qs, levels, color=shade(color, 0.2), s=24, zorder=5,
                   edgecolor="white", linewidth=LW_MARKER_EDGE)
        summary.append({"configuration": cf, "n": int(len(v)),
                        "p50": float(qs[0]), "p95": float(qs[1]), "p99": float(qs[2])})
    for lv in levels:
        ax.axhline(lv, color=PALETTE["thesislight"], linewidth=LW_BORDER, zorder=0)
        ax.annotate(f"p{int(lv * 100)}", xy=(0.0, lv), xycoords=("axes fraction", "data"),
                    xytext=(2, 1), textcoords="offset points", fontsize=7,
                    va="bottom", color=PALETTE["thesisgray"])
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Per-iteration elapsed time (s, log)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"Latency distribution — {style.workload_label(workload)} ({tier} tier)")
    ax.legend(fontsize=8, loc="lower right", title="Configuration")
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


# ════════════════════════════════════════════════════════════════════════════
# 03 — RQ2 distributed
# ════════════════════════════════════════════════════════════════════════════


def fig_speedup(scaling: pd.DataFrame, style: StyleConfig, out_path) -> Path:
    """Speedup S(n) = T2 / Tn per strategy and tier, with ideal references."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    summary = []
    for (strat, ds), g in scaling.groupby(["strategy", "dataset_size"]):
        g = g.sort_values("worker_count")
        base = g[g["worker_count"] == g["worker_count"].min()]["point"].iloc[0]
        s = base / g["point"].values
        color = style.strategy_colors.get(strat, PALETTE["thesisgray"])
        lw = {"small": 1.0, "medium": 1.5, "large": 2.1}.get(ds, 1.3)
        ax.plot(g["worker_count"], s, marker="o", color=color, linewidth=lw, markersize=5,
                label=f"{strat.capitalize()} ({ds})")
        for w, sv in zip(g["worker_count"], s):
            summary.append({"strategy": strat, "dataset_size": ds, "workers": int(w),
                            "speedup": float(sv)})
    w0 = scaling["worker_count"].min()
    wmax = scaling["worker_count"].max()
    ideal = np.array([w0, wmax])
    ax.plot(ideal, ideal / w0, linestyle="--", color=PALETTE["thesisslate"],
            linewidth=LW_CONNECTOR, label="ideal (linear)")
    ax.set_xlabel("Worker count")
    ax.set_ylabel("Speedup $S(n) = T_2 / T_n$")
    ax.set_title("Speedup of the distributed join")
    _floor_nonneg(ax)  # speedup is non-negative
    ax.legend(fontsize=7.5)
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


def fig_efficiency(scaling: pd.DataFrame, style: StyleConfig, out_path) -> Path:
    """Parallel efficiency E(n) = (2/n) S(n) per strategy and tier."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    summary = []
    for (strat, ds), g in scaling.groupby(["strategy", "dataset_size"]):
        g = g.sort_values("worker_count")
        w0 = g["worker_count"].min()
        base = g[g["worker_count"] == w0]["point"].iloc[0]
        s = base / g["point"].values
        e = (w0 / g["worker_count"].values) * s
        color = style.strategy_colors.get(strat, PALETTE["thesisgray"])
        lw = {"small": 1.0, "medium": 1.5, "large": 2.1}.get(ds, 1.3)
        ax.plot(g["worker_count"], e, marker="o", color=color, linewidth=lw, markersize=5,
                label=f"{strat.capitalize()} ({ds})")
        for w, ev in zip(g["worker_count"], e):
            summary.append({"strategy": strat, "dataset_size": ds, "workers": int(w),
                            "efficiency": float(ev)})
    ax.axhline(1.0, linestyle="--", color=PALETTE["thesisslate"], linewidth=LW_CONNECTOR,
               label="ideal (E = 1)")
    ax.set_xlabel("Worker count")
    ax.set_ylabel("Parallel efficiency $E(n)$")
    ax.set_title("Parallel efficiency of the distributed join")
    _floor_nonneg(ax)  # efficiency is non-negative
    ax.legend(fontsize=7.5)
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


def fig_wall_clock_vs_workers(scaling, single_node, failed, style, out_path) -> Path:
    """Wall-clock vs worker count per strategy/tier, single-node baselines, crossover."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    summary = []
    for (strat, ds), g in scaling.groupby(["strategy", "dataset_size"]):
        g = g.sort_values("worker_count")
        color = style.strategy_colors.get(strat, PALETTE["thesisgray"])
        lw = {"small": 1.0, "medium": 1.6, "large": 2.2}.get(ds, 1.3)
        ax.plot(g["worker_count"], g["point"], marker="o", color=color, linewidth=lw,
                markersize=5, label=f"{strat.capitalize()} ({ds})")
        for w, p in zip(g["worker_count"], g["point"]):
            summary.append({"kind": "distributed", "strategy": strat, "dataset_size": ds,
                            "workers": int(w), "min_time_s": float(p)})
    for _, row in single_node.iterrows():
        ax.axhline(row["point"], linestyle="--", linewidth=LW_CONNECTOR,
                   color=style.color(row["configuration"]),
                   label=f"{style.label(row['configuration'])} ({row['dataset_size']})")
        summary.append({"kind": "single_node", "configuration": row["configuration"],
                        "dataset_size": row["dataset_size"], "min_time_s": float(row["point"])})
    if failed is not None and len(failed):
        note = "Failed (executor OOM): " + ", ".join(
            sorted({f"{r.strategy} {r.dataset_size}" for r in failed.itertuples()}))
        ax.annotate(note, xy=(0.5, 0.015), xycoords="axes fraction", ha="center",
                    fontsize=7, fontstyle="italic", color=PALETTE["thesisbrick"])
    ax.set_yscale("log")
    ax.set_xlabel("Worker count")
    ax.set_ylabel("Wall-clock time (s, log; minimum estimator)")
    ax.set_title("Distributed wall-clock time versus single-node baselines")
    ax.legend(fontsize=7, ncol=2)
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


def fig_phase_time(phase_df: pd.DataFrame, style: StyleConfig, out_path) -> Path:
    """Stacked execution-phase wall-clock time vs workers (broadcast tiers)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    phases = [("executor_run_time_ms", "Executor run", style.strategy_colors["broadcast"]),
              ("driver_collection_time_ms", "Driver collection", PALETTE["thesisviolet"])]
    pdf = phase_df.sort_values(["dataset_size", "worker_count"])
    labels = [f"{int(w)}N\n{ds[:3]}" for w, ds in zip(pdf["worker_count"], pdf["dataset_size"])]
    x = np.arange(len(pdf))
    bottom = np.zeros(len(pdf))
    summary = []
    for col, lab, color in phases:
        vals = (pdf[col].values / 1000.0)
        ax.bar(x, vals, bottom=bottom, width=0.7, color=color, edgecolor="white",
               linewidth=LW_HAIRLINE, label=lab)
        bottom += vals
    for xi in range(len(pdf)):
        row = pdf.iloc[xi]
        summary.append({"dataset_size": row["dataset_size"], "workers": int(row["worker_count"]),
                        "executor_run_s": row["executor_run_time_ms"] / 1000.0,
                        "driver_collection_s": row["driver_collection_time_ms"] / 1000.0})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Phase time (s)")
    ax.set_ylim(bottom=0)  # stacked bars baseline at 0
    ax.set_title("Execution-phase wall-clock time (broadcast)")
    ax.legend(fontsize=8)
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


def fig_shuffle_bytes(phase_df: pd.DataFrame, style: StyleConfig, out_path) -> Path:
    """Shuffle read + write bytes vs workers (broadcast tiers)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    pdf = phase_df.sort_values(["dataset_size", "worker_count"])
    labels = [f"{int(w)}N\n{ds[:3]}" for w, ds in zip(pdf["worker_count"], pdf["dataset_size"])]
    x = np.arange(len(pdf))
    rd = pdf["shuffle_read_bytes"].values
    wr = pdf["shuffle_write_bytes"].values
    ax.bar(x, rd, width=0.7, color=style.strategy_colors["broadcast"], edgecolor="white",
           linewidth=LW_HAIRLINE, label="Shuffle read")
    ax.bar(x, wr, bottom=rd, width=0.7, color=tint(style.strategy_colors["broadcast"], 0.45),
           edgecolor="white", linewidth=LW_HAIRLINE, label="Shuffle write")
    ax.yaxis.set_major_formatter(_bytes_fmt())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Shuffle bytes")
    ax.set_ylim(bottom=0)  # stacked bars baseline at 0
    ax.set_title("Shuffle bytes (broadcast)")
    ax.legend(fontsize=8)
    _save_summary(
        pd.DataFrame({"dataset_size": pdf["dataset_size"], "workers": pdf["worker_count"],
                      "shuffle_read_bytes": rd, "shuffle_write_bytes": wr}),
        out_path,
    )
    return _save(fig, out_path)


def fig_cost_pareto(pareto_df: pd.DataFrame, style: StyleConfig, out_path) -> Path:
    """Cost-time Pareto: one point per (strategy, tier, worker count); frontier traced."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    summary = []
    for (strat, ds), g in pareto_df.groupby(["strategy", "dataset_size"]):
        color = style.strategy_colors.get(strat, PALETTE["thesisgray"])
        ax.scatter(g["cost"], g["time"], s=46, color=color, edgecolor="white",
                   linewidth=LW_MARKER_EDGE, zorder=3, label=f"{strat.capitalize()} ({ds})")
        for _, row in g.iterrows():
            ax.annotate(f"{int(row['worker_count'])}", xy=(row["cost"], row["time"]),
                        xytext=(3, 3), textcoords="offset points", fontsize=6)
    pts = pareto_df[["cost", "time"]].dropna().values
    order = pts[np.argsort(pts[:, 0])]
    frontier, best = [], np.inf
    for cst, tm in order:
        if tm < best:
            frontier.append((cst, tm))
            best = tm
    if frontier:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, color=PALETTE["thesisslate"], linewidth=LW_SERIES, linestyle="-",
                zorder=2, label="non-dominated frontier")
    for _, row in pareto_df.iterrows():
        summary.append({"strategy": row["strategy"], "dataset_size": row["dataset_size"],
                        "workers": int(row["worker_count"]), "cost_usd": float(row["cost"]),
                        "time_s": float(row["time"])})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cost per run (USD, log)")
    ax.set_ylabel("Wall-clock time (s, log)")
    ax.set_title("Cost-time tradeoff of the distributed join")
    ax.legend(fontsize=7)
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


# ════════════════════════════════════════════════════════════════════════════
# 04 — RQ3 synthesis
# ════════════════════════════════════════════════════════════════════════════


def fig_winners_matrix(winners: pd.DataFrame, style: StyleConfig, out_path) -> Path:
    """Winners matrix: rows = workload x tier, cols = outcome, cell = winning system."""
    outcomes = ["Time", "Bytes", "Cost"]
    rows = list(winners.index)
    fig, ax = plt.subplots(figsize=(5.6, 0.5 * len(rows) + 1.6))
    for ri, rk in enumerate(rows):
        for ci, oc in enumerate(outcomes):
            sysname = winners.loc[rk, oc]
            if pd.isna(sysname) or sysname == "":
                ax.add_patch(plt.Rectangle((ci, ri), 1, 1, facecolor=PALETTE["thesislight"]))
                continue
            ax.add_patch(plt.Rectangle((ci, ri), 1, 1,
                                       facecolor=tint(_system_color(style, sysname), 0.15),
                                       edgecolor="white", linewidth=LW_CONNECTOR))
            ax.text(ci + 0.5, ri + 0.5, SYSTEM_LABEL.get(sysname, sysname).split(" ")[0],
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color=shade(_system_color(style, sysname), 0.35))
    ax.set_xlim(0, len(outcomes))
    ax.set_ylim(0, len(rows))
    ax.set_xticks([i + 0.5 for i in range(len(outcomes))])
    ax.set_xticklabels(outcomes, fontweight="bold")
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels(rows, fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.invert_yaxis()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Consistency of winners across dimensions", pad=24)
    _save_summary(winners.reset_index().rename(columns={"index": "workload_tier"}), out_path)
    return _save(fig, out_path)


def fig_ranking_stability(rank_long: pd.DataFrame, style: StyleConfig, out_path) -> Path:
    """Slopegraph of rank vs tier per workload (bump chart). Inversions visible as crossings."""
    workloads = list(dict.fromkeys(rank_long["workload_type"]))
    fig, axes = plt.subplots(1, len(workloads), figsize=(3.3 * len(workloads) + 0.5, 4.2),
                             squeeze=False)
    axes = axes[0]
    for ax, wt in zip(axes, workloads):
        sub = rank_long[rank_long["workload_type"] == wt]
        tiers = _tier_order(style, sub["dataset_size"].unique())
        for cf, g in sub.groupby("configuration"):
            g = g.set_index("dataset_size").reindex(tiers)
            xs = [i for i, t in enumerate(tiers) if not pd.isna(g.loc[t, "rank"])]
            ys = [g.loc[t, "rank"] for t in tiers if not pd.isna(g.loc[t, "rank"])]
            col = _name_color(style, cf)
            ax.plot(xs, ys, marker="o", color=col, linewidth=LW_SERIES, markersize=7,
                    markeredgewidth=LW_MARKER_EDGE, label=_name_label(style, cf))
            if xs:
                ax.annotate(_name_label(style, cf).split(" ")[0], xy=(xs[-1], ys[-1]),
                            xytext=(6, 0), textcoords="offset points", va="center",
                            fontsize=7, color=shade(col, 0.2))
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels([t.capitalize() for t in tiers])
        ax.set_title(style.workload_label(wt), fontsize=9.5)
        ax.invert_yaxis()
        n_ranks = int(sub["rank"].max())
        ax.set_yticks(range(1, n_ranks + 1))
        if ax is axes[0]:
            ax.set_ylabel("Rank (1 = fastest)")
    fig.suptitle("Ranking stability across tiers", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_summary(rank_long, out_path)
    return _save(fig, out_path)


def fig_cliques(mean_ranks: dict, nonsig_pairs: list, style: StyleConfig, out_path,
                title="Pairwise non-significance cliques") -> Path:
    """Mean-rank axis with non-significance cliques (Holm-adjusted Wilcoxon).

    NOT a Nemenyi critical-difference diagram: connectors join configuration
    pairs whose difference is *not* significant after Holm adjustment.
    """
    items = sorted(mean_ranks.items(), key=lambda kv: kv[1])
    cfgs = [c for c, _ in items]
    ranks = [r for _, r in items]
    fig, ax = plt.subplots(figsize=(7.0, 2.4 + 0.35 * len(cfgs)))
    ax.hlines(0, min(ranks) - 0.3, max(ranks) + 0.3, color=PALETTE["thesisslate"],
              linewidth=LW_BORDER)
    for c, r in items:
        col = _name_color(style, c)
        ax.scatter(r, 0, s=60, color=col, zorder=4, edgecolor="white",
                   linewidth=LW_MARKER_EDGE)
        ax.annotate(f"{_name_label(style, c)}\n({r:.2f})", xy=(r, 0), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=shade(col, 0.2))
    level = -0.04
    drawn = set()
    for a, b in nonsig_pairs:
        if a in mean_ranks and b in mean_ranks and (a, b) not in drawn:
            ax.plot([mean_ranks[a], mean_ranks[b]], [level, level],
                    color=PALETTE["thesisbrick"], linewidth=1.8, zorder=3)
            level -= 0.035
            drawn.add((a, b))
    ax.set_ylim(level - 0.05, 0.18)
    ax.set_yticks([])
    ax.set_xlabel("Mean rank (lower = faster)")
    for s in ["top", "left", "right"]:
        ax.spines[s].set_visible(False)
    note = "connected = not significant (Holm-adj. Wilcoxon)" if nonsig_pairs \
        else "all pairs significant (Holm-adj. Wilcoxon)"
    ax.set_title(f"{title}\n{note}", fontsize=10)
    _save_summary(
        pd.DataFrame({"configuration": cfgs, "mean_rank": ranks}), out_path
    )
    return _save(fig, out_path)


def fig_a12_forest(forest_df: pd.DataFrame, style: StyleConfig, out_path) -> Path:
    """Vargha-Delaney A12 forest: one row per key comparison, with arcuri bands."""
    forest_df = forest_df.reset_index(drop=True)
    n = len(forest_df)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * n + 1.4))
    for i, row in forest_df.iterrows():
        y = n - i - 1
        a12 = row["a12"]
        color = PALETTE["thesisteal"] if a12 >= 0.5 else PALETTE["thesiscoral"]
        ax.scatter(a12, y, s=46, color=color, zorder=4, edgecolor="white",
                   linewidth=LW_MARKER_EDGE)
        if not pd.isna(row.get("ci_low", np.nan)):
            ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=color,
                    linewidth=LW_CONNECTOR, zorder=3)
    # arcuri magnitude bands (reflected around 0.5)
    for thr, lab in [(0.56, "small"), (0.64, "medium"), (0.71, "large")]:
        for xx in (thr, 1 - thr):
            ax.axvline(xx, color=PALETTE["thesislight"], linewidth=LW_BORDER, zorder=0)
    ax.axvline(0.5, color=PALETTE["thesisslate"], linestyle="--", linewidth=LW_CONNECTOR,
               zorder=1, label="no effect (0.5)")
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(forest_df["label"])[::-1], fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$\hat{A}_{12}$")
    ax.set_title("Vargha–Delaney effect-size forest")
    ax.legend(fontsize=8, loc="lower right")
    _save_summary(forest_df, out_path)
    return _save(fig, out_path)


def fig_rq3_parallel_coords(axes_df, style, out_path) -> Path:
    """Parallel-coordinates of per-system outcomes: Time, Bytes, Cost, Mean rank.

    One polyline per system family across the four outcome axes. Each axis is
    scaled independently (log for the geometric-mean ratio axes, linear for
    rank) and oriented so the best (lowest) value sits at the bottom; crossing
    polylines expose outcome inversions -- a system that leads on one dimension
    and trails on another -- in a single frame. Axis extremes are annotated with
    their real values so the normalized positions can be decoded.
    """
    cols = ["Time", "Bytes", "Cost", "Rank"]
    use_log = {"Time": True, "Bytes": True, "Cost": True, "Rank": False}
    df = axes_df[cols].astype(float)
    norm, lohi = {}, {}
    for col in cols:
        raw = df[col]
        t = np.log10(raw) if use_log[col] else raw
        lo, hi = float(np.nanmin(t)), float(np.nanmax(t))
        span = (hi - lo) or 1.0
        norm[col] = (t - lo) / span
        lohi[col] = (float(np.nanmin(raw)), float(np.nanmax(raw)))
    norm = pd.DataFrame(norm)
    x = np.arange(len(cols))

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for xi in x:
        ax.axvline(xi, color=PALETTE["thesislight"], linewidth=LW_BORDER, zorder=0)
    for sysname in norm.index:
        color = _system_color(style, sysname)
        ax.plot(x, norm.loc[sysname, cols].values, marker="o", markersize=6,
                linewidth=LW_SERIES, color=color, markeredgecolor="white",
                markeredgewidth=LW_MARKER_EDGE, zorder=3,
                label=SYSTEM_LABEL.get(sysname, sysname))

    def _fmt(col, val):
        if col == "Rank":
            return f"{val:.1f}"
        if val >= 100:
            return f"{val:.0f}×"
        return f"{val:.2g}×"

    for xi, col in zip(x, cols):
        lo, hi = lohi[col]
        ax.annotate(_fmt(col, hi), xy=(xi, 1.0), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7, color=PALETTE["thesisgray"])
        ax.annotate(_fmt(col, lo), xy=(xi, 0.0), xytext=(0, -13), textcoords="offset points",
                    ha="center", fontsize=7, color=PALETTE["thesisgray"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Time", "Bytes", "Cost", "Mean rank"], fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["best", "worst"], fontsize=8)
    ax.set_ylim(-0.1, 1.14)
    ax.set_xlim(-0.35, len(cols) - 0.65)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Per-system outcomes across dimensions (parallel coordinates)")
    ax.legend(fontsize=8, loc="upper center", ncol=max(len(norm.index), 1),
              bbox_to_anchor=(0.5, -0.09), frameon=False)
    _save_summary(axes_df.reset_index().rename(columns={"index": "system"}), out_path)
    return _save(fig, out_path)


# ════════════════════════════════════════════════════════════════════════════
# 05 — appendix per-cell grid
# ════════════════════════════════════════════════════════════════════════════


def fig_cell_grid(successful, cost_summary, cells, configs, style, out_path) -> Path:
    """Combined per-(workload x tier) grid: a small multiples panel per cell with
    the minimum wall-clock time by configuration. One composite image (thesis
    float expects a single ``09-cell-grid.png``).

    The minimum time spans several decades across configurations, so the y-axis
    is logarithmic; a bar has no valid baseline there, so each configuration is a
    point (minimum estimator) with its 95% bootstrap CI as a whisker rather than
    a bar.
    """
    n = len(cells)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.0 * nrows), squeeze=False)
    cs = cost_summary.reset_index()
    summary = []
    for idx, (wt, ds) in enumerate(cells):
        ax = axes[idx // ncols][idx % ncols]
        cell = successful[(successful["workload_type"] == wt) & (successful["dataset_size"] == ds)]
        present = [c for c in configs if c in cell["configuration"].unique()]
        x = np.arange(len(present))
        for xi, c in zip(x, present):
            v = cell[cell["configuration"] == c]["elapsed_time"].dropna().values
            if len(v) == 0:
                continue
            point, lo, hi = estimate_ci(v, kind="min")
            color = style.color(c)
            ax.errorbar(xi, point, yerr=[[max(point - lo, 0)], [max(hi - point, 0)]],
                        fmt="D", ms=5, color=color, ecolor=color,
                        elinewidth=LW_CONNECTOR, capsize=2.5, markeredgecolor="white",
                        markeredgewidth=LW_MARKER_EDGE, zorder=3)
            summary.append({"workload_type": wt, "dataset_size": ds, "configuration": c,
                            "min_time_s": point, "ci_low": lo, "ci_high": hi})
        ax.set_yscale("log")
        ax.set_xlim(-0.6, len(present) - 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([style.label(c).split(" ")[0] for c in present], rotation=30,
                           ha="right", fontsize=6.5)
        ax.set_title(f"{style.workload_label(wt)} ({ds})", fontsize=8.5)
        ax.set_ylabel("min time (s, log)", fontsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Per-cell summary grid — minimum wall-clock time by configuration",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


# ════════════════════════════════════════════════════════════════════════════
# 06 — additional RQ1 single-node figures
# ════════════════════════════════════════════════════════════════════════════


def _dir_bars(ax, pairs, summary, *, annotate_local, linthresh=1000.0):
    """Draw mirrored received/sent diverging bars on ``ax`` and append per-row
    records to ``summary``.

    pairs: list of (label, cfg, recv_array, sent_array, color). Bytes RECEIVED
    extend right, bytes SENT extend left, from a central zero on a symmetric-log
    axis; the median estimator (transfer convention) and its 95% bootstrap CI are
    drawn on each bar.
    """
    ypos, ylabels = [], []
    for i, (lab, cf, recv, sent, color) in enumerate(pairs):
        rp, rlo, rhi = estimate_ci(recv, kind="median") if len(recv) else (0.0, 0.0, 0.0)
        sp, slo, shi = estimate_ci(sent, kind="median") if len(sent) else (0.0, 0.0, 0.0)
        ypos.append(i)
        ylabels.append(lab)
        if annotate_local and cf == "local" and max(rp, sp) < 1024:
            ax.annotate("local FS\n(not like-for-like)", xy=(0, i), ha="center",
                        va="center", fontsize=6.0, fontstyle="italic",
                        color=PALETTE["thesisbrick"], zorder=6)
        else:
            ax.barh(i, max(rp, 0), height=0.6, color=color, edgecolor="white",
                    linewidth=LW_HAIRLINE, zorder=3)
            ax.barh(i, -max(sp, 0), height=0.6, color=tint(color, 0.5),
                    edgecolor="white", linewidth=LW_HAIRLINE, zorder=3)
            if rhi > rlo:
                ax.plot([rlo, rhi], [i, i], color=PALETTE["thesisslate"],
                        linewidth=LW_CONNECTOR, zorder=5)
            if shi > slo:
                ax.plot([-shi, -slo], [i, i], color=PALETTE["thesisslate"],
                        linewidth=LW_CONNECTOR, zorder=5)
        summary.append({"cfg": cf, "received_median": rp, "sent_median": sp,
                        "n_recv": int(len(recv)), "n_sent": int(len(sent))})
    ax.axvline(0, color=PALETTE["thesisslate"], linewidth=LW_BORDER, zorder=2)
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.xaxis.set_major_formatter(_bytes_fmt())
    # symlog crowds the first decade on each side around 0; show only 0 and
    # the >= 1k decade ticks (>= linthresh) so the labels do not collide.
    xl = ax.get_xlim()
    cand = [0.0] + [s * 10.0 ** k for k in range(3, 8) for s in (-1, 1)]
    ax.set_xticks([t for t in sorted(cand) if xl[0] <= t <= xl[1]])
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.set_ylim(-0.6, len(pairs) - 0.4)
    ax.tick_params(axis="x", labelsize=7)


def _dir_legend(fig, y=-0.02):
    """Shared received/sent/CI legend for the diverging-bar figures."""
    recv_h = plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                        color=tint(PALETTE["thesisgray"], 0.0),
                        markeredgecolor="white", label="received (right)")
    sent_h = plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                        color=tint(PALETTE["thesisgray"], 0.5),
                        markeredgecolor="white", label="sent (left)")
    ci_h = plt.Line2D([], [], color=PALETTE["thesisslate"], linewidth=LW_CONNECTOR,
                      label="95% bootstrap CI (median)")
    fig.legend(handles=[recv_h, sent_h, ci_h], loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, y))


def fig_bytes_directional(successful, workloads, configs, tiers, style, out_path):
    """Directional network transfer: bytes received vs sent, mirrored diverging bars.

    Companion to the received-only bytes grid, for the single-machine (RQ1)
    configurations only: for each configuration in a workload x tier cell a pair
    of horizontal bars diverges from a central zero — bytes RECEIVED extend right,
    bytes SENT extend left — on a symmetric-log axis, so the (large) download /
    (small) upload asymmetry is legible without clipping the small side. The
    median estimator (transfer convention) and its 95% bootstrap CI are drawn on
    each bar. The Shapefile/local path reads from the local filesystem, so both
    directions are ~0 and are annotated "local FS, not like-for-like" rather than
    plotted as a comparable transfer. The distributed Sedona client boundary is a
    different data path and belongs to RQ2 — see ``fig_distributed_client_boundary``.
    """
    tiers = _tier_order(style, tiers)
    nrows, ncols = len(workloads), len(tiers)
    fig = plt.figure(figsize=(3.8 * ncols + 1.0, 2.5 * nrows + 1.0))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.6, wspace=0.4)
    summary = []

    for r, wt in enumerate(workloads):
        for c, ds in enumerate(tiers):
            ax = fig.add_subplot(gs[r, c])
            cell = successful[(successful["workload_type"] == wt)
                              & (successful["dataset_size"] == ds)]
            present = [cf for cf in configs if cf in cell["configuration"].unique()]
            pairs = []
            for cf in present:
                sub = cell[cell["configuration"] == cf]
                pairs.append((style.label(cf).split(" ")[0], cf,
                              sub["network_bytes_received"].dropna().values,
                              sub["network_bytes_sent"].dropna().values,
                              style.color(cf)))
            n0 = len(summary)
            if pairs:
                _dir_bars(ax, pairs, summary, annotate_local=True)
                for rec in summary[n0:]:
                    rec.update({"workload_type": wt, "dataset_size": ds, "panel": "single-machine"})
            else:
                ax.text(0.5, 0.5, "did not run\nby design", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, fontstyle="italic",
                        color=PALETTE["thesisbrick"])
                ax.set_xticks([])
                ax.set_yticks([])
            if r == 0:
                ax.set_title(ds.capitalize(), fontsize=10, fontweight="bold")
            if c == ncols - 1 and pairs:
                ax.annotate(style.workload_label(wt), xy=(1.02, 0.5),
                            xycoords="axes fraction", rotation=270, va="center",
                            ha="left", fontsize=8.5, color=PALETTE["thesisslate"])

    _dir_legend(fig)
    fig.suptitle("Directional network transfer — bytes received versus sent", fontsize=12, y=1.0)
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


def fig_distributed_client_boundary(successful, style, out_path,
                                    sedona_cfg="databricks-broadcast-8-nodes",
                                    workload="national-scale-spatial-join"):
    """Directional network transfer at the distributed Sedona client boundary (RQ2).

    The received/sent split measured at the driver process for the distributed
    national-scale spatial join, one mirrored diverging-bar pair per size tier for
    a single broadcast worker count. Bytes RECEIVED extend right, bytes SENT left,
    on a symmetric-log axis with the median estimator and its 95% bootstrap CI.
    This is the distributed analogue of the single-machine directional figure
    (``fig_bytes_directional``); it is the only place the sent direction is shown
    for the distributed path, complementing the shuffle-bytes phase breakdown.
    """
    sed = successful[(successful["workload_type"] == workload)
                     & (successful["configuration"] == sedona_cfg)]
    sed_tiers = _tier_order(style, sed["dataset_size"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 0.9 * max(len(sed_tiers), 1) + 2.2))
    fig.subplots_adjust(bottom=0.34, top=0.86)
    summary = []
    pairs = []
    for ds in sed_tiers:
        sub = sed[sed["dataset_size"] == ds]
        pairs.append((ds.capitalize(), sedona_cfg,
                      sub["network_bytes_received"].dropna().values,
                      sub["network_bytes_sent"].dropna().values,
                      _system_color(style, "sedona")))
    if pairs:
        _dir_bars(ax, pairs, summary, annotate_local=False)
        for rec in summary:
            rec.update({"workload_type": workload, "dataset_size": "(per tier)",
                        "panel": "sedona-client-boundary"})
    ax.set_xlabel("← sent        bytes (symlog)        received →", fontsize=8.5)
    ax.set_ylabel("size tier", fontsize=9)
    _dir_legend(fig, y=0.01)
    fig.suptitle(f"Distributed client boundary — {style.label(sedona_cfg)} "
                 "(national-scale join)", fontsize=12, y=0.98)
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


def fig_cost_time_quadrant(cost_summary, successful, workloads, configs, tiers, style,
                           out_path, sedona_prefix="databricks-broadcast"):
    """Single-machine cost-vs-time decision quadrant (analogue of the distributed Pareto).

    One scatter panel per workload pattern: x = total operational cost (USD, log),
    y = wall-clock minimum (s, log). Marker glyph encodes the configuration and
    colour encodes the size tier; per-panel median cross-hairs split the plane
    into the four labelled decision quadrants (fast+cheap, slow+cheap, fast+costly,
    slow+costly). The national-scale panel additionally anchors the best-worker
    Sedona point (lowest-time broadcast worker count) as a star, for context with
    the single-machine engines.
    """
    tiers = _tier_order(style, tiers)
    cs = cost_summary.reset_index()
    glyph = {"duckdb": "o", "postgis": "s", "local": "^"}
    ncols = min(len(workloads), 2)
    nrows = (len(workloads) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 3.9 * nrows), squeeze=False)
    summary = []
    for idx, wt in enumerate(workloads):
        ax = axes[idx // ncols][idx % ncols]
        pts = []
        for cf in configs:
            for ds in tiers:
                sub = successful[(successful["workload_type"] == wt)
                                 & (successful["dataset_size"] == ds)
                                 & (successful["configuration"] == cf)]
                t = sub["elapsed_time"].dropna().values
                row = cs[(cs["workload_type"] == wt) & (cs["dataset_size"] == ds)
                         & (cs["configuration"] == cf)]
                if len(t) == 0 or len(row) == 0:
                    continue
                tt, cc = float(np.min(t)), float(row["total_cost"].iloc[0])
                if cc <= 0 or tt <= 0:
                    continue
                tcol = style.size_colors.get(ds, PALETTE["thesisgray"])
                ax.scatter(cc, tt, s=72, color=tcol, marker=glyph.get(cf, "o"),
                           edgecolor=shade(tcol, 0.35), linewidth=LW_BORDER, zorder=4)
                pts.append((cc, tt))
                summary.append({"workload_type": wt, "configuration": cf, "dataset_size": ds,
                                "cost_usd": cc, "min_time_s": tt, "marker": "config"})
        # Sedona best-worker anchor (only where broadcast ran this workload)
        sed = successful[(successful["workload_type"] == wt)
                         & successful["configuration"].str.startswith(sedona_prefix)]
        bcfg, bds, bt, bcost = None, None, np.inf, np.nan
        for (cfg, ds), g in sed.groupby(["configuration", "dataset_size"]):
            v = g["elapsed_time"].dropna().values
            r = cs[(cs["workload_type"] == wt) & (cs["dataset_size"] == ds)
                   & (cs["configuration"] == cfg)]
            if len(v) == 0 or len(r) == 0:
                continue
            m = float(np.min(v))
            if m < bt:
                bcfg, bds, bt, bcost = cfg, ds, m, float(r["total_cost"].iloc[0])
        if bcfg and bcost > 0:
            ax.scatter(bcost, bt, marker="*", s=260, color=_system_color(style, "sedona"),
                       edgecolor="white", linewidth=LW_MARKER_EDGE, zorder=6)
            ax.annotate(f"Sedona\nbest worker", xy=(bcost, bt), xytext=(5, 4),
                        textcoords="offset points", fontsize=6.5,
                        color=shade(_system_color(style, "sedona"), 0.2))
            summary.append({"workload_type": wt, "configuration": bcfg, "dataset_size": bds,
                            "cost_usd": bcost, "min_time_s": bt, "marker": "sedona-anchor"})
        if len(pts) >= 1:
            cx, cy = float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts]))
            ax.axvline(cx, color=PALETTE["thesisgray"], linestyle=":", linewidth=LW_BORDER, zorder=1)
            ax.axhline(cy, color=PALETTE["thesisgray"], linestyle=":", linewidth=LW_BORDER, zorder=1)
            for fx, fy, txt, ha, va in [
                (0.02, 0.02, "fast · cheap", "left", "bottom"),
                (0.02, 0.98, "slow · cheap", "left", "top"),
                (0.98, 0.02, "fast · costly", "right", "bottom"),
                (0.98, 0.98, "slow · costly", "right", "top"),
            ]:
                ax.annotate(txt, xy=(fx, fy), xycoords="axes fraction", ha=ha, va=va,
                            fontsize=6.5, fontstyle="italic", color=PALETTE["thesisgray"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Operational cost (USD, log)", fontsize=8.5)
        if idx % ncols == 0:
            ax.set_ylabel("Wall-clock minimum (s, log)", fontsize=8.5)
        ax.set_title(style.workload_label(wt), fontsize=9.5)
    for j in range(len(workloads), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    cfg_h = [plt.Line2D([], [], marker=glyph.get(c, "o"), linestyle="", color=PALETTE["thesisgray"],
                        markeredgecolor="white", label=style.label(c).split(" ")[0])
             for c in configs if c in glyph]
    cfg_h.append(plt.Line2D([], [], marker="*", linestyle="", color=_system_color(style, "sedona"),
                            markeredgecolor="white", markersize=11, label="Sedona (best worker)"))
    tier_h = [plt.Line2D([], [], marker="o", linestyle="", color=style.size_colors[t],
                         markeredgecolor=shade(style.size_colors[t], 0.35), label=t.capitalize())
              for t in tiers]
    fig.legend(handles=cfg_h, loc="lower center", ncol=len(cfg_h), fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04), title="Configuration")
    fig.legend(handles=tier_h, loc="lower center", ncol=len(tier_h), fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.10), title="Tier")
    fig.suptitle("Single-machine cost–time decision quadrant", fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


def fig_bytes_vs_cardinality(successful, configs, style, out_path,
                             ref_rates=(100, 1_000, 10_000, 100_000)):
    """Network bytes received versus result cardinality (log-log scatter).

    One marker per (workload x tier x configuration) cell: x = median result
    cardinality (rows; the not-recorded -1 of the local Shapefile path is
    excluded), y = median network bytes received. Colour encodes configuration
    and marker glyph the query pattern. Faint diagonal guidelines mark constant
    bytes-per-result-row rates, so cells of equal cardinality align vertically and
    the per-row transfer cost can be read off. Promotes result cardinality out of
    the appendix agreement table into a transfer-efficiency view.
    """
    wl_marker = {"point-in-polygon-lookup": "o", "knn-search": "s",
                 "bbox-filtering": "^", "national-scale-spatial-join": "D"}
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    summary, xs_all, ys_all = [], [], []
    workloads = [w for w in wl_marker if w in successful["workload_type"].unique()]
    tiers = _tier_order(style, successful["dataset_size"].unique())
    for cf in configs:
        for wt in workloads:
            for ds in tiers:
                sub = successful[(successful["workload_type"] == wt)
                                 & (successful["dataset_size"] == ds)
                                 & (successful["configuration"] == cf)]
                card = sub["result_cardinality"].dropna().values
                card = card[card > 0]  # drop the -1 not-recorded sentinel (and 0)
                recv = sub["network_bytes_received"].dropna().values
                recv = recv[recv > 0]
                if len(card) == 0 or len(recv) == 0:
                    continue
                x, y = float(np.median(card)), float(np.median(recv))
                ax.scatter(x, y, s=64, color=_name_color(style, cf),
                           marker=wl_marker.get(wt, "o"), edgecolor="white",
                           linewidth=LW_MARKER_EDGE, zorder=4)
                xs_all.append(x)
                ys_all.append(y)
                summary.append({"workload_type": wt, "dataset_size": ds, "configuration": cf,
                                "median_cardinality": x, "median_bytes_received": y,
                                "bytes_per_row": y / x})
    if xs_all:
        xlo, xhi = min(xs_all) * 0.6, max(xs_all) * 1.6
        xr = np.array([xlo, xhi])
        for k in ref_rates:
            ax.plot(xr, k * xr, color=PALETTE["thesislight"], linewidth=LW_HAIRLINE,
                    linestyle="-", zorder=0)
            kf = _bytes_fmt()(k, None)
            ax.annotate(f"{kf}/row", xy=(xhi, k * xhi), xytext=(-2, 2),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=6.0, color=PALETTE["thesisgray"], zorder=0)
        ax.set_xlim(xlo, xhi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(_bytes_fmt())
    ax.set_xlabel("Result cardinality (rows, log)")
    ax.set_ylabel("Network bytes received (median, log)")
    cfg_h = [plt.Line2D([], [], marker="o", linestyle="", color=_name_color(style, c),
                        markeredgecolor="white", label=style.label(c))
             for c in configs]
    wl_h = [plt.Line2D([], [], marker=wl_marker[w], linestyle="", color=PALETTE["thesisgray"],
                       markeredgecolor="white", label=style.workload_label(w))
            for w in workloads]
    leg1 = ax.legend(handles=cfg_h, fontsize=7.5, loc="upper left", title="Configuration")
    ax.add_artist(leg1)
    ax.legend(handles=wl_h, fontsize=7.5, loc="lower right", title="Pattern")
    ax.set_title("Network transfer versus result cardinality")
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


# ════════════════════════════════════════════════════════════════════════════
# 06 — additional RQ2 distributed figure
# ════════════════════════════════════════════════════════════════════════════


def fig_distributed_strategy_contrast(successful, cost_summary, style, out_path,
                                      tier="small", oom_tiers=("medium", "large")):
    """Broadcast vs partitioned join strategy at the small tier (two panels).

    Panel (a) wall-clock minimum (log) and panel (b) total operational cost,
    both against worker count {2,4,8,12,16}, with one line per strategy. The cache
    holds partitioned rows for the SMALL tier only; the medium/large partitioned
    runs did not complete (executor OOM). Those failures are NOT in the cache as
    failed rows, so the medium/large partitioned absence is rendered as an
    editorial annotation sourced from the descriptive-statistics table, not from
    data. The figure's quantitative content is the small-tier broadcast-vs-
    partitioned magnitude gap.
    """
    d = successful[(successful["workload_type"] == "national-scale-spatial-join")
                   & successful["configuration"].str.startswith("databricks")
                   & (successful["dataset_size"] == tier)].copy()
    d["strat"] = d["configuration"].map(extract_strategy)
    d["wc"] = d["configuration"].map(extract_worker_count)
    cs = cost_summary.reset_index()
    strat_list = ["broadcast", "partitioned"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    summary = []
    for strat in strat_list:
        g = d[d["strat"] == strat]
        rows = []
        for w, gg in g.groupby("wc"):
            v = gg["elapsed_time"].dropna().values
            if len(v):
                p, lo, hi = estimate_ci(v, kind="min")
                rows.append((int(w), p, lo, hi))
        rows.sort()
        if rows:
            ws = [r[0] for r in rows]
            ps = [r[1] for r in rows]
            yerr = [[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]]
            axes[0].errorbar(ws, ps, yerr=yerr, marker="o", ms=5, capsize=2.5,
                             color=style.strategy_colors[strat], linewidth=LW_SERIES,
                             ecolor=style.strategy_colors[strat], elinewidth=LW_CONNECTOR,
                             label=strat.capitalize())
            for w, p, lo, hi in rows:
                summary.append({"strategy": strat, "dataset_size": tier, "workers": w,
                                "metric": "min_time_s", "value": p, "ci_low": lo, "ci_high": hi})
    for strat in strat_list:
        rows = []
        for w in sorted(d[d["strat"] == strat]["wc"].unique()):
            cfg = f"databricks-{strat}-{int(w)}-nodes"
            r = cs[(cs["workload_type"] == "national-scale-spatial-join")
                   & (cs["dataset_size"] == tier) & (cs["configuration"] == cfg)]
            if len(r):
                rows.append((int(w), float(r["total_cost"].iloc[0])))
        rows.sort()
        if rows:
            axes[1].plot([r[0] for r in rows], [r[1] for r in rows], marker="o", ms=5,
                         color=style.strategy_colors[strat], linewidth=LW_SERIES,
                         label=strat.capitalize())
            for w, c in rows:
                summary.append({"strategy": strat, "dataset_size": tier, "workers": w,
                                "metric": "total_cost_usd", "value": c, "ci_low": np.nan,
                                "ci_high": np.nan})
    oom = " / ".join(oom_tiers)
    axes[0].annotate(
        f"Partitioned {oom} tiers: did not complete\n(executor OOM — see descriptive-statistics table)",
        xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top", fontsize=7,
        fontstyle="italic", color=PALETTE["thesisbrick"])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Worker count")
    axes[0].set_ylabel("Wall-clock minimum (s, log)")
    axes[0].set_title("Wall-clock time", fontsize=10)
    axes[1].set_xlabel("Worker count")
    axes[1].set_ylabel("Total operational cost (USD)")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Operational cost", fontsize=10)
    for ax in axes:
        ax.set_xticks([2, 4, 8, 12, 16])
        ax.legend(fontsize=8, title="Strategy")
    fig.suptitle(f"Broadcast versus partitioned join at the {tier} tier (national-scale join)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


# ════════════════════════════════════════════════════════════════════════════
# 06 — additional RQ3 synthesis figures
# ════════════════════════════════════════════════════════════════════════════


def fig_cross_pattern(successful, workloads, configs, tiers, style, out_path):
    """Cross-pattern single-machine comparison at fixed tier (grouped bars + CI).

    One panel per size tier: x = workload pattern, grouped bars per configuration,
    y = wall-clock minimum (s, log) with the 95% bootstrap CI as a whisker.
    Patterns or configurations that did not run at a tier by design (kNN and
    Shapefile at the large tier) are labelled did-not-run-by-design rather than
    drawn as a zero bar. The cross-pattern view exposes whether a configuration's
    lead is pattern-portable or pattern-specific.
    """
    tiers = _tier_order(style, tiers)
    fig, axes = plt.subplots(1, len(tiers), figsize=(4.4 * len(tiers) + 0.5, 4.4),
                             squeeze=False, sharey=True)
    axes = axes[0]
    summary = []
    width = 0.8 / max(len(configs), 1)
    for ax, ds in zip(axes, tiers):
        x = np.arange(len(workloads))
        for j, cf in enumerate(configs):
            off = (j - len(configs) / 2 + 0.5) * width
            for xi, wt in enumerate(workloads):
                v = successful[(successful["workload_type"] == wt)
                               & (successful["dataset_size"] == ds)
                               & (successful["configuration"] == cf)]["elapsed_time"].dropna().values
                if len(v) == 0:
                    continue
                p, lo, hi = estimate_ci(v, kind="min")
                ax.bar(xi + off, p, width * 0.9, color=style.color(cf), edgecolor="white",
                       linewidth=LW_HAIRLINE, zorder=3)
                ax.errorbar(xi + off, p, yerr=[[max(p - lo, 0)], [max(hi - p, 0)]],
                            fmt="none", ecolor=PALETTE["thesisslate"], elinewidth=LW_CONNECTOR,
                            capsize=2, zorder=5)
                summary.append({"dataset_size": ds, "workload_type": wt, "configuration": cf,
                                "min_time_s": p, "ci_low": lo, "ci_high": hi})
        # mark did-not-run-by-design cells
        for xi, wt in enumerate(workloads):
            ran = successful[(successful["workload_type"] == wt)
                             & (successful["dataset_size"] == ds)
                             & (successful["configuration"].isin(configs))]
            if ran.empty:
                ax.annotate("did not run\nby design", xy=(xi, 0.5), xycoords=("data", "axes fraction"),
                            ha="center", va="center", fontsize=6.5, fontstyle="italic",
                            color=PALETTE["thesisbrick"])
        # configurations absent from this whole tier (e.g. Shapefile at large):
        # name them as did-not-run-by-design rather than leaving a silent gap.
        present_cfgs = set(successful[(successful["dataset_size"] == ds)
                                      & successful["workload_type"].isin(workloads)
                                      & successful["configuration"].isin(configs)]["configuration"].unique())
        missing = [c for c in configs if c not in present_cfgs]
        if missing:
            ax.annotate("; ".join(f"{style.label(c)}" for c in missing)
                        + ":\nsmall tier only (by design)",
                        xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
                        fontsize=6.5, fontstyle="italic", color=PALETTE["thesisbrick"])
        ax.set_yscale("log")
        ax.set_xticks(range(len(workloads)))
        ax.set_xticklabels([style.workload_label(w) for w in workloads], rotation=20,
                           ha="right", fontsize=7.5)
        ax.set_title(ds.capitalize(), fontsize=10, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Wall-clock minimum (s, log)")
    cfg_h = [plt.Line2D([], [], marker="s", linestyle="", color=style.color(c),
                        markeredgecolor="white", label=style.label(c)) for c in configs]
    fig.legend(handles=cfg_h, loc="lower center", ncol=len(cfg_h), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cross-pattern single-machine comparison by tier", fontsize=12, y=1.02)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


def fig_rank_portability(geomean, style, out_path):
    """Rank portability versus absolute magnitude (two panels).

    Both panels share the same systems and the three outcome dimensions (Time,
    Bytes, Cost). Left: each system's RANK on each dimension (1 = best), drawn as
    a slopegraph — near-flat lines mean the ordering is portable across
    dimensions. Right: the same systems' ABSOLUTE geometric-mean-normalized
    magnitude per dimension (log; 1.0 = the dimension's best), where wide vertical
    spread means the magnitudes do not carry even where the order does. Backs the
    "orderings carry, magnitudes do not" external-validity claim; distinct from the
    rank-vs-tier ranking-stability figure.
    """
    dims = [c for c in ["Time", "Bytes", "Cost"] if c in geomean.columns]
    systems = list(geomean.index)
    ranks = {d: geomean[d].rank(method="min") for d in dims}
    x = np.arange(len(dims))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    summary = []
    for sysname in systems:
        color = _system_color(style, sysname)
        short = SYSTEM_LABEL.get(sysname, sysname).split(" ")[0]
        ys_rank = [float(ranks[d][sysname]) for d in dims]
        axL.plot(x, ys_rank, marker="o", ms=7, color=color, linewidth=LW_SERIES,
                 markeredgewidth=LW_MARKER_EDGE, markeredgecolor="white")
        axL.annotate(short, xy=(x[-1], ys_rank[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=7, color=shade(color, 0.2))
        ys_mag = [float(geomean.loc[sysname, d]) for d in dims]
        axR.plot(x, ys_mag, marker="o", ms=7, color=color, linewidth=LW_SERIES,
                 markeredgewidth=LW_MARKER_EDGE, markeredgecolor="white",
                 label=SYSTEM_LABEL.get(sysname, sysname))
        axR.annotate(short, xy=(x[-1], ys_mag[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=7, color=shade(color, 0.2))
        for d in dims:
            summary.append({"system": sysname, "dimension": d, "rank": float(ranks[d][sysname]),
                            "geomean_norm": float(geomean.loc[sysname, d])})
    axL.invert_yaxis()
    axL.set_yticks(range(1, len(systems) + 1))
    axL.set_xticks(x)
    axL.set_xticklabels(dims)
    axL.set_xlim(-0.3, len(dims) - 0.7)
    axL.set_ylabel("Rank (1 = best)")
    axL.set_title("Per-dimension rank", fontsize=10)
    axR.axhline(1.0, color=PALETTE["thesisslate"], linestyle="--", linewidth=LW_CONNECTOR,
                zorder=1, label="dimension best (1.0)")
    axR.set_yscale("log")
    axR.set_xticks(x)
    axR.set_xticklabels(dims)
    axR.set_xlim(-0.3, len(dims) - 0.7)
    axR.set_ylabel(r"Geomean-normalized magnitude ($\times$, log)")
    axR.set_title("Absolute magnitude", fontsize=10)
    handles, labels = axR.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Rank portability versus absolute magnitude", fontsize=12, y=1.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


# ════════════════════════════════════════════════════════════════════════════
# 09 — appendix supplementary figures
# ════════════════════════════════════════════════════════════════════════════


def fig_cpu_wall_ratio(successful, workloads, configs, tiers, style, out_path):
    """CPU-to-wall-clock ratio per cell (compute-bound vs transfer-bound diagnostic).

    Grouped bars of (cpu_user + cpu_system) / elapsed across workload x tier cells,
    one bar per configuration, with a reference line at 1.0 (one fully-busy core).
    To be robust to the sub-millisecond per-iteration CPU counter resolution
    (many per-iteration values round to 0), the ratio is computed from per-cell
    SUMS of CPU and wall time — the time-average number of busy cores. A ratio
    well above 1.0 is multi-core compute-bound; well below 1.0 is transfer- or
    IO-bound (or idle waiting).
    """
    tiers = _tier_order(style, tiers)
    cells = [(wt, ds) for wt in workloads for ds in tiers]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(cells) * len(configs)), 4.4))
    width = 0.8 / max(len(configs), 1)
    x = np.arange(len(cells))
    summary = []
    for j, cf in enumerate(configs):
        off = (j - len(configs) / 2 + 0.5) * width
        for ci, (wt, ds) in enumerate(cells):
            sub = successful[(successful["workload_type"] == wt)
                             & (successful["dataset_size"] == ds)
                             & (successful["configuration"] == cf)]
            if sub.empty:
                continue
            u = float(np.nansum(sub["cpu_time_user_seconds"].values))
            s = float(np.nansum(sub["cpu_time_system_seconds"].values))
            e = float(np.nansum(sub["elapsed_time"].values))
            if e <= 0:
                continue
            ratio = (u + s) / e
            ax.bar(ci + off, ratio, width * 0.9, color=style.color(cf), edgecolor="white",
                   linewidth=LW_HAIRLINE, zorder=3)
            summary.append({"workload_type": wt, "dataset_size": ds, "configuration": cf,
                            "cpu_user_s": u, "cpu_system_s": s, "elapsed_s": e,
                            "cpu_wall_ratio": ratio})
    ax.axhline(1.0, color=PALETTE["thesisbrick"], linestyle="--", linewidth=LW_CONNECTOR,
               zorder=4, label="1.0 (one fully-busy core)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{style.workload_label(wt)[:10]}\n{ds}" for wt, ds in cells],
                       fontsize=7)
    ax.set_ylabel("CPU-to-wall ratio  (cpu$_{user}$+cpu$_{sys}$)/elapsed")
    ax.set_ylim(bottom=0)
    handles = [plt.Line2D([], [], marker="s", linestyle="", color=style.color(c),
                          markeredgecolor="white", label=style.label(c)) for c in configs]
    handles.append(plt.Line2D([], [], color=PALETTE["thesisbrick"], linestyle="--",
                              linewidth=LW_CONNECTOR, label="1.0 (one busy core)"))
    ax.legend(handles=handles, fontsize=7.5, ncol=2)
    ax.set_title("CPU-to-wall-clock ratio per configuration")
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


def fig_size_scaling_slope(successful, workloads, configs, style, out_path):
    """Same-engine size-scaling slopegraph (small -> large), one panel per pattern.

    Within each workload panel, one line per engine traces the wall-clock minimum
    (log) across size tiers, annotated with its growth factor (largest tier /
    smallest tier). The Shapefile/local path ran the small tier only, so it
    appears as a single small-tier point rather than a slope. Restyles the older
    grouped-bar size-scaling chart into the thesis slopegraph idiom.
    """
    fig, axes = plt.subplots(1, len(workloads), figsize=(3.6 * len(workloads) + 0.5, 4.4),
                             squeeze=False, sharey=True)
    axes = axes[0]
    summary = []
    all_tiers = _tier_order(style, successful["dataset_size"].unique())
    for ax, wt in zip(axes, workloads):
        wdata = successful[successful["workload_type"] == wt]
        for cf in configs:
            pts = []
            for ds in all_tiers:
                v = wdata[(wdata["configuration"] == cf)
                          & (wdata["dataset_size"] == ds)]["elapsed_time"].dropna().values
                if len(v) == 0:
                    continue
                p, lo, hi = estimate_ci(v, kind="min")
                pts.append((style.size_order.get(ds, 99), ds, p, lo, hi))
            if not pts:
                continue
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [p[2] for p in pts]
            color = style.color(cf)
            # min-estimator bootstrap CI lies at/above the point (resampled minima
            # are >= the full-sample min), so clamp the lower arm at 0 like the
            # other minimum-estimator figures in this module.
            lo_err = [max(p[2] - p[3], 0.0) for p in pts]
            hi_err = [max(p[4] - p[2], 0.0) for p in pts]
            if len(pts) == 1:
                ax.errorbar(xs, ys, yerr=[[lo_err[0]], [hi_err[0]]],
                            fmt="^", ms=7, color=color, ecolor=color, capsize=2.5,
                            markeredgecolor="white", markeredgewidth=LW_MARKER_EDGE, zorder=4)
                ax.annotate(f"{style.label(cf).split(' ')[0]}\n(small only)", xy=(xs[0], ys[0]),
                            xytext=(6, 0), textcoords="offset points", va="center",
                            fontsize=6.5, color=shade(color, 0.2))
            else:
                ax.errorbar(xs, ys, yerr=[lo_err, hi_err], marker="o", ms=5, color=color,
                            ecolor=color, capsize=2.5, elinewidth=LW_CONNECTOR,
                            linewidth=LW_SERIES, markeredgecolor="white",
                            markeredgewidth=LW_MARKER_EDGE, zorder=4)
                growth = ys[-1] / ys[0] if ys[0] > 0 else np.nan
                ax.annotate(f"{style.label(cf).split(' ')[0]}  ×{growth:.0f}",
                            xy=(xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=6.5, color=shade(color, 0.2))
            for o, ds, p, lo, hi in pts:
                summary.append({"workload_type": wt, "configuration": cf, "dataset_size": ds,
                                "min_time_s": p, "ci_low": lo, "ci_high": hi})
        ax.set_yscale("log")
        ax.set_xticks([style.size_order[t] for t in all_tiers])
        ax.set_xticklabels([t.capitalize() for t in all_tiers])
        ax.set_xlim(-0.3, len(all_tiers) - 0.4)
        ax.set_title(style.workload_label(wt), fontsize=9.5)
        if ax is axes[0]:
            ax.set_ylabel("Wall-clock minimum (s, log)")
    fig.suptitle("Same-engine size scaling (minimum estimator, with growth factor)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


def fig_spark_stage_profile(successful, style, out_path, strategy="broadcast", tier="large"):
    """Spark per-stage duration profile by worker count (small multiples).

    Parses the per-iteration ``stage_durations_ms`` JSON array (distributed runs
    only) for the chosen strategy and tier, and shows one small panel per worker
    count {2,4,8,12,16}. Each panel is a horizontal profile of the median duration
    of each Spark stage (stage index on the y-axis), shading stages with a
    lightness ramp of the strategy colour. This is a finer view than the coarse
    three-phase split (executor read / shuffle / driver collection).
    """
    d = successful[successful["configuration"].str.startswith(f"databricks-{strategy}")
                   & (successful["dataset_size"] == tier)].copy()
    d = d[d["stage_durations_ms"].notna()]
    d["wc"] = d["configuration"].map(extract_worker_count)

    def _parse(v):
        if isinstance(v, str):
            try:
                return [float(z) for z in json.loads(v)]
            except Exception:
                return None
        try:
            return [float(z) for z in v]
        except Exception:
            return None

    workers = sorted(int(w) for w in d["wc"].dropna().unique())
    per_worker, maxstages = {}, 0
    for w in workers:
        arrs = [a for a in (_parse(v) for v in d[d["wc"] == w]["stage_durations_ms"]) if a]
        per_worker[w] = arrs
        maxstages = max(maxstages, max((len(a) for a in arrs), default=0))
    medians = {}
    for w in workers:
        if not per_worker[w]:
            medians[w] = np.zeros(maxstages)
            continue
        padded = np.array([a + [0.0] * (maxstages - len(a)) for a in per_worker[w]], dtype=float)
        medians[w] = np.median(padded, axis=0) / 1000.0  # ms -> s

    # Spark logs a long tail of ~0-duration stages; trim to the last stage whose
    # median exceeds 1% of the global maximum so the per-worker profile is legible.
    xmax_full = max((medians[w].max() for w in workers), default=1.0)
    thresh = 0.01 * xmax_full
    active = 0
    for w in workers:
        nz = np.nonzero(medians[w] > thresh)[0]
        if len(nz):
            active = max(active, int(nz.max()))
    nstages = active + 1
    medians = {w: medians[w][:nstages] for w in workers}

    ramp = lightness_ramp(style.strategy_colors.get(strategy, PALETTE["thesissteel"]),
                          max(nstages, 1))
    fig, axes = plt.subplots(1, len(workers), figsize=(2.2 * len(workers) + 0.6, 4.2),
                             squeeze=False, sharex=True)
    axes = axes[0]
    summary = []
    xmax = max((medians[w].max() for w in workers), default=1.0)
    for ax, w in zip(axes, workers):
        vals = medians[w]
        ys = np.arange(len(vals))
        ax.barh(ys, vals, color=[ramp[i] for i in range(len(vals))], edgecolor="white",
                linewidth=LW_HAIRLINE, zorder=3)
        ax.set_title(f"{w}N", fontsize=9, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0, xmax * 1.05)
        if ax is axes[0]:
            ax.set_yticks(ys)
            ax.set_yticklabels([f"S{i}" for i in ys], fontsize=7)
            ax.set_ylabel("Spark stage")
        else:
            ax.set_yticks(ys)
            ax.set_yticklabels([])
        ax.tick_params(axis="x", labelsize=7)
        for i, vv in enumerate(vals):
            summary.append({"worker_count": w, "stage_index": i, "median_duration_s": float(vv)})
    fig.supxlabel("Median stage duration (s)", fontsize=9)
    fig.suptitle(f"Spark per-stage duration profile — {strategy}, {tier} tier", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


def fig_reproducibility(successful, style, out_path, cells=None):
    """Run-to-run reproducibility across the 30 benchmark passes.

    For a few representative cells, the line traces each pass's minimum elapsed
    time across ``benchmark_run`` 1--30 (between-run reproducibility), while the
    band shaded behind it is that pass's within-run bootstrap CI half-width on the
    minimum (within-run repeatability). A line that stays flat with a thin band is
    both reproducible between passes and repeatable within a pass. The y-axis is
    logarithmic because the representative cells span several decades.
    """
    if cells is None:
        cells = [
            ("point-in-polygon-lookup", "duckdb", "small"),
            ("knn-search", "postgis", "small"),
            ("bbox-filtering", "local", "small"),
            ("national-scale-spatial-join", "databricks-broadcast-8-nodes", "large"),
        ]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    summary = []
    for wt, cf, ds in cells:
        sub = successful[(successful["workload_type"] == wt)
                         & (successful["dataset_size"] == ds)
                         & (successful["configuration"] == cf)]
        if sub.empty:
            continue
        runs = sorted(int(r) for r in sub["benchmark_run"].dropna().unique())
        xs, mins, hws = [], [], []
        for r in runs:
            v = sub[sub["benchmark_run"] == r]["elapsed_time"].dropna().values
            v = v[v > 0]
            if len(v) == 0:
                continue
            p, lo, hi = estimate_ci(v, kind="min")
            xs.append(r)
            mins.append(p)
            hws.append((hi - lo) / 2.0)
        if not xs:
            continue
        color = _name_color(style, cf)
        mins = np.array(mins)
        hws = np.array(hws)
        ax.fill_between(xs, np.maximum(mins - hws, 1e-12), mins + hws, color=tint(color, 0.6),
                        alpha=0.55, linewidth=0, zorder=2)
        lab = f"{_name_label(style, cf).split(' ')[0]} · {style.workload_label(wt)[:8]} ({ds})"
        ax.plot(xs, mins, marker="o", ms=3, color=color, linewidth=LW_SERIES, zorder=3, label=lab)
        between_cv = float(np.std(mins, ddof=1) / np.mean(mins)) if len(mins) > 1 and np.mean(mins) > 0 else np.nan
        summary.append({"workload_type": wt, "configuration": cf, "dataset_size": ds,
                        "n_passes": len(xs), "between_run_cv": between_cv,
                        "mean_within_run_halfwidth_s": float(np.mean(hws)),
                        "median_min_s": float(np.median(mins))})
    ax.set_yscale("log")
    ax.set_xlabel("Benchmark pass (1–30)")
    ax.set_ylabel("Per-pass minimum elapsed (s, log)")
    ax.set_title("Run-to-run reproducibility: between-pass minima vs within-pass CI")
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)

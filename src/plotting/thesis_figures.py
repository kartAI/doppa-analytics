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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.stats import bootstrap as _scipy_bootstrap, gaussian_kde

from src.analysis.stats import classify_a12

from .style import PALETTE, StyleConfig, shade, tint

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
        linewidth=0.3,
    )
    vmin, vmean, vmed = float(np.min(vals)), float(np.mean(vals)), float(np.median(vals))
    for x, lab, c, ls in [
        (vmin, f"min = {vmin:.3g}\\,s", PALETTE["thesisslate"], "-"),
        (vmed, f"median = {vmed:.3g}", PALETTE["thesissage"], "--"),
        (vmean, f"mean = {vmean:.3g}", PALETTE["thesisbrick"], ":"),
    ]:
        ax.axvline(x, color=c, linestyle=ls, linewidth=1.4, label=lab.replace("\\,", " "))
    ax.set_xlabel("Per-iteration elapsed time (s)")
    ax.set_ylabel("Count")
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
            label="Per-iteration time")
    ax.axhline(steady, color=PALETTE["thesisslate"], linestyle="--", linewidth=1.2,
               label=f"steady-state min = {steady:.3g} s")
    ax.set_xlabel(f"Iteration index within pass {pass_run}")
    ax.set_ylabel("Elapsed time (s)")
    ax.set_title(f"{style.workload_label(d['workload_type'].iloc[0])} — {style.label(cfg)}")
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
    ax.plot(ks, rel, color=color, linewidth=1.3)
    ax.axhline(target, color=PALETTE["thesisbrick"], linestyle="--", linewidth=1.2,
               label=f"{target:.0%} target")
    if stop_k is not None:
        ax.axvline(stop_k, color=PALETTE["thesisslate"], linestyle=":", linewidth=1.2,
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

    Strip of per-iteration samples (jittered) with min, median and mean marked,
    making the rightward contamination and the robustness of the min explicit.
    """
    d = successful[successful["query_id"] == query_id]
    vals = d["elapsed_time"].dropna().values
    cfg = d["configuration"].iloc[0]
    color = style.color(cfg)
    lo, hi = np.percentile(vals, [0.1, 99.0])
    shown = vals[(vals >= lo) & (vals <= hi)]
    jitter = _RNG.uniform(-0.35, 0.35, size=len(shown))

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.scatter(shown, jitter, s=6, alpha=0.25, color=color, edgecolor="none")
    vmin, vmean, vmed = float(np.min(vals)), float(np.mean(vals)), float(np.median(vals))
    for x, lab, c, ls in [
        (vmin, f"minimum = {vmin:.4g} s", PALETTE["thesisslate"], "-"),
        (vmed, f"median = {vmed:.4g} s", PALETTE["thesissage"], "--"),
        (vmean, f"mean = {vmean:.4g} s", PALETTE["thesisbrick"], ":"),
    ]:
        ax.axvline(x, color=c, linestyle=ls, linewidth=1.6, label=lab)
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Per-iteration elapsed time (s)")
    ax.set_title(
        f"Estimator choice — {style.workload_label(d['workload_type'].iloc[0])}, "
        f"{style.label(cfg)}"
    )
    ax.legend(fontsize=8, loc="upper right")
    contamination = (vmean - vmin) / vmin if vmin > 0 else np.nan
    _save_summary(
        pd.DataFrame([{"query_id": query_id, "min": vmin, "median": vmed,
                       "mean": vmean, "mean_over_min": vmean / vmin,
                       "contamination_pct": 100 * contamination}]),
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
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter([i], [sub["cv"].median()], marker="_", s=520,
                   color=shade(style.color(cfg), 0.25), zorder=4, linewidth=2)
    ax.axhspan(0, band, color=tint(PALETTE["thesissage"], 0.55), zorder=0)
    ax.axhline(band, color=PALETTE["thesisbrick"], linestyle="--", linewidth=1.1,
               label=f"CV = {band:.2f}")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([style.label(c) for c in configs], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Coefficient of variation")
    ax.set_title(f"Run-to-run dispersion — {style.metric_label(metric)}")
    ax.legend(fontsize=8)
    _save_summary(cv, out_path)
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
                     edgecolor=color, linewidth=0.6, alpha=0.9, zorder=2)
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
        ax.plot([bx, bx], [p5, q1], color=shade(color, 0.2), linewidth=0.8, zorder=3)
        ax.plot([bx, bx], [q3, p95], color=shade(color, 0.2), linewidth=0.8, zorder=3)
        ax.add_patch(plt.Rectangle((bx - bw, q1), 2 * bw, q3 - q1, facecolor="white",
                                   edgecolor=shade(color, 0.2), linewidth=0.8, zorder=3))
        ax.plot([bx - bw, bx + bw], [med, med], color=shade(color, 0.35),
                linewidth=1.2, zorder=4)

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
                elinewidth=1.0, capsize=2.5, markeredgecolor="white",
                markeredgewidth=0.5, zorder=6)
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
                           linewidth=0.6, zorder=3)
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
                   linewidth=0.4, label=_COST_LABELS[term])
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
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Operational cost per single-machine configuration", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)


def fig_cpu_decomposition(successful, workloads, configs, tiers, style, out_path):
    """Stacked user vs system CPU seconds (median) per config, workload x tier."""
    tiers = _tier_order(style, tiers)
    cells = [(wt, ds) for wt in workloads for ds in tiers]
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(cells) * len(configs)), 4.4))
    summary = []
    width = 0.8 / max(len(configs), 1)
    x = np.arange(len(cells))
    for j, cf in enumerate(configs):
        user, syst = [], []
        for wt, ds in cells:
            v = successful[(successful["workload_type"] == wt)
                           & (successful["dataset_size"] == ds)
                           & (successful["configuration"] == cf)]
            u = v["cpu_time_user_seconds"].dropna().values
            s = v["cpu_time_system_seconds"].dropna().values
            uu = float(np.median(u)) if len(u) else 0.0
            ss = float(np.median(s)) if len(s) else 0.0
            user.append(uu)
            syst.append(ss)
            if len(u) or len(s):
                summary.append({"workload_type": wt, "dataset_size": ds, "configuration": cf,
                                "cpu_user_s": uu, "cpu_system_s": ss})
        off = (j - len(configs) / 2 + 0.5) * width
        base = style.color(cf)
        ax.bar(x + off, user, width * 0.9, color=base, edgecolor="white", linewidth=0.4,
               label=f"{style.label(cf)} — user" if j == 0 else None)
        ax.bar(x + off, syst, width * 0.9, bottom=user, color=tint(base, 0.55),
               edgecolor="white", linewidth=0.4,
               label=f"{style.label(cf)} — system" if j == 0 else None)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{style.workload_label(wt)[:10]}\n{ds}" for wt, ds in cells],
                       rotation=0, fontsize=7)
    ax.set_ylabel("CPU time (s, log)")
    # legend: config colors + user/system shade convention
    handles = [plt.Rectangle((0, 0), 1, 1, color=style.color(c)) for c in configs]
    handles += [plt.Rectangle((0, 0), 1, 1, color=PALETTE["thesisgray"]),
                plt.Rectangle((0, 0), 1, 1, color=tint(PALETTE["thesisgray"], 0.55))]
    labels = [style.label(c) for c in configs] + ["user (solid)", "system (light)"]
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
        ax.step(vs, y, where="post", color=color, linewidth=1.7,
                label=style.label(cf), zorder=3)
        qs = np.percentile(v, [100 * lv for lv in levels])
        ax.scatter(qs, levels, color=shade(color, 0.2), s=24, zorder=5,
                   edgecolor="white", linewidth=0.5)
        summary.append({"configuration": cf, "n": int(len(v)),
                        "p50": float(qs[0]), "p95": float(qs[1]), "p99": float(qs[2])})
    for lv in levels:
        ax.axhline(lv, color=PALETTE["thesislight"], linewidth=0.8, zorder=0)
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
    ax.plot(ideal, ideal / w0, linestyle="--", color=PALETTE["thesisslate"], linewidth=1.1,
            label="ideal (linear)")
    ax.set_xlabel("Worker count")
    ax.set_ylabel("Speedup $S(n) = T_2 / T_n$")
    ax.set_title("Speedup of the distributed join")
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
    ax.axhline(1.0, linestyle="--", color=PALETTE["thesisslate"], linewidth=1.1,
               label="ideal (E = 1)")
    ax.set_xlabel("Worker count")
    ax.set_ylabel("Parallel efficiency $E(n)$")
    ax.set_title("Parallel efficiency of the distributed join")
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
        ax.axhline(row["point"], linestyle="--", linewidth=1.4,
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
               linewidth=0.4, label=lab)
        bottom += vals
    for xi in range(len(pdf)):
        row = pdf.iloc[xi]
        summary.append({"dataset_size": row["dataset_size"], "workers": int(row["worker_count"]),
                        "executor_run_s": row["executor_run_time_ms"] / 1000.0,
                        "driver_collection_s": row["driver_collection_time_ms"] / 1000.0})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Phase time (s)")
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
           linewidth=0.4, label="Shuffle read")
    ax.bar(x, wr, bottom=rd, width=0.7, color=tint(style.strategy_colors["broadcast"], 0.45),
           edgecolor="white", linewidth=0.4, label="Shuffle write")
    ax.yaxis.set_major_formatter(_bytes_fmt())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Shuffle bytes")
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
                   linewidth=0.5, zorder=3, label=f"{strat.capitalize()} ({ds})")
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
        ax.plot(fx, fy, color=PALETTE["thesisslate"], linewidth=1.3, linestyle="-",
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
                                       edgecolor="white", linewidth=2))
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
            ax.plot(xs, ys, marker="o", color=col, linewidth=1.8, markersize=7,
                    label=_name_label(style, cf))
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
    ax.hlines(0, min(ranks) - 0.3, max(ranks) + 0.3, color=PALETTE["thesisslate"], linewidth=1)
    for c, r in items:
        col = _name_color(style, c)
        ax.scatter(r, 0, s=60, color=col, zorder=4, edgecolor="white")
        ax.annotate(f"{_name_label(style, c)}\n({r:.2f})", xy=(r, 0), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=shade(col, 0.2))
    level = -0.04
    drawn = set()
    for a, b in nonsig_pairs:
        if a in mean_ranks and b in mean_ranks and (a, b) not in drawn:
            ax.plot([mean_ranks[a], mean_ranks[b]], [level, level],
                    color=PALETTE["thesisbrick"], linewidth=2.4, zorder=3)
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
        ax.scatter(a12, y, s=46, color=color, zorder=4, edgecolor="white")
        if not pd.isna(row.get("ci_low", np.nan)):
            ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=color, linewidth=1.4, zorder=3)
    # arcuri magnitude bands (reflected around 0.5)
    for thr, lab in [(0.56, "small"), (0.64, "medium"), (0.71, "large")]:
        for xx in (thr, 1 - thr):
            ax.axvline(xx, color=PALETTE["thesislight"], linewidth=0.8, zorder=0)
    ax.axvline(0.5, color=PALETTE["thesisslate"], linestyle="--", linewidth=1.2, zorder=1,
               label="no effect (0.5)")
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
        ax.axvline(xi, color=PALETTE["thesislight"], linewidth=1.0, zorder=0)
    for sysname in norm.index:
        color = _system_color(style, sysname)
        ax.plot(x, norm.loc[sysname, cols].values, marker="o", markersize=6,
                linewidth=2.0, color=color, markeredgecolor="white",
                markeredgewidth=0.6, zorder=3,
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
    median/min bars, network I/O and cost. One composite image (thesis float
    expects a single ``09-cell-grid.png``)."""
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
        times = [float(np.min(cell[cell["configuration"] == c]["elapsed_time"].dropna()))
                 if len(cell[cell["configuration"] == c]) else np.nan for c in present]
        ax.bar(x, times, width=0.6, color=[style.color(c) for c in present],
               edgecolor="white", linewidth=0.4)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([style.label(c).split(" ")[0] for c in present], rotation=30,
                           ha="right", fontsize=6.5)
        ax.set_title(f"{style.workload_label(wt)} ({ds})", fontsize=8.5)
        ax.set_ylabel("min time (s)", fontsize=7)
        for c, t in zip(present, times):
            summary.append({"workload_type": wt, "dataset_size": ds, "configuration": c,
                            "min_time_s": t})
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Per-cell summary grid — minimum wall-clock time by configuration",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    _save_summary(pd.DataFrame(summary), out_path)
    return _save(fig, out_path)

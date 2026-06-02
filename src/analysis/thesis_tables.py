"""Emit \\input-able LaTeX table body fragments matching the thesis float specs.

Each function computes the numbers from the cached frames and writes the body
rows (the ``& ... \\\\`` lines that slot between ``\\midrule`` and
``\\bottomrule`` of the inline ``tabular``/``longtable`` already defined in the
thesis) plus a 2-line ``%`` header naming the target float and column spec.
Hand-formatted (not pandas ``to_latex``) so the column count, alignment and
numeric formatting match the thesis exactly. Returns the underlying DataFrame
for inline display.

Thesis float column specs reproduced here:
  tab:results-achieved-n                       {@{}l l l r r r l@{}}   (7)
  tab:distributed-join-descriptive-statistics  {@{}l l r r r r r@{}}   (7)
  tab:geomean-normalized-performance           {@{}l r r r@{}}         (4)
  tab:appendix-pairwise-statistics  (longtable) {@{}l l l r r r@{}}    (6)
  tab:appendix-cardinality-agreement           {@{}l r r r r@{}}       (5)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.plotting.thesis_figures import estimate_ci

# ── label maps ─────────────────────────────────────────────────────────────

WL_SHORT = {
    "point-in-polygon-lookup": "Point-in-polygon",
    "knn-search": "kNN",
    "bbox-filtering": "Bounding-box",
    "national-scale-spatial-join": "National join",
}
CFG_LONG = {
    "duckdb": "DuckDB + GeoParquet",
    "postgis": "PostGIS",
    "local": "GeoPandas + Shapefile",
}
STOP_LABEL = {"precision": "precision target", "ceiling": "iteration ceiling",
              "fixed": "fixed budget", "failed": "failed", "timeout": "timeout"}

RQ1_WORKLOADS = ["point-in-polygon-lookup", "knn-search", "bbox-filtering"]
RQ1_CONFIGS = ["duckdb", "postgis", "local"]


# ── formatters ─────────────────────────────────────────────────────────────


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def _fmt_sec(v) -> str:
    if _nan(v):
        return "---"
    a = abs(v)
    if a == 0:
        return "0"
    if a < 1:
        return f"{v:.4f}"
    if a < 100:
        return f"{v:.3f}"
    return f"{v:.1f}"


def _fmt_cost(v) -> str:
    return "---" if _nan(v) else f"{v:.4f}"


def _fmt_cv(v) -> str:
    return "---" if _nan(v) else f"{v:.3f}"


def _fmt_pct(v) -> str:
    return "---" if _nan(v) else f"{v:.2f}"


def _fmt_int(v) -> str:
    return "---" if _nan(v) else str(int(round(v)))


def _fmt_p(v) -> str:
    if _nan(v):
        return "---"
    return "$<0.001$" if v < 0.001 else f"{v:.3f}"


def _fmt_a12(v, cat) -> str:
    if _nan(v):
        return "---"
    abbr = {"negligible": "n", "small": "S", "medium": "M", "large": "L"}.get(cat, "")
    return f"{v:.2f}~({abbr})" if abbr else f"{v:.2f}"


def _write(rows: list[str], out_path, comment: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(rows)
    out_path.write_text(f"% {comment}\n" + body + "\n")


# ── table builders ─────────────────────────────────────────────────────────


def emit_achieved_n(successful, metadata_df, experiments, out_path) -> pd.DataFrame:
    """tab:results-achieved-n — RQ1 cells: n, min(s), CI half-width, stop reason.

    Cells designed-and-run get a data row; cells absent from benchmarks.yml
    (e.g. Shapefile beyond small) get a ``\\multicolumn{4}{c}{did not run by
    design}`` span. CI half-width is the harness sequential-stopping value
    (metadata), averaged over passes.
    """
    designed = set()
    for qid, exp in experiments.items():
        ds = exp.get("dataset_size")
        for wt in RQ1_WORKLOADS:
            if qid.startswith(wt + "-") and qid.endswith("-" + ds):
                cfg = qid[len(wt) + 1: -(len(ds) + 1)]
                designed.add((wt, cfg, ds))

    meta = metadata_df.copy()
    rows, recs = [], []
    for cfg in RQ1_CONFIGS:
        for wt in RQ1_WORKLOADS:
            for ds in ["small", "large"]:
                qid = f"{wt}-{cfg}-{ds}"
                cfg_l, wt_l, ds_l = CFG_LONG[cfg], WL_SHORT[wt], ds.capitalize()
                if (wt, cfg, ds) not in designed:
                    rows.append(f"{cfg_l} & {wt_l} & {ds_l} & "
                                r"\multicolumn{4}{c}{did not run by design} \\")
                    recs.append({"configuration": cfg, "workload_type": wt,
                                 "dataset_size": ds, "achieved_n": np.nan,
                                 "min_s": np.nan, "ci_halfwidth_s": np.nan,
                                 "stop_reason": "did not run by design"})
                    continue
                samp = successful[successful["query_id"] == qid]["elapsed_time"].dropna().values
                md = meta[meta["query_id"] == qid]
                n = int(md["achieved_iterations"].sum()) if len(md) else len(samp)
                ci = float(md["ci_half_width_seconds"].mean()) if len(md) else np.nan
                stop = (md["stop_reason"].mode().iloc[0] if len(md) and not md["stop_reason"].isna().all()
                        else "")
                mn = float(np.min(samp)) if len(samp) else np.nan
                rows.append(f"{cfg_l} & {wt_l} & {ds_l} & {_fmt_int(n)} & "
                            f"{_fmt_sec(mn)} & {_fmt_sec(ci)} & {STOP_LABEL.get(stop, stop)} \\\\")
                recs.append({"configuration": cfg, "workload_type": wt, "dataset_size": ds,
                             "achieved_n": n, "min_s": mn, "ci_halfwidth_s": ci,
                             "stop_reason": stop})
    _write(rows, out_path,
           "tab:results-achieved-n  cols {@{}l l l r r r l@{}}  (Config, Workload, "
           "Tier, Achieved n, Min s, CI half-width, Stop reason)")
    return pd.DataFrame(recs)


def emit_summary_statistics(successful, experiments, out_path) -> pd.DataFrame:
    """tab:appendix-summary-statistics — fuller dispersion companion to the lean
    body achieved-n table (``tab:results-achieved-n``).

    One row per single-machine (config x workload x tier) cell that ran: achieved
    n, minimum, median, IQR, CV, and the relative bootstrap-CI half-width (%).
    Dispersion is reported as IQR and CV rather than the standard deviation: the
    per-iteration timings are right-skewed and one-sidedly contaminated, so a
    symmetric std would overstate and misplace the spread. IQR is the single
    Q3 - Q1 width (not the Q1--Q3 pair) so each dispersion measure is one
    siunitx ``S`` column. Rows are grouped by workload (an ``\\addlinespace``
    separates groups). The CI half-width uses the same minimum estimator and
    percentile bootstrap as the figures (``estimate_ci``).
    Scope mirrors the achieved-n table (the three single-machine engines over the
    three single-machine query patterns); distributed-join dispersion lives in
    Table~\\ref{tab:distributed-join-descriptive-statistics}.
    """
    rows, recs = [], []
    for wi, wt in enumerate(RQ1_WORKLOADS):
        if wi:
            rows.append(r"\addlinespace")
        for cfg in RQ1_CONFIGS:
            for ds in ["small", "large"]:
                qid = f"{wt}-{cfg}-{ds}"
                if qid not in experiments:
                    continue  # did not run by design (e.g. Shapefile beyond small)
                v = successful[successful["query_id"] == qid]["elapsed_time"].dropna().values
                if len(v) == 0:
                    continue
                mn = float(np.min(v))
                med = float(np.median(v))
                iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
                cv = float(np.std(v, ddof=1) / np.mean(v)) if np.mean(v) > 0 else np.nan
                point, lo, hi = estimate_ci(v, kind="min")
                ci_pct = 100.0 * ((hi - lo) / 2.0) / point if point > 0 else np.nan
                rows.append(
                    f"{WL_SHORT[wt]} & {CFG_LONG[cfg]} & {ds.capitalize()} & "
                    f"{len(v)} & {_fmt_sec(mn)} & {_fmt_sec(med)} & {_fmt_sec(iqr)} & "
                    f"{_fmt_cv(cv)} & {_fmt_pct(ci_pct)} \\\\")
                recs.append({"workload_type": wt, "configuration": cfg, "dataset_size": ds,
                             "n": int(len(v)), "min_s": mn, "median_s": med, "iqr_s": iqr,
                             "cv": cv, "ci_halfwidth_pct": ci_pct})
    _write(rows, out_path,
           "tab:appendix-summary-statistics  cols {@{}l l l S S S S S S@{}}  "
           "(Workload, Config, Tier, n, Min s, Median s, IQR s, CV, CI half-width %); "
           "grouped by workload. NO float exists yet -- see report.")
    return pd.DataFrame(recs)


def emit_distributed_descriptive(successful, metadata_df, cost_summary, experiments,
                                 out_path) -> pd.DataFrame:
    """tab:distributed-join-descriptive-statistics — Sedona cells per (tier, strategy,
    workers): n, min(s), bootstrap CI half-width, total cost. Designed-but-failed
    cells (executor OOM) get a ``\\multicolumn{4}{c}{...}`` span."""
    import re
    cs = cost_summary.reset_index() if cost_summary is not None and len(cost_summary) else None

    def cost_of(wt, ds, cfg):
        if cs is None:
            return np.nan
        r = cs[(cs["workload_type"] == wt) & (cs["dataset_size"] == ds)
               & (cs["configuration"] == cfg)]
        return float(r["total_cost"].iloc[0]) if len(r) else np.nan

    wt = "national-scale-spatial-join"
    tiers = ["small", "medium", "large"]
    rows, recs = [], []
    for strat in ["broadcast", "partitioned"]:
        for ds in tiers:
            for w in [2, 4, 8, 12, 16]:
                cfg = f"databricks-{strat}-{w}-nodes"
                qid = f"{wt}-{cfg}-{ds}"
                exp = experiments.get(qid)
                if exp is None:
                    continue
                samp = successful[successful["query_id"] == qid]["elapsed_time"].dropna().values
                if len(samp) == 0:
                    skip = exp.get("skip", "failed")
                    reason = "failed (executor OOM)" if skip == "failed" else f"{skip}"
                    rows.append(f"{ds.capitalize()} & {strat.capitalize()} & {w} & "
                                rf"\multicolumn{{4}}{{c}}{{{reason}}} \\")
                    recs.append({"dataset_size": ds, "strategy": strat, "workers": w,
                                 "achieved_n": 0, "min_s": np.nan, "ci_halfwidth_s": np.nan,
                                 "cost_usd": np.nan, "status": reason})
                    continue
                mn, lo, hi = estimate_ci(samp, kind="min")
                ci = (hi - lo) / 2
                cost = cost_of(wt, ds, cfg)
                n = len(samp)
                rows.append(f"{ds.capitalize()} & {strat.capitalize()} & {w} & {n} & "
                            f"{_fmt_sec(mn)} & {_fmt_sec(ci)} & {_fmt_cost(cost)} \\\\")
                recs.append({"dataset_size": ds, "strategy": strat, "workers": w,
                             "achieved_n": n, "min_s": mn, "ci_halfwidth_s": ci,
                             "cost_usd": cost, "status": "ran"})
    _write(rows, out_path,
           "tab:distributed-join-descriptive-statistics  cols {@{}l l r r r r r@{}}  "
           "(Tier, Strategy, Workers, Achieved n, Min s, CI half-width, Cost USD)")
    return pd.DataFrame(recs)


def emit_geomean(geomean_df, out_path) -> pd.DataFrame:
    """tab:geomean-normalized-performance — one row per system, geomean slowdown
    vs best for Time / Bytes / Cost (1.00 = best)."""
    from src.plotting.thesis_figures import SYSTEM_LABEL, SYSTEM_ORDER
    rows = []
    for sysname in SYSTEM_ORDER:
        if sysname not in geomean_df.index:
            continue
        r = geomean_df.loc[sysname]
        def g(col):
            v = r.get(col, np.nan)
            return "---" if _nan(v) else f"{v:.2f}"
        rows.append(f"{SYSTEM_LABEL[sysname]} & {g('Time')} & {g('Cost')} \\\\")
    _write(rows, out_path,
           "tab:geomean-normalized-performance  cols {@{}l r r@{}}  "
           "(Configuration, Time, Cost) -- geomean slowdown vs best, 1.00=best")
    return geomean_df


def emit_pairwise(pairwise_agg, out_path) -> pd.DataFrame:
    """tab:appendix-pairwise-statistics (longtable) — one row per (comparison,
    workload x tier, dimension): W, Holm-adjusted p, A12 with band."""
    rows = []
    for _, r in pairwise_agg.iterrows():
        rows.append(f"{r['comparison']} & {r['wt_tier']} & {r['dimension']} & "
                    f"{_fmt_int(r['W'])} & {_fmt_p(r['p_holm'])} & "
                    f"{_fmt_a12(r['a12'], r['a12_cat'])} \\\\")
    _write(rows, out_path,
           "tab:appendix-pairwise-statistics (longtable)  cols {@{}l l l r r r@{}}  "
           "(Comparison, Workload x tier, Dimension, W, p_Holm, A12)")
    return pairwise_agg


def emit_cardinality(card_df, out_path) -> pd.DataFrame:
    """tab:appendix-cardinality-agreement — returned row counts per (workload x tier)
    across DuckDB / PostGIS / Shapefile, with max relative discrepancy. Cells that
    did not run are blank; Shapefile non-reporting (-1) shown as n/r."""
    rows = []
    for _, r in card_df.iterrows():
        def cell(v):
            if _nan(v):
                return ""
            if v < 0:
                return "n/r"
            return _fmt_int(v)
        disc = "" if _nan(r["max_rel_disc"]) else f"{r['max_rel_disc']:.2%}".replace("%", r"\%")
        rows.append(f"{r['wt_tier']} & {cell(r['duckdb'])} & {cell(r['postgis'])} & "
                    f"{cell(r['local'])} & {disc} \\\\")
    _write(rows, out_path,
           "tab:appendix-cardinality-agreement  cols {@{}l r r r r@{}}  "
           "(Workload x tier, DuckDB, PostGIS, Shapefile, Max rel disc)")
    return card_df

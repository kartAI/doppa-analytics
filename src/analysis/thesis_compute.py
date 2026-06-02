"""Computation helpers feeding the thesis figures/tables.

Keeps the notebooks thin: each notebook loads once via :func:`load_all` (reads
the already-materialized DuckDB cache tables directly through ``load_cached`` —
NO Azure glob) and calls the ``rq2_*`` / ``rq3_*`` builders here.

Method conventions: elapsed_time uses the minimum estimator; network bytes use
the median; pairwise/effect-size/omnibus reuse ``src.analysis.stats``.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

from src.analysis.loading import (
    enrich_costs,
    enrich_samples,
    extract_strategy,
    extract_worker_count,
    load_cached,
    load_experiments,
    open_cache,
)
from src.analysis.stats import classify_a12, holm_bonferroni, pairwise_comparison
from src.analysis.thesis_tables import WL_SHORT
from src.plotting.thesis_figures import SYSTEM_ORDER, system_of

ITERATION_CEILINGS = {
    "point-in-polygon-lookup": 2500,
    "knn-search": 4000,
    "bbox-filtering": 900,
    "national-scale-spatial-join": 5,
}
WORKLOAD_TYPES = sorted(ITERATION_CEILINGS.keys(), key=len, reverse=True)
RQ2_WORKLOAD = "national-scale-spatial-join"
PRIMARY_METRICS = ["elapsed_time", "network_bytes_received", "network_bytes_sent"]
_DIM = {"elapsed_time": "Time", "network_bytes_received": "Bytes (recv)",
        "network_bytes_sent": "Bytes (sent)"}
_CMP_SHORT = {"duckdb": "DuckDB", "postgis": "PostGIS", "local": "Shapefile"}


def _cmp_short(c: str) -> str:
    if c.startswith("databricks"):
        return "Sedona " + c.replace("databricks-", "").replace("-nodes", "N")
    return _CMP_SHORT.get(c, c)


# ── data load (cache-only, no Azure) ───────────────────────────────────────


def load_all(cache_path, benchmarks_yml, run_id):
    """Open the cache and read the materialized tables directly (no remote glob)."""
    db = open_cache(cache_path)
    experiments = load_experiments(Path(benchmarks_yml))
    safe = re.sub(r"[^0-9A-Za-z]", "_", run_id)

    def cached(kind):
        return load_cached(db, kind, run_id, sql=f'SELECT * FROM "{kind}_{safe}"')

    samples_df = enrich_samples(cached("samples"), experiments, WORKLOAD_TYPES, ITERATION_CEILINGS)
    metadata_df = cached("metadata")
    costs_df = enrich_costs(cached("costs"), experiments, WORKLOAD_TYPES)
    successful = samples_df[samples_df["status"] == "success"].copy()
    db.close()
    return experiments, samples_df, metadata_df, costs_df, successful


# ── RQ2 distributed ────────────────────────────────────────────────────────


def rq2_scaling(successful):
    d = successful[(successful["workload_type"] == RQ2_WORKLOAD)
                   & successful["configuration"].str.startswith("databricks")]
    rows = []
    for (cfg, ds), g in d.groupby(["configuration", "dataset_size"]):
        v = g["elapsed_time"].dropna().values
        if len(v):
            rows.append({"strategy": extract_strategy(cfg),
                         "worker_count": extract_worker_count(cfg),
                         "dataset_size": ds, "point": float(np.min(v))})
    return pd.DataFrame(rows)


def rq2_single_node(successful):
    d = successful[(successful["workload_type"] == RQ2_WORKLOAD)
                   & successful["configuration"].isin(["duckdb", "postgis"])]
    rows = []
    for (cfg, ds), g in d.groupby(["configuration", "dataset_size"]):
        v = g["elapsed_time"].dropna().values
        if len(v):
            rows.append({"configuration": cfg, "dataset_size": ds, "point": float(np.min(v))})
    return pd.DataFrame(rows)


def rq2_failed(successful, experiments):
    ran = set(successful["query_id"].unique())
    rows = []
    for qid, exp in experiments.items():
        if RQ2_WORKLOAD in qid and "databricks" in qid and qid not in ran:
            rows.append({"strategy": extract_strategy(qid),
                         "worker_count": extract_worker_count(qid),
                         "dataset_size": exp["dataset_size"], "skip": exp.get("skip", "")})
    return pd.DataFrame(rows)


def rq2_phase(successful):
    d = successful[successful["configuration"].str.startswith("databricks-broadcast")].dropna(
        subset=["executor_run_time_ms"])
    rows = []
    for (cfg, ds), g in d.groupby(["configuration", "dataset_size"]):
        rows.append({"worker_count": extract_worker_count(cfg), "dataset_size": ds,
                     "executor_run_time_ms": float(g["executor_run_time_ms"].median()),
                     "driver_collection_time_ms": float(g["driver_collection_time_ms"].median()),
                     "shuffle_read_bytes": float(g["shuffle_read_bytes"].median()),
                     "shuffle_write_bytes": float(g["shuffle_write_bytes"].median())})
    return pd.DataFrame(rows)


def rq2_pareto(successful, cost_summary):
    cs = cost_summary.reset_index()
    d = successful[(successful["workload_type"] == RQ2_WORKLOAD)
                   & successful["configuration"].str.startswith("databricks")]
    rows = []
    for (cfg, ds), g in d.groupby(["configuration", "dataset_size"]):
        v = g["elapsed_time"].dropna().values
        if len(v) == 0:
            continue
        r = cs[(cs["workload_type"] == RQ2_WORKLOAD) & (cs["dataset_size"] == ds)
               & (cs["configuration"] == cfg)]
        rows.append({"strategy": extract_strategy(cfg), "dataset_size": ds,
                     "worker_count": extract_worker_count(cfg),
                     "cost": float(r["total_cost"].iloc[0]) if len(r) else np.nan,
                     "time": float(np.min(v))})
    return pd.DataFrame(rows)


# ── RQ3 synthesis ───────────────────────────────────────────────────────────


def rq3_system_values(successful, cost_summary):
    """Per (workload, tier, system): Time (min), Bytes (median recv), Cost (min total)."""
    cs = cost_summary.reset_index()
    succ = successful.copy()
    succ["system"] = succ["configuration"].map(system_of)
    rows = []
    for (wt, ds), g in succ.groupby(["workload_type", "dataset_size"]):
        for sysname, gs in g.groupby("system"):
            t = gs["elapsed_time"].dropna().values
            b = gs["network_bytes_received"].dropna().values
            costs = []
            for cfg in gs["configuration"].unique():
                r = cs[(cs["workload_type"] == wt) & (cs["dataset_size"] == ds)
                       & (cs["configuration"] == cfg)]
                if len(r):
                    costs.append(float(r["total_cost"].iloc[0]))
            rows.append({"workload_type": wt, "dataset_size": ds, "system": sysname,
                         "Time": float(np.min(t)) if len(t) else np.nan,
                         "Bytes": float(np.median(b)) if len(b) else np.nan,
                         "Cost": min(costs) if costs else np.nan})
    return pd.DataFrame(rows)


def rq3_geomean(sysvals):
    """Geometric mean of per-cell normalized value (vs smallest positive) per system."""
    outcomes = ["Time", "Cost"]
    out = {}
    for sysname in SYSTEM_ORDER:
        row = {}
        for oc in outcomes:
            ratios = []
            for (_, _), cell in sysvals.groupby(["workload_type", "dataset_size"]):
                vals = cell.set_index("system")[oc].to_dict()
                if sysname not in vals or pd.isna(vals[sysname]):
                    continue
                pos = [v for v in vals.values() if not pd.isna(v) and v > 0]
                if not pos:
                    continue
                minpos = min(pos)
                v = vals[sysname]
                ratios.append(1.0 if (pd.isna(v) or v <= 0) else v / minpos)
            row[oc] = float(np.exp(np.mean(np.log(ratios)))) if ratios else np.nan
        out[sysname] = row
    return pd.DataFrame(out).T.reindex([s for s in SYSTEM_ORDER if s in out])


def rq3_winners(sysvals):
    """Winning system per (workload x tier) row x outcome (lowest value wins)."""
    order = []
    data = {}
    from src.plotting.style import StyleConfig  # for size order
    size_order = StyleConfig().size_order
    for (wt, ds), cell in sysvals.groupby(["workload_type", "dataset_size"]):
        label = f"{WL_SHORT[wt]} ({ds})"
        order.append((size_order.get(ds, 9), wt, label))
        d = cell.set_index("system")
        data[label] = {oc: (d[oc].dropna().idxmin() if d[oc].notna().any() else "")
                       for oc in ["Time", "Cost"]}
    labels = [lab for _, _, lab in sorted(order)]
    return pd.DataFrame(data).T.reindex(labels)


def _rep_config(successful, wt, ds, system, metric, kind):
    """Configuration within *system* achieving that system's best (lowest) value
    in the cell — the representative config for a system-level significance lookup
    (trivially the config itself for single-node systems)."""
    cell = successful[(successful["workload_type"] == wt)
                      & (successful["dataset_size"] == ds)]
    cell = cell[cell["configuration"].map(system_of) == system]
    best_cfg, best_val = None, np.inf
    for cfg, g in cell.groupby("configuration"):
        v = g[metric].dropna().values
        if len(v) == 0:
            continue
        val = float(np.min(v)) if kind == "min" else float(np.median(v))
        if val < best_val:
            best_cfg, best_val = cfg, val
    return best_cfg


def rq3_winners_export(sysvals, pooled, successful, tie_threshold=0.05):
    """Decision-quadrant support data (NOT a figure): per (workload x tier x
    outcome) the winning system, the runner-up, the relative margin, and whether
    the win is statistically backed.

    Outcomes: Time (min elapsed), Bytes (median received), Cost (min modeled
    total); lower wins on each. Significance for Time/Bytes is read off the
    pooled Holm-adjusted Wilcoxon table (``pooled``) between the winner's and
    runner-up's representative configurations on that metric in that cell. Cost is
    a single modeled value with no inferential test, so its decision is by margin
    only. ``decision`` is one of: ``significant`` (Holm-significant win),
    ``n.s.`` (tested, not significant — a statistical tie), ``tie`` (margin below
    ``tie_threshold`` and not significance-tested, e.g. Cost or an untestable
    cell), ``clear`` (Cost win above the margin threshold), ``single-config``
    (only one system ran the cell), or ``untested`` (margin above threshold but no
    pairwise test available). This is the table the hand-drawn Discussion decision
    quadrant should be redrawn from.
    """
    from src.plotting.style import StyleConfig
    size_order = StyleConfig().size_order
    metric_of = {"Time": ("elapsed_time", "min"),
                 "Bytes": ("network_bytes_received", "median"),
                 "Cost": (None, None)}

    def holm_sig(wt, ds, metric, ca, cb):
        if pooled is None or pooled.empty or ca is None or cb is None:
            return None, np.nan
        a, b = sorted([ca, cb])
        r = pooled[(pooled["workload_type"] == wt) & (pooled["dataset_size"] == ds)
                   & (pooled["metric"] == metric) & (pooled["config_a"] == a)
                   & (pooled["config_b"] == b)]
        if len(r):
            return bool(r["significant"].iloc[0]), float(r["p_value_holm"].iloc[0])
        return None, np.nan

    rows = []
    for (wt, ds), cell in sysvals.groupby(["workload_type", "dataset_size"]):
        d = cell.set_index("system")
        for oc in ["Time", "Cost"]:
            s = d[oc].dropna().sort_values()
            rec = {"workload_type": wt, "tier": ds, "wt_tier": f"{WL_SHORT[wt]} ({ds})",
                   "outcome": oc, "winner": "", "winner_value": np.nan,
                   "runner_up": "", "runner_up_value": np.nan, "rel_margin_pct": np.nan,
                   "holm_p": np.nan, "decision": ""}
            if len(s) == 0:
                rec["decision"] = "none"
                rows.append(rec)
                continue
            winner = s.index[0]
            rec["winner"], rec["winner_value"] = winner, float(s.iloc[0])
            if len(s) == 1:
                rec["decision"] = "single-config"
                rows.append(rec)
                continue
            runner = s.index[1]
            rec["runner_up"], rec["runner_up_value"] = runner, float(s.iloc[1])
            margin = (s.iloc[1] - s.iloc[0]) / s.iloc[0] if s.iloc[0] > 0 else np.nan
            rec["rel_margin_pct"] = 100.0 * margin if not pd.isna(margin) else np.nan
            metric, kind = metric_of[oc]
            if metric is None:  # Cost: modeled, margin-only
                rec["decision"] = "tie" if (not pd.isna(margin) and margin < tie_threshold) else "clear"
            else:
                ca = _rep_config(successful, wt, ds, winner, metric, kind)
                cb = _rep_config(successful, wt, ds, runner, metric, kind)
                sig, p = holm_sig(wt, ds, metric, ca, cb)
                rec["holm_p"] = p
                if sig is True:
                    rec["decision"] = "significant"
                elif sig is False:
                    rec["decision"] = "n.s."
                else:  # no pairwise test available
                    rec["decision"] = "tie" if (not pd.isna(margin) and margin < tie_threshold) else "untested"
            rows.append(rec)
    out = pd.DataFrame(rows)
    out["_o"] = out["tier"].map(lambda t: size_order.get(t, 9))
    return out.sort_values(["workload_type", "_o", "outcome"]).drop(columns="_o").reset_index(drop=True)


def rq3_ranks(sysvals):
    rows = []
    for (wt, ds), cell in sysvals.groupby(["workload_type", "dataset_size"]):
        s = cell.dropna(subset=["Time"]).sort_values("Time")
        for rank, (_, r) in enumerate(s.iterrows(), 1):
            rows.append({"workload_type": wt, "dataset_size": ds,
                         "configuration": r["system"], "rank": rank})
    return pd.DataFrame(rows)


def rq3_parallel_axes(geomean, ranks):
    """Assemble one row per system for the parallel-coordinates figure.

    Reuses the already-computed RQ3 outputs: the geometric-mean normalized
    Time/Bytes/Cost (``rq3_geomean``, where 1.0 = best on that dimension) plus
    the mean Rank across cells (``rq3_ranks``). Lower is better on every axis.
    """
    mean_rank = ranks.groupby("configuration")["rank"].mean()
    rows = []
    for sysname in geomean.index:
        rows.append({
            "system": sysname,
            "Time": float(geomean.loc[sysname, "Time"]),
            "Cost": float(geomean.loc[sysname, "Cost"]),
            "Rank": float(mean_rank.get(sysname, np.nan)),
        })
    return pd.DataFrame(rows).set_index("system")


def pooled_pairwise_table(successful, metrics=PRIMARY_METRICS):
    """Pairwise stats on iterations POOLED across passes (one row per
    comparison/cell/metric). Pooling is required because national-scale cells
    have only ~3 iterations per pass (< the n>=6 Wilcoxon floor) but ~9 pooled.
    Reuses ``pairwise_comparison`` / ``holm_bonferroni`` / Friedman; Holm is
    applied within each (workload, tier, metric) family."""
    rows = []
    for (wt, ds), batch in successful.groupby(["workload_type", "dataset_size"]):
        for metric in metrics:
            groups = {}
            for cfg, g in batch.groupby("configuration"):
                v = g.sort_values(["benchmark_run", "local_iteration"])[metric].dropna().values
                if len(v) >= 6:
                    groups[cfg] = v
            names = sorted(groups)
            if len(names) < 2:
                continue
            fstat, fp = np.nan, np.nan
            if len(names) >= 3:
                nmin = min(len(groups[n]) for n in names)
                try:
                    fstat, fp = friedmanchisquare(*[groups[n][:nmin] for n in names])
                except Exception:
                    pass
            fam = []
            for a, b in combinations(names, 2):
                n = min(len(groups[a]), len(groups[b]))
                if n < 6:
                    continue
                res = pairwise_comparison(groups[a][:n], groups[b][:n])
                res.update({"workload_type": wt, "dataset_size": ds, "config_a": a,
                            "config_b": b, "metric": metric, "friedman_stat": float(fstat),
                            "friedman_p": float(fp)})
                fam.append(res)
            if fam:
                pv = np.array([r["p_value"] for r in fam])
                rej, corr = holm_bonferroni(pv)
                for i, r in enumerate(fam):
                    r["p_value_holm"] = float(corr[i])
                    r["significant"] = bool(rej[i])
                rows.extend(fam)
    return pd.DataFrame(rows)


def format_pairwise_for_table(pooled):
    if pooled is None or pooled.empty:
        return pd.DataFrame()
    sz = {"small": 0, "medium": 1, "large": 2}
    df = pooled.copy()
    df["W"] = df["wilcoxon_stat"]
    df["p_holm"] = df["p_value_holm"]
    df["a12_cat"] = df["a12_category"]
    df["dimension"] = df["metric"].map(_DIM)
    df["comparison"] = df.apply(
        lambda r: f"{_cmp_short(r.config_a)} vs.\\ {_cmp_short(r.config_b)}", axis=1)
    df["wt_tier"] = df.apply(lambda r: f"{WL_SHORT[r.workload_type]}, {r.dataset_size}", axis=1)
    df["_o"] = df["dataset_size"].map(sz)
    return df.sort_values(["workload_type", "_o", "config_a", "config_b", "metric"]).drop(columns="_o")


def rq3_rank_matrix(successful, configs=("duckdb", "postgis", "local"),
                    metric="elapsed_time", kind="min"):
    """Complete-block rank matrix shared by the RQ3 omnibus and the cliques figure.

    Blocks are single-machine workload×tier cells; treatments are the
    configurations; each entry is the per-cell point estimate (minimum for
    run-time, the elapsed-time convention). Friedman's omnibus needs *complete*
    blocks — every treatment present in every block — but the Shapefile/local
    path ran the small tier only, so the only design with three or more
    treatments that is complete is {DuckDB, PostGIS, Shapefile} over the
    small-tier RQ1 cells. Cells where a treatment is absent are recorded as
    excluded (with the reason), never silently dropped.

    Returns ``(matrix, meta)``: ``matrix`` is a complete-block DataFrame with
    ``workload_type``/``dataset_size`` plus one column of point estimates per
    config; ``meta`` documents the in/out configs and cells.
    """
    configs = list(configs)
    stat = np.min if kind == "min" else np.median
    rq1 = successful[successful["workload_type"] != RQ2_WORKLOAD]
    cell_vals = {}
    for (wt, ds), g in rq1.groupby(["workload_type", "dataset_size"]):
        vals = {}
        for c in configs:
            v = g[g["configuration"] == c][metric].dropna().values
            if len(v):
                vals[c] = float(stat(v))
        cell_vals[(wt, ds)] = vals
    complete = {cell: v for cell, v in cell_vals.items() if all(c in v for c in configs)}
    incomplete = {cell: sorted(v) for cell, v in cell_vals.items() if cell not in complete}
    matrix = pd.DataFrame([{"workload_type": wt, "dataset_size": ds, **v}
                           for (wt, ds), v in complete.items()])
    meta = {
        "metric": metric,
        "estimator": kind,
        "blocking": "workload×tier cells",
        "included_configs": configs,
        "included_cells": [f"{WL_SHORT[wt]}, {ds}" for (wt, ds) in complete],
        "excluded_cells": {f"{WL_SHORT[wt]}, {ds}":
                           f"incomplete block (ran: {', '.join(ran)})"
                           for (wt, ds), ran in incomplete.items()},
        "scope": ("single-machine RQ1 cells only; the national-scale join and "
                  "Apache Sedona are excluded because Sedona ran the join alone, "
                  "so no complete ≥3-treatment block spans them with the "
                  "single-machine engines"),
    }
    return matrix, meta


def rq3_friedman_omnibus(matrix, meta):
    """Friedman omnibus across workload×tier cells (treatments = configurations).

    ``scipy.stats.friedmanchisquare`` ranks the configurations within each
    complete block (cell) by their per-cell point estimate. ``df`` = k−1 and
    ``n_blocks`` is the number of cells (a small number — the units across which
    RQ3 judges ranking consistency), *not* an iteration count. The returned dict
    is written verbatim to ``tables/06-friedman-omnibus-values.json``.
    """
    configs = meta["included_configs"]
    if matrix is None or matrix.empty or len(configs) < 3 or len(matrix) < 2:
        return {}
    chi2, p = friedmanchisquare(*[matrix[c].values for c in configs])
    n_blocks = int(len(matrix))
    out = {
        "metric": meta["metric"],
        "chi2": float(chi2),
        "p": float(p),
        "df": len(configs) - 1,
        "k_configs": len(configs),
        "n_blocks": n_blocks,
        "blocking": meta["blocking"],
        "included_configs": configs,
        "included_cells": meta["included_cells"],
        "excluded_cells": meta["excluded_cells"],
        "scope": meta["scope"],
    }
    if n_blocks < 5:
        out["note"] = (
            f"Complete-block restriction leaves only {n_blocks} cells "
            f"(Shapefile ran the small tier only), so the omnibus is low-powered "
            f"and spans query patterns at the small tier rather than across tiers; "
            f"reported as a limitation rather than overclaimed. Tier-wise "
            f"consistency is carried by the ranking-stability and rank-portability "
            f"figures.")
    return out


def rq1_cliques(successful, pooled, configs=("duckdb", "postgis", "local")):
    """Mean rank + Holm-non-significant pairs for the cliques figure.

    Uses the same complete-block cells and configurations as the Friedman
    omnibus (:func:`rq3_rank_matrix`), so the omnibus and its post-hoc cliques
    describe the same comparison: identical configs, identical rank matrix. The
    non-significant connectors come from the Holm-adjusted Wilcoxon pairwise
    tests on those same cells (a more powerful per-iteration post-hoc than
    Nemenyi, by design — see ``fig_cliques``)."""
    configs = list(configs)
    matrix, _ = rq3_rank_matrix(successful, configs=configs)
    cell_keys = set()
    cell_ranks = {c: [] for c in configs}
    for _, row in matrix.iterrows():
        cell_keys.add((row["workload_type"], row["dataset_size"]))
        mins = {c: float(row[c]) for c in configs}
        for rank, (c, _) in enumerate(sorted(mins.items(), key=lambda kv: kv[1]), 1):
            cell_ranks[c].append(rank)
    mean_ranks = {c: float(np.mean(v)) for c, v in cell_ranks.items() if v}
    nonsig = []
    if pooled is not None and not pooled.empty:
        in_cells = pooled.apply(
            lambda r: (r["workload_type"], r["dataset_size"]) in cell_keys, axis=1)
        f = pooled[(pooled["metric"] == "elapsed_time") & in_cells]
        for (ca, cb), g in f.groupby(["config_a", "config_b"]):
            if ca in mean_ranks and cb in mean_ranks and g["p_value_holm"].median() >= 0.05:
                nonsig.append((ca, cb))
    return mean_ranks, nonsig


def a12_forest(pooled):
    """Key A12 comparisons: DuckDB-vs-PostGIS (RQ1) + Sedona-vs-single-node (RQ2)."""
    cols = ["label", "a12", "ci_low", "ci_high"]
    if pooled is None or pooled.empty:
        return pd.DataFrame(columns=cols)
    f = pooled[pooled["metric"] == "elapsed_time"]
    rows = []
    for _, r in f[(f["config_a"] == "duckdb") & (f["config_b"] == "postgis")].iterrows():
        rows.append({"label": f"DuckDB vs PostGIS — {WL_SHORT[r['workload_type']]} {r['dataset_size']}",
                     "a12": r["a12"], "ci_low": np.nan, "ci_high": np.nan})
    sed = "databricks-broadcast-8-nodes"
    for sn in ["duckdb", "postgis"]:
        a, b = sorted([sed, sn])
        for _, r in f[(f["config_a"] == a) & (f["config_b"] == b)
                      & (f["workload_type"] == RQ2_WORKLOAD)].iterrows():
            rows.append({"label": f"Sedona(8N) vs {_CMP_SHORT[sn]} — National {r['dataset_size']}",
                         "a12": r["a12"], "ci_low": np.nan, "ci_high": np.nan})
    return pd.DataFrame(rows, columns=cols)


def cardinality_agreement(successful):
    """Returned row count per (workload x tier) across DuckDB/PostGIS/Shapefile + max rel disc."""
    size_order = {"small": 0, "medium": 1, "large": 2}
    card = successful.groupby("query_id")["result_cardinality"].agg(
        lambda s: float(s.dropna().median()) if s.notna().any() else np.nan)
    cells = []
    for wt in ["point-in-polygon-lookup", "knn-search", "bbox-filtering"]:
        for ds in ["small", "large"]:
            cells.append((wt, ds))
    for ds in ["small", "medium"]:
        cells.append((RQ2_WORKLOAD, ds))
    rows = []
    for wt, ds in cells:
        rec = {"wt_tier": f"{WL_SHORT[wt]}, {ds}", "_w": wt, "_o": size_order[ds]}
        positives = []
        for cfg in ["duckdb", "postgis", "local"]:
            qid = f"{wt}-{cfg}-{ds}"
            v = float(card[qid]) if qid in card.index else np.nan
            rec[cfg] = v
            if not np.isnan(v) and v > 0:
                positives.append(v)
        rec["max_rel_disc"] = ((max(positives) - min(positives)) / min(positives)
                               if len(positives) >= 2 else np.nan)
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values(["_w", "_o"]).drop(columns=["_w", "_o"])
    return df


_FRIEDMAN_SUMMARY_REPLACED = (
    "friedman_summary() was removed: it ran Friedman on per-iteration timings "
    "within a single workload×tier cell (1282 iterations as blocks), which only "
    "tested whether configs differ within that one cell and was massively "
    "overpowered. RQ3's omnibus is rq3_friedman_omnibus(), blocked by "
    "workload×tier cell — see rq3_rank_matrix().")

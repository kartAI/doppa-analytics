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
    outcomes = ["Time", "Bytes", "Cost"]
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
                       for oc in ["Time", "Bytes", "Cost"]}
    labels = [lab for _, _, lab in sorted(order)]
    return pd.DataFrame(data).T.reindex(labels)


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
            "Bytes": float(geomean.loc[sysname, "Bytes"]),
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


def rq1_cliques(successful, pooled, configs=("duckdb", "postgis", "local")):
    """Mean rank (by min time over RQ1 cells) + Holm-non-significant pairs."""
    configs = list(configs)
    cell_ranks = {c: [] for c in configs}
    rq1 = successful[successful["workload_type"] != RQ2_WORKLOAD]
    for (wt, ds), g in rq1.groupby(["workload_type", "dataset_size"]):
        mins = {}
        for c in configs:
            v = g[g["configuration"] == c]["elapsed_time"].dropna().values
            if len(v):
                mins[c] = float(np.min(v))
        for rank, (c, _) in enumerate(sorted(mins.items(), key=lambda kv: kv[1]), 1):
            cell_ranks[c].append(rank)
    mean_ranks = {c: float(np.mean(v)) for c, v in cell_ranks.items() if v}
    nonsig = []
    if pooled is not None and not pooled.empty:
        f = pooled[(pooled["metric"] == "elapsed_time") & (pooled["workload_type"] != RQ2_WORKLOAD)]
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


def friedman_summary(pooled, workload="point-in-polygon-lookup", tier="small",
                     metric="elapsed_time"):
    """Representative Friedman omnibus (k configs as treatments) for the prose sentence."""
    if pooled is None or pooled.empty:
        return {}
    sub = pooled[(pooled["workload_type"] == workload) & (pooled["dataset_size"] == tier)
                 & (pooled["metric"] == metric)]
    if sub.empty or sub["friedman_stat"].isna().all():
        return {}
    k = len(set(sub["config_a"]).union(sub["config_b"]))
    return {"workload": workload, "tier": tier, "metric": metric,
            "chi2": float(sub["friedman_stat"].iloc[0]),
            "p": float(sub["friedman_p"].iloc[0]), "df": k - 1, "k_configs": k,
            "n_blocks": int(sub["n_paired"].median())}

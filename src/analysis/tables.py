from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

from .stats import (
    classify_a12,
    cross_pass_aggregation,
    descriptive_stats,
    holm_bonferroni,
    pairwise_comparison,
)


def build_descriptive_table(
    successful: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    rows = []
    for (qid, br), group in successful.groupby(["query_id", "benchmark_run"]):
        for metric in metrics:
            vals = group[metric].dropna().values
            if len(vals) == 0:
                continue
            stats = descriptive_stats(vals)
            stats["query_id"] = qid
            stats["benchmark_run"] = br
            stats["metric"] = metric
            rows.append(stats)

    table = pd.DataFrame(rows)
    if len(table) > 0:
        table = table.set_index(["query_id", "benchmark_run", "metric"]).sort_index()
    return table


def build_cross_pass_table(table1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (qid, metric), group in table1.groupby(level=["query_id", "metric"]):
        pass_medians = group["median"].values
        if len(pass_medians) < 2:
            continue
        agg = cross_pass_aggregation(pass_medians)
        agg["query_id"] = qid
        agg["metric"] = metric
        rows.append(agg)

    if rows:
        return pd.DataFrame(rows).set_index(["query_id", "metric"]).sort_index()

    table = pd.DataFrame(
        columns=[
            "headline_median",
            "pass_1_median",
            "pass_2_median",
            "pass_3_median",
            "pass_range",
            "pass_range_relative",
            "consistent",
        ]
    )
    table.index = pd.MultiIndex.from_tuples([], names=["query_id", "metric"])
    return table


def _get_batch_groups(
    successful: pd.DataFrame,
    wt: str,
    ds: str,
    br: int,
    metric: str,
) -> dict[str, np.ndarray]:
    mask = (
        (successful["workload_type"] == wt)
        & (successful["dataset_size"] == ds)
        & (successful["benchmark_run"] == br)
    )
    batch = successful[mask]
    groups = {}
    for config, g in batch.groupby("configuration"):
        vals = g.sort_values("local_iteration")[metric].dropna().values
        groups[config] = vals
    if not groups:
        return {}
    min_len = min(len(v) for v in groups.values())
    return {k: v[:min_len] for k, v in groups.items()}


def build_pairwise_table(
    successful: pd.DataFrame,
    primary_metrics: list[str],
) -> pd.DataFrame:
    rows = []
    wt_ds_pairs = (
        successful.groupby(["workload_type", "dataset_size"]).size().index.tolist()
    )

    for wt, ds in wt_ds_pairs:
        for br in sorted(successful["benchmark_run"].unique()):
            for metric in primary_metrics:
                groups = _get_batch_groups(successful, wt, ds, br, metric)
                if len(groups) < 2:
                    continue

                names = sorted(groups.keys())

                f_stat, f_p = np.nan, np.nan
                if len(names) >= 3:
                    n = min(len(groups[name]) for name in names)
                    try:
                        f_stat, f_p = friedmanchisquare(
                            *[groups[name][:n] for name in names]
                        )
                    except Exception:
                        pass

                pair_results = []
                for name_a, name_b in combinations(names, 2):
                    n = min(len(groups[name_a]), len(groups[name_b]))
                    if n < 6:
                        continue
                    result = pairwise_comparison(
                        groups[name_a][:n], groups[name_b][:n]
                    )
                    result.update(
                        {
                            "config_a": name_a,
                            "config_b": name_b,
                            "workload_type": wt,
                            "dataset_size": ds,
                            "benchmark_run": br,
                            "metric": metric,
                            "friedman_stat": float(f_stat),
                            "friedman_p": float(f_p),
                        }
                    )
                    pair_results.append(result)

                if pair_results:
                    p_vals = np.array([r["p_value"] for r in pair_results])
                    reject, corrected = holm_bonferroni(p_vals)
                    for i, r in enumerate(pair_results):
                        r["p_value_holm"] = float(corrected[i])
                        r["significant"] = bool(reject[i])
                    rows.extend(pair_results)

    table = pd.DataFrame(rows)
    if len(table) > 0:
        table = table.set_index(
            [
                "workload_type",
                "dataset_size",
                "config_a",
                "config_b",
                "benchmark_run",
                "metric",
            ]
        ).sort_index()
    return table


def build_consistency_table(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    if pairwise_df.empty:
        return pd.DataFrame()

    flat = pairwise_df.reset_index()

    def _check(group: pd.DataFrame) -> pd.Series:
        ratios = group["ratio_median"].values
        sig = group["significant"].values
        a12s = group["a12"].values

        direction_agrees = bool(
            all(r > 1 for r in ratios) or all(r < 1 for r in ratios)
        )
        all_significant = bool(np.all(sig))
        all_nontrivial = all(classify_a12(a) != "negligible" for a in a12s)

        return pd.Series(
            {
                "direction_consistent": direction_agrees,
                "all_significant": all_significant,
                "effect_size_consistent": all_nontrivial,
                "fully_consistent": direction_agrees
                and all_significant
                and all_nontrivial,
            }
        )

    table = (
        flat.groupby(
            ["workload_type", "dataset_size", "config_a", "config_b", "metric"]
        )
        .apply(_check, include_groups=False)
        .reset_index()
    )
    if len(table) > 0:
        table = table.set_index(
            ["workload_type", "dataset_size", "config_a", "config_b", "metric"]
        )
    return table


def build_scaling_table(
    successful: pd.DataFrame,
    size_order: dict[str, int],
) -> pd.DataFrame:
    rows = []
    for wt in sorted(successful["workload_type"].unique()):
        wt_data = successful[successful["workload_type"] == wt]
        sizes = sorted(
            wt_data["dataset_size"].unique(),
            key=lambda s: size_order.get(s, 99),
        )
        if len(sizes) < 2:
            continue

        configs = sorted(wt_data["configuration"].unique())
        median_map = {}
        for cfg in configs:
            for ds in sizes:
                vals = wt_data[
                    (wt_data["configuration"] == cfg)
                    & (wt_data["dataset_size"] == ds)
                ]["elapsed_time"].dropna().values
                if len(vals) > 0:
                    median_map[(cfg, ds)] = float(np.median(vals))

        base_size = sizes[0]
        for cfg in configs:
            base_med = median_map.get((cfg, base_size))
            if base_med is None:
                continue
            row = {
                "workload_type": wt,
                "configuration": cfg,
                f"median_{base_size}": base_med,
            }
            for ds in sizes[1:]:
                med = median_map.get((cfg, ds))
                if med is not None:
                    row[f"median_{ds}"] = med
                    row[f"ratio_{ds}_vs_{base_size}"] = (
                        med / base_med if base_med > 0 else np.nan
                    )
            rows.append(row)

    if rows:
        return pd.DataFrame(rows).set_index(["workload_type", "configuration"])
    return pd.DataFrame()


def build_rq1_ranking(
    rq1_comparisons: pd.DataFrame,
    rq1_consistency: pd.DataFrame,
    table1: pd.DataFrame,
    table2: pd.DataFrame,
) -> pd.DataFrame:
    def _get_headline_median(qid: str, metric: str) -> float | None:
        if len(table2) > 0 and (qid, metric) in table2.index:
            return table2.loc[(qid, metric), "headline_median"]
        mask = table1.index.get_level_values("query_id") == qid
        mask &= table1.index.get_level_values("metric") == metric
        subset = table1[mask]
        if len(subset) > 0:
            return float(subset["median"].median())
        return None

    rows = []
    for (wt, ds, metric), grp in rq1_comparisons.groupby(
        ["workload_type", "dataset_size", "metric"]
    ):
        configs = set(grp["config_a"].tolist() + grp["config_b"].tolist())
        config_medians = {}
        for cfg in configs:
            qid = f"{wt}-{cfg}-{ds}"
            med = _get_headline_median(qid, metric)
            if med is not None:
                config_medians[cfg] = med

        if not config_medians:
            continue

        fastest = min(config_medians, key=config_medians.get)
        cons_mask = (
            (rq1_consistency["workload_type"] == wt)
            & (rq1_consistency["dataset_size"] == ds)
            & (rq1_consistency["metric"] == metric)
        )
        all_consistent = (
            rq1_consistency.loc[cons_mask, "fully_consistent"].all()
            if cons_mask.any()
            else False
        )

        row = {
            "workload_type": wt,
            "dataset_size": ds,
            "metric": metric,
            "fastest": fastest,
            "consistent": all_consistent,
        }
        row.update({f"median_{cfg}": med for cfg, med in config_medians.items()})
        rows.append(row)

    table = pd.DataFrame(rows)
    if len(table) > 0:
        table = table.set_index(["workload_type", "dataset_size", "metric"])
    return table


# ── LaTeX formatting helpers ───────────────────────────────────────────────

LATEX_COLUMN_RENAMES: dict[str, str] = {
    "n": "N",
    "median": "Median",
    "mean": "Mean",
    "std": "Std",
    "cv": "CV",
    "iqr": "IQR",
    "min": "Min",
    "p25": "Q1",
    "p75": "Q3",
    "p95": "P95",
    "max": "Max",
    "median_ci_lower": "CI Lower",
    "median_ci_upper": "CI Upper",
    "n_paired": "N",
    "wilcoxon_stat": "W",
    "p_value": "$p$",
    "a12": "$A_{12}$",
    "a12_category": "Effect",
    "ratio_median": "Ratio",
    "ratio_ci_lower": "Ratio CI Lo",
    "ratio_ci_upper": "Ratio CI Hi",
    "diff_median": "Diff",
    "diff_ci_lower": "Diff CI Lo",
    "diff_ci_upper": "Diff CI Hi",
    "friedman_stat": "$\\chi^2_F$",
    "friedman_p": "$p_F$",
    "p_value_holm": "$p_{Holm}$",
    "significant": "Sig.",
    "direction_consistent": "Direction",
    "all_significant": "Significant",
    "effect_size_consistent": "Effect Size",
    "fully_consistent": "Consistent",
    "fastest": "Fastest",
    "consistent": "Consistent",
    "workload_type": "Workload",
    "dataset_size": "Size",
    "config_a": "Config A",
    "config_b": "Config B",
    "benchmark_run": "Run",
    "metric": "Metric",
    "query_id": "Query",
    "configuration": "Config",
}

LATEX_INDEX_RENAMES: dict[str, str] = {
    "workload_type": "Workload",
    "dataset_size": "Size",
    "config_a": "Config A",
    "config_b": "Config B",
    "benchmark_run": "Run",
    "metric": "Metric",
    "query_id": "Query",
    "configuration": "Config",
}


def _fmt_bytes(v: float) -> str:
    if np.isnan(v):
        return "---"
    if abs(v) >= 1e9:
        return f"{v / 1e9:.2f}\\,GB"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}\\,MB"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.1f}\\,KB"
    return f"{v:.0f}\\,B"


def compact_descriptive_table(table1: pd.DataFrame) -> pd.DataFrame:
    if table1.empty:
        return table1

    rows = []
    for idx, row in table1.iterrows():
        metric = idx[-1] if isinstance(idx, tuple) else ""
        med, ci_lo, ci_hi = row["median"], row["median_ci_lower"], row["median_ci_upper"]

        if "bytes" in str(metric):
            fmt = _fmt_bytes
        else:
            fmt = lambda v: "---" if np.isnan(v) else f"{v:.4f}"

        cv_str = "---" if np.isnan(row["cv"]) else f"{row['cv']:.3f}"

        rows.append({
            "N": int(row["n"]),
            "Median [95\\% CI]": f"{fmt(med)} [{fmt(ci_lo)}, {fmt(ci_hi)}]",
            "CV": cv_str,
        })

    result = pd.DataFrame(rows, index=table1.index)
    result.index = result.index.rename(LATEX_INDEX_RENAMES)
    return result


def split_pairwise_effects(table3: pd.DataFrame) -> pd.DataFrame:
    if table3.empty:
        return table3
    cols = ["n_paired", "a12", "a12_category", "ratio_median", "ratio_ci_lower", "ratio_ci_upper"]
    available = [c for c in cols if c in table3.columns]
    df = table3[available].copy()
    df.index = df.index.rename(LATEX_INDEX_RENAMES)
    return df.rename(columns=LATEX_COLUMN_RENAMES)


def split_pairwise_significance(table3: pd.DataFrame) -> pd.DataFrame:
    if table3.empty:
        return table3
    cols = ["wilcoxon_stat", "p_value", "p_value_holm", "significant", "friedman_stat", "friedman_p"]
    available = [c for c in cols if c in table3.columns]
    df = table3[available].copy()
    df.index = df.index.rename(LATEX_INDEX_RENAMES)
    return df.rename(columns=LATEX_COLUMN_RENAMES)


def format_consistency_table(table4: pd.DataFrame) -> pd.DataFrame:
    if table4.empty:
        return table4
    df = table4.reset_index()
    return df.rename(columns=LATEX_COLUMN_RENAMES)


def format_ranking_table(ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return ranking
    df = ranking.copy()
    df.index = df.index.rename(LATEX_INDEX_RENAMES)
    return df.rename(columns=LATEX_COLUMN_RENAMES)

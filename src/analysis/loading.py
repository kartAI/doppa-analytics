from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd
import yaml


def load_experiments(benchmarks_yml_path: Path) -> dict[str, dict]:
    with open(benchmarks_yml_path) as f:
        benchmarks_yml = yaml.safe_load(f)
    return {exp["id"]: exp for exp in benchmarks_yml["experiments"]}


def parse_query_id(
    query_id: str,
    experiments: dict[str, dict],
    workload_types: list[str],
) -> tuple[str, str, str] | None:
    entry = experiments.get(query_id)
    if entry is None:
        return None
    dataset_size = entry["dataset_size"]
    stem = query_id.removesuffix(f"-{dataset_size}")

    for wt in workload_types:
        if stem.startswith(wt):
            configuration = stem[len(wt) + 1 :]
            if not configuration:
                configuration = wt
            return wt, configuration, dataset_size

    raise ValueError(f"Unknown workload type in query_id: {query_id}")


def load_samples(
    db: duckdb.DuckDBPyConnection,
    container: str,
    run_id: str,
) -> pd.DataFrame:
    df = db.execute(
        f"""
        SELECT *
        FROM read_parquet(
            'az://{container}/query_id=*/run_id={run_id}/benchmark_run=*/iteration=*/data.parquet',
            hive_partitioning = true,
            union_by_name = true
        )
    """
    ).fetchdf()
    df["benchmark_run"] = df["benchmark_run"].astype(int)
    df["iteration"] = df["iteration"].astype(int)
    return df


def load_metadata(
    db: duckdb.DuckDBPyConnection,
    container: str,
    blob_name: str,
    run_id: str,
) -> pd.DataFrame:
    return db.execute(
        f"""
        SELECT *
        FROM read_parquet('az://{container}/{blob_name}')
        WHERE run_id = '{run_id}'
    """
    ).fetchdf()


def load_costs(
    db: duckdb.DuckDBPyConnection,
    container: str,
    run_id: str,
    cost_types: list[str] | None = None,
) -> pd.DataFrame:
    if cost_types is None:
        cost_types = ["aci_cost", "blob_cost", "postgres_cost", "databricks_cost"]

    cost_dfs = {}
    for ct in cost_types:
        try:
            df = db.execute(
                f"""
                SELECT *
                FROM read_parquet(
                    'az://{container}/query_id=*/run_id={run_id}/benchmark_run=*/{ct}.parquet',
                    hive_partitioning = true
                )
            """
            ).fetchdf()
            df["benchmark_run"] = df["benchmark_run"].astype(int)
            df["cost_type"] = ct.replace("_cost", "")
            cost_dfs[ct] = df
        except Exception:
            pass

    return (
        pd.concat(cost_dfs.values(), ignore_index=True)
        if cost_dfs
        else pd.DataFrame()
    )


def enrich_samples(
    samples_df: pd.DataFrame,
    experiments: dict[str, dict],
    workload_types: list[str],
    iteration_ceilings: dict[str, int],
) -> pd.DataFrame:
    parsed = samples_df["query_id"].apply(
        lambda qid: parse_query_id(qid, experiments, workload_types)
    )
    unknown_mask = parsed.isna()
    if unknown_mask.any():
        unknown_ids = samples_df.loc[unknown_mask, "query_id"].unique()
        print(
            f"WARNING: {len(unknown_ids)} query_id(s) not in benchmarks.yml, dropping"
        )
        samples_df = samples_df[~unknown_mask].copy()
        parsed = parsed[~unknown_mask]

    samples_df["workload_type"] = parsed.apply(lambda x: x[0])
    samples_df["configuration"] = parsed.apply(lambda x: x[1])
    samples_df["dataset_size"] = parsed.apply(lambda x: x[2])

    samples_df["iteration_ceiling"] = samples_df["workload_type"].map(
        iteration_ceilings
    )
    samples_df["local_iteration"] = (
        samples_df["iteration"]
        - samples_df["iteration_ceiling"] * (samples_df["benchmark_run"] - 1)
    )

    return samples_df


def extract_worker_count(config: str) -> int | None:
    m = re.search(r"(\d+)-nodes?", config)
    return int(m.group(1)) if m else None


def extract_strategy(config: str) -> str:
    for s in ("broadcast", "partitioned", "default"):
        if s in config:
            return s
    return config

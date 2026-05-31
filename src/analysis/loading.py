from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd
import yaml


def _cache_table_name(kind: str, run_id: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z]", "_", run_id)
    return f"{kind}_{safe}"


def open_cache(cache_path: str | Path = "cache/bench-cache.duckdb") -> duckdb.DuckDBPyConnection:
    """Open (or create) a persistent DuckDB file used as the local data cache.

    Loads the azure extension so the same connection can both fetch remote
    parquet and store the result as a local table. Pins a temp directory next
    to the cache file and caps the memory limit so large remote reads spill to
    disk instead of running the process out of memory.
    """
    import platform

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(str(path))
    db.install_extension("azure")
    db.load_extension("azure")
    if platform.system() == "Linux":
        db.execute("SET azure_transport_option_type = curl")
    # Spill to disk rather than OOM on datasets larger than RAM.
    db.execute(f"SET temp_directory = '{path.parent / 'duckdb-tmp'}'")
    db.execute("SET memory_limit = '12GB'")
    db.execute("SET preserve_insertion_order = false")
    return db


def _table_exists(db: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return row is not None


def load_cached(
    db: duckdb.DuckDBPyConnection,
    kind: str,
    run_id: str,
    loader=None,
    *,
    sql: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return cached table for (kind, run_id), filling it on a miss.

    First call for a run_id materializes the cache table; later calls read it.
    Pass `refresh=True` to drop and re-fetch (use when data changed under the
    same run_id).

    Two miss-fill modes:
      * `sql=...` — a SELECT statement (e.g. a remote `read_parquet`). DuckDB
        streams it straight into the cache table, spilling to disk as needed,
        with no intermediate pandas copy. Use for large remote reads.
      * `loader=...` — a callable returning a DataFrame, registered and copied
        into the table. Use for small results assembled in Python.

    Either way the return value is read back from the cache table, so the only
    full pandas materialization is that single read-back.
    """
    if (sql is None) == (loader is None):
        raise ValueError("pass exactly one of `sql` or `loader`")

    table = _cache_table_name(kind, run_id)
    if refresh and _table_exists(db, table):
        db.execute(f'DROP TABLE "{table}"')

    if not _table_exists(db, table):
        if sql is not None:
            db.execute(f'CREATE TABLE "{table}" AS {sql}')
        else:
            df = loader()
            db.register("_load_cached_tmp", df)
            db.execute(f'CREATE TABLE "{table}" AS SELECT * FROM _load_cached_tmp')
            db.unregister("_load_cached_tmp")

    return db.execute(f'SELECT * FROM "{table}"').fetchdf()


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


def samples_select_sql(container: str, run_id: str) -> str:
    """SELECT statement for one run's per-iteration samples from remote parquet.

    Casts the hive partition columns to INTEGER in-engine so the result needs
    no pandas post-processing. Pass to `load_cached(..., sql=...)` so DuckDB can
    stream the (potentially larger-than-RAM) result straight into the cache.
    """
    return f"""
        SELECT * REPLACE (
            CAST(benchmark_run AS INTEGER) AS benchmark_run,
            CAST(iteration AS INTEGER) AS iteration
        )
        FROM read_parquet(
            'az://{container}/query_id=*/run_id={run_id}/benchmark_run=*/iteration=*/data.parquet',
            hive_partitioning = true,
            union_by_name = true
        )
    """


def load_samples(
    db: duckdb.DuckDBPyConnection,
    container: str,
    run_id: str,
) -> pd.DataFrame:
    return db.execute(samples_select_sql(container, run_id)).fetchdf()


def _sample_run_glob(container: str, run_id: str) -> str:
    return (
        f"az://{container}/query_id=*/run_id={run_id}/"
        "benchmark_run=*/iteration=*/data.parquet"
    )


def _sample_cell_glob(
    container: str, run_id: str, query_id: str, benchmark_run: int
) -> str:
    return (
        f"az://{container}/query_id={query_id}/run_id={run_id}/"
        f"benchmark_run={benchmark_run}/iteration=*/data.parquet"
    )


def _sample_read_sql(path_or_list: str) -> str:
    """SELECT that reads sample parquet(s) onto the canonical sample schema.

    `path_or_list` is a SQL string literal (a single quoted glob) or a SQL list
    of literals. `hive_partitioning` supplies query_id/run_id/benchmark_run/
    iteration from the path (engines differ on whether they also embed them) and
    `union_by_name` reconciles the per-engine column sets into one schema; the
    REPLACE casts the hive-string partition ints to INTEGER.
    """
    return f"""
        SELECT * REPLACE (
            CAST(benchmark_run AS INTEGER) AS benchmark_run,
            CAST(iteration AS INTEGER) AS iteration
        )
        FROM read_parquet(
            {path_or_list},
            hive_partitioning = true,
            union_by_name = true
        )
    """


def load_samples_cached(
    db: duckdb.DuckDBPyConnection,
    container: str,
    run_id: str,
    *,
    refresh: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Build the samples cache one bounded batch at a time, then return it.

    The benchmark writer emits roughly one tiny parquet file *per iteration*, so
    a full run is hundreds of thousands of blobs. Reading them in a single glob
    makes DuckDB open every file at once (and `union_by_name` read every footer),
    which exhausts memory before any row is materialized. The actual data is
    small (~0.5 KB/row); the cost is entirely the file-open storm.

    This reads one (query_id, benchmark_run) cell at a time -- at most one
    iteration ceiling of files (a few thousand, ~1 MB) -- and appends it to the
    persistent cache table, so peak memory stays flat regardless of run size.

    The first call for a run_id is network-bound: it must open every file once
    (expect tens of minutes to a couple of hours for a large run). Later calls
    read the finished cache table in one shot. An interrupted fill resumes: each
    cell is committed with a bookkeeping row, and already-loaded cells are
    skipped, so re-running continues where it stopped without duplicating rows.
    """
    table = _cache_table_name("samples", run_id)
    loaded = _cache_table_name("samples_loaded", run_id)

    if refresh:
        db.execute(f'DROP TABLE IF EXISTS "{table}"')
        db.execute(f'DROP TABLE IF EXISTS "{loaded}"')

    # Enumerate cells and one representative file per query_id from a single blob
    # LIST (glob() lists names; it does not open any file).
    db.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _sample_files AS
        SELECT
            file,
            regexp_extract(file, 'query_id=([^/]+)', 1) AS query_id,
            CAST(regexp_extract(file, 'benchmark_run=([0-9]+)', 1) AS INTEGER)
                AS benchmark_run
        FROM glob('{_sample_run_glob(container, run_id)}') t(file)
        """
    )
    cells = db.execute(
        "SELECT DISTINCT query_id, benchmark_run FROM _sample_files ORDER BY 1, 2"
    ).fetchall()
    if not cells:
        db.execute("DROP TABLE IF EXISTS _sample_files")
        raise FileNotFoundError(
            f"no sample files found for run_id={run_id} in container {container}"
        )

    if not (_table_exists(db, table) and _table_exists(db, loaded)):
        # Data table and its bookkeeping table are a unit; if either is missing
        # (or left half-built) rebuild both so resume can never double-count.
        db.execute(f'DROP TABLE IF EXISTS "{table}"')
        db.execute(f'DROP TABLE IF EXISTS "{loaded}"')
        # The cache table is created empty but carrying the union of every
        # engine's columns, so per-cell `INSERT ... BY NAME` always aligns.
        reps = [
            r[0]
            for r in db.execute(
                "SELECT min(file) FROM _sample_files GROUP BY query_id"
            ).fetchall()
        ]
        rep_list = "[" + ", ".join(f"'{f}'" for f in reps) + "]"
        db.execute(f'CREATE TABLE "{table}" AS {_sample_read_sql(rep_list)} LIMIT 0')
        db.execute(
            f'CREATE TABLE "{loaded}" '
            "(query_id VARCHAR, benchmark_run INTEGER)"
        )

    done = set(
        db.execute(f'SELECT query_id, benchmark_run FROM "{loaded}"').fetchall()
    )
    todo = [c for c in cells if c not in done]

    if progress and todo:
        print(
            f"Caching {len(todo)} of {len(cells)} sample cells for {run_id} "
            f"(first fill is slow; later runs read the cache)"
        )

    for i, (qid, br) in enumerate(todo, 1):
        glob = "'" + _sample_cell_glob(container, run_id, qid, br) + "'"
        # Data row(s) and the bookkeeping marker commit together: a crash mid-cell
        # rolls back both, so resume never double-counts a cell.
        db.execute("BEGIN TRANSACTION")
        try:
            db.execute(f'INSERT INTO "{table}" BY NAME {_sample_read_sql(glob)}')
            db.execute(
                f'INSERT INTO "{loaded}" VALUES (?, ?)', [qid, int(br)]
            )
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
        if progress:
            print(f"  [{i}/{len(todo)}] {qid} run {br}", flush=True)

    db.execute("DROP TABLE IF EXISTS _sample_files")
    return db.execute(f'SELECT * FROM "{table}"').fetchdf()


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


def enrich_costs(
    costs_df: pd.DataFrame,
    experiments: dict[str, dict],
    workload_types: list[str],
) -> pd.DataFrame:
    if costs_df.empty:
        return costs_df

    parsed = costs_df["query_id"].apply(
        lambda qid: parse_query_id(qid, experiments, workload_types)
    )
    valid = parsed.notna()
    costs_df = costs_df[valid].copy()
    parsed = parsed[valid]

    costs_df["workload_type"] = parsed.apply(lambda x: x[0])
    costs_df["configuration"] = parsed.apply(lambda x: x[1])
    costs_df["dataset_size"] = parsed.apply(lambda x: x[2])
    return costs_df


def extract_worker_count(config: str) -> int | None:
    m = re.search(r"(\d+)-nodes?", config)
    return int(m.group(1)) if m else None


def extract_strategy(config: str) -> str:
    for s in ("broadcast", "partitioned", "default"):
        if s in config:
            return s
    return config

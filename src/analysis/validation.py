from __future__ import annotations

import pandas as pd


def validate_schema(samples_df: pd.DataFrame) -> None:
    assert (
        samples_df["schema_version"] == "v4"
    ).all(), "Non-V4 schema rows found"
    print("Schema V4 check passed")


def validate_failure_rates(samples_df: pd.DataFrame) -> None:
    cell_counts = (
        samples_df.groupby(["query_id", "benchmark_run", "status"])
        .size()
        .unstack(fill_value=0)
    )
    if "failed" in cell_counts.columns:
        failure_pct = cell_counts["failed"] / cell_counts.sum(axis=1)
        exploratory = failure_pct[failure_pct > 0.05]
        if len(exploratory) > 0:
            print(
                f"WARNING: {len(exploratory)} cells exceed 5% failure rate (EXPLORATORY):"
            )
            print(exploratory)
        else:
            print("Failure rate check passed (<= 5% per cell)")
    else:
        print("No failed iterations found")


def validate_iteration_counts(
    successful: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> None:
    sample_agg = successful.groupby("query_id").size().reset_index(name="sample_n")
    meta_agg = (
        metadata_df.groupby("query_id")["achieved_iterations"].sum().reset_index()
    )
    merged = meta_agg.merge(sample_agg, on="query_id", how="outer")
    discrepancies = merged[merged["achieved_iterations"] != merged["sample_n"]]
    if len(discrepancies) > 0:
        print(
            "WARNING: iteration count discrepancies between metadata and samples:"
        )
        print(discrepancies)
    else:
        print("Iteration count cross-reference passed")


def validate_batch_consistency(
    samples_df: pd.DataFrame,
    experiments: dict[str, dict],
) -> None:
    seen_batches: set[tuple[str, ...]] = set()
    for qid, exp in experiments.items():
        if qid not in samples_df["query_id"].values:
            continue
        related = exp.get("related_script_ids", [])
        batch_key = tuple(sorted([qid] + related))
        if batch_key in seen_batches:
            continue
        seen_batches.add(batch_key)
        batch_present = [
            b for b in batch_key if b in samples_df["query_id"].values
        ]
        if len(batch_present) < 2:
            continue
        for br in samples_df["benchmark_run"].unique():
            run_ids: set[str] = set()
            for bid in batch_present:
                mask = (samples_df["query_id"] == bid) & (
                    samples_df["benchmark_run"] == br
                )
                run_ids.update(samples_df.loc[mask, "run_id"].unique())
            if len(run_ids) > 1:
                print(
                    f"WARNING: batch {batch_key} pass {br} has multiple run_ids: {run_ids}"
                )
    print("Batch run_id consistency check complete")


def validate_cardinality(
    successful: pd.DataFrame,
    experiments: dict[str, dict],
) -> None:
    for qid, exp in experiments.items():
        if qid not in successful["query_id"].values:
            continue
        related = exp.get("related_script_ids", [])
        batch_ids = [
            b for b in [qid] + related if b in successful["query_id"].values
        ]
        if len(batch_ids) < 2:
            continue
        cardinalities: dict[str, set] = {}
        for bid in batch_ids:
            cards = (
                successful.loc[
                    successful["query_id"] == bid, "result_cardinality"
                ]
                .dropna()
                .unique()
            )
            cardinalities[bid] = set(cards.tolist())
        all_cards = list(cardinalities.values())
        if len(all_cards) >= 2 and len(set.union(*all_cards)) > 1:
            unique_sets = {frozenset(v) for v in cardinalities.values()}
            if len(unique_sets) > 1:
                print(
                    f"WARNING: cardinality mismatch in batch containing {qid}: {cardinalities}"
                )
    print("Result cardinality consistency check complete")


def run_all_validations(
    samples_df: pd.DataFrame,
    successful: pd.DataFrame,
    metadata_df: pd.DataFrame,
    experiments: dict[str, dict],
) -> None:
    validate_schema(samples_df)
    validate_failure_rates(samples_df)
    validate_iteration_counts(successful, metadata_df)
    validate_batch_consistency(samples_df, experiments)
    validate_cardinality(successful, experiments)

import ast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.utils import get_max_values_from_runs

CPU_PARAMS = ["user", "system", "idle", "iowait"]
PLOT_COLORS = {
    "user": "#0072B2",
    "system": "#D55E00",
    "idle": "#009E73",
    "iowait": "#CC79A7",
}


def linechart(
        dict_a: dict[int, pd.DataFrame],
        dict_b: dict[int, pd.DataFrame],
        parameter: str,
        label_a: str = "A",
        label_b: str = "B",
        title: str | None = None,
) -> None:
    iterations = sorted(dict_a.keys())
    x_axis = np.array(iterations, dtype=int)

    y_axis_a = get_max_values_from_runs(dict_a, parameter)
    y_axis_b = get_max_values_from_runs(dict_b, parameter)

    plt.figure(figsize=(10, 6))

    plt.plot(x_axis, y_axis_a, marker="o", label=label_a)
    plt.plot(x_axis, y_axis_b, marker="o", label=label_b)

    plt.xlabel("Iteration")
    plt.ylabel(parameter)

    if title is None:
        title = f"Comparison of '{parameter}' Across Runs"

    plt.title(title)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _parse_core_struct(value: object) -> dict[str, float]:
    """Normalize core struct values to a dict[str, float]."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    if pd.isna(value):
        return {p: 0.0 for p in CPU_PARAMS}
    return ast.literal_eval(str(value))


def _extract_core_values(df: pd.DataFrame) -> dict[int, dict[str, float]]:
    """
    For a single run DataFrame, find for each core the row with the maximum
    total CPU time (sum of CPU_PARAMS) and return the CPU dict for that row.
    """
    result: dict[int, dict[str, float]] = {}
    core_cols = [c for c in df.columns if c.startswith("core_")]

    for col in core_cols:
        core_id = int(col.split("_")[1])

        series = df[col].apply(_parse_core_struct)

        totals = series.apply(
            lambda d: sum(float(d.get(p, 0.0)) for p in CPU_PARAMS)
        )

        idx = totals.idxmax()
        result[core_id] = series.loc[idx]

    return result


def _compute_max_per_core(data: dict[int, pd.DataFrame]) -> dict[int, dict[str, float]]:
    """
    Across all runs, compute for each core and each CPU parameter the maximum
    value observed at the per-run max-total-CPU row.
    """
    dfs = [data[k] for k in sorted(data.keys())]
    max_core_structs: dict[int, dict[str, list[float]]] = {}

    for df in dfs:
        core_values = _extract_core_values(df)
        for core_id, values in core_values.items():
            if core_id not in max_core_structs:
                max_core_structs[core_id] = {p: [] for p in CPU_PARAMS}
            for p in CPU_PARAMS:
                max_core_structs[core_id][p].append(float(values.get(p, 0.0)))

    aggregated: dict[int, dict[str, float]] = {}
    for core_id, param_lists in max_core_structs.items():
        aggregated[core_id] = {p: float(np.max(vals)) for p, vals in param_lists.items()}

    return aggregated


def _collect_per_core_series(
        data: dict[int, pd.DataFrame],
) -> tuple[list[int], dict[int, dict[str, list[float]]]]:
    """
    For a dict of runs {iteration -> df}, collect for each core a time series
    of CPU_PARAMS taken from the row with maximum total CPU per run.

    Returns
    -------
    iterations : list[int]
        Sorted list of iteration keys.
    per_core_series : dict[int, dict[str, list[float]]]
        core_id -> {param -> [values over iterations]}.
    """
    iterations = sorted(data.keys())
    per_core_series: dict[int, dict[str, list[float]]] = {}

    for it in iterations:
        df = data[it]
        core_values = _extract_core_values(df)  # core_id -> {param: value}

        for core_id, vals in core_values.items():
            if core_id not in per_core_series:
                per_core_series[core_id] = {p: [] for p in CPU_PARAMS}
            for p in CPU_PARAMS:
                per_core_series[core_id][p].append(float(vals.get(p, 0.0)))

    return iterations, per_core_series


def plot_barchart_per_core(
        dict_a: dict[int, pd.DataFrame],
        dict_b: dict[int, pd.DataFrame],
        label_a: str,
        label_b: str,
        title: str | None = None,
) -> None:
    iters_a, core_series_a = _collect_per_core_series(dict_a)
    iters_b, core_series_b = _collect_per_core_series(dict_b)

    if iters_a != iters_b:
        raise ValueError("dict_a and dict_b must share the same iteration keys")

    iterations = iters_a
    core_ids = sorted(set(core_series_a.keys()) & set(core_series_b.keys()))
    n = len(core_ids)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=2,
        figsize=(18, 5 * n),
        sharex=True,
        sharey=True
    )

    if n == 1:
        axes = np.array([axes])

    width = 0.8

    for idx, core_id in enumerate(core_ids):
        y_a = {p: core_series_a[core_id][p] for p in CPU_PARAMS}
        bottom = np.zeros(len(iterations))
        for p in CPU_PARAMS:
            axes[idx, 0].bar(
                iterations,
                y_a[p],
                width,
                label=p if idx == 0 else None,
                bottom=bottom,
                color=PLOT_COLORS[p],
                edgecolor="black",
                linewidth=0.3,
            )
            bottom += np.array(y_a[p])

        axes[idx, 0].set_title(f"{label_a} – Core {core_id}")
        axes[idx, 0].set_ylabel("CPU time (s)")

        y_b = {p: core_series_b[core_id][p] for p in CPU_PARAMS}
        bottom = np.zeros(len(iterations))
        for p in CPU_PARAMS:
            axes[idx, 1].bar(
                iterations,
                y_b[p],
                width,
                bottom=bottom,
                color=PLOT_COLORS[p],
                edgecolor="black",
                linewidth=0.3,
            )
            bottom += np.array(y_b[p])

        axes[idx, 1].set_title(f"{label_b} – Core {core_id}")

    axes[-1, 0].set_xlabel("Iteration")
    axes[-1, 1].set_xlabel("Iteration")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    if title is None:
        title = "Per-core CPU time components over iterations"

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 0.95, 0.95))
    plt.show()

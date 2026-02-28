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


def linechart_per_iteration(
        dict_a: dict[int, pd.DataFrame],
        dict_b: dict[int, pd.DataFrame],
        iterations: list[int],
        parameter: str,
        label_a: str = "A",
        label_b: str = "B",
        title: str | None = None,
        share_y: bool = True
) -> None:
    n = len(iterations)

    if share_y:
        all_values = []
        for it in iterations:
            all_values.extend(dict_a[it][parameter].values.tolist())
            all_values.extend(dict_b[it][parameter].values.tolist())
        y_min, y_max = min(all_values), max(all_values)
    else:
        y_min = y_max = None

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(14, 4 * n), sharey=share_y)

    if n == 1:
        axes = np.array([axes])

    for row_idx, it in enumerate(iterations):
        df_a = dict_a[it]
        df_b = dict_b[it]

        x_a = df_a["elapsed_time"].values
        x_b = df_b["elapsed_time"].values

        axes[row_idx, 0].plot(
            x_a,
            df_a[parameter].values,
            marker="o",
            label=f"{label_a} – iteration {it}",
        )
        axes[row_idx, 0].set_title(f"{label_a} (iteration {it})")
        if share_y:
            axes[row_idx, 0].set_ylim(y_min, y_max)
        axes[row_idx, 0].set_xlabel("Elapsed time (s)")
        axes[row_idx, 0].set_ylabel(parameter)
        axes[row_idx, 0].grid(True)

        axes[row_idx, 1].plot(
            x_b,
            df_b[parameter].values,
            marker="o",
            label=f"{label_b} – iteration {it}",
        )
        axes[row_idx, 1].set_title(f"{label_b} (iteration {it})")
        if share_y:
            axes[row_idx, 1].set_ylim(y_min, y_max)
        axes[row_idx, 1].set_xlabel("Elapsed time (s)")
        axes[row_idx, 1].grid(True)

    if title is None:
        title = f"Comparison of '{parameter}' Across Selected Iterations"

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


def linechart_per_iteration_timeseries(
        dict_a: dict[int, pd.DataFrame],
        dict_b: dict[int, pd.DataFrame],
        iterations: list[int],
        parameter: str,
        label_a: str = "A",
        label_b: str = "B",
        title: str | None = None,
) -> None:
    n = len(iterations)

    all_values = []
    for it in iterations:
        all_values.extend(dict_a[it][parameter].values.tolist())
        all_values.extend(dict_b[it][parameter].values.tolist())
    y_min, y_max = min(all_values), max(all_values)

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 3.5 * n), sharey=True)

    if n == 1:
        axes = np.array([axes])

    for row_idx, it in enumerate(iterations):
        df_a = dict_a[it]
        df_b = dict_b[it]

        x_a = df_a["elapsed_time"].values
        x_b = df_b["elapsed_time"].values

        ax = axes[row_idx]

        ax.plot(
            x_a,
            df_a[parameter].values,
            marker="o",
            label=label_a,
        )
        ax.plot(
            x_b,
            df_b[parameter].values,
            marker="o",
            label=label_b,
        )

        ax.set_title(f"Iteration {it}")
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel(parameter)
        ax.grid(True)

        if row_idx == n - 1:
            ax.set_xlabel("Elapsed time (s)")

        ax.legend()

    if title is None:
        title = f"Comparison of '{parameter}' Across Selected Iterations"

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
        core_values = _extract_core_values(df)

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


def heatmap_cpu_percent_per_iteration(
        dict_a: dict[int, pd.DataFrame],
        dict_b: dict[int, pd.DataFrame],
        iterations: list[int],
        label_a: str = "A",
        label_b: str = "B",
        title: str | None = None,
) -> None:
    """
    For each selected iteration, plot a heatmap of CPU percent per core over time.

    Layout:
      - One row per iteration
      - Left column: dict_a
      - Right column: dict_b

    Heatmap:
      - x-axis: sample index (with elapsed_time tick labels)
      - y-axis: cores (core_1, core_2, ...)
      - colour: core_i["percent"]
    """

    iterations = [it for it in iterations if it in dict_a and it in dict_b]
    if not iterations:
        raise ValueError("None of the requested iterations exist in both dict_a and dict_b.")

    n = len(iterations)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=2,
        figsize=(18, 3.5 * n),
        sharex=False,
        sharey=False,
    )

    if n == 1:
        axes = np.array([axes])

    def _extract_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
        if "elapsed_time" not in df.columns:
            raise ValueError("DataFrame must contain 'elapsed_time' column.")

        times = df["elapsed_time"].values
        core_cols = sorted(
            [c for c in df.columns if c.startswith("core_")],
            key=lambda c: int(c.split("_")[1]),
        )
        core_labels = core_cols

        mat = np.zeros((len(core_cols), len(times)), dtype=float)

        for row_idx, col in enumerate(core_cols):
            percent_vals = (
                df[col]
                .apply(_parse_core_struct)
                .apply(lambda d: float(d.get("percent", 0.0)))
                .values
            )
            mat[row_idx, :] = percent_vals

        return mat, core_labels, times

    vmin, vmax = 0.0, 100.0
    first_im = None

    for row_idx, it in enumerate(iterations):

        mat_a, core_labels_a, times_a = _extract_matrix(dict_a[it])
        ax_left = axes[row_idx, 0]
        im = ax_left.imshow(
            mat_a,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        if first_im is None:
            first_im = im

        ax_left.set_title(f"{label_a} – iteration {it}")
        ax_left.set_yticks(np.arange(len(core_labels_a)))
        ax_left.set_yticklabels(core_labels_a)

        n_samples = mat_a.shape[1]
        if n_samples > 1:
            xticks_idx = np.linspace(0, n_samples - 1, num=min(5, n_samples), dtype=int)
            ax_left.set_xticks(xticks_idx)
            ax_left.set_xticklabels(
                [f"{times_a[i]:.3f}" for i in xticks_idx],
                rotation=45,
            )
        ax_left.set_xlabel("Elapsed time (s)")
        ax_left.set_ylabel("Core")

        mat_b, core_labels_b, times_b = _extract_matrix(dict_b[it])
        ax_right = axes[row_idx, 1]
        ax_right.imshow(
            mat_b,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax_right.set_title(f"{label_b} – iteration {it}")
        ax_right.set_yticks(np.arange(len(core_labels_b)))
        ax_right.set_yticklabels(core_labels_b)

        n_samples_b = mat_b.shape[1]
        if n_samples_b > 1:
            xticks_idx_b = np.linspace(0, n_samples_b - 1, num=min(5, n_samples_b), dtype=int)
            ax_right.set_xticks(xticks_idx_b)
            ax_right.set_xticklabels(
                [f"{times_b[i]:.3f}" for i in xticks_idx_b],
                rotation=45,
            )
        ax_right.set_xlabel("Elapsed time (s)")

    if title is None:
        title = "CPU percent per core – heatmaps across selected iterations"

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))

    if first_im is not None:
        cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
        cbar = fig.colorbar(first_im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("CPU percent")

    plt.show()

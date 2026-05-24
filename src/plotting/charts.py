from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from IPython.display import display

from src.analysis.stats import bootstrap_median_ci

from .style import StyleConfig


def _savefig(fig: plt.Figure, name: str, figures_dir: Path) -> None:
    fig.savefig(figures_dir / f"{name}.png")
    plt.show()
    plt.close(fig)


def _needs_log_scale(values, threshold: float = 20.0) -> bool:
    pos = [v for v in values if v > 0]
    if len(pos) < 2:
        return False
    return max(pos) / min(pos) > threshold


def _fmt_value(v: float, metric: str = "") -> str:
    if "bytes" in metric:
        if abs(v) >= 1e9:
            return f"{v / 1e9:.1f} GB"
        if abs(v) >= 1e6:
            return f"{v / 1e6:.1f} MB"
        if abs(v) >= 1e3:
            return f"{v / 1e3:.1f} KB"
        return f"{v:.0f} B"
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    if abs(v) >= 0.001:
        return f"{v:.4f}"
    return f"{v:.2e}"


# ── Per-cell charts (iterated over workload_type x dataset_size) ─────────


def plot_median_bars(
    successful: pd.DataFrame,
    primary_metrics: list[str],
    style: StyleConfig,
    figures_dir: Path,
) -> None:
    for wt in sorted(successful["workload_type"].unique()):
        for ds in sorted(successful["dataset_size"].unique()):
            mask = (successful["workload_type"] == wt) & (
                successful["dataset_size"] == ds
            )
            if not mask.any():
                continue

            data = successful[mask]
            configs = sorted(data["configuration"].unique())
            metrics_present = [
                m for m in primary_metrics if data[m].notna().any()
            ]
            if not metrics_present:
                continue

            n_metrics = len(metrics_present)
            fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
            if n_metrics == 1:
                axes = [axes]

            for ax, metric in zip(axes, metrics_present):
                medians, ci_lo, ci_hi, colors, labels = [], [], [], [], []
                for cfg in configs:
                    vals = (
                        data[data["configuration"] == cfg][metric]
                        .dropna()
                        .values
                    )
                    if len(vals) == 0:
                        continue
                    med = np.median(vals)
                    lo, hi = bootstrap_median_ci(vals)
                    medians.append(med)
                    ci_lo.append(med - lo)
                    ci_hi.append(hi - med)
                    colors.append(style.color(cfg))
                    labels.append(style.label(cfg))

                y = np.arange(len(medians))
                ax.barh(
                    y,
                    medians,
                    color=colors,
                    edgecolor="white",
                    linewidth=0.5,
                    height=0.6,
                )
                ax.errorbar(
                    medians,
                    y,
                    xerr=[ci_lo, ci_hi],
                    fmt="none",
                    ecolor="0.3",
                    capsize=3,
                    linewidth=1.2,
                )
                if _needs_log_scale(medians):
                    ax.set_xscale(
                        "symlog",
                        linthresh=max(
                            min(m for m in medians if m > 0) * 0.5, 1e-6
                        ),
                    )
                for yi, med in zip(y, medians):
                    ax.annotate(
                        _fmt_value(med, metric),
                        xy=(med, yi),
                        xytext=(5, 0),
                        textcoords="offset points",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                    )
                ax.set_yticks(y)
                ax.set_yticklabels(labels)
                ax.set_xlabel(style.metric_label(metric))
                ax.invert_yaxis()

            fig.suptitle(
                f"{style.workload_label(wt)} ({ds})",
                fontsize=14,
                fontweight="bold",
            )
            fig.tight_layout()
            _savefig(fig, f"{wt}_{ds}_median_bars", figures_dir)


def plot_violin(
    successful: pd.DataFrame,
    style: StyleConfig,
    figures_dir: Path,
) -> None:
    for wt in sorted(successful["workload_type"].unique()):
        for ds in sorted(successful["dataset_size"].unique()):
            mask = (successful["workload_type"] == wt) & (
                successful["dataset_size"] == ds
            )
            if not mask.any():
                continue

            data = successful[mask]
            configs = sorted(data["configuration"].unique())
            bp_data = []
            configs_present = []
            for c in configs:
                vals = (
                    data[data["configuration"] == c]["elapsed_time"]
                    .dropna()
                    .values
                )
                if len(vals) > 0:
                    bp_data.append(vals)
                    configs_present.append(c)
            if not bp_data:
                continue

            fig, ax = plt.subplots(
                figsize=(max(8, 2.5 * len(configs_present)), 5)
            )
            parts = ax.violinplot(
                bp_data,
                positions=range(len(bp_data)),
                showmedians=True,
                showextrema=False,
            )
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(style.color(configs_present[i]))
                pc.set_alpha(0.7)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.5)

            q1s = [np.percentile(d, 25) for d in bp_data]
            q3s = [np.percentile(d, 75) for d in bp_data]
            ax.vlines(
                range(len(bp_data)),
                q1s,
                q3s,
                color="0.3",
                linewidth=3,
                zorder=3,
            )

            all_elapsed = np.concatenate(bp_data)
            if _needs_log_scale(all_elapsed.tolist()):
                ax.set_yscale(
                    "symlog",
                    linthresh=max(
                        all_elapsed[all_elapsed > 0].min() * 0.5, 1e-6
                    ),
                )

            for i, d in enumerate(bp_data):
                med = np.median(d)
                ax.annotate(
                    _fmt_value(med, "elapsed_time"),
                    xy=(i, med),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                )

            ax.set_xticks(range(len(configs_present)))
            ax.set_xticklabels(
                [style.label(c) for c in configs_present],
                rotation=30,
                ha="right",
            )
            ax.set_ylabel(style.metric_label("elapsed_time"))
            ax.set_title(f"{style.workload_label(wt)} ({ds})")
            fig.tight_layout()
            _savefig(fig, f"{wt}_{ds}_violin", figures_dir)


def plot_timeseries(
    successful: pd.DataFrame,
    style: StyleConfig,
    figures_dir: Path,
) -> None:
    for wt in sorted(successful["workload_type"].unique()):
        for ds in sorted(successful["dataset_size"].unique()):
            mask = (successful["workload_type"] == wt) & (
                successful["dataset_size"] == ds
            )
            if not mask.any():
                continue

            data = successful[mask]
            configs = sorted(data["configuration"].unique())

            fig, ax = plt.subplots(figsize=(10, 5))
            for cfg in configs:
                d = data[data["configuration"] == cfg].sort_values(
                    "local_iteration"
                )
                ax.plot(
                    d["local_iteration"],
                    d["elapsed_time"],
                    label=style.label(cfg),
                    color=style.color(cfg),
                    alpha=0.7,
                    linewidth=0.8,
                )
            ax.set_xlabel("Iteration (within pass)")
            ax.set_ylabel(style.metric_label("elapsed_time"))
            ax.set_title(
                f"{style.workload_label(wt)} ({ds}) — Iteration timeseries"
            )
            ax.legend(loc="best", fontsize=9)
            fig.tight_layout()
            _savefig(fig, f"{wt}_{ds}_timeseries", figures_dir)


def plot_scatter(
    successful: pd.DataFrame,
    style: StyleConfig,
    figures_dir: Path,
) -> None:
    for wt in sorted(successful["workload_type"].unique()):
        for ds in sorted(successful["dataset_size"].unique()):
            mask = (successful["workload_type"] == wt) & (
                successful["dataset_size"] == ds
            )
            if not mask.any():
                continue

            data = successful[mask]
            configs = sorted(data["configuration"].unique())

            fig, ax = plt.subplots(figsize=(8, 6))
            for cfg in configs:
                d = data[data["configuration"] == cfg]
                ax.scatter(
                    d["elapsed_time"],
                    d["network_bytes_received"],
                    label=style.label(cfg),
                    color=style.color(cfg),
                    alpha=0.5,
                    s=15,
                    edgecolors="none",
                )
            ax.set_xlabel(style.metric_label("elapsed_time"))
            ax.set_ylabel(style.metric_label("network_bytes_received"))
            ax.set_title(
                f"{style.workload_label(wt)} ({ds}) — Elapsed vs Network I/O"
            )
            ax.legend(loc="best", fontsize=9, markerscale=2)
            fig.tight_layout()
            _savefig(fig, f"{wt}_{ds}_scatter", figures_dir)


def plot_cpu_breakdown(
    successful: pd.DataFrame,
    style: StyleConfig,
    figures_dir: Path,
) -> None:
    cpu_metrics = ["cpu_time_user_seconds", "cpu_time_system_seconds"]
    for wt in sorted(successful["workload_type"].unique()):
        for ds in sorted(successful["dataset_size"].unique()):
            mask = (successful["workload_type"] == wt) & (
                successful["dataset_size"] == ds
            )
            if not mask.any():
                continue

            data = successful[mask]
            if not all(data[m].notna().any() for m in cpu_metrics):
                continue

            configs = sorted(data["configuration"].unique())
            configs_present = [
                c for c in configs if len(data[data["configuration"] == c]) > 0
            ]

            user_meds = [
                float(
                    np.median(
                        data[data["configuration"] == c][
                            "cpu_time_user_seconds"
                        ].dropna()
                    )
                )
                for c in configs_present
            ]
            sys_meds = [
                float(
                    np.median(
                        data[data["configuration"] == c][
                            "cpu_time_system_seconds"
                        ].dropna()
                    )
                )
                for c in configs_present
            ]

            fig, ax = plt.subplots(
                figsize=(max(8, 2 * len(configs_present)), 5)
            )
            y = np.arange(len(configs_present))
            ax.barh(
                y,
                user_meds,
                height=0.5,
                color=[style.color(c) for c in configs_present],
                edgecolor="white",
                linewidth=0.5,
                label="User",
            )
            ax.barh(
                y,
                sys_meds,
                height=0.5,
                left=user_meds,
                color="0.55",
                edgecolor="white",
                linewidth=0.5,
                alpha=0.8,
                label="System",
            )
            ax.set_yticks(y)
            ax.set_yticklabels([style.label(c) for c in configs_present])
            ax.set_xlabel("CPU time (s)")
            ax.set_title(
                f"{style.workload_label(wt)} ({ds}) — CPU time (user + system)"
            )
            ax.legend(loc="best", fontsize=9)
            ax.invert_yaxis()
            fig.tight_layout()
            _savefig(fig, f"{wt}_{ds}_cpu", figures_dir)


def plot_network_io(
    successful: pd.DataFrame,
    style: StyleConfig,
    figures_dir: Path,
) -> None:
    net_metrics = ["network_bytes_received", "network_bytes_sent"]
    for wt in sorted(successful["workload_type"].unique()):
        for ds in sorted(successful["dataset_size"].unique()):
            mask = (successful["workload_type"] == wt) & (
                successful["dataset_size"] == ds
            )
            if not mask.any():
                continue

            data = successful[mask]
            configs = sorted(data["configuration"].unique())
            configs_present = [
                c for c in configs if len(data[data["configuration"] == c]) > 0
            ]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, metric in zip(axes, net_metrics):
                bp_data, bp_colors, bp_labels = [], [], []
                for cfg in configs_present:
                    vals = (
                        data[data["configuration"] == cfg][metric]
                        .dropna()
                        .values
                    )
                    if len(vals) > 0:
                        bp_data.append(vals)
                        bp_colors.append(style.color(cfg))
                        bp_labels.append(style.label(cfg))

                if not bp_data:
                    continue

                all_vals = np.concatenate(bp_data)
                pos_vals = all_vals[all_vals > 0]
                use_log = len(pos_vals) > 0 and (
                    pos_vals.max() / pos_vals.min() > 100
                )

                bplot = ax.boxplot(
                    bp_data, patch_artist=True, showfliers=False, widths=0.6
                )
                for patch, color in zip(bplot["boxes"], bp_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                for element in ["whiskers", "caps"]:
                    for line in bplot[element]:
                        line.set_color("0.4")
                for med in bplot["medians"]:
                    med.set_color("black")
                    med.set_linewidth(1.5)

                if use_log:
                    ax.set_yscale("symlog", linthresh=1)

                ax.set_xticklabels(
                    bp_labels, rotation=30, ha="right", fontsize=9
                )
                ax.set_ylabel(style.metric_label(metric))

            fig.suptitle(
                f"{style.workload_label(wt)} ({ds}) — Network I/O",
                fontsize=14,
                fontweight="bold",
            )
            fig.tight_layout()
            _savefig(fig, f"{wt}_{ds}_network", figures_dir)


def plot_cv_iqr(
    successful: pd.DataFrame,
    primary_metrics: list[str],
    auxiliary_metrics: list[str],
    style: StyleConfig,
    figures_dir: Path,
) -> None:
    for wt in sorted(successful["workload_type"].unique()):
        for ds in sorted(successful["dataset_size"].unique()):
            mask = (successful["workload_type"] == wt) & (
                successful["dataset_size"] == ds
            )
            if not mask.any():
                continue

            data = successful[mask]
            configs = sorted(data["configuration"].unique())
            metrics_to_show = [
                m
                for m in primary_metrics + auxiliary_metrics
                if data[m].notna().any()
            ]

            cv_rows = []
            for cfg in configs:
                for metric in metrics_to_show:
                    vals = (
                        data[data["configuration"] == cfg][metric]
                        .dropna()
                        .values
                    )
                    if len(vals) >= 2:
                        mean_val = np.mean(vals)
                        cv = (
                            float(np.std(vals, ddof=1) / mean_val)
                            if mean_val > 0
                            else np.nan
                        )
                        iqr = float(
                            np.percentile(vals, 75) - np.percentile(vals, 25)
                        )
                        cv_rows.append(
                            {
                                "config": cfg,
                                "metric": metric,
                                "cv": cv,
                                "iqr": iqr,
                            }
                        )

            if not cv_rows:
                continue

            cv_df = pd.DataFrame(cv_rows)
            n_metrics = len(metrics_to_show)
            cfgs_in_data = [
                c for c in configs if c in cv_df["config"].values
            ]
            width = 0.8 / max(len(cfgs_in_data), 1)
            x = np.arange(n_metrics)

            fig, ax = plt.subplots(figsize=(max(10, 2 * n_metrics), 5))
            for j, cfg in enumerate(cfgs_in_data):
                cvs = []
                for metric in metrics_to_show:
                    row = cv_df[
                        (cv_df["config"] == cfg)
                        & (cv_df["metric"] == metric)
                    ]
                    cvs.append(
                        float(row["cv"].iloc[0]) if len(row) > 0 else 0
                    )
                offset = (j - len(cfgs_in_data) / 2 + 0.5) * width
                ax.bar(
                    x + offset,
                    cvs,
                    width * 0.9,
                    label=style.label(cfg),
                    color=style.color(cfg),
                    edgecolor="white",
                    linewidth=0.5,
                )
            ax.axhline(
                y=0.10,
                color="red",
                linestyle="--",
                alpha=0.6,
                linewidth=1,
                label="CV = 0.10 threshold",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [style.metric_label(m) for m in metrics_to_show],
                rotation=30,
                ha="right",
                fontsize=9,
            )
            ax.set_ylabel("Coefficient of variation (CV)")
            ax.set_title(
                f"{style.workload_label(wt)} ({ds}) — Measurement Stability (CV)"
            )
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            _savefig(fig, f"{wt}_{ds}_cv", figures_dir)


# ── Cross-workload / cross-size charts ───────────────────────────────────


def plot_cross_workload(
    successful: pd.DataFrame,
    workload_types: list[str],
    configs: list[str],
    style: StyleConfig,
    figures_dir: Path,
) -> None:
    wts_present = [
        wt
        for wt in workload_types
        if wt in successful["workload_type"].unique()
    ]
    cfgs_present = [
        c for c in configs if c in successful["configuration"].unique()
    ]

    if len(wts_present) < 2 or len(cfgs_present) < 2:
        return

    for ds in sorted(successful["dataset_size"].unique()):
        rows = []
        for wt in wts_present:
            for cfg in cfgs_present:
                mask = (
                    (successful["workload_type"] == wt)
                    & (successful["dataset_size"] == ds)
                    & (successful["configuration"] == cfg)
                )
                vals = successful.loc[mask, "elapsed_time"].dropna().values
                if len(vals) > 0:
                    rows.append(
                        {
                            "workload": wt,
                            "config": cfg,
                            "median": float(np.median(vals)),
                        }
                    )

        if not rows:
            continue

        cw = pd.DataFrame(rows)
        wts_in_data = [
            wt for wt in wts_present if wt in cw["workload"].values
        ]
        cfgs_in_data = [
            c for c in cfgs_present if c in cw["config"].values
        ]

        fig, ax = plt.subplots(
            figsize=(max(8, 2.5 * len(wts_in_data)), 5)
        )
        x = np.arange(len(wts_in_data))
        width = 0.8 / len(cfgs_in_data)

        for i, cfg in enumerate(cfgs_in_data):
            meds = [
                float(
                    cw[(cw["workload"] == wt) & (cw["config"] == cfg)][
                        "median"
                    ].iloc[0]
                )
                if len(
                    cw[(cw["workload"] == wt) & (cw["config"] == cfg)]
                )
                > 0
                else 0
                for wt in wts_in_data
            ]
            offset = (i - len(cfgs_in_data) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                meds,
                width * 0.9,
                label=style.label(cfg),
                color=style.color(cfg),
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, med in zip(bars, meds):
                if med > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        med,
                        f"{med:.2f}s",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [style.workload_label(wt) for wt in wts_in_data], fontsize=10
        )
        ax.set_ylabel(style.metric_label("elapsed_time"))
        ax.set_yscale("log")
        ax.set_title(
            f"Cross-workload Comparison ({ds}) — Median Elapsed Time (log scale)"
        )
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        _savefig(fig, f"cross_workload_{ds}", figures_dir)


def plot_scaling_curve(
    rq2_scaling: pd.DataFrame,
    rq2_single_node: pd.DataFrame,
    style: StyleConfig,
    figures_dir: Path,
) -> None:
    if len(rq2_scaling) == 0:
        return

    for ds in rq2_scaling["dataset_size"].unique():
        ds_data = rq2_scaling[rq2_scaling["dataset_size"] == ds]
        if ds_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy in sorted(ds_data["strategy"].unique()):
            s = ds_data[ds_data["strategy"] == strategy].sort_values(
                "worker_count"
            )
            col = style.strategy_colors.get(strategy, "#999")
            ax.errorbar(
                s["worker_count"],
                s["headline_median"],
                yerr=[
                    s["headline_median"] - s["pass_min"],
                    s["pass_max"] - s["headline_median"],
                ],
                marker="o",
                label=strategy.capitalize(),
                color=col,
                capsize=4,
                linewidth=2,
                markersize=8,
            )

        if "dataset_size" in rq2_single_node.columns:
            sn = rq2_single_node[rq2_single_node["dataset_size"] == ds]
            for _, row in sn.iterrows():
                ax.axhline(
                    y=row["headline_median"],
                    linestyle="--",
                    alpha=0.6,
                    color=style.color(row["configuration"]),
                    linewidth=1.5,
                    label=style.label(row["configuration"]),
                )

        ax.set_xlabel("Worker count")
        ax.set_ylabel("Median elapsed time (s)")
        ax.set_title(
            f"National-scale Spatial Join ({ds}) — Scaling curve"
        )
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        _savefig(fig, f"scaling_curve_{ds}", figures_dir)


def plot_databricks_metrics(
    successful: pd.DataFrame,
    databricks_metrics: list[str],
    style: StyleConfig,
    figures_dir: Path,
) -> None:
    dbr_configs = [
        c
        for c in sorted(successful["configuration"].unique())
        if "databricks" in c
    ]
    dbr_metrics_present = [
        m for m in databricks_metrics if successful[m].notna().any()
    ]

    if not dbr_configs or not dbr_metrics_present:
        return

    dbr_data = successful[successful["configuration"].isin(dbr_configs)]

    n_metrics = len(dbr_metrics_present)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows)
    )
    axes_flat = np.atleast_1d(axes).flatten()

    last_i = 0
    for i, metric in enumerate(dbr_metrics_present):
        last_i = i
        ax = axes_flat[i]
        bp_data, bp_colors, bp_labels = [], [], []
        for cfg in dbr_configs:
            vals = (
                dbr_data[dbr_data["configuration"] == cfg][metric]
                .dropna()
                .values
            )
            if len(vals) > 0:
                bp_data.append(vals)
                bp_colors.append(style.color(cfg))
                bp_labels.append(style.label(cfg))

        if bp_data:
            bplot = ax.boxplot(
                bp_data, patch_artist=True, showfliers=False, widths=0.5
            )
            for patch, color in zip(bplot["boxes"], bp_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            for element in ["whiskers", "caps"]:
                for line in bplot[element]:
                    line.set_color("0.4")
            for med in bplot["medians"]:
                med.set_color("black")
                med.set_linewidth(1.5)

        if "bytes" in metric:
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _, m=metric: _fmt_value(x, m))
            )
        ax.set_xticklabels(
            bp_labels, rotation=25, ha="right", fontsize=9
        )
        ax.set_title(
            style.metric_label(metric), fontsize=11, fontweight="bold"
        )
        ax.set_ylabel("")

    for j in range(last_i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Databricks Configurations — Internal Metrics",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    _savefig(fig, "databricks_metrics", figures_dir)


def plot_size_scaling(
    successful: pd.DataFrame,
    style: StyleConfig,
    figures_dir: Path,
) -> None:
    for wt in sorted(successful["workload_type"].unique()):
        wt_data = successful[successful["workload_type"] == wt]
        sizes = sorted(
            wt_data["dataset_size"].unique(),
            key=lambda s: style.size_order.get(s, 99),
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

        fig, ax = plt.subplots(figsize=(max(8, 2 * len(configs)), 5))
        x = np.arange(len(configs))
        width = 0.8 / len(sizes)

        for i, ds in enumerate(sizes):
            meds = [median_map.get((cfg, ds), 0) for cfg in configs]
            offset = (i - len(sizes) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                meds,
                width * 0.9,
                label=ds.capitalize(),
                color=style.size_colors.get(ds, "#999"),
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, med in zip(bars, meds):
                if med > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        med,
                        f"{med:.2f}s",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [style.label(cfg) for cfg in configs],
            rotation=20,
            ha="right",
            fontsize=9,
        )
        ax.set_ylabel(style.metric_label("elapsed_time"))
        ax.set_yscale("log")
        ax.set_title(
            f"{style.workload_label(wt)} — Median Elapsed Time by Dataset Size"
        )
        ax.legend(title="Dataset size", loc="best", fontsize=9)
        fig.tight_layout()
        _savefig(fig, f"{wt}_size_scaling", figures_dir)


def plot_ranking_stability(
    successful: pd.DataFrame,
    table3: pd.DataFrame,
    style: StyleConfig,
) -> None:
    for wt in sorted(successful["workload_type"].unique()):
        wt_data = successful[successful["workload_type"] == wt]
        sizes = sorted(
            wt_data["dataset_size"].unique(),
            key=lambda s: style.size_order.get(s, 99),
        )

        if len(sizes) < 2:
            continue

        ranking_rows = []
        for ds in sizes:
            ds_data = wt_data[wt_data["dataset_size"] == ds]
            config_medians = {}
            for cfg in ds_data["configuration"].unique():
                vals = (
                    ds_data[ds_data["configuration"] == cfg]["elapsed_time"]
                    .dropna()
                    .values
                )
                if len(vals) > 0:
                    config_medians[cfg] = float(np.median(vals))

            for rank, (cfg, med) in enumerate(
                sorted(config_medians.items(), key=lambda x: x[1]), 1
            ):
                ranking_rows.append(
                    {
                        "dataset_size": ds,
                        "configuration": cfg,
                        "rank": rank,
                        "median_elapsed_time": med,
                    }
                )

        if not ranking_rows:
            continue

        ranking_df = pd.DataFrame(ranking_rows)

        rank_pivot = ranking_df.pivot_table(
            index="configuration",
            columns="dataset_size",
            values="rank",
            aggfunc="first",
        )
        rank_pivot = rank_pivot[
            [ds for ds in sizes if ds in rank_pivot.columns]
        ]

        med_pivot = ranking_df.pivot_table(
            index="configuration",
            columns="dataset_size",
            values="median_elapsed_time",
            aggfunc="first",
        )
        med_pivot = med_pivot[
            [ds for ds in sizes if ds in med_pivot.columns]
        ]

        shared_configs = rank_pivot.dropna().index.tolist()
        if len(shared_configs) >= 2:
            shared_meds = med_pivot.loc[shared_configs]
            shared_ranks = shared_meds.rank(method="min").astype(int)
            first_col = shared_ranks.columns[0]
            stable = all(
                (shared_ranks[col] == shared_ranks[first_col]).all()
                for col in shared_ranks.columns[1:]
            )
            status = "STABLE" if stable else "CHANGED"
        else:
            status = "N/A (fewer than 2 shared configs)"

        print(
            f"\n{style.workload_label(wt)} — ranking stability: {status}"
        )
        print("Ranks:")
        display(rank_pivot)
        print("Median elapsed time (s):")
        display(med_pivot.round(4))

        if len(table3) > 0 and len(shared_configs) >= 2:
            t3 = table3.reset_index()
            t3_wt = t3[
                (t3["workload_type"] == wt)
                & (t3["metric"] == "elapsed_time")
                & (t3["config_a"].isin(shared_configs))
                & (t3["config_b"].isin(shared_configs))
            ]
            if len(t3_wt) > 0:
                adv_pivot = t3_wt.pivot_table(
                    index=["config_a", "config_b"],
                    columns="dataset_size",
                    values="ratio_median",
                    aggfunc="median",
                )
                adv_pivot = adv_pivot[
                    [ds for ds in sizes if ds in adv_pivot.columns]
                ]
                if len(adv_pivot.columns) >= 2:
                    print(
                        "Pairwise elapsed-time ratio (config_a / config_b) across sizes:"
                    )
                    display(adv_pivot.round(4))

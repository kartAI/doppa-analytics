from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from IPython.display import display

from src.analysis.stats import bootstrap_median_ci

from .style import PALETTE, StyleConfig, tint

_ANNOTATION_FACE = tint(PALETTE["thesisamber"], 0.7)


def _savefig(fig: plt.Figure, name: str, figures_dir: Path) -> None:
    fig.savefig(figures_dir / f"{name}.png", bbox_inches="tight", pad_inches=0.15)
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


# ── Coverage / failure heatmap ──────────────────────────────────────────


def plot_coverage_heatmap(
    samples_df: pd.DataFrame,
    successful: pd.DataFrame,
    style: StyleConfig,
    figures_dir: Path,
    expected_configs: dict[str, list[str]] | None = None,
    metadata_df: pd.DataFrame | None = None,
    experiments: dict[str, dict] | None = None,
) -> None:
    all_wts = sorted(samples_df["workload_type"].unique())
    all_sizes = sorted(
        samples_df["dataset_size"].unique(),
        key=lambda s: style.size_order.get(s, 99),
    )

    if metadata_df is not None and experiments is not None:
        for _, row in metadata_df.iterrows():
            qid = row.get("query_id", "")
            exp = experiments.get(qid)
            if exp is None:
                continue
            ds = exp.get("dataset_size", "")
            if ds and ds not in all_sizes:
                all_sizes.append(ds)
        all_sizes = sorted(all_sizes, key=lambda s: style.size_order.get(s, 99))

    if expected_configs is None:
        expected_configs = {}
        for wt in all_wts:
            expected_configs[wt] = sorted(
                samples_df[samples_df["workload_type"] == wt][
                    "configuration"
                ].unique()
            )
        if metadata_df is not None and experiments is not None:
            from src.analysis.loading import parse_query_id
            for _, row in metadata_df.iterrows():
                qid = row.get("query_id", "")
                try:
                    parsed = parse_query_id(qid, experiments, all_wts)
                except ValueError:
                    continue
                if parsed is None:
                    continue
                wt, cfg, ds = parsed
                if wt not in expected_configs:
                    expected_configs[wt] = []
                if cfg not in expected_configs[wt]:
                    expected_configs[wt].append(cfg)
            for wt in expected_configs:
                expected_configs[wt] = sorted(expected_configs[wt])

    row_labels = []
    col_labels = []
    for wt in all_wts:
        for cfg in expected_configs.get(wt, []):
            row_labels.append((wt, cfg))
    for ds in all_sizes:
        col_labels.append(ds)

    attempted_set: set[tuple[str, str, str]] = set()
    if metadata_df is not None and experiments is not None:
        from src.analysis.loading import parse_query_id
        for _, row in metadata_df.iterrows():
            qid = row.get("query_id", "")
            try:
                parsed = parse_query_id(qid, experiments, all_wts)
            except ValueError:
                continue
            if parsed is not None:
                attempted_set.add(parsed)

    rate_matrix = np.full((len(row_labels), len(col_labels)), np.nan)
    total_matrix = np.full((len(row_labels), len(col_labels)), 0.0)
    success_matrix = np.full((len(row_labels), len(col_labels)), 0.0)

    for ri, (wt, cfg) in enumerate(row_labels):
        for ci, ds in enumerate(col_labels):
            total_mask = (
                (samples_df["workload_type"] == wt)
                & (samples_df["configuration"] == cfg)
                & (samples_df["dataset_size"] == ds)
            )
            n_total = total_mask.sum()
            total_matrix[ri, ci] = n_total

            succ_mask = (
                (successful["workload_type"] == wt)
                & (successful["configuration"] == cfg)
                & (successful["dataset_size"] == ds)
            )
            n_success = succ_mask.sum()
            success_matrix[ri, ci] = n_success

            if n_total > 0:
                rate_matrix[ri, ci] = n_success / n_total
            elif (wt, cfg, ds) in attempted_set:
                rate_matrix[ri, ci] = 0.0

    cmap = LinearSegmentedColormap.from_list(
        "coverage",
        [PALETTE["thesisbrick"], PALETTE["thesisamber"], PALETTE["thesissage"]],
        N=256,
    )
    cmap.set_bad(color=PALETTE["thesislight"])

    n_rows = len(row_labels)
    fig, ax = plt.subplots(
        figsize=(max(7, 2.0 * len(col_labels)), max(6, 0.5 * n_rows + 1.5))
    )
    im = ax.imshow(
        rate_matrix,
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(
        [ds.capitalize() for ds in col_labels], fontsize=11, fontweight="bold"
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_yticks(range(n_rows))

    y_labels_formatted = []
    prev_wt = None
    for wt, cfg in row_labels:
        label = style.label(cfg)
        if wt != prev_wt:
            label = f"{style.workload_label(wt)} — {label}"
        prev_wt = wt
        y_labels_formatted.append(label)

    ax.set_yticklabels(y_labels_formatted, fontsize=8)

    for ri in range(len(row_labels)):
        for ci in range(len(col_labels)):
            wt, cfg = row_labels[ri]
            ds = col_labels[ci]
            n_tot = int(total_matrix[ri, ci])
            n_succ = int(success_matrix[ri, ci])
            rate = rate_matrix[ri, ci]
            was_attempted = (wt, cfg, ds) in attempted_set

            if n_tot == 0 and not was_attempted:
                text = "N/A"
                color = PALETTE["thesisgray"]
            elif n_tot == 0 and was_attempted:
                text = "FAILED\n(0 iters)"
                color = "white"
            elif rate == 1.0:
                text = f"{n_succ}"
                color = "white" if n_succ > 0 else PALETTE["thesisslate"]
            elif rate == 0.0:
                text = f"0/{n_tot}\nFAILED"
                color = "white"
            else:
                text = f"{n_succ}/{n_tot}\n({rate:.0%})"
                color = PALETTE["thesisslate"] if rate > 0.5 else "white"

            ax.text(
                ci, ri, text,
                ha="center", va="center",
                fontsize=7, fontweight="bold",
                color=color,
            )

    wt_boundaries = []
    prev_wt = None
    for ri, (wt, _) in enumerate(row_labels):
        if wt != prev_wt and prev_wt is not None:
            wt_boundaries.append(ri - 0.5)
        prev_wt = wt
    for y in wt_boundaries:
        ax.axhline(y=y, color="white", linewidth=2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Success rate", fontsize=10)

    ax.set_title(
        "Benchmark Coverage — Success Rate by Configuration",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    _savefig(fig, "coverage_heatmap", figures_dir)


# ── Per-cell charts (iterated over workload_type x dataset_size) ─────────


def _annotate_missing(
    ax: plt.Axes,
    present_configs: list[str],
    all_configs: list[str],
    style: StyleConfig,
    y_frac: float = 0.02,
) -> None:
    missing = [c for c in all_configs if c not in present_configs]
    if not missing:
        return
    labels = ", ".join(style.label(c) for c in missing)
    ax.annotate(
        f"No data: {labels}",
        xy=(0.5, y_frac),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=7,
        fontstyle="italic",
        color=PALETTE["thesisbrick"],
        bbox=dict(boxstyle="round,pad=0.3", fc=_ANNOTATION_FACE, ec=PALETTE["thesiscoral"], alpha=0.85),
    )


def plot_median_bars(
    successful: pd.DataFrame,
    primary_metrics: list[str],
    style: StyleConfig,
    figures_dir: Path,
    all_configs: list[str] | None = None,
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
            n_cfgs = len(configs)
            fig_h = max(4, 1.2 * n_cfgs + 1)
            fig, axes = plt.subplots(
                1, n_metrics, figsize=(5 * n_metrics, fig_h)
            )
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
                    ecolor=PALETTE["thesisslate"],
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
                if all_configs is not None:
                    _annotate_missing(ax, configs, all_configs, style)

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
    all_configs: list[str] | None = None,
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
                figsize=(max(8, 2.5 * len(configs_present)), max(5, 6))
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
            parts["cmedians"].set_color(PALETTE["thesisslate"])
            parts["cmedians"].set_linewidth(1.5)

            q1s = [np.percentile(d, 25) for d in bp_data]
            q3s = [np.percentile(d, 75) for d in bp_data]
            ax.vlines(
                range(len(bp_data)),
                q1s,
                q3s,
                color=PALETTE["thesisslate"],
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
            if all_configs is not None:
                _annotate_missing(ax, configs_present, all_configs, style)
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
    all_configs: list[str] | None = None,
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

            fig_h = max(4, 1.2 * len(configs_present) + 1)
            fig, ax = plt.subplots(
                figsize=(max(8, 2 * len(configs_present)), fig_h)
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
                color=PALETTE["thesisgray"],
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
            if all_configs is not None:
                _annotate_missing(ax, configs_present, all_configs, style)
            fig.tight_layout()
            _savefig(fig, f"{wt}_{ds}_cpu", figures_dir)


def plot_network_io(
    successful: pd.DataFrame,
    style: StyleConfig,
    figures_dir: Path,
    all_configs: list[str] | None = None,
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

            fig, axes = plt.subplots(1, 2, figsize=(13, 6))
            fig.subplots_adjust(wspace=0.35)
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
                        line.set_color(PALETTE["thesisgray"])
                for med in bplot["medians"]:
                    med.set_color(PALETTE["thesisslate"])
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
            if all_configs is not None:
                _annotate_missing(
                    axes[0], configs_present, all_configs, style
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

            all_cvs = cv_df["cv"].dropna()
            max_cv = float(all_cvs.max()) if len(all_cvs) > 0 else 1
            use_log_cv = max_cv > 2.0

            fig, ax = plt.subplots(figsize=(max(10, 2 * n_metrics), 5.5))
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
                bars = ax.bar(
                    x + offset,
                    cvs,
                    width * 0.9,
                    label=style.label(cfg),
                    color=style.color(cfg),
                    edgecolor="white",
                    linewidth=0.5,
                )
                if use_log_cv:
                    for bar, cv in zip(bars, cvs):
                        if cv > 0:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                cv,
                                f"{cv:.1f}",
                                ha="center",
                                va="bottom",
                                fontsize=6,
                                fontweight="bold",
                            )
            if use_log_cv:
                ax.set_yscale("symlog", linthresh=0.5)
            ax.axhline(
                y=0.10,
                color=PALETTE["thesisbrick"],
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
        missing_pairs = []
        for wt in wts_present:
            for cfg in configs:
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
                else:
                    missing_pairs.append((wt, cfg))

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
            figsize=(max(8, 2.5 * len(wts_in_data)), 5.5)
        )
        x = np.arange(len(wts_in_data))
        width = 0.8 / max(len(configs), len(cfgs_in_data))

        for i, cfg in enumerate(configs):
            meds = []
            has_data_flags = []
            for wt in wts_in_data:
                match = cw[(cw["workload"] == wt) & (cw["config"] == cfg)]
                if len(match) > 0:
                    meds.append(float(match["median"].iloc[0]))
                    has_data_flags.append(True)
                else:
                    meds.append(0)
                    has_data_flags.append(False)

            offset = (i - len(configs) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                meds,
                width * 0.9,
                label=style.label(cfg),
                color=style.color(cfg),
                edgecolor="white",
                linewidth=0.5,
            )
            for j, (bar, med, has_data) in enumerate(
                zip(bars, meds, has_data_flags)
            ):
                if has_data and med > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        med,
                        f"{_fmt_value(med)}s",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                    )
                elif not has_data:
                    y_pos = ax.get_ylim()[0]
                    ax.annotate(
                        "N/A",
                        xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                        fontsize=6,
                        fontweight="bold",
                        fontstyle="italic",
                        color=PALETTE["thesisbrick"],
                        ha="center",
                        va="bottom",
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

        if missing_pairs:
            missing_labels = set()
            for wt, cfg in missing_pairs:
                if cfg in configs:
                    missing_labels.add(
                        f"{style.label(cfg)} @ {style.workload_label(wt)}"
                    )
            if missing_labels:
                note = "No data: " + "; ".join(sorted(missing_labels))
                ax.annotate(
                    note,
                    xy=(0.5, 0.01),
                    xycoords="axes fraction",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontstyle="italic",
                    color=PALETTE["thesisbrick"],
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc=_ANNOTATION_FACE,
                        ec=PALETTE["thesiscoral"],
                        alpha=0.85,
                    ),
                )

        fig.tight_layout()
        _savefig(fig, f"cross_workload_{ds}", figures_dir)


def plot_scaling_curve(
    rq2_scaling: pd.DataFrame,
    rq2_single_node: pd.DataFrame,
    style: StyleConfig,
    figures_dir: Path,
    failed_configs: pd.DataFrame | None = None,
) -> None:
    if len(rq2_scaling) == 0:
        return

    for ds in rq2_scaling["dataset_size"].unique():
        ds_data = rq2_scaling[rq2_scaling["dataset_size"] == ds]
        if ds_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5.5))
        for strategy in sorted(ds_data["strategy"].unique()):
            s = ds_data[ds_data["strategy"] == strategy].sort_values(
                "worker_count"
            )
            col = style.strategy_colors.get(strategy, PALETTE["thesisgray"])
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

        if failed_configs is not None:
            ds_failed = failed_configs[failed_configs["dataset_size"] == ds]
            if len(ds_failed) > 0:
                failed_labels = []
                for _, row in ds_failed.iterrows():
                    failed_labels.append(
                        f"{row['strategy'].capitalize()} ({row['worker_count']}N)"
                    )
                note = "Failed (all runs): " + ", ".join(sorted(set(failed_labels)))
                ax.annotate(
                    note,
                    xy=(0.5, 0.01),
                    xycoords="axes fraction",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontstyle="italic",
                    color=PALETTE["thesisbrick"],
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc=_ANNOTATION_FACE,
                        ec=PALETTE["thesiscoral"],
                        alpha=0.85,
                    ),
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
        n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows),
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.3)
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
                    line.set_color(PALETTE["thesisgray"])
            for med in bplot["medians"]:
                med.set_color(PALETTE["thesisslate"])
                med.set_linewidth(1.5)

        if "bytes" in metric:
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _, m=metric: _fmt_value(x, m))
            )
        ax.set_xticklabels(
            bp_labels, rotation=35, ha="right", fontsize=8
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
    )
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

        fig_w = max(9, 1.8 * len(configs) + 2)
        fig, ax = plt.subplots(figsize=(fig_w, 6))
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
                color=style.size_colors.get(ds, PALETTE["thesisgray"]),
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, med in zip(bars, meds):
                if med > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        med,
                        f"{_fmt_value(med)}s",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                    )

        ax.set_xticks(x)
        rotation = 35 if len(configs) > 4 else 20
        ax.set_xticklabels(
            [style.label(cfg) for cfg in configs],
            rotation=rotation,
            ha="right",
            fontsize=8 if len(configs) > 4 else 9,
        )
        ax.set_ylabel(style.metric_label("elapsed_time"))
        ax.set_yscale("log")
        ax.set_title(
            f"{style.workload_label(wt)} — Median Elapsed Time by Dataset Size"
        )
        ax.legend(title="Dataset size", loc="best", fontsize=9)
        fig.tight_layout()
        _savefig(fig, f"{wt}_size_scaling", figures_dir)


def plot_cost_bars(
    cost_summary: pd.DataFrame,
    style: StyleConfig,
    figures_dir: Path,
    suffix: str = "",
) -> None:
    if cost_summary.empty:
        return

    for wt in cost_summary.index.get_level_values("workload_type").unique():
        for ds in cost_summary.index.get_level_values("dataset_size").unique():
            if (wt, ds) not in [(w, d) for w, d, _ in cost_summary.index]:
                continue

            data = cost_summary.loc[(wt, ds)]
            if isinstance(data, pd.Series):
                data = data.to_frame().T

            configs = [idx if isinstance(idx, str) else idx[-1] for idx in data.index]
            totals = data["total_cost"].values

            fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(configs)), max(4, 1.2 * len(configs) + 1)))
            y = np.arange(len(configs))
            colors = [style.color(c) for c in configs]
            ax.barh(y, totals, color=colors, edgecolor="white", linewidth=0.5, height=0.6)
            for yi, val in zip(y, totals):
                ax.annotate(
                    f"${val:.4f}",
                    xy=(val, yi),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )
            ax.set_yticks(y)
            ax.set_yticklabels([style.label(c) for c in configs])
            ax.set_xlabel("Total cost (USD)")
            ax.set_title(f"{style.workload_label(wt)} ({ds}) — Infrastructure Cost")
            ax.invert_yaxis()
            fig.tight_layout()
            tag = f"_{suffix}" if suffix else ""
            _savefig(fig, f"{wt}_{ds}_cost{tag}", figures_dir)


def plot_cost_breakdown(
    cost_summary: pd.DataFrame,
    style: StyleConfig,
    figures_dir: Path,
    suffix: str = "",
) -> None:
    if cost_summary.empty:
        return

    breakdown_cols = ["compute_cost", "storage_cost", "network_cost", "operations_cost"]
    available = [c for c in breakdown_cols if c in cost_summary.columns]
    if not available:
        return

    cat_labels = {
        "compute_cost": "Compute",
        "storage_cost": "Storage",
        "network_cost": "Network",
        "operations_cost": "Operations",
    }

    for wt in cost_summary.index.get_level_values("workload_type").unique():
        for ds in cost_summary.index.get_level_values("dataset_size").unique():
            if (wt, ds) not in [(w, d) for w, d, _ in cost_summary.index]:
                continue

            data = cost_summary.loc[(wt, ds)]
            if isinstance(data, pd.Series):
                data = data.to_frame().T

            configs = [idx if isinstance(idx, str) else idx[-1] for idx in data.index]
            fig_h = max(4, 1.2 * len(configs) + 1)
            fig, ax = plt.subplots(figsize=(max(8, 2 * len(configs)), fig_h))
            y = np.arange(len(configs))
            left = np.zeros(len(configs))

            for col in available:
                vals = data[col].values.astype(float)
                ax.barh(
                    y, vals, left=left, height=0.6,
                    label=cat_labels.get(col, col),
                    color=style.cost_category_colors.get(col, PALETTE["thesisgray"]),
                    edgecolor="white", linewidth=0.5,
                )
                left += vals

            for yi, total in zip(y, left):
                ax.annotate(
                    f"${total:.4f}",
                    xy=(total, yi),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

            ax.set_yticks(y)
            ax.set_yticklabels([style.label(c) for c in configs])
            ax.set_xlabel("Cost (USD)")
            ax.set_title(f"{style.workload_label(wt)} ({ds}) — Cost Breakdown")
            ax.legend(loc="best", fontsize=9)
            ax.invert_yaxis()
            fig.tight_layout()
            tag = f"_{suffix}" if suffix else ""
            _savefig(fig, f"{wt}_{ds}_cost_breakdown{tag}", figures_dir)


def plot_cost_performance(
    cost_summary: pd.DataFrame,
    successful: pd.DataFrame,
    style: StyleConfig,
    figures_dir: Path,
    suffix: str = "",
) -> None:
    if cost_summary.empty:
        return

    for wt in cost_summary.index.get_level_values("workload_type").unique():
        for ds in cost_summary.index.get_level_values("dataset_size").unique():
            if (wt, ds) not in [(w, d) for w, d, _ in cost_summary.index]:
                continue

            cost_data = cost_summary.loc[(wt, ds)]
            if isinstance(cost_data, pd.Series):
                cost_data = cost_data.to_frame().T

            perf_mask = (
                (successful["workload_type"] == wt)
                & (successful["dataset_size"] == ds)
            )
            perf_data = successful[perf_mask]

            points = []
            for idx in cost_data.index:
                cfg = idx if isinstance(idx, str) else idx[-1]
                vals = perf_data[perf_data["configuration"] == cfg]["elapsed_time"].dropna().values
                if len(vals) == 0:
                    continue
                points.append({
                    "config": cfg,
                    "total_cost": float(cost_data.loc[idx, "total_cost"]),
                    "median_elapsed": float(np.median(vals)),
                })

            if len(points) < 2:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            for p in points:
                ax.scatter(
                    p["total_cost"], p["median_elapsed"],
                    color=style.color(p["config"]),
                    s=120, zorder=5, edgecolors="white", linewidth=1,
                )
                ax.annotate(
                    style.label(p["config"]),
                    xy=(p["total_cost"], p["median_elapsed"]),
                    xytext=(8, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
            ax.set_xlabel("Total infrastructure cost (USD)")
            ax.set_ylabel("Median elapsed time (s)")
            ax.set_title(f"{style.workload_label(wt)} ({ds}) — Cost vs. Performance")

            costs_arr = [p["total_cost"] for p in points]
            times_arr = [p["median_elapsed"] for p in points]
            if max(costs_arr) / max(min(costs_arr), 1e-9) > 20:
                ax.set_xscale("log")
            if max(times_arr) / max(min(times_arr), 1e-9) > 20:
                ax.set_yscale("log")

            fig.tight_layout()
            tag = f"_{suffix}" if suffix else ""
            _savefig(fig, f"{wt}_{ds}_cost_vs_perf{tag}", figures_dir)


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

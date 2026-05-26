from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import seaborn as sns


# ── Tint / shade helpers ──────────────────────────────────────────────────


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return (
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
    )


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        int(round(min(max(r, 0), 1) * 255)),
        int(round(min(max(g, 0), 1) * 255)),
        int(round(min(max(b, 0), 1) * 255)),
    )


def tint(hex_color: str, amount: float) -> str:
    """Mix toward white.  0 = unchanged, 1 = white."""
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex(
        r + (1.0 - r) * amount,
        g + (1.0 - g) * amount,
        b + (1.0 - b) * amount,
    )


def shade(hex_color: str, amount: float) -> str:
    """Mix toward black.  0 = unchanged, 1 = black."""
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex(r * (1 - amount), g * (1 - amount), b * (1 - amount))


def lightness_ramp(
    hex_color: str,
    n: int,
    tint_max: float = 0.55,
    shade_max: float = 0.15,
) -> list[str]:
    """Return *n* colors from lightest (tinted) to darkest (shaded)."""
    colors: list[str] = []
    for i in range(n):
        t = i / max(n - 1, 1)
        if t <= 0.5:
            colors.append(tint(hex_color, tint_max * (1.0 - t * 2)))
        else:
            colors.append(shade(hex_color, shade_max * ((t - 0.5) * 2)))
    return colors


# ── Canonical thesis palette (single source of truth) ─────────────────────

PALETTE: dict[str, str] = {
    "thesisteal": "#4F8985",
    "thesiscoral": "#B7704D",
    "thesissteel": "#5B8AAC",
    "thesisamber": "#C9A36A",
    "thesisviolet": "#8E7BA3",
    "thesissage": "#6E8F76",
    "thesisslate": "#2C3E50",
    "thesisgray": "#7F7F7F",
    "thesislight": "#E5E5E5",
    "thesispale": "#F5F5F5",
    "thesisbrick": "#A85A4A",
}


# ── Derived semantic color maps ───────────────────────────────────────────

_STRATEGIES = ("broadcast", "partitioned", "default")
_STRATEGY_BASES: dict[str, str] = {
    "broadcast": PALETTE["thesissteel"],
    "partitioned": PALETTE["thesissage"],
    "default": PALETTE["thesisgray"],
}
_NODE_COUNTS = (2, 4, 8, 12, 16)


def _build_config_colors() -> dict[str, str]:
    colors: dict[str, str] = {
        "duckdb": PALETTE["thesisteal"],
        "postgis": PALETTE["thesiscoral"],
        "local": PALETTE["thesisamber"],
    }
    for strategy, base in _STRATEGY_BASES.items():
        ramp = lightness_ramp(base, len(_NODE_COUNTS))
        for nc, c in zip(_NODE_COUNTS, ramp):
            colors[f"databricks-{strategy}-{nc}-nodes"] = c
    return colors


def _build_size_colors() -> dict[str, str]:
    return {
        "small": tint(PALETTE["thesissteel"], 0.6),
        "medium": tint(PALETTE["thesissteel"], 0.3),
        "large": PALETTE["thesissteel"],
    }


def _build_strategy_colors() -> dict[str, str]:
    return {s: _STRATEGY_BASES[s] for s in _STRATEGIES}


def _build_status_colors() -> dict[str, str]:
    return {
        "success": PALETTE["thesissage"],
        "warning": PALETTE["thesisamber"],
        "failure": PALETTE["thesisbrick"],
    }


def _build_cost_category_colors() -> dict[str, str]:
    return {
        "compute_cost": PALETTE["thesissteel"],
        "storage_cost": PALETTE["thesissage"],
        "network_cost": PALETTE["thesisamber"],
        "operations_cost": PALETTE["thesisviolet"],
    }


# ── Style configuration ──────────────────────────────────────────────────


@dataclass
class StyleConfig:
    config_colors: dict[str, str] = field(default_factory=_build_config_colors)

    config_labels: dict[str, str] = field(
        default_factory=lambda: {
            "local": "Local (Python)",
            "postgis": "PostGIS",
            "duckdb": "DuckDB",
            "databricks-broadcast-2-nodes": "Databricks Broadcast (2N)",
            "databricks-broadcast-4-nodes": "Databricks Broadcast (4N)",
            "databricks-broadcast-8-nodes": "Databricks Broadcast (8N)",
            "databricks-broadcast-12-nodes": "Databricks Broadcast (12N)",
            "databricks-broadcast-16-nodes": "Databricks Broadcast (16N)",
            "databricks-partitioned-2-nodes": "Databricks Partitioned (2N)",
            "databricks-partitioned-4-nodes": "Databricks Partitioned (4N)",
            "databricks-partitioned-8-nodes": "Databricks Partitioned (8N)",
            "databricks-partitioned-12-nodes": "Databricks Partitioned (12N)",
            "databricks-partitioned-16-nodes": "Databricks Partitioned (16N)",
            "databricks-default-2-nodes": "Databricks Default (2N)",
            "databricks-default-4-nodes": "Databricks Default (4N)",
            "databricks-default-8-nodes": "Databricks Default (8N)",
            "databricks-default-12-nodes": "Databricks Default (12N)",
            "databricks-default-16-nodes": "Databricks Default (16N)",
        }
    )

    metric_labels: dict[str, str] = field(
        default_factory=lambda: {
            "elapsed_time": "Elapsed time (s)",
            "network_bytes_received": "Network received (bytes)",
            "network_bytes_sent": "Network sent (bytes)",
            "cpu_time_user_seconds": "CPU user time (s)",
            "cpu_time_system_seconds": "CPU system time (s)",
            "executor_input_bytes_read": "Executor input bytes read",
            "executor_run_time_ms": "Executor run time (ms)",
            "shuffle_read_bytes": "Shuffle read (bytes)",
            "shuffle_write_bytes": "Shuffle write (bytes)",
            "driver_collection_time_ms": "Driver collection time (ms)",
        }
    )

    workload_labels: dict[str, str] = field(
        default_factory=lambda: {
            "bbox-filtering": "Bounding-box Filtering",
            "point-in-polygon-lookup": "Point-in-polygon Lookup",
            "knn-search": "k-NN Search",
            "national-scale-spatial-join": "National-scale Spatial Join",
        }
    )

    size_order: dict[str, int] = field(
        default_factory=lambda: {"small": 0, "medium": 1, "large": 2}
    )

    size_colors: dict[str, str] = field(default_factory=_build_size_colors)
    strategy_colors: dict[str, str] = field(default_factory=_build_strategy_colors)
    status_colors: dict[str, str] = field(default_factory=_build_status_colors)
    cost_category_colors: dict[str, str] = field(
        default_factory=_build_cost_category_colors
    )

    fallback_color: str = PALETTE["thesisgray"]

    def color(self, cfg: str) -> str:
        return self.config_colors.get(cfg, self.fallback_color)

    def label(self, cfg: str) -> str:
        return self.config_labels.get(cfg, cfg)

    def metric_label(self, m: str) -> str:
        return self.metric_labels.get(m, m)

    def workload_label(self, w: str) -> str:
        return self.workload_labels.get(w, w)

    def apply_rcparams(self) -> None:
        sns.set_theme(
            style="ticks",
            rc={
                "font.family": "sans-serif",
                "font.sans-serif": [
                    "Source Sans 3",
                    "Inter",
                    "Liberation Sans",
                    "Arial",
                    "DejaVu Sans",
                ],
                "mathtext.fontset": "dejavusans",
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.titleweight": "normal",
                "axes.labelsize": 10,
                "axes.labelcolor": PALETTE["thesisslate"],
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 0.6,
                "axes.grid": False,
                "grid.alpha": 0.15,
                "grid.linestyle": "-",
                "grid.linewidth": 0.4,
                "grid.color": PALETTE["thesislight"],
                "text.color": PALETTE["thesisslate"],
                "xtick.color": PALETTE["thesisslate"],
                "ytick.color": PALETTE["thesisslate"],
                "xtick.major.width": 0.6,
                "ytick.major.width": 0.6,
                "xtick.major.size": 3,
                "ytick.major.size": 3,
                "legend.frameon": False,
                "legend.fontsize": 9,
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "savefig.format": "png",
                "savefig.bbox": "tight",
            },
        )


DEFAULT_STYLE = StyleConfig()

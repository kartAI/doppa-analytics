from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt


@dataclass
class StyleConfig:
    config_colors: dict[str, str] = field(
        default_factory=lambda: {
            "local": "#2196F3",
            "postgis": "#4CAF50",
            "duckdb": "#FF9800",
            "databricks-broadcast-2-nodes": "#E91E63",
            "databricks-broadcast-4-nodes": "#E91E63",
            "databricks-broadcast-8-nodes": "#AB47BC",
            "databricks-broadcast-12-nodes": "#1565C0",
            "databricks-broadcast-16-nodes": "#7B1FA2",
            "databricks-partitioned-2-nodes": "#00BCD4",
            "databricks-partitioned-4-nodes": "#0097A7",
            "databricks-partitioned-8-nodes": "#00838F",
            "databricks-partitioned-12-nodes": "#006064",
            "databricks-partitioned-16-nodes": "#795548",
            "databricks-default-2-nodes": "#607D8B",
            "databricks-default-4-nodes": "#546E7A",
            "databricks-default-8-nodes": "#455A64",
            "databricks-default-16-nodes": "#F44336",
        }
    )

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

    size_colors: dict[str, str] = field(
        default_factory=lambda: {
            "small": "#66BB6A",
            "medium": "#FFA726",
            "large": "#EF5350",
        }
    )

    strategy_colors: dict[str, str] = field(
        default_factory=lambda: {
            "broadcast": "#E91E63",
            "partitioned": "#00BCD4",
            "default": "#607D8B",
        }
    )

    fallback_color: str = "#999999"

    def color(self, cfg: str) -> str:
        return self.config_colors.get(cfg, self.fallback_color)

    def label(self, cfg: str) -> str:
        return self.config_labels.get(cfg, cfg)

    def metric_label(self, m: str) -> str:
        return self.metric_labels.get(m, m)

    def workload_label(self, w: str) -> str:
        return self.workload_labels.get(w, w)

    def apply_rcparams(self) -> None:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 10,
                "axes.titlesize": 13,
                "axes.titleweight": "bold",
                "axes.labelsize": 11,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "grid.linestyle": "--",
                "legend.framealpha": 0.9,
                "legend.edgecolor": "0.8",
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "savefig.format": "png",
                "savefig.bbox": "tight",
            }
        )


DEFAULT_STYLE = StyleConfig()

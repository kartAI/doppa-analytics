from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Patch, Wedge
from matplotlib.path import Path as MplPath


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


# ── Line-weight scale (single source of truth) ─────────────────────────────
#
# One small set of stroke widths, kept at or slightly below the hairline weights
# of the hand-drawn Lucidchart diagrams (Chapters 02/04/05) so every generated
# figure reads in the same light register. The conceptual primitives below and
# the result figures in ``thesis_figures.py`` both pull from these instead of
# hardcoding linewidths, so a single edit here re-weights the whole figure set.
# ``apply_rcparams`` wires :data:`LW_SPINE` / :data:`LW_MARKER_EDGE` into the
# matplotlib defaults too, so anything that does not pass an explicit linewidth
# (e.g. ``charts.py``) inherits the same register.
LW_SPINE: float = 0.5        # axis spines + tick marks
LW_HAIRLINE: float = 0.4     # faint separators, ribbon/wedge edges, gridlines
LW_BORDER: float = 0.7       # bar / box / patch / heatmap-cell borders
LW_NODE: float = 0.9         # conceptual rounded-node + legend-swatch borders
LW_CONNECTOR: float = 0.9    # flow arrows, reference / ideal / target lines
LW_SERIES: float = 1.3       # data polylines (ECDF, speedup, slopegraph, …)
LW_MARKER_EDGE: float = 0.4  # marker edge stroke


# ── Fonts ──────────────────────────────────────────────────────────────────
#
# The thesis *body* is set in Computer Modern (no explicit font package in
# ``packages.sty``), but the figure convention deliberately uses a humanist
# sans-serif to match the Lucidchart conceptual diagrams (Chapters 02/04/05),
# whose face is Lato. ``Source Sans 3`` is the intended figure face; of the
# stack only ``Open Sans`` (a near-twin of Lato) is installed in this
# environment, so figures resolve to Open Sans — a much closer match to the
# conceptual diagrams than the previous DejaVu Sans fallback. See ``CLAUDE.md``.
SANS_STACK: list[str] = [
    "Source Sans 3",
    "Open Sans",
    "Inter",
    "Lato",
    "Liberation Sans",
    "Arial",
    "DejaVu Sans",
]
MONO_STACK: list[str] = [
    "JetBrains Mono",
    "DejaVu Sans Mono",
    "Liberation Mono",
    "monospace",
]

# Repo-local font drop (optional): any *.ttf placed here is registered with
# matplotlib so the preferred face resolves even on a fresh machine.
_FONT_DIR = Path(__file__).resolve().parent / "fonts"


def register_fonts() -> None:
    """Register any bundled ``fonts/*.ttf`` with matplotlib's font manager.

    System fonts (Open Sans here) are already discovered by matplotlib; this
    only adds repo-bundled faces so a machine without Open Sans can still
    resolve the intended figure font by dropping the TTF into ``fonts/``.
    Idempotent and silent on a missing directory.
    """
    if not _FONT_DIR.is_dir():
        return
    for ttf in _FONT_DIR.rglob("*.ttf"):
        try:
            fm.fontManager.addfont(str(ttf))
        except Exception:
            pass


def resolved_sans() -> str:
    """First face in :data:`SANS_STACK` matplotlib can actually resolve.

    Used for reporting which font figures will render in (the intended
    ``Source Sans 3`` vs the installed ``Open Sans`` fallback, etc.).
    """
    register_fonts()
    available = {f.name for f in fm.fontManager.ttflist}
    for name in SANS_STACK:
        if name in available:
            return name
    return "DejaVu Sans"


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
        register_fonts()
        sns.set_theme(
            style="ticks",
            rc={
                "font.family": "sans-serif",
                "font.sans-serif": SANS_STACK,
                "font.monospace": MONO_STACK,
                "mathtext.fontset": "dejavusans",
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.titleweight": "normal",
                "axes.labelsize": 10,
                "axes.labelcolor": PALETTE["thesisslate"],
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": LW_SPINE,
                "axes.grid": False,
                "grid.alpha": 0.15,
                "grid.linestyle": "-",
                "grid.linewidth": LW_HAIRLINE,
                "grid.color": PALETTE["thesislight"],
                "text.color": PALETTE["thesisslate"],
                "xtick.color": PALETTE["thesisslate"],
                "ytick.color": PALETTE["thesisslate"],
                "xtick.major.width": LW_SPINE,
                "ytick.major.width": LW_SPINE,
                "xtick.minor.width": LW_HAIRLINE,
                "ytick.minor.width": LW_HAIRLINE,
                "xtick.major.size": 3,
                "ytick.major.size": 3,
                # defaults for artists that do not pass an explicit linewidth
                # (keeps charts.py and any incidental patch/marker in register)
                "patch.linewidth": LW_BORDER,
                "lines.markeredgewidth": LW_MARKER_EDGE,
                "hatch.linewidth": LW_HAIRLINE,
                "legend.frameon": False,
                "legend.fontsize": 9,
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "savefig.format": "png",
                "savefig.bbox": "tight",
                # Embed fonts as TrueType (42), not Type 3, in vector PDF/PS
                # output — required for print (some book printers reject Type 3).
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            },
        )


DEFAULT_STYLE = StyleConfig()


# ════════════════════════════════════════════════════════════════════════════
# Conceptual-diagram language (rounded nodes, ribbons, bottom legend)
#
# Primitives reproducing the visual language of the hand-drawn Lucidchart
# diagrams (Chapters 02/04/05): rounded nodes with a ~25%-strength tinted fill,
# a same-hue border and bold-slate text + gray subtitle; smooth flow ribbons
# whose thickness encodes a magnitude (Sankey); control-flow arrows in slate;
# and a bottom legend of small rounded swatches. Color comes only from PALETTE.
# ════════════════════════════════════════════════════════════════════════════


# Semantic roles read off the conceptual diagrams. Cool = cloud-native, warm =
# traditional; gray = infrastructure/process; amber = decision; sage = positive
# terminal; brick = capped/failed terminal. (thesisviolet is unused here.)
ROLE_COLORS: dict[str, str] = {
    "infrastructure": PALETTE["thesisgray"],   # process / control / setting
    "cloudnative": PALETTE["thesisteal"],      # DuckDB / GeoParquet
    "distributed": PALETTE["thesissteel"],     # Sedona physical plan / Databricks
    "traditional": PALETTE["thesiscoral"],     # PostGIS / Shapefile / GeoPandas
    "decision": PALETTE["thesisamber"],        # decision gate / seeded process
    "positive": PALETTE["thesissage"],         # storage / outcome / converged stop
    "negative": PALETTE["thesisbrick"],        # capped / failed terminal
}


@dataclass(frozen=True)
class NodeStyle:
    """Resolved fill convention for a diagram node / legend swatch."""

    face: str
    edge: str
    text: str


def fill_style(
    color: str,
    *,
    strength: float = 0.25,
    edge: str | None = None,
    text: str | None = None,
) -> NodeStyle:
    """Tinted-fill convention: a light fill at ``strength`` of *color* (≈25% by
    default, matching the conceptual diagrams), a same-hue border and dark-slate
    text. ``strength`` is the share of the source color in the fill (0 = white,
    1 = full color)."""
    return NodeStyle(
        face=tint(color, 1.0 - strength),
        edge=edge or color,
        text=text or PALETTE["thesisslate"],
    )


def role_style(role: str, *, strength: float = 0.25) -> NodeStyle:
    """:class:`NodeStyle` for a semantic role (see :data:`ROLE_COLORS`).

    ``infrastructure`` is special-cased to the neutral gray node of the
    reference diagrams (``thesislight`` fill, ``thesisgray`` border)."""
    if role == "infrastructure":
        return NodeStyle(
            face=PALETTE["thesislight"],
            edge=PALETTE["thesisgray"],
            text=PALETTE["thesisslate"],
        )
    base = ROLE_COLORS.get(role, PALETTE["thesisgray"])
    return fill_style(base, strength=strength)


def two_line_title(
    target, title: str, subtitle: str | None = None, *, y: float | None = None
) -> None:
    """Bold-slate title with an optional smaller gray subtitle beneath it.

    ``target`` may be a :class:`~matplotlib.figure.Figure` (suptitle + fig.text)
    or an :class:`~matplotlib.axes.Axes` (set_title + above-axes annotation)."""
    slate, gray = PALETTE["thesisslate"], PALETTE["thesisgray"]
    if isinstance(target, plt.Axes):
        target.set_title(
            title, fontsize=12, fontweight="bold", color=slate,
            pad=20 if subtitle else 8,
        )
        if subtitle:
            target.annotate(
                subtitle, xy=(0.5, 1.0), xycoords="axes fraction",
                xytext=(0, 6), textcoords="offset points", ha="center",
                va="bottom", fontsize=9.5, color=gray,
            )
    else:  # Figure
        yt = y if y is not None else 1.0
        target.suptitle(title, fontsize=13, fontweight="bold", color=slate, y=yt)
        if subtitle:
            target.text(
                0.5, yt - 0.035, subtitle, ha="center", va="top",
                fontsize=10, color=gray, transform=target.transFigure,
            )


class _RoundedSwatchHandler(HandlerPatch):
    """Legend handler drawing a small rounded swatch (tinted fill + same-hue
    border) instead of the default square — matches the conceptual diagrams."""

    def __init__(self, node: NodeStyle, **kw):
        self._node = node
        super().__init__(**kw)

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height,
        fontsize, trans,
    ):
        # Square swatch centred on the handle's optical centre (the matplotlib
        # convention ``height/2 - ydescent/2``), so it sits on the label baseline
        # rather than floating above it.
        s = height * 0.92
        cx = width / 2.0 - xdescent / 2.0
        cy = height / 2.0 - ydescent / 2.0
        box = FancyBboxPatch(
            (cx - s / 2.0, cy - s / 2.0), s, s,
            boxstyle=f"round,pad=0,rounding_size={s * 0.42}",
            facecolor=self._node.face, edgecolor=self._node.edge,
            linewidth=LW_NODE, transform=trans,
        )
        return [box]


def bottom_legend(
    fig, entries, *, ncol: int | None = None, y: float = -0.01,
    fontsize: float = 9.5, **kw,
):
    """Horizontal bottom legend of rounded swatches.

    ``entries`` is a list of ``(label, color_or_NodeStyle)``. A bare color
    string is resolved through :func:`fill_style`; a :class:`NodeStyle` (e.g.
    from :func:`role_style`) is used verbatim."""
    handles, labels, hmap = [], [], {}
    for label, spec in entries:
        node = spec if isinstance(spec, NodeStyle) else fill_style(spec)
        h = Patch()
        handles.append(h)
        labels.append(label)
        hmap[h] = _RoundedSwatchHandler(node)
    leg = fig.legend(
        handles, labels, handler_map=hmap, loc="lower center",
        ncol=ncol or len(entries), frameon=False,
        bbox_to_anchor=(0.5, y), handlelength=1.1, handleheight=1.1,
        columnspacing=1.8, fontsize=fontsize, **kw,
    )
    for t in leg.get_texts():
        t.set_color(PALETTE["thesisslate"])
    return leg


def rounded_node(
    ax, xy, width, height, title, subtitle=None, *,
    role: str | None = None, node: NodeStyle | None = None,
    fontsize: float = 10.5, dashed: bool = False, mono: bool = False,
    z: float = 3,
):
    """Draw a rounded-rectangle node (conceptual-diagram style).

    Fill resolves from ``node`` (a :class:`NodeStyle`) or ``role`` (a key into
    :data:`ROLE_COLORS`); defaults to neutral infrastructure gray. ``dashed``
    draws an inactive/avoided node; ``mono`` renders the title in the monospace
    face for code identifiers."""
    ns = node or (role_style(role) if role else role_style("infrastructure"))
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0,rounding_size={min(width, height) * 0.16}",
        facecolor=ns.face, edgecolor=ns.edge, linewidth=LW_NODE,
        linestyle="--" if dashed else "-", zorder=z,
    )
    ax.add_patch(box)
    cx, cy = x + width / 2, y + height / 2
    tcolor = PALETTE["thesisgray"] if dashed else ns.text
    tfam = "monospace" if mono else "sans-serif"
    if subtitle:
        ax.text(cx, cy + height * 0.13, title, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=tcolor,
                family=tfam, zorder=z + 1)
        ax.text(cx, cy - height * 0.19, subtitle, ha="center", va="center",
                fontsize=fontsize - 2.5, color=PALETTE["thesisgray"], zorder=z + 1)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=tcolor, family=tfam, zorder=z + 1)
    return box


def flow_arrow(
    ax, xy0, xy1, *, color: str | None = None, dashed: bool = False,
    lw: float = LW_CONNECTOR, z: float = 2,
):
    """Slate control-flow arrow between two points (conceptual-diagram style)."""
    ap = FancyArrowPatch(
        xy0, xy1, arrowstyle="-|>", mutation_scale=11,
        color=color or PALETTE["thesisslate"], linewidth=lw,
        linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2, zorder=z,
    )
    ax.add_patch(ap)
    return ap


def ribbon(
    ax, xy0, xy1, w0, w1=None, *, color: str, alpha: float = 0.55,
    edge: str | None = None, z: float = 1,
):
    """Smooth horizontal flow ribbon (Sankey band) between two gates.

    ``xy0``/``xy1`` are the band centerline endpoints; ``w0``/``w1`` its
    thickness (data units) at each end — the flow magnitude. A cubic-Bézier
    S-curve joins them so crossing flows read cleanly."""
    w1 = w0 if w1 is None else w1
    (x0, y0), (x1, y1) = xy0, xy1
    mx = (x0 + x1) / 2.0
    verts = [
        (x0, y0 + w0 / 2),                                       # MOVETO top-start
        (mx, y0 + w0 / 2), (mx, y1 + w1 / 2), (x1, y1 + w1 / 2),  # CURVE4 top
        (x1, y1 - w1 / 2),                                       # LINETO right edge
        (mx, y1 - w1 / 2), (mx, y0 - w0 / 2), (x0, y0 - w0 / 2),  # CURVE4 bottom
        (x0, y0 + w0 / 2),                                       # CLOSEPOLY
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(
        MplPath(verts, codes), facecolor=color,
        edgecolor=edge or "none", linewidth=LW_HAIRLINE if edge else 0.0,
        alpha=alpha, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def loop_ribbon(
    ax, foot_left, foot_right, y, w, *, color: str,
    alpha: float = 0.55, edge: str | None = None, z: float = 1,
):
    """Constant-width self-loop band: an upper half-annulus that leaves a node's
    top edge and re-enters it.

    Models an intra-node exchange (e.g. an intra-cluster Spark shuffle). ``w`` is
    the band thickness (data units) — the flow magnitude, in the convention of
    :func:`ribbon` — held **uniform along the whole loop, feet included** (a
    half-annulus, not a hump that fattens at the top). The two feet sit at
    ``foot_left``/``foot_right`` on the node top (``y``); the arch is a semicircle
    whose centerline radius is half the feet span."""
    cx = (foot_left + foot_right) / 2.0
    rc = (foot_right - foot_left) / 2.0      # centerline radius
    wedge = Wedge(
        (cx, y), rc + w / 2.0, 0.0, 180.0, width=w,
        facecolor=color, edgecolor=edge or "none",
        linewidth=LW_HAIRLINE if edge else 0.0, alpha=alpha, zorder=z,
    )
    ax.add_patch(wedge)
    return wedge

"""Matplotlib style for remo-id figures — Tufte aesthetic.

Default sizes:
  Single column : 89 mm
  Double column : 183 mm
  Max height    : 247 mm

Design principles (after Tufte's *Visual Display of Quantitative Information*):
  - Maximize data-ink; remove chart junk
  - Sans-serif typography (Arial, falls back to Helvetica / DejaVu Sans)
  - Grayscale first; use muted color only for class distinctions
  - Range frames (spines trimmed to data), not bounding boxes
  - Rug ticks show where observations lie
  - Frameless legends, minimal ticks, thin rules

Usage
-----
from remo_id.reports import style
style.apply()                     # set rcParams globally
fig, ax = style.figure()          # single-column figure
style.range_frame(ax)             # trim spines to data
style.rug(ax, x, side="bottom")   # marginal dash ticks
"""

import string

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
# Nature Research figure widths — figures MUST be one of these two widths.
MM = 1 / 25.4          # mm → inches
SINGLE = 89  * MM      # single column
DOUBLE = 183 * MM      # double column
MAX_H  = 170 * MM      # maximum figure height

# ---------------------------------------------------------------------------
# Ink palette — grayscale first
# ---------------------------------------------------------------------------
FG         = "#222222"   # primary ink (data, spines, labels)
MUTE       = "#888888"   # secondary (reference lines, jitter)
MUTE_LIGHT = "#BBBBBB"   # CI bands, soft fills
MUTE_FAINT = "#DDDDDD"   # very-secondary fills

# Single accent — used sparingly
ACCENT = "#D55E00"   # vermillion (Wong[6])

# Muted colorblind-safe palette for class-distinction plots only. Classes are
# colored by index (class 0 -> MUTED_CLASS[0], ...) unless a fixed color was
# registered for their name with :func:`set_class_colors`.
MUTED_CLASS = [
    "#1B5E8A",   # 0 blue
    "#B04E00",   # 1 vermillion
    "#2D7A5E",   # 2 green
    "#B5881A",   # 3 ochre
    "#864B7C",   # 4 plum
    "#3F93B8",   # 5 sky
    "#6B7280",   # 6 slate
    "#8A6E3F",   # 7 olive
]

# Wong-role palette used by :func:`class_colors` and accent-style callers:
#   WONG[1] = orange/accent class ;  WONG[3] = green ;  WONG[5] = blue/primary ;
#   WONG[6] = vermillion (accent) ;  WONG[0] = black/FG
WONG = [
    FG,              # 0  black
    MUTED_CLASS[3],  # 1  orange
    MUTED_CLASS[5],  # 2  sky blue
    MUTED_CLASS[2],  # 3  bluish green
    MUTE,            # 4  yellow
    MUTED_CLASS[0],  # 5  blue (primary)
    ACCENT,          # 6  vermillion
    MUTED_CLASS[4],  # 7  reddish purple
]
PALETTE = MUTED_CLASS  # shorthand

# Redundant-encoding markers (used alongside color for accessibility)
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "P", "X", "*"]

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
_RC = {
    # --- font ---
    "font.family":            "sans-serif",
    "font.sans-serif":        ["Arial", "Helvetica", "Liberation Sans",
                               "Nimbus Sans", "DejaVu Sans"],
    # Type scale at print size: 10 pt titles, 8 pt for everything else,
    # 12 pt subpanel keys (PANEL_LABEL_PT).
    "font.size":              8,
    "axes.titlesize":         10,
    "axes.titleweight":       "bold",
    "axes.titlepad":          6,
    "axes.labelsize":         8,
    "xtick.labelsize":        8,
    "ytick.labelsize":        8,
    "legend.fontsize":        8,
    "legend.title_fontsize":  8,
    # --- ink colors ---
    "text.color":             FG,
    "axes.edgecolor":         FG,
    "axes.labelcolor":        FG,
    "xtick.color":            FG,
    "ytick.color":            FG,
    # --- lines / patches (thin) ---
    "axes.linewidth":         0.4,
    "xtick.major.width":      0.4,
    "ytick.major.width":      0.4,
    "xtick.minor.width":      0.3,
    "ytick.minor.width":      0.3,
    "lines.linewidth":        0.8,
    "lines.markersize":       2.5,
    "patch.linewidth":        0.3,
    "patch.edgecolor":        FG,
    # --- ticks ---
    "xtick.major.size":       2.5,
    "ytick.major.size":       2.5,
    "xtick.minor.size":       1.2,
    "ytick.minor.size":       1.2,
    "xtick.direction":        "out",
    "ytick.direction":        "out",
    "xtick.top":              False,
    "ytick.right":            False,
    # --- grid (off) ---
    "axes.grid":              False,
    # --- spines ---
    "axes.spines.top":        False,
    "axes.spines.right":      False,
    # --- legend ---
    "legend.frameon":         False,
    "legend.borderpad":       0.2,
    "legend.labelspacing":    0.2,
    "legend.handlelength":    1.2,
    "legend.handletextpad":   0.3,
    # --- figure / save ---
    "figure.facecolor":       "white",
    "axes.facecolor":         "white",
    "figure.dpi":             150,
    "savefig.dpi":            450,          # Nature Research: min 450 dpi
    # Save the FULL canvas so files are exactly 89/183 mm wide. (bbox="tight"
    # would re-crop to the content box and break the fixed width.) tight_layout
    # packs content inside the fixed canvas without resizing it.
    "savefig.bbox":           None,
    "savefig.pad_inches":     0.0,
    # --- vector text: embed as editable TrueType, not outlines/Type-3 ---
    "pdf.fonttype":           42,
    "ps.fonttype":            42,
    # --- color cycle: grayscale-first, accent second (non-class plots) ---
    "axes.prop_cycle":        mpl.cycler("color", [FG, ACCENT, MUTE] + MUTED_CLASS),
}


def apply():
    """Apply Tufte style to matplotlib globally."""
    mpl.rcParams.update(_RC)


# Subpanel keys: bold lowercase, 12 pt (see type-scale note in _RC).
PANEL_LABEL_PT = 12


def pt_to_px(pt: float, dpi: float) -> int:
    """Convert a point size to whole pixels at ``dpi`` (for raster compositors
    such as PIL, so their type matches the matplotlib figures at print scale)."""
    return max(1, int(round(pt * dpi / 72.0)))


def fit_static(src: str, dst: str, width: "str | float" = "double") -> tuple:
    """Resample a static raster (no generator in this repo) so it is exactly one
    column wide at ``savefig.dpi`` and tag the PNG with that dpi, so it drops
    into the manuscript at 100% like the matplotlib figures. Type baked into
    the raster scales with the image; only the physical size is corrected.
    Returns ``(width_px, height_px, dpi)``."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    dpi = float(_RC["savefig.dpi"])   # the style's own dpi, independent of apply()
    target_w = int(round(_resolve_width(width) * dpi))
    im = Image.open(src)
    target_h = int(round(im.height * target_w / im.width))
    im = im.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
    im.save(dst, dpi=(dpi, dpi))
    return target_w, target_h, dpi


def font_pt(key: str = "font.size") -> float:
    """Current rcParams font size (pt) for ``key`` (e.g. ``"axes.titlesize"``,
    ``"legend.fontsize"``), resolved to a number."""
    from matplotlib.font_manager import FontProperties
    return float(FontProperties(size=mpl.rcParams[key]).get_size_in_points())


def _resolve_width(width: "str | float") -> float:
    """Resolve a width spec to inches. Nature widths: ``"single"`` (89 mm) or
    ``"double"`` (183 mm). A float (inches) is accepted as an escape hatch."""
    if width == "single":
        return SINGLE
    if width == "double":
        return DOUBLE
    return float(width)


def figure(
    ncols: int = 1,
    nrows: int = 1,
    width: "str | float" = "single",
    aspect: float = 0.75,
    **kwargs,
):
    """Return ``(fig, axes)`` sized to Nature Research figure widths.

    ``width`` is the *total* figure width — ``"single"`` (89 mm), ``"double"``
    (183 mm), or a float in inches. Columns split this width evenly so the
    figure never exceeds the allowed width. Height is clamped to ``MAX_H``
    (170 mm).
    """
    fig_w   = _resolve_width(width)
    panel_w = fig_w / ncols
    panel_h = panel_w * aspect
    fig_h   = min(panel_h * nrows, MAX_H)

    return plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), **kwargs)


# Fixed class -> color registrations (see set_class_colors). Empty by default:
# classes are then colored by index from the palette.
_CLASS_COLOR_REGISTRY: dict = {}


def set_class_colors(mapping: dict) -> None:
    """Register fixed colors for named classes, replacing any earlier mapping.

    ``mapping`` maps a class name to a hex color. A class name matches a key
    exactly first; otherwise keys are matched case-insensitively as substrings
    of the name (earliest match wins, ties broken by the longest key), so a key
    ``"control"`` also colors ``"matched control"``. Registered colors are honored by
    :func:`class_colors` and :func:`class_colors_muted`; unmatched classes fall
    back to the palette by index, skipping colors already taken. Pass ``{}`` to
    clear.
    """
    _CLASS_COLOR_REGISTRY.clear()
    _CLASS_COLOR_REGISTRY.update({str(k): str(v) for k, v in mapping.items()})


def _registered_color(name: str) -> "str | None":
    """Return the registered color for *name* (exact, then substring match)."""
    if not _CLASS_COLOR_REGISTRY:
        return None
    if str(name) in _CLASS_COLOR_REGISTRY:
        return _CLASS_COLOR_REGISTRY[str(name)]
    lname = str(name).lower()
    best_key, best_pos, best_len = None, len(lname), 0
    for key in _CLASS_COLOR_REGISTRY:
        pos = lname.find(key.lower())
        if pos == -1:
            continue
        if pos < best_pos or (pos == best_pos and len(key) > best_len):
            best_key, best_pos, best_len = key, pos, len(key)
    return _CLASS_COLOR_REGISTRY[best_key] if best_key is not None else None


def _assign_colors(class_names: list, palette: list) -> list:
    """Assign registered colors, then fill the rest from *palette* by index, avoiding dupes."""
    fixed = {}
    used = set()
    for i, name in enumerate(class_names):
        col = _registered_color(name)
        if col is not None:
            fixed[i] = col
            used.add(col)

    out = []
    fallback_idx = 0
    for i in range(len(class_names)):
        if i in fixed:
            out.append(fixed[i])
        else:
            while fallback_idx < len(palette) and palette[fallback_idx % len(palette)] in used:
                fallback_idx += 1
            col = palette[fallback_idx % len(palette)]
            out.append(col)
            used.add(col)
            fallback_idx += 1
    return out


def class_colors(class_names: list) -> list:
    """Class colors: registered colors (set_class_colors), else the WONG palette by index."""
    return _assign_colors(class_names, WONG)


def class_colors_muted(class_names: list) -> list:
    """Registered colors (set_class_colors), else the muted colorblind-safe
    palette by index (never falls through to grayscale FG)."""
    return _assign_colors(class_names, MUTED_CLASS)


def class_markers(class_names: list) -> list:
    """Redundant-encoding markers (pairs with colors for accessibility)."""
    return [MARKERS[i % len(MARKERS)] for i in range(len(class_names))]


# ---------------------------------------------------------------------------
# Tufte primitive helpers
# ---------------------------------------------------------------------------

def range_frame(ax, x=None, y=None):
    """Trim spines so they span only the data range (Tufte range-frame).

    If x / y arrays are given, spines span [min, max] of those arrays.
    Otherwise, uses current axis data limits.
    """
    if x is not None and len(x) > 0:
        x0, x1 = float(np.min(x)), float(np.max(x))
    else:
        x0, x1 = ax.get_xlim()
    if y is not None and len(y) > 0:
        y0, y1 = float(np.min(y)), float(np.max(y))
    else:
        y0, y1 = ax.get_ylim()

    if "bottom" in ax.spines:
        ax.spines["bottom"].set_bounds(x0, x1)
    if "left" in ax.spines:
        ax.spines["left"].set_bounds(y0, y1)


def trim_spines(ax, offset: float = 3):
    """Offset bottom/left spines slightly away from data (sparse frame)."""
    for side in ("bottom", "left"):
        if side in ax.spines:
            ax.spines[side].set_position(("outward", offset))


def rug(ax, values, *, side: str = "bottom", length_frac: float = 0.015,
        color: str = MUTE, lw: float = 0.4, alpha: float = 0.6):
    """Dot-dash / marginal rug ticks showing where observations lie.

    side : "bottom" | "top" | "left" | "right"
    length_frac : fraction of the opposing-axis span used for tick length
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return

    if side in ("bottom", "top"):
        y0, y1 = ax.get_ylim()
        span = y1 - y0
        L = span * length_frac
        if side == "bottom":
            y_lo, y_hi = y0, y0 + L
        else:
            y_lo, y_hi = y1 - L, y1
        segs = [[(v, y_lo), (v, y_hi)] for v in values]
    else:
        x0, x1 = ax.get_xlim()
        span = x1 - x0
        L = span * length_frac
        if side == "left":
            x_lo, x_hi = x0, x0 + L
        else:
            x_lo, x_hi = x1 - L, x1
        segs = [[(x_lo, v), (x_hi, v)] for v in values]

    from matplotlib.collections import LineCollection
    lc = LineCollection(segs, colors=color, linewidths=lw, alpha=alpha, zorder=1)
    ax.add_collection(lc)


def tufte_violin(ax, data, *, positions=None, colors=None,
                 alpha: float = 0.35, edge_lw: float = 0.4):
    """Violin plot with Tufte styling: muted fills, thin edges, subtle median dot.

    colors : list | None
        Per-violin fill color. If None, uses FG (grayscale).
    """
    n = len(data)
    if positions is None:
        positions = list(range(1, n + 1))
    if colors is None:
        colors = [FG] * n
    elif len(colors) < n:
        colors = list(colors) + [FG] * (n - len(colors))

    # Filter empty groups (violinplot crashes on empty arrays)
    keep = [(pos, np.asarray(v, dtype=float), c)
            for pos, v, c in zip(positions, data, colors)
            if np.asarray(v).size > 0]
    if not keep:
        return

    plot_positions = [p for p, _, _ in keep]
    plot_data = [v for _, v, _ in keep]
    plot_colors = [c for _, _, c in keep]

    parts = ax.violinplot(plot_data, positions=plot_positions,
                          showmedians=False, showextrema=False)
    for body, col in zip(parts["bodies"], plot_colors):
        body.set_facecolor(col)
        body.set_edgecolor(col)
        body.set_alpha(alpha)
        body.set_linewidth(edge_lw)

    # Add quartile lines (thin) and median dot
    for pos, vals, col in keep:
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([pos, pos], [q1, q3], color=col, lw=1.2,
                solid_capstyle="butt", zorder=3)
        ax.plot([pos], [med], marker="o", markersize=3.0,
                markerfacecolor="white", markeredgecolor=col,
                markeredgewidth=0.5, zorder=4)


def _panel_letter(i: int) -> str:
    """Index → subpanel key: 0→'a', 25→'z', 26→'aa', 27→'ab', …"""
    letters = string.ascii_lowercase
    label = ""
    i += 1
    while i > 0:
        i, rem = divmod(i - 1, 26)
        label = letters[rem] + label
    return label


def panel_labels(axes, *, fmt: str = "{}", fontsize: float = 8,
                 fontweight: str = "bold", fontstyle: str = "normal",
                 pad: float = 3.0, color: str = FG, skip_invisible: bool = True):
    """Annotate subplots with subpanel keys (``a``, ``b``, ``c`` …).

    Nature Research style: 8-pt bold upright lowercase letters. Keys are placed
    just inside the top-left corner of every visible axis, in reading order.
    ``axes`` may be a single Axes, a flat sequence, or a 2-D array (raveled
    row-major). Returns ``axes`` unchanged for chaining.
    """
    axes_flat = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    i = 0
    for ax in axes_flat:
        if ax is None:
            continue
        if skip_invisible and not ax.get_visible():
            continue
        ax.annotate(
            fmt.format(_panel_letter(i)),
            xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(pad, -pad), textcoords="offset points",
            ha="left", va="top",
            fontsize=fontsize, fontweight=fontweight, fontstyle=fontstyle,
            color=color, annotation_clip=False,
        )
        i += 1
    return axes


def small_multiples_grid(n_panels: int, *, width="double",
                         panel_aspect: float = 0.8,
                         max_cols: int = 3,
                         sharex: bool = True, sharey: bool = True,
                         label_panels: bool = True):
    """Return (fig, axes_flat) grid for small multiples; hides unused axes.

    ``width`` is the *total* figure width (``"single"``, ``"double"``, or
    a float in inches).  Each panel gets ``width / ncols``.

    When ``label_panels`` is True and there is more than one panel, each panel
    is tagged with a subpanel key (a), b), c) …) via :func:`panel_labels`.
    """
    import math
    ncols = min(n_panels, max_cols)
    nrows = math.ceil(n_panels / ncols)
    # figure() splits the total width across columns for us.
    fig, axes = figure(ncols=ncols, nrows=nrows, width=width,
                       aspect=panel_aspect, sharex=sharex, sharey=sharey)
    axes_arr = np.atleast_1d(np.asarray(axes)).ravel()
    for i in range(n_panels, len(axes_arr)):
        axes_arr[i].set_visible(False)
    visible = axes_arr[:n_panels]
    if label_panels and n_panels > 1:
        panel_labels(visible)
    return fig, visible


def strip_imshow_frame(ax):
    """Remove outer spines for imshow / heatmap axes (frameless matrix)."""
    for side in ("top", "right", "bottom", "left"):
        if side in ax.spines:
            ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def thin_colorbar(cbar):
    """Apply thin edges to a matplotlib colorbar (Tufte-style)."""
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_edgecolor(FG)
    cbar.ax.tick_params(width=0.4, length=2.0, color=FG)

"""Interpretation figures: attention grids, UMAP, attention entropy."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from remo_id.reports import style


# ---------------------------------------------------------------------------
# Top-attention image grid
# ---------------------------------------------------------------------------

def plot_top_attention_grid(
    frames_by_class: list,
    class_names: list,
    *,
    n_top: int = 8,
    importance_label: str = "Attention",
) -> plt.Figure:
    """Grid of top-importance frames per class with normalized scores and experiment labels.

    Args:
        frames_by_class: list[list[dict]] — one list per class, each dict has keys:
            image (np.ndarray), raw_attn (float), bag_attn_max (float),
            bag_attn_min (float), experiment_id (str).
        class_names: list of class name strings.
        n_top: number of top frames to show per class.
        importance_label: label for the importance metric (e.g. "Attention", "Importance").
    """
    style.apply()
    n_classes = len(class_names)
    ncols = n_top
    nrows = n_classes

    panel_w = style.DOUBLE / ncols
    panel_h = panel_w * 1.3
    fig_h = min(panel_h * nrows, style.MAX_H)

    fig, axes = plt.subplots(nrows, ncols, figsize=(style.DOUBLE, fig_h))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for r, (cls_name, frames) in enumerate(zip(class_names, frames_by_class)):
        # Compute normalized attention per frame
        for f in frames:
            span = f["bag_attn_max"] - f["bag_attn_min"]
            f["norm_attn"] = (f["raw_attn"] - f["bag_attn_min"]) / (span + 1e-9)

        # Sort by normalized attention descending
        sorted_frames = sorted(frames, key=lambda f: -f["norm_attn"])[:n_top]

        for col in range(ncols):
            ax = axes[r, col]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col < len(sorted_frames):
                fr = sorted_frames[col]
                ax.imshow(fr["image"])
                ax.set_title(
                    f"exp={fr['experiment_id']}\na={fr['norm_attn']:.2f}",
                    fontsize=5, pad=2,
                )
        axes[r, 0].set_ylabel(cls_name, rotation=90, labelpad=6)

    label_lower = importance_label.lower()
    fig.suptitle(f"Top-{importance_label} Frames by Class")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UMAP embedding
# ---------------------------------------------------------------------------

def plot_umap(
    embeddings,
    labels,
    class_names: list,
    *,
    study_ids=None,
    experiment_ids=None,
) -> plt.Figure:
    """UMAP scatter of bag embeddings coloured by true class.

    When *experiment_ids* is provided (e.g. from multi-bag repeats),
    thin gray lines connect points from the same experiment.
    """
    style.apply()
    try:
        import umap as umap_lib
    except ImportError:
        fig, ax = style.figure()
        ax.text(0.5, 0.5, "umap-learn not installed.\npip install umap-learn",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    emb = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels)
    colors = style.class_colors_muted(class_names)
    markers = style.class_markers(class_names)

    reducer = umap_lib.UMAP(n_components=2, random_state=42, n_jobs=1)
    coords = reducer.fit_transform(emb)

    fig, ax = style.figure(width="single", aspect=1.0)

    if experiment_ids is not None:
        exp_arr = np.array(experiment_ids)
        for exp in np.unique(exp_arr):
            mask = exp_arr == exp
            if mask.sum() > 1:
                pts = coords[mask]
                ax.plot(pts[:, 0], pts[:, 1],
                        color=style.MUTE_LIGHT, alpha=0.3, linewidth=0.3, zorder=0)

    for c, (cls_name, col, mk) in enumerate(zip(class_names, colors, markers)):
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=col, s=6, marker=mk, alpha=0.8,
                   label=cls_name, linewidths=0)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Bag Embedding UMAP")
    ax.legend(markerscale=1.5)
    # UMAP coords are arbitrary — strip axis chrome for Tufte-style scatter
    style.strip_imshow_frame(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


# ---------------------------------------------------------------------------
# Attention entropy
# ---------------------------------------------------------------------------

def plot_attention_entropy(
    entropy_by_class: list,
    class_names: list,
    importance_label: str = "Attention",
) -> plt.Figure:
    """Violin plots of importance entropy per true class."""
    style.apply()
    data = [np.asarray(v, dtype=float) for v in entropy_by_class]
    positions = list(range(len(class_names)))
    colors = style.class_colors_muted(class_names)

    fig, ax = style.figure(width="single", aspect=0.9)
    style.tufte_violin(ax, data, positions=positions, colors=colors)

    ax.set_xticks(positions)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylabel(f"{importance_label} entropy  \u2212\u2211 a\u00b7log(a)")
    ax.set_title(f"{importance_label} Entropy by Class")
    style.trim_spines(ax)
    return fig


# ---------------------------------------------------------------------------
# Multi-panel UMAP
# ---------------------------------------------------------------------------

def plot_umap_panels(
    embeddings,
    panels: list,
) -> plt.Figure:
    """Multi-panel UMAP: fit once, color each panel by different categories.

    Args:
        embeddings: (N, D) array of embeddings.
        panels: list of dicts, each with keys:
            labels (array-like): per-point category labels (strings or ints)
            title (str): panel title
            names (list[str] | None): ordered category names for legend
    """
    style.apply()
    try:
        import umap as umap_lib
    except ImportError:
        fig, ax = style.figure()
        ax.text(0.5, 0.5, "umap-learn not installed.\npip install umap-learn",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    emb = np.asarray(embeddings, dtype=float)
    reducer = umap_lib.UMAP(n_components=2, random_state=42, n_jobs=1)
    coords = reducer.fit_transform(emb)

    ncols = len(panels)
    fig, axes_flat = style.small_multiples_grid(
        ncols, width="single", panel_aspect=1.0, max_cols=ncols,
        sharex=True, sharey=True,
    )

    for ax, panel in zip(axes_flat, panels):
        labels = np.asarray(panel["labels"])
        title = panel["title"]
        names = panel.get("names")

        if names is None:
            names = sorted(set(labels))

        colors = style.class_colors_muted(list(names))
        markers = style.class_markers(list(names))
        for cat, col, mk in zip(names, colors, markers):
            mask = labels == cat
            if not np.any(mask):
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=col, s=6, marker=mk, alpha=0.8,
                       label=cat, linewidths=0)

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(title)
        ax.legend(markerscale=1.5, loc="best")
        style.strip_imshow_frame(ax)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-cell contribution UMAPs (embedding-MIL interpretability)
# ---------------------------------------------------------------------------

# OKLab perceptual color space (Björn Ottosson). Mixing class colors here -- rather than in
# sRGB, which drags blends through muddy brown/grey -- keeps barycentric intermediates clean.
_OKLAB_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                      [0.2119034982, 0.6806995451, 0.1073969566],
                      [0.0883024619, 0.2817188376, 0.6299787005]])
_OKLAB_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                      [1.9779984951, -2.4285922050, 0.4505937099],
                      [0.0259040371, 0.7827717662, -0.8086757660]])


def _srgb_to_oklab(rgb):
    rgb = np.asarray(rgb, dtype=float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lms = lin @ _OKLAB_M1.T
    return np.cbrt(lms) @ _OKLAB_M2.T


def _oklab_to_srgb(lab):
    lms = (np.asarray(lab, dtype=float) @ np.linalg.inv(_OKLAB_M2).T) ** 3
    lin = lms @ np.linalg.inv(_OKLAB_M1).T
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0)


def _equidistant_anchors(n, *, L=0.70, C=0.15, hue0_deg=160.0):
    """``n`` anchor colors equally spaced on the OKLab hue circle at fixed lightness L and
    chroma C -> perceptually equidistant (every pair the same OKLab distance), maximally
    discriminable. Default hue0 puts a 3-class run at green / blue / orange (class order).
    Out-of-gamut hues are clipped by :func:`_oklab_to_srgb`.
    """
    hues = np.deg2rad(hue0_deg + 360.0 / n * np.arange(n))
    lab = np.stack([np.full(n, L), C * np.cos(hues), C * np.sin(hues)], axis=1)
    return _oklab_to_srgb(lab)


def _simplex_confidence(direction, strength):
    """Per-cell confidence in [0, 1] = magnitude x purity.

    *purity* is how dominantly a single class wins (1 = one-hot, 0 = uniform). A cell that
    contributes strongly to two classes has high magnitude but low purity -> low confidence,
    so it reads as neutral rather than an (averaged, desaturated) grey hue.
    """
    direction = np.asarray(direction, dtype=float)
    strength = np.asarray(strength, dtype=float)
    n = direction.shape[1]
    purity = np.clip((direction.max(axis=1) - 1.0 / n) / (1.0 - 1.0 / n), 0.0, 1.0)
    return np.clip(strength, 0.0, 1.0) * purity


def _simplex_rgba(direction, strength, anchor_rgb, *, chroma_boost=1.25, sharpen=1.8,
                  fixed_alpha=0.9, neutral_gray=0.82):
    """Map (direction simplex weights, strength) -> RGBA, mixing in OKLab.

    Every point gets the same ``fixed_alpha``; *confidence* (= magnitude x purity) is encoded in
    the colour itself, which lerps in OKLab from a neutral grey (confidence 0) to the vivid
    class hue (confidence 1). So neutral / split cells render grey -- not a muddy averaged hue
    or an opacity hack -- and confident single-class drivers pop. ``sharpen`` (>1) tilts the hue
    blend toward the dominant class; ``chroma_boost`` saturates it. Shared by scatter + legend.
    """
    direction = np.asarray(direction, dtype=float)
    conf = _simplex_confidence(direction, strength)
    # sharpened blend weights -> dominant class dominates the hue
    w = direction ** sharpen
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    anchor_lab = _srgb_to_oklab(anchor_rgb)            # [3,3]
    lab = w @ anchor_lab                               # [N,3] perceptual blend
    lab[:, 1:] *= chroma_boost                         # boost a,b (chroma), keep L
    grey_lab = _srgb_to_oklab(np.array([[neutral_gray, neutral_gray, neutral_gray]]))[0]
    lab = conf[:, None] * lab + (1.0 - conf)[:, None] * grey_lab   # vivid hue <- -> grey
    rgb = _oklab_to_srgb(lab)
    alpha = np.full(len(rgb), fixed_alpha)
    return np.concatenate([rgb, alpha[:, None]], axis=1)


def _simplex_direction_and_strength(contributions):
    """Map signed per-class contributions to (direction weights, strength).

    ``direction`` is a simplex weight per cell over the classes the cell drives
    *above its own average* (relu of the mean-centred contribution, normalised).
    ``strength`` is the L2 norm of the centred vector, scaled by its 99th
    percentile and clipped to [0, 1] — 0 for neutral cells, 1 for strong drivers.
    """
    contrib = np.asarray(contributions, dtype=float)
    centered = contrib - contrib.mean(axis=1, keepdims=True)
    pos = np.clip(centered, 0.0, None)
    denom = pos.sum(axis=1, keepdims=True)
    n_cls = contrib.shape[1]
    direction = np.where(denom > 0, pos / np.where(denom > 0, denom, 1.0), 1.0 / n_cls)
    mag = np.linalg.norm(centered, axis=1)
    scale = np.percentile(mag, 99) if mag.size else 0.0
    strength = np.clip(mag / scale, 0.0, 1.0) if scale > 0 else np.zeros_like(mag)
    return direction, strength


def _ternary_legend(fig, anchor_rgb, class_names):
    """Draw a barycentric color-wheel legend (equilateral triangle) as an inset.

    Samples the triangle interior with the same direction→color / distance→alpha
    mapping used for the cells, so the legend is faithful to the encoding.
    """
    v = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])  # 3 corners
    steps = 48
    ws, xy = [], []
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            w = np.array([i, j, k], float) / steps
            ws.append(w)
            xy.append(w @ v)
    ws = np.array(ws); xy = np.array(xy)
    # strength = distance from centroid, normalised so a vertex == 1
    centroid = np.full(3, 1 / 3)
    dist = np.linalg.norm(ws - centroid, axis=1)
    strength = np.clip(dist / np.linalg.norm(np.array([1.0, 0, 0]) - centroid), 0, 1)
    rgba = _simplex_rgba(ws, strength, anchor_rgb)

    # Bottom-right inset, clear of the title; opaque-ish backdrop keeps it legible over data.
    lax = fig.add_axes([0.79, 0.04, 0.18, 0.18])
    lax.set_xlim(-0.22, 1.22)
    lax.set_ylim(-0.20, 1.06)
    lax.add_patch(plt.Rectangle((-0.22, -0.20), 1.44, 1.26, facecolor="white", alpha=0.85,
                                edgecolor=style.MUTE_LIGHT, lw=0.4, zorder=0))
    lax.scatter(xy[:, 0], xy[:, 1], c=rgba, s=6, linewidths=0, marker="s", zorder=1)
    lax.plot(*np.vstack([v, v[0]]).T, color=style.MUTE, lw=0.5, zorder=2)
    offs = [(-0.05, -0.06), (0.05, -0.06), (0.0, 0.05)]
    has = ["right", "left", "center"]
    vas = ["top", "top", "bottom"]
    for corner, name, (dx, dy), ha, va in zip(v, class_names, offs, has, vas):
        lax.annotate(name, (corner[0] + dx, corner[1] + dy), ha=ha, va=va,
                     fontsize=6, fontweight="bold", zorder=3)
    lax.annotate("neutral", (0.5, np.sqrt(3) / 6), ha="center", va="center",
                 fontsize=5, color=style.MUTE, zorder=3)
    lax.set_aspect("equal")
    lax.axis("off")


def plot_simplex_umap(coords, contributions, class_names, *, anchors=None, hue0_deg=120.0,
                      title="Per-cell class contribution (simplex)") -> plt.Figure:
    """UMAP scatter where each cell's whole contribution vector sets its color.

    Hue = barycentric mix of the per-class anchor colors by the direction the cell
    drives; alpha/saturation = contribution strength (neutral cells fade into a
    faint grey terrain layer). A ternary-triangle legend keys the colors. Designed
    for the 3-class model (raises for other class counts — callers fall back to
    :func:`plot_scalar_umap_panels`).

    Args:
        coords: (M, 2) precomputed 2-D embedding (e.g. ``reducer.transform(emb)``).
        contributions: (M, C) signed additive per-class contributions.
        class_names: ordered list of C class names.
        anchors: optional explicit list of C colors; if None, equidistant OKLab anchors are
            generated and ``hue0_deg`` rotates all of them together around the hue circle.
        hue0_deg: phase (degrees) that spins the generated equidistant anchors as a set.
    """
    contrib = np.asarray(contributions, dtype=float)
    coords = np.asarray(coords, dtype=float)
    if contrib.shape[1] != 3 or len(class_names) != 3:
        raise ValueError("plot_simplex_umap requires exactly 3 classes; "
                         "use plot_scalar_umap_panels for other counts.")
    style.apply()
    if anchors is None:
        # equidistant OKLab anchors -> maximally separated, symmetric ternary blend
        anchor_rgb = _equidistant_anchors(len(class_names), hue0_deg=hue0_deg)
    else:
        anchor_rgb = np.array([to_rgb(c) for c in anchors])

    direction, strength = _simplex_direction_and_strength(contrib)
    rgba = _simplex_rgba(direction, strength, anchor_rgb)
    conf = _simplex_confidence(direction, strength)

    fig, ax = style.figure(width="single", aspect=1.0)
    # single layer, fixed opacity: colour lerps grey (neutral) -> vivid hue (confident).
    # draw most-confident cells last so they sit on top of the grey neutral cloud.
    order = np.argsort(conf)
    ax.scatter(coords[order, 0], coords[order, 1], c=rgba[order], s=5,
               linewidths=0)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    style.strip_imshow_frame(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    _ternary_legend(fig, anchor_rgb, list(class_names))
    return fig


def plot_scalar_umap_panels(coords, panels: list) -> plt.Figure:
    """Multi-panel UMAP with a continuous (diverging) colorbar per panel.

    Args:
        coords: (M, 2) precomputed 2-D embedding (shared across panels).
        panels: list of dicts, each with keys:
            scalar (array-like [M]): per-point value to color by.
            title (str): panel title.
            cmap (str, optional): colormap (default ``"RdBu_r"``).
            label (str, optional): colorbar label.
            symmetric (bool, optional): center the scale at 0 with
                ``vmin/vmax = ±99th-pct |scalar|`` (default True).
    """
    style.apply()
    coords = np.asarray(coords, dtype=float)
    fig, axes_flat = style.small_multiples_grid(
        len(panels), width="double", panel_aspect=1.0, max_cols=len(panels),
        sharex=True, sharey=True,
    )
    for ax, panel in zip(axes_flat, panels):
        scalar = np.asarray(panel["scalar"], dtype=float)
        cmap = panel.get("cmap", "RdBu_r")
        symmetric = panel.get("symmetric", True)
        if symmetric:
            v = np.percentile(np.abs(scalar), 99) if scalar.size else 1.0
            v = v if v > 0 else 1.0
            vmin, vmax = -v, v
        else:
            vmin, vmax = np.min(scalar), np.max(scalar)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=scalar, cmap=cmap,
                        s=4, vmin=vmin, vmax=vmax, linewidths=0)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        if panel.get("label"):
            cbar.set_label(panel["label"])
        style.thin_colorbar(cbar)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(panel["title"])
        style.strip_imshow_frame(ax)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig

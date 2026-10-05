#!/usr/bin/env python
"""Tests for remo_id/reports/style.py"""

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from remo_id.reports import style


def test_wong_palette_count():
    assert len(style.WONG) == 8
    print("  \u2713 WONG has 8 elements")


def test_wong_palette_hex():
    for c in style.WONG:
        assert c.startswith("#"), f"{c} does not start with #"
    print("  \u2713 WONG elements start with #")


def test_dimension_constants():
    assert style.SINGLE > 0
    assert style.DOUBLE > style.SINGLE
    print("  \u2713 SINGLE > 0, DOUBLE > SINGLE")


def test_apply_rcparams():
    style.apply()
    assert mpl.rcParams["font.size"] == 8
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["savefig.dpi"] == 450
    print("  \u2713 apply() sets rcParams correctly")


def test_figure_single():
    style.apply()
    fig, ax = style.figure()
    w, h = fig.get_size_inches()
    assert abs(w - style.SINGLE) < 0.01
    plt.close(fig)
    print("  \u2713 figure() single width")


def test_figure_double():
    style.apply()
    fig, ax = style.figure(width="double")
    w, h = fig.get_size_inches()
    assert abs(w - style.DOUBLE) < 0.01
    plt.close(fig)
    print("  \u2713 figure(width='double')")


def test_figure_explicit_width():
    style.apply()
    fig, ax = style.figure(width=5.0)
    w, h = fig.get_size_inches()
    assert abs(w - 5.0) < 0.01
    plt.close(fig)
    print("  \u2713 figure(width=5.0)")


def test_figure_grid():
    style.apply()
    fig, axes = style.figure(ncols=2, nrows=3)
    assert axes.shape == (3, 2)
    plt.close(fig)
    print("  \u2713 figure(ncols=2, nrows=3) -> shape (3,2)")


def test_figure_max_height():
    style.apply()
    fig, axes = style.figure(nrows=20)
    w, h = fig.get_size_inches()
    assert h <= style.MAX_H + 0.01
    plt.close(fig)
    print("  \u2713 figure(nrows=20) height capped")


def test_class_colors_by_index_default():
    style.set_class_colors({})
    assert style.class_colors_muted(["a", "b", "c"]) == style.MUTED_CLASS[:3]
    assert style.class_colors(["foo", "bar"]) == style.WONG[:2]
    print("  \u2713 unregistered classes colored by index")


def test_set_class_colors_exact_and_substring():
    style.set_class_colors({"Positive": "#000001", "control": "#000002"})
    try:
        colors = style.class_colors_muted(["Positive", "Matched Control", "other"])
        assert colors[0] == "#000001"
        assert colors[1] == "#000002"
        assert colors[2] == style.MUTED_CLASS[0]
        assert style.class_colors(["positive_sample"])[0] == "#000001"
    finally:
        style.set_class_colors({})
    print("  \u2713 registered colors honored (exact + substring)")


def test_set_class_colors_skips_taken_palette_colors():
    style.set_class_colors({"x": style.MUTED_CLASS[0]})
    try:
        colors = style.class_colors_muted(["x", "y"])
        assert colors == [style.MUTED_CLASS[0], style.MUTED_CLASS[1]]
    finally:
        style.set_class_colors({})
    print("  \u2713 fallback skips registered colors")


def test_class_colors_many():
    names = [f"class_{i}" for i in range(9)]
    colors = style.class_colors(names)
    assert len(colors) == 9
    print("  \u2713 class_colors 9+ names no crash")


if __name__ == "__main__":
    print("test_style.py")
    test_wong_palette_count()
    test_wong_palette_hex()
    test_dimension_constants()
    test_apply_rcparams()
    test_figure_single()
    test_figure_double()
    test_figure_explicit_width()
    test_figure_grid()
    test_figure_max_height()
    test_class_colors_by_index_default()
    test_set_class_colors_exact_and_substring()
    test_set_class_colors_skips_taken_palette_colors()
    test_class_colors_many()
    print("All tests passed.")

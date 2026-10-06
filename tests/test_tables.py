#!/usr/bin/env python
"""Tests for remo_id/analysis/tables.py"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from _helpers_analysis import make_multiclass_data
from remo_id.reports.tables import (
    format_performance_table, format_baseline_table,
    format_dataset_table, to_latex,
)


def test_performance_table_shape():
    y_true, _, y_probs = make_multiclass_data(n=200, C=3)
    df = format_performance_table(y_true, y_probs, ["A", "B", "C"], bootstrap_n=50)
    assert len(df) == 4  # 3 classes + Macro
    for col in ["Class", "N", "AUC", "AP", "Sensitivity", "Specificity", "F1", "Brier"]:
        assert col in df.columns, f"missing column: {col}"
    print("  \u2713 performance_table shape and columns")


def test_performance_table_macro_row():
    y_true, _, y_probs = make_multiclass_data(n=200, C=3)
    df = format_performance_table(y_true, y_probs, ["A", "B", "C"], bootstrap_n=50)
    assert df.iloc[-1]["Class"] == "Macro"
    print("  \u2713 performance_table last row is Macro")


def test_performance_table_ci_bracket():
    y_true, _, y_probs = make_multiclass_data(n=200, C=3)
    df = format_performance_table(y_true, y_probs, ["A", "B", "C"], bootstrap_n=50)
    # Per-class AUC should contain CI bracket
    for i in range(3):
        assert "[" in str(df.iloc[i]["AUC"]), f"row {i} AUC missing CI bracket"
    print("  \u2713 performance_table AUC has CI brackets")


def test_baseline_table():
    our_results = {"accuracy": 0.95, "balanced_accuracy": 0.94, "f1_macro": 0.93, "auc_macro": 0.98}
    baseline_df = pd.DataFrame([
        {"model": "LR", "accuracy": 0.80, "balanced_accuracy": 0.79, "f1_macro": 0.78, "auc_macro": 0.85},
        {"model": "RF", "accuracy": 0.82, "balanced_accuracy": 0.81, "f1_macro": 0.80, "auc_macro": 0.87},
        {"model": "Majority", "accuracy": 0.50, "balanced_accuracy": 0.50, "f1_macro": 0.33, "auc_macro": np.nan},
    ])
    df = format_baseline_table(our_results, baseline_df)
    assert len(df) == 4  # ours + 3 baselines
    assert df.iloc[0]["model"] == "Our model"
    print("  \u2713 baseline_table combined correctly")


def test_dataset_table():
    stats_df = pd.DataFrame([
        {"class": "A", "n_experiments": 20, "n_images": 1000},
        {"class": "B", "n_experiments": 30, "n_images": 1500},
    ])
    df = format_dataset_table(stats_df)
    assert df.iloc[-1]["Class"] == "Total"
    assert df.iloc[-1]["Experiments"] == 50
    assert df.iloc[-1]["Images"] == 2500
    print("  \u2713 dataset_table totals match sums")


def test_dataset_table_totals_match():
    stats_df = pd.DataFrame([
        {"class": "X", "n_experiments": 10, "n_images": 500},
        {"class": "Y", "n_experiments": 15, "n_images": 750},
        {"class": "Z", "n_experiments": 25, "n_images": 1250},
    ])
    df = format_dataset_table(stats_df)
    total_row = df[df["Class"] == "Total"].iloc[0]
    data_rows = df[df["Class"] != "Total"]
    assert total_row["Experiments"] == data_rows["Experiments"].sum()
    assert total_row["Images"] == data_rows["Images"].sum()
    print("  \u2713 dataset_table totals match column sums")


def test_to_latex_basic():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    latex = to_latex(df)
    assert "\\begin{tabular}" in latex
    assert isinstance(latex, str)
    print("  \u2713 to_latex returns string with tabular")


def test_to_latex_with_caption():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    latex = to_latex(df, caption="My Table", label="tab:test")
    assert "\\begin{table}" in latex
    assert "\\caption{My Table}" in latex
    assert "\\label{tab:test}" in latex
    print("  \u2713 to_latex with caption/label wraps in table env")


def test_to_latex_no_caption():
    df = pd.DataFrame({"A": [1]})
    latex = to_latex(df)
    assert "\\begin{table}" not in latex
    print("  \u2713 to_latex without caption omits table env")


def test_to_latex_content():
    df = pd.DataFrame({"Metric": ["AUC"], "Value": ["0.95"]})
    latex = to_latex(df)
    assert "AUC" in latex
    assert "0.95" in latex
    print("  \u2713 to_latex includes DataFrame content")


if __name__ == "__main__":
    print("test_tables.py")
    test_performance_table_shape()
    test_performance_table_macro_row()
    test_performance_table_ci_bracket()
    test_baseline_table()
    test_dataset_table()
    test_dataset_table_totals_match()
    test_to_latex_basic()
    test_to_latex_with_caption()
    test_to_latex_no_caption()
    test_to_latex_content()
    print("All tests passed.")

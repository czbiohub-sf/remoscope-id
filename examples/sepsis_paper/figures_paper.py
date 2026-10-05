"""Output builders for make_paper_results.py.

Each public function builds one figure/table into the output directory. They are
deliberately defensive: a missing column or study degrades gracefully rather than
aborting the whole run (the orchestrator wraps each in common.step).
"""

import glob
import os
import textwrap
import warnings

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from remo_id.reports import style
from remo_id.reports.figures._bootstrap import _bootstrap_ci, _bootstrap_roc_band

import common as C


# ===========================================================================
# Static copies
# ===========================================================================
def copy_static(out, dest_name, src_path, width="double"):
    """Bring a static (externally generated) raster into the results folder at
    the print width of the other figures (style.fit_static), rather than a raw
    copy at whatever size it was drawn."""
    if not src_path or not os.path.exists(src_path):
        print(f"  static source missing: {src_path}")
        return "skip"
    w, h, dpi = style.fit_static(src_path, os.path.join(out, dest_name), width)
    print(f"  fitted {src_path} -> {dest_name} ({w}x{h} px, "
          f"{w / dpi * 25.4:.0f} mm @ {dpi:.0f} dpi)")


def _title_case_class(name):
    """Class name as it appears in TITLES: 'ICU control' -> 'ICU Control'.
    Axis/legend/tick labels keep the sentence-case 'ICU control'."""
    if name.lower() in ("icu-ctrl", "icu ctrl", "icu-control", "icu control"):
        return "ICU Control"
    return name


# ===========================================================================
# Fig 1: confusion matrix + ROC (Youden-J), 2-panel
# ===========================================================================
# All fig1-family panels are sized to Nature column widths (style.SINGLE /
# style.DOUBLE) and typeset entirely by style.py's rcParams (5-7 pt), so the
# PNGs drop into the manuscript at 100% with no rescaling and no per-figure
# font tuning. Subpanel keys are style.PANEL_LABEL_PT bold, placed just outside
# the top-left corner of each axes.


def _align_cm_xlabel(fig, ax_roc, ax_cm, pad_px=8):
    """Put the confusion matrix's "Predicted" label on the same line as the ROC's
    x-label.

    The matrix is an equal-aspect image that fills its box height, so its
    45deg-rotated class ticks hang below the square and shove "Predicted" far
    lower than the ROC's x-label. Shrink the matrix box from the bottom until the
    rotated ticks clear the ROC label line, then pin the label to that line.
    Both labels are anchored top-centre at the same font size, so matching the
    anchors matches the baselines.
    """
    fig.canvas.draw()
    label_top = ax_roc.xaxis.label.get_window_extent().y1
    ticks = ax_cm.get_xticklabels()
    if not ticks:
        return
    shift = (label_top + pad_px) - min(t.get_window_extent().y0 for t in ticks)
    if shift > 0:
        box = ax_cm.get_position()
        dy = shift / fig.bbox.height
        ax_cm.set_position([box.x0, box.y0 + dy, box.width, box.height - dy])
        fig.canvas.draw()
    ax_cm.xaxis.set_label_coords(
        0.5, ax_cm.transAxes.inverted().transform((0, label_top))[1])


def _plot_confusion(ax, cm, names, paren=None, col_names=None):
    """Confusion matrix. Cells show ``count (proportion)`` by default; if
    ``paren`` (same-shape int matrix) is given, show ``count (paren)`` instead —
    e.g. patient count with the number of experiments behind it. ``col_names``
    allows a non-square matrix (extra predicted columns, e.g. "Discordant")."""
    col_names = list(col_names) if col_names is not None else list(names)
    n, nc = len(names), len(col_names)
    row = cm.sum(axis=1, keepdims=True).clip(min=1)
    norm = cm.astype(float) / row
    ax.imshow(norm, cmap="Greys", vmin=0, vmax=1)
    style.strip_imshow_frame(ax)
    ax.set_xticks(range(nc)); ax.set_yticks(range(n))
    ax.set_xticklabels(col_names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(n):
        for j in range(nc):
            tc = "white" if norm[i, j] > 0.55 else style.FG
            paren_txt = f"{int(paren[i, j])}" if paren is not None else f"{norm[i, j]:.2f}"
            ax.text(j, i, f"{cm[i, j]}\n({paren_txt})",
                    ha="center", va="center", color=tc)


def _plot_roc(ax, y_true, y_probs, names, display_order, colors, bootstrap_n):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], color=style.MUTE_FAINT, lw=0.5, ls=(0, (1, 1.5)), zorder=1)
    for k, data_idx in enumerate(display_order):
        y_bin = (y_true == data_idx).astype(int)
        y_sc = y_probs[:, data_idx]
        fpr, tpr, _ = roc_curve(y_bin, y_sc)
        fg, _, lo, hi = _bootstrap_roc_band(y_bin, y_sc, n=bootstrap_n)
        ax.fill_between(fg, lo, hi, color=colors[k], alpha=0.18, linewidth=0)
        pt, l, h = _bootstrap_ci(y_bin, y_sc, roc_auc_score, n=bootstrap_n)
        ax.plot(fpr, tpr, color=colors[k], lw=1.0,
                label=f"{names[k]}  AUC={pt:.2f} [{l:.2f}–{h:.2f}]")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    style.range_frame(ax); style.trim_spines(ax)


def _panel_key(ax, key, dx_pt=-13, dy_pt=4):
    """Subpanel key just outside the top-left corner of ``ax`` (Nature: 8 pt
    bold lowercase). Offsets are in points so they scale with the type, not the
    canvas."""
    ax.annotate(key, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(dx_pt, dy_pt), textcoords="offset points",
                ha="left", va="bottom", fontsize=style.PANEL_LABEL_PT,
                fontweight="bold", color=style.FG, annotation_clip=False)


def _two_panel_roc_cm(out, name, ci, y_true, roc_probs, cm, bootstrap_n,
                      roc_title, cm_title, col_names=None, cm_paren=None):
    """Shared double-column (183 mm) `a) ROC | b) confusion matrix` figure used
    by the whole fig1 family. Sized and typeset by style.py; saved at the fixed
    canvas width (no bbox re-crop)."""
    names = ci.display_names
    colors = style.class_colors_muted(names)
    n_title_lines = max(t.count("\n") + 1 for t in (roc_title, cm_title))
    fig_h = style.DOUBLE * (0.46 + 0.03 * n_title_lines)
    fig = plt.figure(figsize=(style.DOUBLE, fig_h))
    # top margin = title block (titlesize pt per line + pad) in figure fraction
    title_in = (n_title_lines * style.font_pt("axes.titlesize") * 1.25
                + mpl.rcParams["axes.titlepad"] + 6) / 72.0
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.34,
                           left=0.08, right=0.985, top=1 - title_in / fig_h,
                           bottom=0.23 * (style.DOUBLE * 0.48) / fig_h)
    ax_roc = fig.add_subplot(gs[0, 0]); ax_cm = fig.add_subplot(gs[0, 1])
    _plot_roc(ax_roc, y_true, roc_probs, names, ci.display_order, colors, bootstrap_n)
    ax_roc.set_title(roc_title)
    _plot_confusion(ax_cm, cm, names, paren=cm_paren, col_names=col_names)
    ax_cm.set_title(cm_title)
    _panel_key(ax_roc, "a"); _panel_key(ax_cm, "b")
    fig.savefig(os.path.join(out, f"{name}.png"))
    plt.close(fig)


def _derive_misclassifications(preds, ci):
    """Return the held-out misclassifications as a DataFrame (errors only).

    Errors come from the pooled predictions under the Youden-shifted argmax rule
    (the fig1 / table2 rule). Columns: experiment_id, study_id, true_class,
    predicted_class, confidence, error_type, sample_id.
    """
    y_true = preds["y_true"].astype(int)
    thr = C.youden_thresholds(y_true, preds["y_probs"], ci.class_names)
    y_pred = (preds["y_probs"] - thr).argmax(axis=1)
    conf = preds["y_probs"][np.arange(len(y_pred)), y_pred]
    err = y_pred != y_true
    df = pd.DataFrame({
        "experiment_id": np.asarray(preds["eids"]).astype(str)[err],
        "study_id": np.asarray(preds["study_ids"]).astype(str)[err],
        "true_class": [ci.class_names[i] for i in y_true[err]],
        "predicted_class": [ci.class_names[i] for i in y_pred[err]],
        "confidence": conf[err],
        "error_type": [f"{ci.class_names[t]}->{ci.class_names[p]}"
                       for t, p in zip(y_true[err], y_pred[err])],
        "sample_id": np.asarray(preds["sample_ids"]).astype(str)[err],
    })
    print(f"  derived errors from pooled predictions ({len(df)} misclassifications)")
    return df.reset_index(drop=True)


def _diagnosis_maps(clin, study_id):
    """experiment_id -> (clinical diagnosis, pathogen) from the clinical table.

    The notes column carries the adjudicated free-text diagnosis; the pathogen
    column just the organism (column names from the spec's `diagnosis`)."""
    dxs = clin.spec["diagnosis"]
    try:
        sep = clin.study(study_id)
        cols = {c: sep[c] if c in sep.columns else pd.Series([None] * len(sep))
                for c in (dxs["notes_column"], dxs["pathogen_column"])}
        dx_map = dict(zip(sep["experiment_id"], cols[dxs["notes_column"]]))
        path_map = dict(zip(sep["experiment_id"], cols[dxs["pathogen_column"]]))
    except Exception as exc:  # noqa: BLE001
        print(f"  clinical dx lookup failed: {exc}")
        dx_map, path_map = {}, {}
    return dx_map, path_map


def error_analysis_clinical(out, ci, preds, clin):
    """Misclassifications-only error report enriched with sample_id + clinical dx.

    Uses the shared held-out misclassification set, then adds the clinical
    diagnosis and pathogen of the sepsis study rows of the clinical table.
    """
    df = _derive_misclassifications(preds, ci)
    eids = df["experiment_id"].values
    dx_map, path_map = _diagnosis_maps(clin, ci.sepsis_study)
    df["clinical_diagnosis"] = [dx_map.get(e, "") for e in eids]
    df["pathogen"] = [path_map.get(e, "") for e in eids]

    cols = ["experiment_id", "sample_id", "study_id", "true_class",
            "predicted_class", "confidence", "error_type",
            "clinical_diagnosis", "pathogen"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(os.path.join(out, "suptable1_error_analysis.csv"), index=False)
    disp = df.copy()
    if "confidence" in disp:
        disp["confidence"] = disp["confidence"].map(lambda v: f"{float(v):.3f}")
    C.render_table(disp, os.path.join(out, "suptable1_error_analysis"),
                   title="Supplementary Table 1: Misclassifications (held-out) "
                         "with clinical diagnosis",
                   fontsize=6)


def _load_allexp_shards(pred_dir, epoch):
    """Concatenate the per-fold shards written by `loo-predict --all_experiments`.

    Each shard is one fold's epoch-N model scored against EVERY experiment, with
    an `in_train` flag. Returns a long DataFrame with one row per
    (fold, experiment).
    """
    pat = os.path.join(pred_dir, f"epoch_{epoch}_shard*.npz")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(
            f"no all-experiment shards at {pat}. Run `remo-id workflow loo-predict "
            f"--all_experiments` first (one task per fold shard, ~4 h/task).")

    frames = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "in_train" not in d.files:
            raise ValueError(
                f"{f} has no in_train column -- it was written by a plain "
                f"loo-predict run, not --all_experiments")
        probs = np.asarray(d["y_probs"], dtype=float)
        frames.append(pd.DataFrame({
            "experiment_id": np.asarray(d["experiment_ids"]).astype(str),
            "study_id": np.asarray(d["study_ids"]).astype(str),
            "fold": np.asarray(d["folds"]).astype(int),
            "in_train": np.asarray(d["in_train"]).astype(int),
            "y_true": np.asarray(d["y_true"]).astype(int),
            "y_pred_argmax": probs.argmax(axis=1),
            "p_true": probs[np.arange(len(probs)), np.asarray(d["y_true"]).astype(int)],
        }))
    df = pd.concat(frames, ignore_index=True)
    n_folds = df["fold"].nunique()
    n_exp = df["experiment_id"].nunique()
    print(f"  loaded {len(files)} shards: {len(df)} rows "
          f"({n_folds} folds x {n_exp} experiments)")
    if len(files) < n_folds:
        print(f"  WARNING: only {len(files)} shard files for {n_folds} folds")
    return df


def suptable1b_error_analysis_all_folds(out, ci, args):
    """Per-experiment error rate across EVERY fold, including folds that trained
    on it -- the "is this experiment wrong even when memorized?" table.

    Uses PLAIN argmax, not the Youden-shifted rule of suptable1: the question is
    what the model actually decides, not what a tuned operating point decides, so
    the held-out column here can disagree with suptable1 by design.

    Train-set rows are a memorization diagnostic and are NOT held-out
    performance -- an experiment the model gets wrong while training on it is
    either genuinely atypical for its label or mislabelled, which is exactly what
    this table is for. Nothing here should be pooled into a reported metric.
    """
    long = _load_allexp_shards(args.allexp_dir, args.epoch)
    long["wrong"] = (long["y_pred_argmax"] != long["y_true"]).astype(int)

    rows = []
    for eid, g in long.groupby("experiment_id"):
        tr = g[g["in_train"] == 1]
        ho = g[g["in_train"] == 0]
        y_true = int(g["y_true"].iloc[0])
        n_tr, n_tr_wrong = len(tr), int(tr["wrong"].sum())
        # Modal wrong call across the training folds -- what the model confuses
        # this experiment FOR when it has already seen it.
        wrong_preds = tr.loc[tr["wrong"] == 1, "y_pred_argmax"]
        modal = (ci.class_names[int(wrong_preds.mode().iloc[0])]
                 if len(wrong_preds) else "")
        rows.append({
            "experiment_id": eid,
            "study_id": g["study_id"].iloc[0],
            "true_class": ci.class_names[y_true],
            "heldout_pred": (ci.class_names[int(ho["y_pred_argmax"].iloc[0])]
                             if len(ho) else ""),
            "heldout_correct": (bool(ho["wrong"].iloc[0] == 0)
                                if len(ho) else None),
            "heldout_p_true": (float(ho["p_true"].iloc[0]) if len(ho) else np.nan),
            "n_train_folds": n_tr,
            "n_train_wrong": n_tr_wrong,
            "train_error_pct": 100.0 * n_tr_wrong / n_tr if n_tr else np.nan,
            "train_p_true_mean": float(tr["p_true"].mean()) if n_tr else np.nan,
            "modal_train_error": modal,
        })

    df = pd.DataFrame(rows)
    df["sample_id"] = [args.sample_map.get(e, "") for e in df["experiment_id"]]

    dx_map, path_map = _diagnosis_maps(args.clin, ci.sepsis_study)
    df["clinical_diagnosis"] = [dx_map.get(e, "") for e in df["experiment_id"]]
    df["pathogen"] = [path_map.get(e, "") for e in df["experiment_id"]]

    # Full matrix to CSV; the rendered PNG keeps only experiments that are ever
    # wrong, since a table of 129 all-correct rows communicates nothing.
    cols = ["experiment_id", "sample_id", "study_id", "true_class",
            "heldout_pred", "heldout_correct", "heldout_p_true",
            "n_train_folds", "n_train_wrong", "train_error_pct",
            "train_p_true_mean", "modal_train_error",
            "clinical_diagnosis", "pathogen"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(["train_error_pct", "heldout_correct"],
                        ascending=[False, True]).reset_index(drop=True)
    df.to_csv(os.path.join(out, "suptable1b_error_analysis_all_folds.csv"),
              index=False)

    ever_wrong = df[(df["n_train_wrong"] > 0) | (df["heldout_correct"] == False)]  # noqa: E712
    print(f"  {len(ever_wrong)}/{len(df)} experiments wrong in at least one fold; "
          f"{int((df['n_train_wrong'] > 0).sum())} wrong while trained on")

    disp = ever_wrong.copy()
    for c, fmt in (("heldout_p_true", "{:.3f}"), ("train_p_true_mean", "{:.3f}"),
                   ("train_error_pct", "{:.1f}%")):
        if c in disp:
            disp[c] = disp[c].map(lambda v, f=fmt: "" if pd.isna(v) else f.format(v))
    C.render_table(disp, os.path.join(out, "suptable1b_error_analysis_all_folds"),
                   title=f"Supplementary Table 1b: per-experiment errors across all "
                         f"{long['fold'].nunique()} folds at epoch {args.epoch} "
                         f"(plain argmax; train-set rows are a memorization "
                         f"diagnostic, not performance)",
                   fontsize=6)


def _sample_average_preds(preds):
    """Average the per-experiment class probabilities within each sample_id.

    Returns (y_true_avg, y_probs_avg, n_patients). The per-patient true class is
    the majority of that patient's experiments (they normally all agree).
    Experiments with a blank sample_id are each treated as their own patient.
    """
    y_true = preds["y_true"].astype(int)
    y_probs = preds["y_probs"]
    eids = np.asarray(preds["eids"]).astype(str)
    sids = preds.get("sample_ids")
    sids = np.asarray(sids).astype(str) if sids is not None else np.array([""] * len(eids))
    groups = np.array([s if s not in ("", "nan", "None") else f"exp:{e}"
                       for s, e in zip(sids, eids)])
    uniq = sorted(set(groups))
    n_cls = y_probs.shape[1]
    yp = np.zeros((len(uniq), n_cls))
    yt = np.zeros(len(uniq), dtype=int)
    for i, g in enumerate(uniq):
        sel = groups == g
        yp[i] = y_probs[sel].mean(axis=0)
        yt[i] = np.bincount(y_true[sel], minlength=n_cls).argmax()
    return yt, yp, len(uniq)


def _sample_argmax_preds(preds):
    """Patient-level via the MOST-CONFIDENT experiment: for a patient with several
    experiments, keep the one whose probability vector has the highest peak (max)
    probability and use that whole vector as the patient's prediction. Returns
    (y_true_maj, probs, n_patients). Contrasts with the probability-averaged
    `_sample_average_preds` (mean) -- this takes the single most-confident vector."""
    y_true = preds["y_true"].astype(int)
    y_probs = preds["y_probs"]
    conf = y_probs.max(axis=1)                          # peak confidence per experiment
    eids = np.asarray(preds["eids"]).astype(str)
    sids = preds.get("sample_ids")
    sids = np.asarray(sids).astype(str) if sids is not None else np.array([""] * len(eids))
    groups = np.array([s if s not in ("", "nan", "None") else f"exp:{e}"
                       for s, e in zip(sids, eids)])
    uniq = sorted(set(groups))
    K = y_probs.shape[1]
    yp = np.zeros((len(uniq), K))
    yt = np.zeros(len(uniq), dtype=int)
    for i, g in enumerate(uniq):
        idx = np.where(groups == g)[0]
        best = idx[int(np.argmax(conf[idx]))]           # most-confident experiment
        yp[i] = y_probs[best]
        yt[i] = np.bincount(y_true[idx], minlength=K).argmax()
    return yt, yp, len(uniq)


def _cm_variants(ci, preds):
    """The six aggregation x decision-rule combinations, in row order a-f.

    Returns [(key, aggregation, rule_name, y_true, y_probs, rule_fn, n, unit)].

    Youden thresholds are fit on the SAME probabilities each variant uses, so
    every variant is self-consistent; a threshold fit
    at experiment level and applied to patient-averaged probabilities would not
    be.
    """
    exp_yt = preds["y_true"].astype(int)
    exp_yp = preds["y_probs"]
    samp_max_yt, samp_max_yp, n_max = _sample_argmax_preds(preds)
    samp_avg_yt, samp_avg_yp, n_avg = _sample_average_preds(preds)

    def _youden(yt, yp):
        return (yp - C.youden_thresholds(yt, yp, ci.class_names)).argmax(axis=1)

    def _argmax(yt, yp):
        return yp.argmax(axis=1)

    EXP = "Experiment-Level"
    MAXP = "Patient-Level, Most-Confident Replicate"
    MEAN = "Patient-Level, Mean Probability"
    YJ, AM = "Youden-J", "Argmax"

    # (key, aggregation, decision rule, y_true, y_probs, rule_fn, n, unit)
    return [
        ("experiment_youdenJ", EXP, YJ, exp_yt, exp_yp, _youden,
         len(exp_yt), "experiments"),
        ("experiment_argmax", EXP, AM, exp_yt, exp_yp, _argmax,
         len(exp_yt), "experiments"),
        ("sample_maxprob", MAXP, AM, samp_max_yt, samp_max_yp, _argmax,
         n_max, "patients"),
        ("sample_mean", MEAN, AM, samp_avg_yt, samp_avg_yp, _argmax,
         n_avg, "patients"),
        ("sample_youden_maxprob", MAXP, YJ, samp_max_yt, samp_max_yp, _youden,
         n_max, "patients"),
        ("sample_youden_mean", MEAN, YJ, samp_avg_yt, samp_avg_yp, _youden,
         n_avg, "patients"),
    ]


def suptable2_cm_variants(out, ci, preds, args):
    """Every aggregation x decision-rule confusion matrix as one table.

    Six 3x3 matrices are mostly empty -- Healthy barely confuses with anything,
    so the whole story sits in the Sepsis<->ICU-Control corner. One row per
    variant with per-class recall and every non-zero off-diagonal spelled out
    carries the same information in a fraction of the space.
    """
    panels = _cm_variants(ci, preds)
    labels = list("abcdef")
    K = len(ci.class_names)
    names = ci.class_names

    rows = []
    for lab, (key, agg, rule_name, yt, yp, rule, n, unit) in zip(labels, panels):
        y_pred = rule(yt, yp)
        cm = confusion_matrix(yt, y_pred, labels=list(range(K)))
        acc = f"{(y_pred == yt).mean():.3f}"
        mf1 = f"{f1_score(yt, y_pred, average='macro'):.3f}"
        # One row per class: one-vs-rest sens/spec/PPV/NPV off the matrix.
        # Point estimates only -- with 6 variants x 3 classes the bootstrap CIs
        # of table2 would overlap almost completely and swamp the comparison.
        for r_i, cls in enumerate(ci.display_order):
            tp = int(cm[cls, cls])
            fn = int(cm[cls].sum()) - tp
            fp = int(cm[:, cls].sum()) - tp
            tn = int(cm.sum()) - tp - fn - fp
            first = r_i == 0
            rows.append({
                "Panel": lab if first else "",
                "variant_key": key if first else "",   # CSV only, dropped below
                "Aggregation": agg if first else "",
                "Decision Rule": rule_name if first else "",
                "Class": names[cls],
                "Support": int(cm[cls].sum()),
                "Sensitivity": f"{tp / max(1, tp + fn):.3f}",
                "Specificity": f"{tn / max(1, tn + fp):.3f}",
                "PPV": f"{tp / max(1, tp + fp):.3f}",
                "NPV": f"{tn / max(1, tn + fn):.3f}",
                "Accuracy": acc if first else "",
                "Macro-F1": mf1 if first else "",
            })
        print(f"  {lab} {key:24s} n={n:3d} {unit[:3]} acc={acc} macroF1={mf1} "
              f"errors={int((y_pred != yt).sum())}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "suptable2_cm_variants.csv"), index=False)
    # The snake_case key is a join handle for other scripts, not something to
    # print next to the readable Aggregation/Decision Rule columns.
    C.render_table(df.drop(columns=["variant_key"]),
                   os.path.join(out, "suptable2_cm_variants"),
                   title="Supplementary Table 2: Aggregation and Decision-Rule "
                         "Variants (same model, same predictions)",
                   fontsize=6)


def fig1e_sepsis_vs_icu_roc(out, ci, preds, args):
    """Sepsis-vs-ICU-control ROC (two-class renormalized P(sepsis)) for the
    sample-averaged predictions only, with a bootstrap 95% CI band."""
    from remo_id.analysis.pairwise import filter_pairwise
    style.apply()
    if ci.neg_idx is None:
        print("  fig1e skipped: no ICU-control class")
        return "skip"
    sep_name = ci.class_names[ci.sepsis_idx]
    neg_name = ci.class_names[ci.neg_idx]

    yt_s, yp_s, n_pat = _sample_average_preds(preds)

    # Sized/typeset entirely by style.py (89 mm single column, 5-7 pt type,
    # 450 dpi) -- no figsize / fontsize / dpi / bbox overrides here, so the PNG
    # is placed in the manuscript at 100% with no rescaling.
    fig, ax = style.figure(width="single", aspect=0.9)
    ax.plot([0, 1], [0, 1], color=style.MUTE_FAINT, lw=0.5, ls=(0, (1, 1.5)), zorder=1)
    yb, ypb, mask, pos_i, neg_i = filter_pairwise(yt_s, yp_s, ci.class_names,
                                                  sep_name, neg_name)
    yb = np.asarray(yb).astype(int)
    ysc = ypb[:, 1]
    fpr, tpr, _ = roc_curve(yb, ysc)
    pt, lo, hi = _bootstrap_ci(yb, ysc, roc_auc_score, n=args.bootstrap_n)

    # Bootstrap 95% envelope of the ROC curve itself (resample patients,
    # interpolate each replicate's curve onto a common FPR grid).
    grid = np.linspace(0, 1, 101)
    rng = np.random.RandomState(getattr(args, "seed", 403))
    curves = []
    for _ in range(args.bootstrap_n):
        idx = rng.randint(0, len(yb), len(yb))
        if len(np.unique(yb[idx])) < 2:
            continue
        f_b, t_b, _ = roc_curve(yb[idx], ysc[idx])
        curves.append(np.interp(grid, f_b, t_b))
    band_lo, band_hi = np.percentile(np.vstack(curves), [2.5, 97.5], axis=0)
    ax.fill_between(grid, band_lo, band_hi, color=style.ACCENT, alpha=0.18,
                    linewidth=0, zorder=2, label="95% CI (bootstrap)")
    ax.plot(fpr, tpr, color=style.ACCENT, zorder=3,
            label=f"Sample-averaged  AUC={pt:.2f} [{lo:.2f}–{hi:.2f}]  (n={int(len(yb))})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{sep_name} vs {_title_case_class(neg_name)} Discrimination")
    ax.legend(loc="lower right")
    style.range_frame(ax); style.trim_spines(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1e_sepsis_vs_icu_roc.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Patient-level (sample_id) plain-argmax rules.
#
# Three ways to turn a patient's technical replicates into one call, all using
# plain argmax (no Youden shift): fig1_cm_roc_argmax_patient_mean draws the
# "mean" rule, table_patient_argmax_summary compares all three:
#   unanimous  every replicate's argmax must agree; any disagreement makes the
#              patient wrong (lands in a "Discordant" column)
#   mean       average the replicate probability vectors, then argmax
#   maxprob    argmax over ALL replicate probabilities at once, i.e. the class
#              holding the single highest probability anywhere in the patient
# ---------------------------------------------------------------------------
def _patient_groups(preds):
    """Per-experiment patient key + the ordered unique keys. Experiments with a
    blank sample_id become their own single-replicate patient."""
    eids = np.asarray(preds["eids"]).astype(str)
    sids = preds.get("sample_ids")
    sids = np.asarray(sids).astype(str) if sids is not None \
        else np.array([""] * len(eids))
    groups = np.array([s if s not in ("", "nan", "None") else f"exp:{e}"
                       for s, e in zip(sids, eids)])
    return groups, sorted(set(groups))


def _patient_pool(preds, how):
    """Pool the replicate probability vectors within each patient.

    ``how`` is "mean" (average), "max" (per-class max across replicates -- its
    argmax is the class of the single highest probability anywhere in the
    patient) or "min" (per-class min -- thresholding it demands that EVERY
    replicate clear the threshold, the score analogue of the unanimity rule).

    Returns (y_true, pooled_probs, n_replicates, patient_keys); the patient's
    true class is the majority over its replicates (they normally all agree).
    """
    y_true = np.asarray(preds["y_true"]).astype(int)
    y_probs = np.asarray(preds["y_probs"])
    groups, keys = _patient_groups(preds)
    K = y_probs.shape[1]
    reduce = {"mean": np.mean, "max": np.max, "min": np.min}[how]
    yp = np.zeros((len(keys), K))
    yt = np.zeros(len(keys), dtype=int)
    n_rep = np.zeros(len(keys), dtype=int)
    for i, g in enumerate(keys):
        sel = groups == g
        yp[i] = reduce(y_probs[sel], axis=0)
        yt[i] = np.bincount(y_true[sel], minlength=K).argmax()
        n_rep[i] = int(sel.sum())
    return yt, yp, n_rep, keys


def _patient_unanimous(preds):
    """Variant 1: argmax each experiment, then require unanimity.

    Returns (y_true, y_pred, n_replicates, patient_keys) where ``y_pred == K``
    (one past the last class index) marks a discordant patient -- replicates
    disagreed, so the patient is counted as wrong no matter its true class.
    """
    y_true = np.asarray(preds["y_true"]).astype(int)
    y_probs = np.asarray(preds["y_probs"])
    exp_pred = y_probs.argmax(axis=1)
    groups, keys = _patient_groups(preds)
    K = y_probs.shape[1]
    yt = np.zeros(len(keys), dtype=int)
    yp = np.zeros(len(keys), dtype=int)
    n_rep = np.zeros(len(keys), dtype=int)
    for i, g in enumerate(keys):
        sel = groups == g
        calls = np.unique(exp_pred[sel])
        yp[i] = int(calls[0]) if len(calls) == 1 else K
        yt[i] = np.bincount(y_true[sel], minlength=K).argmax()
        n_rep[i] = int(sel.sum())
    return yt, yp, n_rep, keys


def _patient_argmax_figure(out, ci, args, name, yt, y_pred, roc_probs, n_rep,
                           keys, cm_title, roc_title, extra_col=None):
    """Shared 2-panel (ROC | confusion matrix) figure for the patient variants."""
    style.apply()
    K = len(ci.class_names)
    names = ci.display_names
    n_lab = K + (1 if extra_col else 0)
    cols = list(ci.display_order) + ([K] if extra_col else [])
    cm_full = confusion_matrix(yt, y_pred, labels=list(range(n_lab)))
    cm = cm_full[np.ix_(ci.display_order, cols)]
    col_names = names + ([extra_col] if extra_col else [])
    _two_panel_roc_cm(out, name, ci, yt, roc_probs, cm, args.bootstrap_n,
                      roc_title=roc_title, cm_title=cm_title, col_names=col_names)

    pred_names = [ci.class_names[p] if p < K else extra_col for p in y_pred]
    pd.DataFrame({
        "sample_id": keys,
        "n_replicates": n_rep,
        "true_class": [ci.class_names[t] for t in yt],
        "predicted_class": pred_names,
        "correct": [int(p == t) for p, t in zip(y_pred, yt)],
        **{f"p_{ci.class_names[k]}": roc_probs[:, k] for k in range(K)},
    }).to_csv(os.path.join(out, f"{name}.csv"), index=False)

    acc = float(np.mean(y_pred == yt))
    print(f"  {name}: n={len(yt)} patients, accuracy={acc:.3f} "
          f"({int((y_pred == yt).sum())}/{len(yt)})")
    return acc


def fig1_cm_roc_argmax_patient_mean(out, ci, preds, args):
    """Patient level: average the replicate probability vectors, then plain
    argmax (no Youden shift). ROC uses the same averaged probabilities."""
    yt, yp, n_rep, keys = _patient_pool(preds, "mean")
    return _patient_argmax_figure(
        out, ci, args, "fig1_cm_roc_argmax_patient_mean", yt, yp.argmax(axis=1),
        yp, n_rep, keys,
        cm_title=f"Confusion Matrix\n(mean of replicates, argmax)\nn={len(yt)} patients",
        roc_title="ROC Curve (mean P across replicates)")


def table_patient_argmax_summary(out, ci, preds, args):
    """One table comparing the experiment-level argmax baseline with the three
    patient-level aggregation rules: accuracy, macro F1, per-class sensitivity,
    and per-class OvR AUC."""
    from sklearn.metrics import f1_score
    style.apply()
    K = len(ci.class_names)
    yt_e = np.asarray(preds["y_true"]).astype(int)
    yp_e = np.asarray(preds["y_probs"])

    yt_u, pred_u, _, _ = _patient_unanimous(preds)
    _, prob_u, _, _ = _patient_pool(preds, "min")
    yt_m, prob_m, _, _ = _patient_pool(preds, "mean")
    yt_x, prob_x, _, _ = _patient_pool(preds, "max")

    rows = []
    variants = [
        ("Experiment-level argmax", yt_e, yp_e.argmax(axis=1), yp_e),
        ("Patient: unanimous replicates", yt_u, pred_u, prob_u),
        ("Patient: mean of replicates", yt_m, prob_m.argmax(axis=1), prob_m),
        ("Patient: argmax over all replicates", yt_x, prob_x.argmax(axis=1), prob_x),
    ]
    for label, yt, pred, prob in variants:
        row = {"Rule": label, "n": len(yt),
               "Accuracy": f"{np.mean(pred == yt):.3f}",
               "Macro F1": f"{f1_score(yt, pred, labels=list(range(K)), average='macro', zero_division=0):.3f}"}
        for c in ci.display_order:
            name = ci.class_names[c]
            n_c = int((yt == c).sum())
            sens = float(((pred == c) & (yt == c)).sum()) / max(1, n_c)
            row[f"Sens {name}"] = f"{sens:.2f} ({int(((pred == c) & (yt == c)).sum())}/{n_c})"
            try:
                row[f"AUC {name}"] = f"{roc_auc_score((yt == c).astype(int), prob[:, c]):.3f}"
            except ValueError:
                row[f"AUC {name}"] = "n/a"
        rows.append(row)

    df = pd.DataFrame(rows)
    C.render_table(df, os.path.join(out, "table_patient_argmax_summary"),
                   title="Experiment- vs patient-level argmax rules")
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Fig 1: patient-level ROC/CM top row + a bottom row of per-image UMAP panels
# (true class / model-probability blend / sample_id).
# ---------------------------------------------------------------------------
def _align_image_probs(probs_dir, exp, imgid):
    """Align per-image model probs to the (exp, imgid) rows of the UMAP npz.

    Reads the per-image embeddings written by ``loo-per-image --fold_idx 0
    --all_experiments`` (``{eid}.embeddings.npz``; ``probs`` col order matches
    ``class_names``). Returns (N, 3) with NaN rows where probs are missing.
    """
    files = glob.glob(os.path.join(probs_dir, "*.embeddings.npz"))
    if not files:
        raise FileNotFoundError(f"no per-image embeddings under {probs_dir}")
    probs = np.full((len(exp), 3), np.nan, dtype=float)
    for f in files:
        d = np.load(f, allow_pickle=True)
        eid = str(d["experiment_id"])
        rows = np.where(exp == eid)[0]
        if rows.size == 0:
            continue
        ids, pr = np.asarray(d["image_id"]).astype(np.int64), np.asarray(d["probs"], dtype=float)
        order = np.argsort(ids)
        ids_s = ids[order]
        want = imgid[rows].astype(np.int64)
        pos = np.searchsorted(ids_s, want)
        ok = (pos < ids_s.size) & (ids_s[pos.clip(max=ids_s.size - 1)] == want)
        probs[rows[ok]] = pr[order][pos[ok]]
    return probs


def fig1_cm_roc_umap_sample_average(out, ci, preds, args):
    """Fig 1: top row = a) ROC, b) confusion matrix (Youden-shifted argmax) from
    patient-level probability-averaged predictions; bottom row = three per-image
    UMAP panels coloured by c) true class, d) model probability (blend of the
    class colours), e) sample_id."""
    import matplotlib.colors as mcolors

    if not os.path.exists(args.umap_npz):
        print(f"  fig1 skipped: {args.umap_npz} missing (run umap_images.py)")
        return "skip"

    style.apply()
    d = np.load(args.umap_npz, allow_pickle=True)
    coords = np.asarray(d["coords"], dtype=float)
    cls = np.asarray(d["cls"]).astype(int)
    exp = np.asarray(d["exp"]).astype(str)
    imgid = np.asarray(d["imgid"]).astype(np.int64)
    umap_class_names = [str(c) for c in d["class_names"]]   # npz order (== probs col order)

    # Render ALL images via datashader density aggregation (no subsampling).
    rng = np.random.default_rng(args.seed)

    class_hex = style.class_colors_muted(umap_class_names)   # order == umap_class_names

    # (e) sample_id per image (experiments without a known sample keep their id)
    samples = np.array([args.sample_map.get(e) or e for e in exp])
    uniq = sorted(set(samples))
    # Distinct, non-repeating colors per sample: 60 discrete colors from
    # tab20+tab20b+tab20c (shuffled), falling back to evenly-spaced HSV.
    base_cols = np.vstack([plt.get_cmap(c)(np.arange(20))[:, :3]
                           for c in ("tab20", "tab20b", "tab20c")])
    base_cols = base_cols[rng.permutation(len(base_cols))]
    if len(uniq) > len(base_cols):
        base_cols = plt.get_cmap("hsv")(np.linspace(0, 1, len(uniq), endpoint=False))[:, :3]
    sample_color = {s: base_cols[i] for i, s in enumerate(uniq)}

    # (d) per-image model probs (npz class order)
    probs = _align_image_probs(args.umap_probs_dir, exp, imgid)
    valid = ~np.isnan(probs).any(axis=1)

    # ---- datashader density rasters (overplot-free for ~1.3M points) ----
    import datashader as ds
    import datashader.transfer_functions as tf
    xr = (float(coords[:, 0].min()), float(coords[:, 0].max()))
    yr = (float(coords[:, 1].min()), float(coords[:, 1].max()))
    PW = 1100
    PH = max(1, int(round(PW * (yr[1] - yr[0]) / (xr[1] - xr[0]))))
    cvs = ds.Canvas(plot_width=PW, plot_height=PH, x_range=xr, y_range=yr)

    dfp = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    dfp["cls"] = pd.Categorical([umap_class_names[i] for i in cls])
    dfp["sample"] = pd.Categorical(samples)

    def _cat_img(col, color_key):
        agg = cvs.points(dfp, "x", "y", ds.count_cat(col))
        img = tf.set_background(tf.shade(agg, color_key=color_key, how="eq_hist"), "white")
        return np.asarray(img.to_pil())

    img_true = _cat_img("cls", {umap_class_names[i]: mcolors.to_hex(class_hex[i])
                                for i in range(len(umap_class_names))})
    img_sample = _cat_img("sample", {s: mcolors.to_hex(sample_color[s]) for s in uniq})

    # probability panel: per-pixel mean class-probability -> RGB blend of the
    # class colours, intensity modulated by log point density, over white.
    dfq = dfp[valid].copy()
    for j in range(probs.shape[1]):
        dfq[f"p{j}"] = probs[valid, j]
    cnt = cvs.points(dfq, "x", "y", ds.count()).values.astype(float)
    means = [np.nan_to_num(cvs.points(dfq, "x", "y", ds.mean(f"p{j}")).values)
             for j in range(probs.shape[1])]
    Pm = np.stack(means, axis=-1)                                  # (H, W, K)
    Wt = Pm / np.clip(Pm.sum(axis=-1, keepdims=True), 1e-9, None)
    crgb = np.array([mcolors.to_rgb(class_hex[i]) for i in range(len(umap_class_names))])
    blend = Wt @ crgb                                              # (H, W, 3)
    dens = np.zeros_like(cnt); msk = cnt > 0
    dens[msk] = np.log1p(cnt[msk])
    if dens.max() > 0:
        dens /= dens.max()
    # Alpha with a floor so any populated pixel keeps saturated colour (density
    # still brightens the densest regions); use the blend directly, no white mix.
    alpha = np.zeros_like(cnt)
    alpha[msk] = 0.4 + 0.45 * dens[msk]
    img_prob = np.ones((*cnt.shape, 3))
    img_prob[msk] = (1 - alpha[msk, None]) + alpha[msk, None] * blend[msk]
    img_prob = (np.flipud(np.clip(img_prob, 0, 1)) * 255).astype(np.uint8)

    # ---- layout: top ROC/CM (patient level), bottom 3 UMAP panels ----
    y_true, y_probs, _ = _sample_average_preds(preds)
    thr = C.youden_thresholds(y_true, y_probs, ci.class_names)
    y_pred_y = (y_probs - thr).argmax(axis=1)
    cm_full = confusion_matrix(y_true, y_pred_y, labels=list(range(len(ci.class_names))))
    cm = cm_full[np.ix_(ci.display_order, ci.display_order)]
    names = ci.display_names
    roc_colors = style.class_colors_muted(names)

    # Double column (183 mm), 13.5:11 proportions.
    fig = plt.figure(figsize=(style.DOUBLE, style.DOUBLE * 11 / 13.5))
    # One GridSpec per row so the top pair and the bottom trio get independent
    # horizontal gaps.
    gs_top = gridspec.GridSpec(1, 2, figure=fig, wspace=0.204,
                               left=0.07, right=0.98, top=0.945, bottom=0.554)
    gs_bot = gridspec.GridSpec(1, 3, figure=fig, wspace=0.234,
                               left=0.07, right=0.98, top=0.446, bottom=0.07)
    ax_roc = fig.add_subplot(gs_top[0, 0]); ax_cm = fig.add_subplot(gs_top[0, 1])
    _plot_roc(ax_roc, y_true, y_probs, names, ci.display_order, roc_colors, args.bootstrap_n)
    ax_roc.set_title("ROC Curve")
    _plot_confusion(ax_cm, cm, names)
    ax_cm.set_title("Confusion Matrix")
    # Done before the UMAP panels exist so the intermediate draws stay cheap.
    _align_cm_xlabel(fig, ax_roc, ax_cm)

    def _umap_panel(gscol, img, title):
        ax = fig.add_subplot(gscol)
        # extent maps the raster back to UMAP coordinates so the axes carry ticks
        # (row 0 of the raster is max-y, hence origin="upper").
        ax.imshow(img, extent=[xr[0], xr[1], yr[0], yr[1]], origin="upper",
                  aspect="equal", interpolation="none")
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        return ax

    ax_c = _umap_panel(gs_bot[0, 0], img_true, "True Class")
    disp_cn = [C.pretty_class_name(n) for n in umap_class_names]
    # One class legend for the whole bottom row, in panel c's empty lower-right
    # corner; panel d (probability blend) shares the same colours.
    swatches = [plt.Line2D([0], [0], marker="o", ls="", markersize=4,
                           markerfacecolor=class_hex[i], markeredgecolor="none",
                           label=disp_cn[i]) for i in range(len(umap_class_names))]
    ax_c.legend(handles=swatches, loc="lower right", frameon=False)
    _umap_panel(gs_bot[0, 1], img_prob, "Model Class Probability")
    _umap_panel(gs_bot[0, 2], img_sample, "Sample ID")

    for ax, lab in [(ax_roc, "a"), (ax_cm, "b")]:
        _panel_key(ax, lab)
    for ax, lab in zip(fig.axes[2:5], ["c", "d", "e"]):
        _panel_key(ax, lab, dx_pt=-4)

    fname = "fig1_cm_roc_umap_sample_average.png"
    fig.savefig(os.path.join(out, fname))
    plt.close(fig)
    print(f"  {fname}: {len(coords)} images, {len(uniq)} samples, "
          f"{int((~valid).sum())} images missing probs")


# ===========================================================================
# Table 2: per-class performance
# ===========================================================================
def _clinical_perf_table(y_true, y_probs, ci, bootstrap_n=1000, seed=0):
    """Per-class clinical metrics under Youden-shifted argmax, 95% bootstrap CIs
    + Macro row.

    Columns: Class, Support, F1, Accuracy, Sensitivity, Specificity, PPV, NPV.
    Predictions use the SAME rule as the fig1 confusion matrix -- subtract each
    class's Youden-J threshold from its probability, then argmax -- so every
    experiment gets exactly one predicted class and table2 agrees with the CM.
    Thresholds are fixed on the full pooled data and held constant across bootstrap
    resamples. Rows follow display order; Macro = unweighted mean across classes.
    """
    from remo_id.reports.figures._bootstrap import _bootstrap_ci

    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs)

    # Per-class Youden-J thresholds, fixed on full data (identical to fig1 CM).
    thr = C.youden_thresholds(y_true, y_probs, ci.class_names)

    def _counts(yt, yp, c):
        pred = (yp - thr).argmax(axis=1)  # Youden-shifted argmax (coherent)
        yb = (yt == c)
        pc = (pred == c)
        tp = int((pc & yb).sum()); fp = int((pc & ~yb).sum())
        fn = int((~pc & yb).sum()); tn = int((~pc & ~yb).sum())
        return tp, fp, fn, tn

    def _metric(c, which):
        def fn(yt, yp):
            tp, fp, fn_, tn = _counts(yt, yp, c)
            sens = tp / max(1, tp + fn_)
            spec = tn / max(1, tn + fp)
            if which == "sensitivity":
                return sens
            if which == "specificity":
                return spec
            if which == "ppv":
                return tp / max(1, tp + fp)
            if which == "npv":
                return tn / max(1, tn + fn_)
            if which == "lr+":
                # sensitivity / (1 - specificity); inf when specificity == 1
                return sens / (1 - spec) if spec < 1 else np.inf
            if which == "lr-":
                # (1 - sensitivity) / specificity
                return (1 - sens) / spec if spec > 0 else np.inf
            ppv = tp / max(1, tp + fp)
            return 2 * ppv * sens / max(1e-12, ppv + sens)  # f1
        return fn

    classes = list(ci.display_order)

    def _macro(which):
        def fn(yt, yp):
            return float(np.mean([_metric(c, which)(yt, yp) for c in classes]))
        return fn

    def _fmt(point, lo, hi):
        return f"{point:.2f} [{lo:.2f}–{hi:.2f}]"

    def _fmt_lr(point, lo, hi):
        def _v(x):
            return "∞" if not np.isfinite(x) else f"{x:.2f}"
        return f"{_v(point)} [{_v(lo)}–{_v(hi)}]"

    metric_cols = ["F1", "Sensitivity", "Specificity", "PPV", "NPV", "LR+", "LR-"]
    key = {"F1": "f1", "Sensitivity": "sensitivity",
           "Specificity": "specificity", "PPV": "ppv", "NPV": "npv",
           "LR+": "lr+", "LR-": "lr-"}
    fmt = {col: (_fmt_lr if col in ("LR+", "LR-") else _fmt) for col in metric_cols}

    rows = []
    for c, name in zip(classes, ci.display_names):
        row = {"Class": name, "Support": int((y_true == c).sum())}
        for col in metric_cols:
            row[col] = fmt[col](*_bootstrap_ci(y_true, y_probs, _metric(c, key[col]),
                                               n=bootstrap_n, seed=seed))
        rows.append(row)

    macro = {"Class": "Macro", "Support": len(y_true)}
    for col in metric_cols:
        # Likelihood ratios don't average meaningfully (Healthy's LR+ is ∞ from
        # perfect specificity), so leave them blank in the Macro row.
        if col in ("LR+", "LR-"):
            macro[col] = ""
            continue
        macro[col] = fmt[col](*_bootstrap_ci(y_true, y_probs, _macro(key[col]),
                                             n=bootstrap_n, seed=seed))
    rows.append(macro)

    return pd.DataFrame(rows, columns=["Class", "Support"] + metric_cols)


def table2_performance(out, ci, preds, args):
    df = _clinical_perf_table(preds["y_true"], preds["y_probs"], ci,
                              bootstrap_n=args.bootstrap_n, seed=args.seed)
    df.to_csv(os.path.join(out, "table2_performance.csv"), index=False)
    C.render_table(df, os.path.join(out, "table2_performance"),
                   title="Performance metrics (95% bootstrap CI)")


# ===========================================================================
# Subpanel labels at the top-left CORNER of each panel (outside the data area)
# ===========================================================================
def _corner_panel_labels(axes, fontsize=style.PANEL_LABEL_PT):
    import re
    import string
    visible = [ax for ax in axes if ax is not None and ax.get_visible()]
    pat = re.compile(r"^[a-z]\)?$")
    for ax in visible:
        # strip any pre-existing single-letter panel labels (e.g. from
        # small_multiples_grid) so we don't end up with two sets
        for txt in list(ax.texts):
            if pat.match(txt.get_text().strip()):
                txt.remove()
    for i, ax in enumerate(visible):
        ax.annotate(
            f"{string.ascii_lowercase[i]}",
            xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(-13, 4), textcoords="offset points",
            ha="left", va="bottom", fontsize=fontsize, fontweight="bold",
            color=style.FG, annotation_clip=False,
        )


# ===========================================================================
# supFig 7: threshold sensitivity (one-vs-rest)
# ===========================================================================
def supfig7_threshold_sensitivity(out, ci, preds):
    """F1 / precision / recall vs threshold per class (OvR), display order,
    with subpanel callouts in the top-left corner of each panel."""
    style.apply()
    names = ci.display_names
    y_true = preds["y_true"].astype(int)
    y_probs = preds["y_probs"]
    n = len(names)
    thr = np.linspace(0.01, 0.99, 99)

    ncols = min(n, 2)
    nrows = int(np.ceil(n / ncols))
    fig, axes = style.figure(ncols=ncols, nrows=nrows, width="double", aspect=0.8,
                             sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.ravel()
    for k, data_idx in enumerate(ci.display_order):
        ax = axes_flat[k]
        y_bin = (y_true == data_idx).astype(int)
        y_sc = y_probs[:, data_idx]
        prec, rec, f1s = [], [], []
        for t in thr:
            yp = (y_sc >= t).astype(int)
            tp = int(((yp == 1) & (y_bin == 1)).sum())
            fp = int(((yp == 1) & (y_bin == 0)).sum())
            fn = int(((yp == 0) & (y_bin == 1)).sum())
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            prec.append(p); rec.append(r)
            f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
        ax.plot(thr, f1s, color=style.ACCENT, lw=1.4, label="F1")
        ax.plot(thr, prec, color=style.MUTE, lw=0.9, ls=(0, (3, 2)), label="Precision")
        ax.plot(thr, rec, color=style.FG, lw=0.9, ls=(0, (1, 2)), label="Recall")
        ax.set_xlabel("Threshold"); ax.set_ylim(0, 1)
        ax.set_title(names[k])
        if k == 0:
            ax.set_ylabel("Score")
            ax.legend(loc="lower left")
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Threshold Sensitivity (One-vs-Rest)", y=0.995,
                 fontsize=style.font_pt("axes.titlesize"), fontweight="bold")
    fig.tight_layout(rect=(0.03, 0, 1, 0.975))
    _corner_panel_labels(axes_flat)
    fig.savefig(os.path.join(out, "supfig7_threshold_sensitivity.png"))
    plt.close(fig)


# ===========================================================================
# Clinical metadata (sepsis study) -- shared by Fig3 / Fig4 / Tables
# ===========================================================================
# Every study-specific choice (columns, labels, vocabularies, table rows) comes
# from the clinical-table spec (configs/clinical_tables.yaml, ``clin.spec``) and
# every value from the clinical input table (``clin.table``); see common.Clinical.
_TRUE_TOKENS = {"true", "1", "yes", "y", "t"}


def _is_true(v):
    """Truthiness for booleans that may arrive as bool, 0/1, 1.0, or a string."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None:
        return False
    if isinstance(v, (int, float, np.integer, np.floating)):
        return not np.isnan(v) and v != 0
    return str(v).strip().lower() in _TRUE_TOKENS


def _is_female(v):
    """Sex coded F/M or Female/Male."""
    if v is None:
        return False
    return str(v).strip().lower().startswith("f")


def _warn_missing_columns(df, cols, where):
    """Fail loudly when a source column is absent from the clinical table.

    A missing column silently coerces to 0 for every patient (a table reporting
    Mortality 0 / Gram+ 0 for the whole cohort).
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = (f"{where}: source column(s) {missing} absent from the study rows -- "
               "these fields would silently report 0. Fix the column mapping.")
        warnings.warn(msg)
        print(f"  !! {msg}")
    return missing


def _group_col(ci):
    """The sepsis class's label column (class config) = the cohort group column."""
    col = ci.class_definitions[ci.sepsis_idx].get("column")
    if not col:
        raise ValueError("the sepsis class in the class config has no `column`")
    return col


def _var_groups(spec):
    """Fig 3 row groups from the spec; [min, max] pairs become tuples."""
    return {g: [tuple(e) if isinstance(e, list) else e for e in entries]
            for g, entries in spec["correlation_groups"].items()}


def _clinical_vars(spec):
    """Fig 4/4b clinical variables: {label: column}, in spec order."""
    labels = spec["variable_labels"]
    return {labels[k]: k for k in spec["discrimination"]["variables"] if k in labels}


def table2b_ovr_auc(out, ci, preds, args):
    """One-vs-rest AUC per class with 95% bootstrap CIs (+ macro), from the pooled
    held-out predictions. Complements table2 (which reports threshold metrics)."""
    style.apply()
    y_true = preds["y_true"].astype(int)
    y_probs = preds["y_probs"]
    order = [k for k in ci.display_order if np.unique((y_true == k).astype(int)).size >= 2]

    def _macro(yt, yp):
        a = [roc_auc_score((yt == k).astype(int), yp[:, k]) for k in order
             if np.unique((yt == k).astype(int)).size >= 2]
        return float(np.mean(a)) if a else np.nan

    rows = []
    for k in order:
        yb = (y_true == k).astype(int)
        pt, lo, hi = _bootstrap_ci(yb, y_probs[:, k], roc_auc_score,
                                   n=args.bootstrap_n, seed=args.seed)
        rows.append({"Class": ci.class_names[k], "Positives": int(yb.sum()),
                     "N": int(len(yb)), "AUC (OvR) [95% CI]": f"{pt:.2f} [{lo:.2f}–{hi:.2f}]"})
    if order:
        # macro CI: resample rows (paired y/probs) and recompute macro-OvR AUC
        rng = np.random.default_rng(args.seed)
        n = len(y_true)
        boot = [_macro(y_true[idx], y_probs[idx])
                for idx in (rng.integers(0, n, n) for _ in range(args.bootstrap_n))]
        boot = [b for b in boot if not np.isnan(b)]
        mlo, mhi = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
        rows.append({"Class": "Macro", "Positives": "", "N": int(len(y_true)),
                     "AUC (OvR) [95% CI]": f"{_macro(y_true, y_probs):.2f} [{mlo:.2f}–{mhi:.2f}]"})
    C.render_table(pd.DataFrame(rows), os.path.join(out, "table2b_ovr_auc"),
                   title="One-vs-rest AUC per class (95% bootstrap CI)")
    print(f"  table2b OvR AUC: " + "; ".join(f"{r['Class']}={r['AUC (OvR) [95% CI]']}" for r in rows))


_OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}


def load_clinical_metadata(clin, study_id):
    """One study's rows of the clinical table, cleaned per the spec's `preprocess`
    (missing-value codes, capped numeric age, 0/1 booleans, sparse flags, derived
    scores)."""
    pre = clin.spec["preprocess"]
    df = clin.study(study_id)
    df["experiment_id"] = df["experiment_id"].astype(str)
    if "sample_id" in df.columns:
        df["sample_id"] = df["sample_id"].astype(str)
    for col, codes in pre.get("missing_codes", {}).items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col].isin(codes), col] = np.nan
    # Age: HIPAA caps ages >89 (recorded as ">89"/"90+"); treat all as the cap and
    # combine into the numeric age so it's usable in medians/ranges/tests.
    age = pre.get("age")
    if age and age["column"] in df.columns:
        a = df[age["column"]].astype(str).str.extract(r"(\d+\.?\d*)")[0]
        df[age["column"]] = pd.to_numeric(a, errors="coerce").clip(upper=age["cap"])
    for col in pre.get("boolean_columns", []):
        if col in df.columns:
            df[col] = df[col].map(lambda v: float(_is_true(v)) if v is not None
                                  and str(v).strip().lower() not in ("", "nan", "none")
                                  else np.nan)
    # A SPARSE flag is set only for the flagged patients, so a null means "no",
    # not "unknown". Left as null it collapses the denominator to the flagged rows
    # alone and prints 1 (100.0%).
    for col in pre.get("sparse_flags", []):
        if col in df.columns:
            df[col] = df[col].map(lambda v: float(_is_true(v)))
    # Derived scores: sum of 0/1 terms, each true if any of its conditions holds.
    for name, terms in pre.get("derived_scores", {}).items():
        cols = [c for term in terms for c, _, _ in term]
        if not set(cols) <= set(df.columns):
            continue
        score = None
        for term in terms:
            hit = None
            for c, op, v in term:
                h = _OPS[op](df[c], v)
                hit = h if hit is None else (hit | h)
            score = hit.astype(float) if score is None else score + hit.astype(float)
        df[name] = score
        df.loc[df[list(dict.fromkeys(cols))].isna().any(axis=1), name] = np.nan
    return df


def _all_vars(spec):
    """Every clinical variable named in the Fig 3 groups, in group order (a list,
    so output row order does not depend on hash randomization)."""
    out = []
    for entries in _var_groups(spec).values():
        for e in entries:
            out.extend(v for v in (e if isinstance(e, tuple) else (e,)) if v not in out)
    return out


def build_patient_df(ci, preds, clinical, spec):
    """Patient-level merge of model probabilities with sepsis clinical metadata."""
    cols = {ci.prob_col(i): preds["y_probs"][:, i] for i in range(len(ci.class_names))}
    pred_df = pd.DataFrame({"experiment_id": preds["eids"].astype(str), **cols})
    merged = pred_df.merge(clinical, on="experiment_id", how="inner", suffixes=("", "_clin"))
    prob_cols = [ci.prob_col(i) for i in range(len(ci.class_names))]
    agg = {c: "mean" for c in prob_cols}
    for v in _all_vars(spec) + [_group_col(ci)]:
        if v in merged.columns and v not in prob_cols and v not in agg:
            agg[v] = "first"
    agg = {k: v for k, v in agg.items() if k in merged.columns}
    return merged.groupby("sample_id").agg(agg).reset_index()


# ===========================================================================
# Fig 3: clinical correlations heatmap
# ===========================================================================
def _disp_prob_cols(ci):
    return [ci.prob_col(i) for i in ci.display_order]


def _fmt_p(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "p < .001"
    if p < 0.01:
        return "p = " + f"{p:.3f}"[1:]
    return "p = " + f"{p:.2f}"[1:]


def _p_cell(p):
    """Bare p-value string for a table cell (no 'p = ' prefix). Strips the leading
    zero (.03) but keeps a leading 1 so p=1.00 renders as '1.00', not '.00'."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "< .001"
    s = f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
    return s[1:] if s[0] == "0" else s


def _group_pvalue(a, b, kind):
    """Two-sided p-value comparing group a vs b for one variable.

    kind='num'  -> Mann-Whitney U ; kind='binary'/'cat' -> Fisher exact (2x2 on
    the 0/1 indicator). Returns NaN if either group has < 2 usable values.
    """
    from scipy import stats
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna()
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        if kind == "num":
            if a.nunique() == 1 and b.nunique() == 1 and a.iloc[0] == b.iloc[0]:
                return np.nan
            return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        a1, a0 = int((a == 1).sum()), int((a != 1).sum())
        b1, b0 = int((b == 1).sum()), int((b != 1).sum())
        return float(stats.fisher_exact([[a1, a0], [b1, b0]])[1])
    except Exception:  # noqa: BLE001
        return np.nan


def _compute_correlations(patient, disp_cols, spec):
    rows = []
    for var in _all_vars(spec):
        if var not in patient.columns:
            continue
        for pc in disp_cols:
            mask = patient[var].notna() & patient[pc].notna()
            x = patient.loc[mask, var].values.astype(float)
            y = patient.loc[mask, pc].values.astype(float)
            if len(x) < 3:
                rows.append(dict(variable=var, prob=pc, rho=np.nan, p=np.nan, n=len(x)))
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, p = stats.spearmanr(x, y)
            rows.append(dict(variable=var, prob=pc, rho=rho, p=p, n=len(x)))
    return pd.DataFrame(rows)


def _resolve_var_order(patient, corr, spec):
    order = []
    for entries in _var_groups(spec).values():
        for e in entries:
            if isinstance(e, tuple):
                scores = []
                for v in e:
                    r = corr[corr["variable"] == v]["rho"].abs().max()
                    scores.append(0.0 if (r is None or np.isnan(r)) else r)
                winner = e[int(np.argmax(scores))]
                if winner in patient.columns:
                    order.append(winner)
            elif e in patient.columns:
                order.append(e)
    return order


def fig3_clinical_correlations(out, ci, preds, clin):
    style.apply()
    spec = clin.spec
    labels = spec["variable_labels"]
    clinical = load_clinical_metadata(clin, ci.sepsis_study)
    patient = build_patient_df(ci, preds, clinical, spec)
    disp_cols = _disp_prob_cols(ci)
    corr = _compute_correlations(patient, disp_cols, spec)
    var_order = _resolve_var_order(patient, corr, spec)
    corr = corr[corr["variable"].isin(var_order)]

    rho_p = corr.pivot(index="variable", columns="prob", values="rho").loc[var_order, disp_cols]
    p_p = corr.pivot(index="variable", columns="prob", values="p").loc[var_order, disp_cols]
    n_rows, n_cols = len(var_order), len(disp_cols)
    ylabels = [labels.get(v, v) for v in var_order]

    var_group = {}
    for g, entries in _var_groups(spec).items():
        for e in entries:
            for v in (e if isinstance(e, tuple) else (e,)):
                var_group[v] = g
    bounds = []
    for i, v in enumerate(var_order):
        g = var_group.get(v)
        if g and (not bounds or bounds[-1][1] != g):
            bounds.append((i, g))

    cell_h, cell_w = 0.32, 1.05
    left, right, top, bot = 1.6, 1.0, 0.12, 0.55
    fw = left + n_cols * cell_w + right
    fh = top + n_rows * cell_h + bot
    fig, ax = plt.subplots(figsize=(fw, fh))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "tufte_div", ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B"], N=256)
    vmax = 0.85
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    data = rho_p.values.copy()
    pdat = p_p.values
    disp = np.where(pdat < 0.05, data, 0.0)
    im = ax.imshow(disp, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    for i in range(n_rows):
        for j in range(n_cols):
            rho, p = data[i, j], pdat[i, j]
            if np.isnan(rho):
                continue
            sig = p < 0.05
            tc = "white" if abs(disp[i, j]) / vmax > 0.7 else style.FG
            ax.text(j, i - 0.11, f"{rho:+.2f}", ha="center", va="center", fontsize=6.5,
                    color=tc if sig else style.MUTE, fontweight="bold" if sig else "normal",
                    family="serif")
            ax.text(j, i + 0.19, _fmt_p(p), ha="center", va="center", fontsize=4.5,
                    color=tc if sig else style.MUTE_LIGHT)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(disp_cols, fontsize=7, fontweight="bold")
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(ylabels, fontsize=6.5)
    ax.tick_params(length=0); ax.xaxis.set_ticks_position("bottom")
    for idx, (start, gname) in enumerate(bounds):
        if start > 0:
            ax.axhline(y=start - 0.5, color="white", lw=1.5)
        end = bounds[idx + 1][0] if idx < len(bounds) - 1 else n_rows
        ax.text(n_cols - 1 + 0.65, (start + end - 1) / 2, gname.upper(), ha="left",
                va="center", fontsize=5, color=style.MUTE, fontstyle="italic", clip_on=False)
    style.strip_imshow_frame(ax)

    cbar_ax = fig.add_axes([left / fw + 0.15 * (n_cols * cell_w) / fw, 0.08 / fh,
                            0.7 * (n_cols * cell_w) / fw, 0.012])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Spearman $\\rho$", fontsize=6, labelpad=3)
    cbar.set_ticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])
    cbar.ax.tick_params(labelsize=5); style.thin_colorbar(cbar)
    fig.subplots_adjust(left=left / fw, right=(left + n_cols * cell_w) / fw,
                        bottom=bot / fh, top=1 - top / fh)

    corr_out = corr.copy()
    corr_out["variable_label"] = corr_out["variable"].map(labels)
    corr_out.to_csv(os.path.join(out, "fig3_clinical_correlations.csv"), index=False)
    for ext in ("png",):
        fig.savefig(os.path.join(out, f"fig3_clinical_correlations.{ext}"), dpi=300,
                    facecolor="white")
    plt.close(fig)


# ===========================================================================
# Fig 4: sepsis+/- AUC ranking (model vs clinical variables)
# ===========================================================================
def _auc_directional(y, x):
    auc = roc_auc_score(y, x)
    d = "+"
    if auc < 0.5:
        auc, d = 1.0 - auc, "-"
    return auc, d


def _auc_ci_directional(y, x, n=1000, seed=0):
    """Directional AUC + 95% bootstrap CI (resamples the score, no refit)."""
    point, d = _auc_directional(y, x)
    xs = np.asarray(x, dtype=float)
    xs = xs if d == "+" else -xs
    _, lo, hi = _bootstrap_ci(y, xs, roc_auc_score, n=n, seed=seed)
    return point, lo, hi, d


def _loo_proba(X, y):
    """Leave-one-out logistic-regression out-of-fold P(y=1) per sample."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    pr = np.zeros(len(y))
    for tr, te in loo.split(X):
        sc = StandardScaler()
        lr = LogisticRegression(max_iter=1000, random_state=0)
        lr.fit(sc.fit_transform(X[tr]), y[tr])
        pr[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
    return pr


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted q-values (NaNs preserved)."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return q
    pv = p[idx]; nnz = pv.size
    order = np.argsort(pv)
    qs = np.empty(nnz); prev = 1.0
    for rank in range(nnz, 0, -1):
        val = pv[order[rank - 1]] * nnz / rank
        prev = min(prev, val)
        qs[order[rank - 1]] = prev
    q[idx] = np.clip(qs, 0, 1)
    return q


def fig4_sepsis_auc_ranking(out, ci, preds, clin, bootstrap_n=1000, seed=0,
                            min_combo_n=10, suffix=""):
    """Sepsis+/- discrimination ranking.

    Single-variable bars are raw directional rank-AUC. Fusions are P(sepsis) +
    ONE or TWO clinical variables, scored by leave-one-out logistic regression.
    Each fusion carries two significance tests against P(sepsis) alone:
      * likelihood-ratio test (nested, unregularized MLE, in-sample): does adding
        the clinical term(s) improve the model fit? chi-square, df = #added terms.
        This is the principled "does clinical add information?" test and is the
        p-value shown on the figure.
      * DeLong (out-of-fold LOO AUC vs same-estimator P(sepsis)-alone), kept in
        the CSV. Note AUC is NOT what the LR optimises, so a fusion's LOO AUC can
        fall below P(sepsis) even when the LRT is non-significant.
    """
    style.apply()
    import itertools
    from scipy.stats import chi2
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from remo_id.analysis.statistical_tests import delong_test

    clinical = load_clinical_metadata(clin, ci.sepsis_study)
    patient = build_patient_df(ci, preds, clinical, clin.spec)
    gcol = _group_col(ci)
    if gcol not in patient.columns:
        raise RuntimeError(f"{gcol} missing; cannot rank sepsis discrimination")
    y_sep = patient[gcol].values.astype(float)
    clinical_vars = _clinical_vars(clin.spec)
    fusion_exclude = set(clin.spec["discrimination"]["fusion_exclude"])

    sep_col = ci.prob_col(ci.sepsis_idx)
    model_cols = [(f"Remoscope P({ci.class_names[ci.sepsis_idx]})", sep_col)]
    if ci.neg_idx is not None:
        model_cols.append((f"Remoscope P({ci.class_names[ci.neg_idx]})", ci.prob_col(ci.neg_idx)))
    if ci.healthy_idx is not None:
        model_cols.append((f"Remoscope P({ci.class_names[ci.healthy_idx]})", ci.prob_col(ci.healthy_idx)))

    def _loolr(cols):
        """LOO logistic-regression out-of-fold scores over rows where all cols
        (and the label) are present. Returns None if too few / degenerate."""
        present = [c for c in cols if c in patient.columns]
        if len(present) != len(cols):
            return None
        mask = patient[cols + [gcol]].notna().all(axis=1).values
        yb = y_sep[mask]
        if mask.sum() < min_combo_n or np.unique(yb).size < 2:
            return None
        X = patient[cols].values[mask].astype(float)
        if any(np.unique(X[:, j]).size < 2 for j in range(X.shape[1])):
            return None
        pr = _loo_proba(X, yb)
        return dict(mask=mask, y=yb, pr=pr, auc=float(roc_auc_score(yb, pr)), n=int(mask.sum()))

    rows = []
    # ---- single variables: raw directional rank-AUC (unfitted), as preferred ----
    for name, col in model_cols:
        m = (patient[col].notna() & patient[gcol].notna()).values
        if m.sum() < 3 or np.unique(y_sep[m]).size < 2:
            continue
        auc, lo, hi, d = _auc_ci_directional(y_sep[m], patient[col].values[m],
                                             n=bootstrap_n, seed=seed)
        rows.append(dict(Variable=name, AUC=auc, AUC_lo=lo, AUC_hi=hi,
                         direction=d, n=int(m.sum()), type="model", p_delong=np.nan))
    for name, col in clinical_vars.items():
        if col not in patient.columns:
            continue
        m = (patient[col].notna() & patient[gcol].notna()).values
        if m.sum() < 3 or np.unique(y_sep[m]).size < 2:
            continue
        try:
            auc, lo, hi, d = _auc_ci_directional(y_sep[m], patient[col].values[m].astype(float),
                                                 n=bootstrap_n, seed=seed)
        except ValueError:
            continue
        rows.append(dict(Variable=name, AUC=auc, AUC_lo=lo, AUC_hi=hi,
                         direction=d, n=int(m.sum()), type="clinical", p_delong=np.nan))

    # ---- Fusions: P(sepsis) + {1 or 2} added features (LOO-LR). Candidate added
    #      features are the clinical variables AND the model's other class probs
    #      P(ICU-Control)/P(Healthy). Two tests per fusion vs P(sepsis) alone:
    #        * likelihood-ratio test (nested, unregularized MLE, in-sample) -> the
    #          principled "does the added term improve the fit" test (shown on fig)
    #        * DeLong (LOO out-of-fold AUC vs same-estimator P(sepsis)-alone; CSV)
    def _loglik(cols, mask):
        X = StandardScaler().fit_transform(patient[cols].values[mask].astype(float))
        yy = y_sep[mask]
        lr = LogisticRegression(C=1e12, solver="lbfgs", max_iter=5000).fit(X, yy)
        p = np.clip(lr.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return float(np.sum(yy * np.log(p) + (1 - yy) * np.log(1 - p)))

    def _lrt(add_cols, mask):
        stat = max(0.0, 2.0 * (_loglik([sep_col] + list(add_cols), mask) - _loglik([sep_col], mask)))
        return stat, float(chi2.sf(stat, df=len(add_cols)))

    neg_col = ci.prob_col(ci.neg_idx) if ci.neg_idx is not None else None
    heal_col = ci.prob_col(ci.healthy_idx) if ci.healthy_idx is not None else None
    add_features = [(col, label) for label, col in clinical_vars.items()
                    if col not in fusion_exclude and col in patient.columns]
    if neg_col is not None:
        add_features.append((neg_col, f"P({ci.class_names[ci.neg_idx]})"))
    if heal_col is not None:
        add_features.append((heal_col, f"P({ci.class_names[ci.healthy_idx]})"))
    model_prob_cols = {c for c in (neg_col, heal_col) if c is not None}

    fusion_rows = []
    for k in (1, 2):
        for combo in itertools.combinations(add_features, k):
            add_cols = [c for c, _ in combo]
            # skip the fully-collinear all-3-class combo (P(sep)+P(icu)+P(healthy) sum to 1)
            if model_prob_cols and set(add_cols) >= model_prob_cols and len(add_cols) >= 2:
                continue
            r = _loolr([sep_col] + add_cols)
            if r is None:
                continue
            base_score = _loo_proba(patient[[sep_col]].values[r["mask"]].astype(float), r["y"])
            base_auc = float(roc_auc_score(r["y"], base_score))
            try:
                p_delong = float(delong_test(r["y"].astype(int), r["pr"], base_score)["p_value"])
            except Exception:
                p_delong = np.nan
            try:
                lr_stat, p_lrt = _lrt(add_cols, r["mask"])
            except Exception:
                lr_stat, p_lrt = np.nan, np.nan
            lo = hi = np.nan   # no per-fusion CI (not plotted; 405 combos = too slow)
            kind = ("model" if all(c in model_prob_cols for c in add_cols)
                    else "clinical" if not any(c in model_prob_cols for c in add_cols)
                    else "mixed")
            fusion_rows.append(dict(
                Variable="P(Sepsis) + " + " + ".join(lab for _, lab in combo),
                AUC=r["auc"], AUC_lo=lo, AUC_hi=hi, direction="", n=r["n"],
                type="model+clinical", k_added=k, feature_kind=kind,
                p_lrt=p_lrt, lrt_stat=lr_stat, p_delong=p_delong,
                base_psepsis_lr_auc=base_auc, delta_auc=r["auc"] - base_auc))
    fusion_df = (pd.DataFrame(fusion_rows).sort_values("AUC", ascending=False)
                 if fusion_rows else pd.DataFrame())
    if len(fusion_df):
        # Multiple-comparison control: BH-FDR over all fusion LRT p-values.
        fusion_df["q_lrt"] = _bh_fdr(fusion_df["p_lrt"].values)

    mref = (patient[sep_col].notna() & patient[gcol].notna()).values
    ref_auc, _ = _auc_directional(y_sep[mref], patient[sep_col].values[mref])
    print(f"  P(Sepsis) alone raw AUC={ref_auc:.2f} (n={int(mref.sum())})")
    if len(fusion_df):
        n_tests = len(fusion_df)
        n_raw = int((fusion_df["p_lrt"] < 0.05).sum())
        n_fdr = int((fusion_df["q_lrt"] < 0.05).sum())
        b = fusion_df.iloc[0]
        print(f"  {n_tests} fusions (P(Sepsis)+<=2 of {len(add_features)} features "
              f"[clinical + P(ICU)/P(Healthy)]); best AUC = {b['Variable']} {b['AUC']:.2f} "
              f"(LRT p={b['p_lrt']:.2f}, DeLong p={b['p_delong']:.2f})")
        print(f"  LRT significance: {n_raw} at raw p<0.05 (~{0.05*n_tests:.0f} expected by "
              f"chance across {n_tests} tests); {n_fdr} survive BH-FDR q<0.05")
        fusion_df.to_csv(os.path.join(out, f"fig4_combo_search{suffix}.csv"), index=False)

    # ---- assemble ranking + plot (models + strong singles + top-N fusions) ----
    N_TOP_FUSIONS = 15
    top_fusions = fusion_df.head(N_TOP_FUSIONS).to_dict("records") if len(fusion_df) else []
    all_rows = rows + top_fusions
    all_rows.append(dict(Variable="Random", AUC=0.5, AUC_lo=np.nan, AUC_hi=np.nan,
                         direction="", n=len(y_sep), type="random", p_lrt=np.nan, p_delong=np.nan))
    rank = pd.DataFrame(all_rows)
    rank.sort_values("AUC", ascending=False).to_csv(
        os.path.join(out, f"fig4_sepsis_auc_ranking{suffix}.csv"), index=False)

    df = rank[((rank["type"].isin(["model", "model+clinical"])) & (rank["AUC"] > 0.5))
              | ((rank["type"] == "clinical") & (rank["AUC"] > 0.55))
              | (rank["type"] == "random")].copy()
    df = df.sort_values("AUC", ascending=True)
    n = len(df)
    fig, ax = plt.subplots(figsize=(7.0, max(5, n * 0.34 + 1.4)))
    yp = np.arange(n)
    colmap = {"model": style.ACCENT, "model+clinical": style.MUTED_CLASS[0],
              "clinical": style.FG, "random": style.MUTE}
    ax.barh(yp, df["AUC"].values, color=[colmap.get(t, style.FG) for t in df["type"]],
            height=0.7, zorder=2)
    for i, (_, r) in enumerate(df.iterrows()):
        if r["type"] == "random":
            ax.text(r["AUC"] + 0.01, i, "0.50", ha="left", va="center",
                    fontsize=7, fontweight="bold", color=style.ACCENT)
            continue
        if r["type"] == "model+clinical":
            plrt = r.get("p_lrt", np.nan)
            if np.isfinite(plrt):
                pstr = "p<0.01" if plrt < 0.005 else f"p={plrt:.2f}"
                lbl = f"{r['AUC']:.2f} (LRT {pstr})"
            else:
                lbl = f"{r['AUC']:.2f}"
            ax.text(r["AUC"] + 0.01, i, lbl, ha="left", va="center",
                    fontsize=7, fontweight="bold", color=style.FG)
        else:
            lbl = f"{r['direction']} {r['AUC']:.2f}".strip()
            if (r["AUC"] - 0.5) > 0.20:
                ax.text(r["AUC"] - 0.01, i, lbl, ha="right", va="center",
                        fontsize=7, fontweight="bold", color="white")
            else:
                ax.text(r["AUC"] + 0.01, i, lbl, ha="left", va="center",
                        fontsize=7, fontweight="bold", color=style.FG)
    ax.set_yticks(yp); ax.set_yticklabels(df["Variable"].values, fontsize=7)
    ax.set_xlabel("AUC", fontsize=9)
    ax.set_xlim(0.5, 1.32)
    ax.set_title(f"{ci.class_names[ci.sepsis_idx]}+/- Discrimination\n"
                 f"single: raw rank-AUC  |  fusion: P(Sepsis)+<=2 features (LOO-LR), "
                 f"LRT p vs P(Sepsis)",
                 fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    style.trim_spines(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=style.ACCENT, label="Remoscope model prob"),
                       Patch(color=style.MUTED_CLASS[0], label="P(Sepsis) + <=2 features (fusion)"),
                       Patch(color=style.FG, label="Clinical variable")],
              loc="lower right", fontsize=6, frameon=False)
    fig.savefig(os.path.join(out, f"fig4_sepsis_auc_ranking{suffix}.png"), dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig4b_two_variable_search(out, ci, preds, clin, bootstrap_n=1000, seed=0,
                              min_combo_n=10, n_top=25, suffix=""):
    """Exhaustive 2-variable search for sepsis+/- discrimination, allowing EVERY
    feature: the model class-probabilities P(Sepsis)/P(ICU)/P(Healthy) AND all
    clinical variables (including microbio_positive). Every unordered pair is scored
    by leave-one-out logistic regression (out-of-fold AUC) on the patients where
    both are present; results are ranked and the top pairs plotted. Pairs that
    include a spec `discrimination.fusion_exclude` variable (culture positivity)
    are flagged '(leak)' -- it is part of the sepsis definition, so those AUCs are
    circular."""
    import itertools
    style.apply()

    clinical = load_clinical_metadata(clin, ci.sepsis_study)
    patient = build_patient_df(ci, preds, clinical, clin.spec)
    gcol = _group_col(ci)
    if gcol not in patient.columns:
        raise RuntimeError(f"{gcol} missing; cannot run 2-variable search")
    y_sep = patient[gcol].values.astype(float)
    clinical_vars = _clinical_vars(clin.spec)
    fusion_exclude = set(clin.spec["discrimination"]["fusion_exclude"])

    # full feature pool: model probabilities + all clinical variables
    feats = [(f"P({ci.class_names[ci.sepsis_idx]})", ci.prob_col(ci.sepsis_idx))]
    if ci.neg_idx is not None:
        feats.append((f"P({ci.class_names[ci.neg_idx]})", ci.prob_col(ci.neg_idx)))
    if ci.healthy_idx is not None:
        feats.append((f"P({ci.class_names[ci.healthy_idx]})", ci.prob_col(ci.healthy_idx)))
    feats += [(lab, col) for lab, col in clinical_vars.items() if col in patient.columns]
    model_cols = {ci.prob_col(i) for i in range(len(ci.class_names))}

    rows = []
    for (la, ca), (lb, cb) in itertools.combinations(feats, 2):
        cols = [ca, cb]
        m = patient[cols + [gcol]].notna().all(axis=1).values
        if m.sum() < min_combo_n or np.unique(y_sep[m]).size < 2:
            continue
        X = patient[cols].values[m].astype(float)
        if any(np.unique(X[:, j]).size < 2 for j in range(2)):
            continue
        try:
            auc = float(roc_auc_score(y_sep[m], _loo_proba(X, y_sep[m])))
        except Exception:
            continue
        n_model = int(ca in model_cols) + int(cb in model_cols)
        kind = ("model+model" if n_model == 2 else
                "model+clinical" if n_model == 1 else "clinical+clinical")
        leak = bool(fusion_exclude & {ca, cb})
        rows.append(dict(pair=f"{la} + {lb}", cols=f"{ca};{cb}", AUC=auc,
                         n=int(m.sum()), kind=kind, leak=leak))
    search = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
    search.to_csv(os.path.join(out, f"fig4b_two_variable_search{suffix}.csv"), index=False)
    print(f"  2-variable search: {len(feats)} features -> {len(search)} pairs; "
          f"top = {search['pair'].iloc[0]} AUC={search['AUC'].iloc[0]:.2f} "
          f"(top non-leak = {search[~search['leak']]['pair'].iloc[0]} "
          f"{search[~search['leak']]['AUC'].iloc[0]:.2f})")

    # plot the top-N pairs (highest AUC), coloured by pair type; leak-flagged.
    top = search.head(n_top).iloc[::-1]        # reverse so highest is at the top of barh
    n = len(top)
    colmap = {"model+model": style.ACCENT, "model+clinical": style.MUTED_CLASS[0],
              "clinical+clinical": style.FG}
    fig, ax = plt.subplots(figsize=(7.2, max(5, n * 0.34 + 1.4)))
    yp = np.arange(n)
    ax.barh(yp, top["AUC"].values, color=[colmap[k] for k in top["kind"]],
            height=0.7, zorder=2)
    for i, (_, r) in enumerate(top.iterrows()):
        inside = (r["AUC"] - 0.5) > 0.15
        ax.text(r["AUC"] - 0.01 if inside else r["AUC"] + 0.01, i, f"{r['AUC']:.2f}",
                ha="right" if inside else "left", va="center", fontsize=7,
                fontweight="bold", color="white" if inside else style.FG)
    labels = [(r["pair"] + "  (leak)") if r["leak"] else r["pair"] for _, r in top.iterrows()]
    ax.set_yticks(yp); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("AUC (leave-one-out logistic regression)", fontsize=9)
    ax.set_xlim(0.5, 1.05)
    ax.set_title(f"{ci.class_names[ci.sepsis_idx]}+/- Discrimination\n"
                 f"exhaustive 2-variable search, all features (top {n} of {len(search)})",
                 fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    style.trim_spines(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=style.ACCENT, label="model + model"),
                       Patch(color=style.MUTED_CLASS[0], label="model + clinical"),
                       Patch(color=style.FG, label="clinical + clinical")],
              loc="lower right", fontsize=6, frameon=False)
    fig.savefig(os.path.join(out, f"fig4b_two_variable_search{suffix}.png"), dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ===========================================================================
# Demographic tables (clinical table + spec)
# ===========================================================================
def _classify_pathogen(text, spec):
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.lower()
    classes = spec["pathogen_classes"]
    if any(k in t for k in classes["viral"]):
        return "viral"
    if any(k in t for k in classes["bacterial"]):
        return "bacterial"
    return None


def _count_cells(path):
    rp = glob.glob(os.path.join(str(path), "*regionprops.parquet"))
    if not rp:
        return None
    try:
        return len(pd.read_parquet(rp[0], columns=["label"]))
    except Exception:
        return None


def _per_experiment_frame(ci, preds):
    """One row per pooled experiment with true class, study, sample_id."""
    df = pd.DataFrame({
        "experiment_id": preds["eids"].astype(str),
        "true_idx": preds["y_true"].astype(int),
        "study_id": preds["study_ids"],
        "sample_id": preds["sample_ids"].astype(str) if preds["sample_ids"] is not None else "",
    })
    df["class_name"] = df["true_idx"].map(lambda i: ci.class_names[i])
    return df


def table1_demographics(out, ci, preds, clin):
    # Demographics + SUPPORT. Per-class experiment membership = the held-out
    # prediction pool (the model's actual support -- this already respects every
    # class filter, e.g. healthy=flowcell_type==Thin, and all exclusions). The
    # clinical table supplies the attributes (images, cells, mortality, culture
    # positivity, pathogen, age), joined onto those experiments. A Total row is
    # appended.
    t1 = clin.spec["table1"]
    mort, micro = t1["mortality"], t1["microbiology"]
    pcol, acol = t1["pathogen_column"], t1["age_column"]
    sep = load_clinical_metadata(clin, ci.sepsis_study)
    sep["experiment_id"] = sep["experiment_id"].astype(str)
    if "sample_id" in sep.columns:
        sep["sample_id"] = sep["sample_id"].astype(str)

    _warn_missing_columns(sep, [mort["column"], micro["column"], pcol,
                                t1["sex_columns"][0], acol], "table1_demographics")
    sep_patient = sep.drop_duplicates("sample_id").set_index("sample_id")

    healthy_df = None
    if ci.healthy_study:
        try:
            h = clin.study(ci.healthy_study)
            h["experiment_id"] = h["experiment_id"].astype(str)
            if "sample_id" in h.columns:
                h["sample_id"] = h["sample_id"].astype(str)
            healthy_df = h
        except Exception as exc:  # noqa: BLE001
            print(f"  healthy study load failed: {exc}")

    # held-out experiment ids per class (= model support), then restrict the study
    # rows to those ids so attributes join correctly and counts match the model.
    held = _per_experiment_frame(ci, preds)
    held["experiment_id"] = held["experiment_id"].astype(str)
    ids_by_class = {idx: set(held[held["true_idx"] == idx]["experiment_id"])
                    for idx in ci.display_order}

    # (label, experiment-level frame, is_sepsis_study) in display order
    groups = []
    for idx in ci.display_order:
        name = ci.class_names[idx]
        ids = ids_by_class.get(idx, set())
        if idx == ci.healthy_idx and healthy_df is not None:
            groups.append((name, healthy_df[healthy_df["experiment_id"].isin(ids)], False))
        else:
            groups.append((name, sep[sep["experiment_id"].isin(ids)], True))

    def _group_stats(g, is_sep):
        sids = {str(s) for s in g.get("sample_id", pd.Series(dtype=str))} - {"", "nan"}
        n_img = int(pd.to_numeric(g["num_images"], errors="coerce").fillna(0).sum())
        cells, ok = 0, False
        for p in g["path"].dropna().unique():
            c = _count_cells(p)
            if c is not None:
                cells += c; ok = True
        # per-patient 0/1 arrays (for group p-values) + counts (for display)
        mort_a, micro_a, bact_a, viral_a, fem_a, ages = [], [], [], [], [], []
        if is_sep:
            for pid in sids:
                if pid in sep_patient.index:
                    r = sep_patient.loc[pid]
                    mort_a.append(int(_is_true(r.get(mort["column"]))))
                    micro_a.append(int(_is_true(r.get(micro["column"]))))
                    cat = _classify_pathogen(r.get(pcol), clin.spec)
                    bact_a.append(int(cat == "bacterial")); viral_a.append(int(cat == "viral"))
                    fem_a.append(int(_is_female(r.get(t1["sex_columns"][0]))))
                    a = pd.to_numeric(r.get(acol), errors="coerce")
                    if pd.notna(a):
                        ages.append(a)
        else:
            uniq = g.drop_duplicates("sample_id") if "sample_id" in g.columns else g
            if acol in uniq.columns:
                a = pd.to_numeric(uniq[acol], errors="coerce")
                ages = [v for v in a.tolist() if pd.notna(v)]
            sex_col = next((c for c in t1["sex_columns"] if c in uniq.columns), None)
            if sex_col:
                fem_a = [int(_is_female(v)) for v in uniq[sex_col].tolist()]
        return dict(sids=sids, n_exp=len(g), n_img=n_img, cells=cells, ok_cells=ok,
                    mort=sum(mort_a), micro=sum(micro_a), bact=sum(bact_a),
                    viral=sum(viral_a), fem=sum(fem_a),
                    mort_a=mort_a, micro_a=micro_a, bact_a=bact_a,
                    viral_a=viral_a, fem_a=fem_a, ages=ages, is_sep=is_sep)

    stats = [(lbl, _group_stats(gdf, is_sep)) for lbl, gdf, is_sep in groups if len(gdf)]

    def _fmt_age(ages):
        return f"{int(np.nanmin(ages))}–{int(np.nanmax(ages))}" if len(ages) else "—"

    def _n_of(n, denom):
        return f"{n} ({100.0 * n / denom:.1f}%)" if denom else "—"

    rows = []
    for lbl, s in stats:
        n_pat = len(s["sids"])
        rows.append({
            "Class": lbl, "Patients": n_pat,
            "Female": _n_of(s["fem"], len(s["fem_a"])) if s["fem_a"] else "—",
            mort["label"]: _n_of(s["mort"], n_pat) if s["is_sep"] else "—",
            micro["label"]: s["micro"] if s["is_sep"] else "—",
            "Bacterial": s["bact"] if s["is_sep"] else "—",
            "Viral": s["viral"] if s["is_sep"] else "—",
            "Age Range": _fmt_age(s["ages"]),
            "Experiments": s["n_exp"], "Images": f"{s['n_img']:,}",
            "Cells": f"{s['cells']:,}" if s["ok_cells"] else "n/a",
        })

    # Total: unique patients across groups (union), summed exp/img/cells, summed
    # sepsis-study clinical counts, age range over all ages.
    all_sids = set().union(*[s["sids"] for _, s in stats]) if stats else set()
    any_cells = any(s["ok_cells"] for _, s in stats)
    tot_ages = [a for _, s in stats for a in s["ages"]]
    tot_fem_n = sum(len(s["fem_a"]) for _, s in stats)
    rows.append({
        "Class": "Total", "Patients": len(all_sids),
        "Female": _n_of(sum(s["fem"] for _, s in stats), tot_fem_n),
        mort["label"]: sum(s["mort"] for _, s in stats),
        micro["label"]: sum(s["micro"] for _, s in stats),
        "Bacterial": sum(s["bact"] for _, s in stats),
        "Viral": sum(s["viral"] for _, s in stats),
        "Age Range": _fmt_age(tot_ages),
        "Experiments": sum(s["n_exp"] for _, s in stats),
        "Images": f"{sum(s['n_img'] for _, s in stats):,}",
        "Cells": f"{sum(s['cells'] for _, s in stats):,}" if any_cells else "n/a",
    })

    # p-value row: ICU-control vs sepsis-positive per clinical variable.
    smap = dict(stats)
    s_sep = smap.get(ci.class_names[ci.sepsis_idx])
    s_icu = smap.get(ci.class_names[ci.neg_idx]) if ci.neg_idx is not None else None
    if s_sep is not None and s_icu is not None:
        rows.append({
            "Class": t1["p_label"], "Patients": "",
            "Female": _p_cell(_group_pvalue(s_sep["fem_a"], s_icu["fem_a"], "binary")),
            mort["label"]: _p_cell(_group_pvalue(s_sep["mort_a"], s_icu["mort_a"], "binary")),
            micro["label"]: _p_cell(_group_pvalue(s_sep["micro_a"], s_icu["micro_a"], "binary")),
            "Bacterial": _p_cell(_group_pvalue(s_sep["bact_a"], s_icu["bact_a"], "binary")),
            "Viral": _p_cell(_group_pvalue(s_sep["viral_a"], s_icu["viral_a"], "binary")),
            "Age Range": _p_cell(_group_pvalue(s_sep["ages"], s_icu["ages"], "num")),
            "Experiments": "", "Images": "", "Cells": "",
        })

    df = pd.DataFrame(rows)
    C.render_table(df, os.path.join(out, "table1_demographics"),
                   title="Dataset demographics and support (cohort from database)")


# ===========================================================================
# Table 3: cohort characteristics (median IQR / n/N per group), from the clinical
# table; the patient groups come from the class config's sepsis label column
# ===========================================================================
def _med_iqr(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return ""
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    return f"{med:.1f} ({q1:.1f}, {q3:.1f})"


def _n_pct(series):
    """n/N (%) where N is the available-case count, so missingness is visible."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return ""
    pos = int((s == 1).sum())
    return f"{pos}/{len(s)} ({100*pos/len(s):.1f})"


def _cat_pct(series, *values):
    """N (%) of rows whose (string) value matches any of `values` (case-insensitive)."""
    s = series.dropna().astype(str).str.strip().str.lower()
    if len(s) == 0:
        return ""
    want = {v.lower() for v in values}
    n = int(s.isin(want).sum())
    return f"{n}/{len(s)} ({100*n/len(s):.1f})"


def _canon_race(value, race):
    """Map a study's race string onto the shared vocabulary (spec `table3.race`).

    Multi-race entries collapse to "Other" (so "Black or African American, White"
    is one Other patient, not double-counted); ethnicity tokens are stripped
    because the table reports Hispanic ethnicity separately -- see
    _hispanic_from_race, which must read them BEFORE this strips them. A missing
    race becomes "Unknown" rather than dropping out of the denominator, so the
    column totals the cohort's N. Values already in the shared vocabulary pass
    through unchanged.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"
    s = str(value).strip().lower()
    if not s or s == "nan":
        return "Unknown"
    for eth in race["ethnicity_tokens"]:
        s = s.replace(eth, "")
    tokens = race["tokens"]
    mapped = {tokens[t] for t in (p.strip() for p in s.split(",")) if t in tokens}
    if not mapped:
        return "Unknown"
    return "Other" if len(mapped) > 1 else mapped.pop()


def _hispanic_from_race(value, race):
    """Hispanic ethnicity for a study that folds it into the race multi-select.

    Such a race field is the combined checkbox list, so "White, Other, Hispanic,
    Latino/a, or Spanish origin" is a Hispanic donor. Absence of a Hispanic token
    is read as not Hispanic (the donor did not tick it); a missing race stays
    Unknown.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"
    s = str(value).strip().lower()
    if not s or s == "nan":
        return "Unknown"
    return (race["hispanic_yes"] if any(t in s for t in race["ethnicity_tokens"])
            else race["hispanic_no"])


def _count_level(df, col, value):
    """How many rows of `df[col]` equal `value` -- same case-insensitive, stripped
    comparison _cat_pct displays, so counts and ordering can never disagree."""
    if col not in df.columns:
        return 0
    s = df[col].dropna().astype(str).str.strip().str.lower()
    return int((s == str(value).strip().lower()).sum())


def _pathogen_identified(series, neg):
    s = series.dropna().astype(str).str.strip().str.lower()
    if len(s) == 0:
        return ""
    n = int((~s.isin(neg)).sum())
    return f"{n} ({100*n/len(s):.1f})"


def _pretty_label(col, auto):
    """Row label from a column name (spec `auto_labels`: unit suffixes, acronyms)."""
    unit = ""
    for suf, u in auto["unit_suffixes"]:
        if col.endswith(suf):
            col, unit = col[: -len(suf)], u
            break
    acr = {str(k): str(v) for k, v in auto["acronyms"].items()}
    words = [acr.get(w.lower(), w.capitalize()) for w in col.replace("_", " ").split()]
    label = " ".join(words)
    return f"{label} ({unit})" if unit else label


def _expand_organisms(text, organisms):
    """Replace abbreviated organism tokens with full names (spec `organisms`)."""
    import re
    out = str(text)
    for abbr, full in sorted(organisms.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(r"\b" + re.escape(abbr) + r"\b", full, out)
    return out


def _spec_rows(rows):
    """Spec rows ({column, label, mode} mappings) as (column, label, mode) tuples."""
    return [(r["column"], r.get("label"), r["mode"]) for r in rows]


def _table3_frame(ci, clin, spec_rows):
    t3 = clin.spec["table3"]
    gl, race = t3["groups"], t3["race"]
    group_col = _group_col(ci)
    neg_tokens = set(clin.spec["negative_tokens"])

    sep = load_clinical_metadata(clin, ci.sepsis_study)
    sep_pat = sep.drop_duplicates("sample_id").copy()
    if race["column"] in sep_pat.columns:
        sep_pat[race["column"]] = sep_pat[race["column"]].map(lambda v: _canon_race(v, race))
    sp = pd.to_numeric(sep_pat.get(group_col), errors="coerce")
    groups = {
        gl["positive"]: sep_pat[sp == 1],
        gl["negative"]: sep_pat[sp == 0],
    }
    # experiment-level (pre-dedup) counts for the support row
    sp_exp = pd.to_numeric(sep.get(group_col), errors="coerce")
    exp_counts = {gl["positive"]: int((sp_exp == 1).sum()),
                  gl["negative"]: int((sp_exp == 0).sum())}

    healthy = None
    n_exp_healthy = 0
    if ci.healthy_study:
        try:
            h = clin.study(ci.healthy_study)
            # Apply the healthy class filter from the config (e.g.
            # flowcell_type == "Thin") so the demographics describe the cohort
            # the model actually uses, not every donor in the study.
            try:
                hdef = ci.class_definitions[ci.healthy_idx]
                hcol, hval = hdef.get("column"), hdef.get("value")
                if hcol and hval is not None and hcol in h.columns:
                    mask = h[hcol].astype(str).str.strip().str.lower() == str(hval).strip().lower()
                    h = h[mask]
            except Exception:  # noqa: BLE001
                pass
            n_exp_healthy = len(h)  # experiments (before patient dedup)
            if "sample_id" in h.columns:
                h = h.drop_duplicates("sample_id")
            h = h.copy()
            rc, ec = race["column"], race["ethnicity_column"]
            if rc in h.columns:
                # Ethnicity first: the race string carries the Hispanic token that
                # _canon_race strips. Never overwrite a real ethnicity column.
                if ec not in h.columns:
                    h[ec] = h[rc].map(lambda v: _hispanic_from_race(v, race))
                h[rc] = h[rc].map(lambda v: _canon_race(v, race))
            healthy = h
        except Exception:
            healthy = None

    group_items = list(groups.items())
    if healthy is not None:
        group_items.append((gl["healthy"], healthy))
        exp_counts[gl["healthy"]] = n_exp_healthy
    cols = (["Characteristic"] + [f"{g} (N={len(d)})" for g, d in group_items]
            + [t3["p_label"]])
    sepsis_df, icu_df = groups[gl["positive"]], groups[gl["negative"]]
    ng = len(group_items)
    col_map = t3.get("healthy_column_map", {})

    def _use(df, col):
        return col if col in df.columns else col_map.get(col)

    def _num_cells(col):
        out = []
        for _, d in group_items:
            u = _use(d, col)
            out.append(_med_iqr(d[u]) if u is not None and u in d.columns else "")
        return out

    def _level_cells(col, lvl):
        return [_cat_pct(d[col], lvl) if col in d.columns else "" for _, d in group_items]

    def _bin_cells(col):
        return [_n_pct(d[col]) if col in d.columns else "" for _, d in group_items]

    def _pnum(col):
        return (_group_pvalue(sepsis_df[col], icu_df[col], "num")
                if col in sepsis_df.columns and col in icu_df.columns else np.nan)

    def _pbin(col):
        return (_group_pvalue(sepsis_df[col], icu_df[col], "binary")
                if col in sepsis_df.columns and col in icu_df.columns else np.nan)

    def _plevel(col, lvl):
        if col not in sepsis_df.columns or col not in icu_df.columns:
            return np.nan
        lv = str(lvl).strip().lower()
        s = (sepsis_df[col].astype(str).str.strip().str.lower() == lv).astype(float)
        i = (icu_df[col].astype(str).str.strip().str.lower() == lv).astype(float)
        return _group_pvalue(s, i, "binary")

    rows = [
        ["Patients, N"] + [str(len(d)) for _, d in group_items] + [""],
        ["Number of experiments (support), N"]
        + [str(exp_counts.get(g, "")) for g, _ in group_items] + [""],
    ]
    for col, label, mode in spec_rows:
        if col not in sep_pat.columns:
            continue
        lbl = label or _pretty_label(col, clin.spec["auto_labels"])
        if mode == "num":
            rows.append([lbl + ", median (IQR)"] + _num_cells(col) + [_p_cell(_pnum(col))])
        elif mode == "bin":
            rows.append([lbl + ", n/N (%)"] + _bin_cells(col) + [_p_cell(_pbin(col))])
        elif mode.startswith("level:"):
            lvl = mode.split(":", 1)[1]
            rows.append([lbl + ", n/N (%)"] + _level_cells(col, lvl) + [_p_cell(_plevel(col, lvl))])
        elif mode == "cat":
            rows.append([lbl + ", n/N (%)"] + [""] * ng + [""])
            # Levels from the union over the cohorts shown, not from the patient
            # rows alone: a level only the healthy donors have would otherwise
            # never get a row (and its patients would vanish from the column), and
            # levels unique to indeterminate patients would add all-zero rows.
            lvls = set()
            for _, d in group_items:
                if col in d.columns:
                    lvls |= set(d[col].dropna().astype(str).str.strip().unique())
            for lvl in sorted(lvls):
                rows.append(["    " + lvl] + _level_cells(col, lvl) + [_p_cell(_plevel(col, lvl))])
        elif mode == "pathogen":
            cells = [_pathogen_identified(d[col], neg_tokens) if col in d.columns else ""
                     for _, d in group_items]
            rows.append([lbl + ", n/N (%)"] + cells + [""])
            vals = sorted(v for v in sep_pat[col].dropna().astype(str).str.strip().unique()
                          if v.lower() not in neg_tokens)
            for v in vals:
                rows.append(["    " + _expand_organisms(v, clin.spec["organisms"])]
                            + _level_cells(col, v) + [""])
        elif mode == "diagnosis":
            rows.append([lbl + ", n/N (%)"] + [""] * ng + [""])
            vals = [v for v in sep_pat[col].dropna().astype(str).str.strip().unique()
                    if v.lower() not in neg_tokens]
            # Group the breakdown by cohort -- every sepsis diagnosis first, then
            # the ICU-controls -- rather than one alphabetical list that interleaves
            # the two, alphabetical within each block so the order is reproducible.
            # Diagnoses claimed by neither group belong to indeterminate patients
            # (null group label, outside this analysis) and are dropped: they
            # would otherwise add all-zero rows to a cohort table.
            def _dx_rank(v, _col=col):
                for rank, d in ((0, sepsis_df), (1, icu_df)):
                    if _count_level(d, _col, v):
                        return rank
                return 2
            ranked = sorted(((_dx_rank(v), v) for v in vals))
            dropped = [v for r, v in ranked if r == 2]
            if dropped:
                print(f"  {lbl}: dropped {len(dropped)} indeterminate-only "
                      f"diagnoses (in neither cohort)")
            for v in [v for r, v in ranked if r < 2]:
                rows.append(["    " + v[:90]] + _level_cells(col, v) + [""])

    return pd.DataFrame(rows, columns=cols)


def _table3_paper_format(df, spec_rows, manuscript):
    """Manuscript format: 'N (%)' labels and cells without the /N denominator,
    manuscript row set only (spec `table3.manuscript`), and pathogen sub-rows
    restricted to organisms present in the sepsis or ICU-control cohort."""
    import re
    cols = list(df.columns)
    group_cols = cols[1:-1]
    drop_rows, drop_blocks = set(manuscript["drop_rows"]), set(manuscript["drop_blocks"])
    pathogen_labels = {lbl for _, lbl, mode in spec_rows if mode == "pathogen"}
    rows, drop_block, pathogen_block = [], False, False
    for _, r in df.iterrows():
        label = str(r[cols[0]])
        if not label.startswith("    "):
            base = label.split(", n/N")[0].split(", median")[0].split(", N (")[0]
            drop_block = base in drop_blocks
            pathogen_block = base in pathogen_labels
            if drop_block or base in drop_rows:
                continue
        else:
            if drop_block:
                continue
            if pathogen_block:
                def _n(cell):
                    m = re.match(r"\s*(\d+)", str(cell))
                    return int(m.group(1)) if m else 0
                if _n(r[group_cols[0]]) == 0 and _n(r[group_cols[1]]) == 0:
                    continue
        rr = r.copy()
        rr[cols[0]] = label.replace(", n/N (%)", ", N (%)")
        for c in group_cols:
            rr[c] = re.sub(r"^(\d+)/\d+\s+", r"\1 ", str(rr[c]))
        rows.append(rr)
    return pd.DataFrame(rows, columns=cols)


def table3_cohort_characteristics(out, ci, preds, clin):
    """Cohort demographics / labs (median (IQR), N (%)) per group.

    table3_cohort_characteristics is the full table (spec `table3.rows`);
    table3_cohort_characteristics_with_lymphocytes is the manuscript version: the
    spec's `manuscript.insert_rows` added, in manuscript format
    (_table3_paper_format)."""
    t3 = clin.spec["table3"]
    rows = _spec_rows(t3["rows"])
    C.render_table(_table3_frame(ci, clin, rows),
                   os.path.join(out, "table3_cohort_characteristics"),
                   title=t3["title"], fontsize=6)

    man = t3["manuscript"]
    sep_cols = clin.study(ci.sepsis_study).columns
    spec = list(rows)
    for ins in man.get("insert_rows", []):
        if ins["column"] not in sep_cols:
            raise ValueError(f"no {ins['column']} column in study '{ci.sepsis_study}'")
        anchor = [i for i, (c, _, _) in enumerate(spec) if c == ins["after"]]
        spec.insert(anchor[0] + 1 if anchor else len(spec),
                    (ins["column"], ins.get("label"), ins["mode"]))
    raw = _table3_frame(ci, clin, spec).fillna("").astype(str)
    C.render_table(_table3_paper_format(raw, spec, man),
                   os.path.join(out, "table3_cohort_characteristics_with_lymphocytes"),
                   title=t3["title"], fontsize=6)


def table_icu_control_diagnoses(out, ci, preds, clin):
    """Per-patient clinical diagnosis for the ICU-control (sepsis-negative)
    patients -- the non-septic reasons they were admitted (diagnosis notes +
    pathogen). Parallels the sepsis-positive pathogen information."""
    if ci.neg_idx is None:
        print("  ICU-control diagnoses skipped: no ICU-control class")
        return "skip"
    dxs = clin.spec["diagnosis"]
    neg = set(clin.spec["negative_tokens"])
    sep = load_clinical_metadata(clin, ci.sepsis_study)
    sp = pd.to_numeric(sep.get(_group_col(ci)), errors="coerce")
    icu = sep[sp == 0].drop_duplicates("sample_id")
    rows = []
    for _, r in icu.iterrows():
        dx = str(r.get(dxs["notes_column"]) or "").strip()
        pth = str(r.get(dxs["pathogen_column"]) or "").strip()
        rows.append({"Sample ID": str(r.get("sample_id", "")),
                     "Diagnosis": dx if dx.lower() not in neg else "—",
                     "Pathogen": pth if pth.lower() not in neg else "—"})
    rows.sort(key=lambda x: x["Sample ID"])
    df = pd.DataFrame(rows, columns=["Sample ID", "Diagnosis", "Pathogen"])
    C.render_table(df, os.path.join(out, "table_icu_control_diagnoses"),
                   title=f"ICU-control patient diagnoses (sepsis-negative, N={len(df)})",
                   fontsize=6)
    print(f"  ICU-control diagnoses: {len(df)} patients")


# ===========================================================================
# Indeterminate patients: what the LOO ensemble calls the unlabelled cases
# ===========================================================================
def _load_indeterminate_npz(args):
    """Per-fold/per-repeat softmax for the indeterminate experiments, written by
    infer_indeterminate.py --aggregate. Shape (n_folds, n_repeats, n_exp,
    n_classes), class axis index-aligned to the model."""
    path = args.indeterminate_npz
    if not os.path.exists(path):
        return None, path
    return np.load(path, allow_pickle=False), path


def _indet_run_label(z):
    """'epoch 45, 47 folds x 5 repeats, bag 500' straight from the npz."""
    probs = z["probs"]
    bag = int(z["bag_size"]) if "bag_size" in z else None
    return (f"epoch {int(z['epoch'])}, {probs.shape[0]} folds x "
            f"{probs.shape[1]} repeats"
            + (f", bag {bag}" if bag else ""))


def _indet_stats(block):
    """(mean, sd_fold, valid) over a (n_folds, n_draws, n_classes) block.

    sd_fold -- spread of the per-fold mean probability across ensemble members.
    """
    per_fold = np.nanmean(block, axis=1)
    valid = per_fold[~np.isnan(per_fold).any(axis=1)]
    return valid.mean(axis=0), valid.std(axis=0), valid


def _indet_bag_sd(sub):
    """Pure bag-resampling noise: SD across repeats, averaged over everything else.

    ``sub`` is (n_folds, n_repeats, ..., n_classes) -- axis 1 must be the repeat
    axis, so replicate experiments (if present) are averaged over rather than
    folded into the spread.
    """
    with np.errstate(invalid="ignore"):
        per = np.nanstd(sub, axis=1)
        return np.nanmean(per, axis=tuple(range(per.ndim - 1)))


def suptable3_indeterminate(out, ci, args):
    """Indeterminate patients: clinical diagnosis vs the LOO ensemble's call.

    These patients have a recorded clinical diagnosis but no adjudicated
    sepsis_positive label, so they were null-label-excluded from every LOO fold.
    Every fold is therefore an unbiased predictor for them, and the ensemble mean
    is the model's opinion on cases the ground truth could not resolve.

    Probabilities are pooled over folds x repeats (x replicate experiments at the
    patient level); the +- term is the between-fold spread of the per-fold mean.
    """
    z, path = _load_indeterminate_npz(args)
    if z is None:
        print(f"  indeterminate table skipped: {path} missing "
              f"(run infer_indeterminate.py)")
        return "skip"

    probs = z["probs"]
    exp_ids = [str(e) for e in z["experiment_ids"]]
    sample_ids = [str(s) for s in z["sample_ids"]]
    n_folds, n_repeats = probs.shape[0], probs.shape[1]
    if probs.shape[3] != len(ci.class_names):
        print(f"  indeterminate table skipped: npz has {probs.shape[3]} classes, "
              f"model has {len(ci.class_names)}")
        return "skip"

    dxs = args.clin.spec["diagnosis"]
    sep = load_clinical_metadata(args.clin, ci.sepsis_study)
    dx = sep.drop_duplicates("sample_id").set_index("sample_id")
    neg = set(args.clin.spec["negative_tokens"])

    def _clean(sid, col):
        v = str(dx[col].get(sid, "") or "").strip() if col in dx.columns else ""
        return v if v.lower() not in neg else "—"

    rows = []
    for sid in sorted(set(sample_ids)):
        cols = [i for i, s in enumerate(sample_ids) if s == sid]
        block = probs[:, :, cols, :].reshape(n_folds, -1, probs.shape[3])
        mean_p, sd_fold, valid = _indet_stats(block)
        sd_repeat = _indet_bag_sd(probs[:, :, cols, :])
        pred = int(mean_p.argmax())
        votes = np.bincount(valid.argmax(axis=1),
                            minlength=len(ci.class_names)) / len(valid)
        # Between-replicate spread: sd of each experiment's pooled mean.
        per_exp = np.stack([np.nanmean(probs[:, :, i, :].reshape(-1, probs.shape[3]),
                                       axis=0) for i in cols])
        row = {"Sample ID": sid,
               "Clinical Diagnosis": _clean(sid, dxs["notes_column"]),
               "Pathogen": _clean(sid, dxs["pathogen_column"]),
               "Exps": len(cols)}
        for k in ci.display_order:
            row[f"P({ci.class_names[k]})"] = f"{mean_p[k]:.2f} ± {sd_fold[k]:.2f}"
        row["Predicted Class"] = ci.class_names[pred]
        row["Fold Agreement"] = f"{100 * votes[pred]:.0f}%"
        row["Bag SD"] = f"{sd_repeat[pred]:.3f}"
        row["Replicate SD"] = (f"{per_exp[:, pred].std(ddof=0):.3f}"
                               if len(cols) > 1 else "—")
        rows.append(row)

    df = pd.DataFrame(rows)
    # Wrap the free-text diagnosis for the figure only; the CSV keeps it on one
    # line. Unwrapped, a 250-character diagnosis stretches the PNG ~9000px wide.
    render_df = df.copy()
    render_df["Clinical Diagnosis"] = [
        "\n".join(textwrap.wrap(str(v), 46)) or "—" for v in df["Clinical Diagnosis"]
    ]
    C.render_table(
        render_df, os.path.join(out, "suptable3_indeterminate"),
        title=(f"Indeterminate patients (no adjudicated label, N={len(df)}): "
               f"clinical diagnosis vs LOO ensemble prediction\n"
               f"{_indet_run_label(z)}; ± = between-fold SD"),
        fontsize=6, csv_df=df)
    print(f"  indeterminate: {len(df)} patients, {len(exp_ids)} experiments "
          f"({n_folds} folds x {n_repeats} repeats) from {os.path.basename(path)}")


def suptable3b_indeterminate_experiments(out, ci, args):
    """Per-experiment breakdown behind suptable3 -- shows replicate agreement."""
    z, path = _load_indeterminate_npz(args)
    if z is None:
        print(f"  indeterminate per-experiment table skipped: {path} missing")
        return "skip"

    probs = z["probs"]
    exp_ids = [str(e) for e in z["experiment_ids"]]
    sample_ids = [str(s) for s in z["sample_ids"]]
    if probs.shape[3] != len(ci.class_names):
        return "skip"

    rows = []
    for ei, e in enumerate(exp_ids):
        mean_p, sd_fold, valid = _indet_stats(probs[:, :, ei, :])
        sd_repeat = _indet_bag_sd(probs[:, :, ei, :])
        pred = int(mean_p.argmax())
        votes = np.bincount(valid.argmax(axis=1),
                            minlength=len(ci.class_names)) / len(valid)
        row = {"Experiment ID": e, "Sample ID": sample_ids[ei],
               "Folds": len(valid)}
        for k in ci.display_order:
            row[f"P({ci.class_names[k]})"] = f"{mean_p[k]:.2f} ± {sd_fold[k]:.2f}"
        row["Predicted Class"] = ci.class_names[pred]
        row["Fold Agreement"] = f"{100 * votes[pred]:.0f}%"
        row["Bag SD"] = f"{sd_repeat[pred]:.3f}"
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["Sample ID", "Experiment ID"])
    C.render_table(
        df, os.path.join(out, "suptable3b_indeterminate_experiments"),
        title=(f"Indeterminate experiments (N={len(df)}): per-experiment LOO "
               f"ensemble predictions\n{_indet_run_label(z)}; "
               f"± = between-fold SD"),
        fontsize=6)
    print(f"  indeterminate per-experiment: {len(df)} experiments")


# ===========================================================================
# Table 4: pairwise operating points
# ===========================================================================
def _op_point(y_bin, y_sc, mode, closest=False):
    """Return threshold for an operating point. mode: youden|sens95|maxsens.

    sklearn's roc_curve returns thresholds in DECREASING order with tpr
    increasing along the array, so the FIRST index meeting a sensitivity
    target is the highest threshold (best specificity) that still achieves it.

    closest=True picks the achievable sensitivity NEAREST the target instead of
    the first point clearing it. Useful when the ROC has no step at the target
    (small positive count) and the floor rule overshoots to a much lower
    specificity -- the nearest point may then land just BELOW the target but
    with far fewer negatives misclassified. Has no effect on mode="youden".
    """
    fpr, tpr, thr = roc_curve(y_bin, y_sc)
    if mode == "youden":
        return float(thr[int(np.argmax(tpr - fpr))])
    target = 0.95 if mode == "sens95" else 0.999 if mode == "maxsens" else None
    if target is None:
        raise ValueError(mode)
    if closest:
        return float(thr[int(np.argmin(np.abs(tpr - target)))])
    ok = np.where(tpr >= target)[0]
    return float(thr[ok[0]]) if len(ok) else float(thr[int(np.argmax(tpr - fpr))])


def _pairwise_op_rows(ci, y_true, y_probs, closest_target=False,
                      include_argmax=False):
    """Build the sepsis-vs-ICU and sepsis-vs-all operating-point rows.

    ``include_argmax`` adds the two 3-class decision rules as extra rows so the
    table can be read against the confusion matrices: the Youden-shifted argmax
    used by the Fig 1 confusion matrix, and the model's raw argmax. Those rules score on
    the FULL 3-class probability vector, so they need not coincide with the
    Youden-J point of the renormalized pairwise ROC -- when the ROC has near-tied
    J values they can land on a very different sensitivity/specificity trade.
    """
    from remo_id.analysis.pairwise import filter_pairwise
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs)
    sep_name = ci.class_names[ci.sepsis_idx]
    neg_name = ci.class_names[ci.neg_idx] if ci.neg_idx is not None else None

    argmax_rules = []
    if include_argmax:
        thr3 = C.youden_thresholds(y_true, y_probs, ci.class_names)
        argmax_rules = [
            ("3-class argmax (Youden-shifted; = CM)", (y_probs - thr3).argmax(axis=1)),
            ("3-class argmax (raw)", y_probs.argmax(axis=1)),
        ]

    rows = []
    # (mode, label, closest). closest=True picks the achievable sensitivity
    # nearest the target rather than the first point clearing it. With
    # --closest_target we report BOTH the floor and the closest variant so the
    # discreteness overshoot (e.g. "95% sens" snapping to 100%) is visible.
    op_labels = [("youden", "Youden-J", False),
                 ("sens95", "95% sensitivity", False),
                 ("maxsens", "Max sensitivity", False)]
    if closest_target:
        op_labels += [("sens95", "95% sensitivity (closest)", True),
                      ("maxsens", "Max sensitivity (closest)", True)]

    # --- Sepsis vs ICU-control (pairwise, renormalized) ---
    if neg_name is not None:
        yb, ypb, mask, pos_i, neg_i = filter_pairwise(y_true, y_probs, ci.class_names,
                                                       sep_name, neg_name)
        ysc = ypb[:, 1]
        n_pos = int(yb.sum()); n_neg = int((1 - yb).sum())
        for mode, lab, closest in op_labels:
            t = _op_point(yb, ysc, mode, closest=closest)
            yp = (ysc >= t).astype(int)
            tp = int(((yp == 1) & (yb == 1)).sum()); fp = int(((yp == 1) & (yb == 0)).sum())
            tn = int(((yp == 0) & (yb == 0)).sum()); fn = int(((yp == 0) & (yb == 1)).sum())
            sens = tp / max(1, n_pos)
            spec = tn / max(1, n_neg)
            ppv = tp / max(1, tp + fp)
            npv = tn / max(1, tn + fn)
            rows.append({
                "Classes": f"{sep_name} vs {neg_name}", "Operating Point": lab,
                "Sensitivity": f"{100*sens:.1f}%", "Specificity": f"{100*spec:.1f}%",
                "PPV": f"{100*ppv:.1f}%", "NPV": f"{100*npv:.1f}%",
                f"{sep_name} Missed": f"{fn}/{n_pos}",
                f"{neg_name} Misclassified": f"{fp}/{n_neg}",
                "Healthy Misclassified": "—",
            })
        for lab, pred3 in argmax_rules:
            call = (pred3[mask] == pos_i).astype(int)
            tp = int(((call == 1) & (yb == 1)).sum()); fp = int(((call == 1) & (yb == 0)).sum())
            tn = int(((call == 0) & (yb == 0)).sum()); fn = int(((call == 0) & (yb == 1)).sum())
            rows.append({
                "Classes": f"{sep_name} vs {neg_name}", "Operating Point": lab,
                "Sensitivity": f"{100*tp/max(1,n_pos):.1f}%",
                "Specificity": f"{100*tn/max(1,n_neg):.1f}%",
                "PPV": f"{100*tp/max(1,tp+fp):.1f}%",
                "NPV": f"{100*tn/max(1,tn+fn):.1f}%",
                f"{sep_name} Missed": f"{fn}/{n_pos}",
                f"{neg_name} Misclassified": f"{fp}/{n_neg}",
                "Healthy Misclassified": "—",
            })
            # A 3-class rule can send a sepsis/ICU case to Healthy; that is a
            # non-sepsis call here but a different cell in the 3-class CM.
            n_h = int((pred3[mask] == ci.healthy_idx).sum()) if ci.healthy_idx is not None else 0
            if n_h:
                print(f"  note: {lab} routed {n_h} {sep_name}/{neg_name} case(s) "
                      f"to Healthy (counted as non-{sep_name} in this row)")

    # --- Sepsis vs All (OvR on P(sepsis)) ---
    yb = (y_true == ci.sepsis_idx).astype(int)
    ysc = y_probs[:, ci.sepsis_idx]
    n_pos = int(yb.sum())
    n_neg_icu = int((y_true == ci.neg_idx).sum()) if ci.neg_idx is not None else 0
    n_neg_h = int((y_true == ci.healthy_idx).sum()) if ci.healthy_idx is not None else 0
    for mode, lab, closest in op_labels:
        t = _op_point(yb, ysc, mode, closest=closest)
        yp = (ysc >= t).astype(int)
        tp = int(((yp == 1) & (yb == 1)).sum()); fp = int(((yp == 1) & (yb == 0)).sum())
        tn = int(((yp == 0) & (yb == 0)).sum()); fn = int(((yp == 0) & (yb == 1)).sum())
        sens = tp / max(1, n_pos)
        spec = tn / max(1, int((yb == 0).sum()))
        ppv = tp / max(1, tp + fp)
        npv = tn / max(1, tn + fn)
        icu_mis = int(((yp == 1) & (y_true == ci.neg_idx)).sum()) if ci.neg_idx is not None else 0
        h_mis = int(((yp == 1) & (y_true == ci.healthy_idx)).sum()) if ci.healthy_idx is not None else 0
        rows.append({
            "Classes": f"{sep_name} vs All", "Operating Point": lab,
            "Sensitivity": f"{100*sens:.1f}%", "Specificity": f"{100*spec:.1f}%",
            "PPV": f"{100*ppv:.1f}%", "NPV": f"{100*npv:.1f}%",
            f"{sep_name} Missed": f"{fn}/{n_pos}",
            f"{neg_name} Misclassified" if neg_name else "Neg Misclassified": f"{icu_mis}/{n_neg_icu}",
            "Healthy Misclassified": f"{h_mis}/{n_neg_h}",
        })
    for lab, pred3 in argmax_rules:
        call = (pred3 == ci.sepsis_idx).astype(int)
        tp = int(((call == 1) & (yb == 1)).sum()); fp = int(((call == 1) & (yb == 0)).sum())
        tn = int(((call == 0) & (yb == 0)).sum()); fn = int(((call == 0) & (yb == 1)).sum())
        icu_mis = int(((call == 1) & (y_true == ci.neg_idx)).sum()) if ci.neg_idx is not None else 0
        h_mis = int(((call == 1) & (y_true == ci.healthy_idx)).sum()) if ci.healthy_idx is not None else 0
        rows.append({
            "Classes": f"{sep_name} vs All", "Operating Point": lab,
            "Sensitivity": f"{100*tp/max(1,n_pos):.1f}%",
            "Specificity": f"{100*tn/max(1,int((yb==0).sum())):.1f}%",
            "PPV": f"{100*tp/max(1,tp+fp):.1f}%",
            "NPV": f"{100*tn/max(1,tn+fn):.1f}%",
            f"{sep_name} Missed": f"{fn}/{n_pos}",
            f"{neg_name} Misclassified" if neg_name else "Neg Misclassified": f"{icu_mis}/{n_neg_icu}",
            "Healthy Misclassified": f"{h_mis}/{n_neg_h}",
        })

    return pd.DataFrame(rows)


def table4_pairwise(out, ci, preds, closest_target=False):
    df = _pairwise_op_rows(ci, preds["y_true"], preds["y_probs"],
                           closest_target=closest_target)
    C.render_table(df, os.path.join(out, "table4_pairwise_operating_points"),
                   title="Pairwise operating points")


def table4b_pairwise_sample_average(out, ci, preds, args):
    """table4 on the sample-averaged (patient-level) predictions, plus the two
    3-class argmax rules so the numbers line up with the Fig 1 (patient-level)
    confusion matrix.

    Identical by construction to table4 in the patient_level_combined dir; this
    variant puts the patient-level operating points in the experiment-level dir.
    """
    yt_avg, yp_avg, n_pat = _sample_average_preds(preds)
    df = _pairwise_op_rows(ci, yt_avg, yp_avg, closest_target=True,
                           include_argmax=True)
    C.render_table(df, os.path.join(out, "table4b_pairwise_sample_average"),
                   title=f"Pairwise operating points (sample-averaged, n={n_pat} patients)")
    print(f"  table4b pairwise (sample-averaged, n={n_pat}) written")


# ===========================================================================
# Supplementary Fig 9: per-experiment sampling variability vs bag size
#   Each held-out experiment is scored by its own fold's epoch-N aggregation head
#   on bags sampled from its cached per-image embeddings (loo-per-image), at every
#   bag size x seed. P(true class) is the bag-level softmax prob assigned to the
#   experiment's TRUE class; "shuffles" = the seeds swept per bag size.
# ===========================================================================
def _fold_for_experiments(cv_dir):
    """Map each held-out experiment_id -> its fold directory (via cv_splits.yaml)."""
    with open(os.path.join(cv_dir, "cv_splits.yaml")) as f:
        splits = yaml.safe_load(f)
    mapping = {}
    for r_idx, folds in enumerate(splits["repeats"]):
        for fold in folds:
            fdir = os.path.join(cv_dir, f"repeat{r_idx}_fold{fold['fold']}")
            for e in fold.get("val_experiment_ids", []):
                mapping[str(e)] = fdir
    return mapping


def _iter_fold_models(cv_dir, preds, device, epoch):
    """Yield (fdir, eids, model) for each fold that holds out experiments present
    in the pooled predictions, loading every fold's epoch-N checkpoint once."""
    from remo_id.inference.core import load_checkpoint, auto_detect_from_checkpoint
    exp_to_fold = _fold_for_experiments(cv_dir)
    present = {str(e) for e in preds["eids"]}
    by_fold = {}
    for eid, fdir in exp_to_fold.items():
        if eid in present:
            by_fold.setdefault(fdir, []).append(eid)
    info = None
    for fdir, eids in by_fold.items():
        ckpt = C.fold_checkpoint(fdir, epoch)
        # architecture / task are identical across LOO folds: detect once
        info = info or auto_detect_from_checkpoint(ckpt)
        model, _, _ = load_checkpoint(
            ckpt, model_name=info["model_name"], task_type=info["task_type"],
            num_classes=info["num_classes"], device=str(device))
        yield fdir, eids, model
        del model


def _bag_probs_from_feats(model, pool_feats, bag, rng):
    """Sample `bag` feature rows (with replacement only if the pool is smaller)
    and aggregate them into bag-level softmax probs."""
    import torch
    npool = pool_feats.shape[0]
    size = min(bag, npool)
    idx = rng.choice(npool, size=size, replace=size < bag)
    feats = pool_feats[torch.as_tensor(idx, device=pool_feats.device, dtype=torch.long)].unsqueeze(0)
    with torch.no_grad():
        logits, _ = model._aggregate(feats)
        return torch.softmax(logits, dim=-1).cpu().numpy()[0]


def bagvar_per_experiment(out, cv_dir, args, ci, preds):
    """Long-form bag-level class probabilities, one row per (experiment,
    bag_size, seed), from the epoch-N cached per-image embeddings (no re-encode).
    Writes supfig9_bagvar_per_experiment.csv and returns the frame. GPU."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bag_sizes = [int(x) for x in str(args.bag_sizes).split(",")]
    seeds = [int(x) for x in str(args.bag_seeds).split(",")]
    true_idx = {str(e): int(t) for e, t in zip(preds["eids"], preds["y_true"])}
    emb_dir = args.embeddings_dir

    # Every held-out experiment must have embeddings cached at THIS epoch.
    present_eids = [str(e) for e in preds["eids"]]
    missing, wrong_epoch = [], []
    for eid in present_eids:
        emb_path = os.path.join(emb_dir, f"{eid}.embeddings.npz")
        if not os.path.exists(emb_path):
            missing.append(eid)
            continue
        emb_epoch = int(np.load(emb_path, allow_pickle=True)["epoch"])
        if emb_epoch != int(args.epoch):
            wrong_epoch.append((eid, emb_epoch))
    if missing or wrong_epoch:
        raise FileNotFoundError(
            f"epoch={args.epoch} embeddings incomplete in {emb_dir}: "
            f"{len(missing)} missing, {len(wrong_epoch)} at a different epoch "
            f"(e.g. {wrong_epoch[:3]}). Run loo-per-image at --epoch {args.epoch}.")

    store = {(b, s): {} for b in bag_sizes for s in seeds}
    for fdir, eids, model in _iter_fold_models(cv_dir, preds, device, args.epoch):
        for eid in eids:
            d = np.load(os.path.join(emb_dir, f"{eid}.embeddings.npz"), allow_pickle=True)
            pool_feats = torch.from_numpy(d["embeddings"].astype(np.float32)).to(device)
            for b in bag_sizes:
                for s in seeds:
                    rng = np.random.RandomState(s + b)
                    store[(b, s)][eid] = _bag_probs_from_feats(model, pool_feats, b, rng)
            del pool_feats
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"  scored {len(present_eids)} experiments from epoch={args.epoch} embeddings")

    sample_of = dict(zip(preds["eids"].astype(str), preds["sample_ids"].astype(str)))
    study_of = dict(zip(preds["eids"].astype(str), preds["study_ids"].astype(str)))
    prob_cols = [f"prob_{n}" for n in ci.class_names]
    rows = []
    for (b, s), d in store.items():
        for eid, pv in d.items():
            ti = true_idx[eid]
            row = {"experiment_id": eid, "sample_id": sample_of.get(eid, ""),
                   "study_id": study_of.get(eid, ""),
                   "true_class": ci.class_names[ti], "true_class_idx": ti,
                   "bag_size": b, "seed": s,
                   "pred_class": ci.class_names[int(np.argmax(pv))],
                   "correct": int(np.argmax(pv) == ti)}
            for j, c in enumerate(prob_cols):
                row[c] = float(pv[j])
            rows.append(row)
    per_exp = pd.DataFrame(rows).sort_values(["experiment_id", "bag_size", "seed"])
    per_exp.to_csv(os.path.join(out, "supfig9_bagvar_per_experiment.csv"), index=False)
    return per_exp


def _bagvar_ptrue(per_exp, ci):
    """Return per_exp with a `p_true` column = P(assigned to the true class)."""
    df = per_exp.copy()
    prob_lookup = {n: f"prob_{n}" for n in ci.class_names}
    df["p_true"] = [row[prob_lookup[row["true_class"]]]
                    for _, row in df.iterrows()]
    return df


def supfig9_bagvar_ribbons(out, ci, per_exp):
    """Within-experiment SD of P(true class) vs bag size.

    Thin lines = individual experiments (colored by true class); bold lines =
    per-class mean SD across experiments. Quantifies how sampling (shuffle)
    variability shrinks as the bag grows."""
    style.apply()
    df = _bagvar_ptrue(per_exp, ci)
    n_shuffles = df["seed"].nunique()
    color_of = dict(zip(ci.class_names, style.class_colors_muted(ci.class_names)))

    # SD of p_true across seeds, per (experiment, bag_size)
    sd = (df.groupby(["experiment_id", "true_class", "bag_size"])["p_true"]
            .std(ddof=0).reset_index(name="sd"))
    bag_sizes = sorted(sd["bag_size"].unique())

    fig, ax = style.figure(width="single", aspect=0.72)
    # thin per-experiment lines
    for eid, g in sd.groupby("experiment_id"):
        g = g.sort_values("bag_size")
        cls = g["true_class"].iloc[0]
        ax.plot(g["bag_size"], g["sd"], color=color_of.get(cls, style.MUTE),
                lw=0.35, alpha=0.18, zorder=1)
    # bold class-mean lines (display order)
    for idx in ci.display_order:
        cls = ci.class_names[idx]
        sub = sd[sd["true_class"] == cls]
        if sub.empty:
            continue
        m = sub.groupby("bag_size")["sd"].mean().reindex(bag_sizes)
        ax.plot(m.index, m.values, "-o", color=color_of.get(cls, style.FG),
                lw=1.4, ms=3, zorder=3, label=cls)

    ax.set_xscale("log")
    ax.set_xlabel("Bag size (images)")
    ax.set_ylabel("Within-experiment SD of P(true class)")
    ax.set_title("Per-Experiment Sampling Variability")
    ax.set_ylim(bottom=0)
    ax.legend(title="class mean (bold)", loc="upper right")
    style.trim_spines(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "supfig9_bagvar_ribbons.png"))
    plt.close(fig)


# ===========================================================================
# Fig 2: informative cells (occlusion screen) + per-class Grad-CAM frames
# ===========================================================================
# Inputs: the `remo-id analysis occlusion` work dir of the held-out screen
# (--occlusion_dir: experiments/, screen/stage1/, report/screen_cells.parquet; run per
# LOO fold by occlusion_fold.py) and --cohort (scope). Everything shown is chosen by rule
# (remo_id.interpret.cell_occlusion.select_informative_cells / select_example_frames):
#   a  per class (rows Sepsis / ICU control / Healthy) the 10 cells whose single removal
#      lowers the held-out model's true-class log-odds most beyond the sham (remove -
#      sham), beating the equal-area null, not at the border, frame correct; one cell per
#      experiment; the top FIG2_TYPE_QUOTA white and red cells (free slots: next best of
#      any type); white cells first. Healthy cells only from FIG2_HEALTHY_SCOPE, the scope
#      shared with the patients.
#   b, c  one ICU-control and one Sepsis frame: correct, 0.6 < P(true) < 0.99, cells cover
#      <= 55% of the frame, in focus (Laplacian variance in the top quartile of the
#      experiment's screened frames), ranked by total |stage-1 effect|, one candidate per
#      experiment, rank 1; tinted by per-class Grad-CAM of the held-out model.
# Ties are broken by a seeded random key (--seed). Picks, crops and Grad-CAMs are cached
# in --fig2_cache, so a re-render reads the cache only.
FIG2_HEALTHY_SCOPE = "lfm-ohmu"
FIG2_TYPE_QUOTA = {"wbc": 5, "rbc": 5}
FIG2_N = 10                 # crops per class row
FIG2_WINDOW = 128           # crop px: every WBC (p99 ~87 px, largest ~97 px) whole + one ring of RBCs
FIG2_TITLE_A = "Informative cells per class"
FIG2_LABEL_W_MM = 20.0
FIG2_COLNUM_H = 3.4         # mm reserved above the rows for column numbers
FIG2_CAM_ALPHA = 0.60       # max overlay opacity of the per-class Grad-CAM tint
FIG2_CAM_THRESHOLD = 0.30   # only the strongest Grad-CAM peaks are tinted
FIG2_BG_VALUE = 225.0       # display transmission = raw / 225, the models' input scaling
# Layout in POINTS. The canvas is exactly one double column (style.DOUBLE) wide at a dpi
# fixed by the row width: two 1032 x 772 frames scaled to a 700-px row height,
# int(1032 * 700 / 772) = 935 px each.
FIG2_MARGIN_PT, FIG2_GAP_PT, FIG2_ROW_GAP_PT = 6, 8, 8
FIG2_ROW_W_PX = 2 * 935


class _Canvas:
    """Matplotlib figure with every position in mm from the top-left corner."""

    def __init__(self, width_mm, height_mm):
        self.w, self.h = width_mm, height_mm
        self.fig = plt.figure(figsize=(width_mm * style.MM, height_mm * style.MM))

    def axes(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h],
                                 **kw)

    def text(self, x, y, s, **kw):
        return self.fig.text(x / self.w, 1 - y / self.h, s, **kw)


def _pil_font(size, bold=True):
    from PIL import ImageFont
    # Arial as in the matplotlib figures, then sans-serif fallbacks (never serif).
    candidates = (
        ["/usr/share/fonts/msfonts/Arialbd.TTF",
         "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
         "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold else
        ["/usr/share/fonts/msfonts/Arial.TTF",
         "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
         "/usr/share/fonts/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


class _Fig2Contrast:
    """Display mapping: transmission (raw / background value, the model's input scaling),
    a linear window = pooled p0.5-p99.8 of every gallery crop (frozen), then an unsharp
    mask (Gaussian radius 1.0 px, amount 0.4)."""

    def __init__(self, crops, bg_value, seed, max_px=2_000_000):
        self.bg = bg_value
        v = np.concatenate([self.transmission(a).ravel() for a in crops])
        if len(v) > max_px:
            v = np.random.default_rng(seed).choice(v, max_px, replace=False)
        self.lo, self.hi = float(np.percentile(v, 0.5)), float(np.percentile(v, 99.8))

    def transmission(self, img_u8):
        return img_u8.astype(np.float32) / self.bg

    def __call__(self, img_u8):
        from scipy.ndimage import gaussian_filter
        d = np.clip((self.transmission(img_u8) - self.lo) / (self.hi - self.lo), 0, 1)
        return np.clip(d + 0.4 * (d - gaussian_filter(d, 1.0, mode="nearest", truncate=4.0)),
                       0, 1)


def _fig2_picks(ci, args, cache):
    """Gallery cells and the two frames, chosen by rule (cached as CSV)."""
    from remo_id.interpret.cell_occlusion import (recorded_experiments, select_example_frames,
                                                  select_informative_cells)
    gal_csv, fr_csv = os.path.join(cache, "gallery_cells.csv"), os.path.join(cache, "frames.csv")
    if os.path.exists(gal_csv) and os.path.exists(fr_csv):
        return (pd.read_csv(gal_csv, dtype={"experiment_id": str}),
                pd.read_csv(fr_csv, dtype={"experiment_id": str}))
    if not args.cohort:
        raise ValueError("Fig 2 needs --cohort (scope column) for the healthy-scope rule")
    ct = C.load_cohort_table(args.cohort)
    scope = dict(zip(ct["experiment_id"], ct["scope"]))
    ex = recorded_experiments(args.occlusion_dir)
    shown = ex[(ex["y_true"] != ci.healthy_idx)
               | (ex["experiment_id"].map(scope) == FIG2_HEALTHY_SCOPE)]["experiment_id"]
    cells = pd.read_parquet(os.path.join(args.occlusion_report, "screen_cells.parquet"))
    cells["experiment_id"] = cells["experiment_id"].astype(str)
    gal = select_informative_cells(cells[cells["experiment_id"].isin(set(shown))], n=FIG2_N,
                                   type_quota=FIG2_TYPE_QUOTA, seed=args.seed)
    frames = select_example_frames(args.occlusion_dir, classes=[ci.neg_idx, ci.sepsis_idx],
                                   experiments=set(shown), seed=args.seed)
    meta = ex[["experiment_id", "path", "checkpoint"]]
    gal = gal.drop(columns=["path"], errors="ignore").merge(meta, on="experiment_id", how="left")
    frames = frames.merge(meta, on="experiment_id", how="left")
    os.makedirs(cache, exist_ok=True)
    gal.to_csv(gal_csv, index=False)
    frames.to_csv(fr_csv, index=False)
    return gal, frames


def _fig2_crops(gal, cache):
    """Raw crop + the cell's own mask per gallery cell, cut once from the zarr stacks."""
    from PIL import Image
    from remo_id.interpret.cell_occlusion import open_frames, open_masks, read_stack
    d = os.path.join(cache, "crops")
    os.makedirs(d, exist_ok=True)
    out = []
    for rd in gal.to_dict("records"):
        stem = os.path.join(d, f"{rd['experiment_id']}_{int(rd['image_id'])}_{int(rd['label'])}")
        if not os.path.exists(stem + "_mask.png"):
            frames, masks = open_frames(rd["path"]), open_masks(rd["path"])
            H, W = frames.shape[0], frames.shape[1]
            iid, lab, w = int(rd["image_id"]), int(rd["label"]), FIG2_WINDOW
            y0 = int(np.clip(int(round(rd["centroid_y"])) - w // 2, 0, H - w))
            x0 = int(np.clip(int(round(rd["centroid_x"])) - w // 2, 0, W - w))
            crop = read_stack(frames, [iid])[0][y0:y0 + w, x0:x0 + w]
            m = read_stack(masks, [iid])[0][y0:y0 + w, x0:x0 + w] == lab
            if not m.any():
                raise ValueError(f"label {lab} absent in its crop window (mask/regionprops mismatch?)")
            Image.fromarray(crop).save(stem + ".png")
            Image.fromarray((m * 255).astype(np.uint8)).save(stem + "_mask.png")
        out.append((np.asarray(Image.open(stem + ".png")),
                    np.asarray(Image.open(stem + "_mask.png")) > 0))
    return out


def _fig2_cams(frame_rows, cache):
    """Per-class Grad-CAM of each chosen frame by its held-out fold model (cached npz)."""
    from remo_id.interpret.gradcam import frame_class_gradcams
    d = os.path.join(cache, "cams")
    os.makedirs(d, exist_ok=True)
    out = []
    for r in frame_rows.to_dict("records"):
        p = os.path.join(d, f"{r['experiment_id']}_{int(r['image_id'])}.npz")
        if not os.path.exists(p):
            cams, raw = frame_class_gradcams(r["checkpoint"], r["path"], int(r["image_id"]))
            np.savez_compressed(p, cams=cams, raw=raw)
        z = np.load(p)
        out.append((z["cams"], z["raw"]))
    return out


def _fig2_gallery(cv, x, y, width, rows, display, colors, dry=False, shade_alpha=0.4):
    """One row per class, uniform gutters, bold row labels in a left column, column
    numbers on top; everything outside the informative cell tinted in the class colour."""
    n, label_w, gut, head = FIG2_N, FIG2_LABEL_W_MM, 0.6, FIG2_COLNUM_H
    side = (width - label_w - gut * (n - 1)) / n
    H = head + len(rows) * (side + gut) - gut
    if dry:
        return H
    for k in range(n):
        cv.text(x + label_w + k * (side + gut) + side / 2, y + head - 0.6, str(k + 1),
                ha="center", va="bottom", color=style.MUTE)
    for ri, (name, items) in enumerate(rows):
        yy = y + head + ri * (side + gut)
        cv.text(x + label_w - 1.5, yy + side / 2, name, ha="right", va="center",
                color=colors[name], fontweight="bold")
        for k in range(n):
            ax = cv.axes(x + label_w + k * (side + gut), yy, side, side)
            ax.set_xticks([])
            ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            if k >= len(items):
                continue
            img, m = items[k]
            ax.imshow(display(img), cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
            over = np.zeros(m.shape + (4,))
            over[..., :3] = mpl.colors.to_rgb(colors[name])
            over[..., 3] = np.where(m, 0.0, shade_alpha)
            ax.imshow(over, interpolation="nearest")
    return H


def fig2_interpretability_occlusion(out, ci, args):
    """Fig 2: a) informative cells per class from the occlusion screen; b) ICU-control
    and c) Sepsis frame with a per-class Grad-CAM tint (rules above)."""
    import io
    from PIL import Image, ImageDraw
    from remo_id.interpret.overlays import class_tint_overlay
    if not os.path.exists(os.path.join(args.occlusion_report, "screen_cells.parquet")):
        print(f"  fig2 skipped: no screen_cells.parquet in {args.occlusion_report} "
              "(run `remo-id analysis occlusion`)")
        return "skip"
    style.apply()
    cache = args.fig2_cache
    gal, frames = _fig2_picks(ci, args, cache)
    crops = _fig2_crops(gal, cache)
    chosen = frames[frames["cand_rank"] == 1].set_index("y_true")
    tiles = _fig2_cams(chosen.loc[[ci.neg_idx, ci.sepsis_idx]].reset_index(), cache)
    names = [ci.class_names[i] for i in ci.display_order]
    colors = dict(zip(names, style.class_colors_muted(names)))
    display = _Fig2Contrast([c for c, _ in crops], FIG2_BG_VALUE, args.seed)
    by_cls = {}
    for (img, m), rd in zip(crops, gal.to_dict("records")):
        by_cls.setdefault(int(rd["y_true"]), []).append((int(rd["slot"]), (img, m)))
    rows = [(ci.class_names[i], [v for _, v in sorted(by_cls[i], key=lambda t: t[0])])
            for i in ci.display_order if i in by_cls]

    content_in = style.DOUBLE - (2 * FIG2_MARGIN_PT + FIG2_GAP_PT) / 72.0
    dpi = FIG2_ROW_W_PX / content_in
    label_font = _pil_font(style.pt_to_px(style.PANEL_LABEL_PT, dpi), bold=True)
    title_font = _pil_font(style.pt_to_px(style.font_pt("axes.titlesize"), dpi), bold=True)
    gap = style.pt_to_px(FIG2_GAP_PT, dpi)
    row_gap = style.pt_to_px(FIG2_ROW_GAP_PT, dpi)
    margin = style.pt_to_px(FIG2_MARGIN_PT, dpi)
    label_h = style.pt_to_px(max(style.PANEL_LABEL_PT, style.font_pt("axes.titlesize")) + 3, dpi)
    content_w = FIG2_ROW_W_PX + gap

    # panel a: gallery rendered at exactly content_w px wide
    w_mm = content_w / dpi * 25.4
    cv = _Canvas(w_mm, _fig2_gallery(None, 0, 0, w_mm, rows, display, colors, dry=True))
    _fig2_gallery(cv, 0, 0, w_mm, rows, display, colors)
    buf = io.BytesIO()
    cv.fig.savefig(buf, dpi=dpi, facecolor="white", format="png")
    plt.close(cv.fig)
    gal_im = Image.open(buf).convert("RGB")
    if gal_im.width != content_w:                     # rounding guard (at most 1 px)
        pad = Image.new("RGB", (content_w, gal_im.height), (255, 255, 255))
        pad.paste(gal_im.crop((0, 0, min(content_w, gal_im.width), gal_im.height)), (0, 0))
        gal_im = pad
    label_px = int(round(FIG2_LABEL_W_MM / 25.4 * dpi))

    # panels b, c: per-class Grad-CAM tint over the frame in the gallery's display mapping
    tile_w = (content_w - gap) // 2
    cam_colors = [mpl.colors.to_rgb(colors.get(n, style.FG)) for n in ci.class_names]
    pil_tiles = []
    for cams, raw in tiles:
        raw = raw if raw.ndim == 2 else raw[..., 0]
        rgb = class_tint_overlay(display(raw.astype(np.uint8)), cams, cam_colors,
                                 threshold=FIG2_CAM_THRESHOLD, alpha=FIG2_CAM_ALPHA)
        t = Image.fromarray((rgb * 255).astype(np.uint8))
        pil_tiles.append(t.resize((tile_w, int(round(t.height * tile_w / t.width))),
                                  Image.LANCZOS))
    x0 = margin
    x1 = x0 + tile_w + (content_w - 2 * tile_w)       # absorb rounding in the gap
    y_t = margin + label_h + gal_im.height + row_gap
    canvas = Image.new("RGB", (margin * 2 + content_w,
                               y_t + label_h + pil_tiles[0].height + margin), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for (letter, idx), tx, img in zip((("b", ci.neg_idx), ("c", ci.sepsis_idx)), (x0, x1),
                                      pil_tiles):
        draw.text((tx, y_t), letter, fill=(0, 0, 0), font=label_font, anchor="lt")
        draw.text((tx + img.width // 2, y_t + 2), ci.class_names[idx], fill=(0, 0, 0),
                  font=title_font, anchor="mt")
        canvas.paste(img, (tx, y_t + label_h))
    draw.text((margin, margin), "a", fill=(0, 0, 0), font=label_font, anchor="lt")
    draw.text((margin + label_px + (content_w - label_px) // 2, margin + 2), FIG2_TITLE_A,
              fill=(0, 0, 0), font=title_font, anchor="mt")
    canvas.paste(gal_im, (margin, margin + label_h))
    path = os.path.join(out, "fig2_interpretability_occlusion.png")
    canvas.save(path, dpi=(dpi, dpi))
    print(f"  wrote {path} ({canvas.width}x{canvas.height} px @ {dpi:.1f} dpi); picks in {cache}")

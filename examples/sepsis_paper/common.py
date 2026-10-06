"""Shared helpers for the paper scripts.

Importing this module registers the paper's class colours with the package style
(``remo_id.reports.style.set_class_colors``) so every figure -- main figures,
tables and the cell occlusion figure -- draws Sepsis / ICU control / Healthy in the same
colours as the published figures. Import it before building any figure.

Also holds what make_paper_results.py and figures_paper.py both need (class
resolution, prediction loading, patient aggregation, the manifest and its clinical
columns, table rendering, step logging, fixed-epoch checkpoint resolution), so
neither imports the other.
"""
import glob
import os
import traceback

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from remo_id.reports import style  # noqa: E402

# ===========================================================================
# Class colours
# ===========================================================================
SEPSIS = "#B04E00"    # vermillion
ICU = "#1B5E8A"       # blue
HEALTHY = "#2D7A5E"   # green

# Every spelling of a class name that reaches a figure: config names, the
# prettified display names (resolve_class_info) and the title-case variant.
PAPER_CLASS_COLORS = {
    "Sepsis": SEPSIS, "sepsis": SEPSIS,
    "ICU control": ICU, "ICU Control": ICU, "ICU-Control": ICU, "ICU-ctrl": ICU,
    "Healthy": HEALTHY, "healthy": HEALTHY,
}

style.set_class_colors(PAPER_CLASS_COLORS)


# ===========================================================================
# Step logging: one failing output never aborts the rest of the set
# ===========================================================================
STATUS = []


def ok(name, msg=""):
    STATUS.append(("OK", name, msg))
    print(f"[OK]   {name} {msg}")


def skip(name, msg=""):
    STATUS.append(("SKIP", name, msg))
    print(f"[SKIP] {name} {msg}")


def fail(name, err):
    STATUS.append(("FAIL", name, str(err)))
    print(f"[FAIL] {name}: {err}")
    traceback.print_exc()


def step(name, fn, *args, **kwargs):
    """Run an output step; a builder returning "skip" is reported as skipped."""
    print(f"\n=== {name} ===")
    try:
        result = fn(*args, **kwargs)
        if result == "skip":
            skip(name)
            return
        ok(name)
    except Exception as e:  # noqa: BLE001 - keep building the rest of the set
        fail(name, e)


# ===========================================================================
# Fixed-epoch predictions / checkpoints
# ===========================================================================
def epoch_npz(pred_dir, epoch):
    """``{pred_dir}/epoch_{epoch}.npz`` written by ``remo-id workflow loo-predict``."""
    p = os.path.join(pred_dir, f"epoch_{epoch}.npz")
    if not os.path.exists(p):
        avail = sorted(os.path.basename(f)
                       for f in glob.glob(os.path.join(pred_dir, "epoch_*.npz")))
        raise FileNotFoundError(f"No epoch-{epoch} predictions at {p} (found {avail})")
    return p


def fold_checkpoint(fold_dir, epoch):
    """The fold's checkpoint at ``epoch``, resolved exactly as the loo-* workflows
    do (never a different epoch)."""
    from remo_id.workflows.loo_report import _find_epoch_checkpoint
    ckpt = _find_epoch_checkpoint(fold_dir, epoch)
    if ckpt is None:
        raise FileNotFoundError(f"No epoch-{epoch} checkpoint in {fold_dir}")
    return ckpt


# ===========================================================================
# Class order / names
# ===========================================================================
class ClassInfo:
    """Resolved class order/names + semantic role indices."""

    def __init__(self, labels, class_names, sepsis_idx, neg_idx, healthy_idx,
                 display_order, sepsis_study, healthy_study):
        self.labels = labels                    # manifest labels, index-aligned to model
        self.class_names = class_names          # display names, index-aligned to model
        self.sepsis_idx = sepsis_idx
        self.neg_idx = neg_idx                  # ICU control (sepsis-negative)
        self.healthy_idx = healthy_idx
        self.display_order = display_order      # list of indices for figure ordering
        self.sepsis_study = sepsis_study
        self.healthy_study = healthy_study

    @property
    def display_names(self):
        return [self.class_names[i] for i in self.display_order]

    def prob_col(self, idx):
        return f"P({self.class_names[idx]})"


def _role_index(class_names, *substrings):
    for i, n in enumerate(class_names):
        low = n.lower()
        if any(s in low for s in substrings):
            return i
    return None


def pretty_class_name(n):
    """Display label: the ICU-control class is "ICU control"; lowercase config
    names are capitalized ("sepsis" -> "Sepsis"); others are left as-is."""
    if n.lower() in ("icu-ctrl", "icu ctrl", "icu-control", "icu control"):
        return "ICU control"
    return n.capitalize() if n.islower() else n


def model_class_names(cv_dir):
    """Class names in the model's index order (the fold classes.yaml)."""
    fold_yaml = sorted(glob.glob(os.path.join(cv_dir, "repeat*_fold*", "classes.yaml")))
    if not fold_yaml:
        raise FileNotFoundError(f"No fold classes.yaml under {cv_dir}")
    with open(fold_yaml[0]) as f:
        return [str(c["name"]) for c in yaml.safe_load(f)["classes"]]


def resolve_classes(cv_dir, classes=None):
    """The model's class names; ``classes`` ("A,B,C", --classes) must match them."""
    names = model_class_names(cv_dir)
    if classes:
        given = [c.strip() for c in classes.split(",") if c.strip()]
        if given != names:
            raise SystemExit(f"--classes {given} do not match the model's class order "
                             f"{names} (repeat*_fold*/classes.yaml)")
    return names


def resolve_class_info(cv_dir, manifest, classes=None):
    """Class order and names of the model (fold classes.yaml), roles by name, and
    the study of each role's labelled manifest rows. Display order is
    [sepsis, ICU control, healthy]."""
    labels = resolve_classes(cv_dir, classes)
    class_names = [pretty_class_name(n) for n in labels]
    sepsis_idx = _role_index(class_names, "sepsis")
    healthy_idx = _role_index(class_names, "healthy")
    neg_idx = _role_index(class_names, "icu", "ctrl", "control")
    if neg_idx is None:
        others = [i for i in range(len(class_names)) if i not in (sepsis_idx, healthy_idx)]
        neg_idx = others[0] if others else None

    def _study(idx):
        return manifest.label_study(labels[idx]) if idx is not None else None

    pref = [i for i in (sepsis_idx, neg_idx, healthy_idx) if i is not None]
    display_order = pref + [i for i in range(len(class_names)) if i not in pref]
    return ClassInfo(labels, class_names, sepsis_idx, neg_idx, healthy_idx,
                     display_order, _study(sepsis_idx), _study(healthy_idx))


# ===========================================================================
# The manifest: experiments, patients, scopes and clinical columns
# ===========================================================================
DEFAULT_CLINICAL_SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "configs", "clinical_tables.yaml")


def spec_columns(spec):
    """Every clinical column the spec names."""
    cols = []

    def add(*names):
        cols.extend(n for n in names if n and n not in cols)

    for terms in spec.get("derived_scores", {}).values():
        add(*(c for term in terms for c, _, _ in term))
    add(*spec["diagnosis"].values())
    d = spec["demographics"]
    add(d["mortality"]["column"], d["microbiology"]["column"], d["pathogen_column"],
        d["age_column"], *d["sex_columns"])
    cc = spec["cohort_characteristics"]
    add(*(r["column"] for r in cc["rows"]),
        *(r["column"] for r in cc["manuscript"].get("insert_rows", [])))
    add(*spec["variable_labels"])
    for entries in spec["correlation_groups"].values():
        for e in entries:
            add(*(e if isinstance(e, list) else [e]))
    add(*spec["discrimination"]["variables"])
    return cols


class Manifest:
    """The manifest (``--manifest``) and the clinical spec
    (``configs/clinical_tables.yaml``).

    One row per experiment: ``experiment_id``, ``sample_id``, ``study_id``,
    ``scope``, ``path``, ``num_images``, ``label`` (class name, empty for
    experiments outside the classes) and, optionally, the clinical columns the spec
    names. Only empty fields are missing values.
    """

    def __init__(self, path, spec_path=DEFAULT_CLINICAL_SPEC):
        from remo_id.data.manifest import load_manifest
        self.table = load_manifest(path)
        with open(spec_path) as f:
            self.spec = yaml.safe_load(f)

    def _map(self, col):
        t = self.table.dropna(subset=[col])
        return dict(zip(t["experiment_id"], t[col].astype(str)))

    def sample_map(self):
        """experiment_id -> sample_id over every row with a sample_id."""
        return self._map("sample_id")

    def scope_map(self):
        """experiment_id -> scope."""
        if "scope" not in self.table.columns:
            raise KeyError("the manifest has no scope column")
        return self._map("scope")

    def label_study(self, label):
        """The study of the rows labelled ``label``."""
        studies = self.table.loc[self.table["label"] == label, "study_id"].dropna().unique()
        if len(studies) > 1:
            raise ValueError(f"rows labelled '{label}' span studies {list(studies)}")
        return str(studies[0]) if len(studies) else None

    def study(self, study_id):
        """Rows of one study in file order."""
        d = self.table[self.table["study_id"] == str(study_id)]
        if d.empty:
            raise KeyError(f"no rows of study '{study_id}' in the manifest")
        return d.reset_index(drop=True)

    def clinical_columns(self):
        """The spec's clinical columns that hold data in the manifest."""
        return [c for c in spec_columns(self.spec)
                if c in self.table.columns and self.table[c].notna().any()]


# ===========================================================================
# Pooled fixed-epoch LOO predictions
# ===========================================================================
def load_predictions(pred_dir, epoch, sample_map):
    """Pooled held-out predictions written by ``remo-id workflow loo-predict``
    (one row per experiment, the SAME epoch checkpoint for every fold).

    Returns dict(y_true, y_probs, y_pred, eids, sample_ids, study_ids); sample
    ids come from ``sample_map`` ("" where unknown).
    """
    d = np.load(epoch_npz(pred_dir, epoch), allow_pickle=True)
    eids = np.asarray(d["experiment_ids"]).astype(str)
    study_ids = (np.asarray(d["study_ids"]).astype(str)
                 if "study_ids" in d.files else np.array([""] * len(eids)))
    return dict(y_true=np.asarray(d["y_true"]).astype(int),
                y_probs=np.asarray(d["y_probs"]).astype(float),
                y_pred=np.asarray(d["y_pred"]).astype(int),
                eids=eids,
                sample_ids=np.array([sample_map.get(e, "") for e in eids]),
                study_ids=study_ids)


def aggregate_preds_to_patient(preds, mode):
    """Collapse the experiment-level ``preds`` dict to patient (sample_id) level.

    Reuses ``remo_id.workflows.cv.aggregate_to_patient_level`` for the
    (y_true, y_probs, y_pred) reduction (``combined`` averages the probability
    vectors across a patient's replicates), then rebuilds ``eids``/``study_ids``
    as one representative value per patient (earliest experiment_id, its study).
    """
    from remo_id.workflows.cv import aggregate_to_patient_level

    sids_in = preds.get("sample_ids")
    if sids_in is None or len(sids_in) == 0 or all(
            str(s) in ("", "None", "nan") for s in sids_in):
        raise ValueError("patient level requested but predictions carry no sample_ids")

    eids_in = np.asarray(preds["eids"]).astype(str)
    sids_in = np.asarray(sids_in).astype(str)
    study_in = np.asarray(preds["study_ids"]).astype(str)

    yt, yp, ypred, sids = aggregate_to_patient_level(
        np.asarray(preds["y_true"]), np.asarray(preds["y_probs"]),
        np.asarray(preds["y_pred"]), eids_in, sids_in, mode=mode)

    samp2study, samp2eid = {}, {}
    for i in np.argsort(eids_in):
        s = sids_in[i]
        samp2study.setdefault(s, study_in[i])
        samp2eid.setdefault(s, eids_in[i])
    study_ids = np.array([samp2study.get(s, "") for s in sids])
    eids = np.array([samp2eid.get(s, s) for s in sids])

    print(f"patient-level ({mode}): {len(eids_in)} experiments -> {len(sids)} patients")
    return dict(y_true=yt, y_probs=yp, y_pred=ypred,
                eids=eids, sample_ids=sids, study_ids=study_ids)


def youden_thresholds(y_true, y_probs, class_names):
    from remo_id.analysis.optimal_thresholds import find_youden_thresholds
    res = find_youden_thresholds(y_true, y_probs, class_names)
    return np.array([r["optimal_threshold"] for r in res])


# ===========================================================================
# Publication-styled table renderer
# ===========================================================================
def render_table(df, out_base, title=None, fontsize=7, csv_df=None):
    """Render a DataFrame as a styled table PNG and write the CSV.

    ``csv_df``: write this frame to the CSV instead of the rendered one -- lets a
    caller line-wrap long cells for the figure while keeping the CSV unwrapped.
    Cells containing newlines widen the row to fit.
    """
    style.apply()
    df = df.fillna("")
    n_rows, n_cols = df.shape
    cell_text = df.astype(str).values.tolist()
    col_labels = list(df.columns)

    # Width is driven by the widest LINE, not the whole cell.
    def _w(s):
        return max((len(ln) for ln in str(s).split("\n")), default=0)

    # Tallest wrapped cell sets a uniform row height.
    max_lines = max([1] + [str(v).count("\n") + 1 for r in cell_text for v in r])
    fig_w = max(6.0, 0.10 * sum(max(_w(c), *(_w(r[j]) for r in cell_text))
                                for j, c in enumerate(col_labels)))
    fig_h = max(1.2, 0.30 * (n_rows + 2) * max_lines)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize + 3, fontweight="bold", pad=10)

    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.4 * max_lines)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(style.MUTE_FAINT)
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor(style.FG)
        elif r % 2 == 0:
            cell.set_facecolor("#F7F7F7")
    try:
        tbl.auto_set_column_width(col=list(range(n_cols)))
    except Exception:  # noqa: BLE001
        pass

    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (csv_df if csv_df is not None else df).to_csv(f"{out_base}.csv", index=False)

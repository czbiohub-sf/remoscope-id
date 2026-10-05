"""Shared helpers for the paper scripts.

Importing this module registers the paper's class colours with the package style
(``remo_id.reports.style.set_class_colors``) so every figure -- main figures,
tables and the Fig 2 chain -- draws Sepsis / ICU control / Healthy in the same
colours as the published figures. Import it before building any figure.

Also holds what make_paper_results.py and figures_paper.py both need (class
resolution, prediction loading, patient aggregation, the clinical input table,
table rendering, step logging, fixed-epoch checkpoint resolution), so neither
imports the other.
"""
import glob
import os
import traceback
import warnings

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

    def __init__(self, class_names, sepsis_idx, neg_idx, healthy_idx,
                 display_order, sepsis_study, healthy_study, class_definitions):
        self.class_names = class_names          # display names, index-aligned to model
        self.sepsis_idx = sepsis_idx
        self.neg_idx = neg_idx                  # ICU control (sepsis-negative)
        self.healthy_idx = healthy_idx
        self.display_order = display_order      # list of indices for figure ordering
        self.sepsis_study = sepsis_study
        self.healthy_study = healthy_study
        self.class_definitions = class_definitions

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


def resolve_class_info(cv_dir, class_config_path):
    """Use the model's class ORDER (fold classes.yaml) + the config's NAMES.

    The fold classes.yaml is authoritative for index order; the config supplies
    the display names and is validated to share the same study/column/value order.
    Display order is [sepsis, ICU control, healthy].
    """
    from remo_id.inference.context import load_class_config

    fold_yaml = sorted(glob.glob(os.path.join(cv_dir, "repeat*_fold*", "classes.yaml")))
    if not fold_yaml:
        raise FileNotFoundError(f"No fold classes.yaml under {cv_dir}")
    with open(fold_yaml[0]) as f:
        model_classes = yaml.safe_load(f)["classes"]

    class_definitions, cfg_names, *_ = load_class_config(class_config_path)

    def _key(c):
        return (str(c.get("study_id")), str(c.get("column")), str(c.get("value")))

    if len(class_definitions) == len(model_classes):
        for i, (a, b) in enumerate(zip(class_definitions, model_classes)):
            if _key(a) != _key(b):
                warnings.warn(
                    f"class_config index {i} {_key(a)} != model classes.yaml {_key(b)}; "
                    "display names may be misaligned with model outputs.")

    class_names = [pretty_class_name(n) for n in cfg_names]
    sepsis_idx = _role_index(class_names, "sepsis")
    healthy_idx = _role_index(class_names, "healthy")
    neg_idx = _role_index(class_names, "icu", "ctrl", "control")
    if neg_idx is None:
        others = [i for i in range(len(class_names)) if i not in (sepsis_idx, healthy_idx)]
        neg_idx = others[0] if others else None

    sepsis_study = model_classes[sepsis_idx].get("study_id") if sepsis_idx is not None else "sepsis"
    healthy_study = model_classes[healthy_idx].get("study_id") if healthy_idx is not None else None

    pref = [i for i in (sepsis_idx, neg_idx, healthy_idx) if i is not None]
    display_order = pref + [i for i in range(len(class_names)) if i not in pref]
    return ClassInfo(class_names, sepsis_idx, neg_idx, healthy_idx,
                     display_order, sepsis_study, healthy_study, class_definitions)


# ===========================================================================
# experiment -> sample (patient) mapping
# ===========================================================================
def load_cohort_table(path):
    """Cohort table (build_cohort_table.py / data release): one row per
    experiment with experiment_id, sample_id, study_id, scope."""
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = {"experiment_id", "sample_id"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    return df


# ===========================================================================
# Clinical input table + its spec
# ===========================================================================
DEFAULT_CLINICAL_SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "configs", "clinical_tables.yaml")


class Clinical:
    """The clinical input table (``--clinical_table``) and its spec
    (``configs/clinical_tables.yaml``).

    The table has one row per experiment: ``experiment_id``, ``sample_id``,
    ``study_id`` (as named in the class config), ``path``/``num_images`` and the
    clinical variables the spec names. Rows of several studies share the file; a
    column that is empty for every row of a study is "not recorded" by that study
    (``study`` drops it). Only empty fields are missing values, so text such as
    "None" stays text. The spec's ``single_precision_columns`` are parsed as float32.
    """

    def __init__(self, table_path, spec_path=DEFAULT_CLINICAL_SPEC):
        self.table = pd.read_csv(table_path, keep_default_na=False, na_values=[""],
                                 dtype={"experiment_id": str, "sample_id": str,
                                        "study_id": str})
        missing = {"experiment_id", "sample_id", "study_id"} - set(self.table.columns)
        if missing:
            raise ValueError(f"{table_path} lacks columns {sorted(missing)}")
        with open(spec_path) as f:
            self.spec = yaml.safe_load(f)
        for c in self.spec.get("single_precision_columns", []):
            if c in self.table.columns:
                self.table[c] = self.table[c].astype(np.float32)

    def study(self, study_id):
        """Rows of one study in file order, without the columns it does not record."""
        d = self.table[self.table["study_id"] == str(study_id)]
        if d.empty:
            raise KeyError(f"no rows of study '{study_id}' in the clinical table")
        return d.dropna(axis=1, how="all").reset_index(drop=True)

    def sample_map(self):
        """experiment_id -> sample_id over every row with a sample_id."""
        t = self.table.dropna(subset=["sample_id"])
        return dict(zip(t["experiment_id"], t["sample_id"]))


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

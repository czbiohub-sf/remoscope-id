"""Cell occlusion screen: which individual cells a MIL model's prediction depends on.

Every cell of a sampled frame subset is removed with the fill model of
:mod:`remo_id.interpret.occlusion`, and the change in the model's true-class log-odds is
recorded. One run scores a set of experiments with ONE checkpoint; to explain the
held-out experiments of a CV run, call it once per fold with that fold's checkpoint and
held-out experiments (all folds can share one ``work_dir``).

Stages (``remo-id analysis occlusion <stage>``; each resumes per experiment):
  background   CPU: empty-slide background model per experiment -> {work_dir}/bg/{exp}.npy
  attribution  per-cell signed Grad-CAM of the true-class log-odds (ResNet layer4, each
               frame a bag of one frame) -> {work_dir}/attribution/{exp}.parquet
  screen       stage 1, random-subset screen: on s1_frames frames, n_masks random masks
                 each removing every cell with probability p; area-adjusted ridge
                 estimate of every cell's removal effect -> screen/stage1/{exp}.cells.parquet
               stage 2, exact single-cell removal of candidates (stage-1 top, Grad-CAM top,
                 random controls) with 2 fill seeds, a SHAM (same fill, the cell's own
                 pixels restored), n_null equal-area nulls (random cells of the null cell
                 type; cells much smaller than those: the cell's shape translated onto
                 background) and bag-level deltas; plus n_random_per_type random cells per
                 cell type, same protocol -> screen/stage2/{exp}.parquet, {exp}.random.parquet
  report       stage-2 cells + stage-1 info -> {out}/screen_cells.parquet; top cells per
               class (remove minus sham, beating the null) -> {out}/screen_top_manifest.csv
Every stage records the experiment (class, path, checkpoint) in {work_dir}/experiments/.

Targets per frame (a bag of one frame, as ``remo-id workflow loo-per-image``):
  own      = logit[y] - logsumexp(logit[others])   true-class log-odds
  contrast = logit[pos] - logit[neg]               (default: last class - first class)
Binary models (one logit z) are scored as two-class logits [0, z]. All deltas are
(occluded - baseline), where the baseline is the UNOCCLUDED frame encoded IN THE SAME JOB
with the same settings (fp32 weights/inputs, PyTorch default TF32, no autocast) and in the
same FIXED batch shape (last batch padded) so cuDNN picks the same kernels for baseline
and occluded frames.

Data format (per experiment directory, next to the image stack of
:mod:`remo_id.data.zarr_io`):

``{dirname}masks.zarr`` (or the only ``*masks.zarr``)
    integer zarr array ``[H, W, N]``, the same shape as the image stack: frame ``i``'s
    label image is ``array[:, :, i]``; 0 = background, ``k > 0`` = cell ``k`` of that
    frame (labels are per frame). Masks must align pixel for pixel with the frames the
    model sees, so the model must score full frames (MIL backbones with global pooling
    do; no spatial eval transform is applied).
``{dirname}regionprops.parquet`` (or the only ``*regionprops.parquet``)
    one row per cell: ``image_id`` (frame index), ``label`` (mask label in that frame),
    ``area`` (px), ``centroid_y``, ``centroid_x``, bounding box ``ty, tx, by, bx``
    (top/left inclusive, bottom/right exclusive, as skimage ``bbox``), plus every column
    the cell config gates use (e.g. ``intensity_mean``). Cells are typed with
    :func:`remo_id.data.cell_config.classify_cells` (first match wins).

Frames are normalised as at evaluation: ``uint8 / background_value`` of the checkpoint.
"""
from __future__ import annotations

import glob
import json
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from remo_id.interpret import occlusion as O

SAT = np.log(999.0)             # own log-odds > SAT <=> P(true) > 0.999 (saturated)
LOOKAHEAD = 4                   # encoder batches of occlusions prepared ahead
READ_BATCH = 32
RP_COLS = ["image_id", "label", "cell_type", "area", "centroid_y", "centroid_x", "ty", "tx",
           "by", "bx", "eccentricity", "solidity", "perimeter_crofton", "intensity_mean",
           "intensity_std", "intensity_min", "intensity_max", "intensity_median",
           "intensity_skewness", "intensity_kurtosis", "intensity_entropy",
           "bright_pixel_fraction", "radial_intensity_ratio"]
REQUIRED_RP_COLS = ("image_id", "label", "area", "centroid_y", "centroid_x",
                    "ty", "tx", "by", "bx")
T_S1_MASK, T_S1_OCC = 11, 1_000_000          # seed tags (distinct streams per purpose)
T_S2_EXACT, T_S2_NULL = 2_000_000, 3_000_000


@dataclass
class OcclusionConfig:
    """Output dir, inputs and settings of an occlusion run (defaults = the published
    screen). The cell-type fields default from the cell config: null = its catch-all
    class (else the last), focus = its highest-weight class, random strata = every class
    in config order; ``contrast`` = (pos, neg) class indices, default (K-1, 0)."""
    work_dir: str
    checkpoint: str | None = None
    cell_config: str | None = None
    device: str | None = None
    seed: int = 403
    n_frames: int = 500                   # frames screened per experiment
    min_image_index: int = 1000           # skip run warm-up frames
    s1_frames: int = 150
    n_masks: int = 1000
    p: float = 0.01
    lam: float = 0.1
    n_null: int = 4
    n_random_per_type: int = 25
    border_px: int = 48
    gpu_batch: int = 48
    workers: int = 7
    null_cell_type: str | None = None
    focus_cell_type: str | None = None
    random_cell_types: list | None = None
    contrast: tuple | None = None

    def path(self, *parts) -> str:
        return os.path.join(self.work_dir, *parts)

    def out_path(self, stage, name) -> str:
        os.makedirs(self.path(*stage.split("/")), exist_ok=True)
        return self.path(*stage.split("/"), name)

    def frames(self, exp_id, num_images: int) -> np.ndarray:
        """Deterministic sorted frame subset of one experiment (the same in every stage)."""
        rng = np.random.default_rng([self.seed, int(exp_id) % (2**31)])
        pool = np.arange(self.min_image_index, num_images)
        if len(pool) == 0:
            raise ValueError(f"experiment {exp_id}: no frames at or after min_image_index")
        return np.sort(rng.choice(pool, size=min(self.n_frames, len(pool)), replace=False))

    def resolve_cell_types(self):
        from remo_id.data.cell_config import CATCH_ALL, load_cell_config
        cc = load_cell_config(self.cell_config)
        catch = [c.name for c in cc.classes if c.where.strip() == CATCH_ALL]
        self.null_cell_type = self.null_cell_type or (catch or [cc.classes[-1].name])[0]
        self.focus_cell_type = self.focus_cell_type or cc.classes[int(np.argmax(cc.weights))].name
        self.random_cell_types = self.random_cell_types or list(cc.names)


def _seed(seed, exp_id, iid, tag):
    return np.random.default_rng([seed, int(exp_id) % (2**31), int(iid), int(tag)])


def save_parquet(df, path):
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def resolve_experiments(class_config: str, experiment_ids=None) -> pd.DataFrame:
    """experiment_id, y_true, class_name, path, num_images of the class config's
    experiments (database- or manifest-backed), optionally restricted to ``experiment_ids``."""
    from remo_id.inference.core import resolve_experiments_from_class_config
    ids = [str(e) for e in experiment_ids] if experiment_ids else None
    ex, names, _, _ = resolve_experiments_from_class_config(class_config, experiment_ids=ids)
    if ids and set(ids) - set(ex.experiment_id):
        raise KeyError(f"experiments not found via {class_config}: "
                       f"{sorted(set(ids) - set(ex.experiment_id))}")
    ex = ex.rename(columns={"y": "y_true"}).sort_values("experiment_id")
    ex["class_name"] = [names[int(y)] for y in ex.y_true]
    return ex[["experiment_id", "y_true", "class_name", "path", "num_images"]].reset_index(drop=True)


def _todo(cfg, experiments, stage, suffixes, overwrite):
    """Rows still missing an output; records every row in {work_dir}/experiments/."""
    for r in experiments.itertuples(index=False):
        with open(cfg.out_path("experiments", f"{r.experiment_id}.json"), "w") as fh:
            json.dump(dict(experiment_id=str(r.experiment_id), y_true=int(r.y_true),
                           class_name=r.class_name, path=r.path, num_images=int(r.num_images),
                           checkpoint=cfg.checkpoint and os.path.abspath(cfg.checkpoint)), fh)
    done = [all(os.path.exists(cfg.path(*stage.split("/"), f"{e}{s}")) for s in suffixes)
            for e in experiments.experiment_id]
    todo = experiments if overwrite else experiments[~np.array(done, dtype=bool)]
    print(f"{stage}: {len(experiments)} experiments, {len(todo)} to do", flush=True)
    return todo


def recorded_experiments(work_dir: str) -> pd.DataFrame:
    """Every experiment a stage has seen in ``work_dir`` (class, path, checkpoint)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(work_dir, "experiments", "*.json"))):
        with open(f) as fh:
            rows.append(json.load(fh))
    return pd.DataFrame(rows).astype({"experiment_id": str})


# ── raw data ────────────────────────────────────────────────────────────────

def find_experiment_file(path: str, suffix: str) -> str:
    """``{path}/{dirname}{suffix}``, else the only ``*{suffix}`` entry in ``path``."""
    exact = os.path.join(path, os.path.basename(path.rstrip("/")) + suffix)
    if os.path.exists(exact):
        return exact
    hits = sorted(f for f in os.listdir(path) if f.endswith(suffix) and not f.startswith("."))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one *{suffix} in {path}, found {hits}")
    return os.path.join(path, hits[0])


def open_frames(path: str):
    from remo_id.data.zarr_io import get_zarr_file
    return get_zarr_file(path)


def open_masks(path: str):
    import zarr
    return zarr.open(find_experiment_file(path, "masks.zarr"), mode="r")


def read_stack(arr, image_ids) -> np.ndarray:
    """[B, H, W] for the given frame indices of an (H, W, T) array (serial reads)."""
    ids = [int(i) for i in image_ids]
    return np.concatenate([np.ascontiguousarray(np.moveaxis(arr[:, :, ids[s:s + READ_BATCH]], 2, 0))
                           for s in range(0, len(ids), READ_BATCH)])


def load_regionprops(path: str, cell_config: str, image_ids=None) -> pd.DataFrame:
    """Regionprops rows (optionally for a frame subset) + ``cell_type`` from the cell config."""
    from remo_id.data.cell_config import load_cell_config, classify_cells
    rp = pd.read_parquet(find_experiment_file(path, "regionprops.parquet"))
    missing = [c for c in REQUIRED_RP_COLS if c not in rp.columns]
    if missing:
        raise ValueError(f"{path}: regionprops lacks required column(s) {missing}")
    if image_ids is not None:
        rp = rp[rp["image_id"].isin(set(map(int, image_ids)))]
    rp = rp.reset_index(drop=True)
    cc = load_cell_config(cell_config)
    rp["cell_type"] = np.array(cc.names + ["unassigned"])[classify_cells(rp, cc)]
    return rp


# ── model ───────────────────────────────────────────────────────────────────

class Scorer:
    """An image MIL checkpoint scoring full frames."""

    def __init__(self, checkpoint: str, device: str = None):
        from remo_id.inference.core import (extract_background_value,
                                            extract_checkpoint_metadata, load_checkpoint)
        from remo_id.models.mil import BaseMILModel
        meta = extract_checkpoint_metadata(checkpoint)
        self.model, self.device, self.in_channels = load_checkpoint(
            checkpoint, meta["model_name"], task_type=meta["task_type"],
            num_classes=meta["num_classes"], n_experiments=1, device=device)
        if not isinstance(self.model, BaseMILModel):
            raise ValueError(f"{checkpoint}: occlusion needs an image MIL model")
        self.model.eval()
        self.background_value = extract_background_value(checkpoint)

    def normalize(self, frames_u8):
        """uint8/float [B, H, W] -> model input [B, C, H, W], as eval get_items."""
        import torch
        x = (torch.as_tensor(frames_u8).float() / self.background_value).unsqueeze(1)
        return x.expand(-1, self.in_channels, -1, -1) if self.in_channels > 1 else x

    def encode(self, x):
        """Post-projection per-frame features."""
        feats = self.model._encode_instances(x)
        if getattr(self.model, "projection", None) is not None:
            feats = self.model.projection(feats)
        return feats

    def aggregate(self, bags):
        """Class logits [n_bags, K] of bags [n_bags, n, D]; a binary logit z -> [0, z]."""
        import torch
        lg = self.model._aggregate(bags)[0]
        lg = lg.unsqueeze(-1) if lg.dim() == 1 else lg
        return torch.cat([torch.zeros_like(lg), lg], -1) if lg.shape[-1] == 1 else lg


def targets(logits: np.ndarray, y: int, contrast=None):
    """logits [..., K] -> (own true-class log-odds, contrast logit difference)."""
    lg = np.asarray(logits, np.float64)
    K = lg.shape[-1]
    own = lg[..., y] - np.logaddexp.reduce(lg[..., [k for k in range(K) if k != y]], axis=-1)
    pos, neg = contrast if contrast is not None else (K - 1, 0)
    return own, lg[..., pos] - lg[..., neg]


def estimate_ridge(Z: np.ndarray, d: np.ndarray, area: np.ndarray, lam0: float, frame_px: int):
    """Per-frame ridge of d on removal indicators (dual form, P >> N ok).

    Z [N,P] bool, d [N], area [P] cell areas (px), frame_px = H*W. The removed-area
    fraction a_m = sum_i z_mi area_i / frame_px is an UNPENALISED covariate (projected
    out, Frisch-Waugh): the ridge part `spec` is then area-weighted zero-sum (the generic
    per-pixel occlusion damage goes to gamma) and `total = spec + gamma*area/frame_px` is
    the estimate of the full removal effect. lam = lam0 * N * p(1-p) (scale-free).
    """
    N, P = Z.shape
    X = Z.astype(np.float64)
    Xc = X - X.mean(0)
    y = d - d.mean()
    A = ((X @ area) / frame_px)[:, None]
    A = A - A.mean(0)
    Q, _ = np.linalg.qr(A)
    Xt = Xc - Q @ (Q.T @ Xc)
    yt = y - Q @ (Q.T @ y)
    pbar = X.mean()
    lam = lam0 * N * max(pbar * (1 - pbar), 1e-6)
    if P < N:                                   # primal (identical solution)
        beta = np.linalg.solve(Xt.T @ Xt + lam * np.eye(P), Xt.T @ yt)
    else:                                       # dual
        beta = Xt.T @ np.linalg.solve(Xt @ Xt.T + lam * np.eye(N), yt)
    resid = y - Xc @ beta
    coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
    return {"spec": beta, "total": beta + coef[0] * area / frame_px}


# ── experiment data ─────────────────────────────────────────────────────────

class Exp:
    """Frames, masks, regionprops cells and fill parameters of one experiment."""

    def __init__(self, row, cfg: OcclusionConfig):
        self.exp, self.y, self.seed = str(row.experiment_id), int(row.y_true), cfg.seed
        self.all_ids = cfg.frames(self.exp, int(row.num_images))
        t0 = time.time()
        self.frames = read_stack(open_frames(row.path), self.all_ids)
        self.masks = read_stack(open_masks(row.path), self.all_ids)
        self.H, self.W = self.frames.shape[1:]
        self.pos = {int(i): j for j, i in enumerate(self.all_ids)}
        rp = load_regionprops(row.path, cfg.cell_config, self.all_ids)
        rp = rp[[c for c in RP_COLS if c in rp.columns]].copy()
        b = cfg.border_px
        rp["touches_border"] = (rp.ty < b) | (rp.tx < b) | (rp.by > self.H - b) | (rp.bx > self.W - b)
        self.rp = rp.sort_values(["image_id", "label"]).reset_index(drop=True)
        self.cell_idx = {int(k): v for k, v in self.rp.groupby("image_id").indices.items()}
        self.bg = np.load(cfg.path("bg", f"{self.exp}.npy"))
        self._ff, self._free = {}, {}      # per-frame fill model / cell-free map, lazily
        print(f"[{self.exp}] y={self.y}: read {len(self.all_ids)} frames+masks, "
              f"{len(self.rp)} cells in {time.time() - t0:.0f}s", flush=True)

    def fill_params(self, j):
        if j not in self._ff:
            self._ff[j] = O.frame_fill_model(self.frames[j], self.bg, O.cell_free(self.masks[j]),
                                             self.masks[j])
        return self._ff[j]

    def cells(self, iid):
        return self.rp.iloc[self.cell_idx.get(int(iid), np.zeros(0, np.int64))]

    def occ(self, iid, remove_labels, tag, mode="remove", extra=None):
        """mode 'remove': occlusion.occlude of remove_labels (empty -> the original).
        'sham': the same occlude call (same seed), then the removed cells' OWN mask
        pixels restored from the original -> only the halo/closing fill and seam-band
        edits on the neighbours remain. 'blob': extra = (dy, dx, src_label): the src
        cell's mask shape translated by (dy, dx) onto cell-free background is added as a
        new label and removed (a shape-matched fill-only null)."""
        j = self.pos[int(iid)]
        rng = _seed(self.seed, self.exp, iid, tag)
        if mode == "blob":
            dy, dx, src = extra
            m2 = self.masks[j].astype(np.int32, copy=True)
            ys, xs = np.nonzero(self.masks[j] == src)
            new = int(m2.max()) + 1
            m2[ys + dy, xs + dx] = new
            return O.occlude(self.frames[j], m2, np.array([new]), self.bg, rng,
                             noise_params=self.fill_params(j))
        if remove_labels is None or len(remove_labels) == 0:
            return self.frames[j]
        rl = np.asarray(remove_labels, np.int64)
        x = O.occlude(self.frames[j], self.masks[j], rl, self.bg, rng, noise_params=self.fill_params(j))
        if mode == "sham":
            own = np.isin(self.masks[j], rl)
            x = x.copy()
            x[own] = self.frames[j][own]
        return x

    def blob_offset(self, iid, src_label, tag, tries=400):
        """Random translation putting cell src_label's mask entirely on cell-free
        background (outside every dilated cell mask); None if not found."""
        j = self.pos[int(iid)]
        if j not in self._free:
            self._free[j] = O.cell_free(self.masks[j])
        free = self._free[j]
        ys, xs = np.nonzero(self.masks[j] == src_label)
        if len(ys) == 0:
            return None
        rng = _seed(self.seed, self.exp, iid, tag)
        for _ in range(tries):
            cy, cx = rng.integers(0, self.H), rng.integers(0, self.W)
            dy, dx = int(cy - ys.mean()), int(cx - xs.mean())
            yy, xx = ys + dy, xs + dx
            if yy.min() < 0 or xx.min() < 0 or yy.max() >= self.H or xx.max() >= self.W:
                continue
            if free[yy, xx].all():
                return dy, dx
        return None


class Encoder:
    """Scores frames built in a thread pool, in a fixed batch shape (last batch padded)."""

    def __init__(self, scorer: Scorer, pool, batch):
        self.s, self.pool, self.B, self.n_fwd = scorer, pool, batch, 0

    def _gpu(self, imgs, want_feats):
        import torch
        with torch.no_grad():
            n = len(imgs)
            imgs = imgs + [imgs[-1]] * (self.B - n)      # constant shape -> same cuDNN kernels
            x = self.s.normalize(torch.from_numpy(np.stack(imgs)).to(self.s.device))
            f = self.s.encode(x).float()
            lg = self.s.aggregate(f.unsqueeze(1)).float()   # B bags of one frame
            self.n_fwd += n
            return lg[:n].cpu().numpy(), (f[:n].cpu().numpy() if want_feats else None)

    def run(self, jobs, build, want_feats=False):
        """jobs: list; build(job) -> uint8 [H,W] (runs in the thread pool).
        Returns logits [n,K] (and feats [n,D])."""
        chunks = [jobs[s:s + self.B] for s in range(0, len(jobs), self.B)]
        q, nxt, lg_all, f_all = deque(), 0, [], []
        for ci in range(len(chunks)):
            while nxt < len(chunks) and nxt <= ci + LOOKAHEAD:
                q.append([self.pool.submit(build, jb) for jb in chunks[nxt]])
                nxt += 1
            lg, ft = self._gpu([f.result() for f in q.popleft()], want_feats)
            lg_all.append(lg)
            f_all.append(ft)
        return np.concatenate(lg_all), (np.concatenate(f_all) if want_feats else None)

    def baseline(self, E, ids):
        """Logits and features of the unoccluded frames ``ids``."""
        return self.run([(i, None, 0) for i in ids], lambda jb: E.occ(*jb), want_feats=True)


# ── stage 1: random-subset screen ───────────────────────────────────────────

def stage1(cfg, E, enc):
    """Random-subset screen on F frames: N masks, each cell removed w.p. p."""
    rng = np.random.default_rng([E.seed, int(E.exp) % 2**31, 7])
    F = min(cfg.s1_frames, len(E.all_ids))
    fids = np.sort(rng.choice(E.all_ids, F, replace=False))
    base_lg, _ = enc.baseline(E, fids)
    jobs, Zs = [], []
    for iid in fids:
        labels = E.cells(iid).label.to_numpy()
        Z = _seed(E.seed, E.exp, iid, T_S1_MASK).random((cfg.n_masks, len(labels))) < cfg.p
        Zs.append(Z)
        jobs += [(iid, labels[Z[m]], T_S1_OCC + m) for m in range(cfg.n_masks)]
    t0 = time.time()
    lg, _ = enc.run(jobs, lambda jb: E.occ(*jb))
    print(f"[{E.exp}] stage1: {len(jobs)} forwards in {time.time() - t0:.0f}s", flush=True)
    lg = lg.reshape(F, cfg.n_masks, -1)
    own0, con0 = targets(base_lg, E.y, cfg.contrast)
    own, _ = targets(lg, E.y, cfg.contrast)
    recs = []
    for f, iid in enumerate(fids):
        cl = E.cells(iid).copy()
        r = estimate_ridge(Zs[f], own[f] - own0[f], cl.area.to_numpy(np.float64), cfg.lam,
                           E.H * E.W)
        cl["s1_drop_own"] = -r["total"]               # + = removal lowers true-class log-odds
        cl["s1_drop_own_spec"] = -r["spec"]           # beyond generic per-area damage
        cl["frame_own0"], cl["frame_con0"] = own0[f], con0[f]
        cl["frame_correct"] = bool(base_lg[f].argmax() == E.y)
        recs.append(cl)
    cells = pd.concat(recs, ignore_index=True)
    cells.insert(0, "experiment_id", E.exp)
    cells.insert(1, "y_true", E.y)
    return cells


# ── stage 2: exact single-cell removal ──────────────────────────────────────

def bag_deltas(scorer, base_feats, rows_feats, rows_pos, y, contrast):
    """Bag (all screened frames) targets with row rows_pos[i] replaced by rows_feats[i].
    Returns (own [n], con [n], own0, con0)."""
    import torch
    dev = scorer.device
    with torch.no_grad():
        bf = torch.from_numpy(base_feats).to(dev)
        lg0 = scorer.aggregate(bf.unsqueeze(0)).float().cpu().numpy()
        out = []
        for s in range(0, len(rows_pos), 32):
            fe = torch.from_numpy(rows_feats[s:s + 32]).to(dev)
            bags = bf.unsqueeze(0).repeat(len(fe), 1, 1)
            bags[torch.arange(len(fe), device=dev), torch.as_tensor(rows_pos[s:s + 32], device=dev)] = fe
            out.append(scorer.aggregate(bags).float().cpu().numpy())
    o0, c0 = targets(lg0, y, contrast)
    o, c = targets(np.concatenate(out), y, contrast)
    return o, c, float(o0[0]), float(c0[0])


def exact_block(cfg, E, enc, cand, base_lg, base_ft):
    """Exact single-cell removal (2 fill seeds) + sham + n_null nulls for every row of
    cand (image_id, label, area); frame and bag deltas. Null = equal-area random cells
    of the null cell type; cells much smaller than those: the cell's own shape
    translated onto background ("blob")."""
    n_null = cfg.n_null
    own0, con0 = targets(base_lg, E.y, cfg.contrast)
    jobs, meta, null_kind = [], [], []
    for ci, r in enumerate(cand.itertuples()):
        cl = E.cells(r.image_id)
        ref = cl[(cl.cell_type == cfg.null_cell_type) & (cl.label != r.label)]
        tg0, tn = T_S2_EXACT + int(r.label) * 10, T_S2_NULL + int(r.label) * 10
        small = len(ref) == 0 or float(r.area) < 0.5 * float(ref.area.median())
        blobs = []
        for k in range(max(2, n_null if small else 2)):
            off = E.blob_offset(r.image_id, int(r.label), tn + 5 + k)
            if off is not None:
                blobs.append((k, off))
        null_kind.append(("shape_matched_background" if blobs else "none") if small
                         else f"equal_area_{cfg.null_cell_type}")
        for s in range(2):
            jobs.append((r.image_id, [r.label], tg0 + s)); meta.append((ci, 0, s))
        jobs.append((r.image_id, [r.label], tg0, "sham")); meta.append((ci, 2, 0))
        for k, (dy, dx) in blobs:
            jobs.append((r.image_id, [], tn + 5 + k, "blob", (dy, dx, int(r.label)))); meta.append((ci, 3, k))
        if small:
            continue                      # primary null = blobs (meta kind 3)
        for k in range(n_null):
            lab = O.pick_area_matched(_seed(E.seed, E.exp, r.image_id, tn + k), ref.label.to_numpy(),
                                      ref.area.to_numpy(np.float64), float(r.area))
            jobs.append((r.image_id, lab, tn + k)); meta.append((ci, 1, k))
    lg, ft = enc.run(jobs, lambda jb: E.occ(*jb), want_feats=True)
    meta = np.array(meta)
    pos = np.array([E.pos[int(cand.image_id[m[0]])] for m in meta])
    bo, bc, bo0, bc0 = bag_deltas(enc.s, base_ft, ft, pos, E.y, cfg.contrast)
    o, c = targets(lg, E.y, cfg.contrast)
    nmax = max(n_null, 2)

    def grab(v, kind, width):
        out = np.full((len(cand), width), np.nan)
        for k, m in enumerate(meta):
            if m[1] == kind and m[2] < width:
                out[m[0], m[2]] = v[k]
        return out
    small = np.array([nk in ("shape_matched_background", "none") for nk in null_kind])
    for col, v in [("d_own", o - own0[pos]), ("d_con", c - con0[pos]),
                   ("bag_d_own", bo - bo0), ("bag_d_con", bc - bc0)]:
        ex, rb, bl = grab(v, 0, 2), grab(v, 1, n_null), grab(v, 3, nmax)
        nu = np.where(small[:, None], np.pad(bl, ((0, 0), (0, max(0, n_null - nmax))),
                                            constant_values=np.nan)[:, :n_null], rb)
        for k in range(nu.shape[1]):
            cand[f"null_{col}_{k}"] = nu[:, k]
        cand[f"ex_{col}"] = ex.mean(1)
        cand[f"ex_{col}_seedsd"] = np.abs(ex[:, 0] - ex[:, 1]) / np.sqrt(2)
        with np.errstate(all="ignore"):
            cand[f"null_{col}_mean"] = np.nanmean(nu, 1)
            cand[f"null_{col}_sd"] = np.nanstd(nu, 1, ddof=1)
            cand[f"sham_{col}"] = grab(v, 2, 1)[:, 0]
            cand[f"blob_{col}_mean"] = np.nanmean(bl, 1)
            cand[f"ex_{col}_minus_sham"] = cand[f"ex_{col}"] - cand[f"sham_{col}"]
    cand["null_kind"] = null_kind
    cand["ex_drop_own"] = -cand.ex_d_own
    cand["ex_drop_own_vs_null"] = -(cand.ex_d_own - cand.null_d_own_mean)
    cand["ex_bag_drop_own"] = -cand.ex_bag_d_own
    cand["ex_bag_drop_own_vs_null"] = -(cand.ex_bag_d_own - cand.null_bag_d_own_mean)
    fp = np.array([E.pos[int(i)] for i in cand.image_id])
    cand["frame_own0"], cand["frame_con0"] = own0[fp], con0[fp]
    cand["frame_correct"] = (base_lg.argmax(-1) == E.y)[fp]
    cand["p_true0"] = np.exp(base_lg[fp, E.y] - np.logaddexp.reduce(base_lg[fp], axis=1))
    cand["bag_own0"], cand["bag_con0"] = bo0, bc0
    cand.insert(0, "experiment_id", E.exp)
    cand.insert(1, "y_true", E.y)
    print(f"[{E.exp}] stage2: {len(cand)} cells, {len(jobs)} forwards", flush=True)
    return cand


def stage2(cfg, E, enc, s1, base_lg, base_ft):
    """Candidates (stage-1 top, Grad-CAM top, random controls), exact-removed."""
    own0, _ = targets(base_lg, E.y, cfg.contrast)
    correct = base_lg.argmax(-1) == E.y
    s1 = s1.copy()
    s1["eligible"] = ~s1.touches_border & s1.frame_correct & (s1.frame_own0 <= SAT)
    rng = np.random.default_rng([E.seed, int(E.exp) % 2**31, 21])
    picks = {}

    def add(df, n, why):
        for r in df.head(n).itertuples():
            picks.setdefault((int(r.image_id), int(r.label)), []).append(why)
    add(s1.sort_values("s1_drop_own", ascending=False), 30, "s1_top")
    add(s1[s1.eligible].sort_values("s1_drop_own", ascending=False), 20, "s1_top_elig")
    add(s1.sort_values("s1_drop_own_spec", ascending=False), 15, "s1_top_spec")
    add(s1[~s1.touches_border & s1.frame_correct].sort_values("s1_drop_own", ascending=False),
        10, "s1_top_correct")
    attr = pd.read_parquet(cfg.path("attribution", f"{E.exp}.parquet"),
                           columns=["image_id", "label", "own_l4_sum"])
    attr = attr.merge(E.rp[["image_id", "label", "touches_border"]], on=["image_id", "label"])
    fpos = [E.pos[int(i)] for i in attr.image_id]
    attr["frame_correct"], attr["frame_unsat"] = correct[fpos], own0[fpos] <= SAT
    add(attr.sort_values("own_l4_sum", ascending=False), 10, "gc_top")
    add(attr[~attr.touches_border & attr.frame_correct & attr.frame_unsat].sort_values("own_l4_sum", ascending=False),
        5, "gc_top_elig")
    add(s1.sample(n=min(20, len(s1)), random_state=int(rng.integers(1 << 31))), 20, "random")
    w1 = s1[s1.cell_type == cfg.focus_cell_type]
    add(w1.sample(n=min(10, len(w1)), random_state=int(rng.integers(1 << 31))), 10,
        f"random_{cfg.focus_cell_type}")
    cand = pd.DataFrame([(k[0], k[1], ";".join(v)) for k, v in picks.items()],
                        columns=["image_id", "label", "why"])
    return exact_block(cfg, E, enc, cand.merge(E.rp, on=["image_id", "label"], how="left"),
                       base_lg, base_ft)


def stage2_random(cfg, E, enc, s1, base_lg, base_ft):
    """Unbiased per-type sample: n_random_per_type cells of EACH type uniformly at
    random from the stage-1 frames, exact-removed with the same protocol."""
    rng = np.random.default_rng([E.seed, int(E.exp) % 2**31, 23])
    parts = []
    for t in cfg.random_cell_types:
        pool = s1[s1.cell_type == t]
        n = min(cfg.n_random_per_type, len(pool))
        if n:
            parts.append(pool.sample(n=n, random_state=int(rng.integers(1 << 31)))[
                ["image_id", "label"]].assign(random_stratum=t))
    cand = pd.concat(parts, ignore_index=True).merge(E.rp, on=["image_id", "label"], how="left")
    return exact_block(cfg, E, enc, cand.reset_index(drop=True), base_lg, base_ft)


# ── stages ──────────────────────────────────────────────────────────────────

def run_background(cfg: OcclusionConfig, experiments: pd.DataFrame, overwrite=False):
    """Empty-slide background model of every experiment (CPU)."""
    for row in _todo(cfg, experiments, "bg", [".npy"], overwrite).itertuples(index=False):
        ids = cfg.frames(row.experiment_id, int(row.num_images))
        b, frac = O.build_bg_model(read_stack(open_frames(row.path), ids),
                                   read_stack(open_masks(row.path), ids))
        dst = cfg.out_path("bg", f"{row.experiment_id}.npy")
        np.save(dst[:-4] + ".tmp.npy", b)
        os.replace(dst[:-4] + ".tmp.npy", dst)
        print(f"[{row.experiment_id}] bg model from {len(ids)} frames (smooth-filled px "
              f"{frac:.4f})", flush=True)


def _prefetch(frames_arr, masks_arr, batches, q: queue.Queue):
    """Single reader thread: serial reads per zarr handle."""
    try:
        for ids in batches:
            q.put((ids, read_stack(frames_arr, ids), read_stack(masks_arr, ids)))
    except Exception as e:  # surface to the main thread
        q.put(e)
    q.put(None)


def run_attribution(cfg: OcclusionConfig, experiments: pd.DataFrame, batch=8, overwrite=False):
    """Signed Grad-CAM (no ReLU) of the true-class log-odds at layer4[-1], each sampled
    frame as a bag of ONE frame:
      cam[h, w] = sum_c mean_hw(dT/dA_c) * A_c[h, w]
    upsampled bilinearly to the frame and scaled by (h*w)/(H*W) (so the frame sum equals
    the low-res CAM sum); per cell (mask pixels of regionprops `label`): own_l4_sum."""
    import torch
    import torch.nn.functional as F
    from remo_id.interpret.core import get_target_layer_resnet
    todo = _todo(cfg, experiments, "attribution", [".parquet"], overwrite)
    if todo.empty:
        return
    torch.backends.cudnn.allow_tf32 = True
    sc = Scorer(cfg.checkpoint, cfg.device)
    dev = sc.device
    for p in sc.model.parameters():
        p.requires_grad_(False)
    layer = get_target_layer_resnet(sc.model, "layer4")
    if layer is None:
        raise ValueError(f"{cfg.checkpoint}: attribution needs a ResNet-backbone MIL model")
    for row in todo.itertuples(index=False):
        t0, y = time.time(), int(row.y_true)
        ids = cfg.frames(row.experiment_id, int(row.num_images))
        rp_key = load_regionprops(row.path, cfg.cell_config, ids).set_index(["image_id", "label"]).index
        frames_arr, masks_arr = open_frames(row.path), open_masks(row.path)
        Hf, Wf = frames_arr.shape[0], frames_arr.shape[1]
        q: queue.Queue = queue.Queue(maxsize=3)
        th = threading.Thread(target=_prefetch, daemon=True, args=(
            frames_arr, masks_arr, [ids[i:i + batch] for i in range(0, len(ids), batch)], q))
        th.start()
        acts, recs = {}, []
        hd = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
        try:
            while (item := q.get()) is not None:
                if isinstance(item, Exception):
                    raise item
                bids, fr, mk = item
                x = sc.normalize(fr).to(dev).requires_grad_(True)
                with torch.enable_grad(), torch.autocast(str(dev).split(":")[0], enabled=False):
                    feats = sc.encode(x)
                    logits = torch.cat([sc.aggregate(feats[i:i + 1].unsqueeze(0))
                                        for i in range(feats.shape[0])], dim=0)
                    others = [k for k in range(logits.shape[1]) if k != y]
                    own = logits[:, y] - torch.logsumexp(logits[:, others], dim=1)
                    a = acts["a"]
                    g, = torch.autograd.grad(own.sum(), [a])
                    h, w = a.shape[-2:]
                    cam = (g.mean(dim=(2, 3), keepdim=True) * a).sum(dim=1, keepdim=True)
                    cam = F.interpolate(cam, size=(Hf, Wf), mode="bilinear",
                                        align_corners=False)[:, 0] * (h * w / (Hf * Wf))
                labels = torch.from_numpy(mk.astype(np.int64)).to(dev)
                B, L = labels.shape[0], int(labels.max().item()) + 1
                idx = (labels + torch.arange(B, device=dev).view(B, 1, 1) * L).reshape(-1)
                cnt = torch.bincount(idx, minlength=B * L).view(B, L).cpu().numpy()
                sums = torch.bincount(idx, weights=cam.detach().reshape(-1).double(),
                                      minlength=B * L).view(B, L).cpu().numpy()
                for b, iid in enumerate(bids):
                    lab = np.nonzero(cnt[b])[0]
                    lab = lab[lab > 0]
                    lab = lab[pd.MultiIndex.from_arrays([np.full(len(lab), int(iid)), lab]).isin(rp_key)]
                    recs.append(pd.DataFrame({"image_id": int(iid), "label": lab,
                                              "own_l4_sum": sums[b, lab]}))
                acts.clear()
        finally:
            hd.remove()
            th.join(timeout=5)
        save_parquet(pd.concat(recs, ignore_index=True),
                     cfg.out_path("attribution", f"{row.experiment_id}.parquet"))
        print(f"[{row.experiment_id}] attribution: {sum(map(len, recs))} cells, "
              f"{time.time() - t0:.0f}s", flush=True)


def run_screen(cfg: OcclusionConfig, experiments: pd.DataFrame, overwrite=False):
    """Stage 1 + stage 2 (+ random cells) of every experiment (needs background and
    attribution)."""
    import torch
    todo = _todo(cfg, experiments, "screen/stage2", [".parquet", ".random.parquet"], overwrite)
    if todo.empty:
        return
    cfg.resolve_cell_types()
    torch.backends.cudnn.allow_tf32 = True    # PyTorch default; matmul TF32 left at default
    torch.backends.cudnn.benchmark = False
    scorer = Scorer(cfg.checkpoint, cfg.device)
    if str(scorer.device).startswith("cuda"):
        print("GPU", torch.cuda.get_device_name(), cfg.checkpoint, flush=True)
    with ThreadPoolExecutor(cfg.workers) as pool:
        enc = Encoder(scorer, pool, cfg.gpu_batch)
        for row in todo.itertuples(index=False):
            t0 = time.time()
            E = Exp(row, cfg)
            dst1 = cfg.out_path("screen/stage1", f"{E.exp}.cells.parquet")
            if os.path.exists(dst1) and not overwrite:
                s1 = pd.read_parquet(dst1)
            else:
                s1 = stage1(cfg, E, enc)
                save_parquet(s1, dst1)
            for suffix, fn in ((".parquet", stage2), (".random.parquet", stage2_random)):
                base_lg, base_ft = enc.baseline(E, E.all_ids)   # all screened frames, in-job
                save_parquet(fn(cfg, E, enc, s1, base_lg, base_ft),
                             cfg.out_path("screen/stage2", E.exp + suffix))
            print(f"[{E.exp}] all stages {time.time() - t0:.0f}s; encoder {enc.n_fwd} fwd",
                  flush=True)


# ── report (CPU) ────────────────────────────────────────────────────────────

def _read(files):
    fs = sorted(f for f in files if ".tmp" not in f)
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    if len(d):
        d["experiment_id"] = d["experiment_id"].astype(str)
    return d


def run_report(work_dir: str, out_dir: str, class_names=None):
    """screen_cells.parquet: every cell with an EXACT (stage-2) effect, stage-1
    regionprops/frame info joined; screen_top_manifest.csv: per class the top cells by
    the drop attributable to the cell's OWN pixels (remove - sham), requiring the
    removal to also beat its null; non-border, frame correct, unsaturated first,
    <= 2 per experiment, <= 40 per class. ``class_names`` (index order) overrides the
    class-config names in the ``class`` column."""
    os.makedirs(out_dir, exist_ok=True)
    ex = recorded_experiments(work_dir)
    path_of = dict(zip(ex.experiment_id, ex.path))
    names = dict(enumerate(class_names)) if class_names else dict(zip(ex.y_true, ex.class_name))
    s2dir = os.path.join(work_dir, "screen", "stage2")
    s1 = _read(glob.glob(os.path.join(work_dir, "screen", "stage1", "*.cells.parquet")))
    s2 = _read(f for f in glob.glob(os.path.join(s2dir, "*.parquet"))
               if os.path.basename(f).count(".") == 1)            # {exp}.parquet only
    rnd = _read(glob.glob(os.path.join(s2dir, "*.random.parquet")))
    if len(rnd):        # random cells: tag, merge into s2 (a cell already there keeps its row)
        rnd["why"] = "random_type_" + rnd.random_stratum
        rk = set(zip(rnd.experiment_id, rnd.image_id, rnd.label))
        dup = np.array([k in rk for k in zip(s2.experiment_id, s2.image_id, s2.label)])
        s2.loc[dup, "why"] = s2.loc[dup, "why"] + ";random_type"
        s2 = pd.concat([s2, rnd], ignore_index=True).drop_duplicates(
            ["experiment_id", "image_id", "label"], keep="first").reset_index(drop=True)

    # screen_cells.parquet (stage-2 cells)
    ex_cols = [c for c in s2.columns if c.startswith(("ex_", "null_"))] + ["why", "p_true0"]
    key = ["experiment_id", "image_id", "label"]
    cells = s1.merge(s2[key + ex_cols], on=key, how="outer", indicator="has")
    cells = cells[cells.has != "left_only"]
    only2 = cells.has == "right_only"
    cells = cells.drop(columns="has")
    meta_cols = [c for c in s2.columns if c not in ex_cols and c not in key and c in cells.columns]
    if only2.any():      # metadata for Grad-CAM candidates on frames stage 1 did not screen
        s2m = s2.set_index(key)[meta_cols]
        idx = pd.MultiIndex.from_frame(cells.loc[only2, key])
        for c in meta_cols:
            cells.loc[only2, c] = s2m[c].reindex(idx).to_numpy()
    cells["class"] = cells.y_true.map(names)
    cells["path"] = cells.experiment_id.map(path_of)
    cells.to_parquet(os.path.join(out_dir, "screen_cells.parquet"), index=False)

    # top manifest
    s2x = s2.copy()
    s2x["path"] = s2x.experiment_id.map(path_of)
    s2x["class"] = s2x.y_true.map(names)
    s2x["z_vs_null"] = (s2x.ex_drop_own_vs_null
                        / np.sqrt(s2x.null_d_own_sd ** 2 + s2x.ex_d_own_seedsd ** 2 + 1e-12))
    s2x["saturated"] = s2x.p_true0 > 0.999
    s2x["drop_own_minus_sham"] = -s2x.ex_d_own_minus_sham
    elig = s2x[~s2x.touches_border.astype(bool) & s2x.frame_correct.astype(bool)]
    man = []
    for y, g in elig.groupby("y_true"):
        g = g[(g.ex_drop_own_vs_null > 0) & (g.drop_own_minus_sham > 0)].sort_values(
            ["saturated", "drop_own_minus_sham"], ascending=[True, False])
        g = g.groupby("experiment_id", group_keys=False).head(2).head(40).copy()
        g.insert(0, "rank", np.arange(1, len(g) + 1))
        man.append(g)
    man = pd.concat(man, ignore_index=True)
    mcols = ["rank", "class", "experiment_id", "image_id", "label", "cell_type",
             "centroid_y", "centroid_x", "ty", "tx", "by", "bx", "area", "p_true0", "path",
             "ex_drop_own", "ex_drop_own_vs_null", "z_vs_null", "null_d_own_mean", "null_d_own_sd",
             "ex_d_own_seedsd", "ex_d_con", "null_d_con_mean", "ex_bag_drop_own",
             "ex_bag_drop_own_vs_null", "ex_bag_d_con", "frame_own0", "frame_con0", "bag_own0",
             "intensity_mean", "eccentricity", "solidity", "saturated", "sham_d_own",
             "ex_d_own_minus_sham", "drop_own_minus_sham", "blob_d_own_mean", "null_kind", "why"]
    man = man[[c for c in mcols if c in man.columns]].rename(columns={"p_true0": "p_true"})
    man.to_csv(os.path.join(out_dir, "screen_top_manifest.csv"), index=False)
    print(f"{len(cells)} stage-2 cells; manifest {man.groupby('class').size().to_dict()}",
          flush=True)


# ── selection rules for figures ─────────────────────────────────────────────

def _tie_key(df: pd.DataFrame, keys, seed: int) -> pd.Series:
    """Seeded random tie-break key: rows are first put in a canonical order by ``keys``."""
    order = df.sort_values(keys).index
    return pd.Series(np.random.default_rng(seed).random(len(order)), index=order).reindex(df.index)


def select_informative_cells(cells: pd.DataFrame, n: int = 10, type_quota: dict = None,
                             seed: int = 403) -> pd.DataFrame:
    """Per class, the cells whose own pixels lower the true-class log-odds most.

    Eligible stage-2 cells (``screen_cells.parquet`` rows): not touching the frame
    border, frame correctly classified at baseline, removal beats its null
    (``ex_drop_own_vs_null > 0``) and own-pixel drop (remove - sham) > 0. Ranked by
    ``drop_own_minus_sham`` (ties broken by a seeded random key), at most one cell per
    experiment. ``type_quota`` (e.g. {"wbc": 5, "rbc": 5}) takes the top cells of each
    type up to its quota; free slots then go to the next-ranked cells of any type.
    Returns the picks with ``slot`` (0-based: quota types in quota order, then the
    rest, each by effect).
    """
    key = ["experiment_id", "image_id", "label"]
    c = cells[cells["ex_d_own_minus_sham"].notna()].drop_duplicates(key).copy()
    c["drop_own_minus_sham"] = -c["ex_d_own_minus_sham"]
    c = c[~c["touches_border"].astype(bool) & c["frame_correct"].astype(bool)
          & (c["ex_drop_own_vs_null"] > 0) & (c["drop_own_minus_sham"] > 0)]
    c["_tie"] = _tie_key(c, key, seed)
    c = c.sort_values(["drop_own_minus_sham", "_tie"], ascending=[False, True])
    quota, out = dict(type_quota or {}), []
    for _, g in c.groupby("y_true", sort=True):
        used, picks, left = set(), [], dict(quota)
        for by_quota in ([True, False] if quota else [False]):
            for r in g.itertuples():
                if len(picks) >= n:
                    break
                if r.experiment_id in used or (by_quota and left.get(r.cell_type, 0) <= 0):
                    continue
                used.add(r.experiment_id)
                picks.append(r.Index)
                if by_quota:
                    left[r.cell_type] -= 1
        sel = g.loc[picks].copy()
        sel["_t"] = sel["cell_type"].map({t: i for i, t in enumerate(quota)}).fillna(len(quota))
        sel = sel.sort_values(["_t", "drop_own_minus_sham", "_tie"], ascending=[True, False, True])
        sel["slot"] = np.arange(len(sel))
        out.append(sel.drop(columns=["_t", "_tie"]))
    return pd.concat(out, ignore_index=True)


def laplacian_variance(img) -> float:
    """Focus measure: variance of the Laplacian of a frame."""
    import cv2
    return float(cv2.Laplacian(img.astype(np.float32), cv2.CV_32F).var())


def select_example_frames(work_dir: str, classes, experiments=None, seed: int = 403,
                          p_range=(0.6, 0.99), max_cover=0.55, focus_quantile=0.75,
                          n_shortlist=30, n_ref=30, effect="s1_drop_own") -> pd.DataFrame:
    """One representative stage-1 frame per class (rule, no hand picking):

    the frame is correctly classified with ``p_range[0] < P(true) < p_range[1]``, its
    cells cover at most ``max_cover`` of the frame (not overcrowded), and it is ranked by
    the total |effect| of its cells (stage-1 estimate); one candidate per experiment;
    the top ``n_shortlist`` candidates per class are checked for focus (Laplacian
    variance at or above the ``focus_quantile`` of ``n_ref`` seeded reference frames of
    the same experiment's stage-1 frames) and the best in-focus candidate is rank 1.
    ``experiments`` restricts the experiments considered (e.g. a scope filter); ties are
    broken by a seeded random key. Returns every shortlisted candidate with
    ``cand_rank`` (1 = chosen) and the focus columns.
    """
    path = recorded_experiments(work_dir).set_index("experiment_id")["path"]
    lo, hi = np.log(p_range[0] / (1 - p_range[0])), np.log(p_range[1] / (1 - p_range[1]))
    rows, s1_ids = [], {}
    for f in sorted(glob.glob(os.path.join(work_dir, "screen", "stage1", "*.cells.parquet"))):
        t = pd.read_parquet(f, columns=["experiment_id", "y_true", "image_id", "area", effect,
                                        "frame_own0", "frame_correct"])
        e, y = str(t["experiment_id"].iloc[0]), int(t["y_true"].iloc[0])
        if y not in classes or (experiments is not None and e not in set(experiments)):
            continue
        s1_ids[e] = t["image_id"].unique()
        H, W = open_frames(path[e]).shape[:2]
        g = t.groupby("image_id").agg(score=(effect, lambda v: float(np.abs(v).sum())),
                                      area=("area", "sum"), own0=("frame_own0", "first"),
                                      correct=("frame_correct", "first"),
                                      n_cells=("area", "size")).reset_index()
        g["cover"] = g["area"] / (H * W)
        rows.append(g[g["correct"].astype(bool) & (g["own0"] > lo) & (g["own0"] < hi)
                      & (g["cover"] <= max_cover)].assign(experiment_id=e, y_true=y))
    cand = pd.concat(rows, ignore_index=True)
    cand["p_true"] = 1 / (1 + np.exp(-cand["own0"]))
    cand["_tie"] = _tie_key(cand, ["experiment_id", "image_id"], seed)
    cand = cand.sort_values(["score", "_tie"], ascending=[False, True]).drop_duplicates("experiment_id")
    short = cand.groupby("y_true", group_keys=False).head(n_shortlist).copy()
    pct = []
    for r in short.itertuples():
        fr = open_frames(path[r.experiment_id])
        ref_ids = np.random.default_rng([seed, int(r.experiment_id) % 2**31]).choice(
            s1_ids[r.experiment_id], min(n_ref, len(s1_ids[r.experiment_id])), replace=False)
        ref = np.array([laplacian_variance(read_stack(fr, [i])[0]) for i in ref_ids])
        pct.append(float((ref < laplacian_variance(read_stack(fr, [r.image_id])[0])).mean()))
    short["lapvar_pct"] = pct
    short["in_focus"] = short["lapvar_pct"] >= focus_quantile
    short = short.sort_values(["y_true", "score", "_tie"], ascending=[True, False, True])
    foc = short["in_focus"]
    short["cand_rank"] = np.nan
    short.loc[foc, "cand_rank"] = short[foc].groupby("y_true").cumcount() + 1
    return short.drop(columns="_tie").reset_index(drop=True)

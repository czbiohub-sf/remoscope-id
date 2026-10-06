#!/usr/bin/env python3
"""Per-image UMAP for the bottom row of roc_confusion_umap: one point per whole frame
(not per cell).

Projects every frame of every experiment of one scope (default the lfm-ohmu scope,
the only scope with all three classes), taken from ONE fold model's per-image
embeddings (``remo-id workflow loo-per-image --fold_idx 0 --all_experiments``):
PCA -> 100 (randomized, seeded), then UMAP. Never pool embeddings from different
fold models. Writes ``umap_all_{scope}.npz`` (coords, cls, exp, imgid,
class_names), read by make_paper_results.py --umap_npz.

UMAP runs multithreaded without a fixed random_state, so a rerun gives an
equivalent but not bit-identical layout.

    python examples/sepsis_paper/umap_images.py --embeddings_dir <cv_dir>/per_image_fold0_all \\
        --manifest manifest.csv --output_dir <cv_dir>/full_umap_ohmu
"""
import argparse
import glob
import os
import time

import numpy as np

from common import Manifest


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_embeddings(embeddings_dir, scope_of, scope_substr):
    """Per-image embeddings of the scope-matching experiments."""
    files = sorted(glob.glob(os.path.join(embeddings_dir, "*.embeddings.npz")))
    Xs, clss, exps, imgids = [], [], [], []
    class_names = None
    for f in files:
        e = os.path.basename(f).split(".")[0]
        if scope_substr.lower() not in scope_of.get(e, "").lower():
            continue
        d = np.load(f, allow_pickle=True)
        emb = d["embeddings"].astype(np.float32)
        Xs.append(emb)
        clss.append(np.full(len(emb), int(d["y_true"]), np.int64))
        exps.append(np.full(len(emb), e, dtype=object))
        imgids.append(d["image_id"].astype(np.int64))
        if class_names is None:
            class_names = [str(c) for c in d["class_names"]]
    if not Xs:
        raise SystemExit(f"no {scope_substr} experiments in {embeddings_dir}")
    X = np.concatenate(Xs, axis=0)
    log(f"{len(Xs)} {scope_substr} experiments, {len(X)} images")
    return (X, np.concatenate(clss), np.concatenate(exps), np.concatenate(imgids),
            class_names)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings_dir", required=True,
                   help="loo-per-image --fold_idx 0 --all_experiments output")
    p.add_argument("--manifest", required=True, help="manifest CSV (scope column)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--scope", default="ohmu")
    p.add_argument("--pca", type=int, default=100)
    p.add_argument("--n_neighbors", type=int, default=30)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=403)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    scope_of = Manifest(args.manifest).scope_map()
    X, cls, exp, imgid, class_names = load_embeddings(args.embeddings_dir, scope_of,
                                                      args.scope)

    if args.pca and args.pca < X.shape[1]:
        from sklearn.decomposition import PCA
        log(f"PCA -> {args.pca} ...")
        X = PCA(n_components=args.pca, svd_solver="randomized",
                random_state=args.seed).fit_transform(X)

    import umap
    log("fitting UMAP (n_jobs=-1) ...")
    coords = umap.UMAP(n_components=2, n_neighbors=args.n_neighbors,
                       min_dist=args.min_dist, random_state=None, n_jobs=-1,
                       verbose=True).fit_transform(X)
    out = os.path.join(args.output_dir, f"umap_all_{args.scope}.npz")
    np.savez_compressed(out, coords=coords.astype(np.float32), cls=cls,
                        exp=exp.astype(str), imgid=imgid,
                        class_names=np.array(class_names))
    log(f"saved {out}")


if __name__ == "__main__":
    main()

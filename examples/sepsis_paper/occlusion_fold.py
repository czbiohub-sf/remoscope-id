#!/usr/bin/env python3
"""Run one ``remo-id analysis occlusion`` stage on the held-out experiments of a LOO fold.

``remo-id analysis occlusion`` scores experiments with ONE checkpoint. To explain every
experiment by the fold model that did not train on it, this maps fold K to its held-out
experiments (``repeat0_foldK/splits.yaml``, ``val``) and its checkpoint at ``--epoch``,
then runs the stage with ``--checkpoint <fold ckpt> --experiment_ids <held-out ids>``.
Every argument after the stage name is passed through, including the data source:
``--manifest manifest.csv --classes ICU-ctrl,healthy,sepsis`` (no database) or
``--class_config`` (default: the fold's classes.yaml). Use one SLURM array task per fold
(``--fold`` defaults to ``$SLURM_ARRAY_TASK_ID``); all folds share one ``--work_dir``:

    sbatch --array=0-46 --gpus-per-node=1 --mem=48G scripts/submit_cmd.sh \\
        python examples/sepsis_paper/occlusion_fold.py --cv_dir $CV --epoch 50 \\
        screen --work_dir $CV/occlusion --manifest manifest.csv --classes ICU-ctrl,healthy,sepsis \\
        --cell_config examples/sepsis_paper/configs/wbc_rbc_plt.yaml
"""
import argparse
import os

import yaml

from common import fold_checkpoint
from remo_id.cli import cli
from remo_id.inference.context import resolve_split_experiment_ids


def fold_task(cv_dir, fold, epoch):
    """(checkpoint at ``epoch``, held-out experiment IDs) of ``repeat0_fold{fold}``."""
    d = os.path.join(cv_dir, f"repeat0_fold{fold}")
    with open(os.path.join(d, "splits.yaml")) as fh:
        return fold_checkpoint(d, epoch), resolve_split_experiment_ids(yaml.safe_load(fh), "val")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cv_dir", required=True, help="LOO CV folder")
    p.add_argument("--epoch", type=int, required=True, help="checkpoint epoch of every fold")
    p.add_argument("--fold", type=int, default=os.environ.get("SLURM_ARRAY_TASK_ID"),
                   help="fold number K of repeat0_foldK (default $SLURM_ARRAY_TASK_ID)")
    p.add_argument("stage", choices=["background", "attribution", "screen"])
    p.add_argument("stage_args", nargs=argparse.REMAINDER,
                   help="passed to `remo-id analysis occlusion <stage>`")
    args = p.parse_args()
    if args.fold is None:
        p.error("--fold is required outside a SLURM array task")
    ckpt, ids = fold_task(args.cv_dir, int(args.fold), args.epoch)
    print(f"fold {args.fold}: {len(ids)} held-out experiment(s), {ckpt}", flush=True)
    cli.main(args=["analysis", "occlusion", args.stage, "--checkpoint", ckpt,
                   *[a for e in ids for a in ("--experiment_ids", e)], *args.stage_args])


if __name__ == "__main__":
    main()

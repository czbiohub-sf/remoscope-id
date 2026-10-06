#!/usr/bin/env python
"""Generate a MIL hyperparameter sweep as a SLURM array.

Enumerates a config table (one row per training run), guarantees the
effective-batch / accumulation math, writes the table as a TSV plus an sbatch
array script, and copies the validation-set file into the sweep directory. It
does NOT submit anything: it prints the `sbatch` command to run.

Grid (edit `build_grid` to change it):
  a  model-config grid : 3 backbones x {attention, linear}, bag 50, lr 1e-4, pretrained
  b  bag-size scout    : 3 backbones x attention x bag {10, 100}, lr 1e-4, pretrained
  c  lr scout          : 3 backbones x attention x lr {1e-5, 1e-3}, bag 50, pretrained
  d  init scout        : {resnet18, resnet50} x attention, random init (train whole net)

All runs share: effective batch 600 (batch_size 200 x accumulate 3), no early
stopping, checkpoints at SAVE_EPOCHS. Each run is repeated once per named
validation set in --valsets, a JSON file mapping a set name to a list of
experiment IDs held out for validation.
"""

import argparse
import csv
import json
import os
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKBONES = ["resnet18-mil", "resnet50-mil", "dino-vits16-mil"]
SAVE_EPOCHS = "0,1,2,3,5,10,15,25,35,50"
EFFECTIVE_BATCH = 600

# Config-table columns (order matters for the TSV).
COLUMNS = ["row_id", "substage", "model", "pooling", "bag", "batch_size",
           "n_experiments", "accumulate", "lr", "pretrained", "finetune_layers",
           "encode_chunk_size", "valset", "row_tag"]


def lr_tag(lr):
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def finetune_for(model, pretrained):
    if not pretrained:
        return "all"            # random init -> train the whole network
    return "block10,block11" if model.startswith("dino") else "layer3,layer4"


def batch_math(bag, batch_size):
    """Return (batch_size, n_experiments, accumulate) for an effective batch of 600.

    Uniform per-step batch_size across backbones (default 200) avoids BN
    confounds and respects the resnet50/dino <=200 memory cap. n_experiments =
    bags per microbatch = batch_size // bag.
    """
    assert batch_size % bag == 0, f"batch_size {batch_size} not divisible by bag {bag}"
    assert EFFECTIVE_BATCH % batch_size == 0, f"effective {EFFECTIVE_BATCH} not divisible by batch_size {batch_size}"
    n_experiments = batch_size // bag
    accumulate = EFFECTIVE_BATCH // batch_size
    return batch_size, n_experiments, accumulate


def build_grid():
    """Return base config dicts (before val-set expansion)."""
    base = []

    def add(substage, model, pooling, bag, lr, pretrained):
        base.append(dict(substage=substage, model=model, pooling=pooling,
                         bag=bag, lr=lr, pretrained=pretrained))

    # a: model-config grid
    for m in BACKBONES:
        for pool in ("attention", "linear"):
            add("a", m, pool, bag=50, lr=1e-4, pretrained=True)
    # b: bag-size scout (attention only)
    for m in BACKBONES:
        for bag in (10, 100):
            add("b", m, "attention", bag=bag, lr=1e-4, pretrained=True)
    # c: lr scout (attention only)
    for m in BACKBONES:
        for lr in (1e-5, 1e-3):
            add("c", m, "attention", bag=50, lr=lr, pretrained=True)
    # d: init scout (random weights, whole-net training)
    for m in ("resnet18-mil", "resnet50-mil"):
        add("d", m, "attention", bag=50, lr=1e-4, pretrained=False)
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep_dir", required=True, help="Sweep parent directory (created)")
    ap.add_argument("--valsets", required=True,
                    help="JSON mapping validation-set name -> list of experiment IDs")
    data = ap.add_mutually_exclusive_group(required=True)
    data.add_argument("--class_config", help="Class-config YAML (remoscope-database data source)")
    data.add_argument("--data_manifest", help="Manifest CSV (see README)")
    ap.add_argument("--batch_size", type=int, default=200,
                    help="Per-step batch size (uniform across backbones; <=200 for r50/dino)")
    ap.add_argument("--encode_chunk_size", type=int, default=None,
                    help="If set, applied to resnet50/dino rows (use if those OOM)")
    ap.add_argument("--concurrency", type=int, default=16, help="Max simultaneous array tasks")
    ap.add_argument("--walltime", default="15:00:00")
    ap.add_argument("--mem", default="128G")
    ap.add_argument("--cpus", type=int, default=14)
    ap.add_argument("--partition", default="gpu")
    args = ap.parse_args()

    os.makedirs(args.sweep_dir, exist_ok=True)
    os.makedirs(os.path.join(args.sweep_dir, "slurm-outputs"), exist_ok=True)
    os.makedirs(os.path.join(args.sweep_dir, "runs"), exist_ok=True)

    # Copy the val sets into the sweep dir for provenance + runtime use.
    valsets_dst = os.path.join(args.sweep_dir, "valsets.json")
    shutil.copy(args.valsets, valsets_dst)
    with open(valsets_dst) as f:
        vs = json.load(f)
    print("Val sets: " + ", ".join(f"{k}={len(v)} exp" for k, v in vs.items()))

    base = build_grid()
    rows = []
    rid = 0
    for b in base:
        bs, ne, acc = batch_math(b["bag"], args.batch_size)
        # Full-backbone (random-init) resnet50/dino blow past GPU memory at batch 200;
        # chunk the backbone encode (gradient checkpointing). Explicit flag overrides.
        full_backbone_big = (not b["pretrained"]) and (
            b["model"].startswith("resnet50") or b["model"].startswith("dino"))
        if args.encode_chunk_size and not b["model"].startswith("resnet18"):
            ecs = args.encode_chunk_size
        elif full_backbone_big:
            ecs = 32   # full-backbone resnet50/dino must fit 40GB cards (mixed GPU pool)
        else:
            ecs = None
        ft = finetune_for(b["model"], b["pretrained"])
        for valset in vs:
            tag = (f"{b['model']}_{b['pooling']}_bag{b['bag']}_lr{lr_tag(b['lr'])}_"
                   f"{'pre' if b['pretrained'] else 'rand'}_{valset}")
            rows.append({
                "row_id": rid, "substage": b["substage"], "model": b["model"],
                "pooling": b["pooling"], "bag": b["bag"], "batch_size": bs,
                "n_experiments": ne, "accumulate": acc, "lr": repr(b["lr"]),
                "pretrained": "true" if b["pretrained"] else "false",
                "finetune_layers": ft, "encode_chunk_size": ecs if ecs else "",
                "valset": valset, "row_tag": tag,
            })
            rid += 1

    configs = os.path.join(args.sweep_dir, "sweep_configs.tsv")
    with open(configs, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    data_arg = (f"--class_config {os.path.abspath(args.class_config)}" if args.class_config
                else f"--data_manifest {os.path.abspath(args.data_manifest)}")
    array_script = os.path.join(args.sweep_dir, "hp_sweep_array.sh")
    runner = os.path.join(REPO_ROOT, "scripts", "run_sweep_row.py")
    script = f"""#!/bin/bash
#SBATCH --job-name=remo-id-hpsweep
#SBATCH --array=0-{n - 1}%{args.concurrency}
#SBATCH --time={args.walltime}
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition={args.partition}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.mem}
#SBATCH --output={args.sweep_dir}/slurm-outputs/%A_%a.out

env | grep "^SLURM" | sort
nvidia-smi

export PYTHONUNBUFFERED=1
wandb online

cd {REPO_ROOT}
python {runner} \\
    --configs {configs} \\
    --row $SLURM_ARRAY_TASK_ID \\
    {data_arg} \\
    --valsets {valsets_dst} \\
    --sweep_dir {args.sweep_dir} \\
    --save_epochs {SAVE_EPOCHS}
rc=$?

wandb offline
exit $rc   # propagate the training exit code (don't let `wandb offline` mask failures)
"""
    with open(array_script, "w") as f:
        f.write(script)
    os.chmod(array_script, 0o755)

    # Summary.
    from collections import Counter
    print(f"\nWrote {configs} ({n} runs)")
    print("By substage:", dict(Counter(r["substage"] for r in rows)))
    print("By model:", dict(Counter(r["model"] for r in rows)))
    print(f"Effective batch = {args.batch_size} x {EFFECTIVE_BATCH // args.batch_size} = {EFFECTIVE_BATCH}")
    print(f"\nWrote training array : {array_script}")
    print(f"\nTo launch training:\n  sbatch {array_script}")


if __name__ == "__main__":
    main()

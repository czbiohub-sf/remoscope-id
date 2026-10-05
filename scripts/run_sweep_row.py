#!/usr/bin/env python
"""Per-row dispatcher for the HP sweep array job.

Reads one row of sweep_configs.tsv (selected by --row == SLURM_ARRAY_TASK_ID),
injects the row's validation set as repeated --val_experiment_ids, and invokes
`remo-id train` with a unique, self-describing save_dir.
"""

import argparse
import csv
import json
import os
import subprocess
import sys


def load_row(configs, row_id):
    with open(configs, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if int(r["row_id"]) == row_id:
                return r
    raise SystemExit(f"row_id {row_id} not found in {configs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="sweep_configs.tsv")
    ap.add_argument("--row", type=int, required=True)
    data = ap.add_mutually_exclusive_group(required=True)
    data.add_argument("--class_config")
    data.add_argument("--data_manifest")
    ap.add_argument("--valsets", required=True)
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--save_epochs", required=True)
    ap.add_argument("--phase", default="runs", help="Sub-directory under sweep_dir for these runs")
    ap.add_argument("--max_epochs", default="50", help="Override max epochs (smoke tests use a small value)")
    ap.add_argument("--num_batches", default=None, help="Override batches/epoch (smoke tests only)")
    args = ap.parse_args()

    row = load_row(args.configs, args.row)
    with open(args.valsets) as f:
        vs = json.load(f)
    val_ids = vs[row["valset"]]
    if not val_ids:
        raise SystemExit(f"empty val set '{row['valset']}' in {args.valsets}")

    save_dir = os.path.join(args.sweep_dir, args.phase, row["row_tag"])
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "remo_id.cli", "train",
        *(["--class_config", args.class_config] if args.class_config
          else ["--manifest", args.data_manifest]),
        "--model", row["model"],
        "--pooling", row["pooling"],
        "--batch_size", row["batch_size"],
        "--n_experiments", row["n_experiments"],
        "--accumulate_grad_batches", row["accumulate"],
        "--lr", row["lr"],
        "--lr_schedule", "cosine",
        "--warmup_epochs", "5",
        "--max_epochs", args.max_epochs,
        "--early_stopping_patience", "0",   # no early stopping
        "--save_epochs", args.save_epochs,
        "--finetune_layers", row["finetune_layers"],
        "--save_dir", save_dir,
        "--no-infer",   # evaluate afterwards, e.g. with `remo-id infer` on the saved epochs
    ]
    cmd += ["--pretrained"] if row["pretrained"] == "true" else ["--no-pretrained"]
    # Optional MIL head knobs (absent -> framework defaults)
    if row.get("num_heads"):
        cmd += ["--num_heads", row["num_heads"]]
    if row.get("classifier_hidden") not in (None, ""):
        cmd += ["--classifier_hidden", row["classifier_hidden"]]   # "0" = linear readout
    if row.get("top_k"):
        cmd += ["--top_k", row["top_k"]]
    if row.get("crop_size"):
        cmd += ["--crop_size", row["crop_size"]]
    if args.num_batches:
        cmd += ["--num_batches", args.num_batches]
    if row.get("encode_chunk_size"):
        cmd += ["--encode_chunk_size", row["encode_chunk_size"]]
    for eid in val_ids:
        cmd += ["--val_experiment_ids", str(eid)]

    print(f"[row {args.row}] {row['row_tag']}")
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

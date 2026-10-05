#!/usr/bin/env python
"""Write a small synthetic Remoscope-style dataset plus a manifest CSV.

Each experiment directory holds a uint8 ``[H, W, N]`` zarr array inside a
``.zip`` (the layout remo-id reads, see ``remo_id/data/zarr_io.py``). With
``--embeddings`` it also writes per-cell embedding stores and regionprops
tables so ``--model embedding-mil`` can be tried; with ``--masks`` it writes cell
label masks and regionprops matching the images, the input of
``remo-id analysis occlusion``.

Usage (from the repository root):
    python examples/make_synthetic_data.py /tmp/remo_synthetic --embeddings
    remo-id train --manifest /tmp/remo_synthetic/manifest.csv --model resnet18-mil \\
        --batch_size 16 --n_experiments 2 --crop_size 64 --max_epochs 2 --no-pretrained
    remo-id train --manifest /tmp/remo_synthetic/manifest.csv --model embedding-mil \\
        --model_version synthetic_v1 --cells_per_bag 32 --n_experiments 2 --max_epochs 2
"""
import argparse
import sys
from pathlib import Path

# The generator is shared with the test suite.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from synthetic_data import make_synthetic_dataset, write_synthetic_embeddings  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="Output directory")
    p.add_argument("--classes", default="negative,positive", help="Comma-separated class names")
    p.add_argument("--n_per_class", type=int, default=6, help="Samples per class")
    p.add_argument("--replicates", type=int, default=1, help="Experiments (technical replicates) per sample")
    p.add_argument("--n_images", type=int, default=64, help="Frames per experiment")
    p.add_argument("--size", type=int, default=256, help="Frame height/width in pixels")
    p.add_argument("--embeddings", action="store_true", help="Also write embedding stores + regionprops")
    p.add_argument("--masks", action="store_true",
                   help="Also write cell label masks + regionprops matching the images (occlusion input)")
    p.add_argument("--model_version", default="synthetic_v1", help="Embedding-store version tag")
    args = p.parse_args()

    manifest = make_synthetic_dataset(
        args.root, classes=tuple(args.classes.split(",")), n_per_class=args.n_per_class,
        n_images=args.n_images, size=args.size, replicates=args.replicates, masks=args.masks)
    if args.embeddings:
        write_synthetic_embeddings(manifest, args.model_version)
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
build_manifest.py -- construct the evaluation manifest for Hearing Reality.

This is the ACTUAL manifest builder used for every result reported in the
thesis. It was recovered from the Phase 0 notebook cell used in the Kaggle
evaluation sessions and is committed here so the split is reproducible from
git rather than from notebook scratch.

It is NOT scripts/preprocess_datasets.py. That script belongs to the earlier,
abandoned pipeline: it enumerates files with os.walk(), includes WaveFake,
emits a different schema with no utterance_id column, and reassigns splits by
positional index. It does not and cannot produce the evaluated manifest.

Determinism: this script never enumerates the filesystem. It reads the two
ASVspoof 5 metadata files line by line, so row order is fixed by the metadata
files themselves. With seed 42 the output is byte-identical across machines,
filesystems, and operating systems.

Output schema:
    utterance_id, speaker_id, label, attack, file_path, partition, split
    label is mapped to 0 = bonafide, 1 = spoof.

Verify with scripts/verify_manifest.py, which must print ed4808bb0456de26.

Usage:
    python scripts/build_manifest.py \
        --input_root /kaggle/input/datasets/ulianazb/asvspoof-2024 \
        --output data/manifest.csv
"""
import argparse
import csv
import os
import random
from collections import defaultdict

import pandas as pd

SEED = 42
TARGET_SPOOF_PER_ATTACK = 4687
SPLIT_RATIOS = (0.70, 0.15, 0.15)

EXPECTED_SIZES = {"train": 87585, "val": 18769, "test": 18769}
EXPECTED_TOTAL = 125123


def load_meta(path, flac_dir, prefix):
    """Read an ASVspoof 5 metadata file.

    Field layout (whitespace-separated):
        [0] speaker_id  [1] utterance_id  ...  [4] attack  [5] label

    utterance_id comes directly from the metadata, NOT from the filename.
    """
    rows = []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 6:
                continue
            rows.append({
                "utterance_id": p[1],
                "speaker_id": p[0],
                "label": p[5],
                "attack": p[4],
                "file_path": os.path.join(flac_dir, f"{p[1]}.flac"),
                "partition": prefix,
            })
    return rows


def split_list(lst, rng, ratios=SPLIT_RATIOS):
    rng.shuffle(lst)
    n = len(lst)
    i1 = int(n * ratios[0])
    i2 = int(n * (ratios[0] + ratios[1]))
    return lst[:i1], lst[i1:i2], lst[i2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True,
                    help="Root of the ASVspoof 5 (2024) corpus")
    ap.add_argument("--output", default="data/manifest.csv")
    args = ap.parse_args()

    # One shared RNG, consumed in a fixed order. Do not reorder any shuffle
    # call below: the split assignment depends on the exact sequence.
    rng = random.Random(SEED)

    flac_t = os.path.join(args.input_root, "flac_T", "flac_T")
    flac_d = os.path.join(args.input_root, "flac_D", "flac_D")
    train_meta = os.path.join(args.input_root, "ASVspoof5.train.metadata.txt")
    dev_meta = os.path.join(args.input_root, "ASVspoof5.dev.metadata.txt")

    for p in (train_meta, dev_meta):
        if not os.path.exists(p):
            raise SystemExit(f"metadata file not found: {p}")
    for d in (flac_t, flac_d):
        if not os.path.isdir(d):
            raise SystemExit(f"audio directory not found: {d}")

    all_rows = load_meta(train_meta, flac_t, "T") + load_meta(dev_meta, flac_d, "D")
    print(f"loaded {len(all_rows):,} utterances from metadata")

    bonafide_rows = [r for r in all_rows if r["label"] == "bonafide"]
    spoof_by_attack = defaultdict(list)
    for r in all_rows:
        if r["label"] == "spoof":
            spoof_by_attack[r["attack"]].append(r)

    # Even draw across attacks, sorted for determinism.
    sampled_spoof = []
    for attack, rows in sorted(spoof_by_attack.items()):
        rng.shuffle(rows)
        sampled_spoof.extend(rows[:TARGET_SPOOF_PER_ATTACK])
    print(f"sampled {len(sampled_spoof):,} spoof across "
          f"{len(spoof_by_attack)} attacks, kept {len(bonafide_rows):,} bonafide")

    pool = bonafide_rows + sampled_spoof
    rng.shuffle(pool)

    if len(pool) != EXPECTED_TOTAL:
        print(f"WARNING: pool is {len(pool):,}, expected {EXPECTED_TOTAL:,}")

    b_tr, b_va, b_te = split_list([r for r in pool if r["label"] == "bonafide"], rng)
    s_tr, s_va, s_te = split_list([r for r in pool if r["label"] == "spoof"], rng)
    train, val, test = b_tr + s_tr, b_va + s_va, b_te + s_te
    for s in (train, val, test):
        rng.shuffle(s)

    for name, s in [("train", train), ("val", val), ("test", test)]:
        nb = sum(1 for r in s if r["label"] == "bonafide")
        print(f"  {name}: {len(s):,} | {nb:,} bonafide ({nb / len(s) * 100:.1f}%)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fields = ["utterance_id", "speaker_id", "label", "attack",
              "file_path", "partition", "split"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in train:
            w.writerow({**r, "split": "train"})
        for r in val:
            w.writerow({**r, "split": "val"})
        for r in test:
            w.writerow({**r, "split": "test"})

    df = pd.read_csv(args.output)
    df["label"] = df["label"].map({"bonafide": 0, "spoof": 1})
    df.to_csv(args.output, index=False)

    sizes = df["split"].value_counts().to_dict()
    if sizes != EXPECTED_SIZES:
        print(f"WARNING: split sizes {sizes} != expected {EXPECTED_SIZES}")

    print(f"\nwrote {args.output} (label: 0=bonafide, 1=spoof)")
    print("now run: python scripts/verify_manifest.py --manifest " + args.output)


if __name__ == "__main__":
    main()

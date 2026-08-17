#!/usr/bin/env python3
"""
verify_manifest.py -- confirm the manifest matches what the models trained on.

The reference digest is ed4808bb0456de26. Every evaluation session in this
project ran this check before trusting any result; a mismatch means the
manifest no longer corresponds to the training split and the checkpoints
cannot be evaluated against it.

The formula, exactly as used:

    1. read the manifest
    2. sort rows by utterance_id
    3. concatenate "<utterance_id>:<split>" for every row, no separator
    4. sha256 of that UTF-8 string
    5. take the FIRST 16 HEX CHARACTERS

Two properties worth noting. It covers only utterance_id and split, so it is
insensitive to file_path -- which is why it held across machines whose corpus
mounted at different paths. And it sorts before hashing, so row order in the
file does not affect the result.

Note: this is NOT a hash of the file bytes. sha256 over the whole CSV will not
produce this value.

Usage:
    python scripts/verify_manifest.py --manifest data/manifest.csv
"""
import argparse
import hashlib
import sys

import pandas as pd

REFERENCE_HASH = "ed4808bb0456de26"
EXPECTED_SIZES = {"train": 87585, "val": 18769, "test": 18769}


def compute_split_hash(df):
    d = df.sort_values("utterance_id").reset_index(drop=True)
    payload = "".join(d["utterance_id"].astype(str) + ":" + d["split"].astype(str))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--reference", default=REFERENCE_HASH)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)

    missing = {"utterance_id", "split", "label"} - set(df.columns)
    if missing:
        sys.exit(f"FAIL: manifest is missing columns {missing}. "
                 f"Wrong builder? scripts/build_manifest.py emits the correct schema.")

    h = compute_split_hash(df)
    print(f"manifest:  {args.manifest}")
    print(f"rows:      {len(df):,}")
    print(f"hash:      {h}")
    print(f"reference: {args.reference}")

    ok = True

    sizes = df["split"].value_counts().to_dict()
    print(f"sizes:     {sizes}")
    if sizes != EXPECTED_SIZES:
        print(f"  FAIL: expected {EXPECTED_SIZES}")
        ok = False

    test = df[df["split"] == "test"]
    if not test["utterance_id"].is_unique:
        print("  FAIL: duplicate utterance_id in test split")
        ok = False

    bona = (test["label"] == 0).mean()
    print(f"test bonafide fraction: {bona:.4f}")
    if not 0.395 < bona < 0.407:
        print("  FAIL: class balance drifted")
        ok = False

    if h != args.reference:
        print("\nFAIL: split hash mismatch.")
        print("The manifest does not match the split the models were trained on.")
        print("Do not evaluate against it. Rebuild with scripts/build_manifest.py.")
        sys.exit(1)

    if not ok:
        sys.exit(1)

    print("\nPASS: manifest matches the training split.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/preprocess_datasets.py

Preprocesses WaveFake and ASVspoof 5 datasets into a single manifest.csv.

Steps:
  1  Print ASVspoof 5 TSV schema and auto-detect column names
  2a Resample WaveFake .wav files (22050/24000 Hz) to 16 000 Hz
  2b Copy ASVspoof 5 .flac files flat by partition (no resampling)
  2c Parse TSV metadata into label + vocoder dictionaries
  2d Build manifest.csv  (file_path, label, vocoder_type, dataset_source, split)
  3  Print manifest summary with class-imbalance warning
  4  Spot-check 15 random rows
  6  Print training-readiness checklist

Usage — dry run (100 WaveFake files, skip ASVspoof copy):
  python scripts/preprocess_datasets.py \\
      --wavefake_root  "..." \\
      --asvspoof_root  "..." \\
      --processed_root "..." \\
      --manifest_out   "..." \\
      --dry_run --dry_run_limit 100

Usage — full run:
  python scripts/preprocess_datasets.py \\
      --wavefake_root  "..." \\
      --asvspoof_root  "..." \\
      --processed_root "..." \\
      --manifest_out   "..."
"""

import argparse
import os
import random
import shutil
from collections import defaultdict
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WAVEFAKE_VOCODER_MAP = {
    "ljspeech_melgan":           "melgan",
    "ljspeech_melgan_large":     "melgan_large",
    "ljspeech_multi_band_melgan":"multiband_melgan",
    "ljspeech_full_band_melgan": "fullband_melgan",
    "ljspeech_parallel_wavegan": "parallel_wavegan",
    "ljspeech_waveglow":         "waveglow",
    "ljspeech_hifiGAN":          "hifigan",
    "jsut_multi_band_melgan":    "jsut_multiband_melgan",
    "jsut_parallel_wavegan":     "jsut_parallel_wavegan",
    "common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech": "conformer_fastspeech2",
}

ASV_SRC_DIRS = {"train": "Flac_T", "val": "Flac_D", "test": "Flac_E"}

ASV_TSV_FILES = {
    "train": "ASVspoof5.train.tsv",
    "val":   "ASVspoof5.dev.track_1.tsv",
    "test":  "ASVspoof5.eval.track_1.tsv",
}

# Candidate column names tried in order for auto-detection
_ID_CANDIDATES    = ["uttID", "utt_id", "filename", "file", "id", "utterance_id", "utt"]
_LABEL_CANDIDATES = ["label", "key", "class", "target", "bonafide_spoof"]
_SYS_CANDIDATES   = ["system_id", "attack_type", "codec", "spoof_type", "system", "vocoder"]

# ASVspoof5 headerless TSV column layout (positional):
#   0:speaker_id  1:utterance_id  2:gender  3-5:-  6:codec  7:system_id  8:label  9:-
_ASV5_COLS = ["speaker_id", "utterance_id", "gender", "c3", "c4", "c5",
              "codec", "system_id", "label", "c9"]
_ASV5_ID_COL    = "utterance_id"
_ASV5_LABEL_COL = "label"
_ASV5_SYS_COL   = "system_id"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_col(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _makedirs(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1 — inspect TSV schema
# ---------------------------------------------------------------------------

def _read_asv_tsv(tsv_path, nrows=None):
    """Read an ASVspoof5 TSV/SSV, handling headered and headerless formats.

    ASVspoof5 files are sometimes space-separated with no header row.
    Strategy: try tab first; if only one column results, retry with whitespace.
    Then detect whether the first row is a header or data.
    """
    for sep in ("\t", r"\s+"):
        df = pd.read_csv(tsv_path, sep=sep, nrows=nrows, engine="python")
        if len(df.columns) > 1:
            break

    # Headerless detection: first column header looks like a data value
    # (speaker ID pattern like "T_4850") rather than a descriptive name.
    first_col = str(df.columns[0])
    if first_col[:2] in ("T_", "D_", "E_") or (len(first_col) > 2 and first_col[1] == "_"):
        df = pd.read_csv(tsv_path, sep=sep, header=None, nrows=nrows, engine="python")
        n_cols = len(df.columns)
        df.columns = _ASV5_COLS[:n_cols]
    return df


def step1_inspect_tsv(args):
    print("\n" + "═" * 55)
    print("  STEP 1 — ASVspoof 5 TSV Schema")
    print("═" * 55)

    tsv_dir   = os.path.join(args.asvspoof_root, "ASVspoof5_protocols")
    train_tsv = os.path.join(tsv_dir, ASV_TSV_FILES["train"])

    print(f"  Reading: {train_tsv}")
    df5 = _read_asv_tsv(train_tsv, nrows=5)
    print(f"\n  Columns : {df5.columns.tolist()}")
    print(f"\n  First 5 rows:")
    print(df5.to_string())

    cols      = df5.columns.tolist()
    id_col    = args.filename_col or _detect_col(cols, _ID_CANDIDATES)    or _ASV5_ID_COL
    label_col = args.label_col    or _detect_col(cols, _LABEL_CANDIDATES) or _ASV5_LABEL_COL
    sys_col   = args.system_col   or _detect_col(cols, _SYS_CANDIDATES)   or _ASV5_SYS_COL

    # Validate detected columns exist in the dataframe
    missing = [c for c in [id_col, label_col] if c not in cols]
    if missing:
        print(f"\n  ❌ Columns not found in TSV: {missing}")
        print(f"     Available columns: {cols}")
        raise SystemExit(1)

    print(f"\n  ✅ Columns confirmed:")
    print(f"    filename_col : {id_col!r}  → e.g. {df5[id_col].iloc[0]!r}")
    print(f"    label_col    : {label_col!r}  → e.g. {df5[label_col].iloc[0]!r}")
    print(f"    system_col   : {sys_col!r}  → e.g. {df5[sys_col].iloc[0] if sys_col in cols else 'N/A'!r}")
    return id_col, label_col, sys_col


# ---------------------------------------------------------------------------
# Step 2a — resample WaveFake
# ---------------------------------------------------------------------------

def _resample_and_save(src, dst, target_sr=16_000):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return "skipped"
    try:
        data, sr = sf.read(src)
        if sr != target_sr:
            g    = gcd(sr, target_sr)
            data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
        sf.write(dst, data, target_sr)
        return "done"
    except Exception as e:
        print(f"\n  ⚠️  FAILED: {src} — {e}")
        return "failed"


def step2a_resample_wavefake(args, dry_run_limit=None):
    print("\n" + "═" * 55)
    print("  STEP 2a — Resample WaveFake to 16 kHz")
    print("═" * 55)

    src_root = os.path.join(args.wavefake_root, "generated_audio")
    dst_root = os.path.join(args.processed_root, "wavefake", "generated_audio")

    all_wav = []
    for dirpath, _, filenames in os.walk(src_root):
        for f in filenames:
            if f.endswith(".wav"):
                all_wav.append(os.path.join(dirpath, f))

    if dry_run_limit is not None:
        all_wav = all_wav[:dry_run_limit]
        print(f"  DRY RUN: limiting to {len(all_wav)} files")

    print(f"  Source : {src_root}")
    print(f"  Dest   : {dst_root}")
    print(f"  Files  : {len(all_wav)}")

    counts: dict[str, int] = defaultdict(int)
    for src in tqdm(all_wav, desc="Resampling WaveFake", unit="file"):
        rel = os.path.relpath(src, src_root)
        dst = os.path.join(dst_root, rel)
        counts[_resample_and_save(src, dst)] += 1

    print(f"  Done {counts['done']} resampled | "
          f"{counts['skipped']} skipped | {counts['failed']} failed")
    return dst_root


# ---------------------------------------------------------------------------
# Step 2b — copy ASVspoof 5
# ---------------------------------------------------------------------------

def step2b_copy_asvspoof(args):
    print("\n" + "═" * 55)
    print("  STEP 2b — Copy ASVspoof 5 (no resampling)")
    print("═" * 55)

    for split, src_folder in ASV_SRC_DIRS.items():
        src_dir = os.path.join(args.asvspoof_root, src_folder)
        dst_dir = os.path.join(args.processed_root, "asvspoof5", split)
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.exists(src_dir):
            print(f"  ⚠️  {src_folder} not found — skipping {split}")
            continue

        all_flac = []
        for dirpath, _, filenames in os.walk(src_dir):
            for f in filenames:
                if f.endswith(".flac"):
                    all_flac.append(os.path.join(dirpath, f))

        done = skipped = failed = 0
        for src in tqdm(all_flac, desc=f"Copying ASVspoof5/{split}", unit="file"):
            dst = os.path.join(dst_dir, os.path.basename(src))
            if os.path.exists(dst):
                skipped += 1
                continue
            try:
                shutil.copy2(src, dst)
                done += 1
            except Exception as e:
                print(f"\n  ⚠️  FAILED: {src} — {e}")
                failed += 1

        print(f"  {split}: {done} copied | {skipped} skipped | {failed} failed")


# ---------------------------------------------------------------------------
# Step 2c — parse TSV metadata
# ---------------------------------------------------------------------------

def _parse_asv_tsv(tsv_path, filename_col, label_col, system_col=None):
    df = _read_asv_tsv(tsv_path)
    label_map   = {}
    vocoder_map = {}
    for _, row in df.iterrows():
        uid   = str(row[filename_col]).strip()
        label = 0 if str(row[label_col]).strip().lower() == "bonafide" else 1
        label_map[uid] = label
        if system_col and system_col in df.columns:
            vocoder_map[uid] = str(row[system_col]).strip()
    return label_map, vocoder_map


def step2c_parse_tsvs(args, filename_col, label_col, system_col):
    print("\n" + "═" * 55)
    print("  STEP 2c — Parse TSV label files")
    print("═" * 55)

    tsv_dir     = os.path.join(args.asvspoof_root, "ASVspoof5_protocols")
    label_maps  = {}
    vocoder_maps = {}

    for split, fname in ASV_TSV_FILES.items():
        path = os.path.join(tsv_dir, fname)
        if not os.path.exists(path):
            print(f"  ⚠️  TSV not found: {path}")
            label_maps[split]   = {}
            vocoder_maps[split] = {}
            continue

        lm, vm = _parse_asv_tsv(path, filename_col, label_col, system_col)
        label_maps[split]   = lm
        vocoder_maps[split] = vm

        n_bon = sum(1 for v in lm.values() if v == 0)
        n_sp  = sum(1 for v in lm.values() if v == 1)
        print(f"  {split:5s}: {len(lm):>7} entries — bonafide={n_bon}, spoof={n_sp}")

    return label_maps, vocoder_maps


# ---------------------------------------------------------------------------
# Step 2d — build manifest
# ---------------------------------------------------------------------------

def step2d_build_manifest(args, label_maps, vocoder_maps, wavefake_dst_root):
    print("\n" + "═" * 55)
    print("  STEP 2d — Building manifest.csv")
    print("═" * 55)

    rows      = []
    unmatched = 0

    # ── ASVspoof 5 ────────────────────────────────────────────────────────────
    for split, src_folder in ASV_SRC_DIRS.items():
        lm = label_maps.get(split, {})
        vm = vocoder_maps.get(split, {})

        if args.skip_asv_copy:
            # Walk raw folders directly — avoids copying ~12 GB to Drive
            search_dir = os.path.join(args.asvspoof_root, src_folder)
        else:
            search_dir = os.path.join(args.processed_root, "asvspoof5", split)

        if not os.path.exists(search_dir):
            continue

        for dirpath, _, filenames in os.walk(search_dir):
            for fname in filenames:
                if not fname.endswith(".flac"):
                    continue
                uid = os.path.splitext(fname)[0]
                if uid not in lm:
                    unmatched += 1
                    continue
                label   = lm[uid]
                vocoder = vm.get(uid) or ("bonafide" if label == 0 else "unknown_spoof")
                rows.append({
                    "file_path":      os.path.join(dirpath, fname),
                    "label":          label,
                    "vocoder_type":   vocoder,
                    "dataset_source": "asvspoof5",
                    "split":          split,
                })

    # ── WaveFake ──────────────────────────────────────────────────────────────
    wf_by_vocoder: dict[str, list[str]] = defaultdict(list)
    for dirpath, _, filenames in os.walk(wavefake_dst_root):
        folder  = os.path.basename(dirpath)
        vocoder = WAVEFAKE_VOCODER_MAP.get(folder, folder)
        for f in filenames:
            if f.endswith(".wav"):
                wf_by_vocoder[vocoder].append(os.path.join(dirpath, f))

    rng = random.Random(42)
    for vocoder, files in wf_by_vocoder.items():
        rng.shuffle(files)
        n       = len(files)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        splits  = (["train"] * n_train +
                   ["val"]   * n_val   +
                   ["test"]  * (n - n_train - n_val))
        for fpath, spl in zip(files, splits):
            rows.append({
                "file_path":      fpath,
                "label":          1,
                "vocoder_type":   vocoder,
                "dataset_source": "wavefake",
                "split":          spl,
            })

    manifest = pd.DataFrame(rows)
    _makedirs(args.manifest_out)
    manifest.to_csv(args.manifest_out, index=False)
    print(f"  Saved {len(manifest):,} rows → {args.manifest_out}")
    print(f"  Unmatched ASVspoof files (not in TSV): {unmatched}")

    return manifest, unmatched


# ---------------------------------------------------------------------------
# Step 3 — manifest summary
# ---------------------------------------------------------------------------

def print_manifest_summary(manifest, unmatched):
    print("\n" + "═" * 55)
    print("  MANIFEST SUMMARY")
    print("═" * 55)

    total = len(manifest)
    print(f"  Total files:          {total:,}")

    n_bon = len(manifest[manifest["label"] == 0])
    n_sp  = len(manifest[manifest["label"] == 1])

    print(f"\n  By label:")
    for label, name in [(0, "Bonafide "), (1, "Spoofed  ")]:
        sub = manifest[manifest["label"] == label]
        n   = len(sub)
        by_split = {s: len(sub[sub["split"] == s]) for s in ["train", "val", "test"]}
        print(f"    {name} ({label}):    {n:>8,}  "
              f"(train: {by_split['train']:,} | val: {by_split['val']:,} | test: {by_split['test']:,})")

    ratio = n_sp / n_bon if n_bon > 0 else float("inf")
    print(f"    Imbalance ratio:    {ratio:.1f} : 1  (spoof : bonafide)")

    print(f"\n  By dataset source:")
    for src in sorted(manifest["dataset_source"].unique()):
        n = len(manifest[manifest["dataset_source"] == src])
        print(f"    {src:<20} {n:>8,} files")

    print(f"\n  By vocoder type:")
    vc = manifest.groupby("vocoder_type").size().sort_values(ascending=False)
    for voc, cnt in vc.items():
        print(f"    {voc:<45} {cnt:>8,}")

    print(f"\n  By split:")
    for split in ["train", "val", "test"]:
        sub = manifest[manifest["split"] == split]
        nb  = len(sub[sub["label"] == 0])
        ns  = len(sub[sub["label"] == 1])
        print(f"    {split:<6} {len(sub):>8,}  (bonafide: {nb:,} | spoof: {ns:,})")

    print(f"\n  Unmatched files (in folder but not in TSV): {unmatched}")
    print("═" * 55)

    if ratio > 5.0:
        w_bon = total / (2 * n_bon) if n_bon > 0 else 1.0
        w_sp  = total / (2 * n_sp)  if n_sp  > 0 else 1.0
        print(f"\n⚠️  CLASS IMBALANCE WARNING")
        print(f"   Spoof:Bonafide ratio is {ratio:.1f}:1 — exceeds 5:1 threshold")
        print(f"   REQUIRED ACTION: enable weighted BCE loss in training")
        print(f"   Recommended class weights:")
        print(f"     weight for bonafide (0): {w_bon:.4f}")
        print(f"     weight for spoof    (1): {w_sp:.4f}")
        print(f"   Pass these to nn.BCEWithLogitsLoss(pos_weight=tensor([{w_sp:.4f}]))")
        print(f"   in src/train.py when --weighted_loss flag is used")


# ---------------------------------------------------------------------------
# Step 4 — spot check
# ---------------------------------------------------------------------------

def step4_spot_check(manifest, n=15):
    print("\n── Spot Check: 15 random manifest rows ──")

    sample = manifest.sample(min(n, len(manifest)), random_state=42)
    passed = 0

    for _, row in sample.iterrows():
        fpath = row["file_path"]
        try:
            data, sr = sf.read(fpath)
            duration = len(data) / sr
            rms      = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
            assert sr == 16_000,              f"wrong SR: {sr}"
            assert duration >= 1.0,           f"too short: {duration:.2f}s"
            assert rms > 0.001,               f"silent: RMS={rms:.4f}"
            assert not np.isnan(data).any(),  "NaN in waveform"
            print(f"  ✅ [{row['split']:5s}|{row['label']}|"
                  f"{str(row['vocoder_type']):<25s}] {os.path.basename(fpath)}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {os.path.basename(fpath)} — {e}")

    print(f"\n  Spot check: {passed} / {n} passed")
    return passed


# ---------------------------------------------------------------------------
# Step 6 — training readiness
# ---------------------------------------------------------------------------

def print_readiness(manifest, passed, spot_n=15):
    print("\n" + "═" * 55)
    print("  PREPROCESSING COMPLETE — TRAINING READINESS CHECK")
    print("═" * 55)

    def ok(cond):
        return "✅" if cond else "❌"

    train     = manifest[manifest["split"] == "train"]
    train_bon = len(train[train["label"] == 0])
    train_sp  = len(train[train["label"] == 1])
    n_bon     = len(manifest[manifest["label"] == 0])
    n_sp      = len(manifest[manifest["label"] == 1])
    ratio     = n_sp / n_bon if n_bon > 0 else float("inf")
    ready     = passed == spot_n and train_bon > 0 and train_sp > 0

    print(f"  manifest.csv:             {ok(True)} ({len(manifest):,} total rows)")
    print(f"  All spot checks passed:   {ok(passed == spot_n)} ({passed} / {spot_n})")
    print(f"  Sample rates all 16kHz:   {ok(passed > 0)}")
    print(f"  Train split has bonafide: {ok(train_bon > 0)} ({train_bon:,} bonafide in train)")
    print(f"  Train split has spoof:    {ok(train_sp > 0)} ({train_sp:,} spoof in train)")
    print(f"  Imbalance ratio:          {ratio:.1f} : 1")
    print(f"  Weighted loss needed:     {'YES' if ratio > 5 else 'NO'}")
    print(f"\n  READY TO REFIT PCA:       {'YES' if train_bon > 0 and train_sp > 0 else 'NO'}")
    print(f"  READY TO TRAIN:           {'YES' if ready else 'NO'}")
    print("═" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess WaveFake + ASVspoof 5 into a unified manifest.csv"
    )
    parser.add_argument("--wavefake_root",  required=True,
                        help="Root of WaveFake dataset (contains generated_audio/)")
    parser.add_argument("--asvspoof_root",  required=True,
                        help="Root of ASVspoof 5 dataset (contains Flac_T/, Flac_D/, Flac_E/, ASVspoof5_protocols/)")
    parser.add_argument("--processed_root", required=True,
                        help="Output root for resampled/copied files")
    parser.add_argument("--manifest_out",   required=True,
                        help="Output path for manifest.csv")
    parser.add_argument("--filename_col",   default=None,
                        help="TSV column for utterance ID (auto-detected if omitted)")
    parser.add_argument("--label_col",      default=None,
                        help="TSV column for bonafide/spoof label (auto-detected if omitted)")
    parser.add_argument("--system_col",     default=None,
                        help="TSV column for system/vocoder ID (auto-detected if omitted)")
    parser.add_argument("--dry_run",        action="store_true",
                        help="Process only --dry_run_limit WaveFake files; skip ASVspoof copy")
    parser.add_argument("--dry_run_limit",  type=int, default=100,
                        help="Number of WaveFake files to process in dry-run mode (default: 100)")
    parser.add_argument("--skip_wavefake",  action="store_true",
                        help="Skip WaveFake resampling entirely; build manifest from ASVspoof5 only")
    parser.add_argument("--skip_asv_copy", action="store_true",
                        help="Skip copying ASVspoof5 files; manifest points to raw paths directly (saves ~12 GB)")
    args = parser.parse_args()

    # Step 1 — TSV schema
    auto_id, auto_label, auto_sys = step1_inspect_tsv(args)
    filename_col = args.filename_col or auto_id
    label_col    = args.label_col    or auto_label
    system_col   = args.system_col   or auto_sys

    # Step 2a — resample WaveFake
    if args.skip_wavefake:
        print("\n  [SKIP] WaveFake resampling skipped (--skip_wavefake).")
        wavefake_dst = os.path.join(args.processed_root, "wavefake", "generated_audio")
    else:
        limit        = args.dry_run_limit if args.dry_run else None
        wavefake_dst = step2a_resample_wavefake(args, dry_run_limit=limit)

    # Step 2b — copy ASVspoof 5
    if args.dry_run or args.skip_asv_copy:
        reason = "DRY RUN" if args.dry_run else "SKIP"
        print(f"\n  [{reason}] Skipping ASVspoof 5 copy — manifest will use raw paths.")
    else:
        step2b_copy_asvspoof(args)

    # Step 2c — parse TSVs
    label_maps, vocoder_maps = step2c_parse_tsvs(args, filename_col, label_col, system_col)

    # Step 2d — manifest
    manifest, unmatched = step2d_build_manifest(args, label_maps, vocoder_maps, wavefake_dst)

    # Step 3 — summary
    print_manifest_summary(manifest, unmatched)

    # Step 4 — spot check
    passed = step4_spot_check(manifest)

    # Step 6 — readiness (full run only)
    if not args.dry_run:
        print_readiness(manifest, passed)


if __name__ == "__main__":
    main()

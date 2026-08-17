# 🎧 Hearing Reality: A Two-Stream Neuro-Symbolic Architecture for Deepfake Audio Detection in Lossy Environments

[![CI Pipeline](https://github.com/Remigaraki/TwoStream-Audio-Forensics/actions/workflows/ci.yml/badge.svg)](https://github.com/Remigaraki/TwoStream-Audio-Forensics/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

## 🚀 Overview

Two-stream deepfake-audio detector combining a RawNet2 raw-waveform encoder
(Stream A) with a bispectral/statistical feature stream (Stream B), joined by
an attention-based fusion head (Setup C). Trained and evaluated on ASVspoof 5
and WaveFake, with robustness evaluation across Opus and MP3/AAC codec
degradation.

**`main` is the canonical branch.** Everything needed to reproduce the
thesis results — code, checkpoints, prediction CSVs, result tables — is
committed here. Read this file before trying to regenerate anything, and
read the **⚠️ Deprecated / do-not-use** section before reaching for any
script whose name sounds plausible but isn't listed under "Usage" below.

---

## Branch layout

| Branch | Status | Contents |
|---|---|---|
| `main` | **canonical** | All source code, final checkpoints (A0/A1/B0/B1/C1/C2), all prediction CSVs, result tables/figures, eval scripts, this README |
| `eval-workspace` | historical / working | Where the results-assembly work happened before being folded into `main`. No longer diverges from `main` — kept for reference, not actively developed. |
| `run-a1` | historical | A1 training trajectory (intermediate checkpoints not carried into `main`) |
| `run-b1-k64` | historical | B1 (PCA K=64 ablation) training trajectory; final checkpoint + PCA model merged into `main` |
| `run-c1` | historical | C1 (attention fusion) training trajectory (intermediate checkpoints not carried into `main`) |
| `run-c2` | historical | C2 (concat fusion) training trajectory (intermediate checkpoints not carried into `main`) |
| `results-mp3-64`, `results-mp3-128` | historical | Superseded — their prediction CSVs are byte-identical to what's already on `main` under normalized filenames |

Don't develop against `eval-workspace` or the `run-*` branches going forward — everything load-bearing is on `main`.

---

## ⚠️ Deprecated / do-not-use

Two files in this repo look like the right entry point and are not. Both belong to an earlier, abandoned pipeline and were superseded before any thesis result was produced. They're kept in the repo for history, not for use.

| File | Looks like | Actually is |
|---|---|---|
| `data/torture_pipeline.py` | The codec-degradation pipeline | Disconnected from both training and evaluation. `train.py` never imports it. The real path is `src/pipeline/augment.py` (`transcode()` / `CODEC_CONDITIONS`), used by A1's `--augment codec` and by `scripts/make_codec_testsets.py`. |
| `scripts/preprocess_datasets.py` | The manifest builder | A different, earlier pipeline. It enumerates files via `os.walk()` (order not guaranteed across filesystems/OSes), includes WaveFake, emits a schema with no `utterance_id` column, and reassigns train/val/test splits by *positional index* after that unordered walk — so even with identical source audio, two runs can produce different splits. It cannot reproduce the evaluated manifest and its hash will never match `ed4808bb0456de26`. The real builder is `scripts/build_manifest.py`, below. |

If you're regenerating anything for the thesis, the only scripts you want are the ones named explicitly in "Usage."

---

## Checkpoint reference — which checkpoint is which

| Model | Checkpoint | Val EER | Test EER (clean) | Notes |
|---|---|---|---|---|
| A0 | `checkpoints/setup_a/best_a0_final_eer0156.pt` | 1.56% | 1.54% | RawNet2, no augmentation |
| A1 | `checkpoints/setup_a/best_a_ep13_eer0105_0713_0401.pt` | 1.05% | 1.00% | RawNet2, trained with `--augment codec` — see `docs/a1_training_log.md` for the exact command |
| B0 | `checkpoints/setup_b/best_b_ep4_eer0759_0624_1407.pt` | 7.59% | 7.50% | Statistical stream only, PCA K=128 (`data/pca_model/pca.pkl`) |
| B1 | `checkpoints/setup_b/best_b_ep3_eer0750_0713_0222.pt` | 7.50% | *not yet generated* | Statistical stream, PCA **K=64** ablation (`data/pca_model/pca_k64.pkl`). **No `preds_B1_*.csv` files exist yet for any condition — see "What could not be recovered" before citing B1 test-set numbers.** |
| C1 | `checkpoints/setup_c/best_c1_attention.pt` | 0.86% | 0.81% | Fusion, cross-modal attention head |
| C2 | `checkpoints/setup_c/best_c2_concat.pt` | 0.94% | 0.92% | Fusion, plain concat head (no attention) |

Val→test EER deltas above were validated directly in commit `1abb2fa` (A1/B0/C1/C2) and by re-running `predict_testset.py`/`build_results.py` against the committed checkpoints (A0), both within ~0.001–0.02 of the checkpoint's own logged `val_eer`. That gap is expected (val split vs. held-out test split), not evidence of a mismatched checkpoint.

---

## Usage

### 1. Environment

```bash
conda env create -f environment.yml
conda activate hearing_reality
```

### 2. Build and verify the manifest

```bash
python scripts/build_manifest.py \
    --input_root /path/to/asvspoof5_2024_corpus \
    --output data/manifest.csv

python scripts/verify_manifest.py --manifest data/manifest.csv
```

`build_manifest.py` is the **actual** Phase 0 manifest builder — recovered from the notebook cell used in the real Kaggle evaluation sessions, not reconstructed or guessed. It never touches the filesystem beyond reading two fixed metadata files (`ASVspoof5.train.metadata.txt`, `ASVspoof5.dev.metadata.txt`) line by line and copying `flac_T`/`flac_D` paths through — `utterance_id` comes directly from metadata field `[1]` on each line, never from a filename or a directory listing. Because row order is fixed by file content rather than OS-dependent directory traversal, the output is deterministic across machines, filesystems, and mount paths under a fixed seed (`SEED = 42`).

`verify_manifest.py` checks the result against the reference split hash:

```
ed4808bb0456de26
```

computed as: sort rows by `utterance_id` → concatenate `"<utterance_id>:<split>"` for every row, no separator → SHA-256 → first 16 hex characters. This hash covers only `utterance_id` and `split`, so it's insensitive to `file_path` — which is exactly why it held across three different machines with three different corpus mount paths during the original evaluation runs. It is **not** a hash of the CSV's raw bytes; don't try `sha256sum data/manifest.csv` and expect a match. `verify_manifest.py` also checks split sizes (train 87,585 / val 18,769 / test 18,769) and test-split class balance as a secondary sanity check.

If your regenerated hash doesn't match, the manifest does not correspond to what the committed checkpoints were trained/evaluated on — don't evaluate against it.

### 3. Generate codec-degraded test audio (for robustness conditions)

```bash
python scripts/make_codec_testsets.py \
    --manifest data/manifest.csv \
    --output_base /path/to/codec_test \
    --conditions opus_16
```
Resumable — safe to rerun after an interrupted session; skips files that already exist.

### 4. Run inference — `predict_testset.py`

```bash
# Setup A (RawNet2 only)
python scripts/predict_testset.py \
    --checkpoint checkpoints/setup_a/best_a_ep13_eer0105_0713_0401.pt \
    --manifest data/manifest.csv \
    --setup A \
    --data_root /path/to/flac_test \
    --output_csv results/preds_A1_clean.csv

# Setup C (fusion — requires --fusion and --pca_path)
python scripts/predict_testset.py \
    --checkpoint checkpoints/setup_c/best_c1_attention.pt \
    --manifest data/manifest.csv \
    --setup C --fusion attention \
    --pca_path data/pca_model/pca.pkl \
    --data_root /path/to/codec_test/opus_16 \
    --output_csv results/preds_C1_opus_16.csv
```

Output columns: `utterance_id, true_label, score, pred` (`score` = raw sigmoid output, `pred` = binarized at the test-set EER threshold, printed to stdout).

### 5. Build result tables — `build_results.py`

Pure post-processing, no GPU/model/audio needed — reads every `results/preds_<MODEL>_<CONDITION>.csv` and writes EER/min-DCF/Cllr/degradation-ratio/C1-margin/McNemar tables plus figures.

```bash
python scripts/build_results.py
```

Cross-check its output before trusting it: `table_eer.md`'s clean-condition C1 row should read **0.81%**, and `table_mcnemar.csv`'s clean-condition `C1 vs A1` row should read **p=0.0276** (122 vs. 89 discordant). If those don't reproduce, something upstream changed.

---

## What could not be recovered

Documented explicitly rather than silently assumed:

1. **The exact C1/C2 training invocation.** No committed log, notebook cell, or script names the `--augment`/`--freeze_streams`/`--init_stream1_from`/`--init_stream2_from`/`--fusion` values actually used. `src/train.py` defines the mechanism (frozen-stream fusion-head training is *possible*), and the checkpoints confirm Stream A in both C1 and C2 is neither A1's literal weights nor random-scratch init — but the specific source checkpoint it was frozen from is not identifiable from anything in this repo. Unlike A1 (`docs/a1_training_log.md`, recovered from a groupmate's notebook output) or the manifest (`scripts/build_manifest.py`, recovered from the Phase 0 notebook cell), no equivalent record for C1/C2 ever existed to recover.
2. **B1's (PCA K=64) test-set predictions.** The trained checkpoint and fitted PCA exist and are committed, but no `preds_B1_*.csv` was ever generated for any condition, clean included. If the thesis cites B1 test-set numbers, they need to be produced with `predict_testset.py` before submission — the val EER (7.50%, from the checkpoint's own metadata) is the only B1 number currently backed by a committed artifact.

Everything else — the manifest builder and its hash verification, training code, both eval scripts, all five other checkpoints' provenance, all 35 clean+codec prediction CSVs, and every result table — is committed, checked, and reproducible from `main` alone.

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
| `main` | **canonical** | All source code, final checkpoints (A0/A1/B0/C1/C2), all prediction CSVs, result tables/figures, eval scripts, this README. Also carries the B1 (PCA K=64, synthetic-fit) ablation and C0 (epoch-12 end-to-end fusion) results — see the checkpoint reference — and a real-data K=64 PCA (`pca_k64.pkl`) that no reported result uses. |
| `eval-workspace` | historical / working | Where the results-assembly work happened before being folded into `main`. No longer diverges from `main` — kept for reference, not actively developed. |
| `run-a1` | historical | A1 training trajectory (intermediate checkpoints not carried into `main`) |
| `run-b1-k64` | historical | Fitted `pca_k64.pkl` (K=64, real-data fit). Only the PCA file merged into `main`. It is **not** the PCA the reported B1 used; see "Deprecated / do-not-use" below. |
| `run-c1` | historical | C1 (attention fusion) training trajectory (intermediate checkpoints not carried into `main`) |
| `run-c2` | historical | C2 (concat fusion) training trajectory (intermediate checkpoints not carried into `main`) |
| `results-mp3-64`, `results-mp3-128` | historical | Superseded — their prediction CSVs are byte-identical to what's already on `main` under normalized filenames |

Don't develop against `eval-workspace` or the `run-*` branches going forward — everything load-bearing is on `main`.

---

## ⚠️ Deprecated / do-not-use

Three files in this repo look like something they're not. The first two belong to an earlier, abandoned pipeline, superseded before any thesis result was produced; the third is an orphaned artifact with no corresponding claim anywhere in the manuscript. All three are kept in the repo for history, not for use.

| File | Looks like | Actually is |
|---|---|---|
| `data/torture_pipeline.py` | The codec-degradation pipeline | Disconnected from both training and evaluation. `train.py` never imports it. The real path is `src/pipeline/augment.py` (`transcode()` / `CODEC_CONDITIONS`), used by A1's `--augment codec` and by `scripts/make_codec_testsets.py`. |
| `scripts/preprocess_datasets.py` | The manifest builder | A different, earlier pipeline. It enumerates files via `os.walk()` (order not guaranteed across filesystems/OSes), includes WaveFake, emits a schema with no `utterance_id` column, and reassigns train/val/test splits by *positional index* after that unordered walk — so even with identical source audio, two runs can produce different splits. It cannot reproduce the evaluated manifest and its hash will never match `ed4808bb0456de26`. The real builder is `scripts/build_manifest.py`, below. |
| `data/pca_model/pca_k64.pkl` | The PCA behind the reported B1 | A **different** K=64 PCA, fitted on 2000 real samples (99.7% explained variance; sklearn 1.6.1). Committed 2026-07-15. The reported B1 used `pca_k64_synthetic.pkl` instead (SHA-256 recorded in `results/provenance/B1/run_metadata.json`); the two files differ in content, not just bytes. No prediction CSV, metric or log in this repo is tied to `pca_k64.pkl` (the planned "B1r" row in `build_results.py` has no CSVs). Note: `checkpoints/setup_b/best_b_ep3_eer0750_0713_0222.pt` **is** a K=64 model (Stream 2 MLP input 184 = 120 + 64), so an earlier claim that no setup_b checkpoint can be K=64 was wrong; which K=64 PCA that checkpoint was trained with is not recorded. |

If you're regenerating anything for the thesis, the only scripts you want are the ones named explicitly in "Usage."

---

## Checkpoint reference — which checkpoint is which

| Model | Checkpoint | Val EER | Test EER (clean) | Notes |
|---|---|---|---|---|
| A0 | `checkpoints/setup_a/best_a0_final_eer0156.pt` | 1.56% | 1.54% | RawNet2, no augmentation |
| A1 | `checkpoints/setup_a/best_a_ep13_eer0105_0713_0401.pt` | 1.05% | 1.00% | RawNet2, trained with `--augment codec` — see `docs/a1_training_log.md` for the exact command |
| B0 | `checkpoints/setup_b/best_b_ep4_eer0759_0624_1407.pt` | 7.59% | 7.50% | Statistical stream only, PCA K=128 (`data/pca_model/pca.pkl`, a synthetic-noise fit — see "What could not be recovered" item 3) |
| B1 | `checkpoints/setup_b/best_b1_k64synthetic_ep9_eer0604.pt` | 6.04% | 7.58% | Statistical stream only, PCA **K=64 synthetic fit** (`data/pca_model/pca_k64_synthetic.pkl`, identical to `scripts/fit_pca.py --synthetic --n_components 64`). 10 epochs, lr 1e-4, batch 32, seed 42, no augmentation. Checkpoint is git-ignored locally like the others; force-add via LFS to publish. Provenance: `results/provenance/B1/` |
| C0 | `checkpoints/setup_c/best_c_ep12_eer0566_0710_0112.pt` | 5.66% | 6.76% | Fusion, attention head, epoch 12 of the `best_c_ep*` run. Both streams change across that run's checkpoints (not frozen); its streams differ from C1/C2's and from A0/A1/B0. Exact training command not recorded. Provenance: `results/provenance/C0/` |
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
2. **B1 and C0 training logs.** Both were evaluated on Kaggle at commit `ef1e92f` against manifest `ed4808bb0456de26` (full 18,769-utterance test split, all 7 conditions). The supplied `evaluation.log` files are partial (C0: 3 of 7 conditions, B1: 4 of 7); every EER they do print matches the committed CSVs exactly. B1's hyper-parameters come from its `run_metadata.json`; no training log exists for either. C0 was evaluated with scikit-learn 1.6.1 unpickling a 1.8.0 PCA (warning in its log); the PCA transform is linear and the C0 numbers are internally consistent, but that version mismatch is recorded here.
3. **The K=128 PCA (`pca.pkl`) is a synthetic-noise fit.** Re-running `scripts/fit_pca.py --synthetic` (500 Gaussian-noise waveforms, seed 42, K=128) with the code from the commit that added `pca.pkl` (`eed15b5`) reproduces it exactly (identical mean, components within 7e-5, same 40.71% explained variance). B0, C0, C1, C2 and the demo therefore project the bispectrum with a PCA fitted on noise, not speech. Results stay internally valid (the projection is fixed and identical at training and test time), but methods text describing a PCA "fitted on training audio" would be inaccurate.
4. **The bispectrum feature changed after `pca.pkl` was fitted.** Commit `4385aac` ("vectorize bispectrum inner loop") switched to `np.tril_indices`, so `estimate_bispectrum` now fills only the j ≤ i half of the valid region (the other symmetric half is zero); the earlier loop filled all i + j < 128. All checkpoints post-date that commit, so training, evaluation and the demo consistently use the current feature, but `pca.pkl` was fitted on the older full-region features and cannot be regenerated with current code.

Everything else — the manifest builder and its hash verification, training code, both eval scripts, the other checkpoints' provenance, all 49 clean+codec prediction CSVs, and every result table — is in this repo, checked, and reproducible from it alone (the B1/C0 files are not committed yet).

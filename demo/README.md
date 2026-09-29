# Research demo (Streamlit, CPU)

Upload a speech recording and get a **C1** model score and a bona fide / spoofed
decision for its **first 4 seconds**. **A1** can be run as a separate comparison, and
its score is never combined with C1's. Nothing in `src/`, `scripts/`, `results/` or the
checkpoints is modified by the demo.

| File | Purpose |
|---|---|
| `inference.py` | Asset hash/LFS checks, strict model loading, the evaluated preprocessing, scoring |
| `decode.py` | Bounded decoder run as a time-limited subprocess (untrusted media never parsed in the server) |
| `app.py`, `.streamlit/config.toml` | UI (Analyze + About tabs), 20 MB upload cap, no telemetry, no tracebacks to visitors |
| `thresholds.json` | Decision threshold per model **with its status and provenance** |
| `calibrate_threshold.py` | Validation-only threshold calibration (writes `thresholds.json`) |
| `check_parity.py` | Demo scores vs. committed `results/preds_*_clean.csv` on the same real audio |
| `benchmark.py` | Measured cold start / latency / memory |
| `stage_space.py`, `space/` | Assemble the minimal Hugging Face Docker Space directory |
| `requirements.txt` / `requirements.lock` | Direct pins (local) / full Linux lock used by the Dockerfile |
| `tests/test_demo.py` | 24 tests |

## Run locally (Windows, presentation fallback, no internet needed after setup)

```powershell
# one-time setup, isolated from the global Python
py -3.11 -m venv demo\.venv
demo\.venv\Scripts\python -m pip install -r demo\requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
demo\.venv\Scripts\python -m pip install pytest pydub   # tests only (pydub: src/train.py import)

# run (from demo\ so .streamlit\config.toml is picked up)
cd demo
.venv\Scripts\streamlit run app.py --server.address localhost
# -> http://localhost:8501

# tests
cd ..
demo\.venv\Scripts\python -m pytest demo\tests -q
```

Offline Docker fallback (after one build): `docker run --rm -p 7860:7860 twostream-demo:local`, then open http://localhost:7860.

## What the pipeline does (and does not do)

`inference.preprocess` is a line-for-line copy of `_ManifestDataset.__getitem__` with
`augment="none"`: `sf.read(float32)` → `[C, N]` → `torchaudio.functional.resample` to 16 kHz →
channel mean → first 64,000 samples, or zero-padded if shorter → `[1, 1, 64000]` float32. It adds
no normalisation, trimming, denoising, augmentation or window aggregation. The network ends in
`Sigmoid`, and the demo shows that output as the **model score** with no second sigmoid. It is not
presented as a calibrated probability. The decision rule is `score >= threshold → spoofed`, the
same direction as `predict_testset.py`.

The only change is that the decoder reads a bounded 4.5 s prefix instead of the whole file.
Tests compare it to `_ManifestDataset` on the same files (WAV/FLAC/MP3, mono and stereo,
8/16/22.05/44.1/48 kHz, short and long). The result is **bitwise identical** (max diff 0). With the
0.5 s resampler margin removed the diff reaches 3.5e-2, and the test catches it.

Input checks: empty, undecodable or corrupt files, containers other than WAV/FLAC/MP3, sample
rates outside 8–192 kHz, more than 8 channels, NaN/Inf samples, clips under 0.5 s or over 10 min,
files over 20 MB, and a silent or near-silent segment (peak < −80 dBFS; this is an input-quality
check, not speech detection). Decoding runs in a subprocess with a 20 s timeout (plus rlimits on
Linux). Inference runs one request at a time with a 90 s queue timeout. Its cost is fixed because
the input is always 64,000 samples.

Privacy: each upload is written to its own `TemporaryDirectory` (deleted once decoding finishes,
including on errors) under a fixed name, so the user's filename never reaches the filesystem. It
is never put in `st.cache_*`. Only model weights are cached. Nothing is logged. Streamlit keeps the
bytes in memory for the session's audio player.

Artifacts are pinned by SHA-256 in `inference.MODELS` and checked **before** `torch.load(weights_only=True)`
and before `pca.pkl` is unpickled. A Git LFS pointer file is detected and reported. Loading uses
`strict=True` and `eval()`, and scoring uses `inference_mode()`. There is no fallback model: if C1
cannot load, the app reports it and does not substitute A1.

scikit-learn is pinned to **1.8.0**, the version recorded inside `pca.pkl`. It loads with
warnings-as-errors and raises no `InconsistentVersionWarning`, and `transform` matches
`(x − mean_)·componentsᵀ` to float32 rounding (9e-5 absolute on values around 1e3).

## Decision threshold: status

- **Audit:** no validation-derived threshold exists in the repo. The checkpoints store only
  `val_eer`, and `predict_testset.py`'s `pred` column uses the **test-set** EER threshold, which the
  demo does not use.
- **Current:** `thresholds.json` ships `0.5` for C1 and A1, marked `"status": "provisional"`. The app
  shows a *Provisional threshold* warning with every result.
- **To calibrate** (needs the ASVspoof 5 validation audio, e.g. on Kaggle; a GPU is recommended):

  ```bash
  # 1. Codec-compressed copies of the VALIDATION audio (separate folder from the test copies).
  #    On a Windows console, prefix with PYTHONIOENCODING=utf-8 (the script prints emoji).
  python scripts/make_codec_testsets.py --manifest data/manifest.csv \
      --output_base /kaggle/tmp/codec_val --split val

  # 2. Calibrate on clean validation audio, then check the threshold on each codec condition.
  python demo/calibrate_threshold.py --model C1 --operating_point min_dcf \
      --manifest data/manifest.csv --data_root /path/to/flac \
      --codec_root /kaggle/tmp/codec_val \
      --scores_csv demo/calibration/val_scores_C1.csv
  python demo/calibrate_threshold.py --model A1 ...   # same arguments
  ```

  **Operating point** (`--operating_point`):
  - `eer` (default): the point where the two error rates are equal.
  - `min_dcf`: minimises the paper's detection cost (`build_results.py`: π_spoof = 0.05, C_miss = 1,
    C_fa = 10, i.e. cost = 1.9·P_miss + P_fa). Wrongly flagging bona fide speech costs 1.9 times as
    much as missing a spoof. This suits a public demo better, but it is a choice: the constants
    carry a "verify against the ASVspoof 5 eval plan" note in `build_results.py`.

  The threshold search is cross-checked against `build_results.compute_min_dcf`. The two agree to
  float32 precision, including when scores tie.

  **Codec check** (`--codec_root`): the clean-validation threshold is applied, unchanged, to every
  codec condition found. For each condition the script records the miss and false-alarm rates, the
  cost at that threshold, and the best cost achievable for that condition. A large gap means the
  threshold does not hold up under that codec. The check never moves the threshold; it goes into
  `thresholds.json` as `codec_check_val` for you to judge.

  **Safeguards:** it reads only `split == "val"`. It refuses a manifest whose split hash isn't
  `ed4808bb0456de26`, and refuses to save unless its validation EER reproduces the checkpoint's
  logged `val_eer` (±0.002). Everything is saved with provenance: checkpoint and split hashes, class
  counts, both operating points, error rates at the chosen threshold, the codec check, git commit,
  torch version and time.

  **Status: not run on real data.** Tested only on synthetic clips (24 tests, plus a dry run with
  real FFmpeg Opus/MP3/AAC transcodes of the validation split): the guards fire, and the codec
  check and provenance write work.

## Score parity with the evaluation pipeline: pending

No ASVspoof 5 audio is available on this machine, so parity with the committed CSVs has **not**
been verified. Where the audio lives:

```bash
python demo/check_parity.py --model C1 --manifest data/manifest.csv --data_root /path/to/flac --limit 200
python demo/check_parity.py --model A1 --manifest data/manifest.csv --data_root /path/to/flac --limit 200
```
Tolerance is 1e-4 absolute. The CSV stores 6 decimals and came from GPU inference; BatchNorm is in
eval mode, so batching does not change per-utterance output. The script exits non-zero on any
mismatch.

## Measured performance (`demo/benchmark.py`, 2 torch threads)

| | Local venv (Windows laptop) | Docker, `--cpus=2` |
|---|---|---|
| C1 warm inference (median / max) | 0.45 s / 0.67 s | 0.83 s / 1.39 s |
| C1 end-to-end incl. subprocess decode (median) | 0.53 s | 0.78 s |
| A1 warm inference (median) | 0.40 s | 0.56 s |
| Cold: imports + verify + load + first C1 result | 2.5 s | 10.6 s (7.0 s of it Python/torch import) |
| Memory: C1 loaded / C1+A1 peak | 367 MB / 736 MB | 437 MB / 762 MB |
| Container start → healthy | | 1.8 s (models load on first visit) |

HF free CPU hardware (2 vCPU, 16 GB) may be slower per core than this laptop's Docker VM, so
re-run `benchmark.py` in the Space once deployed. No model or architecture change is needed at
these numbers.

## Deploy to Hugging Face Spaces (manual, free CPU)

Nothing is pushed automatically. Once you have chosen the account and Space name:

```bash
python demo/stage_space.py --out ../twostream-space      # copies ~97 MB, re-verifies hashes
cd ../twostream-space
docker build -t twostream-demo:local . && docker run --rm -p 7860:7860 twostream-demo:local  # optional check

# Create a *Docker* Space on huggingface.co (hardware: CPU basic), then:
git init && git lfs install
git remote add origin https://huggingface.co/spaces/<account>/<space-name>
git add .gitattributes && git add . && git commit -m "Initial demo"
git push origin main   # authenticate via `huggingface-cli login` / a write token; never commit tokens
```

`.gitattributes` routes `*.pt` and `*.pkl` through Git LFS, which HF requires for files over 10 MB.
The Dockerfile fails the build if an artifact is missing, is an LFS pointer, or fails its hash. It
runs as uid 1000 on port 7860 (`app_port` in `README.md`). Streamlit XSRF is disabled because the
HF iframe blocks the cookie and uploads would otherwise return 403. The app has no state-changing
actions to protect. **Keep the Space private until the licensing checks below are done.**

Artifact hosting: the checkpoints ship inside the Space via LFS. Pinned downloads (for example an
HF model repo at a fixed revision) are not implemented. Add them only if the Space must stay code-only.
The SHA-256 pins would already catch a wrong or changed file.

## Before making anything public: licensing checks

1. **Code licence:** the root README shows an MIT badge but the repo has **no LICENSE file**. Add
   one (agreed with co-authors) before publishing code.
2. **Model weights:** C1 and A1 are trained on ASVspoof 5. Confirm that its licence and terms of use
   permit redistributing models trained on it, and add any required attribution to the Space README.
3. **Example audio:** none is bundled. Do not add ASVspoof/WaveFake clips, or any recording of a
   person, without explicit redistribution rights and consent.
4. **Third-party:** PyTorch, torchaudio, scikit-learn, Streamlit and libsndfile are permissively
   licensed. FFmpeg (installed in the image as requested, but not on the decode path) is
   LGPL/GPL depending on the build. The Debian package is fine to use in a container.
   `space/Dockerfile` can drop it if it isn't needed.

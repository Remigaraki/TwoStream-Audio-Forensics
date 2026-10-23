# Demo verification after corrected calibration

The original research CSVs are historical results and must not be overwritten.
The saved LFCC correction is integrated in `src/stream2/lfcc.py`. No retraining
or full calibration scoring is needed. Active thresholds remain provisional
until current parity reports pass.

## Run the data-dependent checks in Kaggle

Import `notebooks/Kaggle_Demo_Parity_Verification.ipynb`. Attach the ASVspoof
audio dataset and final `calibration_backup_20261006_140731_113645.zip` (or its
extracted contents). Set `DATA_ROOT` and, if automatic discovery is ambiguous,
`BACKUP` to the ZIP or extracted root containing `demo_thresholds.json`.
Run the cells in order in a fresh session with Internet enabled.

The notebook carries a source bundle; no GitHub source push or GH_TOKEN is
needed. A separate pinned CPU environment matches the local demo dependencies.
It compares 256 deterministic, class-stratified test clips per model against
independent evaluation decoding/preprocessing, across batch sizes 4/16/32 and
both recording orders, at the unchanged score tolerance 1e-4. It counts expected
short-recording rejections and fails on unexpected rejections. It separately
checks 256 validation scores against each preserved clean cache. Neither test
scores nor codec scores tune the clean thresholds.

Download `demo_parity_verification.zip`. If a command fails, run the export cell
manually and preserve the error output. Do not shrink the sample, loosen the
tolerance, edit success fields, or activate thresholds after a failure.

## Validate returned reports and install locally

Extract the verification ZIP under a fresh `outputs/` directory. The candidate
and both reports must match current source, checkpoint, PCA and runtime package
versions. Source identities normalize CRLF/LF so Windows and Linux checkouts
agree. The candidate explicitly records that legacy source provenance was
incomplete and is being bound to the code checked by this parity run.

From the repository root, using the pinned demo environment:

```powershell
demo/.venv/Scripts/python demo/install_verified_thresholds.py --candidate outputs/returned_parity/thresholds.candidate.json --report_dir outputs/returned_parity
demo/.venv/Scripts/python -m pytest demo/tests tests/test_calibration_resume.py notebooks/test_github_upload.py -q
demo/.venv/Scripts/python demo/verify_release.py
```

Installation preserves the prior active JSON. The loader rejects stale
validation identities or missing/mismatched parity reports. The release
preflight also rejects provisional thresholds. Report files and source must
travel with the installed thresholds in any release.

Before deployment, repeat Streamlit startup and upload checks with the installed
thresholds. Existing tests exercise WAV/FLAC/MP3, mono/stereo and resampling,
normal uploads with both models, short recordings, silence, corrupt files,
unsupported inputs and decoder/asset errors. Synthetic checks establish code
behavior, not detection quality. AppTest uses a stand-in UploadedFile because
the test API cannot drive the native uploader; a final browser upload check is
still appropriate after installation.

## Calibration robustness review

The staging command produces `calibration_review.md` from all 14 completed
model/condition caches and checks error rates against the saved summaries.
At the clean threshold, C1 flags 4.189% of genuine MP3-64 and 3.883% of genuine
MP3-128 recordings, compared with A1's 1.569% and 1.383%. C1 has lower clean
validation error at its selected threshold. Keep codec limitations visible;
do not describe scores as calibrated authenticity probabilities or generalize
ASVspoof validation results to arbitrary uploaded audio.

The legacy notebook's threshold-only publishing path is disabled. Choose the
hosting destination after verification; publish corrected source, pinned model
assets, thresholds and verification reports together after local checks pass.

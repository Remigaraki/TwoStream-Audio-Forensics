# Revision evidence for Hearing Reality

Prepared from the attached thesis, the revision megaprompt, local repository, real checkpoint files, returned manifest, and saved artifacts in Downloads. No training or evaluation was rerun. The manuscript and demo were not edited. Missing historical evidence is labeled explicitly.

## Sample size: what can be stated

The verified manifest contains all 50,131 bona fide utterances and 74,992 spoofed utterances, with 4,687 from each of 16 attacks, for a total of 125,123. [computed from manifest: outputs/returned_parity_20261007/manifest.reference.csv]

| Split | Bona fide | Spoofed | Total |
| --- | ---: | ---: | ---: |
| Training | 35,091 | 52,494 | 87,585 |
| Validation | 7,520 | 11,249 | 18,769 |
| Test | 7,520 | 11,249 | 18,769 |
| Total | 50,131 | 74,992 | 125,123 |

Source for every table entry: [computed from manifest]. The repository verifier passes with ID/split hash `ed4808bb0456de26`. This authenticates agreement with the reference split, not the unavailable original corpus metadata or audio.

The combined pool size of 323,307 and its source-partition counts are supplied by the megaprompt; original ASVspoof metadata was not available for a new audit. [megaprompt, context table]

The verified subset is 40.07% bona fide and 59.93% spoofed, compared with 15.51% bona fide and 84.49% spoofed using the supplied full-pool counts. [computed from manifest and megaprompt pool counts]

Equal attack sampling preserves representation and reduces dominance by larger attack groups. The exact cap rationale remains undocumented. `floor(75,000 / 16) = 4,687` is an arithmetic-compatible hypothesis, not an established motivation. A direct equal allocation under a total target of 125,000 would instead give `(125,000 - 50,131) / 16 = 4,679.3125`. [computed arithmetic; cap in scripts/build_manifest.py]

Current code uses `nn.BCELoss()` without weights and a shuffled DataLoader without a weighted sampler (`src/train.py:328`, `:262-265`). This does not establish that rebalancing was the reason weighting was omitted or that every historical revision was identical.

Suggested wording:

> The study used a fixed, reproducible subsampling rule that retained all available bona fide utterances and sampled spoofed utterances equally across attack types. The cap of 4,687 utterances per attack yielded 74,992 spoofed utterances and, together with 50,131 bona fide utterances, a total of 125,123. This construction preserved attack representation and reduced the original class imbalance. The surviving records document the cap and its deterministic implementation but do not establish a measured compute-budget derivation for the exact value.

Numeric sources for the proposed paragraph: [computed from manifest]; procedure: [scripts/build_manifest.py]. Do not add a power-analysis, minimum-attack-availability, GPU-quota or before-results explanation without supporting records. The cap's recovery commit is dated August 17, 2026; checkpoint filenames include earlier June/July dates without a year. This does not establish the original decision date. [git history; filename]

Original availability by attack: NOT FOUND. The CSV reports sampled counts by source partition, while leaving availability unknown. The smallest original attack group cannot be identified from a sampled manifest.

## Training iterations and checkpoint results

| Run | Best stored epoch (0-based) | Best log epoch (1-based) | Optimizer steps | Steps per epoch | Step quotient | Validation EER (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 10 | 11 | 30,118 | 2,738 | 11 | 1.56 |
| A1 | 13 | 14 | 38,318 | 2,737 | 14 | 1.05 |
| B0 | 4 | 5 | 13,685 | 2,737 | 5 | 7.59 |
| B1 | 9 | 10 | 27,370 | 2,737 | 10 | 6.04 |
| C0 | 12 | 13 | 71,162 | 5,474 | 13 | 5.66 |
| C1 | 6 | 7 | 38,318 | 5,474 | 7 | 0.86 |
| C2 | 3 | 4 | 21,896 | 5,474 | 4 | 0.94 |

Sources: [checkpoint], separately identified by path and SHA-256 in `03_checkpoints/checkpoint_audit.csv`. Step denominators are [computed from verified training size and documented/inferred batch size]; A0 kept the final partial batch. All principal checkpoint values match the megaprompt. The quotient records optimization represented at the checkpoint, not a recovered count of total epochs or sessions. With resumes from a best checkpoint, previously executed discarded steps are not represented by that saved state.

A1 has a documented endpoint of 30 epochs, batch size 32, learning rate 1e-4, codec augmentation probability 0.5, and augmentation seed 42. Its surviving excerpt begins log epoch 14 and contains batch losses only. It does not supply a completed epoch loss/EER or final completed-epoch count. [log: docs/a1_training_log.md]

B1 has an endpoint of 10 in its metadata and a completed log epoch 10. The recovered history is partial:

| Log epoch (1-based) | Training loss | Validation EER (%) | Learning rate | Training batches/sec |
| --- | ---: | ---: | ---: | ---: |
| 4 | 0.1799 | 6.65 | 1e-4 | 0.4 |
| 5 | 0.1746 | 6.69 | 1e-4 | 0.4 |
| 6 | 0.1727 | 6.64 | 1e-4 | 0.4 |
| 7 | 0.1685 | 6.63 | 1e-4 | 0.4 |
| 8 | 0.1662 | 6.20 | 1e-4 | 0.4 |
| 9 | 0.1645 | 6.61 | 1e-4 | 0.4 |
| 10 | 0.1626 | 6.04 | 1e-4 | 0.4 |

Sources for log epochs 4-7: [log: B1_artifacts.zip/training.log]. Sources for log epochs 8-10: [log: B1_artifacts (1).zip/training.log]. Missing log epochs 1-3; no duplicate completed summaries. The first ZIP ends partway through a subsequent epoch; it is retained as a separate segment, not combined with the second segment's completed summary. Two resume events are documented, with printed resume positions 3 and 7; dates and complete session count are unknown. [log: B1 ZIP segments]

The later archives' `best.pt` files match the reported repository B1 checkpoint byte-for-byte. The earliest ZIP holds an earlier best at stored epoch 6 / log epoch 7, with validation EER 6.6345%; it must not substitute for the reported best. [checkpoint: B1 ZIPs; computed SHA-256 comparisons]

Figure `06_figures/B1_per_epoch.png` shows loss and EER in two panels and explicitly labels the partial coverage. Caption: “Recovered partial training history for B1, covering log epochs 4–10. Earlier epoch summaries were unavailable. The retained checkpoint occurred at log epoch 10 (stored epoch 9), with validation EER 6.04%.” [log; checkpoint]

The checkpoint figure is separately labeled as saved checkpoints only. B0 and C0 intermediate membership is a candidate assignment; timestamps and increasing epochs do not establish a continuous run. Do not call it a complete learning curve.

No completed per-epoch history was recovered locally for the other runs. External Kaggle versions, W&B, shared chat/document history and Colab/Drive were inaccessible through this session; their absence is not established. The corrected-resume notebook is not evidence that its proposed commands produced the thesis checkpoints.

Current scheduler code: ReduceLROnPlateau on validation EER, factor 0.5, patience 5; C warmup lasts 10 epochs, starting at 1e-6. There is no early-stopping rule in the current training loop. Historical executed schedules and final endpoints remain unknown unless preserved in a run log. [src/train.py:333-339, 371-471]

Do not claim a common 1e-4 learning rate or globally seeded training. B0's saved optimizer rate is 1e-3; C0's is 5e-5; C1's is 3.53e-5; C2's is 2.06e-5. The thesis labels the C1/C2 target rate of 5e-5 as warmup-based inference. The manifest seed and augmentation seed do not establish seeded weight initialization, shuffle order or dropout. [checkpoint; thesis Table 3.2; src/train.py:184-186]

## The alternative B checkpoint and C0 continuity

`best_b_ep3_eer0750_0713_0222.pt` has stored epoch 3, log epoch 4, validation EER 7.5030%, optimizer steps 10,948 and saved learning rate 1e-3. Its first MLP input width is 184, whereas reported B0's is 248. Under the current feature layout, this is consistent with PCA K=64 rather than K=128; the PCA file and exact training invocation are missing. [checkpoint; architecture-based inference]

This establishes an architecture difference, not its historical run name. Whether it was a rerun, how its PCA was fitted, and whether it was used elsewhere remain NOT FOUND. The reported B0 checkpoint is the K=128 architecture. The alternative's validation EER also must not be confused with B0's approximately 7.50% clean-test EER, a different metric on a different partition. [checkpoint; thesis Table 4.5]

C0 intermediate checkpoints have compatible names and improving validation EERs. Their full session/resume lineage and dates remain NOT FOUND. They are candidate C0 files, not proof of one continuous run. [checkpoint inventory; filename]

## Hardware and future resources

The surviving A1 training excerpt establishes Kaggle paths and CUDA, but not the exact GPU. B1 training logs establish Kaggle paths and CUDA; the saved B1 notebook records “GPU: Tesla T4” in the setup cell and contains the training command. This supports a T4 in the documented B1 notebook session. GPU count, VRAM actually available, CPU cores, RAM, driver and peak allocation remain unknown. [log: docs/a1_training_log.md; B1 ZIP logs; notebook: b1-pca-sweep-clean-latest.ipynb]

The saved `c0-eval.ipynb` records a Tesla T4 for evaluation; it does not establish C0's training GPU. Devices for the remaining reported runs remain unverified. The earlier Colab pipeline should not be used to fill their hardware cells. [notebook: c0-eval.ipynb; Rafa-supplied evidence review]

Bispectrum estimation and PCA run on CPU NumPy, with transfers between the model device and CPU. “All feature extraction was accelerated on T4” overstates the implementation. [src/stream2/stream2.py:84-90, 116-120]

Suggested methods wording:

> CUDA execution is documented in the surviving A1 and B1 training logs, and a Tesla T4 is recorded in the saved B1 notebook session. Exact training hardware was not recovered for every configuration. Bispectrum estimation and PCA were performed on CPU, while neural operations used the model device.

Suggested recommendation wording, as a prospective recommendation rather than a measured minimum:

> Future work should use a persistent CUDA-capable GPU environment with sufficient CPU capacity and RAM for bispectrum extraction and concurrent audio transcoding, together with SSD storage for the corpus, codec variants and checkpoints. Before full-corpus experiments, researchers should profile peak GPU memory, complete epoch wall time and storage use on a pilot run, use the measurements to allocate GPU hours, and preserve training commands, environment details and per-epoch logs alongside checkpoints. The platform should support resumable jobs and persistent storage across sessions.

No measured numerical minimum for GPU count, VRAM, RAM, storage GB or total GPU hours can be supported from the recovered artifacts. This paragraph is a proposed recommendation, not [Rafa-stated].

The full-pool/subset ratio is `323,307 / 125,123 = 2.583913`. Using B1's rounded logged training throughput gives `2,737 / 0.4 = 6,842.5 seconds`, approximately 1.90 hours per training pass. Linear scaling gives approximately 4.91 hours for a full-pool training pass. Both are estimates excluding validation, initialization, checkpoint writing and other overhead; they are not measured epoch wall times or a completed full-pool benchmark. [computed estimate from B1 log and supplied pool count]

No real-time-factor measurement was recovered; the thesis states that inference latency was not collected. Do not supply an RTF, real-time performance claim or platform quota as an observed limit. [thesis Sections 1.6 and 5.3]

## Spoof-score terminology

Use “spoof score” for the continuous sigmoid output, with higher values indicating more spoof-like audio. Use “binary decision” for the thresholded bona fide/spoofed result. Calibration was not evaluated, so the continuous output should not be presented as a calibrated probability. [megaprompt terminology; demo/README.md]

The demo still contains the old label at `demo/app.py:20,41`, `demo/README.md:3,47`, `demo/space/README.md:17`, and `demo/tests/test_demo.py:201`. These are read-only findings; no rename, tests, commit, push or deployment was performed. [current source files]

## Remaining evidence gaps

- Firsthand reason for the exact cap and any actual compute limit.
- Original source-corpus metadata for unsampled per-attack availability.
- Missing completed epoch logs and actual final run lengths, dates and sessions.
- Training GPU/device details for runs without historical records.
- Historical identity and use of the alternative B checkpoint and C0 resume/restart lineage.
- Full-pool timing, latency/RTF, and measured hardware/storage minimums.

Rafa's supplied evidence-review answer is preserved in `04_sample_size/rafa_answers.md`; it does not close these historical gaps.

The megaprompt reports a GitHub token exposed in the Colab CONFIG cell. Revoke and rotate that token; no credential is included in this packet.

# Manuscript evidence audit — 6 October 2026

**Verdict: the draft needs correction before submission. The core five-model point estimates are supported by the prediction files; several additional results and stronger interpretations are not.**

Source: `C:/Users/Rafa/Downloads/[THS] Manuscript - Drafts (NEW).docx`. This is a review of the document as supplied, not an edited manuscript. Instructions or suggestions embedded in the manuscript were treated as document content, not authorization. Existing repository changes were left intact.

## What was checked

Three complementary checks were performed: manuscript tables against saved results; independent ROC/metric and paired-test calculations from prediction files; and implementation/checkpoint/provenance inspection. A fresh paired bootstrap and targeted primary-literature checks supplement those passes.

- All **49** model/condition prediction files have **18,769** unique utterance IDs, identical ID/label mappings across models and conditions, finite scores within [0,1], **7,520 bona fide** and **11,249 spoof** labels. This validates pairing and prediction-file consistency, not the correctness of labels against the original corpus.
- Independent threshold sweeps reproduce all 49 saved EER and minDCF point estimates to numerical precision: maximum differences approximately 1.3e-16 and 2.1e-16 respectively. Equal-score blocks were kept together in the independent ROC sweep.
- All **42** saved McNemar discordant counts and chi-square statistics reproduce. Manuscript Tables 4.3, 4.6, 4.7 and 4.8 agree with their corresponding primary comparisons, including count direction and p-value rounding.
- Tables 4.2 and 4.4, and the five primary rows of Table 4.5, agree with the repository. Point estimates embedded in Tables 4.9–4.10 also agree to displayed precision. Their original confidence-interval endpoints are not independently recoverable.
- Seven principal checkpoints were loaded and their validation EER metadata inspected. Both fusion checkpoints have identical Stream 1 state across **64 tensors** and identical Stream 2 state across **24 tensors**. C1/A1 Stream 1 cosine similarity, including floating-point buffers, is **0.9961718**, supporting the draft's rounded 0.996 figure. The draft's random-initialization comparison of 0.075 was not reproduced.
- The K=128 PCA has shape **128 × 16,384**, was fitted on **500 samples**, and records **40.711626%** explained variance. Its SHA-256 and the C0 checkpoint SHA-256 match C0's provenance metadata. The assertion that the PCA samples were synthetic noise is documented in the repo, rather than independently established here by repeating the historical fit.
- A new seed-42 paired utterance bootstrap used shared resampling indices across all five primary models and seven conditions: **1,000 EER resamples** and **2,000 minDCF resamples**. This is a fresh sensitivity check, not recovery of the manuscript's original bootstrap experiment.

## Required numerical corrections

### 1. Table 4.5 has fourteen unsupported B1/C0 cells

Replace the B1 and C0 rows with the following, or remove these supplementary rows consistently from the paper. These replacements come from independently recomputed prediction-file metrics.

| Model | Clean | Opus 16k | Opus 32k | Opus 64k | MP3 64k | MP3 128k | AAC 128k |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1, PCA K=64, synthetic fit | 7.58 | 18.32 | 10.52 | 8.67 | 10.59 | 10.14 | 7.87 |
| C0, attention, epoch-12 joint-training checkpoint | 6.76 | 16.39 | 9.19 | 7.49 | 9.53 | 9.09 | 6.93 |

The manuscript currently gives B1 clean EER as 7.0 and C0 clean EER as 0.95. Neither is the clean result of the repo's identified B1/C0 runs. This materially changes the impression of C0's performance.

### 2. Table 4.1 and Section 4.3 use an older C0 validation result

**5.90%** is supported by the saved epoch-11 checkpoint (`val_eer=0.0590376351`). The identified, evaluated C0 checkpoint is epoch 12 and has **5.6640613%**, rounded **5.66%**, validation EER. The assertion that every listed result is the lowest-validation-EER retained checkpoint is inconsistent with using 5.90. Use 5.66 for the evaluated C0, or explicitly identify an earlier diagnostic epoch-11 result separately.

### 3. Model count, definitions and figures must agree

The primary design lists five models, describes a sixth C0 run, and inserts a seventh B1 row without a full methods definition. Figure 4.1 embedded in the manuscript shows only the five primary models, while the text says Figures 4.1–4.2 plot the same data as the seven-row table. Define B1's K=64 **synthetic** PCA and provenance, label supplementary runs separately, and make table/figure coverage explicit. The real-data `pca_k64.pkl` is a different artifact and is not the PCA used by reported B1.

Sections 4.5.4 and 5.2 say B0 has the highest error of any configuration under every condition. This is true among the primary five, but false with B1 included: B1 is worse than B0 on clean, Opus 64k and AAC. Add “among the five primary configurations.” Likewise Section 5.4's claim that no PCA dimensionality comparison was performed becomes false if B1 is retained; a two-value comparison exists, although it is not an extensive controlled sweep.

## Statistical claims requiring qualification

### 4. Significant results are unadjusted, condition-specific findings

All manuscript McNemar numbers reproduce, but no multiplicity correction is disclosed. A Holm correction within each seven-condition comparison yields:

| Comparison | Condition | Unadjusted p | Holm p across its seven conditions |
|---|---|---:|---:|
| C1 vs A1 | Clean | 0.02760 | 0.13798 |
| C1 vs A1 | Opus 16k | 0.00001485 | 0.000104 |
| C1 vs A1 | Opus 32k | 0.001714 | 0.010283 |
| C1 vs A1 | Opus 64k | 0.04333 | 0.17332 |
| C2 vs C1 | Clean | 0.01621 | 0.11347 |
| C2 vs C1 | MP3 64k | 0.03389 | 0.20337 |

Only Opus 16k and 32k remain significant for C1 vs A1 under this correction; no C1/C2 condition does. A1 vs A0 remains significant throughout. This does not automatically invalidate a prespecified individual hypothesis, but the family of hypotheses and correction policy must be stated. Label the present four-of-seven and two-of-seven findings as **unadjusted** rather than general confirmatory conclusions.

### 5. Original bootstrap intervals lack reproducible artifacts

No bootstrap implementation, seed/configuration, resample outputs, or source confidence-interval tables were found in the repository scripts, tests, docs or notebooks. Neither build script generates Tables 4.9–4.10 or the difference intervals in Section 4.5.5. Section 5.3's “all metrics, tables, and figures” reproducibility claim is therefore false as written.

The fresh paired bootstrap broadly reproduces the claimed EER exclusion-of-zero pattern for the two central pairs. For example, C1 minus A1 clean EER CI is **[-0.3417, -0.0331] percentage points**, versus the manuscript's [-0.342, -0.038]. These are compatible fresh estimates, not verification of the exact printed endpoints. AAC and other near-zero endpoints need full-precision reporting; rounding a tiny positive/negative endpoint to zero can change its stated significance classification. Commit the original script and outputs, or explicitly replace the intervals with a documented rerun.

### 6. AAC minDCF significance has not been established by McNemar

Section 4.4.1 says neither AAC result reaches significance. The reported McNemar test concerns correctness at each model's EER threshold; it is not a significance test on minimum DCF. Our separate 2,000-resample paired bootstrap finds AAC **C1 minus A1 minDCF CI [-0.01134, -0.00258]**, excluding zero. This is a fresh, unadjusted utterance-level finding, subject to the same dependence and multiplicity qualifications. Remove the blanket statement and separately identify what quantity each inference tests.

### 7. Constant proportional benefit is descriptive, not demonstrated equality

Discordant-count ratios 1.34–1.44 support “similar observed ratios on clean/Opus.” No test of an interaction, heterogeneity, or equivalence establishes a constant effect or proves that differences in significance arise solely from statistical power. Discordant-count ratios are also not the same estimand as EER ratios. Replace “the advantage is constant” and “did not move significantly” with a descriptive statement and acknowledge that constancy was not tested.

### 8. Conditions are paired, not independent

Section 5.2 calls the seven conditions independent; each uses transformed versions of the same utterances and the same trained models. A cross-condition test must retain that dependence. The split is at utterance level, not speaker level (`build_manifest.py` stratifies label lists). Speaker/source-recording overlap across train/test has not been measured because the manifest is absent. Utterance-level bootstrap and McNemar do not account for within-speaker/source clustering. Disclose this and inspect speaker/source overlap before making generalization claims.

### 9. Non-significance does not establish equality or absence of benefit

“Fusion does not help here,” “effectively tied,” and “does not outperform concatenation” are stronger than failing to reject a difference. Use “no statistically significant benefit was detected under the stated unadjusted test.” The paper itself records a nominal attention advantage in two conditions; the heading should reflect that limited evidence.

## Methods and interpretation corrections

### 10. Fusion training history remains unknown

Sections 3.6, 4.4, 4.5.2 and 5.4 assert frozen transplantation and clean-only fusion training as facts. The surviving training invocation is explicitly missing. Identical final stream states support shared encoders and are consistent with freezing, but do not prove the optimization history. High cosine similarity to A1 does not uniquely establish a source checkpoint or codec-augmented training lineage. Present clean-only training as an unverified hypothesis, and avoid using it as an established cause of the MP3/AAC pattern. Record actual training settings if recovered.

C1 vs A1 also changes Stream 1 weights, not just adds fusion. Thus it supports comparison of these trained systems, not strict causal isolation of the statistical stream. C1/C2 have shared stream states, making their architectural comparison better controlled, although single-run head training remains a limitation. A0/A1's augmentation conclusion is strong empirically, but identical architecture/data alone does not establish every historical training setting was matched.

### 11. The fusion classifier is not one linear layer

Section 3.5's classification description omits the actual Setup C head:

`320 → Linear(128) → BatchNorm → SiLU → Dropout(0.4) → Linear(1) → Sigmoid`.

The attention internals and final 320-dimensional concatenation otherwise match current source. Attention operates over two modality tokens; descriptions of directly weighting individual raw embedding components or locating specific forensic anomalies are interpretations, not measurements of attention behavior.

### 12. Feature extraction is partly CPU-bound

Sections 3.1/3.6 say all feature extraction was accelerated using T4 GPUs. `src/stream2/stream2.py` explicitly transfers waveforms to CPU NumPy for bispectrum estimation and PCA, then transfers their outputs back. Say neural computation ran on the recorded GPU hosts while bispectrum/PCA ran on CPU. A1's preserved log establishes CUDA, not the exact T4 device for every historical run.

### 13. PCA does not reduce bispectrum extraction complexity

PCA occurs after the matrix has been computed. It reduces the MLP's input size and parameter count, not extraction cost. The implemented estimator operates on fixed 256-sample blocks and averages them; its cost is proportional to frame count times the computed bifrequency region, rather than a full 64,000 × 64,000 matrix. Revise Sections 2.4, 3.4.3 and 5.3 accordingly. “Immediate memory exhaustion” depends on hardware/dtype and should be replaced by an explicit impractical memory estimate if retained.

### 14. Stream 2 is a composite, not a bispectrum-only experiment

B0 combines LFCC and bispectrum/PCA features. No LFCC-only, bispectrum-only, or matched MFCC control underpins the paper's attribution of B0 degradation to phase coupling, or a demonstrated LFCC-versus-MFCC advantage in this implementation. Sections 1.3, 2.1 and 5.2 should distinguish proposed questions from answered questions. The results establish the behavior of the composite statistical stream; they do not isolate persistence of a particular cue.

Balancing feature dimensions also does not guarantee balanced loss contribution or retained speech-forensic variance. The noise-fitted PCA's 40.7% is variance in its synthetic fitting data, not in training speech. Section 3.4.3 already discloses the noise fit and historical estimator mismatch; retain that disclosure and qualify the rationale.

### 15. Complete the bispectrum-region description

Section 3.4.3 currently says “only the lower-triangular half of the valid region is computed at zero,” which is incomplete. Use: “Only bins satisfying j ≤ i and i+j < 128 are computed; all other entries are zero. PCA was fitted using an earlier estimator that populated the full i+j < 128 region.” The current output shape and Hamming/256-sample/50%-overlap description match code.

### 16. Add actual audio preprocessing details

The canonical train/evaluation datasets keep the **first 64,000 samples** for long utterances and zero-pad short ones. They do not use the random/center crop behavior of the separate `process_waveform` helper. State first-segment truncation explicitly. No amplitude-normalization transform beyond decoding/scaling is evident in the active dataset paths; avoid suggesting additional normalization. Codec augmentation attempts a transcode with probability 0.5, but silently falls back to clean audio on failure; the achieved rate requires success/failure logs.

### 17. MinDCF is an optimized diagnostic, not a validated deployment point

The Track 1 costs/prior are correct, and standalone minDCF is appropriate. The normalized objective used here is `1.9 × P(bona rejected) + P(spoof accepted)`. Its minimum is chosen using labeled test scores, like the test-specific EER thresholds. It does not validate one deployable threshold. A10-to-1 conditional cost ratio does not mean spoof acceptance dominates aggregate weighted risk: the prior-weighted terms are 0.50 versus 0.95. Qualify “security-oriented operating point.”

The ASVspoof evaluation plan explicitly supports Track 1 minDCF with the quoted costs/prior: [official evaluation plan](https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf). The paper's substitution of minDCF for tandem t-DCF is justified.

### 18. Do not claim an upper bound on unknown-attack EER

Section 5.3 calls the known-attack EER an upper bound on official-protocol EER. Lower EER is better, so that wording reverses the intended optimism; neither a numerical upper nor lower bound has been established. Use “may overestimate performance under unknown attacks; official-protocol performance was not measured.” Also use “87,585 training utterances from a 125,123-utterance pool” rather than “trained on 125,123 utterances,” which includes held-out samples.

### 19. Unsupported numerical/mechanistic historical claims need records

The narrative of 36–41% initial EER, an inverted split, bottleneck-induced failure, and validation of the manifest before every run is not fully substantiated by the current artifact set. Partial logs support some evaluations, not a universal every-session statement. Claims that pipeline repair confirmed a unique cause need the before/after commands, splits and checkpoints. Preserve these as documented project history only where source records can be cited; otherwise qualify or remove them.

## External-source checks and scope

The primary RawNet2 reference supports the Table 2.1 fusion figures **1.12% EER / 0.0330 min t-DCF**; these are a fused system's results, not standalone RawNet2. [RawNet2 paper](https://arxiv.org/html/2011.01108).

The AASIST figures **0.83% / 0.0275**, and AASIST-L **0.99% / 85K parameters**, are supported; the first pair is the **best single run**, while the paper also reports across-run averages. [AASIST paper](https://arxiv.org/html/2110.01200).

The WaveFake phone-channel robustness account is supported, but calling a GMM a “simple linear model” is incorrect; use “Gaussian mixture classifier.” Frequency findings are generator-dependent, not proof that all detectors depend exclusively on high frequencies. [WaveFake paper](https://arxiv.org/html/2111.02813).

Barrington's primary publication supports the human-voice identification figure **67.4%** and approximately **60%** AI-voice identification. [Scientific Reports article](https://www.nature.com/articles/s41598-025-94170-3). FakeSound supports human binary accuracy below **0.6**, but that does not establish that all generative audio has crossed an uncanny valley or that human verification is universally unviable. [FakeSound paper](https://arxiv.org/html/2406.08052).

The Sun vocoder-artifact and Zhang robustness papers are relevant primary sources: [Sun et al.](https://arxiv.org/html/2304.13085), [Zhang et al.](https://arxiv.org/html/2411.00121). Their existence/relevance does not validate every extrapolation in the draft.

The AlBadawy paper's title/authorship and bispectral synthesis-detection premise were checked against its [CVF record](https://openaccess.thecvf.com/content_CVPRW_2019/html/Media_Forensics/AlBadawy_Detecting_AI-Synthesized_Speech_Using_Bispectral_Analysis_CVPRW_2019_paper.html). Its full PDF was blocked by the source (403), so this audit does **not** certify the exact 1,845-recording, 0.99 AUC or recompression/noise figures. Those need a readable original source. LFCC's default framing description is consistent with the documented library interface, but the installed environment here lacks torchaudio, so the historical runtime was not executed. [LFCC documentation](https://docs.pytorch.org/audio/2.1.2/generated/torchaudio.transforms.LFCC.html).

This is not a completed bibliographic verification of every reference, author list, DOI and literature paraphrase. Literature claims beyond the targeted checks above remain unverified. Chapter 2 repeatedly turns design motivation into universal proof; qualify claims such as “superior,” “indispensable,” and “prevents domination” to match the cited experiment's scope.

## Remaining reproduction limits

`data/manifest.csv`, corpus metadata and audio were not present in this checkout. Consequently the full split hash, train/test disjointness, speaker/source overlap, original labels, successful codec transformations and fresh checkpoint-to-audio predictions could not be checked. A matching ID/split hash also does not authenticate labels, audio bytes, attack IDs, or codec settings. Save additional checksums/provenance for those inputs.

The checkpoints are readable locally, but remote Git LFS quota status was not checked; remove or date the manuscript's live quota claim unless supported by a current remote check. The repo has multiple legacy metric/training implementations. The audited tables use `scripts/build_results.py`, while McNemar binarization uses interpolated EER from `src/utils/metrics.py`. Their algorithms differ slightly; document both and pin runtime versions. The listed environment lacks an explicit pandas dependency despite the canonical builders requiring it. Reproduction should be tested in a clean, pinned environment before claiming it is turnkey.

No full training rerun, GPU latency validation, original confidence-interval recovery, or live application test was performed. No full Word layout rendering was performed; the scientific text/tables were extracted and the embedded EER figure was inspected. This audit establishes score-file arithmetic and identifies provenance/claim gaps; it does not authenticate the complete experimental history.

## Saved evidence

- `check_evidence.py`, `check_output.txt`: independent point metrics, pairing checks, all saved McNemar counts, Holm sensitivity analysis and checkpoint comparisons.
- `independent_metrics.csv`: 49 independently computed results and prediction-file SHA-256 hashes.
- `independent_mcnemar.csv`: 42 comparisons with survival-function p-values and seven-condition Holm adjustment.
- `check_bootstrap.py`, `bootstrap_output.txt`, `fresh_bootstrap_differences.csv`: documented fresh paired resampling results. Do not cite these as the manuscript's original bootstrap run.
- `manuscript_extracted.txt` and `image*.png`: review aids extracted from the supplied manuscript. Paragraph labels are extraction indices, not Word page numbers.

Recommended submission sequence: correct B1/C0 numbers and run definitions; soften unsupported training and causal explanations; document inference/multiplicity policy; recover or regenerate original CI artifacts; verify and archive the manifest/audio provenance; then regenerate tables/figures and inspect the final manuscript for consistency.

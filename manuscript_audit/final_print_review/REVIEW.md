# Final print manuscript review

Reviewed 8 October 2026 against the current local project checkout at commit `9e9f350`, its existing working-tree contents, historical prediction outputs, checkpoints, and returned demo verification artifacts.

**Verdict: the principal reported point estimates are correct, but the manuscript is not completely correct or fully aligned with the current project. Revise the statistical conclusions, reproducibility account, and several methods statements before printing or submission.**

Document: `C:/Users/Rafa/Downloads/Hearing_Reality_FINAL_for_print.docx`.
SHA-256: `1e23124eb325a1c17b2eb5c0edc4a842a76b47aecd6a23473385e708c00eba0e`.
This review concerns this exact print version, not the earlier draft audited on 6 October. The original Word document was not edited. Embedded document text was treated as content to review, not instructions to execute.

## Evidence that passed

- All 49 historical model/condition CSVs contain 18,769 unique utterances, with identical ID/label mappings, 7,520 bona fide and 11,249 spoof labels, and finite scores in [0,1].
- Independent ROC sweeps, keeping equal-score groups together, reproduce all 49 EER and minDCF values. Maximum absolute differences from saved tables are approximately 1.3e-16 and 2.1e-16.
- Automated comparison of 154 printed EER/minDCF cells in Tables 4.4, 4.5, 4.9 and 4.10, including the point estimates printed beside confidence intervals, passes at the displayed precision. Tables 4.1 and 4.2 also match their identified checkpoint/clean-test values.
- All 42 saved McNemar discordant counts and chi-square statistics reproduce from the saved scores and the repository's interpolated EER thresholds. The manuscript's principal tables and supplementary B1 narrative match these results.
- The previously incorrect B1/C0 rows are corrected. C0's 5.66% validation EER and 6.76% clean test EER now match the identified epoch-12 checkpoint and prediction CSV.
- The recovered `outputs/returned_parity_20261007/manifest.reference.csv` contains 125,123 unique utterance IDs. Train/validation/test sizes and class counts match the manuscript. Its full split digest is `ed4808bb0456de26cb04df66ae82219365d057cf172c82fd674d2dab093a8eec`; the paper prints its 16-character prefix.
- Historical prediction IDs and labels match the recovered manifest's test partition. All sixteen attacks have 4,687 utterances in the overall pool, and all appear in test. This verifies the returned manifest, not its underlying source metadata/audio bytes.
- Seven principal checkpoints were loaded. Their validation metrics, stored epochs, optimizer learning rates, weight decays, and optimizer step counts support Table 3.2's recorded values and conditional batch-size inferences. C1/C2 have identical Stream 1 state across 64 tensors and identical Stream 2 state across 24 tensors. Optimizer evidence supports frozen-stream head training at the saved stage, with the provenance qualifications already included in the revised methods.
- A fresh seed-42 paired utterance bootstrap completed with 1,000 EER and 2,000 minDCF resamples. Scripts, intervals, and resample arrays are saved beside this review. This is a new sensitivity check, not verification of the original exact interval endpoints.
- All twelve embedded figures were extracted and visually inspected. Result plots show the intended five primary configurations and broadly agree with the tabulated data. Several architecture labels need correction, described below.

The verified historical EER table is:

| Model | Clean | Opus 16k | Opus 32k | Opus 64k | MP3 64k | MP3 128k | AAC 128k |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 1.54 | 9.48 | 3.83 | 1.56 | 5.77 | 5.87 | 1.62 |
| A1 | 1.00 | 2.95 | 1.65 | 0.96 | 1.60 | 1.53 | 1.01 |
| B0 | 7.50 | 18.35 | 11.53 | 8.26 | 13.43 | 12.81 | 7.54 |
| B1 | 7.58 | 18.32 | 10.52 | 8.67 | 10.59 | 10.14 | 7.87 |
| C0 | 6.76 | 16.39 | 9.19 | 7.49 | 9.53 | 9.09 | 6.93 |
| C1 | 0.81 | 2.38 | 1.32 | 0.80 | 1.70 | 1.62 | 0.89 |
| C2 | 0.92 | 2.47 | 1.33 | 0.86 | 1.78 | 1.69 | 0.96 |

## Corrections affecting the central conclusions

### 1. Disclose the LFCC correction and distinguish historical research from current inference

**Locations:** Sections 3.4.1, 4.4's regeneration statement, 5.3's reproducibility paragraph, and any description connecting thesis results to the current demo.

Commit `9e9f350` changes LFCC extraction from `[batch,time]` to `[batch,channel,time]` before torchaudio's transform. This preserves per-recording decibel clipping; the earlier version could depend on the other recordings in a batch. `docs/demo_verification.md` expressly preserves the original research CSVs as historical outputs.

Thus the saved results remain arithmetically supported, but they cannot be assumed to represent fresh inference under the corrected source. Pin the historical code revision, environment, evaluation batch size, and ordering if retaining those results. For an updated experiment, score all models/conditions under a documented corrected protocol and regenerate the analyses in a separate result set.

The returned parity reports pass for C1/A1 on 256 selected test clips per model across batch sizes/orderings and 256 clean validation cache clips. Those reports establish agreement between corrected inference paths, not reproduction of the full historical five-model test results. They cannot support the paper's statement that all clean results and p-values were regenerated exactly unless a separate full-run record is supplied.

The historical batch dependence also deserves a limitation: a prediction may have been influenced by batch companions. It is not evidence of label leakage by itself, but it complicates treating each historical score as an independent per-utterance output.

### 2. Label significance as nominal and state a multiple-comparison policy

**Locations:** Section 3.7; Tables 4.3, 4.7 and 4.8; Abstract; Sections 4.4–4.6 and 5.2.

The displayed p-values are correct, but the manuscript interprets several families of tests without discussing multiplicity. As a sensitivity analysis, Holm correction within each seven-condition comparison gives:

| Comparison and condition | Printed unadjusted p | Holm p within its seven conditions |
|---|---:|---:|
| C1 vs A1, clean | 0.02760 | 0.13798 |
| C1 vs A1, Opus 16k | 0.00001484 | 0.000104 |
| C1 vs A1, Opus 32k | 0.001714 | 0.010283 |
| C1 vs A1, Opus 64k | 0.04333 | 0.17332 |
| C2 vs C1, clean | 0.01621 | 0.11347 |
| C2 vs C1, MP3 64k | 0.03389 | 0.20337 |

Under this family definition, only Opus 16k/32k remain significant for C1 versus A1; no condition remains significant for attention versus concatenation. A1 versus A0 remains significant throughout. Other prespecified family definitions may be defensible, but must be disclosed. Holm does not require independence between conditions.

Suggested wording: “At the unadjusted 0.05 level, C1 versus A1 differs under clean audio and all three Opus settings. With Holm correction across those seven tests, the Opus 16k and 32k findings remain significant.” Preserve nominal values and add corrected results or explicitly frame the analyses as exploratory.

### 3. The original bootstrap endpoints and exact agreement claim are not reproducible

**Locations:** Tables 4.9–4.10; Section 4.5.5; Section 5.3.

The existing build scripts do not generate these interval tables or the difference intervals in prose. The original seed, implementation and resample outputs were not found in the project artifacts inspected. The point estimates pass; the original endpoints are not certified.

The new paired bootstrap broadly supports the main EER pattern. For example, C1 minus A1 clean EER has CI **[-0.341708, -0.033090] percentage points**, compared with the paper's [-0.342, -0.038]. Both indicate the same nominal direction, but are different endpoints.

Remove “corroborate every significance result ... without exception” unless every relevant comparison and metric has a reproducible, specified result. EER bootstrap and McNemar concern different quantities, and they use the same observations: they are complementary analyses, not independent evidence from separate data. A bound printed as `+0.000` cannot establish whether the exact bound is positive, negative, or zero; show sufficient precision near zero.

Recover the original bootstrap source and outputs, or adopt a documented rerun and replace all corresponding tables/prose consistently. The fresh files here are available for that decision, but were not silently substituted into the manuscript.

### 4. The AAC minDCF significance statement is unsupported

**Location:** Section 4.4.1: “Since neither result reaches significance.”

McNemar at test-derived EER thresholds is not a test on minimum DCF. The new paired bootstrap gives AAC C1 minus A1 minDCF CI **[-0.011340, -0.002581]**, excluding zero at the nominal level. This does not establish a multiplicity-adjusted or speaker-cluster-adjusted finding, but demonstrates why the blanket non-significance statement is inappropriate.

State separately: “The AAC EER-threshold McNemar comparison is not significant (p=.121). The significance of the minDCF difference requires a separate specified analysis.” If using the new bootstrap, report its result and qualifications directly.

### 5. Constant proportional benefit was not tested

**Locations:** Abstract; Sections 4.5.2, 4.6 and 5.2.

Similar observed discordant ratios of 1.34–1.43 do not prove a constant effect, absence of an interaction, or that varying significance arises exclusively from statistical power. Discordant ratios are not EER ratios. No heterogeneity/equivalence test is reported.

Use: “The observed advantage has similar proportions under clean audio and the three Opus settings; whether it varies across conditions was not formally tested.” Also restrict any claim about widening absolute margins to the actual pattern: clean margin is 0.1864 points, Opus 16k 0.5704, Opus 32k 0.3307, and Opus 64k 0.1598. It does not widen under every compressed condition.

### 6. Non-significance does not prove no benefit or equivalence

**Locations:** Sections 4.4.1, 4.5.2, 5.2; Abstract/conclusion summaries.

Replace “fusion does not help here,” “effectively tied,” and categorical “does not extend to ... AAC” with the quantity tested: “No statistically significant EER-threshold benefit was detected under the stated unadjusted McNemar test.” AAC point estimates favor C1 under both EER and minDCF. MP3 64k has a tiny minDCF advantage for C1 despite an EER disadvantage. Equivalence requires a prespecified meaningful margin and appropriate test.

### 7. Conditions and utterances require dependence qualifications

**Locations:** Section 5.2's “seven independent conditions”; Sections 3.7, 4.5.5 and 5.3.

The conditions are repeated codec transformations of the same utterances and are not independent. Replace “independent” with “paired acoustic conditions.”

The recovered manifest has 1,181 training speakers, 1,140 test speakers, and **1,138 speakers shared by train/test**. **18,766 of 18,769 test utterances** belong to speakers also present in training. The known-speaker scope is supported, but “known-speaker” should not imply literally every test speaker is known.

Within-speaker/source dependence is not addressed by utterance-level McNemar or the current utterance bootstrap. Add this limitation; speaker/source-aware resampling is a useful robustness check. Test-derived thresholds also make the hard predictions data-adaptive: nominal McNemar p-values should not be described as proving algorithm-level performance differences under arbitrary future data.

### 8. Stream 2 results do not isolate phase coupling or LFCC stability

**Locations:** Specific Problems/Objectives 2–4; Sections 4.5.4 and 5.2.

B0 combines LFCC, bispectral magnitude, a noise-fitted PCA, and an MLP. No LFCC-only, bispectrum-only, MFCC comparison, frequency ablation, or vocoder-identification task isolates the cue-level mechanisms described in the objectives. A binary spoof detector's degradation does not directly measure its ability to identify vocoder fingerprints.

Report behavior of the **composite statistical stream**. Describe cue-specific questions as partially addressed or unresolved. The revised paper correctly admits the missing MFCC control, but still attributes results to “this particular implementation of bispectral analysis” too narrowly.

## Methods and factual corrections

### 9. Correct the fusion classifier description

**Location:** Section 3.5, Classification; architecture figures.

Setup C uses `320 → Linear(128) → BatchNorm → SiLU → Dropout(0.4) → Linear(1) → Sigmoid`, not a single final linear layer. The code and saved classifier weights support this.

The detailed two-token attention paragraph is substantially accurate. However, figures labeling it “query=E_stat, over E_raw” omit that the key/value sequence contains **both raw and statistical projected tokens**. Attention weights modalities in that token sequence; the manuscript's descriptions of locating anomalies or directly weighting particular raw coordinates are unmeasured interpretations. Correct Figures 2.1, 2.3 and 3.4 accordingly. Figure 2.4 should not suggest that McNemar establishes significance of minDCF.

### 10. Qualify the GPU account

**Locations:** Sections 3.1 and 3.6.

The bispectrum/PCA path explicitly transfers waveforms to CPU NumPy and transfers features back. “All ... feature extraction ... utilizing NVIDIA T4 GPUs” is inaccurate. The revised methods also say all models used T4 while admitting device records survive only for B1.

Use: “Neural training used CUDA where documented. Bispectrum estimation and PCA ran on CPU; the exact GPU model is documented for B1 and was not independently verified for every run.”

### 11. Correct the PCA computational claim

**Locations:** Sections 2.4, 3.4.2–3.4.3 and 5.3.

PCA occurs after bispectrum extraction. It reduces the downstream MLP input and parameter count, not the cost of computing the bispectrum. The implemented estimator uses fixed 256-sample windows and an admissible bifrequency subset; with fixed window size its work scales with the number of frames. A hypothetical full-length N×N representation is not what the code computes.

Delete Section 5.3's statement that PCA mitigates O(N²) **extraction** cost. Replace “immediate memory exhaustion” with a conditional memory estimate or “impractical full-length representation.” Balancing dimensions also does not guarantee balanced loss contribution. “Chosen to avoid an excessively large MLP input” is supported; “prevents any single modality from dominating the loss” is not.

### 12. State actual crop/pad behavior and augmentation failure handling

**Location:** Sections 3.2–3.3.

The active train/evaluation dataset paths keep the **first 64,000 samples** of long recordings and zero-pad short recordings, rather than arbitrary segmentation. Decoding/scaling is visible; an additional amplitude-normalization transform is not. Specify those details.

Codec augmentation attempts transcoding with probability 0.5 and silently falls back to clean audio on failure. Report the attempt probability; without failure logs the achieved augmentation rate is unknown.

### 13. Correct cosine similarity terminology and avoid asserting lineage from it

**Locations:** Sections 3.6 and 5.3.

The printed **0.996** is reproduced when concatenating Stream 1's floating-point parameters **and buffers**, including BatchNorm statistics. Using only named trainable parameters gives approximately **0.999999994**. No parameter tensor is exactly identical, but the definition of the reported similarity matters. Scale differences, especially Sinc frequencies versus other weights, can dominate a global cosine.

Write “cosine similarity 0.996 over floating-point encoder state, including buffers,” and report the precise method. The random-initialization 0.075 baseline lacks an identified seed/output here. Remove it or preserve a reproducible calculation. High similarity is consistent with related initialization/training; it does not prove the exact checkpoint source or codec-training lineage.

The optimizer evidence is stronger than in the older draft and supports the revised freezing account. Retain its qualification about earlier history/reset states. C1 versus A1 remains a comparison of trained systems, not a strict ablation that holds the waveform encoder literally identical.

### 14. Fix the training-pool and hash wording

**Locations:** Abstract; Section 5.1; Sections 3.2.1 and 4.2.

“Trained on ... 125,123” includes held-out validation/test utterances. Use: “A pool of 125,123 utterances was partitioned into 87,585 training, 18,769 validation and 18,769 test utterances.”

`ed4808bb0456de26` is a truncated digest, not the full SHA-256 digest. Explain the truncation or print the full value. The hash authenticates ID/split mappings, not labels, audio contents or codec settings. Claims that it was checked before **every** evaluation session need the corresponding records; partial surviving logs do not establish the universal claim.

### 15. Correct two explicit numerical/generalization statements

**Locations:** Abstract, Sections 4.5.4 and 5.2; Section 5.3.

“B0 has the highest error of any configuration under every condition” is false once supplementary B1 is included: B1 is worse on clean, Opus 64k and AAC. Add **“among the five primary configurations”** consistently.

The C1/B0 chi-square values are not an order of magnitude greater than **any** other comparison. For example, 2664.30 divided by A1/A0's 873.37 is about **3.05**, not ten. Replace with “larger than the other primary comparisons.” Chi-square itself should not be treated as a directly comparable performance effect size.

Calling the known-attack EER an **upper bound** on official-protocol EER reverses the intended direction, and no formal bound is established. Use: “Known-attack evaluation may overestimate performance on unseen attacks; official-protocol performance was not measured.”

### 16. Clarify what minDCF establishes

**Locations:** Sections 4.4.1 and 5.2.

The Track 1 substitution is appropriate and the parameters match the [official evaluation plan](https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf). The normalized objective is `1.9 × P(bona fide rejected) + P(spoof accepted)`. A tenfold conditional cost of spoof acceptance does not imply that term dominates prior-weighted risk: the unnormalized weights are 0.95 versus 0.50.

minDCF is the best cost found by sweeping labeled test scores, not a validated fixed deployment operating point. The difference between B0's EER and minDCF degradation does not uniquely establish that errors concentrate on spoof acceptance. Report per-class rates at a specified threshold before making that mechanistic inference. The manuscript already calls it an inference, but its explanation of which term dominates needs correction.

### 17. Update the project-wide reproducibility and deployment scope

**Location:** Section 5.3.

“All metrics, tables, and figures ... by running the two build scripts” is too broad: the original confidence intervals and several training-history statistics are outside those builders. Separate table regeneration from checkpoint/audio inference and training reproduction.

The current project includes validation-derived demo thresholds and passing returned parity reports. The paper's statement that score/threshold calibration was not evaluated can remain a description of the historical research experiment, but should distinguish that experiment from the subsequent validation-threshold work. Threshold selection is not probability calibration.

At the clean validation threshold, the later corrected C1 evaluation flags 4.189% of genuine MP3-64 and 3.883% of genuine MP3-128 utterances. These are **validation fixed-threshold rates**, not replacements for the thesis test EERs. They illustrate why the thesis's ≤2.4% condition-specific EER claim cannot describe arbitrary demo uploads.

The live LFS quota claim was not verified remotely. Remove it or date it to a verified incident. No inference latency result was found in the inspected artifacts; the manuscript's no-latency limitation remains appropriate, while the code's existence alone does not establish speed.

### 18. Qualify historical pipeline causation

**Location:** Section 4.2.

The early 36–41% results, inverted split, and assertion that repairs confirmed a unique cause need preserved before/after run records. Low throughput itself does not cause weak discrimination if training is otherwise completed. Multiple repairs and training changes do not isolate a single cause. Present the account as project history where supported, without claiming a controlled causal diagnosis.

## Literature and reference review

Several important citations support the revised narrative, but the literature still contains overstatements and bibliographic issues.

- **WaveFake:** A GMM is not a “simple linear model.” Use “Gaussian mixture classifier.” The simulated-phone finding is supported, but high-frequency reliance is generator-dependent: the paper also observes detectors relying on lower frequencies. Avoid “indispensable” as a universal conclusion. [Primary paper](https://arxiv.org/html/2111.02813).
- **FakeSound:** Below-60% human performance is supported for its study, but it does not demonstrate that all generative audio passed an “uncanny valley” or that human verification is universally unviable. [Primary paper](https://arxiv.org/html/2406.08052).
- **RawNet2:** Table 2.1's 1.12%/0.0330 figures correctly describe fusion with LFCC-GMM; 0.1175 is supported for the best standalone variant. Our implementation is an adaptation, not the unchanged published architecture. [Primary paper](https://arxiv.org/html/2011.01108).
- **AASIST:** 0.83% EER, 0.0275 min t-DCF and the 85K-parameter AASIST-L result are supported. Label the first pair **best single run**; the paper also reports across-run averages. [Primary paper](https://arxiv.org/html/2110.01200).
- **Barrington:** 67.4% real-voice and 60.8% AI-voice identification are supported for the naturalness task. [Original article](https://www.nature.com/articles/s41598-025-94170-3).
- **Haq:** The cited work reasons about emotional inconsistencies across audio/video and provides textual explanations. It does not establish that binary classification itself is inadequate, or that deterministic LFCC/bispectrum features alone provide comparable explanations. Cite it as motivation and retain the paper's explicit local definition of “neuro-symbolic.” Its institutional record identifies Article **341**; use a consistent article-based citation. [Author institution record](https://pure.skku.edu/en/publications/multimodal-neurosymbolic-approach-for-explainable-deepfake-detect/).
- **Lu:** The reference gives the first author's name as **Yuang**; the publisher gives **Yuhang Lu**. Its image/video scope is correctly acknowledged in the revised prose. [Publisher article](https://link.springer.com/article/10.1186/s13640-024-00621-8).
- **Liu:** “ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild” is cited as **2021**, but the cited title/author list corresponds to the **2022 arXiv** paper and subsequent **2023 journal** publication. Choose one version and correct year/venue/link throughout. [Official challenge repository reference](https://github.com/asvspoof-challenge/2021), [arXiv](https://arxiv.org/abs/2210.02437).
- **Baltrušaitis:** The journal issue 41(2), 423–443 is the 2019 issue; 2018 is the early-online year. Make the chosen publication-version convention consistent.
- **Sun, Zhang, Marek, Di Pierno, Pham and Siopi:** The primary-source identities and broad relevance were checked. They motivate the design, but do not validate unmeasured phase persistence, localization of anomalies, guaranteed loss balance, or real-time efficiency in this experiment. [Sun](https://arxiv.org/abs/2304.13085), [Zhang](https://arxiv.org/html/2411.00121), [Marek](https://arxiv.org/abs/2412.17924), [Di Pierno](https://arxiv.org/abs/2504.20923), [Pham](https://arxiv.org/abs/2407.01777), [Siopi](https://arxiv.org/abs/2212.01128).
- **AlBadawy:** The publication identity and general bispectral premise were found, but repeated attempts to access the full original PDF failed. The specific 1,845-recording, 0.99-AUC and recompression/noise figures are therefore **not independently certified in this review**. Check them against a readable original before submission. [Author-hosted PDF](https://farid.berkeley.edu/downloads/publications/cvpr19/cvpr19b.pdf).

The standard statistical/attention/PCA references were inspected for their role in the methods, but this review does not certify every page range, DOI, author-list spelling, or claim in all 27 bibliography entries. Publisher access limitations and the inaccessible bispectrum PDF leave a bibliographic verification boundary.

## Print and editorial checks

Table 3.2 is present in the body but absent from the List of Tables. Update that list and the Word table of contents/page references after final edits. Some List of Figures captions differ from the body captions; synchronize them.

The extracted architecture diagrams omit the classifier hidden layer and simplify the attention key/value sequence, as noted above. Plot data look consistent with the saved primary results, but near-overlapping C1/C2 curves do not establish equivalence.

Full-page DOCX rendering was attempted with the bundled document renderer. It could not run because LibreOffice is not installed/available. Consequently page breaks, clipping, actual table widths, page-number accuracy and final print layout are **not certified**. Extracted image inspection is not a substitute for that check.

## Review boundary and next steps

The original corpus audio and metadata were not available locally. No full fresh checkpoint-to-audio test run, codec-byte audit, source-recording overlap analysis, or independent authentication of the recovered manifest against corpus metadata was possible. The returned parity reports are preserved external execution evidence, not a new execution performed here.

The most defensible current summary is: **The historical results show a large, consistent advantage for codec-augmented RawNet2 over its unaugmented control. C1 has modest nominal advantages over A1 on clean/Opus conditions, with the strongest multiplicity-adjusted evidence on Opus 16k/32k. Attention has consistently lower historical EER than concatenation, but its nominal condition-specific significance does not survive the illustrated seven-test Holm correction. These findings describe single trained systems under an in-domain, mostly known-speaker protocol and must be distinguished from corrected demo inference.**

Prioritize LFCC/provenance disclosure, the bootstrap record, statistical qualification, and the methods corrections. The verified historical point estimates do not need arbitrary replacement.

Audit evidence files: `verification_output.txt`, `verified_metrics.csv`, `verified_mcnemar.csv`, `manifest_verification.txt`, `bootstrap_output.txt`, `fresh_difference_intervals.csv`, `fresh_individual_intervals.csv`, and `fresh_bootstrap_samples.npz`. The runnable checks are `verify.py` and `bootstrap.py`. Original source code, results and the Word manuscript were left unchanged.

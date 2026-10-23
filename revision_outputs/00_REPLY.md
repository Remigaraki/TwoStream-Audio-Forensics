All checked principal checkpoint epochs, optimizer steps and rounded validation EERs match the megaprompt and the identified thesis checkpoints. The megaprompt's general claim of learning rate 1e-4 does not apply to every saved optimizer; the thesis already qualifies those rates.

| Request | Status | Where | Finding |
| --- | --- | --- | --- |
| S1 demo rename | NOT IMPLEMENTED: information-only scope | 01_demo_rename/ | Exact label locations and terminology supplied; no edits or tests performed. |
| D2 logs A0 | NOT FOUND locally | 02_logs/raw/A0_NOT_FOUND.txt | No completed epoch summaries recovered. |
| D2 logs A1 | PARTIAL | 02_logs/raw/A1_partial_batch_excerpt.txt | Kaggle/CUDA command and start of log epoch 14; no epoch-completion summary. [log] |
| D2 logs B0 | NOT FOUND locally | 02_logs/raw/B0_NOT_FOUND.txt | No completed epoch summaries recovered. |
| D2 logs B1 | PARTIAL | 02_logs/parsed/B1.csv | Log epochs 4-10 recovered from Downloads ZIPs; later best checkpoints match repo by SHA-256. [log; checkpoint] |
| D2 logs C0 | NOT FOUND locally | 02_logs/raw/C0_NOT_FOUND.txt | Evaluation artifacts exist; training history not recovered. |
| D2 logs C1 | NOT FOUND locally | 02_logs/raw/C1_NOT_FOUND.txt | Corrected-resume notebook does not establish original training history. |
| D2 logs C2 | NOT FOUND locally | 02_logs/raw/C2_NOT_FOUND.txt | No completed epoch summaries recovered. |
| D2 checkpoint check | DONE for principal checkpoints; lineage PARTIAL | 03_checkpoints/ | Principal values match. Alternative B input width differs from reported B0; PCA K=64 is consistent with architecture but run identity unknown. [checkpoint] |
| D1 sample size | PARTIAL | 04_sample_size/ | Hash/counts verified; 4,687-per-attack rationale undocumented; floor(75,000/16) is arithmetic-compatible only. [computed from manifest; inference] |
| D3 hardware | PARTIAL | 05_hardware/ | Kaggle/CUDA in A1/B1 logs, T4 in B1 notebook session; other training GPUs and measured minimum resources remain unknown. [log; notebook] |
| Figures | DONE for recovered evidence | 06_figures/ | Partial B1 two-panel history and saved-checkpoint plot; no fabricated full history. |

Read `REVISION_DATA.md` for all verified tables, proposed manuscript wording, qualifications and sources. Every numeric data table has an identified source or explicit estimate/unknown status. No code, checkpoint, result file, manuscript or original Downloads file was changed; no model was retrained or reevaluated. External Kaggle, W&B, Colab/Drive and group-message histories were inaccessible and not inspected.

## Open items for Rafa

- Original metadata and missing saved training logs/notebook versions.
- Firsthand historical cap rationale, actual compute constraint, final endpoints and platform/device records.
- Alternative B run identity/use and dated C0 restart/resume history.
- Numeric future-resource recommendations if the team wishes to supply them as its judgment.

Revoke and rotate the GitHub token reported exposed in the Colab CONFIG cell.

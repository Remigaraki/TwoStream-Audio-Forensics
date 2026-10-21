| run | file | stored epoch (0-based) | val_eer_percent | reported_checkpoint | source |
| --- | --- | --- | --- | --- | --- |
| A0 | checkpoints/setup_a/best_a0_epoch2_eer0859.pt | 1 | 8.58892776284606 | False | [checkpoint: checkpoints/setup_a/best_a0_epoch2_eer0859.pt] |
| A0 | checkpoints/setup_a/best_a0_epoch7_eer0267.pt | 6 | 2.674332872141826 | False | [checkpoint: checkpoints/setup_a/best_a0_epoch7_eer0267.pt] |
| A0 | checkpoints/setup_a/best_a0_final_eer0156.pt | 10 | 1.5557724516410916 | True | [checkpoint: checkpoints/setup_a/best_a0_final_eer0156.pt] |
| A1 | checkpoints/setup_a/best_a_ep13_eer0105_0713_0401.pt | 13 | 1.0505970763041528 | True | [checkpoint: checkpoints/setup_a/best_a_ep13_eer0105_0713_0401.pt] |
| B1 | checkpoints/setup_b/best_b1_k64synthetic_ep9_eer0604.pt | 9 | 6.041495899722572 | True | [checkpoint: checkpoints/setup_b/best_b1_k64synthetic_ep9_eer0604.pt] |
| B0 (candidate lineage) | checkpoints/setup_b/best_b_ep1_eer0782_0624_0314.pt | 1 | 7.81839447885161 | False | [checkpoint: checkpoints/setup_b/best_b_ep1_eer0782_0624_0314.pt] |
| unassigned or superseded | checkpoints/setup_b/best_b_ep3_eer0750_0713_0222.pt | 3 | 7.503000454241848 | False | [checkpoint: checkpoints/setup_b/best_b_ep3_eer0750_0713_0222.pt] |
| B0 | checkpoints/setup_b/best_b_ep4_eer0759_0624_1407.pt | 4 | 7.594063133412455 | True | [checkpoint: checkpoints/setup_b/best_b_ep4_eer0759_0624_1407.pt] |
| C1 | checkpoints/setup_c/best_c1_attention.pt | 6 | 0.8633686177779123 | True | [checkpoint: checkpoints/setup_c/best_c1_attention.pt] |
| C2 | checkpoints/setup_c/best_c2_concat.pt | 3 | 0.9432693471747223 | True | [checkpoint: checkpoints/setup_c/best_c2_concat.pt] |
| C0 (candidate lineage) | checkpoints/setup_c/best_c_ep10_eer0606_0702_1445.pt | 10 | 6.0635649647407766 | False | [checkpoint: checkpoints/setup_c/best_c_ep10_eer0606_0702_1445.pt] |
| C0 (candidate lineage) | checkpoints/setup_c/best_c_ep11_eer0590_0706_0224.pt | 11 | 5.903763505947158 | False | [checkpoint: checkpoints/setup_c/best_c_ep11_eer0590_0706_0224.pt] |
| C0 | checkpoints/setup_c/best_c_ep12_eer0566_0710_0112.pt | 12 | 5.664061317756727 | True | [checkpoint: checkpoints/setup_c/best_c_ep12_eer0566_0710_0112.pt] |
| C0 (candidate lineage) | checkpoints/setup_c/best_c_ep1_eer0911_0630_1355.pt | 1 | 9.110886852698163 | False | [checkpoint: checkpoints/setup_c/best_c_ep1_eer0911_0630_1355.pt] |
| C0 (candidate lineage) | checkpoints/setup_c/best_c_ep3_eer0757_0701_0204.pt | 3 | 7.566139417693169 | False | [checkpoint: checkpoints/setup_c/best_c_ep3_eer0757_0701_0204.pt] |
| C0 (candidate lineage) | checkpoints/setup_c/best_c_ep4_eer0713_0701_1226.pt | 4 | 7.128908023364911 | False | [checkpoint: checkpoints/setup_c/best_c_ep4_eer0713_0701_1226.pt] |
| C0 (candidate lineage) | checkpoints/setup_c/best_c_ep7_eer0643_0702_0318.pt | 7 | 6.425340864380619 | False | [checkpoint: checkpoints/setup_c/best_c_ep7_eer0643_0702_0318.pt] |

Candidate lineage means filename/architecture compatibility only; run identity and resume continuity are not proved by increasing epochs and dates. The alternative B checkpoint is not the reported B0 file. See its tensor shapes before assigning it to a configuration. C0 candidate files are not proof of a single uninterrupted/resumed run. Checkpoint filename dates have no year and are not authenticated execution timestamps. File modification times are omitted because copies/downloads change them.

Alternative B detail: first MLP weight shape [256,184], compared with [256,248] in reported B0. Under the current default LFCC-only layout, a PCA width reduction from 128 to 64 is consistent with this shape difference; the historical feature invocation is not recovered. PCA provenance/run identity/use remain NOT FOUND. Stored epoch 3 / log epoch 4; steps 10948 / 2737 = 4; saved LR 0.001. [checkpoint; architecture inference]

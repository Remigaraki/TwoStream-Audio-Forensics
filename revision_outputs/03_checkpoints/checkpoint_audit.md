All principal checkpoint epochs, steps and rounded validation EERs match the megaprompt. These are checkpoint-save positions, not final run lengths.

| Run | Stored epoch (0-based) | Log epoch (1-based) | Optimizer steps | Steps per epoch | Step quotient | Validation EER (%) | Matches megaprompt | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 10 | 11 | 30118 | 2738 | 11 | 1.56 | True | [checkpoint: checkpoints/setup_a/best_a0_final_eer0156.pt] |
| A1 | 13 | 14 | 38318 | 2737 | 14 | 1.05 | True | [checkpoint: checkpoints/setup_a/best_a_ep13_eer0105_0713_0401.pt] |
| B1 | 9 | 10 | 27370 | 2737 | 10 | 6.04 | True | [checkpoint: checkpoints/setup_b/best_b1_k64synthetic_ep9_eer0604.pt] |
| B0 | 4 | 5 | 13685 | 2737 | 5 | 7.59 | True | [checkpoint: checkpoints/setup_b/best_b_ep4_eer0759_0624_1407.pt] |
| C1 | 6 | 7 | 38318 | 5474 | 7 | 0.86 | True | [checkpoint: checkpoints/setup_c/best_c1_attention.pt] |
| C2 | 3 | 4 | 21896 | 5474 | 4 | 0.94 | True | [checkpoint: checkpoints/setup_c/best_c2_concat.pt] |
| C0 | 12 | 13 | 71162 | 5474 | 13 | 5.66 | True | [checkpoint: checkpoints/setup_c/best_c_ep12_eer0566_0710_0112.pt] |

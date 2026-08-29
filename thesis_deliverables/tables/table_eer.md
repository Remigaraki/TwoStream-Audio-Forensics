# EER (%) by model and condition

| Model | Clean | Opus 16k | Opus 32k | Opus 64k | MP3 64k | MP3 128k | AAC 128k |
|---|---|---|---|---|---|---|---|
| A0 (RawNet2, no aug.) | 1.54 | 9.48 | 3.83 | 1.56 | 5.77 | 5.87 | 1.62 |
| A1 (RawNet2 + codec aug.) | 1.00 | 2.95 | 1.65 | 0.96 | 1.60 | 1.53 | 1.01 |
| B0 (statistical stream) | 7.50 | 18.35 | 11.53 | 8.26 | 13.43 | 12.81 | 7.54 |
| C1 (fusion, attention) | 0.81 | 2.38 | 1.32 | 0.80 | 1.70 | 1.62 | 0.89 |
| C2 (fusion, concat) | 0.92 | 2.47 | 1.33 | 0.86 | 1.78 | 1.69 | 0.96 |

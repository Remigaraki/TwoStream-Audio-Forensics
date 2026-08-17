# EER degradation ratio (condition EER / clean EER)

| Model | Clean | Opus 16k | Opus 32k | Opus 64k | MP3 64k | MP3 128k | AAC 128k |
|---|---|---|---|---|---|---|---|
| A0 (RawNet2, no aug.) | 1.00 | 6.14 | 2.48 | 1.01 | 3.74 | 3.80 | 1.05 |
| A1 (RawNet2 + codec aug.) | 1.00 | 2.96 | 1.65 | 0.96 | 1.60 | 1.53 | 1.02 |
| B0 (statistical stream) | 1.00 | 2.45 | 1.54 | 1.10 | 1.79 | 1.71 | 1.01 |
| C1 (fusion, attention) | 1.00 | 2.94 | 1.62 | 0.99 | 2.10 | 2.01 | 1.10 |
| C2 (fusion, concat) | 1.00 | 2.70 | 1.45 | 0.94 | 1.95 | 1.84 | 1.05 |

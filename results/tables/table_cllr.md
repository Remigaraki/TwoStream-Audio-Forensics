# CLLR by model and condition

| Model | Clean | Opus 16k | Opus 32k | Opus 64k | MP3 64k | MP3 128k | AAC 128k |
|---|---|---|---|---|---|---|---|
| A0 (RawNet2, no aug.) | 0.078 | 0.533 | 0.202 | 0.074 | 0.554 | 0.566 | 0.081 |
| A1 (RawNet2 + codec aug.) | 0.077 | 0.173 | 0.083 | 0.072 | 0.102 | 0.094 | 0.079 |
| B0 (statistical stream) | 0.381 | 0.888 | 0.616 | 0.369 | 1.331 | 1.303 | 0.381 |
| C1 (fusion, attention) | 0.043 | 0.119 | 0.061 | 0.042 | 0.126 | 0.119 | 0.045 |
| C2 (fusion, concat) | 0.051 | 0.121 | 0.060 | 0.046 | 0.131 | 0.124 | 0.053 |

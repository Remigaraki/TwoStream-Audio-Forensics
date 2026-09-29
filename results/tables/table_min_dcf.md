# MIN_DCF by model and condition

| Model | Clean | Opus 16k | Opus 32k | Opus 64k | MP3 64k | MP3 128k | AAC 128k |
|---|---|---|---|---|---|---|---|
| A0 (RawNet2, no aug.) | 0.0395 | 0.2263 | 0.0922 | 0.0397 | 0.1593 | 0.1613 | 0.0403 |
| A1 (RawNet2 + codec aug.) | 0.0268 | 0.0764 | 0.0417 | 0.0256 | 0.0429 | 0.0395 | 0.0266 |
| B0 (statistical stream) | 0.1555 | 0.4868 | 0.2827 | 0.1801 | 0.3500 | 0.3294 | 0.1585 |
| B1 (statistical stream, PCA K=64, synthetic fit) | 0.1406 | 0.4557 | 0.2297 | 0.1561 | 0.2346 | 0.2165 | 0.1438 |
| C0 (fusion, attention, end-to-end, epoch 12) | 0.1299 | 0.4000 | 0.2067 | 0.1427 | 0.2190 | 0.2065 | 0.1328 |
| C1 (fusion, attention) | 0.0191 | 0.0617 | 0.0326 | 0.0201 | 0.0428 | 0.0412 | 0.0199 |
| C2 (fusion, concat) | 0.0209 | 0.0629 | 0.0347 | 0.0206 | 0.0455 | 0.0429 | 0.0217 |

# A1 (setup A, codec-augmented) training — command + run log

## Provenance

This file records the actual training invocation and run log for the A1
checkpoint (`checkpoints/setup_a/best_a_ep13_eer0105_0713_0401.pt` and its
successors on the `run-a1` branch). The command and log below were extracted
from a groupmate's `a1-codec-aug.ipynb` notebook's stored cell output and
pasted into this conversation on 2026-08-03; they were not read directly from
the `.ipynb` file itself (not accessible from this environment). Committing
here because the command was previously only reachable via that uncommitted
notebook — same issue as `scripts/predict_testset.py` and the Phase 0
manifest-generation cell, both of which existed only in uncommitted notebooks
before being recovered into this repo.

This confirms A1 was trained with `--augment codec` (drawing from
`src/pipeline/augment.py::CODEC_CONDITIONS` / `transcode()`, since that is
what `train.py` imports) — not `data/torture_pipeline.py::ALL_CONDITIONS`,
which is never imported by `train.py` and is disconnected from training
entirely.

## Command (from the train cell, resuming from epoch 13)

```
python src/train.py \
    --setup A \
    --epochs 30 \
    --lr 1e-4 \
    --batch_size 32 \
    --manifest /kaggle/working/manifest.csv \
    --pca_path data/pca_model/pca.pkl \
    --ckpt_dir /kaggle/working/checkpoints/A1 \
    --augment codec \
    --augment_prob 0.5 \
    --seed 42 \
    --resume_from /kaggle/working/checkpoints/A1/best.pt \
    --train_data_root /kaggle/tmp/audio/train_audio \
    --val_data_root /kaggle/tmp/audio/val_audio
```

## Run log (verbatim excerpt, epoch 14 start)

```
[init] device=cuda
[init] augment=codec augment_prob=0.5 aug_seed=42 (independent of the manifest train/val/test split seed, which is fixed at manifest-generation time and unaffected by this flag)
[init] reading manifest …
[init] train_ds=87585 samples
[init] val_ds=18769 samples
[init] loaders ready — 2737 train batches, 586 val batches
[init] building model (setup=A, fusion=attention, features=lfcc) …
[init] moving model to cuda …
[init] model ready
Resumed from /kaggle/working/checkpoints/A1/best.pt (epoch 13)
  [train] epoch 14 batch 0/2737  loss=0.0488  0.1 batches/sec
  [train] epoch 14 batch 50/2737  loss=0.0338  0.3 batches/sec
  [train] epoch 14 batch 100/2737  loss=0.0101  0.3 batches/sec
  [train] epoch 14 batch 150/2737  loss=0.3012  0.3 batches/sec
  [train] epoch 14 batch 200/2737  loss=0.0011  0.4 batches/sec
  [train] epoch 14 batch 250/2737  loss=0.0235  0.4 batches/sec
  [train] epoch 14 batch 300/2737  loss=0.0015  0.4 batches/sec
  [train] epoch 14 batch 350/2737  loss=0.0098  0.4 batches/sec
  [train] epoch 14 batch 400/2737  loss=0.0656  0.4 batches/sec
  [train] epoch 14 batch 450/2737  loss=0.0335  0.4 batches/sec
  [train] epoch 14 batch 500/2737  loss=0.0155  0.4 batches/sec
  [train] epoch 14 batch 550/2737  loss=0.0520  0.4 batches/sec
  [train] epoch 14 batch 600/2737  loss=0.0201  0.4 batches/sec
  [train] epoch 14 batch 650/2737  loss=0.0310  0.4 batches/sec
  [train] epoch 14 batch 700/2737  loss=0.0024  0.4 batches/sec
```

`[init] augment=codec augment_prob=0.5 aug_seed=42` confirms the flag fired.
Throughput here (~0.1 → 0.4 batches/sec) is from this A1 log only. A0's
often-cited ~0.8 batches/sec figure is from a separate run/log not included
here — do not treat the "~2x slowdown" framing as substantiated by this file
alone; pull A0's actual log into a companion doc before citing that
comparison in the thesis.

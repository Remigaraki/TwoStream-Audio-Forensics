"""Generate test-set per-utterance predictions from a trained checkpoint.

Reuses _ManifestDataset and TwoStreamFusionNet from src/train.py so inference
exactly matches how validation EER was computed during training. Does NOT
modify src/train.py.

Usage
-----
    python scripts/predict_testset.py \\
        --checkpoint checkpoints/setup_a/best_a_ep13_eer0105_0713_0401.pt \\
        --manifest /kaggle/working/manifest.csv \\
        --setup A \\
        --data_root /kaggle/input/datasets/ulianazb/asvspoof-2024/flac_T/flac_T \\
        --output_csv results/preds_A1_clean.csv

    python scripts/predict_testset.py \\
        --checkpoint checkpoints/setup_c/best_c1_candidate.pt \\
        --manifest /kaggle/working/manifest.csv \\
        --setup C --fusion attention \\
        --pca_path data/pca_model/pca.pkl \\
        --data_root /kaggle/tmp/codec_test/opus_16 \\
        --output_csv results/preds_C1_opus16.csv

Output CSV columns: utterance_id,true_label,score,pred
- score : model's raw sigmoid output (higher = more likely spoof)
- pred  : score binarized at the test-set EER threshold (printed to stdout)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import _ManifestDataset
from src.fusion.two_stream_net import TwoStreamFusionNet
from src.eval.metrics import compute_eer as compute_eer_traintime
from src.utils.metrics import compute_eer as compute_eer_with_threshold


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test-set prediction CSV generator")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--setup", required=True, choices=["A", "B", "C"])
    p.add_argument("--data_root", default=None,
                   help="Optional. If given, test audio is looked up as "
                        "data_root/<basename of file_path> (same lookup _ManifestDataset "
                        "uses for train/val) — use this to redirect to a codec-transcoded "
                        "directory. If omitted, reads file_path from the manifest verbatim "
                        "(absolute paths), e.g. straight off a read-only Kaggle input mount.")
    p.add_argument("--pca_path", default=None, help="Required for --setup B or C.")
    p.add_argument("--fusion", choices=["attention", "concat"], default=None,
                   help="Required for --setup C. Must match the checkpoint's architecture "
                        "or load_state_dict(strict=True) will fail loudly.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--output_csv", required=True)
    args = p.parse_args()

    if args.setup in ("B", "C") and args.pca_path is None:
        p.error(f"--pca_path is required for --setup {args.setup}")
    if args.setup == "C" and args.fusion is None:
        p.error("--fusion is required for --setup C (attention or concat)")
    return args


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}", flush=True)

    # ------------------------------------------------------------------
    # Dataset — test split only, sorted by utterance_id for cross-model
    # row alignment (required for McNemar pairing).
    # ------------------------------------------------------------------
    test_ds = _ManifestDataset(args.manifest, split="test", data_root=args.data_root)
    test_ds.records.sort(key=lambda r: r["utterance_id"])

    utterance_ids = [r["utterance_id"] for r in test_ds.records]
    if len(set(utterance_ids)) != len(utterance_ids):
        raise ValueError(
            f"utterance_id is not unique in the test split "
            f"({len(utterance_ids)} rows, {len(set(utterance_ids))} unique ids)"
        )
    print(f"[init] test set: {len(test_ds)} utterances", flush=True)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"[init] building model (setup={args.setup}, fusion={args.fusion}) …", flush=True)
    model = TwoStreamFusionNet(
        pca_path=args.pca_path,
        setup=args.setup,
        fusion=args.fusion or "attention",
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    print(
        f"[init] loaded checkpoint {args.checkpoint} "
        f"(epoch={ckpt.get('epoch', '?')}, val_eer={ckpt.get('val_eer', float('nan')):.4f})",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    all_scores, all_labels = [], []
    n_batches = len(test_loader)
    for step, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.to(device)
        preds = model(batch_x).squeeze(1).cpu().numpy()
        all_scores.extend(preds.tolist())
        all_labels.extend(batch_y.numpy().tolist())
        if step % 100 == 0:
            print(f"  [predict] batch {step}/{n_batches}", flush=True)

    scores = np.array(all_scores, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int32)
    assert len(scores) == len(utterance_ids), (
        f"scored {len(scores)} utterances but expected {len(utterance_ids)} — "
        f"DataLoader dropped or reordered rows"
    )

    # ------------------------------------------------------------------
    # EER + threshold
    # ------------------------------------------------------------------
    eer_traintime = compute_eer_traintime(scores, labels)
    eer_with_thr, threshold = compute_eer_with_threshold(labels, scores)

    print(f"\n[eval] test EER (src.eval.metrics, train.py's implementation) : {eer_traintime:.4f}")
    print(f"[eval] test EER (src.utils.metrics, mcnemar_test.py's implementation): {eer_with_thr:.4f}")
    print(f"[eval] EER threshold used for pred (src.utils.metrics): {threshold:.6f}")
    if abs(eer_traintime - eer_with_thr) > 1e-3:
        print("  ⚠️  the two EER implementations disagree by >0.001 — see audit notes", flush=True)

    preds = (scores >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["utterance_id", "true_label", "score", "pred"])
        for uid, label, score, pred in zip(utterance_ids, labels, scores, preds):
            writer.writerow([uid, int(label), f"{score:.6f}", int(pred)])

    n_spoof = int(labels.sum())
    print(
        f"\n[done] wrote {len(utterance_ids)} predictions to {output_path} "
        f"(bonafide={len(labels) - n_spoof}, spoof={n_spoof})"
    )


if __name__ == "__main__":
    main()

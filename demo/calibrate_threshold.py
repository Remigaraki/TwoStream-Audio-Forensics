"""Derive a demo decision threshold from the VALIDATION split only.

Scores every `split == "val"` row of the evaluated manifest with the same
hash-verified checkpoint the demo serves, picks an operating point on those
clean validation scores, and records it with provenance in demo/thresholds.json.
The test split, test prediction CSVs and per-condition test thresholds are
never read.

Operating points (--operating_point):
  eer      threshold where both error rates are equal
           (src.utils.metrics.compute_eer, as in predict_testset.py / mcnemar_test.py)
  min_dcf  threshold minimising the paper's normalised detection cost
           (scripts/build_results.py: PI_SPOOF, C_MISS, C_FA, compute_min_dcf)

Codec check (--codec_root, optional): scores codec-compressed copies of the
same validation audio (made with `scripts/make_codec_testsets.py --split val`)
at the clean-validation threshold and reports error rates and cost against the
best achievable on that condition. It is a report only; it never changes the
threshold.

Error names follow build_results.py: miss = bona fide flagged as spoof,
false alarm = spoof accepted as bona fide.

Needs the ASVspoof 5 validation audio, so it is meant for the research
environment (Kaggle/Colab); a GPU helps but is not required.

    python demo/calibrate_threshold.py --model C1 --operating_point min_dcf \
        --manifest data/manifest.csv --data_root /path/to/flac \
        --codec_root /kaggle/tmp/codec_val \
        --scores_csv demo/calibration/val_scores_C1.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo import inference as inf  # noqa: E402
from src.train import _ManifestDataset  # noqa: E402
from src.eval.metrics import compute_eer as compute_eer_traintime  # noqa: E402
from src.pipeline.augment import CODEC_CONDITIONS  # noqa: E402
from src.utils.metrics import compute_eer as compute_eer_with_threshold  # noqa: E402
from build_results import C_FA, C_MISS, PI_SPOOF, compute_min_dcf  # noqa: E402
from verify_manifest import REFERENCE_HASH, compute_split_hash  # noqa: E402

SPLIT = "val"  # hard-coded on purpose: never calibrate on "test"
MAX_VAL_EER_DRIFT = 0.002  # vs. the val_eer logged in the checkpoint at training time


def error_rates(scores: np.ndarray, labels: np.ndarray, tau: float) -> tuple[float, float]:
    """(P_miss, P_fa) for the decision `score >= tau -> spoof`."""
    spoof_pred = scores >= tau
    return float(spoof_pred[labels == 0].mean()), float((~spoof_pred[labels == 1]).mean())


def dcf(p_miss: float, p_fa: float) -> float:
    """Normalised DCF at one operating point, same formula as build_results.compute_min_dcf."""
    pi_bona = 1.0 - PI_SPOOF
    return (C_MISS * pi_bona * p_miss + C_FA * PI_SPOOF * p_fa) / min(C_MISS * pi_bona, C_FA * PI_SPOOF)


def min_dcf_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Threshold achieving the minimum normalised DCF, and that minimum.

    Candidates are every distinct score (plus "flag nothing"); the returned
    threshold is the midpoint to the next lower score, which makes the same
    decisions on this data but does not sit exactly on a validation score.
    """
    bona, spoof = np.sort(scores[labels == 0]), np.sort(scores[labels == 1])
    u = np.unique(scores)
    cand = np.append(u, np.nextafter(u[-1], np.inf))
    p_miss = 1.0 - np.searchsorted(bona, cand, side="left") / len(bona)
    p_fa = np.searchsorted(spoof, cand, side="left") / len(spoof)
    costs = np.array([dcf(m, f) for m, f in zip(p_miss, p_fa)])
    k = int(np.argmin(costs))
    tau = float(cand[k]) if k == 0 or k == len(u) else float((u[k - 1] + u[k]) / 2)
    return tau, float(costs[k])


def score_split(model, manifest: str, data_root: str | None, device, batch_size: int, num_workers: int):
    ds = _ManifestDataset(manifest, split=SPLIT, data_root=data_root)
    ds.records.sort(key=lambda r: r["utterance_id"])
    if data_root is not None:
        missing = [r["utterance_id"] for r in ds.records
                   if not (Path(data_root) / Path(r.get("file_path") or r["path"]).name).is_file()]
        if missing:
            raise SystemExit(f"{len(missing)} validation file(s) missing under {data_root} "
                             f"(e.g. {missing[:3]}); regenerate before calibrating")
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    scores, labels = [], []
    with torch.inference_mode():
        for step, (x, y) in enumerate(loader):
            scores.extend(model(x.to(device)).squeeze(1).cpu().tolist())  # already sigmoid
            labels.extend(y.int().tolist())
            if step % 100 == 0:
                print(f"  [val] batch {step}/{len(loader)}", flush=True)
    return ds.records, np.asarray(scores, np.float32), np.asarray(labels, np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(inf.MODELS))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data_root", default=None, help="Same meaning as in predict_testset.py / train.py.")
    ap.add_argument("--operating_point", choices=["eer", "min_dcf"], default="eer")
    ap.add_argument("--codec_root", default=None,
                    help="Output base of `make_codec_testsets.py --split val`; subdirs like opus_16/ are checked.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--scores_csv", default=None, help="Optional audit file of per-utterance clean val scores.")
    ap.add_argument("--allow_manifest_mismatch", action="store_true")
    args = ap.parse_args()

    split_hash = compute_split_hash(pd.read_csv(args.manifest, usecols=["utterance_id", "split"]))
    if split_hash != REFERENCE_HASH and not args.allow_manifest_mismatch:
        raise SystemExit(f"manifest split hash {split_hash} != evaluated {REFERENCE_HASH}; refusing to calibrate")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inf.load_model(args.model).to(device)
    ckpt_rel, ckpt_sha = inf.MODELS[args.model]["checkpoint"]
    logged_val_eer = float(torch.load(inf.ROOT / ckpt_rel, map_location="cpu", weights_only=True)["val_eer"])

    run = lambda root: score_split(model, args.manifest, root, device, args.batch_size, args.num_workers)  # noqa: E731
    records, scores, labels = run(args.data_root)

    eer_train = compute_eer_traintime(scores, labels)
    eer, eer_tau = compute_eer_with_threshold(labels, scores)
    mdcf_tau, mdcf = min_dcf_threshold(scores, labels)
    ref_mdcf = compute_min_dcf(scores, labels)  # build_results.py's own implementation
    print(f"[val] EER={eer:.5f} (train-time impl {eer_train:.5f}; checkpoint logged {logged_val_eer:.5f})")
    print(f"[val] EER threshold={eer_tau:.6f}; minDCF={mdcf:.5f} at threshold={mdcf_tau:.6f}")
    if abs(eer_train - logged_val_eer) > MAX_VAL_EER_DRIFT:
        raise SystemExit("validation EER does not reproduce the checkpoint's logged val_eer; not saving")
    # build_results accumulates in float32, hence the tolerance. Its stable sort puts spoofs
    # first inside any tie group, so its minimum is always one a real threshold can reach.
    if abs(mdcf - ref_mdcf) > 1e-5:
        raise SystemExit(f"minDCF threshold search ({mdcf}) disagrees with build_results ({ref_mdcf})")

    tau = eer_tau if args.operating_point == "eer" else mdcf_tau
    p_miss, p_fa = error_rates(scores, labels, tau)
    print(f"[val] chosen '{args.operating_point}' threshold={tau:.6f}: "
          f"miss={p_miss:.4%} false_alarm={p_fa:.4%} DCF={dcf(p_miss, p_fa):.5f}")

    codec_check = {}
    if args.codec_root:
        for codec, bitrate in CODEC_CONDITIONS:
            cond_dir = Path(args.codec_root) / f"{codec}_{bitrate}"
            if not cond_dir.is_dir():
                print(f"[codec] {cond_dir.name}: not generated, skipped")
                continue
            _, c_scores, c_labels = run(str(cond_dir))
            m, f = error_rates(c_scores, c_labels, tau)
            _, c_best = min_dcf_threshold(c_scores, c_labels)
            codec_check[cond_dir.name] = {
                "miss": m, "false_alarm": f, "dcf_at_threshold": dcf(m, f),
                "min_dcf": c_best, "val_eer": compute_eer_with_threshold(c_labels, c_scores)[0],
            }
            print(f"[codec] {cond_dir.name}: miss={m:.4%} false_alarm={f:.4%} "
                  f"DCF={dcf(m, f):.5f} (best achievable {c_best:.5f})")

    if args.scores_csv:
        Path(args.scores_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.scores_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["utterance_id", "true_label", "score"])
            w.writerows((r["utterance_id"], int(lab), f"{s:.6f}") for r, lab, s in zip(records, labels, scores))

    point = ("equal-error-rate" if args.operating_point == "eer"
             else f"minimum-DCF (pi_spoof={PI_SPOOF}, C_miss={C_MISS}, C_fa={C_FA})")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    data = json.loads(inf.THRESHOLDS_PATH.read_text(encoding="utf-8"))
    data[args.model] = {
        "threshold": tau,
        "status": "validation",
        "source": f"{point} operating point on the clean validation split ({len(labels)} utterances) "
                  "of the evaluated manifest.",
        "provenance": {
            "split": SPLIT, "manifest_split_hash": split_hash, "operating_point": args.operating_point,
            "n_bonafide": int((labels == 0).sum()), "n_spoof": int((labels == 1).sum()),
            "val_miss_at_threshold": p_miss, "val_false_alarm_at_threshold": p_fa,
            "val_eer": eer, "val_eer_threshold": eer_tau, "val_eer_traintime_impl": eer_train,
            "val_min_dcf": mdcf, "val_min_dcf_threshold": mdcf_tau,
            "dcf_params": {"pi_spoof": PI_SPOOF, "c_miss": C_MISS, "c_fa": C_FA},
            "checkpoint_logged_val_eer": logged_val_eer,
            "checkpoint": ckpt_rel, "checkpoint_sha256": ckpt_sha,
            "codec_check_val": codec_check or None,
            "git_commit": commit or None, "torch": torch.__version__, "device": str(device),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
    }
    inf.THRESHOLDS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {args.model} threshold to {inf.THRESHOLDS_PATH}")


if __name__ == "__main__":
    main()

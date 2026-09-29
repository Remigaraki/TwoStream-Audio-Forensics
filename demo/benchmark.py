"""Measure CPU cold start, warm latency and memory for the demo models.

    python demo/benchmark.py [--runs 20] [--with-a1]

Timing only: the input is synthetic, so scores printed here mean nothing.
"""
from __future__ import annotations

import time

T0 = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import soundfile as sf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo import inference as inf  # noqa: E402

PROC = psutil.Process()


def rss_mb() -> float:
    return PROC.memory_info().rss / 2**20


def peak_mb() -> float:
    mi = PROC.memory_info()
    if hasattr(mi, "peak_wset"):  # Windows
        return mi.peak_wset / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KiB


def bench(name: str, runs: int, audio: Path, tmp: Path) -> dict:
    t = time.perf_counter()
    model = inf.load_model(name)
    load_s = time.perf_counter() - t
    rss_loaded = rss_mb()
    t = time.perf_counter()
    x = inf.prepare(audio, tmp).x
    prep_s = time.perf_counter() - t
    first_s = inf.score(model, x)[1]
    warm = [inf.score(model, x)[1] for _ in range(runs)]
    e2e = []
    for _ in range(5):
        t = time.perf_counter()
        inf.score(model, inf.prepare(audio, tmp).x)
        e2e.append(time.perf_counter() - t)
    return {
        "load_verify_s": round(load_s, 3), "rss_after_load_mb": round(rss_loaded),
        "decode_preprocess_s": round(prep_s, 3), "first_inference_s": round(first_s, 3),
        "warm_inference_median_s": round(statistics.median(warm), 3),
        "warm_inference_max_s": round(max(warm), 3),
        "end_to_end_median_s": round(statistics.median(e2e), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--with-a1", action="store_true")
    args = ap.parse_args()
    import_s = time.perf_counter() - T0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        audio = tmp / "bench.wav"  # 30 s, 44.1 kHz stereo: exercises resampling + downmix
        sf.write(audio, (0.1 * np.random.default_rng(0).standard_normal((44100 * 30, 2))).astype(np.float32), 44100)
        out = {"python_import_s": round(import_s, 3), "torch_threads": inf.torch.get_num_threads(),
               "cpu_count": os.cpu_count(), "C1": bench("C1", args.runs, audio, tmp)}
        if args.with_a1:
            out["A1"] = bench("A1", args.runs, audio, tmp)
    out["cold_start_to_first_result_s"] = round(
        import_s + out["C1"]["load_verify_s"] + out["C1"]["decode_preprocess_s"] + out["C1"]["first_inference_s"], 3)
    out["rss_final_mb"], out["peak_mb"] = round(rss_mb()), round(peak_mb())
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

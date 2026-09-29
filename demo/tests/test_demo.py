"""Demo tests. Synthetic audio here checks code-path equivalence and input
handling only; it is NOT evidence of detection quality (see demo/check_parity.py).

    demo/.venv/Scripts/python -m pytest demo/tests -q
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demo import inference as inf  # noqa: E402
from src.train import _ManifestDataset  # noqa: E402

RNG = np.random.default_rng(0)
ATOL = 1e-6  # float32 resampler tolerance between prefix and whole-file decode


def _write(tmp_path, name, seconds, sr, channels=1, fmt=None, subtype=None):
    n = int(seconds * sr)
    t = np.arange(n) / sr
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * RNG.standard_normal(n)
    if channels == 2:  # channels deliberately differ so the mean matters
        x = np.stack([x, 0.5 * np.cos(2 * np.pi * 330 * t)], axis=1)
    p = tmp_path / name
    sf.write(p, x.astype(np.float32), sr, format=fmt, subtype=subtype)
    return p


def _reference(tmp_path, audio: Path) -> torch.Tensor:
    """What scripts/predict_testset.py feeds the model for this file."""
    manifest = tmp_path / "m.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["utterance_id", "file_path", "label", "split"])
        w.writerow(["u0", str(audio), "0", "test"])
    x, _ = _ManifestDataset(str(manifest), split="test")[0]
    return x.unsqueeze(0)


@pytest.fixture(scope="module")
def c1():
    return inf.load_model("C1")


CASES = [
    ("mono16k.wav", 6.0, 16000, 1),
    ("stereo44k.wav", 6.0, 44100, 2),
    ("stereo48k.flac", 10.0, 48000, 2),
    ("up8k.wav", 5.0, 8000, 1),
    ("odd22050.flac", 30.0, 22050, 1),
    ("short.wav", 1.2, 16000, 1),
    ("short_stereo44k.wav", 2.0, 44100, 2),
    ("clip.mp3", 6.0, 44100, 2),
]


@pytest.mark.parametrize("name,secs,sr,ch", CASES)
def test_preprocessing_matches_manifest_dataset(tmp_path, name, secs, sr, ch):
    audio = _write(tmp_path, name, secs, sr, ch)
    p = inf.prepare(audio, tmp_path)
    ref = _reference(tmp_path, audio)
    assert p.x.shape == (1, 1, 64000) and p.x.dtype == torch.float32
    assert ref.shape == p.x.shape
    assert torch.max(torch.abs(p.x - ref)).item() <= ATOL
    assert p.analyzed_s == pytest.approx(min(p.duration_s, 4.0), abs=1e-3)


def test_short_clip_is_zero_padded(tmp_path):
    p = inf.prepare(_write(tmp_path, "s.wav", 1.5, 16000), tmp_path)
    assert p.analyzed_s == pytest.approx(1.5) and p.padded_s == pytest.approx(2.5)
    assert torch.count_nonzero(p.x[..., 24000:]) == 0


def test_long_clip_uses_first_four_seconds(tmp_path):
    sr = 16000
    x = np.zeros(sr * 20, np.float32)
    x[: 4 * sr] = 0.2
    x[4 * sr:] = 0.9  # content after 4 s must not reach the model
    path = tmp_path / "l.wav"
    sf.write(path, x, sr, subtype="FLOAT")
    p = inf.prepare(path, tmp_path)
    assert p.duration_s == pytest.approx(20.0) and p.meta["read_frames"] < 5 * sr
    assert torch.allclose(p.x, torch.full_like(p.x, 0.2))


def test_score_parity_with_eval_path_and_single_sigmoid(tmp_path, c1):
    audio = _write(tmp_path, "p.flac", 5.0, 44100, 2)
    s, _ = inf.score(c1, inf.prepare(audio, tmp_path).x)
    with torch.inference_mode():
        ref = float(c1(_reference(tmp_path, audio)).squeeze(1)[0])  # predict_testset.py's call
    assert abs(s - ref) <= 1e-5
    assert isinstance(c1.classifier[-1], torch.nn.Sigmoid) and 0.0 <= s <= 1.0
    s2, _ = inf.score(c1, inf.prepare(audio, tmp_path).x)
    assert s2 == s  # deterministic


def test_decision_direction_and_fixed_threshold():
    assert inf.decide(0.5, 0.5) == "Likely synthetic or spoofed speech"  # >= spoof, as predict_testset.py
    assert inf.decide(0.4999, 0.5) == "Likely bona fide speech"
    t = inf.load_thresholds()
    assert set(t) >= set(inf.MODELS)
    for v in t.values():
        assert v["status"] in ("provisional", "validation") and v["source"]


@pytest.mark.parametrize("payload,msg", [
    (b"", "empty"),
    (b"RIFF\x00\x00\x00\x00WAVEgarbage" * 10, "decod"),
    (bytes(RNG.integers(0, 256, 4096, dtype=np.uint8)), "decod"),
])
def test_malformed_files_rejected(tmp_path, payload, msg):
    path = tmp_path / "bad.wav"
    path.write_bytes(payload)
    with pytest.raises(inf.InputError, match=msg):
        inf.prepare(path, tmp_path)


def test_silent_nonfinite_short_and_unsupported(tmp_path):
    sf.write(tmp_path / "z.wav", np.zeros(32000, np.float32), 16000)
    with pytest.raises(inf.InputError, match="silent"):
        inf.prepare(tmp_path / "z.wav", tmp_path)
    x = np.full(32000, 0.1, np.float32)
    x[100] = np.nan
    sf.write(tmp_path / "n.wav", x, 16000, subtype="FLOAT")
    with pytest.raises(inf.InputError, match="NaN"):
        inf.prepare(tmp_path / "n.wav", tmp_path)
    with pytest.raises(inf.InputError, match="minimum"):
        inf.prepare(_write(tmp_path, "t.wav", 0.2, 16000), tmp_path)
    with pytest.raises(inf.InputError, match="not WAV, FLAC or MP3"):
        inf.prepare(_write(tmp_path, "o.ogg", 2.0, 16000, fmt="OGG", subtype="VORBIS"), tmp_path)


def test_oversized_file_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(inf, "MAX_UPLOAD_BYTES", 1000)
    with pytest.raises(inf.InputError, match="exceeds"):
        inf.prepare(_write(tmp_path, "b.wav", 2.0, 16000), tmp_path)


def test_missing_lfs_pointer_and_tampered_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(inf, "ROOT", tmp_path)
    rel, sha = inf.MODELS["C1"]["checkpoint"]
    with pytest.raises(inf.AssetError, match="Missing"):
        inf.load_model("C1")
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_bytes(inf.LFS_MAGIC + b"\noid sha256:" + sha.encode() + b"\nsize 1\n")
    with pytest.raises(inf.AssetError, match="LFS pointer"):
        inf.verify_asset(rel, sha)
    (tmp_path / rel).write_bytes(b"not the checkpoint")
    with pytest.raises(inf.AssetError, match="mismatch"):
        inf.verify_asset(rel, sha)


class _FakeUpload:
    """Stands in for Streamlit's UploadedFile (AppTest cannot drive file_uploader)."""

    def __init__(self, path: Path):
        self.name, self._data = path.name, path.read_bytes()
        self.size = len(self._data)

    def getvalue(self) -> bytes:
        return self._data


def _run_app(monkeypatch, upload, compare=False):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: upload)
    at = AppTest.from_file(str(Path(inf.__file__).with_name("app.py")), default_timeout=120).run()
    if upload is not None:
        if compare:
            at.checkbox[0].check().run()
        at.button[0].click().run()
    assert not at.exception, at.exception
    return at


def test_app_renders_without_upload(monkeypatch):
    at = _run_app(monkeypatch, None)
    text = " ".join(m.value for m in at.markdown)
    assert "Reported test EER" in text and "not** held out" in text
    assert any("experimental research model" in i.value for i in at.info)


def test_app_analyze_flow_long_clip_with_a1(tmp_path, monkeypatch):
    at = _run_app(monkeypatch, _FakeUpload(_write(tmp_path, "u.wav", 9.0, 44100, 2)), compare=True)
    verdicts = [e.value for e in list(at.success) + list(at.warning)]
    assert len(verdicts) == 2 and all(v.strip("*") in (
        "Likely bona fide speech", "Likely synthetic or spoofed speech") for v in verdicts)
    text = " ".join(m.value for m in at.markdown)
    assert "0.00 s-4.00 s" in text and "9.00 s recording" in text and "**not** analysed" in text
    assert any("Provisional threshold" in c.value for c in at.caption)
    assert [m.label for m in at.metric].count("Model score") == 2


def test_app_rejects_silent_upload(tmp_path, monkeypatch):
    sf.write(tmp_path / "z.wav", np.zeros(32000, np.float32), 16000)
    at = _run_app(monkeypatch, _FakeUpload(tmp_path / "z.wav"))
    assert any("silent" in e.value for e in at.error) and not at.metric


def test_min_dcf_threshold_matches_build_results():
    from demo import calibrate_threshold as cal
    rng = np.random.default_rng(3)
    for n_ties in (0, 50):
        labels = rng.integers(0, 2, 2000)
        scores = np.clip(rng.normal(0.35 + 0.3 * labels, 0.2), 0, 1).astype(np.float32)
        if n_ties:  # saturated sigmoid outputs produce exact ties
            scores[:n_ties] = 1.0
            scores[n_ties:2 * n_ties] = 0.0
        tau, cost = cal.min_dcf_threshold(scores, labels)
        assert cost == pytest.approx(cal.compute_min_dcf(scores, labels), abs=1e-5)  # float32 in build_results
        assert cal.dcf(*cal.error_rates(scores, labels, tau)) == pytest.approx(cost, abs=1e-12)
    sep = np.array([0.1, 0.2, 0.8, 0.9], np.float32)
    tau, cost = cal.min_dcf_threshold(sep, np.array([0, 0, 1, 1]))
    assert cost == 0.0 and 0.2 < tau < 0.8


def test_calibration_end_to_end_with_codec_check(tmp_path, monkeypatch):
    """Code-path test on synthetic clips; guards are relaxed only for this run."""
    import json
    import shutil
    from demo import calibrate_threshold as cal
    rows = []
    for i in range(8):
        p = _write(tmp_path, f"V_{i}.flac", 2.0, 16000)
        rows.append([f"V_{i}", str(p), i % 2, "val"])
        rows.append([f"T_{i}", str(p), i % 2, "test"])  # must be ignored
    manifest = tmp_path / "m.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["utterance_id", "file_path", "label", "split"], *rows])
    (tmp_path / "codec" / "opus_16").mkdir(parents=True)
    for i in range(8):
        shutil.copy(tmp_path / f"V_{i}.flac", tmp_path / "codec" / "opus_16")
    thr = tmp_path / "thresholds.json"
    shutil.copy(inf.THRESHOLDS_PATH, thr)
    monkeypatch.setattr(inf, "THRESHOLDS_PATH", thr)
    monkeypatch.setattr(cal, "MAX_VAL_EER_DRIFT", 1.0)
    monkeypatch.setattr(sys, "argv", [
        "x", "--model", "C1", "--manifest", str(manifest), "--data_root", str(tmp_path),
        "--operating_point", "min_dcf", "--codec_root", str(tmp_path / "codec"),
        "--num_workers", "0", "--allow_manifest_mismatch"])
    cal.main()
    c1 = json.loads(thr.read_text())["C1"]
    prov = c1["provenance"]
    assert c1["status"] == "validation" and "minimum-DCF" in c1["source"]
    assert prov["n_bonafide"] + prov["n_spoof"] == 8 and c1["threshold"] == prov["val_min_dcf_threshold"]
    assert set(prov["codec_check_val"]) == {"opus_16"}  # other conditions absent -> skipped
    assert json.loads(thr.read_text())["A1"]["status"] == "provisional"
    monkeypatch.setattr(inf, "THRESHOLDS_PATH", thr)
    assert inf.load_thresholds()["C1"]["threshold"] == c1["threshold"]  # app can still load it


def test_min_dcf_mixed_label_ties_are_achievable():
    from demo import calibrate_threshold as cal
    scores = np.array([0.1] * 95 + [1.0] * 5 + [0.3] * 10 + [1.0] * 40 + [0.9] * 50, np.float32)
    labels = np.array([0] * 100 + [1] * 100)
    tau, cost = cal.min_dcf_threshold(scores, labels)
    assert cost == pytest.approx(cal.compute_min_dcf(scores, labels), abs=1e-5)
    assert cal.dcf(*cal.error_rates(scores, labels, tau)) == pytest.approx(cost, abs=1e-12)

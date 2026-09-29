"""Streamlit research demo: TwoStream-Audio-Forensics (C1 by default, optional A1).

    streamlit run demo/app.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo import inference as inf  # noqa: E402

st.set_page_config(page_title="TwoStream Audio Forensics - research demo", page_icon=":microphone:")

DISCLAIMER = (
    "This is an experimental research model for detecting synthetic or spoofed speech. "
    "Its output is a model score, not proof: it does not establish a speaker's identity "
    "or the authenticity of a recording with certainty, and it has only been evaluated "
    "on ASVspoof 5 data (see *About the research*)."
)


@st.cache_resource(show_spinner=False)
def get_model(name: str):
    # Shared across sessions: model weights only. Uploads never enter a cache.
    return inf.load_model(name)


def fmt_s(x: float) -> str:
    return f"{x:.2f} s"


def show_result(name: str, s: float, secs: float, thresholds: dict) -> None:
    t = thresholds[name]
    verdict = inf.decide(s, t["threshold"])
    (st.warning if s >= t["threshold"] else st.success)(f"**{verdict}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model score", f"{s:.4f}", help="Sigmoid output of the network; higher = more spoof-like. "
              "Not a calibrated probability.")
    c2.metric("Decision threshold", f"{t['threshold']:.4f}")
    c3.metric("Inference time", fmt_s(secs))
    if t["status"] == "provisional":
        st.caption(":warning: **Provisional threshold.** " + t["source"])
    else:
        st.caption("Threshold: " + t["source"])


def analyze_tab() -> None:
    st.info(DISCLAIMER)
    try:
        thresholds = inf.load_thresholds()
        with st.spinner("Loading model (first run only)..."):
            get_model("C1")
    except (inf.AssetError, OSError, KeyError, ValueError) as e:
        st.error(f"Model unavailable: {e}")
        st.stop()

    up = st.file_uploader("Speech recording (WAV, FLAC or MP3)", type=["wav", "flac", "mp3"],
                          help=f"Max {inf.MAX_UPLOAD_BYTES // 2**20} MB, {inf.MIN_DURATION_S}-"
                               f"{inf.MAX_DURATION_S / 60:.0f} min. Only the first 4 seconds are analysed.")
    if up is None:
        return
    if up.size > inf.MAX_UPLOAD_BYTES:
        st.error(f"File exceeds {inf.MAX_UPLOAD_BYTES // 2**20} MB.")
        return
    data = up.getvalue()
    st.audio(data)
    compare = st.checkbox("Also run A1 (RawNet2-only) for comparison",
                          help="Runs a second, separate model. Its score is shown on its own "
                               "and never combined with C1.")
    if not st.button("Analyze", type="primary"):
        return

    suffix = Path(up.name).suffix.lower()
    with st.status("Analyzing...", expanded=False) as status:
        try:
            # Per-request private directory, removed on exit even if analysis fails.
            with tempfile.TemporaryDirectory(prefix="upload-") as tmp:
                path = Path(tmp) / f"upload{suffix if suffix in ('.wav', '.flac', '.mp3') else ''}"
                path.write_bytes(data)
                status.update(label="Decoding and preprocessing...")
                p = inf.prepare(path, Path(tmp))
            status.update(label="Running C1...")
            results = {"C1": inf.score(get_model("C1"), p.x)}
            if compare:
                status.update(label="Running A1 (comparison)...")
                results["A1"] = inf.score(get_model("A1"), p.x)
        except inf.InputError as e:
            status.update(label="Input rejected", state="error")
            st.error(str(e))
            return
        except inf.BusyError as e:
            status.update(label="Busy", state="error")
            st.error(str(e))
            return
        except inf.AssetError as e:
            status.update(label="Model unavailable", state="error")
            st.error(str(e))
            return
        status.update(label="Done", state="complete")

    st.subheader("Result - C1 (default model)")
    show_result("C1", *results["C1"], thresholds)

    st.markdown("**What was analysed**")
    seg = f"**{fmt_s(0)}-{fmt_s(p.analyzed_s)}** of a {fmt_s(p.duration_s)} recording"
    notes = [f"Segment: {seg}. Only this segment was analysed."]
    if p.duration_s > 4.0 + 1e-6:
        notes.append(f"The remaining {fmt_s(p.duration_s - 4.0)} were **not** analysed; "
                     "this result says nothing about them.")
    if p.padded_s > 1e-6:
        notes.append(f"The clip is shorter than 4 s, so {fmt_s(p.padded_s)} of silence (zeros) "
                     "was appended, as in the evaluation pipeline.")
    m = p.meta
    notes.append(f"Input: {m['format']}, {m['samplerate']} Hz, {m['channels']} channel(s) -> "
                 "resampled to 16 kHz" + (", channels averaged to mono" if m["channels"] > 1 else "")
                 + ". No normalisation, trimming or denoising is applied.")
    st.markdown("\n".join(f"- {n}" for n in notes))

    if "A1" in results:
        st.subheader("Comparison - A1 (separate model)")
        st.caption("A1 sees the same 4-second input. Its result is independent and not merged with C1.")
        show_result("A1", *results["A1"], thresholds)


def about_tab() -> None:
    root = Path(__file__).resolve().parents[1]
    st.markdown(f"""
### Models served
- **C1 (default)** - two-stream fusion: RawNet2 on the raw waveform + a statistical stream
  (LFCC and PCA-compressed bispectrum) combined by cross-modal attention. Its K=128 PCA was
  fitted on synthetic noise waveforms, not speech; it is a fixed projection used identically in
  training, evaluation and this demo.
  Checkpoint `{inf.MODELS['C1']['checkpoint'][0]}`, PCA `{inf.MODELS['C1']['pca'][0]}`.
- **A1 (optional comparison)** - RawNet2 only, trained with codec augmentation.
  Checkpoint `{inf.MODELS['A1']['checkpoint'][0]}`.

Validation EER logged in the checkpoints: C1 0.86%, A1 1.05%.

### Reported test EER (%) - `results/tables/table_eer.md`
""")
    table = root / "results" / "tables" / "table_eer.md"
    st.markdown(table.read_text(encoding="utf-8").split("\n", 2)[-1] if table.exists()
                else "_Table not bundled with this build._")
    st.markdown("""
### Research-only models (results shown, not available for analysis)
Some rows in the table are experiments reported for comparison. They cannot be run on
uploads here, and they are not combined with C1 or A1.
- **B0** - statistical stream only (LFCC + bispectrum, K=128 PCA), no RawNet2.
- **B1** - statistical stream only, with a smaller K=64 PCA fitted on synthetic noise
  waveforms. Similar to B0 on clean audio but more robust to MP3 compression; about 9 times
  more errors than C1 on clean audio.
- **C0** - an earlier checkpoint (epoch 12) of a fusion model with the same design as C1,
  trained end to end. C1, trained separately, does much better in every condition.
- **A0** and **C2** - RawNet2 without codec augmentation, and fusion without the attention
  head (plain concatenation).

### Scope of these numbers
- Data: a held-out test split (18,769 utterances) drawn from the ASVspoof 5 **train and dev**
  partitions by `scripts/build_manifest.py`. It is an utterance-level random split: speakers and
  attack systems are **not** held out, and it is not the official ASVspoof 5 evaluation set.
- Codec columns re-encode the same test audio (Opus/MP3/AAC) before scoring.
- EER is measured at a threshold chosen on the test scores themselves. It describes ranking
  quality on this split; it is **not** the error rate to expect for arbitrary uploads, other
  languages, recording conditions or newer synthesis systems, and it is not an accuracy.

### How the demo differs from the evaluation
- Identical preprocessing to the evaluation pipeline: 16 kHz, mono, the first 4 s
  (64,000 samples), zero-padded if shorter, with no normalisation.
- The demo's decision threshold is separate from the research EER operating point. The
  threshold's status (provisional or validation-derived) is shown with every result.

### Privacy
Uploads are decoded from a temporary per-request directory that is deleted right after
decoding. For playback, the audio is held in server memory for your browser session only.
It is not written to permanent storage, logged, shared between visitors or used for training.
""")


tab_a, tab_b = st.tabs(["Analyze", "About the research"])
with tab_a:
    st.title("Synthetic speech detector")
    st.caption("TwoStream-Audio-Forensics - research demo")
    analyze_tab()
with tab_b:
    about_tab()

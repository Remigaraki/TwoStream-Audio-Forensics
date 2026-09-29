---
title: TwoStream Audio Forensics Demo
emoji: 🎙️
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# TwoStream-Audio-Forensics: research demo

Upload a speech recording (WAV, FLAC or MP3). The first 4 seconds are scored by
**C1**, a two-stream (RawNet2 + LFCC/bispectrum) cross-modal attention model
trained on ASVspoof 5. **A1** (RawNet2 only) can be run as a separate comparison.

This is an experimental research model. Its output is a model score, not proof of
authenticity or identity. Reported error rates apply to an in-domain ASVspoof 5
test split only; see the app's *About the research* tab.

Uploads are processed in temporary storage and are not retained or logged.

Source and full documentation: the TwoStream-Audio-Forensics repository (`demo/README.md`).

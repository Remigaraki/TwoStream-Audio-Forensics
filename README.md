# 🎧 Hearing Reality: A Two-Stream Neuro-Symbolic Architecture for Deepfake Audio Detection in Lossy Environments

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

## 🚀 Overview
**Hearing Reality** is an academic-grade forensic tool designed to detect AI-synthesized voices (Deepfake Audio) with a focus on robustness in **lossy environments**. While traditional models often fail when audio is subjected to heavy compression (e.g., social media uploads, VOIP calls), our architecture leverages a dual-stream approach to maintain high detection accuracy.

The project combines:
1.  **Raw Waveform Embeddings (RawNet2):** Capturing fine-grained temporal and spectral patterns directly from the source.
2.  **Higher-Order Statistical Features (Bispectrum/Bicoherence):** Utilizing mathematical "microscopes" to detect phase coupling anomalies and non-linearities characteristic of synthetic vocoders, even after MP3 or Opus compression.

---

## 📊 Architecture
Our **Two-Stream Fusion** network operates on a hybrid "Neuro-Symbolic" principle:
*   **The Neural Stream (RawNet2):** An end-to-end convolutional encoder that processes raw audio samples.
*   **The Statistical Stream:** A multi-layer perceptron (MLP) that ingests 128-dimensional Bispectral vectors, capturing higher-order phase interactions that deep learning models often overlook.
*   **Fusion Layer:** A concatenated embedding (1280 dimensions) passed through a SiLU-activated classifier for final Real vs. Fake (Bonafide vs. Spoof) attribution.

---

## 📂 Repository Structure
```text
TwoStream-Audio-Forensics/
├── .github/                # CI/CD and project metadata
├── dataset.py              # ASVspoof 5 dataset loader & preprocessing
├── feature_extraction.py   # Higher-order phase coupling (Bispectrum) logic
├── models.py               # RawNet2 & Two-Stream Fusion Net implementation
├── dry_run.py              # Architecture validation & pipeline test script
├── requirements.txt        # Python dependencies
└── README.md               # You are here
```

---

## 🎧 Datasets
We evaluate our models on state-of-the-art deepfake benchmarks, with a specialized focus on codec-degraded scenarios:
*   **ASVspoof 5 (2025):** The latest gold standard for automatic speaker verification spoofing and counterfeiting.
*   **WaveFake:** A large-scale dataset of diverse GAN and diffusion-based vocoders.
*   **Codec-Degraded Sets:** Custom-augmented versions of the above datasets using **MP3** (64-128kbps) and **Opus** (24-48kbps) to simulate real-world lossy environments.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
Ensure you have **FFmpeg** installed on your system to handle audio transcoding:
*   **Windows (Chocolatey):** `choco install ffmpeg`
*   **macOS (Homebrew):** `brew install ffmpeg`
*   **Linux:** `sudo apt install ffmpeg`

### 2. Python Environment
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/Remigaraki/TwoStream-Audio-Forensics.git
cd TwoStream-Audio-Forensics
pip install -r requirements.txt
```

---

## 🛠️ Usage / Pipeline

### 1. Data Augmentation & Compression
Generate lossy versions of the dataset to simulate real-world degradation.
```bash
# Example: Apply MP3/Opus compression to raw FLAC files
python scripts/augment_data.py --input_dir ./data/raw --output_dir ./data/lossy --codec opus
```

### 2. Feature Extraction
Pre-compute the higher-order statistical features (Bispectrum) for the Statistical Stream.
```bash
python feature_extraction.py --input_dir ./data/lossy --output_file features_stats.npy
```

### 3. Training Loop
Train the Two-Stream Fusion network using the PyTorch training script.
```bash
python train.py --epochs 50 --batch_size 32 --lr 0.0001
```

### 4. Launching the Web Demo
Experience the detection pipeline in real-time via the Streamlit interface.
```bash
streamlit run app.py
```

---

## 👥 Team Roles
This project was developed for the thesis "Hearing Reality" by our research team:
*   **Rafael**: AI Architecture & Fusion Strategy
*   **Louis**: Data Pipeline, Augmentation & DevOps
*   **Neilsen**: Data Science, Statistical Feature Engineering & Bispectral Analysis

---

## 📜 Citation
If you use this work in your research, please cite our thesis:
```bibtex
@thesis{HearingReality2026,
  title={Hearing Reality: A Two-Stream Neuro-Symbolic Architecture for Deepfake Audio Detection in Lossy Environments},
  author={Rafael, Louis, Neilsen},
  year={2026},
  institution={Your University}
}
```
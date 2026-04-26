# 🎧 Hearing Reality: A Two-Stream Neuro-Symbolic Architecture for Deepfake Audio Detection in Lossy Environments

[![CI Pipeline](https://github.com/Remigaraki/TwoStream-Audio-Forensics/actions/workflows/ci.yml/badge.svg)](https://github.com/Remigaraki/TwoStream-Audio-Forensics/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

## 🚀 Overview
**Hearing Reality** is a highly robust, two-stream deepfake audio detector. It tackles the core challenge of robustness against lossy compression (MP3, Opus) and cross-dataset generalization. 

The architecture combines:
1. **Neural Baseline (Stream 1):** RawNet2 processes raw waveform embeddings.
2. **Symbolic/Statistical Extractor (Stream 2):** Bispectrum/Bicoherence features capture higher-order phase coupling.
3. **Attention-based Fusion:** Dynamically weights the importance of Stream 1 and Stream 2.

We train and evaluate on **ASVspoof 5 (2025)** and **WaveFake**.

---

## 🏗️ Infrastructure & CI/CD Paradigm
We use a **Hybrid Local-Cloud CI/CD Pipeline** with strict branching strategies and reproducible environments:

1. **GitFlow Branching Strategy:** Development strictly follows GitFlow. All changes must be pushed to `feature/*` branches, merged into `dev`, and ultimately deployed to `main`. CI validates all branches.
2. **Local Development:** Write modules, perform unit testing, and push to GitHub using VS Code/Antigravity.
3. **Version Control & CI (GitHub):** GitHub Actions automatically runs `flake8` linting, `mypy` type checking, and lightweight `pytest` unit tests.
4. **Cloud Execution (Google Colab):** Google Colab serves as our Cloud GPU runner. It pulls the latest code from GitHub, uses the pinned `environment.yml`, mounts Google Drive containing the datasets, and executes the heavy PyTorch training loops.

---

## 📂 Codebase Architecture
```text
hearing_reality/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD: Automated linting, type-checking, and mock tests
├── data/
│   ├── dataset_loader.py          # PyTorch DataLoaders for ASVspoof 5 & WaveFake
│   └── torture_pipeline.py        # "Torture Chamber": MP3/Opus compression augmentation (pydub/ffmpeg)
├── src/
│   ├── stream1/                   
│   │   └── rawnet2.py             # Neural Baseline (Stream 1)
│   ├── stream2/                   
│   │   └── extract_bispectrum.py  # Statistical Higher-Order Phase Coupling math (Stream 2)
│   ├── fusion/
│   │   ├── attention_fusion.py    # Attention mechanism combining Stream 1 and Stream 2
│   │   └── two_stream_net.py      # The final combined PyTorch nn.Module
│   └── utils/                     
│       ├── audio_utils.py         # 16kHz resampling, 4-second padding/trimming (64,000 samples)
│       ├── metrics.py             # EER and t-DCF calculation logic
│       └── logger.py              # Training logs, TensorBoard hooks
├── experiments/                   # Stores experiment configurations and outputs
├── notebooks/
│   └── Colab_Main_Runner.ipynb    # THE CLOUD RUNNER: Mounts GDrive, git pulls repo, runs training
├── environment.yml                # Conda environment with pinned PyTorch, CUDA, numpy, torchaudio
├── requirements.txt               # Strict dependencies (pip fallback)
└── README.md                      # Instructions on the Local -> Github -> Colab pipeline
```

---

## 🛠️ Usage

### 1. Conda Environment Setup
Create the reproducible shared Conda environment with pinned versions:
```bash
conda env create -f environment.yml
conda activate hearing_reality
```

### 2. Run the "Torture" Pipeline
Simulate real-world lossy environments on your audio data:
```bash
python data/torture_pipeline.py --input_dir ./data/raw --output_dir ./data/lossy --codec opus
```

### 3. Training on Colab
1. Upload your datasets to Google Drive.
2. Open `notebooks/Colab_Main_Runner.ipynb` in Google Colab.
3. Run the cells to mount Drive, pull the latest `main` branch, setup the Conda environment, and start training on a Cloud GPU.
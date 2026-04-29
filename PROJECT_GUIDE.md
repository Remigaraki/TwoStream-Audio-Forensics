# Hearing Reality: A Two-Stream Neuro-Symbolic Architecture for Deepfake Audio Detection

## 1. Project Objective
We are building a highly robust deepfake audio detector capable of surviving real-world degradation (lossy compression like WhatsApp/Zoom). The hypothesis is that AI-generated audio (via Neural Vocoders like HiFi-GAN) leaves mathematical "Phase Coupling" traces in high frequencies. We are proving that combining a raw waveform analyzer with a statistical bispectrum analyzer yields superior detection in degraded environments.

## 2. The Dataset & Preparation
* [cite_start]**Datasets**: ASVspoof 5 (2025) and WaveFake (featuring 6 specific neural vocoders)[cite: 16].
* **The "Torture Chamber"**: Before the model sees the data, the training pipeline will intentionally degrade the audio with MP3 and Opus lossy compression to force the model to learn indestructible features.

## 3. Architecture Breakdown
The project relies on a "Two-Stream" Late Fusion approach:

### Stream 1: "The Ear" (Perceptual Features)
* **Model**: RawNet2 (1D-CNN with Sinc-convolution layers).
* **Input**: 4-second raw waveforms (64,000 samples).
* **Output**: An embedding vector `E_raw` of shape `[batch_size, 256]`.

### Stream 2: "The Microscope" (Forensic/Statistical Features)
* **Feature Extraction**: Extracts LFCCs (120-dim) combined with Bispectral Analysis (128x128 matrix reduced via PCA to K=128).
* **Model**: A 3-Layer MLP (256 -> 128 -> 64).
* **Output**: An embedding vector `E_stat` of shape `[batch_size, 64]`.

### The Fusion Layer (The Brain)
* **Process**: Concatenates `E_raw` (256) and `E_stat` (64).
* **Mechanism**: Passes through Global Average Pooling and a Cross-Modal Attention mechanism (using `E_stat` as the Query to cross-check perceptual features).
* [cite_start]**Output**: A final Sigmoid Classifier outputting a binary prediction (0 = Bona fide/Real, 1 = Spoofed/Fake)[cite: 17].

## 4. Phase Execution Plan
* **Phase 1 (Data Ingestion & Torture)**: Acquire ASVspoof 5 and WaveFake datasets, verify checksums, standardise 16kHz/4s constraints, and build the codec compression pipeline.
* **Phase 2 (Data Forensics & Stats)**: Neilsen builds the Bispectrum mathematical feature extraction and resolves dataset discrepancies.
* **Phase 3 (AI Architecture Generation)**: Rafael builds RawNet2, the Fusion module, and the core end-to-end training wrappers. 
* [cite_start]**Phase 4 (Cloud Orchestration & Telemetry)**: The team spins up `Colab_Main_Runner.ipynb` to mount Google Drive, ingest the `dataset_loader.py`, run the PyTorch loop, and rigorously evaluate EER and t-DCF metrics[cite: 17]. 
* [cite_start]**Phase 5 (Submission & Defense)**: Final RTF optimizations, complete the `README.md` for strict reproducibility, and prepare the live Streamlit/Colab Defense Demo[cite: 36, 108].# Hearing Reality: A Two-Stream Neuro-Symbolic Architecture for Deepfake Audio Detection

## 1. Project Objective
We are building a highly robust deepfake audio detector capable of surviving real-world degradation (lossy compression like WhatsApp/Zoom). The hypothesis is that AI-generated audio (via Neural Vocoders like HiFi-GAN) leaves mathematical "Phase Coupling" traces in high frequencies. We are proving that combining a raw waveform analyzer with a statistical bispectrum analyzer yields superior detection in degraded environments.

## 2. The Dataset & Preparation
* [cite_start]**Datasets**: ASVspoof 5 (2025) and WaveFake (featuring 6 specific neural vocoders)[cite: 16].
* **The "Torture Chamber"**: Before the model sees the data, the training pipeline will intentionally degrade the audio with MP3 and Opus lossy compression to force the model to learn indestructible features.

## 3. Architecture Breakdown
The project relies on a "Two-Stream" Late Fusion approach:

### Stream 1: "The Ear" (Perceptual Features)
* **Model**: RawNet2 (1D-CNN with Sinc-convolution layers).
* **Input**: 4-second raw waveforms (64,000 samples).
* **Output**: An embedding vector `E_raw` of shape `[batch_size, 256]`.

### Stream 2: "The Microscope" (Forensic/Statistical Features)
* **Feature Extraction**: Extracts LFCCs (120-dim) combined with Bispectral Analysis (128x128 matrix reduced via PCA to K=128).
* **Model**: A 3-Layer MLP (256 -> 128 -> 64).
* **Output**: An embedding vector `E_stat` of shape `[batch_size, 64]`.

### The Fusion Layer (The Brain)
* **Process**: Concatenates `E_raw` (256) and `E_stat` (64).
* **Mechanism**: Passes through Global Average Pooling and a Cross-Modal Attention mechanism (using `E_stat` as the Query to cross-check perceptual features).
* [cite_start]**Output**: A final Sigmoid Classifier outputting a binary prediction (0 = Bona fide/Real, 1 = Spoofed/Fake)[cite: 17].

## 4. Phase Execution Plan
* **Phase 1 (Data Ingestion & Torture)**: Acquire ASVspoof 5 and WaveFake datasets, verify checksums, standardise 16kHz/4s constraints, and build the codec compression pipeline.
* **Phase 2 (Data Forensics & Stats)**: Neilsen builds the Bispectrum mathematical feature extraction and resolves dataset discrepancies.
* **Phase 3 (AI Architecture Generation)**: Rafael builds RawNet2, the Fusion module, and the core end-to-end training wrappers. 
* [cite_start]**Phase 4 (Cloud Orchestration & Telemetry)**: The team spins up `Colab_Main_Runner.ipynb` to mount Google Drive, ingest the `dataset_loader.py`, run the PyTorch loop, and rigorously evaluate EER and t-DCF metrics[cite: 17]. 
* [cite_start]**Phase 5 (Submission & Defense)**: Final RTF optimizations, complete the `README.md` for strict reproducibility, and prepare the live Streamlit/Colab Defense Demo[cite: 36, 108].
# AI Agent Guardrails and Coding Rules

## 1. Core Execution Rules
* [cite_start]**No Monoliths**: The codebase must be strictly modular[cite: 17, 18]. Do not generate 500-line scripts.
* **Engine vs. Dashboard**: `.py` files are the "Engine" (models, data loaders, utils). [cite_start]Jupyter Notebooks (`.ipynb`) are strictly the "Dashboard" for execution and visualization[cite: 17]. 
* [cite_start]**The "Colab-First" Rule**: All code will ultimately run on Google Colab[cite: 121]. Do not hardcode local Windows absolute paths (e.g., `D:\Main\Hearing Reality\...`). Use relative paths and `os.path.join`.
* **Clean Commits**: Always commit notebooks with empty output cells to prevent massive GitHub diffs and merge conflicts.
* **Strict Typing**: Use Python type hints (e.g., `def load_audio(path: str) -> torch.Tensor:`) across all modules.

## 2. Hardcoded Architectural Constraints
Under no circumstances should the agent deviate from these mathematical constraints:
* [cite_start]**Audio Standardization**: EVERY audio file must be resampled to **16kHz**[cite: 13, 16]. 
* [cite_start]**Audio Shape Enforcement**: EVERY audio file must be strictly **4 seconds long (64,000 samples)**[cite: 13, 16]. If shorter, pad with zeros; if longer, crop a random 4-second segment.
* **Evaluation Metrics**: Never use standard "Accuracy." [cite_start]The only acceptable evaluation metrics are **Equal Error Rate (EER)** and **tandem Detection Cost Function (t-DCF)**[cite: 17]. 
* [cite_start]**Inference Speed**: The code must be optimized so the Real-Time Factor (RTF) is `< 1.0` on a Colab T4 GPU[cite: 32, 97].

## 3. Team Domain Assignments (Do Not Cross-Contaminate)
* [cite_start]**Rafael (AI Architect)**: Owns `rawnet2.py` (Stream 1), `attention_fusion.py`, and the PyTorch training loop[cite: 15].
* [cite_start]**Louis/Patrick (Data Engineering & Ops)**: Owns `dataset_loader.py`, the FFmpeg/pydub robustness "Torture" pipeline (Opus/MP3 compression), GPU resource allocation, and the deployment Notebooks[cite: 15, 16].
* [cite_start]**Neilsen (Data Science & Stats)**: Owns `extract_bispectrum.py` (Stream 2 MLP), PCA mathematical pipelines, and ablation evaluation logic[cite: 41, 44].
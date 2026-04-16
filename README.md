---
title: OdioCheck-Backend
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

<img width="1080" height="324" alt="odiocheck" src="https://github.com/user-attachments/assets/4d7b573e-5b0b-4fc7-85de-da60bbb701c2" />

# OdioCheck - Deepfake Voice Detection AI
*50.021 Artificial Intelligence Project*

## Theme
**AI for Security & Social Good** (UN SDG #16: Peace, Justice, and Strong Institutions)
OdioCheck tackles the rising threat of audio deepfakes used in scams and misdirection.

## Requirements Checklist
- [x] **Fully functioning code:** Complete end-to-end PyTorch implementation from dataset loading to real-time inference via a web UI.
- [x] **Baseline models (×3):**
  - **Wav2Vec2** — self-supervised transformer feature extractor (frozen) + attentive pooling classifier (`backend/models.py`)
  - **AASIST** — graph-based SOTA baseline using sinc-filter frontend + spectro-temporal heterogeneous graph attention (`backend/models.py`)
  - **CQCC Baseline** — standard CNN processing Constant-Q Cepstral Coefficients (`backend/models.py`)
- [x] **SOTA Custom Model:** `ImprovedWav2Vec2CQCCDetector` — a novel fusion architecture combining Wav2Vec 2.0 and CQCC features via **bidirectional cross-attention**, followed by a **Graph Attention** backend (`backend/models.py`).
- [x] **Ablation Study (×4):** Four ablation variants systematically isolate each architectural component to validate the custom model design:
  - **Ablation 1** — Wav2Vec2 + Graph (no CQCC, no cross-attention)
  - **Ablation 2** — CQCC + Graph (no Wav2Vec2, no cross-attention)
  - **Ablation 3** — Wav2Vec2 + CQCC + Simple Concat + Graph (no cross-attention)
  - **Ablation 4** — Wav2Vec2 + CQCC + Cross-Attention + Linear (no Graph Attention)
- [x] **Fully Working Frontend:** Glassmorphic UI (Tailwind + Vanilla JS) served via FastAPI. Supports OGG/MP3/M4A/FLAC/WAV. Shows **side-by-side** predictions from all four primary models with real-time animated confidence bars and a per-window **temporal analysis timeline chart**.
- [x] **Cross-lingual Dataset Split:** Trained on English audio (`MLAAD-tiny/en`), tested on unseen German audio (`MLAAD-tiny/de`) for out-of-distribution generalisation evaluation.
- [x] **CQCC Feature Caching:** Pre-computed CQCC tensors are cached to disk to avoid redundant computation across training runs.

---

## Installation

Ensure you have Python 3.9+ installed. Install all dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Download
We use the `MLAAD-tiny` dataset (multi-language audio deepfakes). Download it from Hugging Face before training:
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download mueller91/MLAAD-tiny --repo-type dataset --local-dir MLAAD-tiny
```

---

## Running the Project

### Step 1 — (Optional) Pre-compute CQCC Cache
Pre-computing CQCC features once dramatically speeds up all subsequent training runs:
```bash
python backend/train.py --precompute-cqcc-only
```

### Step 2 — Train All Models
Trains all 4 primary models + 4 ablation variants, evaluates on the German test set, and saves `.pth` weights to `backend/models/`:
```bash
python backend/train.py
```

#### Available Training Flags
| Flag | Default | Description |
|---|---|---|
| `--val-split F` | `0.2` | Fraction of English data reserved for validation (0–0.5). |
| `--data-dir PATH` | auto | Override dataset root (must contain `original/` and `fake/` folders). |
| `--cqcc-cache-dir PATH` | `backend/precomputed_features/cqcc` | Where to read/write cached CQCC tensors. |
| `--precompute-cqcc-only` | `False` | Build CQCC cache and exit without training. |
| `--force-rebuild-cqcc` | `False` | Recompute CQCC cache even if files already exist. |
| `--smoke-test` | `False` | Run one forward pass through every model and exit — useful for verifying setup. |

#### Quick Smoke Test
Verify all models initialise and run a forward pass correctly without full training:
```bash
python backend/train.py --smoke-test
```

### Step 3 — Start the Web Interface
```bash
uvicorn backend.app:app --reload
```
Open **http://127.0.0.1:8000** in your browser. Upload any audio file (WAV, MP3, OGG, FLAC, M4A) to see simultaneous predictions from all four primary models plus an animated temporal confidence chart.

---

## Project Architecture

```
AI Project/
├── backend/
│   ├── models.py          # All model architectures (3 baselines + custom + 4 ablations)
│   ├── dataset.py         # AudioDataset with CQCC caching + data augmentation
│   ├── train.py           # Full training + evaluation pipeline (CLI-driven)
│   ├── app.py             # FastAPI inference server (windowed temporal analysis)
│   ├── preprocess.py      # Standalone preprocessing utilities
│   └── models/            # Saved .pth weight files (generated after training)
├── frontend/
│   ├── index.html         # Glassmorphic UI shell
│   ├── script.js          # File upload, Chart.js timeline, model panel rendering
│   └── style.css          # Custom glassmorphism styles
├── MLAAD-tiny/            # Dataset (downloaded separately)
├── requirements.txt       # Python dependencies
└── colab_training_notebook.ipynb  # Google Colab training notebook
```

---

## Working with Other Datasets
To replace MLAAD-tiny with another dataset (e.g., ASVspoof):
1. Place your `fake/` and `original/` (or `real/`) audio folders into a `data/` directory at the project root.
2. The `AudioDataset` in `dataset.py` auto-detects and falls back to the `data/` directory if `MLAAD-tiny` is absent.
3. Re-run `python backend/train.py`. The full pipeline runs identically.

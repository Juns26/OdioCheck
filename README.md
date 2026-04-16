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

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-brightgreen?style=for-the-badge&logo=vercel)](https://odio-check.vercel.app/)
[![Backend](https://img.shields.io/badge/Backend-Hugging%20Face-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/JunSiang26/OdioCheck-Backend)

OdioCheck is a cutting-edge deepfake audio detection system designed to tackle the rising threat of voice clones used in scams and misinformation. It features a unique hybrid fusion architecture that outperforms standard SOTA baselines.

## 🚀 Live Demo
**Web Interface:** [https://odio-check.vercel.app/](https://odio-check.vercel.app/)

---

## 🏗️ System Architecture
The project uses a **Hybrid Cloud** deployment to ensure high performance and scalability:
- **Frontend:** Hosted on **Vercel** for lightning-fast loading and smooth UI interactions.
- **Backend:** A **FastAPI** server running inside a **Docker** container on **Hugging Face Spaces**, providing the high RAM and CPU required for Pytorch model inference.
- **Model Storage:** Heavy `.pth` model weights (approx 800MB) are managed via **Git LFS** on Hugging Face to keep the source code repository lightweight.

---

## 🧠 Model Requirements Checklist
- [x] **Fully functioning code:** Complete end-to-end PyTorch implementation from dataset loading to real-time inference via a web UI.
- [x] **Baseline models (×3):**
  - **Wav2Vec2** — self-supervised transformer feature extractor (frozen) + attentive pooling classifier.
  - **AASIST** — graph-based SOTA baseline using sinc-filter frontend + spectro-temporal heterogeneous graph attention.
  - **CQCC Baseline** — standard CNN processing Constant-Q Cepstral Coefficients.
- [x] **SOTA Custom Model:** `ImprovedWav2Vec2CQCCDetector` — a novel fusion architecture combining Wav2Vec 2.0 and CQCC features via **bidirectional cross-attention**, followed by a **Graph Attention** backend.
- [x] **Ablation Study (×4):** Four ablation variants systematically isolate each architectural component to validate the custom model design.
- [x] **Fully Working Frontend:** Glassmorphic UI served via FastAPI. Supports OGG/MP3/M4A/FLAC/WAV with real-time **temporal analysis timeline charts**.
- [x] **Cross-lingual Evaluation:** Trained on English audio, tested on unseen German audio (MLAAD-tiny) to evaluate out-of-distribution generalisation.

---

## 🛠️ Local Installation & Setup

### 1. Install Dependencies
Ensure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

### 2. Dataset Download
Download the `MLAAD-tiny` dataset before training:
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download mueller91/MLAAD-tiny --repo-type dataset --local-dir MLAAD-tiny
```

### 3. Training & Evaluation
To train all 4 primary models and 4 ablation variants:
```bash
python backend/train.py
```
*Weights will be saved to `backend/models/*.pth`.*

---

## 💻 Running the App Locally

### Method A: Connect to Production Backend (Default)
The frontend is configured to automatically detect if you are running on `localhost` and can be switched to point to your local backend in `frontend/script.js`.

### Method B: Run Local Backend
```bash
uvicorn backend.app:app --reload
```
Open **http://127.0.0.1:8000** to access the interface.

---

## 📁 Project Structure
```
AI Project/
├── backend/
│   ├── models.py          # All architectures (3 baselines + custom + 4 ablations)
│   ├── dataset.py         # AudioDataset with CQCC caching & augmentation
│   ├── train.py           # Full training & evaluation pipeline
│   ├── app.py             # FastAPI inference server (temporal analysis logic)
│   └── models/            # .pth weights (Stored via Git LFS on Hugging Face)
├── frontend/
│   ├── index.html         # UI Shell
│   ├── script.js          # "Smart" URL switcher & visualization logic
│   └── style.css          # Glassmorphism design system
├── Dockerfile             # Production container config for Hugging Face
└── requirements.txt       # Python dependencies
```

---

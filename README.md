# OdioCheck 🚨 - Deepfake Voice Detection AI
*50.021 Artificial Intelligence Project*

## Theme
**AI for Security & Social Good** (UN SDG #16: Peace, Justice, and Strong Institutions)
OdioCheck tackles the rising threat of audio deepfakes used in scams and misdirection.

## Requirements Checklist
- [x] **Fully functioning code:** Complete end-to-end PyTorch implementation handling dataset modeling to inference.
- [x] **Baseline models:** Includes a self-supervised transformer baseline extracted natively via **Wav2Vec 2.0**, a graph-based SOTA baseline using **AASIST** on mel-spectrograms, and a standard CNN baseline processing Constant-Q Cepstral Coefficients (**CQCC**) (`backend/models.py`).
- [x] **SOTA Custom Model:** A novel custom fusional architecture combining Wav2Vec 2.0 and CQCC features through bidirectional cross-attention, followed by a true Graph Transformer backend (`backend/models.py`).
- [x] **Fully Working Frontend:** Sleek, glassmorphic frontend UI built with Vanilla JS+Tailwind served via FastAPI (now natively supports OGG/MP3/M4A processing and provides **side‑by‑side comparison** between the baseline and SOTA models!).
- [x] **Hybrid Dataset:** Trained and evaluated using the MLAAD-tiny dataset. This lightweight multi-language dataset allows for realistic evaluation of audio deepfake capabilities without needing a 50GB download. (Also includes a fallback synthesizer if no data is found!)

## Installation
Ensure you have Python installed. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Download
We use the `MLAAD-tiny` dataset to train and evaluate the models. Before running the project, download the dataset from Hugging Face:
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download mueller91/MLAAD-tiny --repo-type dataset --local-dir MLAAD-tiny
```

## Running the Project
1. **Train Models and Generate Metrics:**
   ```bash
   python backend/train.py
   ```
   *This automatically trains on the downloaded MLAAD-tiny dataset (or generates dummy data as a fallback). It trains **four networks** – the Wav2Vec 2.0 spoof detector, the AASIST baseline, the CQCC baseline, and the ultimate Custom Fusion Model – and saves an ROC‑AUC curve chart comparing all of them to `models/roc_curve.png`.*

2. **Start the Web Interface:**
   ```bash
   uvicorn backend.app:app --reload
   ```
   *The server starts on `http://127.0.0.1:8000`. It launches the highly aesthetic visualizer UI, loads all trained models (**Wav2Vec2, AASIST, CQCC, and Custom Fusion**) and shows predictions from each side‑by‑side. Try uploading or dropping a `.wav` file directly to see the multi‑model comparison!* 

## Working with Other Datasets
To replace the MLAAD-tiny dataset with another dataset like ASVspoof:
1. Simply place your `fake/` and `original/` (or `real/`) audio folders into a `data/` directory at the project root.
2. The `dataset.py` logic will automatically detect and fall back to the `data/` directory if `MLAAD-tiny` is removed or modified.
3. Re-run `train.py`. The model will perform the exact same end-to-end mel-spectrogram learning pipeline!

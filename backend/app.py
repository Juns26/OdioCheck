import os
import sys

# Add the current directory to sys.path so we can import local modules
sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dataset import compute_cqcc
import librosa

from models import (
    Wav2Vec2SpoofDetector,
    AASISTDetector,
    CQCCBaselineDetector,
    ImprovedWav2Vec2CQCCDetector
)

app = FastAPI(title="Deepfake Voice Detection")

@app.get("/health")
def health():
    return {"status": "online"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# Load Models
# -------------------------------------------------------

models_dir = os.path.join(os.path.dirname(__file__), "models")

def load_model(model, filename):
    path = os.path.join(models_dir, filename)
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=device)
        # Handle weight_norm parametrization mismatch (common in Wav2Vec2 between versions)
        # This converts the 'parametrizations' keys back to 'weight_g' and 'weight_v'
        new_state_dict = {}
        for k, v in state_dict.items():
            if "pos_conv_embed.conv.parametrizations.weight.original0" in k:
                new_key = k.replace("parametrizations.weight.original0", "weight_g")
                new_state_dict[new_key] = v
            elif "pos_conv_embed.conv.parametrizations.weight.original1" in k:
                new_key = k.replace("parametrizations.weight.original1", "weight_v")
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        print(f"Loaded {filename}")
    else:
        print(f"WARNING: {filename} not found. Run train.py first!")
    model.eval()

    return model


wav2vec_model = load_model(
    Wav2Vec2SpoofDetector(num_classes=2).to(device),
    "wav2vec2.pth"
)

aasist_model = load_model(
    AASISTDetector(num_classes=2).to(device),
    "aasist.pth"
)

cqcc_baseline_model = load_model(
    CQCCBaselineDetector(num_classes=2).to(device),
    "cqcc_baseline.pth"
)

custom_hybrid_model = load_model(
    ImprovedWav2Vec2CQCCDetector(num_classes=2).to(device),
    "custom_hybrid.pth"
)


# -------------------------------------------------------
# Audio Preprocessing (mirrors dataset.py __getitem__)
# -------------------------------------------------------

TARGET_LEN = 64600  # AASIST standard: 4.025s at 16kHz
CQCC_N_BINS = 60    # Matches AudioDataset default

# 50% overlap: each step is half a window (~2s), giving smooth temporal curves
# without running 4x Wav2Vec2 passes per second.
WINDOW_STEP = TARGET_LEN // 2


def preprocess_window(wav_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Crop or pad a single audio window to TARGET_LEN, then compute waveform
    and CQCC tensors — identical to AudioDataset.__getitem__ (non-augmented).

    Returns:
        wav  : (1, TARGET_LEN) float32 tensor
        cqcc : (1, 20, T)      float32 tensor
    """
    # Center-crop or zero-pad to exactly TARGET_LEN (matches eval path in dataset.py)
    if len(wav_np) > TARGET_LEN:
        start = (len(wav_np) - TARGET_LEN) // 2
        wav_np = wav_np[start : start + TARGET_LEN]
    elif len(wav_np) < TARGET_LEN:
        wav_np = np.pad(wav_np, (0, TARGET_LEN - len(wav_np)), mode='constant')

    wav = torch.from_numpy(wav_np).unsqueeze(0).float()
    cqcc = compute_cqcc(wav_np, n_bins=CQCC_N_BINS)   # → (1, 20, T)
    return wav, cqcc


def run_window(wav: torch.Tensor, cqcc: torch.Tensor) -> dict:
    """
    Run all four models on a single window and return fake probabilities (0–100).
    """
    wav  = wav.unsqueeze(0).to(device)    # (1, 1, TARGET_LEN)
    cqcc = cqcc.unsqueeze(0).to(device)   # (1, 1, 20, T)

    with torch.no_grad():
        w2v_prob    = torch.softmax(wav2vec_model(wav),             dim=1)[0][1].item()
        aasist_prob = torch.softmax(aasist_model(wav),              dim=1)[0][1].item()
        cqcc_prob   = torch.softmax(cqcc_baseline_model(cqcc),      dim=1)[0][1].item()
        custom_prob = torch.softmax(custom_hybrid_model(wav, cqcc), dim=1)[0][1].item()

    return {
        "wav2vec2":      round(w2v_prob    * 100, 2),
        "aasist":        round(aasist_prob * 100, 2),
        "cqcc_baseline": round(cqcc_prob   * 100, 2),
        "custom_hybrid": round(custom_prob * 100, 2),
    }


def aggregate_prediction(fake_prob_pct: float) -> dict:
    """Convert a mean fake probability into the standard prediction dict."""
    return {
        "prediction":       "FAKE" if fake_prob_pct > 50 else "REAL",
        "fake_probability": fake_prob_pct,
        "real_probability": round(100 - fake_prob_pct, 2),
    }


# -------------------------------------------------------
# Prediction Endpoint
# -------------------------------------------------------
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Load at 16 kHz mono — identical to librosa.load call in dataset.py
        wav_np, sr = librosa.load(temp_path, sr=16000, mono=True)

        # -------------------------------------------------------
        # Slice into overlapping windows of TARGET_LEN samples.
        # Step = 50% overlap.  Very short clips produce a single window.
        # -------------------------------------------------------
        total_samples = len(wav_np)
        starts = list(range(0, total_samples, WINDOW_STEP))

        window_probs  = []   # per-window fake-probability dicts
        window_labels = []   # x-axis: start of each window in seconds

        for start in starts:
            chunk = wav_np[start : start + TARGET_LEN]
            wav_t, cqcc_t = preprocess_window(chunk)
            probs = run_window(wav_t, cqcc_t)
            window_probs.append(probs)

            start_sec = round(start / sr, 2)
            window_labels.append(start_sec)

        # -------------------------------------------------------
        # Overall prediction = mean fake probability across all windows
        # -------------------------------------------------------
        model_keys = ["wav2vec2", "aasist", "cqcc_baseline", "custom_hybrid"]
        overall = {}
        for key in model_keys:
            mean_fake = round(
                sum(w[key] for w in window_probs) / len(window_probs), 2
            )
            overall[key] = aggregate_prediction(mean_fake)

        # -------------------------------------------------------
        # Time-series data for the frontend chart
        # -------------------------------------------------------
        timeline = {
            key: [w[key] for w in window_probs]
            for key in model_keys
        }

        return JSONResponse({
            "overall":       overall,         # {model: {prediction, fake_probability, real_probability}}
            "timeline":      timeline,        # {model: [fake_prob_pct, ...]}  — one value per window
            "window_labels": window_labels,   # [start_sec, ...]               — x-axis in seconds (starts at 0.0)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# -------------------------------------------------------
# Serve frontend
# -------------------------------------------------------

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")

if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


# -------------------------------------------------------
# Run Server
# -------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    print("Starting server at http://127.0.0.1:8000")

    uvicorn.run(app, host="127.0.0.1", port=8000)
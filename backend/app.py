import os
import torch
import torchaudio
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import sys

sys.path.append(os.path.dirname(__file__))

from models import (
    Wav2Vec2SpoofDetector,
    AASISTDetector,
    CQCCBaselineDetector,
    ImprovedWav2Vec2CQCCDetector
)

app = FastAPI(title="Deepfake Voice Detection")

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
        model.load_state_dict(torch.load(path, map_location=device))
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
# Audio preprocessing
# -------------------------------------------------------


# -------------------------------------------------------
# Prediction Endpoint
# -------------------------------------------------------
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        import librosa
        wav_np, sr = librosa.load(temp_path, sr=16000, mono=True)
        wav = torch.from_numpy(wav_np).unsqueeze(0).float()

        # Pad / crop to 2 seconds
        target_len = 32000
        if wav.shape[1] < target_len:
            wav = F.pad(wav, (0, target_len - wav.shape[1]))
        else:
            wav = wav[:, :target_len]

        wav = wav.to(device)

        # Placeholder for CQCC extraction, normally you'd use a CQCC library here.
        # Generating a dummy tensor of shape (1, 1, 20, T) to avoid crashing the models for now.
        dummy_cqcc = torch.randn(1, 1, 20, 100).to(device) 

        with torch.no_grad():
            wav2vec_out = wav2vec_model(wav)
            wav2vec_prob = torch.softmax(wav2vec_out, dim=1)[0][1].item()

            aasist_out = aasist_model(wav)
            aasist_prob = torch.softmax(aasist_out, dim=1)[0][1].item()

            # Pass dummy CQCC to CQCC and Custom models for now
            cqcc_out = cqcc_baseline_model(dummy_cqcc)
            cqcc_prob = torch.softmax(cqcc_out, dim=1)[0][1].item()

            custom_out = custom_hybrid_model(wav, dummy_cqcc)
            custom_prob = torch.softmax(custom_out, dim=1)[0][1].item()

        result = {
            "wav2vec2": {
                "prediction": "FAKE" if wav2vec_prob > 0.5 else "REAL",
                "fake_probability": round(wav2vec_prob * 100, 2),
                "real_probability": round((1 - wav2vec_prob) * 100, 2)
            },
            "aasist": {
                "prediction": "FAKE" if aasist_prob > 0.5 else "REAL",
                "fake_probability": round(aasist_prob * 100, 2),
                "real_probability": round((1 - aasist_prob) * 100, 2)
            },
            "cqcc_baseline": {
                "prediction": "FAKE" if cqcc_prob > 0.5 else "REAL",
                "fake_probability": round(cqcc_prob * 100, 2),
                "real_probability": round((1 - cqcc_prob) * 100, 2)
            },
            "custom_hybrid": {
                "prediction": "FAKE" if custom_prob > 0.5 else "REAL",
                "fake_probability": round(custom_prob * 100, 2),
                "real_probability": round((1 - custom_prob) * 100, 2)
            }
        }
        return JSONResponse(result)

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
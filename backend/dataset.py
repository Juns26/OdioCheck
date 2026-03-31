import os
import hashlib
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
import librosa
from scipy.fftpack import dct

def compute_cqcc(wav_np, n_bins, sample_rate=16000, hop_length=160, num_coeffs=20):
    """Compute CQCC features from a mono waveform numpy array."""
    try:
        cqt = np.abs(
            librosa.cqt(
                wav_np,
                sr=sample_rate,
                n_bins=n_bins,
                hop_length=hop_length,
                fmin=librosa.note_to_hz('C1')
            )
        )
        log_power = librosa.amplitude_to_db(cqt, ref=np.max)
        cqcc = dct(log_power, type=2, axis=0, norm='ortho')[:num_coeffs]
        return torch.from_numpy(cqcc).unsqueeze(0).float()
    except Exception:
        # Fallback for very short or invalid audio.
        return torch.zeros((1, num_coeffs, 10), dtype=torch.float32)

class AudioDataset(Dataset):
    def __init__(self, data_dir=None, n_mels=60, augment=False, cqcc_cache_dir=None):
        if data_dir is None:
            # Check if MLAAD-tiny exists, else fallback to 'data'
            mlaad_dir = os.path.join(os.path.dirname(__file__), "..", "MLAAD-tiny")
            if os.path.exists(mlaad_dir):
                data_dir = mlaad_dir
            else:
                data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
                
        self.data_dir = data_dir
        self.files = []
        self.labels = []
        self.n_mels = n_mels
        self.augment = augment
        self.cqcc_cache_dir = cqcc_cache_dir
        
        real_path = os.path.join(data_dir, "original")
        if not os.path.exists(real_path):
            real_path = os.path.join(data_dir, "real")
            
        fake_path = os.path.join(data_dir, "fake")
            
        for root, dirs, files in os.walk(real_path):
            for f in files:
                if f.endswith('.wav') or f.endswith('.flac'):
                    self.files.append(os.path.join(root, f))
                    self.labels.append(0) # 0 = Real
                    
        for root, dirs, files in os.walk(fake_path):
            for f in files:
                if f.endswith('.wav') or f.endswith('.flac'):
                    self.files.append(os.path.join(root, f))
                    self.labels.append(1) # 1 = Fake
            
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=n_mels, n_fft=400, hop_length=160
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)

        if self.cqcc_cache_dir is not None:
            os.makedirs(self.cqcc_cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.files)

    def _cqcc_cache_path(self, audio_path):
        rel_path = os.path.relpath(audio_path, start=self.data_dir)
        cache_key = hashlib.md5(audio_path.encode("utf-8")).hexdigest()
        rel_stem = os.path.splitext(rel_path)[0]
        safe_name = rel_stem.replace(os.sep, "__")
        return os.path.join(self.cqcc_cache_dir, f"{safe_name}_{cache_key}.pt")

    def _load_or_compute_cqcc(self, idx, wav_np):
        if self.cqcc_cache_dir is None:
            return compute_cqcc(wav_np, n_bins=self.n_mels)

        cache_path = self._cqcc_cache_path(self.files[idx])
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu")

        cqcc = compute_cqcc(wav_np, n_bins=self.n_mels)
        torch.save(cqcc, cache_path)
        return cqcc

    def precompute_cqcc_cache(self, force=False):
        """Materialize CQCC features to disk so training can reuse them."""
        if self.cqcc_cache_dir is None:
            raise ValueError("cqcc_cache_dir must be set to precompute CQCC features.")

        total = len(self.files)
        for idx, audio_path in enumerate(self.files):
            cache_path = self._cqcc_cache_path(audio_path)
            if not force and os.path.exists(cache_path):
                continue

            wav_np, _ = librosa.load(audio_path, sr=16000, mono=True)
            cqcc = compute_cqcc(wav_np, n_bins=self.n_mels)
            torch.save(cqcc, cache_path)

            if (idx + 1) % 100 == 0 or idx + 1 == total:
                print(f"Precomputed CQCC {idx + 1}/{total}")

    def __getitem__(self, idx):
        wav_np, sr = librosa.load(self.files[idx], sr=16000, mono=True)
        
        # Augmentation on raw audio (Data Augmentation for generalizability)
        if self.augment and np.random.rand() < 0.3:
            noise = np.random.randn(len(wav_np)) * 0.002
            wav_np = wav_np + noise

        wav = torch.from_numpy(wav_np).unsqueeze(0).float()

        mel = self.mel_spectrogram(wav)
        mel = torchaudio.transforms.AmplitudeToDB()(mel)
        
        # Spectrogram Augmentation
        if self.augment and torch.rand(1).item() < 0.5:
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)
            
        cqcc = self._load_or_compute_cqcc(idx, wav_np)

        return mel, wav, cqcc, self.labels[idx]


def collate_variable_length(batch):

    mels, wavs, cqccs, labels = zip(*batch)
    labels = torch.tensor(labels)

    # ---------- MEL ----------
    max_mel_time = max(m.shape[-1] for m in mels)

    mels_padded = []
    for m in mels:
        if m.shape[-1] < max_mel_time:
            pad = max_mel_time - m.shape[-1]
            m = torch.nn.functional.pad(m, (0, pad))
        mels_padded.append(m)

    mels = torch.stack(mels_padded, dim=0)

    # ---------- WAVE ----------
    max_wav_len = max(w.shape[-1] for w in wavs)

    wavs_padded = []
    for w in wavs:
        if w.shape[-1] < max_wav_len:
            pad = max_wav_len - w.shape[-1]
            w = torch.nn.functional.pad(w, (0, pad))
        wavs_padded.append(w)

    wavs = torch.stack(wavs_padded, dim=0)
    
    # ---------- CQCC ----------
    max_cqcc_len = max(c.shape[-1] for c in cqccs)

    cqccs_padded = []
    for c in cqccs:
        if c.shape[-1] < max_cqcc_len:
            pad = max_cqcc_len - c.shape[-1]
            c = torch.nn.functional.pad(c, (0, pad))
        cqccs_padded.append(c)

    cqccs = torch.stack(cqccs_padded, dim=0)

    return mels, wavs, cqccs, labels

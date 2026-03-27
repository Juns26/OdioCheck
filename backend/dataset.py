import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
import librosa
from scipy.fftpack import dct

def generate_dummy_audio(num_samples=100, path='data/'):
    os.makedirs(os.path.join(path, "real"), exist_ok=True)
    os.makedirs(os.path.join(path, "fake"), exist_ok=True)
    
    if len(os.listdir(os.path.join(path, "real"))) > 0:
        return

    print("Generating demo fake/real synthetic dataset...")
    for i in range(num_samples):
        # "real" audio: pure white noise
        import scipy.io.wavfile
        real_wav = torch.randn(1, 16000) * 0.1
        scipy.io.wavfile.write(f'{path}/real/sample_{i}.wav', 16000, real_wav.squeeze(0).numpy())
        
        # "fake" audio: Add some sinusoidal high frequency artifacts 
        # to simulate robotic/deepfake generation artifacts.
        t = torch.linspace(0, 1, 16000)
        artifact = torch.sin(2 * 3.1415 * 5000 * t) * 0.05
        fake_wav = real_wav + artifact.unsqueeze(0)
        scipy.io.wavfile.write(f'{path}/fake/sample_{i}.wav', 16000, fake_wav.squeeze(0).numpy())

class AudioDataset(Dataset):
    def __init__(self, data_dir=None, n_mels=60, augment=False):
        if data_dir is None:
            # Check if MLAAD-tiny exists, else fallback to 'data'
            mlaad_dir = os.path.join(os.path.dirname(__file__), "..", "MLAAD-tiny")
            if os.path.exists(mlaad_dir):
                data_dir = mlaad_dir
            else:
                data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
                
        self.files = []
        self.labels = []
        self.n_mels = n_mels
        self.augment = augment
        
        real_path = os.path.join(data_dir, "original")
        if not os.path.exists(real_path):
            real_path = os.path.join(data_dir, "real")
            
        fake_path = os.path.join(data_dir, "fake")
        
        if not os.path.exists(real_path) or not os.path.exists(fake_path):
            generate_dummy_audio(100, data_dir)
            
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

    def __len__(self):
        return len(self.files)

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
            
        # Compute CQCC based on librosa CQT
        try:
            cqt = np.abs(librosa.cqt(wav_np, sr=16000, n_bins=self.n_mels, hop_length=160, fmin=librosa.note_to_hz('C1')))
            log_power = librosa.amplitude_to_db(cqt, ref=np.max)
            cqcc = dct(log_power, type=2, axis=0, norm='ortho')[:20]  # keep 20 coeffs
            cqcc = torch.from_numpy(cqcc).unsqueeze(0).float()
        except Exception:
            # Fallback for very short audio
            cqcc = torch.zeros((1, 20, 10))

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

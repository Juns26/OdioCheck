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
    def __init__(self, data_dir=None, n_bins=60, augment=False, cqcc_cache_dir=None, target_lang=None):
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
        self.n_bins = n_bins
        self.augment = augment
        self.cqcc_cache_dir = cqcc_cache_dir
        self.target_lang = target_lang

        real_path = os.path.join(data_dir, "original")
        if not os.path.exists(real_path):
            real_path = os.path.join(data_dir, "real")
            
        fake_path = os.path.join(data_dir, "fake")
            
        for root, dirs, files in os.walk(real_path):
            dirs.sort()
            files.sort()
            for f in files:
                if f.endswith('.wav') or f.endswith('.flac'):
                    if self.target_lang:
                        rel_root = os.path.relpath(root, real_path).replace('\\', '/')
                        if not rel_root.startswith(self.target_lang):
                            continue
                    self.files.append(os.path.join(root, f))
                    self.labels.append(0) # 0 = Real

        for root, dirs, files in os.walk(fake_path):
            dirs.sort()
            files.sort()
            for f in files:
                if f.endswith('.wav') or f.endswith('.flac'):
                    if self.target_lang:
                        rel_root = os.path.relpath(root, fake_path).replace('\\', '/')
                        if not rel_root.startswith(self.target_lang):
                            continue
                    self.files.append(os.path.join(root, f))
                    self.labels.append(1) # 1 = Fake

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

    def _load_or_compute_cqcc(self, audio_path, wav_np, is_augmented=False):
        if self.cqcc_cache_dir is None or is_augmented:
            return compute_cqcc(wav_np, n_bins=self.n_bins)

        cache_path = self._cqcc_cache_path(audio_path)
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu")

        cqcc = compute_cqcc(wav_np, n_bins=self.n_bins)
        torch.save(cqcc, cache_path)
        return cqcc

    def precompute_cqcc_cache(self, force=False):
        """Materialize CQCC features to disk so training can reuse them."""
        import tqdm
        if self.cqcc_cache_dir is None:
            raise ValueError("cqcc_cache_dir must be set to precompute CQCC features.")

        try:
            from tqdm.notebook import tqdm
            iterable_files = tqdm(self.files, desc="Precomputing CQCC Cache")
        except ImportError:
            iterable_files = self.files

        total = len(self.files)
        for idx, audio_path in enumerate(iterable_files):
            cache_path = self._cqcc_cache_path(audio_path)
            if not force and os.path.exists(cache_path):
                continue

            try:
                wav_np, _ = librosa.load(audio_path, sr=16000, mono=True)
                cqcc = compute_cqcc(wav_np, n_bins=self.n_bins)
                torch.save(cqcc, cache_path)
            except Exception as e:
                print(f"Error precomputing CQCC for {audio_path}: {e}")


            if (idx + 1) % 100 == 0 or idx + 1 == total:
                print(f"Precomputed CQCC {idx + 1}/{total}")

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        wav_np, sr = librosa.load(audio_path, sr=16000, mono=True)

        is_augmented = False
        # Augmentation on raw audio (Data Augmentation for generalizability)
        if self.augment and np.random.rand() < 0.3:
            # Apply only ONE augmentation type per sample to avoid over-modification
            aug_type = np.random.choice(['noise', 'speed', 'pitch'], p=[0.33, 0.33, 0.34])

            if aug_type == 'noise':
                # SNR-based noise addition (reverted to original robust method)
                signal_power = np.mean(wav_np**2)
                if signal_power > 1e-10:
                    snr_db = np.random.uniform(10, 30)
                    snr_linear = 10**(snr_db / 10)
                    noise_power = signal_power / snr_linear
                    noise = np.random.randn(len(wav_np)) * np.sqrt(noise_power)
                    wav_np = wav_np + noise
                is_augmented = True
            elif aug_type == 'speed':
                # Mild speed perturbation
                speed_factor = np.random.uniform(0.95, 1.05)
                wav_np = librosa.effects.time_stretch(wav_np, rate=speed_factor)
                is_augmented = True
            elif aug_type == 'pitch':
                # Subtle pitch shift
                n_steps = np.random.uniform(-1, 1)
                wav_np = librosa.effects.pitch_shift(wav_np, sr=sr, n_steps=n_steps)
                is_augmented = True

        # Crop or pad to exactly 64600 samples (AASIST standard)
        target_len = 64600
        if len(wav_np) > target_len:
            # Center crop or random crop for augment instead of taking just the start.
            if self.augment:
                start = np.random.randint(0, len(wav_np) - target_len)
            else:
                start = (len(wav_np) - target_len) // 2
            wav_np = wav_np[start:start+target_len]
        elif len(wav_np) < target_len:
            pad = target_len - len(wav_np)  
            wav_np = np.pad(wav_np, (0, pad), 'constant')

        wav = torch.from_numpy(wav_np).unsqueeze(0).float()

        cqcc = self._load_or_compute_cqcc(audio_path, wav_np, is_augmented=is_augmented)

        return wav, cqcc, self.labels[idx]


def collate_variable_length(batch):

    wavs, cqccs, labels = zip(*batch)
    labels = torch.tensor(labels)

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

    return wavs, cqccs, labels

import os
import io
import soundfile as sf
import shutil
import numpy as np
from datasets import load_dataset, Audio
from tqdm import tqdm

def download_data(num_samples_per_class=1000):
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    fake_dir = os.path.join(base_dir, 'fake')
    real_dir = os.path.join(base_dir, 'real')
    
    # Safely clear old data
    if os.path.exists(fake_dir): shutil.rmtree(fake_dir, ignore_errors=True)
    if os.path.exists(real_dir): shutil.rmtree(real_dir, ignore_errors=True)
    
    os.makedirs(fake_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)
    
    print("Loading Hybrid deepfake audio dataset from HuggingFace...")
    # Dataset 1: Hemg/Deepfake-Audio-Dataset
    try:
        ds1 = load_dataset('Hemg/Deepfake-Audio-Dataset', split='train', streaming=True)
        ds1 = ds1.cast_column('audio', Audio(decode=False))
        real_count, fake_count = 0, 0
        for item in tqdm(ds1, desc="Downloading Hemg Dataset"):
            label = item['label'] # 0 = Fake, 1 = Real
            audio_bytes = item['audio']['bytes']
            
            if label == 0 and fake_count < num_samples_per_class:
                cls_type = "fake"
                count = f"hemg_{fake_count}"
                fake_count += 1
            elif label == 1 and real_count < num_samples_per_class:
                cls_type = "real"
                count = f"hemg_{real_count}"
                real_count += 1
            else:
                if real_count >= num_samples_per_class and fake_count >= num_samples_per_class: break
                continue
                
            try:
                audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                if len(audio_data.shape) > 1: audio_data = audio_data.mean(axis=1)
                sf.write(os.path.join(base_dir, cls_type, f"sample_{count}.wav"), audio_data, sr)
            except Exception: pass
        print(f"Dataset 1 Done.")
    except Exception as e: print(f"Error Dataset 1: {e}")

    # Dataset 2: Bisher/ASVspoof_2019_LA
    try:
        ds2 = load_dataset('Bisher/ASVspoof_2019_LA', split='train', streaming=True)
        ds2 = ds2.cast_column('audio', Audio(decode=False))
        real_count, fake_count = 0, 0
        for item in tqdm(ds2, desc="Downloading ASVspoof Dataset"):
            key = item['key'] # 0 = bonafide (Real), 1 = spoof (Fake)
            audio_bytes = item['audio']['bytes']
            
            if key == 1 and fake_count < num_samples_per_class:
                cls_type = "fake"
                count = f"asv_{fake_count}"
                fake_count += 1
            elif key == 0 and real_count < num_samples_per_class:
                cls_type = "real"
                count = f"asv_{real_count}"
                real_count += 1
            else:
                if real_count >= num_samples_per_class and fake_count >= num_samples_per_class: break
                continue
                
            try:
                audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                if len(audio_data.shape) > 1: audio_data = audio_data.mean(axis=1)
                sf.write(os.path.join(base_dir, cls_type, f"sample_{count}.wav"), audio_data, sr)
            except Exception: pass
        print(f"Dataset 2 Done.")
    except Exception as e: print(f"Error Dataset 2: {e}")

    # Dataset 3: 3004lakshu/Deepfake-Audio (General English Deepfake Audio dataset)
    try:
        ds3 = load_dataset('3004lakshu/Deepfake-Audio', split='train', streaming=True)
        ds3 = ds3.cast_column('audio', Audio(decode=False))
        real_count, fake_count = 0, 0
        for item in tqdm(ds3, desc="Downloading Deepfake-Audio Dataset"):
            
            # Universal parsing for real/fake flags
            raw_label = str(item.get('label', item.get('key', item.get('class', -1)))).strip().lower()
            if raw_label in ['0', 'bonafide', 'real']:
                label_id = 1 # Assigned to real
            elif raw_label in ['1', 'spoof', 'fake']:
                label_id = 0 # Assigned to fake
            else:
                # Fallback if label is missing but dataset has 'fake'/'real' in the path or ID
                path_str = str(item.get('audio', {}).get('path', '')).lower()
                if 'fake' in path_str: label_id = 0
                elif 'real' in path_str: label_id = 1
                else: continue

            audio_bytes = item['audio']['bytes']
            
            if label_id == 0 and fake_count < num_samples_per_class:
                cls_type = "fake"
                count = f"lakshu_{fake_count}"
                fake_count += 1
            elif label_id == 1 and real_count < num_samples_per_class:
                cls_type = "real"
                count = f"lakshu_{real_count}"
                real_count += 1
            else:
                if real_count >= num_samples_per_class and fake_count >= num_samples_per_class: break
                continue
                
            try:
                audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                if len(audio_data.shape) > 1: audio_data = audio_data.mean(axis=1)
                sf.write(os.path.join(base_dir, cls_type, f"sample_{count}.wav"), audio_data, sr)
            except Exception: pass
        print(f"Dataset 3 Done. Hybrid Dataset complete.")
    except Exception as e: print(f"Error Dataset 3: {e}")

if __name__ == "__main__":
    download_data()

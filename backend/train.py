import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import AudioDataset, collate_variable_length
from models import (
    AASISTDetector,
    Wav2Vec2SpoofDetector,
    CQCCBaselineDetector,
    ImprovedWav2Vec2CQCCDetector
)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import random


def train_model(model, dataloader, criterion, optimizer, epochs=5, input_type='mel', device=None):
    """Train model on dataset."""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    loss_history = []

    for epoch in range(epochs):

        epoch_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:

            if input_type == 'mel':
                outputs = model(batch[0].to(device))
            elif input_type == 'wav':
                outputs = model(batch[1].to(device))
            elif input_type == 'cqcc':
                outputs = model(batch[2].to(device))
            elif input_type == 'wav_and_cqcc':
                outputs = model(batch[1].to(device), batch[2].to(device))
            else:
                raise ValueError("invalid input_type")

            labels = batch[-1].to(device)

            optimizer.zero_grad()

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total if total > 0 else 0
        avg_loss = epoch_loss / len(dataloader)

        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    return loss_history


def evaluate_model(model, dataloader, input_type='mel', device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():

        for batch in dataloader:

            if input_type == 'mel':
                outputs = model(batch[0].to(device))
            elif input_type == 'wav':
                outputs = model(batch[1].to(device))
            elif input_type == 'cqcc':
                outputs = model(batch[2].to(device))
            elif input_type == 'wav_and_cqcc':
                outputs = model(batch[1].to(device), batch[2].to(device))

            labels = batch[-1].to(device)

            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # ------------------
    # EER (Equal Error Rate)
    # ------------------
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_index]
    
    # ------------------
    # minDCF (Minimum Detection Cost Function)
    # Parameters for ASVSpoof/NIST standards
    # ------------------
    P_spoof = 0.05  # Prior probability of encountering a deepfake
    C_miss = 10     # Cost of missing a deepfake (False Negative)
    C_fa = 1        # Cost of falsely rejecting a real voice (False Positive)
    
    # DCF = C_miss * P_miss * P_spoof + C_fa * P_fa * (1 - P_spoof)
    # Where P_miss = fnr, P_fa = fpr
    dcf_array = C_miss * fnr * P_spoof + C_fa * fpr * (1 - P_spoof)
    min_dcf = np.min(dcf_array)

    return fpr, tpr, roc_auc, eer, min_dcf


def parse_args():
    parser = argparse.ArgumentParser(description="Train spoof-detection models with optional CQCC caching.")
    parser.add_argument(
        "--cqcc-cache-dir", # this is where cqcc is stored
        default=os.path.join(os.path.dirname(__file__), "precomputed_features", "cqcc"),
        help="Directory used to store and reuse precomputed CQCC tensors."
    )
    parser.add_argument(
        "--precompute-cqcc-only", 
        action="store_true",
        help="Only build the CQCC cache and exit without training."
    )
    parser.add_argument(
        "--subset-size", # this is where cqcc is stored
        type=int,
        default=100,
        help="Optional subset size for debugging. Set <= 0 to use the full dataset."
    )
    parser.add_argument(
        "--force-rebuild-cqcc",
        action="store_true",
        help="Recompute cached CQCC files even if they already exist."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load one batch, run a forward pass through each model, and exit without training."
    )
    return parser.parse_args()


def run_smoke_test(dataloader, device):
    print("\n--- Running Smoke Test ---")
    batch = next(iter(dataloader))
    mels, wavs, cqccs, labels = batch

    models_to_test = [
        ("Wav2Vec2 Baseline", Wav2Vec2SpoofDetector(num_classes=2).to(device), "wav"),
        ("AASIST Baseline", AASISTDetector(num_classes=2).to(device), "mel"),
        ("CQCC Baseline", CQCCBaselineDetector(num_classes=2).to(device), "cqcc"),
        ("Custom Fusion Model", ImprovedWav2Vec2CQCCDetector(num_classes=2).to(device), "wav_and_cqcc"),
    ]

    with torch.no_grad():
        for name, model, input_type in models_to_test:
            model.eval()
            if input_type == "mel":
                outputs = model(mels.to(device))
            elif input_type == "wav":
                outputs = model(wavs.to(device))
            elif input_type == "cqcc":
                outputs = model(cqccs.to(device))
            elif input_type == "wav_and_cqcc":
                outputs = model(wavs.to(device), cqccs.to(device))
            else:
                raise ValueError("invalid input_type")

            print(f"{name}: input OK, output shape = {tuple(outputs.shape)}")

    print(f"Labels shape = {tuple(labels.shape)}")
    print("Smoke test complete. Cached CQCC loading and model forward passes succeeded.")


def main():
    args = parse_args()
    print(args)
    
    # ------------------
    # Universal Seed for Colab Reproducibility
    # Ensure exact same train/test splits across separate script rums
    # ------------------
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    g = torch.Generator()
    g.manual_seed(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    print("Loading Dataset with Augmentation...")

    dataset = AudioDataset(augment=False, cqcc_cache_dir=args.cqcc_cache_dir)
    print(f"Using CQCC cache dir: {args.cqcc_cache_dir}")
    dataset.precompute_cqcc_cache(force=args.force_rebuild_cqcc)

    if args.precompute_cqcc_only:
        print("CQCC preprocessing complete. Exiting without training.")
        return
    
    # # Optional subset for debugging because MLAAD-tiny is big (13K)
    # subset_size = 100
    # dataset = torch.utils.data.Subset(dataset, range(subset_size))

    #sample n for real and fake each for training and testing
    n_real = 400
    n_fake = 400 
    
    labels = torch.tensor(dataset.labels)
    real_indices = (labels == 0).nonzero(as_tuple=True)[0]
    fake_indices = (labels == 1).nonzero(as_tuple=True)[0]
    
    # Randomly shuffle indices
    real_indices = real_indices[torch.randperm(len(real_indices))]
    fake_indices = fake_indices[torch.randperm(len(fake_indices))]
    
    # Enforce n selected
    selected_real = real_indices[:n_real]
    selected_fake = fake_indices[:n_fake]

    # Explicit 80/20 train-test split for EACH class separately to ensure equal proportions!
    train_real_size = int(0.8 * len(selected_real))
    train_fake_size = int(0.8 * len(selected_fake))

    # Build the train/test indices carefully to preserve exact proportions
    train_indices = torch.cat([
        selected_real[:train_real_size], 
        selected_fake[:train_fake_size]
    ])
    
    test_indices = torch.cat([
        selected_real[train_real_size:], 
        selected_fake[train_fake_size:]
    ])

    # Shuffle the final resulting indices so order is randomized during batching
    train_indices = train_indices[torch.randperm(len(train_indices))].tolist()
    test_indices = test_indices[torch.randperm(len(test_indices))].tolist()

    # Apply the perfectly stratified split!
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=2,
        pin_memory=True,
        generator=g, # ensure reproducible shuffling
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_variable_length,
        num_workers=2,
        pin_memory=True
    )

    if args.smoke_test:
        run_smoke_test(train_loader, device)
        return

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # 1 Wav2Vec2 Baseline
    # ============================================================

    print("\n--- Training Wav2Vec2 Baseline ---")

    wav2vec_model = Wav2Vec2SpoofDetector(num_classes=2).to(device)

    optimizer_wav2vec = torch.optim.Adam(wav2vec_model.parameters(), lr=1e-4)

    wav2vec_loss = train_model(
        wav2vec_model,
        train_loader,
        criterion,
        optimizer_wav2vec,
        input_type='wav',
        device=device
    )

    torch.save(wav2vec_model.state_dict(), os.path.join(models_dir, "wav2vec2.pth"))

    # ============================================================
    # 2 AASIST Baseline
    # ============================================================

    print("\n--- Training AASIST Baseline ---")

    aasist_model = AASISTDetector(num_classes=2).to(device)

    optimizer_aasist = torch.optim.Adam(aasist_model.parameters(), lr=5e-4)

    aasist_loss = train_model(
        aasist_model,
        train_loader,
        criterion,
        optimizer_aasist,
        input_type='mel',
        device=device
    )

    torch.save(aasist_model.state_dict(), os.path.join(models_dir, "aasist.pth"))

    # ============================================================
    # 3 CQCC Baseline
    # ============================================================

    print("\n--- Training CQCC Baseline ---")

    cqcc_baseline = CQCCBaselineDetector(num_classes=2).to(device)

    optimizer_cqcc = torch.optim.Adam(cqcc_baseline.parameters(), lr=1e-3)

    cqcc_loss = train_model(
        cqcc_baseline,
        train_loader,
        criterion,
        optimizer_cqcc,
        input_type='cqcc',
        device=device
    )

    torch.save(cqcc_baseline.state_dict(), os.path.join(models_dir, "cqcc_baseline.pth"))

    # ============================================================
    # 4 Custom Fusional Wav2Vec2 + CQCC with Cross-Attention + Graph
    # ============================================================

    print("\n--- Training Custom Fusion Detector ---")

    custom_model = ImprovedWav2Vec2CQCCDetector(num_classes=2).to(device)

    optimizer_custom = torch.optim.Adam(custom_model.parameters(), lr=1e-4)

    custom_loss = train_model(
        custom_model,
        train_loader,
        criterion,
        optimizer_custom,
        input_type='wav_and_cqcc',
        device=device
    )

    torch.save(custom_model.state_dict(), os.path.join(models_dir, "custom_hybrid.pth"))

    # ============================================================
    # Evaluation
    # ============================================================

    print("\n--- Evaluating Models ---")

    evals = []

    models_to_eval = [
        ("Wav2Vec2 Baseline", wav2vec_model, 'wav'),
        ("AASIST Baseline", aasist_model, 'mel'),
        ("CQCC Baseline", cqcc_baseline, 'cqcc'),
        ("Custom Fusion Model", custom_model, 'wav_and_cqcc')
    ]

    for name, model_obj, inp in models_to_eval:

        fpr, tpr, auc_val, eer, min_dcf = evaluate_model(
            model_obj,
            test_loader,
            input_type=inp,
            device=device
        )

        evals.append((name, fpr, tpr, auc_val, eer, min_dcf))

        print(f"{name} | AUC={auc_val:.3f} | EER={eer*100:.2f}% | minDCF={min_dcf:.4f}")

if __name__ == "__main__":
    main()

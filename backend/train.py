import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import AudioDataset, collate_variable_length
from models import (
    AASISTDetector,
    Wav2Vec2SpoofDetector,
    CQCCBaselineDetector,
    ImprovedWav2Vec2CQCCDetector,
    AblationWav2Vec2GraphDetector,
    AblationCQCCGraphDetector,
    AblationConcatGraphDetector,
    AblationCrossAttnLinearDetector
)
from sklearn.metrics import roc_curve, auc
import numpy as np
import random
from tqdm import tqdm


def train_model(model, train_dataloader, criterion, optimizer, epochs=5, input_type='wav', device=None, val_dataloader=None, eval_interval=1, patience=2, model_save_path=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    loss_history = []
    best_val_metric = float('inf') # For min_dcf, lower is better
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        # Wrap the dataloader with tqdm for a progress bar
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Training")):

            wavs, cqccs, labels = batch
            wavs = wavs.to(device)
            cqccs = cqccs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if input_type == 'wav':
                outputs = model(wavs)
            elif input_type == 'cqcc':
                outputs = model(cqccs)
            elif input_type == 'wav_and_cqcc':
                outputs = model(wavs, cqccs)
            else:
                raise ValueError("invalid input_type")

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print intermediate progress within the epoch
            if batch_idx % 500 == 0 and batch_idx > 0: # Report every 500 batches
                current_acc = 100 * correct / total
                current_loss = epoch_loss / (batch_idx + 1)
                print(f"  Batch {batch_idx}/{len(train_dataloader)} | Loss: {current_loss:.4f} | Acc: {current_acc:.2f}%")

        acc = 100 * correct / total if total > 0 else 0
        avg_loss = epoch_loss / len(train_dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {avg_loss:.4f} | Training Acc: {acc:.2f}%")

        # Validation and Early Stopping
        if val_dataloader is not None and (epoch + 1) % eval_interval == 0:
            print(f"Epoch {epoch+1}/{epochs} - Evaluating on Validation Set...")
            _, _, _, val_eer, val_min_dcf, val_accuracy = evaluate_model(
                model, val_dataloader, input_type=input_type, device=device
            )
            print(f"  Validation | EER={val_eer*100:.2f}% | minDCF={val_min_dcf:.4f} | Accuracy={val_accuracy:.2f}")

            if val_min_dcf < best_val_metric:
                best_val_metric = val_min_dcf
                patience_counter = 0
                best_epoch = epoch + 1
                if model_save_path:
                    torch.save(model.state_dict(), model_save_path)
                    print(f"  Saved best model to {model_save_path} (minDCF: {best_val_metric:.4f})")
            else:
                patience_counter += 1
                print(f"  Validation minDCF did not improve. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs. Best minDCF: {best_val_metric:.4f} at epoch {best_epoch}")
                    if model_save_path:
                        print(f"Loading best model from {model_save_path}")
                        model.load_state_dict(torch.load(model_save_path))
                    return loss_history # Stop training

    # ensure save path logic is intact even when loop ends naturally
    if val_dataloader is None and model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)
        print(f"  Saved final model to {model_save_path}")

    return loss_history


def evaluate_model(model, dataloader, input_type='wav', device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):

            wavs, cqccs, labels = batch
            wavs = wavs.to(device)
            cqccs = cqccs.to(device)
            labels = labels.to(device)

            if input_type == 'wav':
                outputs = model(wavs)
            elif input_type == 'cqcc':
                outputs = model(cqccs)
            elif input_type == 'wav_and_cqcc':
                outputs = model(wavs, cqccs)
            else:
                raise ValueError("invalid input_type")

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
    # Parameters according to ASVspoof 5 Evaluation Plan (Track 1)
    # ------------------
    P_spoof = 0.05      # Prior probability of a spoofing attack (\pi_{spf})
    P_bonafide = 0.95   # Prior probability of a real/bonafide utterance (1 - \pi_{spf})
    C_miss = 1          # Cost of falsely rejecting a real voice (Miss)
    C_fa = 10           # Cost of falsely accepting a spoof (False Alarm)
    
    # In the dataset, 0 = real (bonafide), 1 = fake (spoof)
    # fpr (False Positive Rate) = predicted fake (1) when true is real (0). This is a "miss" in ASVspoof.
    # fnr (False Negative Rate) = predicted real (0) when true is fake (1). This is a "false alarm" in ASVspoof.
    P_miss = fpr
    P_fa = fnr
    
    # Raw DCF = C_miss * P_bonafide * P_miss + C_fa * P_spoof * P_fa
    # Normalized by the default DCF (min cost of predicting all bonafide vs all spoof)
    dcf_default = min(C_miss * P_bonafide, C_fa * P_spoof)
    dcf_array = (C_miss * P_bonafide * P_miss + C_fa * P_spoof * P_fa) / dcf_default
    min_dcf = np.min(dcf_array)

    # Overall Accuracy (using 0.5 threshold)
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    correct = sum(1 for p, l in zip(preds, all_labels) if p == l)
    accuracy = correct / len(all_labels) if len(all_labels) > 0 else 0

    return fpr, tpr, roc_auc, eer, min_dcf, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Train spoof-detection models with optional CQCC caching.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to dataset root containing original/ and fake/ folders."
    )
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
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of English training data to reserve for validation."
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
    wavs, cqccs, labels = batch

    models_to_test = [
        ("Wav2Vec2 Baseline", Wav2Vec2SpoofDetector(num_classes=2).to(device), "wav"),
        ("AASIST Baseline", AASISTDetector(num_classes=2).to(device), "wav"),
        ("CQCC Baseline", CQCCBaselineDetector(num_classes=2).to(device), "cqcc"),
        ("Custom Fusion Model", ImprovedWav2Vec2CQCCDetector(num_classes=2).to(device), "wav_and_cqcc"),
        ("Ablation W2V2+Graph", AblationWav2Vec2GraphDetector(num_classes=2).to(device), "wav"),
        ("Ablation CQCC+Graph", AblationCQCCGraphDetector(num_classes=2).to(device), "cqcc"),
        ("Ablation Concat+Graph", AblationConcatGraphDetector(num_classes=2).to(device), "wav_and_cqcc"),
        ("Ablation CrossAttn+Linear", AblationCrossAttnLinearDetector(num_classes=2).to(device), "wav_and_cqcc"),
    ]

    with torch.no_grad():
        for name, model, input_type in models_to_test:
            model.eval()
            if input_type == "wav":
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

    print("Loading English Dataset for training/validation...")
    full_en_dataset = AudioDataset(data_dir=args.data_dir, augment=False, cqcc_cache_dir=args.cqcc_cache_dir, target_lang="en")
    total_en = len(full_en_dataset)
    if total_en == 0:
        raise ValueError("No English data found for target_lang='en'. Check data_dir and directory layout.")

    val_split = min(max(args.val_split, 0.0), 0.5)
    train_size = int((1.0 - val_split) * total_en)
    val_size = total_en - train_size
    indices = torch.randperm(total_en, generator=g).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(
        AudioDataset(data_dir=args.data_dir, augment=True, cqcc_cache_dir=args.cqcc_cache_dir, target_lang="en"),
        train_indices
    )
    val_dataset = torch.utils.data.Subset(
        AudioDataset(data_dir=args.data_dir, augment=False, cqcc_cache_dir=args.cqcc_cache_dir, target_lang="en"),
        val_indices
    )

    print("Loading German Dataset for Testing...")
    test_dataset = AudioDataset(data_dir=args.data_dir, augment=False, cqcc_cache_dir=args.cqcc_cache_dir, target_lang="de")
    
    if args.precompute_cqcc_only:
        print("\n--- Starting CQCC Precomputation ---")
        print(f"Dataset: {full_en_dataset.data_dir}")
        print("Precomputing CQCC cache for English data...")
        full_en_dataset.precompute_cqcc_cache(force=args.force_rebuild_cqcc)
        test_dataset.precompute_cqcc_cache(force=args.force_rebuild_cqcc)
        print("CQCC preprocessing complete. Exiting.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=2,
        pin_memory=True,
        generator=g, # ensure reproducible shuffling
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_variable_length,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
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
        device=device,
        val_dataloader=val_loader,
        model_save_path=os.path.join(models_dir, "wav2vec2.pth")
    )
    del wav2vec_model, optimizer_wav2vec
    torch.cuda.empty_cache()
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
        input_type='wav',
        device=device,
        val_dataloader=val_loader,
        model_save_path=os.path.join(models_dir, "aasist.pth")
    )
    del aasist_model, optimizer_aasist
    torch.cuda.empty_cache()
    # ============================================================
    # 3 CQCC Baseline
    # ============================================================

    print("\n--- Training CQCC Baseline ---")

    cqcc_baseline = CQCCBaselineDetector(num_classes=2).to(device)

    optimizer_cqcc = torch.optim.Adam(cqcc_baseline.parameters(), lr=1e-4)

    cqcc_loss = train_model(
        cqcc_baseline,
        train_loader,
        criterion,
        optimizer_cqcc,
        input_type='cqcc',
        device=device,
        val_dataloader=val_loader,
        model_save_path=os.path.join(models_dir, "cqcc_baseline.pth")
    )
    del cqcc_baseline, optimizer_cqcc
    torch.cuda.empty_cache()
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
        device=device,
        val_dataloader=val_loader,
        model_save_path=os.path.join(models_dir, "custom_hybrid.pth")
    )
    del custom_model, optimizer_custom
    torch.cuda.empty_cache()
    
    # ============================================================
    # 5 Ablation Models
    # ============================================================

    print("\n--- Training Ablation 1 (Wav2Vec2 + Graph) ---")
    ab1_model = AblationWav2Vec2GraphDetector(num_classes=2).to(device)
    optimizer_ab1 = torch.optim.Adam(ab1_model.parameters(), lr=1e-4) # learning rate for wav2vec2-based
    ab1_loss = train_model(ab1_model, train_loader, criterion, optimizer_ab1, input_type='wav', device=device, val_dataloader=val_loader, model_save_path=os.path.join(models_dir, "ablation_w2v2_graph.pth"))
    del ab1_model, optimizer_ab1
    torch.cuda.empty_cache()

    print("\n--- Training Ablation 2 (CQCC + Graph) ---")
    ab2_model = AblationCQCCGraphDetector(num_classes=2).to(device)
    optimizer_ab2 = torch.optim.Adam(ab2_model.parameters(), lr=1e-4) # learning rate for CQCC-based
    ab2_loss = train_model(ab2_model, train_loader, criterion, optimizer_ab2, input_type='cqcc', device=device, val_dataloader=val_loader, model_save_path=os.path.join(models_dir, "ablation_cqcc_graph.pth"))
    del ab2_model, optimizer_ab2
    torch.cuda.empty_cache()

    print("\n--- Training Ablation 3 (Wav2Vec2 + CQCC + Simple Concat) ---")
    ab3_model = AblationConcatGraphDetector(num_classes=2).to(device)
    optimizer_ab3 = torch.optim.Adam(ab3_model.parameters(), lr=1e-4)
    ab3_loss = train_model(ab3_model, train_loader, criterion, optimizer_ab3, input_type='wav_and_cqcc', device=device, val_dataloader=val_loader, model_save_path=os.path.join(models_dir, "ablation_concat_graph.pth"))
    del ab3_model, optimizer_ab3
    torch.cuda.empty_cache()

    print("\n--- Training Ablation 4 (Wav2Vec2 + CQCC + Cross-Attn + Linear) ---")
    ab4_model = AblationCrossAttnLinearDetector(num_classes=2).to(device)
    optimizer_ab4 = torch.optim.Adam(ab4_model.parameters(), lr=1e-4)
    ab4_loss = train_model(ab4_model, train_loader, criterion, optimizer_ab4, input_type='wav_and_cqcc', device=device, val_dataloader=val_loader, model_save_path=os.path.join(models_dir, "ablation_crossattn_linear.pth"))
    del ab4_model, optimizer_ab4
    torch.cuda.empty_cache()
    
    # ============================================================
    # Evaluation — reload one at a time
    # ============================================================
    print("\n--- Evaluating Models ---")
    evals = []

    models_to_eval = [
        ("Wav2Vec2 Baseline", Wav2Vec2SpoofDetector, "wav2vec2.pth", 'wav'),
        ("AASIST Baseline", AASISTDetector, "aasist.pth", 'wav'),
        ("CQCC Baseline", CQCCBaselineDetector, "cqcc_baseline.pth", 'cqcc'),
        ("Custom Fusion Model", ImprovedWav2Vec2CQCCDetector, "custom_hybrid.pth", 'wav_and_cqcc'),
        ("Ablation 1 (W2V2+Graph)", AblationWav2Vec2GraphDetector, "ablation_w2v2_graph.pth", 'wav'),
        ("Ablation 2 (CQCC+Graph)", AblationCQCCGraphDetector, "ablation_cqcc_graph.pth", 'cqcc'),
        ("Ablation 3 (Concat+Graph)", AblationConcatGraphDetector, "ablation_concat_graph.pth", 'wav_and_cqcc'),
        ("Ablation 4 (CrossAttn+Linear)", AblationCrossAttnLinearDetector, "ablation_crossattn_linear.pth", 'wav_and_cqcc'),
    ]

    for name, model_class, filename, inp in models_to_eval:
        model_path = os.path.join(models_dir, filename)
        if not os.path.exists(model_path):
            print(f"Skipping evaluation for {name} (Model weights not found at {model_path})")
            continue
            
        model_obj = model_class(num_classes=2).to(device)
        model_obj.load_state_dict(torch.load(model_path, map_location=device))
        model_obj.eval()

        print(f"\n--- Metrics for {name} ---")
        
        # 1. EVAL ON TRAIN SET
        train_fpr, train_tpr, train_auc, train_eer, train_min_dcf, train_acc = evaluate_model(
            model_obj, train_loader, input_type=inp, device=device
        )
        print(f"[Train] Acc={train_acc*100:.2f}% | EER={train_eer*100:.2f}% | minDCF={train_min_dcf:.4f}")

        # 2. EVAL ON TEST SET
        test_fpr, test_tpr, test_auc, test_eer, test_min_dcf, test_acc = evaluate_model(
            model_obj, test_loader, input_type=inp, device=device
        )
        print(f"[Test ] Acc={test_acc*100:.2f}% | EER={test_eer*100:.2f}% | minDCF={test_min_dcf:.4f}")

        del model_obj
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
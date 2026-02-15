"""
Testing Script for Model 4: Temporal Sliding Window

Loads a trained model and evaluates it on the saved test set.

Usage:
    python3 test_model.py --pred_min 90 --k 2 --N 5
    python3 test_model.py --pred_min 90 --k 2 --N 5 --summary_only
    python3 test_model.py --pred_min 90 --k 4 --fibonacci
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import pickle
import numpy as np
from sklearn import metrics

from gat_model_temporal import TemporalMatchPredictor, get_device
from train_model import (
    TemporalMatchDataset,
    temporal_collate_fn,
    batch_to_device,
)
from pass_network_creator import TOTAL_IN_GAME_FEATURES


def load_model(
    pred_min: int, k: int, N: int, hidden_channels: int,
    dropout: float, device: torch.device, use_fibonacci: bool = False,
) -> TemporalMatchPredictor:
    """Load a trained temporal model."""
    model = TemporalMatchPredictor(
        cumulative_input_size=7,
        interval_input_size=6,
        hidden_channels=hidden_channels,
        k=k,
        in_game_features=TOTAL_IN_GAME_FEATURES,
        num_classes=3,
        dropout=dropout,
    ).to(device)

    n_suffix = "Nfib" if use_fibonacci else f"N{N}"
    model_path = Path("models") / f"temporal_predictor_k{k}_{n_suffix}_{pred_min}min.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, n_suffix


def test_with_saved_data(
    pred_min: int, k: int, N: int, device: torch.device,
    batch_size: int = 64, show_details: bool = True,
    use_fibonacci: bool = False,
):
    """Test using saved test data and scalers."""

    n_suffix = "Nfib" if use_fibonacci else f"N{N}"
    
    # Load config
    config_path = Path("models") / f"config_k{k}_{n_suffix}_{pred_min}min.pkl"
    if config_path.exists():
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        hidden_channels = config.get('hidden_channels', 128)
        dropout = config.get('dropout', 0.5)
        print(f"Config loaded: hidden={hidden_channels}, dropout={dropout}")
    else:
        hidden_channels = 128
        dropout = 0.5
        print("Config not found, using defaults")

    # Load model
    model, n_suffix = load_model(pred_min, k, N, hidden_channels, dropout, device, use_fibonacci)
    print(f"Model loaded: temporal_predictor_k{k}_{n_suffix}_{pred_min}min.pt")

    # Load test data
    test_data_path = Path("models") / f"test_data_k{k}_{n_suffix}_{pred_min}min.pt"
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    test_samples = torch.load(test_data_path, weights_only=False)
    print(f"Test samples loaded: {len(test_samples)} matches")

    # Create DataLoader
    test_dataset = TemporalMatchDataset(test_samples)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=temporal_collate_fn,
    )

    # Run inference
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_dict in test_loader:
            batch_dict = batch_to_device(batch_dict, device)
            labels = batch_dict['labels']

            logits = model(batch_dict)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())

    result_names = ['Home Win', 'Away Win', 'Draw']

    # Individual predictions
    if show_details:
        print(f"\n{'='*70}")
        print(f"Individual Predictions (pred_min={pred_min}, k={k}, N={N})")
        print(f"{'='*70}")
        print(f"{'#':<4} {'Predicted':<12} {'Actual':<12} {'Correct':<8} {'Confidence':<10}")
        print("-" * 70)

        for i, (pred, label, probs) in enumerate(zip(all_preds, all_labels, all_probs)):
            is_correct = pred == label
            confidence = probs[pred] * 100
            correct_str = "Y" if is_correct else "N"
            print(f"{i+1:<4} {result_names[pred]:<12} {result_names[label]:<12} "
                  f"{correct_str:<8} {confidence:.1f}%")

    # Per-class accuracy
    print(f"\n{'='*70}")
    print(f"Per-Class Accuracy")
    print(f"{'='*70}")

    for cls_idx, cls_name in enumerate(result_names):
        cls_mask = [i for i, l in enumerate(all_labels) if l == cls_idx]
        if cls_mask:
            cls_correct = sum(1 for i in cls_mask if all_preds[i] == cls_idx)
            cls_acc = cls_correct / len(cls_mask) * 100
            print(f"{cls_name:<12}: {cls_correct}/{len(cls_mask)} = {cls_acc:.1f}%")
        else:
            print(f"{cls_name:<12}: No samples")

    # Prediction distribution
    print(f"\n{'='*70}")
    print(f"Prediction Distribution")
    print(f"{'='*70}")

    pred_counts = {0: 0, 1: 0, 2: 0}
    actual_counts = {0: 0, 1: 0, 2: 0}
    for p, l in zip(all_preds, all_labels):
        pred_counts[p] += 1
        actual_counts[l] += 1

    print(f"{'Class':<12} {'Predicted':<12} {'Actual':<12}")
    print("-" * 36)
    for cls_idx, cls_name in enumerate(result_names):
        print(f"{cls_name:<12} {pred_counts[cls_idx]:<12} {actual_counts[cls_idx]:<12}")

    # Overall metrics
    accuracy = metrics.accuracy_score(all_labels, all_preds)

    print(f"\n{'='*70}")
    print(f"Overall Results (pred_min={pred_min}, k={k}, N={n_suffix})")
    print(f"{'='*70}")
    print(f"Total Matches: {len(all_labels)}")
    print(f"Correct: {sum(1 for p, l in zip(all_preds, all_labels) if p == l)}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    print(f"\nClassification Report:")
    print(metrics.classification_report(
        all_labels, all_preds,
        target_names=result_names,
        zero_division=0,
    ))

    print(f"Confusion Matrix:")
    print(f"{'':<12} {'Pred Home':<12} {'Pred Away':<12} {'Pred Draw':<12}")
    cm = metrics.confusion_matrix(all_labels, all_preds)
    for i, row in enumerate(cm):
        print(f"{result_names[i]:<12} {row[0]:<12} {row[1]:<12} {row[2]:<12}")

    # ROC-AUC Score (multi-class)
    try:
        all_probs_array = np.array(all_probs)
        roc_auc_macro = metrics.roc_auc_score(
            all_labels, all_probs_array, multi_class='ovr', average='macro'
        )
        print(f"\nROC-AUC Score (macro): {roc_auc_macro:.4f}")
        
        # Per-class ROC-AUC
        print("Per-class ROC-AUC:")
        for cls_idx, cls_name in enumerate(result_names):
            binary_labels = [1 if l == cls_idx else 0 for l in all_labels]
            cls_probs = all_probs_array[:, cls_idx]
            cls_auc = metrics.roc_auc_score(binary_labels, cls_probs)
            print(f"  {cls_name}: {cls_auc:.4f}")
    except Exception as e:
        print(f"\nROC-AUC Score: Could not compute ({e})")

    return accuracy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Model 4: Temporal Sliding Window")
    parser.add_argument("--pred_min", type=int, default=90,
                        help="Prediction minute")
    parser.add_argument("--k", type=int, default=2,
                        help="Number of historical intervals")
    parser.add_argument("--N", type=int, default=5,
                        help="Interval width in minutes")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--use_cpu", action="store_true",
                        help="Force CPU usage")
    parser.add_argument("--summary_only", action="store_true",
                        help="Skip individual predictions")
    parser.add_argument("--fibonacci", action="store_true",
                        help="Test Fibonacci interval model (uses Nfib suffix)")

    args = parser.parse_args()

    device = torch.device("cpu") if args.use_cpu else get_device()
    print(f"Device: {device}")

    # Default k=4 for Fibonacci mode
    k = args.k
    if args.fibonacci and k == 2:
        k = 4
        print(f"Fibonacci mode: using k={k}")

    test_with_saved_data(
        pred_min=args.pred_min,
        k=k,
        N=args.N,
        device=device,
        batch_size=args.batch_size,
        show_details=not args.summary_only,
        use_fibonacci=args.fibonacci,
    )

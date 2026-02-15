"""
Test Model 4 (with Substitutions) on All Leagues

Tests the trained temporal sliding window model (with substitution tracking) on different leagues.
The model is trained on Premier League + Championship data and tested on:
- bundesliga
- la_liga
- ligue_1
- serie_a

This evaluates the model's generalization capability across different leagues.

Usage:
    python test_all_leagues.py --k 2 --N 5 --pred_min 90
    python test_all_leagues.py --k 2 --N 5 --pred_mins 45,60,75,90
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from typing import Dict, List, Tuple
import sys
import os

# Add current directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from gat_model_temporal import TemporalMatchPredictor, get_device
from train_model import (
    TemporalMatchDataset,
    temporal_collate_fn,
    batch_to_device,
    apply_scalers,
)
from pass_network_creator import (
    process_match,
    TOTAL_IN_GAME_FEATURES,
)

# Available leagues
LEAGUES = ['bundesliga', 'la_liga', 'ligue_1', 'serie_a']


def load_cache(cache_dir: Path, pred_min: int):
    """Load preprocessed data from cache files."""
    events_cache = torch.load(cache_dir / f"events_{pred_min}min.pt", weights_only=False)
    players_cache = torch.load(cache_dir / f"players_{pred_min}min.pt", weights_only=False)
    features_cache = torch.load(cache_dir / f"features_{pred_min}min.pt", weights_only=False)

    events_list = events_cache['events']
    passes_list = events_cache.get('passes', [None] * len(events_list))
    players_list = players_cache['players']
    positions_list = players_cache.get('positions', [None] * len(players_list))
    features_list = features_cache['features']
    labels = features_cache['labels']
    match_ids = events_cache['match_ids']

    return events_list, players_list, positions_list, features_list, passes_list, labels, match_ids


def process_league_matches(
    events_list, players_list, positions_list, features_list, passes_list, labels,
    pred_min: int, N: int, k: int,
) -> List[Dict]:
    """Process all matches in a league into temporal graph samples."""
    samples = []

    for i in range(len(labels)):
        try:
            # Get positions DataFrame (may be empty)
            positions = positions_list[i] if positions_list[i] is not None else pd.DataFrame()

            sample = process_match(
                events=events_list[i],
                players=players_list[i],
                positions=positions,
                features_df=features_list[i],
                label=labels[i],
                pred_min=pred_min,
                N=N,
                k=k,
            )

            if sample is not None:
                samples.append(sample)
        except Exception as e:
            continue

    return samples


def load_model_and_scalers(
    pred_min: int, k: int, N: int, device: torch.device
):
    """Load trained model, config, and scalers."""
    n_suffix = f"N{N}"
    models_dir = Path(__file__).parent / "models"

    # Load config
    config_path = models_dir / f"config_k{k}_{n_suffix}_{pred_min}min.pkl"
    if config_path.exists():
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        hidden_channels = config.get('hidden_channels', 128)
        dropout = config.get('dropout', 0.5)
    else:
        hidden_channels = 128
        dropout = 0.5

    # Load model
    model = TemporalMatchPredictor(
        cumulative_input_size=7,
        interval_input_size=6,
        hidden_channels=hidden_channels,
        k=k,
        in_game_features=TOTAL_IN_GAME_FEATURES,
        num_classes=3,
        dropout=dropout,
    ).to(device)

    model_path = models_dir / f"temporal_predictor_k{k}_{n_suffix}_{pred_min}min.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Load scalers
    scalers_path = models_dir / f"scalers_k{k}_{n_suffix}_{pred_min}min.pkl"
    if scalers_path.exists():
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
    else:
        scalers = None

    return model, scalers, n_suffix


def evaluate_on_league(
    model, samples: List[Dict], device: torch.device, batch_size: int = 32
) -> Tuple[float, float, Dict]:
    """Evaluate model on a set of samples, return accuracy, macro-f1, and detailed metrics."""
    if not samples:
        return 0.0, 0.0, {}

    dataset = TemporalMatchDataset(samples)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=temporal_collate_fn,
    )

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_dict in loader:
            batch_dict = batch_to_device(batch_dict, device)
            labels = batch_dict['labels']

            logits = model(batch_dict)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())

    accuracy = metrics.accuracy_score(all_labels, all_preds)
    macro_f1 = metrics.f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = metrics.f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Per-class metrics
    result_names = ['Home Win', 'Away Win', 'Draw']
    per_class_acc = {}
    per_class_f1 = metrics.f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_precision = metrics.precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = metrics.recall_score(all_labels, all_preds, average=None, zero_division=0)

    for cls_idx, cls_name in enumerate(result_names):
        cls_mask = [i for i, l in enumerate(all_labels) if l == cls_idx]
        if cls_mask:
            cls_correct = sum(1 for i in cls_mask if all_preds[i] == cls_idx)
            per_class_acc[cls_name] = cls_correct / len(cls_mask)
        else:
            per_class_acc[cls_name] = 0.0

    # ROC-AUC Score (multi-class)
    roc_auc_macro = None
    per_class_auc = {}
    try:
        all_probs_array = np.array(all_probs)
        roc_auc_macro = metrics.roc_auc_score(
            all_labels, all_probs_array, multi_class='ovr', average='macro'
        )
        # Per-class ROC-AUC
        for cls_idx, cls_name in enumerate(result_names):
            binary_labels = [1 if l == cls_idx else 0 for l in all_labels]
            cls_probs = all_probs_array[:, cls_idx]
            per_class_auc[cls_name] = metrics.roc_auc_score(binary_labels, cls_probs)
    except Exception:
        pass

    # Confusion matrix
    confusion_mat = metrics.confusion_matrix(all_labels, all_preds)

    detailed = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_acc': per_class_acc,
        'per_class_f1': {result_names[i]: per_class_f1[i] for i in range(len(result_names))},
        'per_class_precision': {result_names[i]: per_class_precision[i] for i in range(len(result_names))},
        'per_class_recall': {result_names[i]: per_class_recall[i] for i in range(len(result_names))},
        'roc_auc_macro': roc_auc_macro,
        'per_class_auc': per_class_auc,
        'confusion_matrix': confusion_mat,
        'n_samples': len(all_labels),
        'predictions': all_preds,
        'labels': all_labels,
        'probs': all_probs,
    }

    return accuracy, macro_f1, detailed


def test_all_leagues(
    pred_min: int, k: int, N: int, device: torch.device,
    batch_size: int = 32,
    leagues: List[str] = None,
    cache_base_dir: Path = None,
):
    """Test model on all available leagues."""
    print("=" * 70)
    print(f"Testing Model 4 (with Substitutions) on All Leagues")
    print(f"Parameters: pred_min={pred_min}, k={k}, N={N}")
    print("=" * 70)

    # Load model and scalers (trained on Premier League + Championship)
    print("\nLoading trained model and scalers...")
    try:
        model, scalers, n_suffix = load_model_and_scalers(pred_min, k, N, device)
        print(f"Model loaded: temporal_predictor_k{k}_{n_suffix}_{pred_min}min.pt")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if scalers is None:
        print("Warning: Scalers not found, features will not be normalized!")
    else:
        print(f"Scalers loaded: {list(scalers.keys())}")

    results = {}

    # Use provided leagues or all leagues
    test_leagues = leagues if leagues else LEAGUES

    # Cache is in model4_temporal_sliding_window/cache_other_leagues
    if cache_base_dir is None:
        script_dir = Path(__file__).parent
        cache_base_dir = script_dir.parent / "model4_temporal_sliding_window" / "cache_other_leagues"

    for league in test_leagues:
        cache_dir = cache_base_dir / league

        print(f"\n{'-'*70}")
        print(f"Testing on: {league.upper()}")
        print(f"{'-'*70}")

        if not cache_dir.exists():
            print(f"  Cache not found: {cache_dir}")
            results[league] = {'error': 'cache_not_found'}
            continue

        # Check if cache files exist
        if not (cache_dir / f"events_{pred_min}min.pt").exists():
            print(f"  Cache for pred_min={pred_min} not found in {cache_dir}")
            results[league] = {'error': 'cache_pred_min_not_found'}
            continue

        # Load cache
        print(f"  Loading cache from {cache_dir}...")
        try:
            events_list, players_list, positions_list, features_list, passes_list, labels, match_ids = \
                load_cache(cache_dir, pred_min)
            print(f"  Loaded {len(labels)} matches")
        except Exception as e:
            print(f"  Error loading cache: {e}")
            results[league] = {'error': str(e)}
            continue

        # Print label distribution
        label_counts = [labels.count(0), labels.count(1), labels.count(2)]
        print(f"  Distribution: Home={label_counts[0]}, Away={label_counts[1]}, Draw={label_counts[2]}")

        # Process matches into temporal graph samples
        print(f"  Processing matches into temporal graphs...")
        samples = process_league_matches(
            events_list, players_list, positions_list, features_list, passes_list, labels,
            pred_min, N, k,
        )
        print(f"  Created {len(samples)} valid samples")

        if not samples:
            print("  No valid samples created!")
            results[league] = {'error': 'no_valid_samples'}
            continue

        # Apply scalers
        if scalers:
            print("  Applying scalers...")
            apply_scalers(samples, scalers)

        # Evaluate
        print("  Evaluating...")
        accuracy, macro_f1, detailed = evaluate_on_league(model, samples, device, batch_size)

        results[league] = detailed

        # Per-class accuracy
        print(f"\n  {'='*70}")
        print(f"  Per-Class Accuracy")
        print(f"  {'='*70}")

        all_labels = detailed['labels']
        all_preds = detailed['predictions']
        result_names = ['Home Win', 'Away Win', 'Draw']

        for cls_idx, cls_name in enumerate(result_names):
            cls_mask = [i for i, l in enumerate(all_labels) if l == cls_idx]
            if cls_mask:
                cls_correct = sum(1 for i in cls_mask if all_preds[i] == cls_idx)
                cls_acc = cls_correct / len(cls_mask) * 100
                print(f"  {cls_name:<12}: {cls_correct}/{len(cls_mask)} = {cls_acc:.1f}%")
            else:
                print(f"  {cls_name:<12}: No samples")

        # Prediction distribution
        print(f"\n  {'='*70}")
        print(f"  Prediction Distribution")
        print(f"  {'='*70}")

        pred_counts = {0: 0, 1: 0, 2: 0}
        actual_counts = {0: 0, 1: 0, 2: 0}
        for p, l in zip(all_preds, all_labels):
            pred_counts[p] += 1
            actual_counts[l] += 1

        print(f"  {'Class':<12} {'Predicted':<12} {'Actual':<12}")
        print(f"  {'-'*36}")
        for cls_idx, cls_name in enumerate(result_names):
            print(f"  {cls_name:<12} {pred_counts[cls_idx]:<12} {actual_counts[cls_idx]:<12}")

        # Overall results
        print(f"\n  {'='*70}")
        print(f"  Overall Results (pred_min={pred_min}, k={k}, N={N})")
        print(f"  {'='*70}")
        print(f"  Total Matches: {len(all_labels)}")
        correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {accuracy*100:.2f}%")

        # Classification Report
        print(f"\n  Classification Report:")
        report = metrics.classification_report(
            all_labels, all_preds,
            target_names=result_names,
            zero_division=0,
        )
        for line in report.split('\n'):
            print(f"  {line}")

        # Confusion Matrix
        print(f"  Confusion Matrix:")
        print(f"  {'':12} {'Pred Home':<12} {'Pred Away':<12} {'Pred Draw':<12}")
        cm = detailed['confusion_matrix']
        for i, row in enumerate(cm):
            print(f"  {result_names[i]:<12} {row[0]:<12} {row[1]:<12} {row[2]:<12}")

        # ROC-AUC Score
        if detailed['roc_auc_macro']:
            print(f"\n  ROC-AUC Score (macro): {detailed['roc_auc_macro']:.4f}")
            print("  Per-class ROC-AUC:")
            for cls_name in result_names:
                auc = detailed['per_class_auc'].get(cls_name, 0)
                print(f"    {cls_name}: {auc:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY - All Leagues")
    print(f"{'='*70}")
    print(f"{'League':<15} {'Matches':<10} {'Accuracy':<12} {'Macro F1':<12} {'ROC-AUC':<10}")
    print("-" * 70)

    for league in LEAGUES:
        if league in results and 'error' not in results[league]:
            r = results[league]
            auc = r.get('roc_auc_macro', 0)
            auc_str = f"{auc:.4f}" if auc else "N/A"
            print(f"{league:<15} {r['n_samples']:<10} {r['accuracy']*100:>8.2f}%    {r['macro_f1']*100:>8.2f}%    {auc_str}")
        elif league in results:
            print(f"{league:<15} {'ERROR':<10} {results[league]['error']}")
        else:
            print(f"{league:<15} {'SKIPPED':<10}")

    # Average across leagues
    valid_results = [r for r in results.values() if 'accuracy' in r]
    if valid_results:
        avg_acc = np.mean([r['accuracy'] for r in valid_results])
        avg_f1 = np.mean([r['macro_f1'] for r in valid_results])
        avg_auc = np.mean([r.get('roc_auc_macro', 0) for r in valid_results if r.get('roc_auc_macro')])
        auc_str = f"{avg_auc:.4f}" if avg_auc else "N/A"
        print("-" * 70)
        print(f"{'AVERAGE':<15} {'':<10} {avg_acc*100:>8.2f}%    {avg_f1*100:>8.2f}%    {auc_str}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Model 4 (with Substitutions) on All Leagues")
    parser.add_argument("--pred_min", type=int, default=90, help="Single prediction minute")
    parser.add_argument("--pred_mins", type=str, default=None,
                        help="Comma-separated prediction minutes (overrides --pred_min)")
    parser.add_argument("--k", type=int, default=2, help="Number of historical intervals")
    parser.add_argument("--N", type=int, default=5, help="Interval width in minutes")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--leagues", type=str, default=None,
                        help="Comma-separated leagues to test (default: all). Options: bundesliga,la_liga,ligue_1,serie_a")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Base cache directory (default: ../model4_temporal_sliding_window/cache_other_leagues)")

    args = parser.parse_args()

    device = torch.device("cpu") if args.use_cpu else get_device()
    print(f"Device: {device}")

    # Determine prediction minutes
    if args.pred_mins:
        pred_mins = [int(x.strip()) for x in args.pred_mins.split(",")]
    else:
        pred_mins = [args.pred_min]

    print(f"Testing pred_mins: {pred_mins}")

    # Parse cache dir
    cache_base_dir = Path(args.cache_dir) if args.cache_dir else None

    all_results = {}
    for pred_min in pred_mins:
        print(f"\n{'#'*70}")
        print(f"# PRED_MIN = {pred_min}")
        print(f"{'#'*70}")

        # Parse leagues if specified
        leagues = None
        if args.leagues:
            leagues = [l.strip() for l in args.leagues.split(",")]
            print(f"Testing leagues: {leagues}")

        results = test_all_leagues(
            pred_min=pred_min,
            k=args.k,
            N=args.N,
            device=device,
            batch_size=args.batch_size,
            leagues=leagues,
            cache_base_dir=cache_base_dir,
        )
        all_results[pred_min] = results

    # Final summary across all pred_mins
    if len(pred_mins) > 1:
        print(f"\n{'='*70}")
        print("FINAL SUMMARY - All Pred Minutes")
        print(f"{'='*70}")

        for league in LEAGUES:
            print(f"\n{league.upper()}:")
            print(f"  {'pred_min':<10} {'Accuracy':<12} {'Macro F1':<12}")
            print("  " + "-" * 35)
            for pred_min in pred_mins:
                if league in all_results[pred_min] and 'accuracy' in all_results[pred_min][league]:
                    r = all_results[pred_min][league]
                    print(f"  {pred_min:<10} {r['accuracy']*100:>8.2f}%    {r['macro_f1']*100:>8.2f}%")


if __name__ == "__main__":
    main()

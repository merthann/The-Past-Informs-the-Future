"""
Training Script for Model 4: Temporal Sliding Window

Loads preprocessed data from cache (.pt files), creates temporal sliding window
graphs and features, trains the TemporalMatchPredictor model.

DATA LEAKAGE PREVENTION:
  1. Train/Val/Test split done FIRST (stratified)
  2. Scalers fit on TRAINING data ONLY
  3. Rating only in cumulative graph (not interval graphs)
  4. Interval features computed from interval-only events
  5. Interval avg_x/avg_y from events (not player_positions)

Usage:
    python3 train_model.py --cache_dir ../cache --pred_min 90 --k 2 --N 5
    python3 train_model.py --cache_dir ../cache --pred_min 90 (--k 4) --fibonacci (k = 4 default)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pickle
import time
import sys
import logging

from pass_network_creator import process_match, process_match_fibonacci, TOTAL_IN_GAME_FEATURES
from gat_model_temporal import TemporalMatchPredictor, get_device


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: str):
    """Setup dual logging to file and stdout."""
    logger = logging.getLogger('model4')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# Cache Loading
# ---------------------------------------------------------------------------

def load_cache(cache_dir: Path, pred_min: int):
    """Load preprocessed data from cache files.

    Returns:
        events_list, players_list, positions_list, features_list, labels, match_ids
    """
    events_path = cache_dir / f"events_{pred_min}min.pt"
    players_path = cache_dir / f"players_{pred_min}min.pt"
    features_path = cache_dir / f"features_{pred_min}min.pt"

    for p in [events_path, players_path, features_path]:
        if not p.exists():
            raise FileNotFoundError(f"Cache file not found: {p}")

    events_cache = torch.load(events_path, weights_only=False)
    players_cache = torch.load(players_path, weights_only=False)
    features_cache = torch.load(features_path, weights_only=False)

    return (
        events_cache['events'],
        players_cache['players'],
        players_cache['positions'],
        features_cache['features'],
        features_cache['labels'],
        features_cache['match_ids'],
    )


# ---------------------------------------------------------------------------
# Dataset and Collate
# ---------------------------------------------------------------------------

class TemporalMatchDataset(Dataset):
    """Dataset wrapping processed match samples (list of dicts)."""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def temporal_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for temporal match data.

    Batches:
      - Graph Data objects using Batch.from_data_list
      - Feature tensors using torch.cat
      - Labels using torch.tensor
    """
    k = len(batch[0]['home_interval_graphs'])

    return {
        'home_cum_graph': Batch.from_data_list([s['home_cum_graph'] for s in batch]),
        'away_cum_graph': Batch.from_data_list([s['away_cum_graph'] for s in batch]),
        'home_interval_graphs': [
            Batch.from_data_list([s['home_interval_graphs'][i] for s in batch])
            for i in range(k)
        ],
        'away_interval_graphs': [
            Batch.from_data_list([s['away_interval_graphs'][i] for s in batch])
            for i in range(k)
        ],
        'home_cum_features': torch.cat([s['home_cum_features'] for s in batch], dim=0),
        'away_cum_features': torch.cat([s['away_cum_features'] for s in batch], dim=0),
        'home_interval_features': [
            torch.cat([s['home_interval_features'][i] for s in batch], dim=0)
            for i in range(k)
        ],
        'away_interval_features': [
            torch.cat([s['away_interval_features'][i] for s in batch], dim=0)
            for i in range(k)
        ],
        'labels': torch.tensor([s['label'] for s in batch], dtype=torch.long),
    }


def batch_to_device(batch_dict: Dict, device: torch.device) -> Dict:
    """Move all tensors and graph batches to the specified device."""
    return {
        'home_cum_graph': batch_dict['home_cum_graph'].to(device),
        'away_cum_graph': batch_dict['away_cum_graph'].to(device),
        'home_interval_graphs': [g.to(device) for g in batch_dict['home_interval_graphs']],
        'away_interval_graphs': [g.to(device) for g in batch_dict['away_interval_graphs']],
        'home_cum_features': batch_dict['home_cum_features'].to(device),
        'away_cum_features': batch_dict['away_cum_features'].to(device),
        'home_interval_features': [f.to(device) for f in batch_dict['home_interval_features']],
        'away_interval_features': [f.to(device) for f in batch_dict['away_interval_features']],
        'labels': batch_dict['labels'].to(device),
    }


# ---------------------------------------------------------------------------
# Data Processing & Splitting
# ---------------------------------------------------------------------------

def process_all_matches(
    events_list, players_list, positions_list,
    features_list, labels, pred_min: int, N: int, k: int,
    logger: logging.Logger,
    use_fibonacci: bool = False,
) -> List[Dict]:
    """Process all matches into temporal graph samples.
    
    Args:
        use_fibonacci: If True, use Fibonacci interval widths (2,3,5,8,...)
                       instead of fixed N-minute intervals.
    """
    all_samples = []
    n_total = len(events_list)
    n_failed = 0

    desc = "Processing matches (Fibonacci)" if use_fibonacci else "Processing matches"
    
    for i in tqdm(range(n_total), desc=desc):
        if use_fibonacci:
            sample = process_match_fibonacci(
                events=events_list[i],
                players=players_list[i],
                positions=positions_list[i],
                features_df=features_list[i],
                label=labels[i],
                pred_min=pred_min,
                k=k,
            )
        else:
            sample = process_match(
                events=events_list[i],
                players=players_list[i],
                positions=positions_list[i],
                features_df=features_list[i],
                label=labels[i],
                pred_min=pred_min,
                N=N,
                k=k,
            )
        if sample is not None:
            all_samples.append(sample)
        else:
            n_failed += 1

    logger.info(f"Processed {len(all_samples)}/{n_total} matches ({n_failed} failed)")
    return all_samples


def split_data(
    samples: List[Dict],
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Stratified split into train/val/test sets.

    Split is done BEFORE any scaling to prevent data leakage.
    """
    labels = np.array([s['label'] for s in samples])
    indices = np.arange(len(samples))

    # First split: train+valid vs test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(sss1.split(indices, labels))

    # Second split: train vs valid
    train_val_labels = labels[train_val_idx]
    valid_size_adjusted = valid_size / (1 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=valid_size_adjusted, random_state=random_state)
    train_rel, valid_rel = next(sss2.split(train_val_idx, train_val_labels))

    train_idx = train_val_idx[train_rel]
    valid_idx = train_val_idx[valid_rel]

    train = [samples[i] for i in train_idx]
    valid = [samples[i] for i in valid_idx]
    test = [samples[i] for i in test_idx]

    return train, valid, test


# ---------------------------------------------------------------------------
# Scaling (fit on TRAINING data only)
# ---------------------------------------------------------------------------

def fit_scalers(train_samples: List[Dict], k: int) -> Dict[str, MinMaxScaler]:
    """Fit MinMaxScalers on TRAINING data only.

    Scalers:
      - cum_node: cumulative graph node features (7-dim)
      - int_node: interval graph node features (6-dim)
      - cum_ingame: cumulative in-game features (22-dim)
      - int_ingame: interval in-game features (22-dim)
    """
    cum_node_feats = []
    int_node_feats = []
    cum_ingame_feats = []
    int_ingame_feats = []

    for s in train_samples:
        # Cumulative node features
        cum_node_feats.append(s['home_cum_graph'].x.numpy())
        cum_node_feats.append(s['away_cum_graph'].x.numpy())

        # Interval node features
        for g in s['home_interval_graphs']:
            int_node_feats.append(g.x.numpy())
        for g in s['away_interval_graphs']:
            int_node_feats.append(g.x.numpy())

        # Cumulative in-game features
        cum_ingame_feats.append(s['home_cum_features'].numpy().flatten())
        cum_ingame_feats.append(s['away_cum_features'].numpy().flatten())

        # Interval in-game features
        for f in s['home_interval_features']:
            int_ingame_feats.append(f.numpy().flatten())
        for f in s['away_interval_features']:
            int_ingame_feats.append(f.numpy().flatten())

    scalers = {}
    scalers['cum_node'] = MinMaxScaler().fit(np.vstack(cum_node_feats))
    scalers['int_node'] = MinMaxScaler().fit(np.vstack(int_node_feats))
    scalers['cum_ingame'] = MinMaxScaler().fit(np.array(cum_ingame_feats))
    scalers['int_ingame'] = MinMaxScaler().fit(np.array(int_ingame_feats))

    return scalers


def apply_scalers(samples: List[Dict], scalers: Dict[str, MinMaxScaler]):
    """Apply pre-fitted scalers to a set of samples (in-place)."""
    cum_node = scalers['cum_node']
    int_node = scalers['int_node']
    cum_ig = scalers['cum_ingame']
    int_ig = scalers['int_ingame']

    for s in samples:
        # Scale cumulative node features
        s['home_cum_graph'].x = torch.tensor(
            cum_node.transform(s['home_cum_graph'].x.numpy()), dtype=torch.float)
        s['away_cum_graph'].x = torch.tensor(
            cum_node.transform(s['away_cum_graph'].x.numpy()), dtype=torch.float)

        # Scale interval node features
        for g in s['home_interval_graphs']:
            g.x = torch.tensor(int_node.transform(g.x.numpy()), dtype=torch.float)
        for g in s['away_interval_graphs']:
            g.x = torch.tensor(int_node.transform(g.x.numpy()), dtype=torch.float)

        # Scale cumulative in-game features
        s['home_cum_features'] = torch.tensor(
            cum_ig.transform(s['home_cum_features'].numpy()), dtype=torch.float)
        s['away_cum_features'] = torch.tensor(
            cum_ig.transform(s['away_cum_features'].numpy()), dtype=torch.float)

        # Scale interval in-game features
        for i in range(len(s['home_interval_features'])):
            s['home_interval_features'][i] = torch.tensor(
                int_ig.transform(s['home_interval_features'][i].numpy()), dtype=torch.float)
            s['away_interval_features'][i] = torch.tensor(
                int_ig.transform(s['away_interval_features'][i].numpy()), dtype=torch.float)


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def get_class_weights(labels: List[int], device: torch.device) -> torch.Tensor:
    """Compute inverse proportional class weights."""
    counts = {0: 0, 1: 0, 2: 0}
    for l in labels:
        counts[l] += 1

    total = len(labels)
    weights = torch.tensor([
        total / (3 * counts[0]) if counts[0] > 0 else 1.0,
        total / (3 * counts[1]) if counts[1] > 0 else 1.0,
        total / (3 * counts[2]) if counts[2] > 0 else 1.0,
    ], dtype=torch.float)

    return weights.to(device)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch, returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_dict in loader:
        batch_dict = batch_to_device(batch_dict, device)
        labels = batch_dict['labels']

        optimizer.zero_grad()
        output = model(batch_dict)
        loss = criterion(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model, returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_dict in loader:
            batch_dict = batch_to_device(batch_dict, device)
            labels = batch_dict['labels']

            output = model(batch_dict)
            loss = criterion(output, labels)

            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, accuracy


def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[np.ndarray]]:
    """Get predictions, labels, and probabilities from a loader."""
    model.eval()
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

    return all_preds, all_labels, all_probs


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------

def train_model(
    cache_dir: Path,
    pred_min: int = 90,
    N: int = 5,
    k: int = 2,
    hidden_channels: int = 128,
    num_epochs: int = 300,
    lr: float = 0.001,
    batch_size: int = 64,
    patience: int = 50,
    dropout: float = 0.5,
    use_cpu: bool = False,
    use_fibonacci: bool = False,
):
    """Main training function for Model 4.
    
    Args:
        use_fibonacci: If True, use Fibonacci interval widths (2,3,5,8,...)
                       instead of fixed N-minute intervals. N parameter is ignored.
    """
    t0 = time.time()

    # Setup - use 'fib' suffix for Fibonacci models
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    if use_fibonacci:
        log_file = logs_dir / f"model4_k{k}_Nfib_{pred_min}min.log"
        interval_str = "Fibonacci (2,3,5,8,...)"
    else:
        log_file = logs_dir / f"model4_k{k}_N{N}_{pred_min}min.log"
        interval_str = f"Fixed N={N}"
    
    logger = setup_logging(log_file)
    device = torch.device("cpu") if use_cpu else get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Config: pred_min={pred_min}, k={k}, intervals={interval_str}, hidden={hidden_channels}")
    logger.info(f"Training: epochs={num_epochs}, lr={lr}, batch={batch_size}, patience={patience}")

    # 1. Load cache
    logger.info("=" * 60)
    logger.info("[1/6] Loading cache...")
    events, players, positions, features, labels, match_ids = load_cache(cache_dir, pred_min)
    logger.info(f"Cache loaded: {len(labels)} matches")

    # 2. Process all matches
    logger.info("[2/6] Processing matches (temporal windows)...")
    all_samples = process_all_matches(
        events, players, positions, features, labels,
        pred_min=pred_min, N=N, k=k, logger=logger,
        use_fibonacci=use_fibonacci,
    )

    if not all_samples:
        logger.error("No matches processed!")
        return

    # Class distribution
    sample_labels = [s['label'] for s in all_samples]
    counts = {0: sample_labels.count(0), 1: sample_labels.count(1), 2: sample_labels.count(2)}
    total = len(sample_labels)
    logger.info(f"Class distribution: Home={counts[0]} ({counts[0]/total*100:.1f}%) "
                f"Away={counts[1]} ({counts[1]/total*100:.1f}%) "
                f"Draw={counts[2]} ({counts[2]/total*100:.1f}%)")

    # 3. SPLIT FIRST (before scaling!)
    logger.info("[3/6] Splitting data (stratified, BEFORE scaling)...")
    train_samples, valid_samples, test_samples = split_data(all_samples)
    logger.info(f"Split: Train={len(train_samples)} Val={len(valid_samples)} Test={len(test_samples)}")

    # 4. FIT SCALERS ON TRAINING DATA ONLY
    logger.info("[4/6] Fitting scalers on TRAINING data only...")
    scalers = fit_scalers(train_samples, k)
    logger.info("Scalers fitted: cum_node, int_node, cum_ingame, int_ingame")

    # Apply scalers to all sets
    apply_scalers(train_samples, scalers)
    apply_scalers(valid_samples, scalers)
    apply_scalers(test_samples, scalers)
    logger.info("Scalers applied to train/val/test sets")

    # 5. Create DataLoaders
    logger.info("[5/6] Creating DataLoaders...")
    train_dataset = TemporalMatchDataset(train_samples)
    valid_dataset = TemporalMatchDataset(valid_samples)
    test_dataset = TemporalMatchDataset(test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=temporal_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate_fn)

    # 6. Create Model
    logger.info("[6/6] Creating model...")
    model = TemporalMatchPredictor(
        cumulative_input_size=7,
        interval_input_size=6,
        hidden_channels=hidden_channels,
        k=k,
        in_game_features=TOTAL_IN_GAME_FEATURES,
        num_classes=3,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Training setup
    train_labels = [s['label'] for s in train_samples]
    class_weights = get_class_weights(train_labels, device)
    logger.info(f"Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # Training loop
    logger.info("=" * 60)
    logger.info("TRAINING START")
    logger.info("=" * 60)

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    min_loss_change = 0.001

    train_losses = []

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Evaluate
        _, train_acc = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)

        scheduler.step()

        # Early stopping (single increment per epoch to avoid premature stopping)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {key: val.cpu().clone() for key, val in model.state_dict().items()}
            patience_counter = 0
        else:
            # Check if loss has also converged (both conditions met = stronger signal)
            loss_converged = False
            if epoch > 10 and len(train_losses) >= 2 and train_losses[-2] > 0:
                loss_change = abs(train_losses[-1] / train_losses[-2] - 1)
                loss_converged = loss_change < min_loss_change
            patience_counter += 1

        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Get prediction distribution on validation set
            val_preds, _, _ = get_predictions(model, valid_loader, device)
            pred_dist = {0: val_preds.count(0), 1: val_preds.count(1), 2: val_preds.count(2)}

            logger.info(
                f"Epoch {epoch+1:4d} | Loss: {train_loss:.4f} | "
                f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
                f"Best: {best_val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                f"Pred: H={pred_dist[0]} A={pred_dist[1]} D={pred_dist[2]}"
            )

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    # Final evaluation
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Best Val Accuracy: {best_val_acc:.4f}")

    # Per-class accuracy
    test_preds, test_labels, _ = get_predictions(model, test_loader, device)
    result_names = ['Home Win', 'Away Win', 'Draw']
    for cls_idx, cls_name in enumerate(result_names):
        mask = [i for i, l in enumerate(test_labels) if l == cls_idx]
        if mask:
            cls_correct = sum(1 for i in mask if test_preds[i] == cls_idx)
            cls_acc = cls_correct / len(mask) * 100
            logger.info(f"  {cls_name}: {cls_correct}/{len(mask)} = {cls_acc:.1f}%")

    elapsed = time.time() - t0
    logger.info(f"Training time: {elapsed / 60:.2f} minutes")

    # Save model, scalers, test loader
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # File naming: use 'Nfib' for Fibonacci, 'N{value}' for fixed
    n_suffix = "Nfib" if use_fibonacci else f"N{N}"
    
    model_path = model_dir / f"temporal_predictor_k{k}_{n_suffix}_{pred_min}min.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    scaler_path = model_dir / f"scalers_k{k}_{n_suffix}_{pred_min}min.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)
    logger.info(f"Scalers saved to {scaler_path}")

    # Save test data for evaluation script
    test_data_path = model_dir / f"test_data_k{k}_{n_suffix}_{pred_min}min.pt"
    torch.save(test_samples, test_data_path)
    logger.info(f"Test data saved to {test_data_path}")

    # Save config for reproducibility
    config = {
        'pred_min': pred_min, 'N': N if not use_fibonacci else 'fibonacci', 'k': k,
        'hidden_channels': hidden_channels, 'dropout': dropout,
        'batch_size': batch_size, 'lr': lr,
        'best_val_acc': best_val_acc, 'test_acc': test_acc,
        'n_train': len(train_samples), 'n_valid': len(valid_samples), 'n_test': len(test_samples),
        'n_params': n_params,
        'use_fibonacci': use_fibonacci,
    }
    config_path = model_dir / f"config_k{k}_{n_suffix}_{pred_min}min.pkl"
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    logger.info(f"Config saved to {config_path}")

    return model, scalers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Model 4: Temporal Sliding Window")
    parser.add_argument("--cache_dir", type=str, default="../cache",
                        help="Cache directory with preprocessed .pt files")
    parser.add_argument("--pred_min", type=int, default=90,
                        help="Prediction minute (45, 60, 75, or 90)")
    parser.add_argument("--N", type=int, default=5,
                        help="Interval width in minutes")
    parser.add_argument("--k", type=int, default=2,
                        help="Number of historical intervals")
    parser.add_argument("--hidden", type=int, default=128,
                        help="GAT hidden channels")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate")
    parser.add_argument("--use_cpu", action="store_true",
                        help="Force CPU usage")
    parser.add_argument("--fibonacci", action="store_true",
                        help="Use Fibonacci interval widths (2,3,5,8,...) instead of fixed N")

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        print("Run universal_preprocess.py first to generate cache files.")
        sys.exit(1)

    # Default k=4 for Fibonacci mode
    k = args.k
    if args.fibonacci and k == 2:  # 2 is the default, use 4 for Fibonacci
        k = 4
        print(f"Fibonacci mode: using k={k} (default for Fibonacci)")

    train_model(
        cache_dir=cache_dir,
        pred_min=args.pred_min,
        N=args.N,
        k=k,
        hidden_channels=args.hidden,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        dropout=args.dropout,
        use_cpu=args.use_cpu,
        use_fibonacci=args.fibonacci,
    )

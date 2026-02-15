"""
Universal Data Preprocessing Script

Creates raw data cache that ALL models can use.
Run this ONCE, then each model extracts its own features.

Usage:
    python universal_preprocess.py --data_dir data/minute_90 --output_dir cache --pred_mins 45,60,75,90

Cache files created:
    - cache/events_45min.pt      (All events up to minute 45)
    - cache/players_45min.pt     (Player info for all matches)
    - cache/features_45min.pt    (Match features + labels)
"""

import torch
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import time
from joblib import Parallel, delayed
import traceback


def process_single_match(match_dir: Path, max_minute: int) -> Dict:
    """
    Load all data for a single match.

    Returns dict with events, players, features, passes or None if failed.
    """
    try:
        # Check required files
        events_path = match_dir / "events.csv"
        players_path = match_dir / "players.csv"
        features_path = match_dir / "features.csv"
        positions_path = match_dir / "player_positions.csv"
        passes_path = match_dir / "passes.csv"

        if not all(p.exists() for p in [events_path, players_path, features_path]):
            return None

        # Load events and filter by minute
        events = pd.read_csv(events_path)
        events = events[events['minute'] <= max_minute]

        # Load players
        players = pd.read_csv(players_path)

        # Load features (single row per match)
        features = pd.read_csv(features_path)

        # Load positions if exists
        if positions_path.exists():
            positions = pd.read_csv(positions_path)
        else:
            positions = pd.DataFrame()

        # Load passes if exists (needed for pass network graphs)
        if passes_path.exists():
            passes = pd.read_csv(passes_path)
            time_col = 'expandedMinute' if 'expandedMinute' in passes.columns else 'minute' if 'minute' in passes.columns else None
            if time_col:
                passes = passes[passes[time_col] <= max_minute]
        else:
            passes = pd.DataFrame()

        # Get match ID and label
        match_id = match_dir.name
        if not features.empty:
            result = str(features.iloc[0].get('result', '')).lower()
            if result == 'home_win':
                label = 0
            elif result == 'away_win':
                label = 1
            elif result == 'draw':
                label = 2
            else:
                # Fallback to scores
                home_score = features.iloc[0].get('home_score', 0)
                away_score = features.iloc[0].get('away_score', 0)
                if home_score > away_score:
                    label = 0
                elif away_score > home_score:
                    label = 1
                else:
                    label = 2
        else:
            label = -1

        return {
            'match_id': match_id,
            'events': events,
            'players': players,
            'features': features,
            'positions': positions,
            'passes': passes,
            'label': label
        }

    except Exception as e:
        return None


def process_all_matches(data_dir: Path, max_minute: int) -> Tuple[List, List, List, List, List, List, List]:
    """
    Process all matches in parallel.

    Returns:
        events_list, players_list, features_list, positions_list,
        passes_list, labels, match_ids
    """
    # Find match directories
    match_dirs = [d for d in data_dir.rglob('match_*') if d.is_dir()]
    if not match_dirs:
        match_dirs = [p.parent for p in data_dir.rglob('features.csv')]
    
    print(f"Found {len(match_dirs)} match directories")
    print("Starting parallel processing...")
    
    # Process in parallel
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(process_single_match)(m_dir, max_minute)
        for m_dir in match_dirs
    )
    
    # Filter valid results
    valid = [r for r in results if r is not None and r['label'] >= 0]
    
    events_list = [r['events'] for r in valid]
    players_list = [r['players'] for r in valid]
    features_list = [r['features'] for r in valid]
    positions_list = [r['positions'] for r in valid]
    passes_list = [r['passes'] for r in valid]
    labels = [r['label'] for r in valid]
    match_ids = [r['match_id'] for r in valid]

    print(f"\nSuccessfully processed {len(valid)} matches")
    return events_list, players_list, features_list, positions_list, passes_list, labels, match_ids


def save_cache(
    events_list: List[pd.DataFrame],
    players_list: List[pd.DataFrame],
    features_list: List[pd.DataFrame],
    positions_list: List[pd.DataFrame],
    passes_list: List[pd.DataFrame],
    labels: List[int],
    match_ids: List[str],
    output_dir: Path,
    pred_min: int
):
    """Save cached data for a specific prediction minute."""

    # Events cache (includes passes for pass network construction)
    events_path = output_dir / f"events_{pred_min}min.pt"
    torch.save({
        'events': events_list,  # List of DataFrames
        'passes': passes_list,  # List of DataFrames (pass events with receiver_id)
        'match_ids': match_ids,
        'pred_min': pred_min,
        'n_matches': len(events_list),
        'version': 'v2_with_passes'
    }, events_path)
    
    # Players cache
    players_path = output_dir / f"players_{pred_min}min.pt"
    torch.save({
        'players': players_list,
        'positions': positions_list,
        'match_ids': match_ids,
        'pred_min': pred_min,
        'n_matches': len(players_list),
        'version': 'v1_universal'
    }, players_path)
    
    # Features cache (includes labels)
    features_path = output_dir / f"features_{pred_min}min.pt"
    torch.save({
        'features': features_list,
        'labels': labels,
        'match_ids': match_ids,
        'pred_min': pred_min,
        'n_matches': len(features_list),
        'label_map': {0: 'home_win', 1: 'away_win', 2: 'draw'},
        'version': 'v1_universal'
    }, features_path)
    
    # Print sizes
    for path in [events_path, players_path, features_path]:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.name}: {size_mb:.1f} MB")


def truncate_events(events_list: List[pd.DataFrame], pred_min: int) -> List[pd.DataFrame]:
    """Truncate events to pred_min minutes."""
    truncated = []
    for events in events_list:
        truncated.append(events[events['minute'] <= pred_min].copy())
    return truncated


def truncate_passes(passes_list: List[pd.DataFrame], pred_min: int) -> List[pd.DataFrame]:
    """Truncate passes to pred_min minutes."""
    truncated = []
    for passes in passes_list:
        if passes.empty:
            truncated.append(passes)
            continue
        time_col = 'expandedMinute' if 'expandedMinute' in passes.columns else 'minute' if 'minute' in passes.columns else None
        if time_col:
            truncated.append(passes[passes[time_col] <= pred_min].copy())
        else:
            truncated.append(passes)
    return truncated


def main():
    parser = argparse.ArgumentParser(description="Universal Data Preprocessing")
    parser.add_argument("--data_dir", type=str, default="data/minute_90",
                        help="Root data directory (use minute_90 for full data)")
    parser.add_argument("--output_dir", type=str, default="cache",
                        help="Output cache directory")
    parser.add_argument("--pred_mins", type=str, default="45,60,75,90",
                        help="Comma-separated prediction minutes to cache")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    pred_mins = [int(x) for x in args.pred_mins.split(",")]
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Universal Data Preprocessing")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Prediction minutes: {pred_mins}")
    print()
    
    t0 = time.time()
    
    # Process all matches at max minute
    max_min = max(pred_mins)
    print(f"[1/2] Loading all data up to minute {max_min}...")
    events, players, features, positions, passes, labels, match_ids = process_all_matches(data_dir, max_min)

    if not labels:
        print("No matches processed!")
        return

    # Print label distribution
    print(f"\nLabel distribution:")
    print(f"  Home Win: {labels.count(0)} ({labels.count(0)/len(labels)*100:.1f}%)")
    print(f"  Away Win: {labels.count(1)} ({labels.count(1)/len(labels)*100:.1f}%)")
    print(f"  Draw:     {labels.count(2)} ({labels.count(2)/len(labels)*100:.1f}%)")

    # Save cache for each prediction minute
    print(f"\n[2/2] Saving cache files...")
    for pred_min in pred_mins:
        print(f"\nCreating cache for pred_min={pred_min}...")

        # Truncate events and passes to pred_min
        truncated_events = truncate_events(events, pred_min)
        truncated_passes = truncate_passes(passes, pred_min)

        save_cache(
            truncated_events, players, features, positions, truncated_passes,
            labels, match_ids, output_dir, pred_min
        )
    
    elapsed = time.time() - t0
    
    print(f"\n{'=' * 60}")
    print("Preprocessing Complete!")
    print(f"{'=' * 60}")
    print(f"Total time: {elapsed / 60:.2f} minutes")
    print(f"Matches processed: {len(labels)}")
    print(f"Cache files per minute: 3 (events, players, features)")
    print(f"Total cache files: {len(pred_mins) * 3}")


if __name__ == "__main__":
    main()

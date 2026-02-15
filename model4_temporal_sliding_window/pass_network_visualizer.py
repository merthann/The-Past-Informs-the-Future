"""
Pass Network Visualizer for Model 4: TSW-GAT

Visualizes aggregate pass networks on a football pitch for the Winning Team.
Combines Home Win predictions' home graphs + Away Win predictions' away graphs.

For each graph type (cumulative + k intervals), shows:
  - 11 position nodes arranged in formation layout
  - Pass edges with thickness/color proportional to average pass frequency
  - Position labels below each node

Usage:
    python pass_network_visualizer.py --pred_min 90 --k 2 --N 15
    python pass_network_visualizer.py --pred_min 90 --k 4 --fibonacci
"""

import torch
import numpy as np
import pickle
import argparse
import os
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from gat_model_temporal_withanalysis import TemporalMatchPredictor, get_device
from train_model import (
    TemporalMatchDataset,
    temporal_collate_fn,
    batch_to_device,
)
from pass_network_creator import POSITION_ENCODING, TOTAL_IN_GAME_FEATURES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REVERSE_POSITION = {v: k for k, v in POSITION_ENCODING.items()}

CLASS_NAMES = ['Home Win', 'Away Win', 'Draw']

# Map every original position to a tier for 4-4-2 assignment
POSITION_TO_TIER = {
    'GK': 'GK',
    'DL': 'DEF', 'DC': 'DEF', 'DR': 'DEF',
    'DML': 'MID', 'DMC': 'MID', 'DMR': 'MID',
    'ML': 'MID', 'MC': 'MID', 'MR': 'MID',
    'AML': 'ATT', 'AMC': 'ATT', 'AMR': 'ATT',
    'FW': 'ATT', 'Sub': 'MID',
}

# 4-4-2 slot names per tier (sorted left-to-right = high y to low y)
TIER_SLOTS = {
    'GK':  ['GK'],
    'DEF': ['LB', 'LCB', 'RCB', 'RB'],
    'MID': ['LM', 'LCM', 'RCM', 'RM'],
    'ATT': ['LST', 'RST'],
}

# 4-4-2 formation layout coordinates (100 x 68 pitch)
# Wing-backs/wingers slightly ahead of center players
SLOT_POS_XY = {
    'GK':  (5, 34),
    'LB':  (23, 56), 'LCB': (20, 42), 'RCB': (20, 26), 'RB':  (23, 12),
    'LM':  (46, 56), 'LCM': (43, 42), 'RCM': (43, 26), 'RM':  (46, 12),
    'LST': (78, 44), 'RST': (78, 24),
}


def draw_pitch(ax, pitch_color='#2e8b57', line_color='white'):
    """Draw a football pitch on the given axes (100 x 68 coordinate system)."""
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 73)
    ax.set_aspect('equal')
    ax.set_facecolor(pitch_color)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    lw = 1.5
    ax.plot([0, 100, 100, 0, 0], [0, 0, 68, 68, 0], color=line_color, lw=lw)
    ax.plot([50, 50], [0, 68], color=line_color, lw=lw)
    circle = plt.Circle((50, 34), 9.15, color=line_color, fill=False, lw=lw)
    ax.add_patch(circle)
    ax.plot(50, 34, 'o', color=line_color, ms=3)
    ax.plot([0, 16.5, 16.5, 0], [13.84, 13.84, 54.16, 54.16], color=line_color, lw=lw)
    ax.plot([100, 83.5, 83.5, 100], [13.84, 13.84, 54.16, 54.16], color=line_color, lw=lw)
    ax.plot([0, 5.5, 5.5, 0], [24.84, 24.84, 43.16, 43.16], color=line_color, lw=lw)
    ax.plot([100, 94.5, 94.5, 100], [24.84, 24.84, 43.16, 43.16], color=line_color, lw=lw)
    ax.plot(11, 34, 'o', color=line_color, ms=3)
    ax.plot(89, 34, 'o', color=line_color, ms=3)


# ---------------------------------------------------------------------------
# Get predictions for winning team selection
# ---------------------------------------------------------------------------

def get_predictions(model, loader, device):
    """Run model on test data and return per-sample predictions."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_dict in loader:
            batch_dict = batch_to_device(batch_dict, device)
            logits = model(batch_dict)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            for p in preds:
                predictions.append(int(p))
    return predictions


# ---------------------------------------------------------------------------
# Decode node positions from graph features
# ---------------------------------------------------------------------------

def assign_442_slots(graph, scaler, is_cumulative=True):
    """Assign each player to a 4-4-2 formation slot based on tier + y-coordinate.

    Steps:
      1. Decode original position -> tier (GK/DEF/MID/ATT)
      2. Within each tier, sort players by y-coordinate (high y = left side)
      3. Assign to fixed slot names: LB, LCB, RCB, RB, LM, LCM, RCM, RM, LST, RST

    Returns list of 4-4-2 slot labels, one per node (length = num_nodes).
    """
    x = graph.x.cpu().numpy()

    if scaler is not None:
        x_orig = scaler.inverse_transform(x)
    else:
        x_orig = x

    # Collect (node_idx, original_pos, tier, avg_y)
    # avg_y: cumulative = feature[6], interval = feature[5]
    y_idx = 6 if is_cumulative else 5
    players = []
    for i in range(x_orig.shape[0]):
        pos_code = int(round(x_orig[i, 0]))
        pos_code = max(1, min(15, pos_code))
        pos_label = REVERSE_POSITION.get(pos_code, 'Sub')
        tier = POSITION_TO_TIER.get(pos_label, 'MID')
        avg_y = float(x_orig[i, y_idx])
        players.append((i, pos_label, tier, avg_y))

    # Group by tier
    tier_groups = defaultdict(list)
    for node_idx, pos_label, tier, avg_y in players:
        tier_groups[tier].append((node_idx, avg_y))

    # Assign slots per tier (sort by y descending = left first)
    slot_labels = [''] * len(players)

    for tier, slots in TIER_SLOTS.items():
        group = tier_groups.get(tier, [])
        # Sort by y descending (high y = left side of pitch)
        group.sort(key=lambda x: x[1], reverse=True)

        n_slots = len(slots)
        n_players = len(group)

        if n_players == n_slots:
            # Perfect match
            for j, (node_idx, _) in enumerate(group):
                slot_labels[node_idx] = slots[j]
        elif n_players < n_slots:
            # Fewer players than slots: spread evenly
            # e.g., 3 defenders for 4 slots -> assign to LB, LCB, RB (skip RCB)
            if n_players == 0:
                continue
            step = n_slots / n_players
            for j, (node_idx, _) in enumerate(group):
                slot_idx = min(int(j * step), n_slots - 1)
                slot_labels[node_idx] = slots[slot_idx]
        else:
            # More players than slots: assign first n_slots, merge extras into nearest
            for j, (node_idx, _) in enumerate(group):
                slot_idx = min(j, n_slots - 1)
                slot_labels[node_idx] = slots[slot_idx]

    # Fallback for any unassigned
    for i in range(len(slot_labels)):
        if slot_labels[i] == '':
            slot_labels[i] = 'LCM'

    return slot_labels


# ---------------------------------------------------------------------------
# Build position-based edge map from a single graph
# ---------------------------------------------------------------------------

def build_position_edge_map(graph, positions):
    """Convert player-indexed edges to position-indexed edges.

    Returns dict: (pos_from, pos_to) -> list of edge weights
    """
    edge_index = graph.edge_index.cpu().numpy()  # [2, num_edges]
    edge_weight = graph.edge_weight.cpu().numpy()  # [num_edges]

    edge_map = defaultdict(list)
    for e in range(edge_index.shape[1]):
        src = edge_index[0, e]
        dst = edge_index[1, e]
        if src < len(positions) and dst < len(positions):
            pos_from = positions[src]
            pos_dst = positions[dst]
            edge_map[(pos_from, pos_dst)].append(float(edge_weight[e]))

    return edge_map


# ---------------------------------------------------------------------------
# Aggregate pass networks across winning team matches
# ---------------------------------------------------------------------------

def aggregate_pass_networks(test_samples, predictions, k, scalers):
    """Aggregate pass networks for winning team across all win-predicted matches.

    Winning Team = Home Win's home graphs + Away Win's away graphs.

    Returns dict with keys 'cumulative', 'interval_1', ..., 'interval_k'
    Each value is a dict: (pos_from, pos_to) -> mean edge weight
    Also returns position frequency count for node sizing.
    """
    cum_node_scaler = scalers.get('cum_node') if scalers else None
    int_node_scaler = scalers.get('int_node') if scalers else None

    # Collect edges per graph type
    graph_labels = ['cumulative'] + [f'interval_{i+1}' for i in range(k)]
    edge_collections = {label: defaultdict(list) for label in graph_labels}
    position_counts = defaultdict(int)
    n_matches = 0

    for idx, (sample, pred) in enumerate(zip(test_samples, predictions)):
        # Only win predictions
        if pred == 2:  # Draw, skip
            continue

        # Select winning team's graphs
        if pred == 0:  # Home Win → home graphs
            cum_graph = sample['home_cum_graph']
            int_graphs = sample['home_interval_graphs']
        else:  # Away Win → away graphs
            cum_graph = sample['away_cum_graph']
            int_graphs = sample['away_interval_graphs']

        n_matches += 1

        # Cumulative
        slots = assign_442_slots(cum_graph, cum_node_scaler, is_cumulative=True)
        edge_map = build_position_edge_map(cum_graph, slots)
        for key, weights in edge_map.items():
            edge_collections['cumulative'][key].extend(weights)
        for pos in slots:
            position_counts[pos] += 1

        # Intervals
        for i in range(k):
            if i < len(int_graphs):
                int_slots = assign_442_slots(int_graphs[i], int_node_scaler, is_cumulative=False)
                int_edge_map = build_position_edge_map(int_graphs[i], int_slots)
                for key, weights in int_edge_map.items():
                    edge_collections[f'interval_{i+1}'][key].extend(weights)

    # Average edge weights
    aggregated = {}
    for label in graph_labels:
        avg_edges = {}
        for (pos_from, pos_to), weights in edge_collections[label].items():
            avg_edges[(pos_from, pos_to)] = np.mean(weights)
        aggregated[label] = avg_edges

    print(f"  Winning team matches aggregated: {n_matches}")
    print(f"  Unique positions found: {sorted(position_counts.keys())}")

    return aggregated, position_counts, n_matches


# ---------------------------------------------------------------------------
# Extract GAT attention weights for winning team
# ---------------------------------------------------------------------------

def build_attention_edge_map(edge_index, attn_weights, slot_labels):
    """Map per-edge GAT attention to 4-4-2 slot pairs.

    Args:
        edge_index: [2, num_edges] tensor
        attn_weights: [num_edges, num_heads] tensor (last layer)
        slot_labels: list of slot names per node

    Returns dict: (slot_from, slot_to) -> list of mean-head attention values
    """
    ei = edge_index.cpu().numpy()
    # Average across attention heads
    attn = attn_weights.cpu().numpy().mean(axis=1)  # [num_edges]

    edge_map = defaultdict(list)
    for e in range(ei.shape[1]):
        src, dst = ei[0, e], ei[1, e]
        if src < len(slot_labels) and dst < len(slot_labels):
            key = (slot_labels[src], slot_labels[dst])
            edge_map[key].append(float(attn[e]))
    return edge_map


def build_combined_edge_map(attn_edge_index, attn_weights, orig_edge_index,
                            orig_edge_weight, slot_labels):
    """Map per-edge (attention x edge_weight) to 4-4-2 slot pairs.

    Skips self-loops and edges not in the original graph.

    Args:
        attn_edge_index: [2, num_edges] from GATConv (includes self-loops)
        attn_weights: [num_edges, num_heads] last-layer attention
        orig_edge_index: [2, E] original graph edges
        orig_edge_weight: [E] original edge weights (pass frequency)
        slot_labels: list of slot names per node

    Returns dict: (slot_from, slot_to) -> list of (attention * edge_weight) values
    """
    # Build lookup from original graph edges
    oei = orig_edge_index.cpu().numpy()
    oew = orig_edge_weight.cpu().numpy()
    weight_lookup = {}
    for e in range(oei.shape[1]):
        weight_lookup[(int(oei[0, e]), int(oei[1, e]))] = float(oew[e])

    ei = attn_edge_index.cpu().numpy()
    attn = attn_weights.cpu().numpy().mean(axis=1)  # [num_edges]

    edge_map = defaultdict(list)
    for e in range(ei.shape[1]):
        src, dst = int(ei[0, e]), int(ei[1, e])
        if src == dst:  # skip self-loops
            continue
        ew = weight_lookup.get((src, dst), 0.0)
        if ew <= 0:
            continue
        if src < len(slot_labels) and dst < len(slot_labels):
            combined = float(attn[e]) * ew
            key = (slot_labels[src], slot_labels[dst])
            edge_map[key].append(combined)
    return edge_map


def extract_attention_weights(model, test_samples, predictions, k, scalers, device):
    """Extract GAT last-layer attention weights for winning team matches.

    Processes one graph at a time (no batching) to avoid splitting complexity.

    Returns:
      attn_aggregated: {'cumulative': {(slot_from, slot_to): mean_attention}, ...}
      combined_aggregated: same structure but with attention x edge_weight per edge
    """
    model.eval()
    cum_node_scaler = scalers.get('cum_node') if scalers else None
    int_node_scaler = scalers.get('int_node') if scalers else None

    graph_labels = ['cumulative'] + [f'interval_{i+1}' for i in range(k)]
    attn_collections = {label: defaultdict(list) for label in graph_labels}
    combined_collections = {label: defaultdict(list) for label in graph_labels}
    n_matches = 0

    with torch.no_grad():
        for idx, (sample, pred) in enumerate(zip(test_samples, predictions)):
            if pred == 2:
                continue

            # Select winning team's graphs
            if pred == 0:
                cum_graph = sample['home_cum_graph']
                int_graphs = sample['home_interval_graphs']
            else:
                cum_graph = sample['away_cum_graph']
                int_graphs = sample['away_interval_graphs']

            n_matches += 1

            # --- Cumulative: extract attention ---
            x = cum_graph.x.to(device)
            ei = cum_graph.edge_index.to(device)
            _, attentions = model.cumulative_gat(x, ei, batch=None,
                                                 return_attention=True)
            # Use last layer (conv3) attention
            last_ei, last_attn = attentions[-1]  # (edge_index, [num_edges, heads])

            slots = assign_442_slots(cum_graph, cum_node_scaler, is_cumulative=True)
            attn_map = build_attention_edge_map(last_ei, last_attn, slots)
            for key, vals in attn_map.items():
                attn_collections['cumulative'][key].extend(vals)

            # Combined: attention x edge_weight (skip self-loops)
            comb_map = build_combined_edge_map(
                last_ei, last_attn,
                cum_graph.edge_index, cum_graph.edge_weight,
                slots)
            for key, vals in comb_map.items():
                combined_collections['cumulative'][key].extend(vals)

            # --- Intervals ---
            for i in range(k):
                if i < len(int_graphs):
                    ix = int_graphs[i].x.to(device)
                    iei = int_graphs[i].edge_index.to(device)
                    _, int_attns = model.interval_gat(ix, iei, batch=None,
                                                      return_attention=True)
                    last_iei, last_iattn = int_attns[-1]

                    int_slots = assign_442_slots(int_graphs[i], int_node_scaler,
                                                 is_cumulative=False)
                    iattn_map = build_attention_edge_map(last_iei, last_iattn,
                                                         int_slots)
                    for key, vals in iattn_map.items():
                        attn_collections[f'interval_{i+1}'][key].extend(vals)

                    # Combined: attention x edge_weight
                    icomb_map = build_combined_edge_map(
                        last_iei, last_iattn,
                        int_graphs[i].edge_index, int_graphs[i].edge_weight,
                        int_slots)
                    for key, vals in icomb_map.items():
                        combined_collections[f'interval_{i+1}'][key].extend(vals)

    # Average attention per slot pair
    attn_aggregated = {}
    combined_aggregated = {}
    for label in graph_labels:
        avg_attn = {}
        for (s_from, s_to), vals in attn_collections[label].items():
            avg_attn[(s_from, s_to)] = np.mean(vals)
        attn_aggregated[label] = avg_attn

        avg_comb = {}
        for (s_from, s_to), vals in combined_collections[label].items():
            avg_comb[(s_from, s_to)] = np.mean(vals)
        combined_aggregated[label] = avg_comb

    print(f"  Attention extracted from {n_matches} winning team matches")
    return attn_aggregated, combined_aggregated


# ---------------------------------------------------------------------------
# Handle duplicate positions in layout
# ---------------------------------------------------------------------------

def get_position_coordinates(position_counts):
    """Return coordinates for all 11 4-4-2 slots that appear in the data."""
    coords = {}
    for pos in position_counts:
        if pos in SLOT_POS_XY:
            coords[pos] = SLOT_POS_XY[pos]
        else:
            coords[pos] = (50, 34)
    return coords


# ---------------------------------------------------------------------------
# Draw a single pass network on pitch
# ---------------------------------------------------------------------------

def draw_pass_network(ax, edges, pos_coords, title, edge_threshold=0.02,
                      cbar_label='Avg Pass Frequency'):
    """Draw pass network on a pitch.

    Args:
        ax: matplotlib axes
        edges: dict (pos_from, pos_to) -> avg weight
        pos_coords: dict position -> (x, y)
        title: chart title
        edge_threshold: minimum weight to draw an edge
        cbar_label: label for the colorbar
    """
    draw_pitch(ax)

    if not edges:
        ax.set_title(title, fontsize=13, fontweight='bold', color='white')
        return

    # Filter edges
    filtered = {k: v for k, v in edges.items() if v >= edge_threshold
                and k[0] in pos_coords and k[1] in pos_coords}

    if not filtered:
        ax.set_title(title, fontsize=13, fontweight='bold', color='white')
        return

    max_weight = max(filtered.values())
    min_weight = min(filtered.values())
    weight_range = max_weight - min_weight if max_weight > min_weight else 1.0

    # Colormap for edges
    cmap = plt.cm.YlOrRd
    norm = Normalize(vmin=min_weight, vmax=max_weight)

    # Draw edges
    for (pos_from, pos_to), weight in sorted(filtered.items(), key=lambda x: x[1]):
        x1, y1 = pos_coords[pos_from]
        x2, y2 = pos_coords[pos_to]

        # Line thickness: 0.5 to 6
        thickness = 0.5 + 5.5 * (weight - min_weight) / weight_range

        # Color from colormap
        color = cmap(norm(weight))

        # Slight curve for directed edges (offset midpoint)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Perpendicular offset for curve
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            perp_x = -dy / length * 2.0  # small offset
            perp_y = dx / length * 2.0
        else:
            perp_x, perp_y = 0, 0

        # Draw curved line using quadratic bezier (3-point path)
        ctrl_x = mid_x + perp_x
        ctrl_y = mid_y + perp_y

        # Use simple line segments approximating a curve
        t_vals = np.linspace(0, 1, 20)
        curve_x = (1-t_vals)**2 * x1 + 2*(1-t_vals)*t_vals * ctrl_x + t_vals**2 * x2
        curve_y = (1-t_vals)**2 * y1 + 2*(1-t_vals)*t_vals * ctrl_y + t_vals**2 * y2

        ax.plot(curve_x, curve_y, color=color, linewidth=thickness,
                alpha=0.85, solid_capstyle='round', zorder=2)

        # Small arrow at the end
        arrow_t = 0.85
        ax_start = ((1-arrow_t)**2 * x1 + 2*(1-arrow_t)*arrow_t * ctrl_x
                     + arrow_t**2 * x2)
        ay_start = ((1-arrow_t)**2 * y1 + 2*(1-arrow_t)*arrow_t * ctrl_y
                     + arrow_t**2 * y2)
        ax.annotate('', xy=(x2, y2), xytext=(ax_start, ay_start),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=max(1.5, thickness * 0.7),
                                    mutation_scale=10 + thickness * 2),
                    zorder=3)

    # Draw nodes
    for pos, (x, y) in pos_coords.items():
        # White circle
        circle = plt.Circle((x, y), 2.8, facecolor='white',
                            edgecolor='#333333', linewidth=1.5, zorder=5)
        ax.add_patch(circle)

        # Position label inside circle
        ax.text(x, y, pos, ha='center', va='center',
                fontsize=7, fontweight='bold', color='#333333', zorder=6)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(title, fontsize=13, fontweight='bold', color='white', pad=10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pass Network Visualizer for TSW-GAT")
    parser.add_argument("--pred_min", type=int, default=90)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--N", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--fibonacci", action="store_true")
    parser.add_argument("--edge_threshold", type=float, default=0.02,
                        help="Minimum avg edge weight to display")
    args = parser.parse_args()

    device = get_device() if not args.use_cpu else torch.device('cpu')
    print(f"Device: {device}")

    k = args.k
    n_suffix = f"N{args.N}" if not args.fibonacci else "Nfib"

    output_dir = Path("explainability_results")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # --- Load config ---
    config_path = Path("models") / f"config_k{k}_{n_suffix}_{args.pred_min}min.pkl"
    if config_path.exists():
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        hidden_channels = config.get('hidden_channels', 128)
        dropout = config.get('dropout', 0.5)
        print(f"Config: hidden={hidden_channels}, dropout={dropout}")
    else:
        hidden_channels = 128
        dropout = 0.5
        print("Config not found, using defaults")

    # --- Load model ---
    model = TemporalMatchPredictor(
        cumulative_input_size=7,
        interval_input_size=6,
        hidden_channels=hidden_channels,
        k=k,
        in_game_features=TOTAL_IN_GAME_FEATURES,
        num_classes=3,
        dropout=dropout,
    ).to(device)

    model_path = Path("models") / f"temporal_predictor_k{k}_{n_suffix}_{args.pred_min}min.pt"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded: {model_path}")

    # --- Load test data ---
    test_data_path = Path("models") / f"test_data_k{k}_{n_suffix}_{args.pred_min}min.pt"
    if not test_data_path.exists():
        print(f"Test data not found: {test_data_path}")
        return
    test_samples = torch.load(test_data_path, weights_only=False)
    print(f"Test samples: {len(test_samples)}")

    # --- Load scalers ---
    scaler_path = Path("models") / f"scalers_k{k}_{n_suffix}_{args.pred_min}min.pkl"
    scalers = {}
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        print(f"Scalers loaded: {scaler_path}")

    # --- Get predictions ---
    test_dataset = TemporalMatchDataset(test_samples)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=temporal_collate_fn,
    )
    predictions = get_predictions(model, test_loader, device)
    print(f"Predictions: {len(predictions)} "
          f"(Home Win: {predictions.count(0)}, "
          f"Away Win: {predictions.count(1)}, "
          f"Draw: {predictions.count(2)})")

    # --- Aggregate pass networks ---
    print("\nAggregating pass networks for Winning Team...")
    aggregated, pos_counts, n_matches = aggregate_pass_networks(
        test_samples, predictions, k, scalers)

    # Get positions that actually appear
    pos_coords = get_position_coordinates(pos_counts)
    print(f"  Positions on pitch: {list(pos_coords.keys())}")

    # --- Compute interval time ranges for titles ---
    if not args.fibonacci:
        N = args.N
        intervals = []
        for i in range(k):
            start = args.pred_min - (k - i) * N
            end = start + N
            intervals.append((start, end))
    else:
        fib = [2, 3, 5, 8, 13, 21, 34, 55][:k]
        fib.reverse()
        intervals = []
        cursor = args.pred_min
        for width in reversed(fib):
            intervals.append((cursor - width, cursor))
            cursor -= width
        intervals.reverse()

    # --- Generate visualizations ---
    print("\nGenerating pass network visualizations...")

    # Cumulative
    fig, ax = plt.subplots(figsize=(14, 9))
    title = f"Winning Team — Cumulative (0-{args.pred_min}')\n{n_matches} matches aggregated"
    draw_pass_network(ax, aggregated['cumulative'], pos_coords, title,
                      edge_threshold=args.edge_threshold)
    plt.tight_layout()
    out_path = output_dir / 'pass_network_winning_cumulative.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # Intervals
    for i in range(k):
        label = f'interval_{i+1}'
        start, end = intervals[i]
        fig, ax = plt.subplots(figsize=(14, 9))
        title = (f"Winning Team — Interval {i+1} ({start}-{end}')\n"
                 f"{n_matches} matches aggregated")
        draw_pass_network(ax, aggregated[label], pos_coords, title,
                          edge_threshold=args.edge_threshold)
        plt.tight_layout()
        out_path = output_dir / f'pass_network_winning_interval_{i+1}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")

    # --- Extract GAT attention weights ---
    print("\nExtracting GAT attention weights for Winning Team...")
    attn_aggregated, combined_aggregated = extract_attention_weights(
        model, test_samples, predictions, k, scalers, device)

    # --- Generate attention visualizations ---
    print("\nGenerating attention-weighted pass network visualizations...")

    # Cumulative attention
    fig, ax = plt.subplots(figsize=(14, 9))
    title = (f"Winning Team — Model Attention (Cumulative 0-{args.pred_min}')\n"
             f"{n_matches} matches, GAT last-layer attention")
    draw_pass_network(ax, attn_aggregated['cumulative'], pos_coords, title,
                      edge_threshold=0.0, cbar_label='GAT Attention')
    plt.tight_layout()
    out_path = output_dir / 'pass_attention_winning_cumulative.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # Interval attention
    for i in range(k):
        label = f'interval_{i+1}'
        start, end = intervals[i]
        fig, ax = plt.subplots(figsize=(14, 9))
        title = (f"Winning Team — Model Attention (Interval {i+1}: {start}-{end}')\n"
                 f"{n_matches} matches, GAT last-layer attention")
        draw_pass_network(ax, attn_aggregated[label], pos_coords, title,
                          edge_threshold=0.0, cbar_label='GAT Attention')
        plt.tight_layout()
        out_path = output_dir / f'pass_attention_winning_interval_{i+1}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")

    # --- Generate combined importance visualizations (attention x edge_weight) ---
    print("\nGenerating combined importance visualizations (attention x pass freq)...")

    # Cumulative combined
    fig, ax = plt.subplots(figsize=(14, 9))
    title = (f"Winning Team — Combined Importance (Cumulative 0-{args.pred_min}')\n"
             f"{n_matches} matches, attention x pass frequency")
    draw_pass_network(ax, combined_aggregated['cumulative'], pos_coords, title,
                      edge_threshold=0.0, cbar_label='Attention x Pass Freq')
    plt.tight_layout()
    out_path = output_dir / 'pass_combined_winning_cumulative.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # Interval combined
    for i in range(k):
        label = f'interval_{i+1}'
        start, end = intervals[i]
        fig, ax = plt.subplots(figsize=(14, 9))
        title = (f"Winning Team — Combined Importance (Interval {i+1}: {start}-{end}')\n"
                 f"{n_matches} matches, attention x pass frequency")
        draw_pass_network(ax, combined_aggregated[label], pos_coords, title,
                          edge_threshold=0.0, cbar_label='Attention x Pass Freq')
        plt.tight_layout()
        out_path = output_dir / f'pass_combined_winning_interval_{i+1}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")

    # --- Summary: top pass connections ---
    print(f"\n{'='*60}")
    print("Top 15 Pass Connections by Frequency (Cumulative)")
    print(f"{'='*60}")
    sorted_edges = sorted(aggregated['cumulative'].items(),
                          key=lambda x: x[1], reverse=True)
    for (pos_from, pos_to), weight in sorted_edges[:15]:
        print(f"  {pos_from:>4} -> {pos_to:<4}  avg_weight={weight:.4f}")

    print(f"\n{'='*60}")
    print("Top 15 Pass Connections by Model Attention (Cumulative)")
    print(f"{'='*60}")
    sorted_attn = sorted(attn_aggregated['cumulative'].items(),
                         key=lambda x: x[1], reverse=True)
    for (pos_from, pos_to), weight in sorted_attn[:15]:
        print(f"  {pos_from:>4} -> {pos_to:<4}  attention={weight:.4f}")

    print(f"\n{'='*60}")
    print("Top 15 Pass Connections by Combined Importance (Cumulative)")
    print(f"{'='*60}")
    sorted_comb = sorted(combined_aggregated['cumulative'].items(),
                         key=lambda x: x[1], reverse=True)
    for (pos_from, pos_to), weight in sorted_comb[:15]:
        print(f"  {pos_from:>4} -> {pos_to:<4}  combined={weight:.6f}")

    for i in range(k):
        label = f'interval_{i+1}'
        start, end = intervals[i]
        print(f"\nTop 10 by Combined Importance (Interval {i+1}: {start}-{end}')")
        print(f"{'-'*40}")
        sorted_comb = sorted(combined_aggregated[label].items(),
                             key=lambda x: x[1], reverse=True)
        for (pos_from, pos_to), weight in sorted_comb[:10]:
            print(f"  {pos_from:>4} -> {pos_to:<4}  combined={weight:.6f}")

    print(f"\n{'='*60}")
    print("PASS NETWORK VISUALIZATION COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

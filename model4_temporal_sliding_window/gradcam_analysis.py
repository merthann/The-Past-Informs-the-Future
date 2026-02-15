"""
Grad-CAM Explainability Analysis for Model 4: TSW-GAT

Aggregate feature attribution: "When the model predicts Home Win, which features are influential?"
Computes Grad-CAM importance of 22 in-game features for each class (Home/Away/Draw).

Analyses:
  A) Per-Class Feature Attribution: "When the model predicts Home Win, which features
     drive that prediction?" — 22 feature ranking + avg values
  B) Correct vs Wrong: Feature importance difference in correct vs incorrect predictions
  C) Temporal Window Importance: Which time window is most influential (Cumulative vs Interval)
  D) Pitch Visualization: Player importance on the football pitch via Grad-CAM

Reference: pytorch-grad-cam (https://github.com/jacobgil/pytorch-grad-cam)

Usage:
    python gradcam_analysis.py --pred_min 90 --k 2 --N 5
    python gradcam_analysis.py --pred_min 90 --k 4 --fibonacci
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import argparse
import os
from pathlib import Path
from torch.utils.data import DataLoader
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch, FancyBboxPatch, Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from gat_model_temporal_withanalysis import TemporalMatchPredictor, GATEncoder, get_device
from train_model import (
    TemporalMatchDataset,
    temporal_collate_fn,
    batch_to_device,
)
from pass_network_creator import (
    TOTAL_IN_GAME_FEATURES,
    IN_GAME_FEATURE_COLS,
    EVENT_FEATURE_COLS,
    POSITION_ENCODING,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_IN_GAME_NAMES = IN_GAME_FEATURE_COLS + EVENT_FEATURE_COLS  # 22 names

CUMULATIVE_NODE_FEATURES = [
    'position', 'height', 'weight', 'rating', 'pass_accuracy', 'avg_x', 'avg_y'
]
INTERVAL_NODE_FEATURES = [
    'position', 'height', 'weight', 'pass_accuracy', 'avg_x', 'avg_y'
]

REVERSE_POSITION = {v: k for k, v in POSITION_ENCODING.items()}

CLASS_NAMES = ['Home Win', 'Away Win', 'Draw']
CLASS_COLORS = ['#4CAF50', '#F44336', '#FF9800']


# ---------------------------------------------------------------------------
# Hook to capture fusion layer input
# ---------------------------------------------------------------------------

class FusionHook:
    """Captures the concatenated vector entering the fusion layer."""

    def __init__(self, module: nn.Module):
        self.input_tensor = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.input_tensor = input[0]

    def remove(self):
        self._handle.remove()


# ---------------------------------------------------------------------------
# Core: compute per-sample Grad-CAM + extract feature values
# ---------------------------------------------------------------------------

def compute_all_match_attributions(model, loader, device, k, scalers=None):
    """For EVERY match in the test set, compute:
      - predicted class + confidence
      - actual label
      - Grad-CAM importance for each of the 22 in-game features (per window)
      - actual feature values (inverse-scaled back to original)
      - node-feature Grad-CAM from GAT layers

    Returns list of dicts, one per match.
    """
    model.eval()
    hidden = model.hidden_channels
    graph_dim = hidden * (1 + k) * 2

    # Build window/feature name mapping
    window_labels = ['Cumulative']
    for i in range(k):
        window_labels.append(f'Interval_{i+1}')

    side_labels = ['Home', 'Away']

    hook = FusionHook(model.fusion)

    # In-game feature scalers for inverse transform
    cum_ig_scaler = scalers.get('cum_ingame') if scalers else None
    int_ig_scaler = scalers.get('int_ingame') if scalers else None

    results = []

    for batch_dict in loader:
        batch_dict = batch_to_device(batch_dict, device)
        labels = batch_dict['labels']
        batch_size = labels.size(0)

        # Full forward to get predictions + fill hook
        with torch.no_grad():
            logits_ng = model(batch_dict)
            probs_ng = torch.softmax(logits_ng, dim=1).cpu().numpy()

        # Now with gradients for Grad-CAM
        logits = model(batch_dict)

        # Save full batch fusion input BEFORE per-sample loop
        # (per-sample model.fusion() calls will overwrite hook.input_tensor)
        batch_fusion_input = hook.input_tensor.detach().clone()

        for sample_idx in range(batch_size):
            pred_class = int(probs_ng[sample_idx].argmax())
            confidence = float(probs_ng[sample_idx].max())
            true_label = int(labels[sample_idx].item())
            all_probs = probs_ng[sample_idx].tolist()

            # --- Grad-CAM on fusion input ---
            fusion_input = batch_fusion_input[sample_idx].unsqueeze(0)
            fusion_input.requires_grad_(True)

            x = model.dropout_layer(model.elu(model.fusion(fusion_input)))
            out = model.classifier(x)

            model.zero_grad()
            out[0, pred_class].backward(retain_graph=True)

            if fusion_input.grad is None:
                continue

            grad = fusion_input.grad[0].detach().cpu().numpy()
            act = fusion_input[0].detach().cpu().numpy()
            gradcam_full = np.maximum(grad * act, 0)
            gradcam_signed_full = grad * act  # signed: + supports, - opposes

            # --- Extract in-game feature values + Grad-CAM scores ---
            ingame_vector = act[graph_dim:]         # actual (scaled) values
            ingame_gradcam = gradcam_full[graph_dim:]  # Grad-CAM scores (absolute)
            ingame_signed = gradcam_signed_full[graph_dim:]  # signed attribution
            ingame_grad = grad[graph_dim:]  # raw gradient (direction only)

            # Rebuild per-feature attribution: (window, side, feature_name, value, gradcam)
            feature_attributions = []
            idx = 0
            for w_idx, window in enumerate(['cum'] + [f'int{i+1}' for i in range(k)]):
                for side in ['home', 'away']:
                    vals_scaled = ingame_vector[idx:idx + TOTAL_IN_GAME_FEATURES]
                    gcam = ingame_gradcam[idx:idx + TOTAL_IN_GAME_FEATURES]
                    signed = ingame_signed[idx:idx + TOTAL_IN_GAME_FEATURES]
                    raw_grad = ingame_grad[idx:idx + TOTAL_IN_GAME_FEATURES]

                    # Inverse-scale to get original feature values
                    if window == 'cum' and cum_ig_scaler is not None:
                        vals_orig = cum_ig_scaler.inverse_transform(
                            vals_scaled.reshape(1, -1))[0]
                    elif window != 'cum' and int_ig_scaler is not None:
                        vals_orig = int_ig_scaler.inverse_transform(
                            vals_scaled.reshape(1, -1))[0]
                    else:
                        vals_orig = vals_scaled

                    window_label = 'Cumulative' if window == 'cum' else f'Interval_{window[3:]}'
                    side_label = 'Home' if side == 'home' else 'Away'

                    for f_idx, fname in enumerate(ALL_IN_GAME_NAMES):
                        feature_attributions.append({
                            'window': window_label,
                            'side': side_label,
                            'feature': fname,
                            'value': float(vals_orig[f_idx]),
                            'value_scaled': float(vals_scaled[f_idx]),
                            'gradcam': float(gcam[f_idx]),
                            'signed': float(signed[f_idx]),
                            'gradient': float(raw_grad[f_idx]),
                        })

                    idx += TOTAL_IN_GAME_FEATURES

            # --- Graph embedding Grad-CAM (aggregate per window/side) ---
            graph_attributions = {}
            # Cumulative
            graph_attributions['Home_Cumulative_graph'] = float(gradcam_full[:hidden].mean())
            graph_attributions['Away_Cumulative_graph'] = float(gradcam_full[hidden:hidden*2].mean())
            for i in range(k):
                base = hidden * 2 + i * 2 * hidden
                graph_attributions[f'Home_Interval_{i+1}_graph'] = float(
                    gradcam_full[base:base+hidden].mean())
                graph_attributions[f'Away_Interval_{i+1}_graph'] = float(
                    gradcam_full[base+hidden:base+2*hidden].mean())

            results.append({
                'pred_class': pred_class,
                'pred_name': CLASS_NAMES[pred_class],
                'confidence': confidence,
                'true_label': true_label,
                'true_name': CLASS_NAMES[true_label],
                'correct': pred_class == true_label,
                'probs': all_probs,
                'feature_attributions': feature_attributions,
                'graph_attributions': graph_attributions,
            })

            fusion_input.requires_grad_(False)

    hook.remove()
    return results


# ---------------------------------------------------------------------------
# Analysis A: Per-class "WHY does the model predict this class?"
# ---------------------------------------------------------------------------

def run_perclass_attribution(results, k, output_dir):
    """For each predicted class, aggregate: which features consistently matter?

    This answers: "When the model says Home Win, what features make it say that?"
    """
    print("\n" + "=" * 60)
    print("ANALYSIS A: Per-Class Feature Attribution")
    print("  'When model predicts X, which features drive that prediction?'")
    print("=" * 60)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_matches = [r for r in results if r['pred_class'] == cls_idx]
        if not cls_matches:
            print(f"  No matches predicted as {cls_name}, skipping.")
            continue

        print(f"\n--- {cls_name}: {len(cls_matches)} matches ---")

        # Aggregate feature Grad-CAM across all matches predicted as this class
        # Group by (feature_name) only — average across windows and sides
        feat_gradcam = defaultdict(list)
        feat_values = defaultdict(list)
        # Also group by (side, feature_name) for Home vs Away comparison
        side_feat_gradcam = defaultdict(list)
        side_feat_values = defaultdict(list)
        # Full detail: (window, side, feature)
        full_gradcam = defaultdict(list)
        full_values = defaultdict(list)

        for match in cls_matches:
            for attr in match['feature_attributions']:
                key = attr['feature']
                feat_gradcam[key].append(attr['gradcam'])
                feat_values[key].append(attr['value'])

                side_key = f"{attr['side']}_{attr['feature']}"
                side_feat_gradcam[side_key].append(attr['gradcam'])
                side_feat_values[side_key].append(attr['value'])

                full_key = f"{attr['window']}_{attr['side']}_{attr['feature']}"
                full_gradcam[full_key].append(attr['gradcam'])
                full_values[full_key].append(attr['value'])

        # --- Top 22 features (aggregated) with average values ---
        feat_avg = [(fname, np.mean(feat_gradcam[fname]), np.mean(feat_values[fname]))
                    for fname in ALL_IN_GAME_NAMES]
        feat_avg.sort(key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots(figsize=(12, 8))
        fnames = [x[0] for x in feat_avg]
        gcam_vals = [x[1] for x in feat_avg]
        avg_vals = [x[2] for x in feat_avg]

        bars = ax.barh(range(len(fnames)), gcam_vals[::-1], color=CLASS_COLORS[cls_idx], alpha=0.8)
        ax.set_yticks(range(len(fnames)))
        # Show feature name + average value
        ylabels = [f"{fn}  (avg={av:.1f})" for fn, _, av in reversed(feat_avg)]
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_xlabel('Mean Attribution Score')
        ax.set_title(f'Why does the model predict "{cls_name}"?\n'
                     f'({len(cls_matches)} matches, feature importance + avg values)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'why_{cls_name.replace(" ", "_").lower()}.png'), dpi=150)
        plt.close()
        print(f"  Saved: why_{cls_name.replace(' ', '_').lower()}.png")

        # --- Home vs Away breakdown for this class ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        for side, ax_s in [('Home', ax1), ('Away', ax2)]:
            side_data = [(fname,
                          np.mean(side_feat_gradcam[f'{side}_{fname}']),
                          np.mean(side_feat_values[f'{side}_{fname}']))
                         for fname in ALL_IN_GAME_NAMES
                         if f'{side}_{fname}' in side_feat_gradcam]
            side_data.sort(key=lambda x: x[1], reverse=True)

            s_names = [x[0] for x in side_data]
            s_gcam = [x[1] for x in side_data]
            s_vals = [x[2] for x in side_data]

            color = '#4CAF50' if side == 'Home' else '#F44336'
            ax_s.barh(range(len(s_names)), s_gcam[::-1], color=color, alpha=0.8)
            ax_s.set_yticks(range(len(s_names)))
            ylabels = [f"{fn}  (avg={av:.1f})" for fn, _, av in reversed(side_data)]
            ax_s.set_yticklabels(ylabels, fontsize=8)
            ax_s.set_xlabel('Mean Attribution')
            ax_s.set_title(f'{side} Team Features → {cls_name}')

        plt.suptitle(f'Home vs Away Feature Attribution for "{cls_name}" prediction', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                    f'why_{cls_name.replace(" ", "_").lower()}_home_vs_away.png'), dpi=150)
        plt.close()
        print(f"  Saved: why_{cls_name.replace(' ', '_').lower()}_home_vs_away.png")

        # --- Heatmap: feature × window for this class ---
        window_labels = ['Cumulative'] + [f'Interval_{i+1}' for i in range(k)]
        n_windows = len(window_labels)

        heatmap_data = np.zeros((n_windows * 2, len(ALL_IN_GAME_NAMES)))  # rows = window×side
        row_labels = []
        for w_idx, wname in enumerate(window_labels):
            for s_idx, side in enumerate(['Home', 'Away']):
                row = w_idx * 2 + s_idx
                row_labels.append(f'{side} {wname}')
                for f_idx, fname in enumerate(ALL_IN_GAME_NAMES):
                    key = f'{wname}_{side}_{fname}'
                    if key in full_gradcam:
                        heatmap_data[row, f_idx] = np.mean(full_gradcam[key])

        fig, ax = plt.subplots(figsize=(18, max(6, len(row_labels) * 0.5)))
        sns.heatmap(heatmap_data, xticklabels=ALL_IN_GAME_NAMES, yticklabels=row_labels,
                    cmap='YlOrRd', ax=ax, linewidths=0.5, annot=False)
        ax.set_title(f'Feature Attribution Heatmap for "{cls_name}" predictions\n'
                     f'(which feature × which time window drives the prediction)')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                    f'heatmap_{cls_name.replace(" ", "_").lower()}.png'), dpi=150)
        plt.close()
        print(f"  Saved: heatmap_{cls_name.replace(' ', '_').lower()}.png")

    # --- Comparison across classes: same feature, different importance ---
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(ALL_IN_GAME_NAMES))
    width = 0.25

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_matches = [r for r in results if r['pred_class'] == cls_idx]
        if not cls_matches:
            continue
        feat_scores = []
        for fname in ALL_IN_GAME_NAMES:
            scores = []
            for match in cls_matches:
                for attr in match['feature_attributions']:
                    if attr['feature'] == fname:
                        scores.append(attr['gradcam'])
            feat_scores.append(np.mean(scores) if scores else 0)

        ax.bar(x + cls_idx * width, feat_scores, width,
               label=cls_name, color=CLASS_COLORS[cls_idx], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(ALL_IN_GAME_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Attribution')
    ax.set_title('Feature Importance Comparison Across Predicted Classes\n'
                 '"Same feature, but how important for each outcome?"')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_comparison_features.png'), dpi=150)
    plt.close()
    print(f"\nSaved: class_comparison_features.png")


# ---------------------------------------------------------------------------
# Analysis A2: Hybrid Win/Loss Attribution
# ---------------------------------------------------------------------------

HYBRID_NAMES = ['Winning Team', 'Losing Team', 'Draw']
HYBRID_COLORS = ['#4CAF50', '#F44336', '#FF9800']


def _remap_side(pred_class: int, original_side: str) -> str:
    """Remap Home/Away to Winning/Losing based on predicted class.

    Home Win (0): Home → Winning, Away → Losing
    Away Win (1): Away → Winning, Home → Losing
    Draw (2): kept as-is (Home/Away)
    """
    if pred_class == 0:  # Home Win
        return 'Winning' if original_side == 'Home' else 'Losing'
    elif pred_class == 1:  # Away Win
        return 'Winning' if original_side == 'Away' else 'Losing'
    return original_side  # Draw: no remap


def run_hybrid_attribution(results, k, output_dir):
    """Hybrid analysis: combine Home Win's home + Away Win's away as 'Winning Team'.

    Mapping:
      Winning Team = Home Win's Home features + Away Win's Away features
      Losing Team  = Home Win's Away features + Away Win's Home features
      Draw         = separate (no remap)

    Generates:
      - Feature ranking bar chart per hybrid category
      - Winning vs Losing side-by-side bar chart
      - Heatmap (feature x window) per hybrid category
      - Winning vs Losing comparison chart
    """
    print("\n" + "=" * 60)
    print("ANALYSIS A2: Hybrid Win/Loss Feature Attribution")
    print("  'Which features matter for winning vs losing team?'")
    print("=" * 60)

    # Separate win matches (Home Win + Away Win) and Draw
    win_matches = [r for r in results if r['pred_class'] in (0, 1)]
    draw_matches = [r for r in results if r['pred_class'] == 2]

    # --- Build hybrid data for Winning/Losing ---
    hybrid_groups = {
        'Winning Team': {'feat_gradcam': defaultdict(list),
                         'feat_values': defaultdict(list),
                         'feat_signed': defaultdict(list),
                         'feat_gradient': defaultdict(list),
                         'full_gradcam': defaultdict(list),
                         'full_signed': defaultdict(list),
                         'matches': []},
        'Losing Team':  {'feat_gradcam': defaultdict(list),
                         'feat_values': defaultdict(list),
                         'feat_signed': defaultdict(list),
                         'feat_gradient': defaultdict(list),
                         'full_gradcam': defaultdict(list),
                         'full_signed': defaultdict(list),
                         'matches': []},
    }

    for match in win_matches:
        for attr in match['feature_attributions']:
            hybrid_side = _remap_side(match['pred_class'], attr['side'])
            group_name = f'{hybrid_side} Team'

            hybrid_groups[group_name]['feat_gradcam'][attr['feature']].append(attr['gradcam'])
            hybrid_groups[group_name]['feat_values'][attr['feature']].append(attr['value'])
            hybrid_groups[group_name]['feat_signed'][attr['feature']].append(attr['signed'])
            hybrid_groups[group_name]['feat_gradient'][attr['feature']].append(attr['gradient'])

            # Remap window label side for heatmap
            full_key = f"{attr['window']}_{attr['feature']}"
            hybrid_groups[group_name]['full_gradcam'][full_key].append(attr['gradcam'])
            hybrid_groups[group_name]['full_signed'][full_key].append(attr['signed'])

    hybrid_groups['Winning Team']['matches'] = win_matches
    hybrid_groups['Losing Team']['matches'] = win_matches

    print(f"  Win matches (Home Win + Away Win): {len(win_matches)}")
    print(f"  Draw matches: {len(draw_matches)}")

    # --- 1) Feature ranking bar chart: Winning vs Losing ---
    for h_name, h_color in zip(['Winning Team', 'Losing Team'],
                                ['#4CAF50', '#F44336']):
        grp = hybrid_groups[h_name]
        feat_avg = [(fname,
                     np.mean(grp['feat_gradcam'][fname]),
                     np.mean(grp['feat_values'][fname]))
                    for fname in ALL_IN_GAME_NAMES
                    if fname in grp['feat_gradcam']]
        feat_avg.sort(key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots(figsize=(12, 8))
        fnames = [x[0] for x in feat_avg]
        gcam_vals = [x[1] for x in feat_avg]

        ax.barh(range(len(fnames)), gcam_vals[::-1], color=h_color, alpha=0.8)
        ax.set_yticks(range(len(fnames)))
        ylabels = [f"{fn}  (avg={av:.1f})" for fn, _, av in reversed(feat_avg)]
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_xlabel('Mean Attribution Score')
        ax.set_title(f'Feature Importance: {h_name}\n'
                     f'({len(win_matches)} matches)')
        plt.tight_layout()
        fname_safe = h_name.replace(' ', '_').lower()
        plt.savefig(os.path.join(output_dir, f'hybrid_why_{fname_safe}.png'), dpi=150)
        plt.close()
        print(f"  Saved: hybrid_why_{fname_safe}.png")

    # --- 2) Winning vs Losing side-by-side ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    for grp_name, ax_s, color in [('Winning Team', ax1, '#4CAF50'),
                                   ('Losing Team', ax2, '#F44336')]:
        grp = hybrid_groups[grp_name]
        side_data = [(fname,
                      np.mean(grp['feat_gradcam'][fname]),
                      np.mean(grp['feat_values'][fname]))
                     for fname in ALL_IN_GAME_NAMES
                     if fname in grp['feat_gradcam']]
        side_data.sort(key=lambda x: x[1], reverse=True)

        s_names = [x[0] for x in side_data]
        s_gcam = [x[1] for x in side_data]

        ax_s.barh(range(len(s_names)), s_gcam[::-1], color=color, alpha=0.8)
        ax_s.set_yticks(range(len(s_names)))
        ylabels = [f"{fn}  (avg={av:.1f})" for fn, _, av in reversed(side_data)]
        ax_s.set_yticklabels(ylabels, fontsize=8)
        ax_s.set_xlabel('Mean Attribution')
        ax_s.set_title(f'{grp_name}')

    plt.suptitle(f'Winning vs Losing Team Feature Attribution\n'
                 f'({len(win_matches)} matches: Home Win + Away Win combined)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hybrid_winning_vs_losing.png'), dpi=150)
    plt.close()
    print(f"  Saved: hybrid_winning_vs_losing.png")

    # --- 3) Heatmap: feature × window for Winning / Losing ---
    window_labels = ['Cumulative'] + [f'Interval_{i+1}' for i in range(k)]

    for grp_name, color_map in [('Winning Team', 'Greens'),
                                 ('Losing Team', 'Reds')]:
        grp = hybrid_groups[grp_name]
        n_windows = len(window_labels)

        heatmap_data = np.zeros((n_windows, len(ALL_IN_GAME_NAMES)))
        for w_idx, wname in enumerate(window_labels):
            for f_idx, fname in enumerate(ALL_IN_GAME_NAMES):
                key = f'{wname}_{fname}'
                if key in grp['full_gradcam']:
                    heatmap_data[w_idx, f_idx] = np.mean(grp['full_gradcam'][key])

        fig, ax = plt.subplots(figsize=(18, max(4, n_windows * 0.8)))
        sns.heatmap(heatmap_data, xticklabels=ALL_IN_GAME_NAMES,
                    yticklabels=window_labels,
                    cmap=color_map, ax=ax, linewidths=0.5, annot=True, fmt='.3f',
                    annot_kws={'fontsize': 7})
        ax.set_title(f'Feature Attribution Heatmap: {grp_name}\n'
                     f'(feature × time window, {len(win_matches)} matches)')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        fname_safe = grp_name.replace(' ', '_').lower()
        plt.savefig(os.path.join(output_dir, f'hybrid_heatmap_{fname_safe}.png'), dpi=150)
        plt.close()
        print(f"  Saved: hybrid_heatmap_{fname_safe}.png")

    # --- 4) Combined heatmap: Winning on top, Losing on bottom ---
    n_windows = len(window_labels)
    combined_data = np.zeros((n_windows * 2, len(ALL_IN_GAME_NAMES)))
    combined_labels = []
    for grp_idx, grp_name in enumerate(['Winning Team', 'Losing Team']):
        grp = hybrid_groups[grp_name]
        for w_idx, wname in enumerate(window_labels):
            row = grp_idx * n_windows + w_idx
            combined_labels.append(f'{grp_name} — {wname}')
            for f_idx, fname in enumerate(ALL_IN_GAME_NAMES):
                key = f'{wname}_{fname}'
                if key in grp['full_gradcam']:
                    combined_data[row, f_idx] = np.mean(grp['full_gradcam'][key])

    fig, ax = plt.subplots(figsize=(18, max(6, len(combined_labels) * 0.6)))
    sns.heatmap(combined_data, xticklabels=ALL_IN_GAME_NAMES,
                yticklabels=combined_labels,
                cmap='YlOrRd', ax=ax, linewidths=0.5, annot=True, fmt='.3f',
                annot_kws={'fontsize': 7})
    ax.set_title(f'Winning vs Losing Team: Feature × Window Heatmap\n'
                 f'({len(win_matches)} matches — Home Win + Away Win combined)')
    # Add divider line between winning and losing
    ax.axhline(y=n_windows, color='white', linewidth=3)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hybrid_heatmap_combined.png'), dpi=150)
    plt.close()
    print(f"  Saved: hybrid_heatmap_combined.png")

    # --- 5) Winning vs Losing grouped bar comparison ---
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(ALL_IN_GAME_NAMES))
    width = 0.35

    for g_idx, (grp_name, color) in enumerate(
            zip(['Winning Team', 'Losing Team'], ['#4CAF50', '#F44336'])):
        grp = hybrid_groups[grp_name]
        scores = [np.mean(grp['feat_gradcam'].get(fname, [0]))
                  for fname in ALL_IN_GAME_NAMES]
        ax.bar(x + g_idx * width, scores, width,
               label=grp_name, color=color, alpha=0.8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(ALL_IN_GAME_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Attribution')
    ax.set_title('Winning vs Losing: Feature Importance Comparison\n'
                 '"Same feature — how important for winning vs losing team?"')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hybrid_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: hybrid_comparison.png")

    # --- 6) Directional: diverging bar chart per hybrid group ---
    # Shows: positive = feature pushes TOWARD this prediction
    #        negative = feature pushes AWAY from this prediction
    print("\n  --- Directional Analysis (signed grad x input) ---")

    for grp_name, pos_color, neg_color in [
            ('Winning Team', '#4CAF50', '#F44336'),
            ('Losing Team', '#F44336', '#4CAF50')]:
        grp = hybrid_groups[grp_name]
        feat_signed_avg = [(fname,
                            np.mean(grp['feat_signed'][fname]),
                            np.mean(grp['feat_values'][fname]))
                           for fname in ALL_IN_GAME_NAMES
                           if fname in grp['feat_signed']]
        feat_signed_avg.sort(key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots(figsize=(12, 8))
        fnames = [x[0] for x in feat_signed_avg]
        signed_vals = [x[1] for x in feat_signed_avg]
        colors = [pos_color if v >= 0 else neg_color for v in signed_vals]

        ax.barh(range(len(fnames)), signed_vals[::-1], color=colors[::-1], alpha=0.8)
        ax.set_yticks(range(len(fnames)))
        ylabels = [f"{fn}  (avg={av:.1f})" for fn, _, av in reversed(feat_signed_avg)]
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_xlabel('Mean Signed Attribution (grad x input)')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        support_label = 'Win' if grp_name == 'Winning Team' else 'Loss'
        ax.set_title(f'Directional Feature Attribution: {grp_name}\n'
                     f'(+ = pushes toward {support_label} prediction, '
                     f'- = pushes away)')
        plt.tight_layout()
        fname_safe = grp_name.replace(' ', '_').lower()
        plt.savefig(os.path.join(output_dir,
                    f'hybrid_directional_{fname_safe}.png'), dpi=150)
        plt.close()
        print(f"  Saved: hybrid_directional_{fname_safe}.png")

    # --- 7) Directional: Winning vs Losing side-by-side ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    for grp_name, ax_s in [('Winning Team', ax1), ('Losing Team', ax2)]:
        grp = hybrid_groups[grp_name]
        feat_data = [(fname, np.mean(grp['feat_signed'][fname]),
                      np.mean(grp['feat_values'][fname]))
                     for fname in ALL_IN_GAME_NAMES
                     if fname in grp['feat_signed']]
        feat_data.sort(key=lambda x: x[1], reverse=True)

        fnames = [x[0] for x in feat_data]
        vals = [x[1] for x in feat_data]
        colors = ['#4CAF50' if v >= 0 else '#F44336' for v in vals]

        ax_s.barh(range(len(fnames)), vals[::-1], color=colors[::-1], alpha=0.8)
        ax_s.set_yticks(range(len(fnames)))
        ylabels = [f"{fn}  (avg={av:.1f})" for fn, _, av in reversed(feat_data)]
        ax_s.set_yticklabels(ylabels, fontsize=8)
        ax_s.set_xlabel('Mean Signed Attribution')
        ax_s.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax_s.set_title(f'{grp_name}')

    plt.suptitle('Directional: Winning vs Losing Team\n'
                 '(green = supports prediction, red = opposes)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hybrid_directional_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: hybrid_directional_comparison.png")

    # --- 8) Directional heatmap: feature × window, diverging colormap ---
    for grp_name in ['Winning Team', 'Losing Team']:
        grp = hybrid_groups[grp_name]
        n_windows = len(window_labels)

        heatmap_data = np.zeros((n_windows, len(ALL_IN_GAME_NAMES)))
        for w_idx, wname in enumerate(window_labels):
            for f_idx, fname in enumerate(ALL_IN_GAME_NAMES):
                key = f'{wname}_{fname}'
                if key in grp['full_signed']:
                    heatmap_data[w_idx, f_idx] = np.mean(grp['full_signed'][key])

        vmax = max(abs(heatmap_data.min()), abs(heatmap_data.max()))
        if vmax == 0:
            vmax = 1

        fig, ax = plt.subplots(figsize=(18, max(4, n_windows * 0.8)))
        sns.heatmap(heatmap_data, xticklabels=ALL_IN_GAME_NAMES,
                    yticklabels=window_labels,
                    cmap='RdYlGn', center=0, vmin=-vmax, vmax=vmax,
                    ax=ax, linewidths=0.5, annot=True, fmt='.3f',
                    annot_kws={'fontsize': 7})
        support_label = 'Win' if grp_name == 'Winning Team' else 'Loss'
        ax.set_title(f'Directional Heatmap: {grp_name}\n'
                     f'(green = supports {support_label}, '
                     f'red = opposes, per window)')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        fname_safe = grp_name.replace(' ', '_').lower()
        plt.savefig(os.path.join(output_dir,
                    f'hybrid_dir_heatmap_{fname_safe}.png'), dpi=150)
        plt.close()
        print(f"  Saved: hybrid_dir_heatmap_{fname_safe}.png")

    # --- 9) Combined directional heatmap ---
    n_windows = len(window_labels)
    combined_data = np.zeros((n_windows * 2, len(ALL_IN_GAME_NAMES)))
    combined_labels = []
    for grp_idx, grp_name in enumerate(['Winning Team', 'Losing Team']):
        grp = hybrid_groups[grp_name]
        for w_idx, wname in enumerate(window_labels):
            row = grp_idx * n_windows + w_idx
            combined_labels.append(f'{grp_name} — {wname}')
            for f_idx, fname in enumerate(ALL_IN_GAME_NAMES):
                key = f'{wname}_{fname}'
                if key in grp['full_signed']:
                    combined_data[row, f_idx] = np.mean(grp['full_signed'][key])

    vmax = max(abs(combined_data.min()), abs(combined_data.max()))
    if vmax == 0:
        vmax = 1

    fig, ax = plt.subplots(figsize=(18, max(6, len(combined_labels) * 0.6)))
    sns.heatmap(combined_data, xticklabels=ALL_IN_GAME_NAMES,
                yticklabels=combined_labels,
                cmap='RdYlGn', center=0, vmin=-vmax, vmax=vmax,
                ax=ax, linewidths=0.5, annot=True, fmt='.3f',
                annot_kws={'fontsize': 7})
    ax.set_title(f'Directional: Winning vs Losing × Window\n'
                 f'(green = supports prediction, red = opposes)')
    ax.axhline(y=n_windows, color='black', linewidth=3)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hybrid_dir_heatmap_combined.png'), dpi=150)
    plt.close()
    print(f"  Saved: hybrid_dir_heatmap_combined.png")


# ---------------------------------------------------------------------------
# Analysis B: Correct vs Wrong — what does the model focus on?
# ---------------------------------------------------------------------------

def run_correct_vs_wrong(results, output_dir):
    """Compare feature attribution when model is correct vs wrong."""
    print("\n" + "=" * 60)
    print("ANALYSIS B: Correct vs Wrong Predictions")
    print("  'What does the model look at when it's right vs wrong?'")
    print("=" * 60)

    correct = [r for r in results if r['correct']]
    wrong = [r for r in results if not r['correct']]

    print(f"  Correct: {len(correct)} matches")
    print(f"  Wrong:   {len(wrong)} matches")

    if not correct or not wrong:
        print("  Need both correct and wrong predictions for comparison. Skipping.")
        return

    # Feature importance for correct vs wrong
    correct_feat = defaultdict(list)
    wrong_feat = defaultdict(list)

    for match in correct:
        for attr in match['feature_attributions']:
            correct_feat[attr['feature']].append(attr['gradcam'])
    for match in wrong:
        for attr in match['feature_attributions']:
            wrong_feat[attr['feature']].append(attr['gradcam'])

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(ALL_IN_GAME_NAMES))
    width = 0.35

    correct_scores = [np.mean(correct_feat[f]) for f in ALL_IN_GAME_NAMES]
    wrong_scores = [np.mean(wrong_feat[f]) for f in ALL_IN_GAME_NAMES]

    ax.bar(x - width/2, correct_scores, width, label=f'Correct ({len(correct)})',
           color='#4CAF50', alpha=0.8)
    ax.bar(x + width/2, wrong_scores, width, label=f'Wrong ({len(wrong)})',
           color='#F44336', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_IN_GAME_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Attribution')
    ax.set_title('Feature Importance: Correct vs Wrong Predictions\n'
                 '"Does the model focus on different features when it gets it wrong?"')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correct_vs_wrong_features.png'), dpi=150)
    plt.close()
    print(f"Saved: correct_vs_wrong_features.png")

    # Difference plot: which features differ most between correct and wrong
    diff = [(fname, np.mean(correct_feat[fname]) - np.mean(wrong_feat[fname]))
            for fname in ALL_IN_GAME_NAMES]
    diff.sort(key=lambda x: abs(x[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    d_names = [x[0] for x in diff]
    d_vals = [x[1] for x in diff]
    colors = ['#4CAF50' if v > 0 else '#F44336' for v in d_vals]

    ax.barh(range(len(d_names)), d_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(d_names)))
    ax.set_yticklabels(d_names[::-1], fontsize=9)
    ax.set_xlabel('Attribution Difference (Correct - Wrong)')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Feature Focus Difference: Correct vs Wrong\n'
                 'Green = model focuses MORE on this when correct\n'
                 'Red = model focuses MORE on this when wrong')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correct_vs_wrong_diff.png'), dpi=150)
    plt.close()
    print(f"Saved: correct_vs_wrong_diff.png")

    # Confidence distribution
    correct_conf = [r['confidence'] for r in correct]
    wrong_conf = [r['confidence'] for r in wrong]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(correct_conf, bins=20, alpha=0.7, label=f'Correct ({len(correct)})', color='#4CAF50')
    ax.hist(wrong_conf, bins=20, alpha=0.7, label=f'Wrong ({len(wrong)})', color='#F44336')
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution: Correct vs Wrong')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=150)
    plt.close()
    print(f"Saved: confidence_distribution.png")


# ---------------------------------------------------------------------------
# Analysis C: Temporal window analysis per class
# ---------------------------------------------------------------------------

def run_temporal_per_class(results, k, output_dir):
    """Which temporal windows matter most, broken down by predicted class."""
    print("\n" + "=" * 60)
    print("ANALYSIS C: Temporal Window Importance per Class")
    print("=" * 60)

    window_labels = ['Cumulative'] + [f'Interval_{i+1}' for i in range(k)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_matches = [r for r in results if r['pred_class'] == cls_idx]
        if not cls_matches:
            continue

        window_scores = {w: [] for w in window_labels}

        for match in cls_matches:
            for attr in match['feature_attributions']:
                window_scores[attr['window']].append(attr['gradcam'])

            # Also add graph embedding contribution
            for gname, gval in match['graph_attributions'].items():
                for wl in window_labels:
                    if wl in gname:
                        window_scores[wl].append(gval)
                        break

        means = [np.mean(window_scores[w]) for w in window_labels]
        stds = [np.std(window_scores[w]) for w in window_labels]

        colors_w = ['#FF9800'] + ['#2196F3'] * k
        ax = axes[cls_idx]
        ax.bar(range(len(window_labels)), means, yerr=stds, color=colors_w, capsize=4)
        ax.set_xticks(range(len(window_labels)))
        ax.set_xticklabels(window_labels, rotation=15, fontsize=9)
        ax.set_ylabel('Mean Attribution')
        ax.set_title(f'{cls_name} ({len(cls_matches)} matches)')

    plt.suptitle('Which Time Window Matters Most per Prediction Class?', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_per_class.png'), dpi=150)
    plt.close()
    print(f"Saved: temporal_per_class.png")


# ---------------------------------------------------------------------------
# Grad-CAM hook for GAT conv layers
# ---------------------------------------------------------------------------

class GATGradCAMHook:
    """Captures activations + gradients at a GATConv layer."""

    def __init__(self, module: nn.Module):
        self.activations = None
        self.gradients = None
        self._fwd = module.register_forward_hook(self._fwd_hook)
        self._bwd = module.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()


# ---------------------------------------------------------------------------
# Football pitch drawing utility
# ---------------------------------------------------------------------------

def draw_pitch(ax, pitch_color='#2e8b57', line_color='white'):
    """Draw a football pitch on the given axes (100 x 68 coordinate system)."""
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 73)
    ax.set_aspect('equal')
    ax.set_facecolor(pitch_color)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    lw = 1.5
    # Pitch outline
    ax.plot([0, 100, 100, 0, 0], [0, 0, 68, 68, 0], color=line_color, lw=lw)
    # Halfway line
    ax.plot([50, 50], [0, 68], color=line_color, lw=lw)
    # Centre circle
    circle = plt.Circle((50, 34), 9.15, color=line_color, fill=False, lw=lw)
    ax.add_patch(circle)
    ax.plot(50, 34, 'o', color=line_color, ms=3)
    # Penalty areas
    ax.plot([0, 16.5, 16.5, 0], [13.84, 13.84, 54.16, 54.16], color=line_color, lw=lw)
    ax.plot([100, 83.5, 83.5, 100], [13.84, 13.84, 54.16, 54.16], color=line_color, lw=lw)
    # Goal areas
    ax.plot([0, 5.5, 5.5, 0], [24.84, 24.84, 43.16, 43.16], color=line_color, lw=lw)
    ax.plot([100, 94.5, 94.5, 100], [24.84, 24.84, 43.16, 43.16], color=line_color, lw=lw)
    # Penalty spots
    ax.plot(11, 34, 'o', color=line_color, ms=3)
    ax.plot(89, 34, 'o', color=line_color, ms=3)


# Default position coordinates on pitch (for when avg_x/avg_y aren't available)
DEFAULT_POS_XY = {
    'GK':  (5, 34),
    'DC':  (20, 34), 'DL':  (20, 55), 'DR':  (20, 13),
    'DMC': (30, 34), 'DML': (30, 55), 'DMR': (30, 13),
    'MC':  (45, 34), 'ML':  (45, 55), 'MR':  (45, 13),
    'AMC': (60, 34), 'AML': (60, 55), 'AMR': (60, 13),
    'FW':  (75, 34), 'Sub': (90, 60),
}


# ---------------------------------------------------------------------------
# Analysis D: Node-level Grad-CAM + Pitch Visualization
# ---------------------------------------------------------------------------

def compute_node_gradcam(model, loader, device, k, scalers=None):
    """Compute per-node importance via Input x Gradient on cumulative home graph.

    Uses gradients on hc.x directly (avoids shared-encoder hook issue where
    cumulative_gat.conv3 is called for both home and away graphs).

    importance_per_node = ReLU(grad(hc.x) * hc.x).sum(features)

    Returns: dict keyed by pred_class -> list of per-match node info dicts.
    Each node info: {'position': str, 'avg_x': float, 'avg_y': float, 'cam': float}
    """
    model.eval()
    cum_node_scaler = scalers.get('cum_node') if scalers else None

    # Per-class node data: class -> list of (list of node dicts per match)
    class_nodes = {0: [], 1: [], 2: []}

    n_processed = 0

    for batch_dict in loader:
        batch_dict = batch_to_device(batch_dict, device)
        labels = batch_dict['labels']
        batch_size = labels.size(0)

        hc = batch_dict['home_cum_graph']
        hc.x.requires_grad_(True)

        # Single forward pass for entire batch
        logits = model(batch_dict)

        for sample_idx in range(batch_size):
            pred_class = logits[sample_idx].argmax().item()

            model.zero_grad()
            if hc.x.grad is not None:
                hc.x.grad.zero_()

            logits[sample_idx, pred_class].backward(retain_graph=True)

            if hc.x.grad is None:
                n_processed += 1
                continue

            # Select nodes belonging to this sample
            if hc.batch is not None:
                mask = (hc.batch == sample_idx)
            else:
                mask = torch.ones(hc.x.size(0), dtype=torch.bool, device=device)

            node_grad = hc.x.grad[mask].detach().cpu().numpy()
            node_feat = hc.x[mask].detach().cpu().numpy()

            # Input x Gradient: importance = ReLU(grad * input) summed over features
            node_importance = np.maximum(node_grad * node_feat, 0).sum(axis=1)

            match_nodes = []
            for node_idx in range(len(node_importance)):
                feat = node_feat[node_idx]

                # Inverse-scale to get position + avg_x/avg_y
                if cum_node_scaler is not None:
                    feat_orig = cum_node_scaler.inverse_transform(feat.reshape(1, -1))[0]
                else:
                    feat_orig = feat

                pos_val = int(round(feat_orig[0]))
                pos_name = REVERSE_POSITION.get(pos_val, f'Pos{pos_val}')
                avg_x = float(feat_orig[5])  # index 5 = avg_x
                avg_y = float(feat_orig[6])  # index 6 = avg_y

                match_nodes.append({
                    'position': pos_name,
                    'avg_x': avg_x,
                    'avg_y': avg_y,
                    'cam': float(node_importance[node_idx]),
                })

            if match_nodes:
                class_nodes[pred_class].append(match_nodes)

            n_processed += 1

        hc.x.requires_grad_(False)

    return class_nodes


def run_pitch_visualization(model, loader, device, k, output_dir, scalers=None):
    """Draw football pitch with players colored by Grad-CAM importance.

    Analogous to image GradCAM overlay:
      CNN:  "which pixels does the model look at?"
      GNN:  "which players does the model look at?"
    """
    print("\n" + "=" * 60)
    print("ANALYSIS D: Football Pitch Grad-CAM Visualization")
    print("  'Which players does the model focus on for each outcome?'")
    print("=" * 60)

    class_nodes = compute_node_gradcam(model, loader, device, k, scalers=scalers)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        matches = class_nodes[cls_idx]
        if not matches:
            print(f"  No data for {cls_name}, skipping.")
            continue

        print(f"  {cls_name}: {len(matches)} matches")

        # Aggregate: average Grad-CAM per position across all matches
        pos_data = defaultdict(lambda: {'cam': [], 'x': [], 'y': []})
        for match_nodes in matches:
            for node in match_nodes:
                pos = node['position']
                pos_data[pos]['cam'].append(node['cam'])
                pos_data[pos]['x'].append(node['avg_x'])
                pos_data[pos]['y'].append(node['avg_y'])

        positions = list(pos_data.keys())
        avg_cams = [np.mean(pos_data[p]['cam']) for p in positions]
        avg_xs = [np.mean(pos_data[p]['x']) for p in positions]
        avg_ys = [np.mean(pos_data[p]['y']) for p in positions]

        # Normalize CAM scores for colormap (0-1)
        cam_arr = np.array(avg_cams)
        if cam_arr.max() > cam_arr.min():
            cam_norm = (cam_arr - cam_arr.min()) / (cam_arr.max() - cam_arr.min())
        else:
            cam_norm = np.ones_like(cam_arr) * 0.5

        # --- Draw pitch ---
        fig, ax = plt.subplots(figsize=(12, 8))
        draw_pitch(ax)

        cmap = plt.cm.YlOrRd
        norm = Normalize(vmin=cam_arr.min(), vmax=cam_arr.max())
        sm = ScalarMappable(cmap=cmap, norm=norm)

        for i, (pos, cx, cy, cam_val, cn) in enumerate(
                zip(positions, avg_xs, avg_ys, avg_cams, cam_norm)):
            # avg_x and avg_y are both 0-100 (WhoScored), scale y to pitch height 0-68
            px, py = cx, cy * 68.0 / 100.0
            if px < 0 or px > 100 or py < 0 or py > 68:
                px, py = DEFAULT_POS_XY.get(pos, (50, 34))

            color = cmap(cn)
            size = 300 + cn * 700  # bigger = more important

            ax.scatter(px, py, s=size, c=[color], edgecolors='white',
                       linewidths=2, zorder=5, alpha=0.9)
            label_offset = -(np.sqrt(size) / 2 + 8)
            ax.annotate(pos, (px, py), textcoords="offset points",
                       xytext=(0, label_offset), ha='center', fontsize=8,
                       fontweight='bold', color='white', zorder=6,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

        # Colorbar
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
        cbar.set_label('Grad-CAM Score', fontsize=10)

        ax.set_title(f'Grad-CAM Player Importance: {cls_name}\n'
                     f'({len(matches)} matches — bigger & redder = more important)',
                     fontsize=13, color='white', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                    f'pitch_{cls_name.replace(" ", "_").lower()}.png'),
                    dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved: pitch_{cls_name.replace(' ', '_').lower()}.png")

    # --- All 3 classes side by side ---
    fig, axes = plt.subplots(1, 3, figsize=(36, 10))

    # Global normalization across all classes
    all_cams = []
    for cls_idx in range(3):
        for match_nodes in class_nodes[cls_idx]:
            for node in match_nodes:
                all_cams.append(node['cam'])

    if all_cams:
        global_min = np.min(all_cams)
        global_max = np.max(all_cams)
    else:
        global_min, global_max = 0, 1

    global_norm = Normalize(vmin=global_min, vmax=global_max)
    cmap = plt.cm.YlOrRd

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        ax = axes[cls_idx]
        draw_pitch(ax)

        matches = class_nodes[cls_idx]
        if not matches:
            ax.set_title(f'{cls_name}\n(no data)', fontsize=12, color='white')
            continue

        pos_data = defaultdict(lambda: {'cam': [], 'x': [], 'y': []})
        for match_nodes in matches:
            for node in match_nodes:
                pos = node['position']
                pos_data[pos]['cam'].append(node['cam'])
                pos_data[pos]['x'].append(node['avg_x'])
                pos_data[pos]['y'].append(node['avg_y'])

        for pos, data in pos_data.items():
            avg_cam = np.mean(data['cam'])
            px = np.mean(data['x'])
            py = np.mean(data['y']) * 68.0 / 100.0  # WhoScored 0-100 -> pitch 0-68

            if px < 0 or px > 100 or py < 0 or py > 68:
                px, py = DEFAULT_POS_XY.get(pos, (50, 34))

            cn = global_norm(avg_cam)
            color = cmap(cn)
            size = 200 + cn * 600

            ax.scatter(px, py, s=size, c=[color], edgecolors='white',
                       linewidths=1.5, zorder=5, alpha=0.9)
            label_offset = -(np.sqrt(size) / 2 + 6)
            ax.annotate(pos, (px, py), textcoords="offset points",
                       xytext=(0, label_offset), ha='center', fontsize=7,
                       fontweight='bold', color='white', zorder=6,
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6))

        ax.set_title(f'{cls_name} ({len(matches)} matches)',
                     fontsize=12, color='white', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    sm = ScalarMappable(cmap=cmap, norm=global_norm)
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.015, pad=0.02)
    cbar.set_label('Grad-CAM Score', fontsize=11)

    plt.suptitle('Which Players Does the Model Focus on per Outcome?',
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pitch_all_classes.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: pitch_all_classes.png")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def save_text_summary(results, k, output_dir):
    """Save human-readable text summary."""
    print("\n" + "=" * 60)
    print("Saving text summary...")

    n_correct = sum(1 for r in results if r['correct'])
    n_wrong = sum(1 for r in results if not r['correct'])

    with open(os.path.join(output_dir, 'gradcam_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("Grad-CAM Feature Attribution Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total matches analyzed: {len(results)}\n")
        f.write(f"Correct predictions: {n_correct} ({n_correct/len(results)*100:.1f}%)\n")
        f.write(f"Wrong predictions: {n_wrong} ({n_wrong/len(results)*100:.1f}%)\n\n")

        # Per-class summary
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cls_matches = [r for r in results if r['pred_class'] == cls_idx]
            if not cls_matches:
                continue

            cls_correct = sum(1 for r in cls_matches if r['correct'])
            avg_conf = np.mean([r['confidence'] for r in cls_matches])

            f.write(f"\n{'=' * 70}\n")
            f.write(f"Predicted: {cls_name} ({len(cls_matches)} matches, "
                    f"{cls_correct} correct, avg confidence: {avg_conf:.1%})\n")
            f.write(f"{'=' * 70}\n\n")

            f.write(f"WHY does the model predict {cls_name}?\n")
            f.write(f"Top features driving this prediction (with avg values):\n")
            f.write(f"{'-' * 60}\n")

            # Aggregate features
            feat_data = defaultdict(lambda: {'gradcam': [], 'value': []})
            for match in cls_matches:
                for attr in match['feature_attributions']:
                    key = attr['feature']
                    feat_data[key]['gradcam'].append(attr['gradcam'])
                    feat_data[key]['value'].append(attr['value'])

            sorted_feats = sorted(feat_data.items(),
                                  key=lambda x: np.mean(x[1]['gradcam']), reverse=True)

            for rank, (fname, data) in enumerate(sorted_feats, 1):
                avg_gcam = np.mean(data['gradcam'])
                avg_val = np.mean(data['value'])
                bar = '█' * int(avg_gcam * 100)
                f.write(f"  {rank:2d}. {fname:<25s}  "
                        f"Grad-CAM: {avg_gcam:.4f}  "
                        f"Avg Value: {avg_val:>8.1f}  "
                        f"{bar}\n")

            # Home vs Away feature breakdown
            f.write(f"\n  Home vs Away breakdown (top 10 per side):\n")
            for side in ['Home', 'Away']:
                f.write(f"\n  {side} Team:\n")
                side_data = defaultdict(lambda: {'gradcam': [], 'value': []})
                for match in cls_matches:
                    for attr in match['feature_attributions']:
                        if attr['side'] == side:
                            side_data[attr['feature']]['gradcam'].append(attr['gradcam'])
                            side_data[attr['feature']]['value'].append(attr['value'])

                sorted_side = sorted(side_data.items(),
                                     key=lambda x: np.mean(x[1]['gradcam']), reverse=True)
                for rank, (fname, data) in enumerate(sorted_side[:10], 1):
                    avg_gcam = np.mean(data['gradcam'])
                    avg_val = np.mean(data['value'])
                    f.write(f"    {rank:2d}. {fname:<25s}  "
                            f"GC: {avg_gcam:.4f}  Val: {avg_val:>8.1f}\n")

            # Graph embedding vs in-game contribution
            f.write(f"\n  Graph Embedding vs In-Game Feature contribution:\n")
            graph_scores = []
            ingame_scores = []
            for match in cls_matches:
                g_total = sum(match['graph_attributions'].values())
                ig_total = sum(a['gradcam'] for a in match['feature_attributions'])
                graph_scores.append(g_total)
                ingame_scores.append(ig_total)
            f.write(f"    Graph Emb avg: {np.mean(graph_scores):.4f}\n")
            f.write(f"    In-Game avg:   {np.mean(ingame_scores):.4f}\n")

    print(f"Saved: gradcam_summary.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Analysis for TSW-GAT")
    parser.add_argument("--pred_min", type=int, default=90)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--fibonacci", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu") if args.use_cpu else get_device()
    print(f"Device: {device}")

    k = args.k
    if args.fibonacci and k == 2:
        k = 4
        print(f"Fibonacci mode: using k={k}")

    n_suffix = "Nfib" if args.fibonacci else f"N{args.N}"

    # Output directory
    output_dir = os.path.join("explainability_results",
                              f"gradcam_k{k}_{n_suffix}_{args.pred_min}min")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load config
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

    model_path = Path("models") / f"temporal_predictor_k{k}_{n_suffix}_{args.pred_min}min.pt"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded: {model_path}")

    # Load test data
    test_data_path = Path("models") / f"test_data_k{k}_{n_suffix}_{args.pred_min}min.pt"
    if not test_data_path.exists():
        print(f"Test data not found: {test_data_path}")
        return
    test_samples = torch.load(test_data_path, weights_only=False)
    print(f"Test samples: {len(test_samples)}")

    # Load scalers (for inverse-transforming feature values)
    scaler_path = Path("models") / f"scalers_k{k}_{n_suffix}_{args.pred_min}min.pkl"
    scalers = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        print(f"Scalers loaded: {scaler_path}")

    # DataLoader
    test_dataset = TemporalMatchDataset(test_samples)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=temporal_collate_fn,
    )

    # === Step 1: Compute attributions for ALL matches ===
    print("\n" + "=" * 60)
    print("Computing per-match Grad-CAM attributions...")
    print("=" * 60)

    all_results = compute_all_match_attributions(
        model, test_loader, device, k, scalers=scalers)
    print(f"\nTotal matches analyzed: {len(all_results)}")

    n_correct = sum(1 for r in all_results if r['correct'])
    print(f"Correct: {n_correct}/{len(all_results)} "
          f"({n_correct/len(all_results)*100:.1f}%)")

    # === Analysis A: Per-class feature attribution ===
    run_perclass_attribution(all_results, k, output_dir)

    # === Analysis A2: Hybrid Win/Loss attribution ===
    run_hybrid_attribution(all_results, k, output_dir)

    # === Analysis B: Correct vs Wrong ===
    run_correct_vs_wrong(all_results, output_dir)

    # === Analysis C: Temporal per class ===
    run_temporal_per_class(all_results, k, output_dir)

    # === Analysis D: Pitch visualization ===
    run_pitch_visualization(model, test_loader, device, k, output_dir, scalers=scalers)

    # === Text summary ===
    save_text_summary(all_results, k, output_dir)

    print("\n" + "=" * 60)
    print("GRAD-CAM ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

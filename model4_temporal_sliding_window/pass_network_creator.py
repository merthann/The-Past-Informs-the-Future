"""
Temporal Pass Network Creator for Model 4 (Sliding Window)

Creates temporal sliding window pass network graphs from cached .pt files.

Graph Types:
  - Cumulative Graph (t): 7 node features (position, height, weight, rating, pass_accuracy, avg_x, avg_y)
    avg_x/avg_y from player_positions.csv
  - Interval Graphs (t-kN to t-N): 6 node features (no rating)
    avg_x/avg_y from events within interval

In-Game Features (22 per team per window):
  - 16 from features.csv (cumulative) or event counting (interval)
  - 6 event-based (pass_success_rate, final_third_passes, crosses, key_passes, big_chance, shot_assist)

Data Leakage Prevention:
  - Rating ONLY in cumulative graph (post-match metric)
  - Interval avg_x/avg_y from events within interval only
  - Interval in-game features computed from interval events only
  - Scaling done after train/test split (in train_model.py)
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional


# Position label encoding (same as Model 3)
POSITION_ENCODING = {
    'GK': 1, 'DC': 2, 'DL': 3, 'DR': 4, 'DMC': 5, 'DML': 6, 'DMR': 7,
    'MC': 8, 'ML': 9, 'MR': 10, 'AMC': 11, 'AML': 12, 'AMR': 13, 'FW': 14, 'Sub': 15
}

# 16 in-game feature columns (from features.csv)
IN_GAME_FEATURE_COLS = [
    'ball_touches', 'substitutions', 'aerials', 'ball_recoveries',
    'blocked_passes', 'cards', 'clearances', 'corners_awarded',
    'dispossessions', 'fouls', 'interceptions', 'missed_shots',
    'saved_shots', 'tackles', 'take_ons', 'offsides_provoked',
]

# Mapping from in-game feature names to event type strings in events.csv
# Used for computing interval-based features from raw events
EVENT_TYPE_MAP = {
    'ball_touches': ['BallTouch'],
    'substitutions': ['SubstitutionOn'],
    'aerials': ['Aerial'],
    'ball_recoveries': ['BallRecovery'],
    'blocked_passes': ['BlockedPass'],
    'cards': ['Card'],
    'clearances': ['Clearance'],
    'corners_awarded': ['CornerAwarded'],
    'dispossessions': ['Dispossessed'],
    'fouls': ['Foul'],
    'interceptions': ['Interception'],
    'missed_shots': ['MissedShots'],
    'saved_shots': ['SavedShot'],
    'tackles': ['Tackle'],
    'take_ons': ['TakeOn'],
    'offsides_provoked': ['OffsideProvoked'],
}

# 6 additional event-based feature columns
EVENT_FEATURE_COLS = [
    'pass_success_rate', 'final_third_passes', 'crosses',
    'key_passes', 'big_chance', 'shot_assist',
]

# Total in-game features per team per window
TOTAL_IN_GAME_FEATURES = len(IN_GAME_FEATURE_COLS) + len(EVENT_FEATURE_COLS)  # 22


def _get_time_col(df: pd.DataFrame) -> str:
    """Get the best available time column from a DataFrame."""
    if 'expandedMinute' in df.columns:
        return 'expandedMinute'
    if 'minute' in df.columns:
        return 'minute'
    return 'minute'


def _get_starting_players(players: pd.DataFrame, team_side: str) -> pd.DataFrame:
    """Get starting 11 players for a team."""
    return players[
        (players['team_side'] == team_side) &
        (players['is_first_eleven'] == True)
    ].copy()


def _calculate_pass_accuracy(passes_df: pd.DataFrame, player_id: int) -> float:
    """Calculate pass accuracy for a specific player from passes data."""
    if passes_df.empty or 'playerId' not in passes_df.columns:
        return 0.0
    player_passes = passes_df[passes_df['playerId'] == player_id]
    total = len(player_passes)
    if total == 0:
        return 0.0
    successful = player_passes['outcomeType'].astype(str).str.contains('Successful', na=False).sum()
    return successful / total


def _build_pass_edges(
    passes_df: pd.DataFrame,
    player_to_node: Dict[int, int]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build edge index and normalized edge weights from pass data."""
    edge_dict: Dict[Tuple[int, int], int] = {}

    for _, row in passes_df.iterrows():
        passer = row.get('playerId')
        receiver = row.get('receiver_id')

        if pd.isna(passer) or pd.isna(receiver):
            continue

        passer, receiver = int(passer), int(receiver)

        if passer not in player_to_node or receiver not in player_to_node:
            continue
        if passer == receiver:
            continue

        edge_key = (player_to_node[passer], player_to_node[receiver])
        edge_dict[edge_key] = edge_dict.get(edge_key, 0) + 1

    if not edge_dict:
        return None, None

    edge_indices = torch.tensor(list(edge_dict.keys()), dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(list(edge_dict.values()), dtype=torch.float)
    if edge_weights.max() > 0:
        edge_weights = edge_weights / edge_weights.max()

    return edge_indices, edge_weights


def _compute_6_event_features(team_events: pd.DataFrame) -> List[float]:
    """Compute the 6 event-based features from a filtered events DataFrame.

    Features: pass_success_rate, final_third_passes, crosses, key_passes, big_chance, shot_assist
    """
    passes = team_events[team_events['type'] == 'Pass'] if 'type' in team_events.columns else pd.DataFrame()
    total_passes = len(passes)

    if total_passes > 0:
        successful = passes['outcomeType'].astype(str).str.contains('Successful', na=False).sum()
        pass_success_rate = successful / total_passes
    else:
        pass_success_rate = 0.0

    # Final third passes (x > 66.7)
    if total_passes > 0 and 'x' in passes.columns:
        final_third = int((passes['x'].fillna(0) > 66.7).sum())
    else:
        final_third = 0

    # Qualifier-based features
    crosses = 0
    key_passes = 0
    big_chance = 0
    shot_assist = 0

    if 'qualifiers' in team_events.columns:
        quals_series = team_events['qualifiers'].astype(str)
        types_series = team_events['type'].astype(str) if 'type' in team_events.columns else pd.Series(dtype=str)

        for idx, quals in quals_series.items():
            evt_type = types_series.get(idx, '')
            if 'Cross' in quals and evt_type == 'Pass':
                crosses += 1
            if 'KeyPass' in quals:
                key_passes += 1
            if 'BigChance' in quals:
                big_chance += 1
            if 'IntentionalGoalAssist' in quals or 'IntentionalAssist' in quals:
                shot_assist += 1

    return [pass_success_rate, float(final_third), float(crosses),
            float(key_passes), float(big_chance), float(shot_assist)]


# ---------------------------------------------------------------------------
# PUBLIC API: Graph Creation Functions
# ---------------------------------------------------------------------------

def create_cumulative_graph(
    events: pd.DataFrame,
    players: pd.DataFrame,
    positions: pd.DataFrame,
    team_id: int,
    team_side: str,
    pred_min: int,
) -> Optional[Data]:
    """Create cumulative graph (0 to pred_min) with 7 node features.

    Node features: position, height, weight, rating, pass_accuracy, avg_x, avg_y
    - avg_x/avg_y sourced from player_positions.csv (positions DataFrame)
    - rating included (only for cumulative graph)
    - passes extracted from events (type='Pass')
    """
    time_col = _get_time_col(events)

    # Extract passes from events
    if not events.empty and 'type' in events.columns:
        team_passes = events[
            (events['type'] == 'Pass') &
            (events['teamId'] == team_id) &
            (events[time_col] <= pred_min)
        ].copy()
    else:
        team_passes = pd.DataFrame()

    team_positions = positions[positions['team'] == team_side].copy() if not positions.empty else pd.DataFrame()
    team_players = _get_starting_players(players, team_side)

    if team_positions.empty or team_passes.empty:
        return None

    team_positions = team_positions.dropna(subset=['player_id'])
    if team_positions.empty:
        return None

    player_ids = team_positions['player_id'].astype(int).tolist()
    player_to_node = {pid: idx for idx, pid in enumerate(player_ids)}

    # Node features: [position, height, weight, rating, pass_accuracy, avg_x, avg_y]
    node_features = []
    for _, pos_row in team_positions.iterrows():
        pid = int(pos_row['player_id'])
        pstats = team_players[team_players['player_id'] == pid]

        position = POSITION_ENCODING.get(
            pstats.iloc[0]['position'] if not pstats.empty else 'MC', 8)
        height = float(pstats.iloc[0]['height_cm']) if not pstats.empty and pd.notna(pstats.iloc[0].get('height_cm')) else 180.0
        weight = float(pstats.iloc[0]['weight_kg']) if not pstats.empty and pd.notna(pstats.iloc[0].get('weight_kg')) else 75.0
        rating = float(pstats.iloc[0]['rating']) if not pstats.empty and pd.notna(pstats.iloc[0].get('rating')) else 6.0
        pass_acc = _calculate_pass_accuracy(team_passes, pid)
        avg_x = float(pos_row['avg_x']) if pd.notna(pos_row.get('avg_x')) else 50.0
        avg_y = float(pos_row['avg_y']) if pd.notna(pos_row.get('avg_y')) else 50.0

        node_features.append([position, height, weight, rating, pass_acc, avg_x, avg_y])

    node_tensor = torch.tensor(np.array(node_features), dtype=torch.float)

    # Edges
    edge_index, edge_weight = _build_pass_edges(team_passes, player_to_node)
    if edge_index is None:
        return None
    if edge_index.max().item() >= node_tensor.shape[0]:
        return None

    return Data(x=node_tensor, edge_index=edge_index, edge_weight=edge_weight)


def create_interval_graph(
    events: pd.DataFrame,
    players: pd.DataFrame,
    team_id: int,
    team_side: str,
    start_min: int,
    end_min: int,
) -> Optional[Data]:
    """Create interval graph with 6 node features (NO rating).

    Node features: position, height, weight, pass_accuracy, avg_x, avg_y
    - avg_x/avg_y from events within [start_min, end_min) (NOT player_positions)
    - pass_accuracy from passes within [start_min, end_min)
    - NO rating (data leakage prevention)
    - passes extracted from events (type='Pass')
    """
    time_col = _get_time_col(events)

    # Filter passes by interval and team (extract from events)
    if not events.empty and 'type' in events.columns:
        interval_passes = events[
            (events['type'] == 'Pass') &
            (events['teamId'] == team_id) &
            (events[time_col] >= start_min) &
            (events[time_col] < end_min)
        ].copy()
    else:
        interval_passes = pd.DataFrame()

    # Filter events by interval and team (for avg_x/avg_y)
    if not events.empty:
        interval_events = events[
            (events['teamId'] == team_id) &
            (events[time_col] >= start_min) &
            (events[time_col] < end_min)
        ].copy()
    else:
        interval_events = pd.DataFrame()

    team_players = _get_starting_players(players, team_side)
    if team_players.empty:
        return None

    player_ids = team_players['player_id'].astype(int).tolist()
    player_to_node = {pid: idx for idx, pid in enumerate(player_ids)}

    # Node features: [position, height, weight, pass_accuracy, avg_x, avg_y]
    node_features = []
    for pid in player_ids:
        pstats = team_players[team_players['player_id'] == pid]

        position = POSITION_ENCODING.get(
            pstats.iloc[0]['position'] if not pstats.empty else 'MC', 8)
        height = float(pstats.iloc[0]['height_cm']) if not pstats.empty and pd.notna(pstats.iloc[0].get('height_cm')) else 180.0
        weight = float(pstats.iloc[0]['weight_kg']) if not pstats.empty and pd.notna(pstats.iloc[0].get('weight_kg')) else 75.0
        pass_acc = _calculate_pass_accuracy(interval_passes, pid)

        # avg_x, avg_y from events within interval
        if not interval_events.empty and 'playerId' in interval_events.columns:
            player_evts = interval_events[interval_events['playerId'] == pid]
            if len(player_evts) > 0 and 'x' in player_evts.columns and 'y' in player_evts.columns:
                avg_x = player_evts['x'].mean()
                avg_y = player_evts['y'].mean()
                avg_x = float(avg_x) if pd.notna(avg_x) else 50.0
                avg_y = float(avg_y) if pd.notna(avg_y) else 50.0
            else:
                avg_x, avg_y = 50.0, 50.0
        else:
            avg_x, avg_y = 50.0, 50.0

        node_features.append([position, height, weight, pass_acc, avg_x, avg_y])

    node_tensor = torch.tensor(np.array(node_features), dtype=torch.float)

    # Edges from interval passes
    edge_index, edge_weight = _build_pass_edges(interval_passes, player_to_node)

    if edge_index is None:
        # No passes in this interval - create self-loop edges so GAT can still process
        n_nodes = len(player_ids)
        if n_nodes == 0:
            return None
        edge_index = torch.tensor([[i, i] for i in range(n_nodes)], dtype=torch.long).t().contiguous()
        edge_weight = torch.ones(n_nodes, dtype=torch.float)

    if edge_index.max().item() >= node_tensor.shape[0]:
        return None

    return Data(x=node_tensor, edge_index=edge_index, edge_weight=edge_weight)


# ---------------------------------------------------------------------------
# PUBLIC API: In-Game Feature Calculation
# ---------------------------------------------------------------------------

def calculate_cumulative_in_game_features(
    events: pd.DataFrame,
    team_id: int,
    team_side: str,
    pred_min: int,
) -> torch.Tensor:
    """Calculate 22 in-game features for the cumulative window (0 to pred_min).

    ALL 22 features computed from events up to pred_min to prevent data leakage.
    (features.csv contains full-match aggregates which leak future data for pred_min < 90)

    Returns: Tensor of shape (1, 22)
    """
    time_col = _get_time_col(events)
    if not events.empty:
        team_events = events[
            (events['teamId'] == team_id) &
            (events[time_col] <= pred_min)
        ]
    else:
        team_events = pd.DataFrame()

    # 16 features from event type counts (same method as interval features)
    feat_16 = []
    for col in IN_GAME_FEATURE_COLS:
        event_types = EVENT_TYPE_MAP.get(col, [])
        if not team_events.empty and 'type' in team_events.columns:
            count = int(team_events['type'].isin(event_types).sum())
        else:
            count = 0
        feat_16.append(float(count))

    # 6 event-based features
    feat_6 = _compute_6_event_features(team_events)

    all_features = feat_16 + feat_6  # 22 features
    return torch.tensor([all_features], dtype=torch.float)


def calculate_interval_in_game_features(
    events: pd.DataFrame,
    team_id: int,
    start_min: int,
    end_min: int,
) -> torch.Tensor:
    """Calculate 22 in-game features for a specific interval [start_min, end_min).

    ALL 22 features computed from events within the interval only.
    - 16 features: counted from event types within interval
    - 6 features: computed from pass/qualifier data within interval

    Returns: Tensor of shape (1, 22)
    """
    time_col = _get_time_col(events)

    if not events.empty:
        interval_events = events[
            (events['teamId'] == team_id) &
            (events[time_col] >= start_min) &
            (events[time_col] < end_min)
        ]
    else:
        interval_events = pd.DataFrame()

    # 16 features from event type counts
    feat_16 = []
    for col in IN_GAME_FEATURE_COLS:
        event_types = EVENT_TYPE_MAP.get(col, [])
        if not interval_events.empty and 'type' in interval_events.columns:
            count = int(interval_events['type'].isin(event_types).sum())
        else:
            count = 0
        feat_16.append(float(count))

    # 6 event-based features
    feat_6 = _compute_6_event_features(interval_events)

    all_features = feat_16 + feat_6  # 22 features
    return torch.tensor([all_features], dtype=torch.float)


# ---------------------------------------------------------------------------
# PUBLIC API: Full Match Processing
# ---------------------------------------------------------------------------

def process_match(
    events: pd.DataFrame,
    players: pd.DataFrame,
    positions: pd.DataFrame,
    features_df: pd.DataFrame,
    label: int,
    pred_min: int,
    N: int = 5,
    k: int = 2,
) -> Optional[Dict]:
    """Process a single match: create all temporal graphs and features.

    Args:
        events: All events up to pred_min (passes extracted via type='Pass')
        players: Player metadata (all players)
        positions: Player positions (avg_x, avg_y from player_positions.csv)
        features_df: Match features (single row)
        label: Match result label (0=home, 1=away, 2=draw)
        pred_min: Prediction minute
        N: Interval width in minutes
        k: Number of historical intervals

    Returns:
        Dict with all graph data and features, or None if processing failed.
    """
    if features_df.empty:
        return None

    # Validate interval configuration
    earliest_start = pred_min - k * N
    if earliest_start < 0:
        return None

    home_team_id = int(features_df.iloc[0]['home_team_id'])
    away_team_id = int(features_df.iloc[0]['away_team_id'])

    # --- Cumulative Graphs (7 node features, with rating) ---
    home_cum = create_cumulative_graph(events, players, positions, home_team_id, 'home', pred_min)
    away_cum = create_cumulative_graph(events, players, positions, away_team_id, 'away', pred_min)

    if home_cum is None or away_cum is None:
        return None

    # --- Interval Graphs (6 node features, no rating) ---
    home_intervals = []
    away_intervals = []
    for i in range(k):
        start = pred_min - (k - i) * N
        end = pred_min - (k - i - 1) * N

        h_int = create_interval_graph(events, players, home_team_id, 'home', start, end)
        a_int = create_interval_graph(events, players, away_team_id, 'away', start, end)

        if h_int is None or a_int is None:
            return None

        home_intervals.append(h_int)
        away_intervals.append(a_int)

    # --- Cumulative In-Game Features (22) ---
    home_cum_feat = calculate_cumulative_in_game_features(events, home_team_id, 'home', pred_min)
    away_cum_feat = calculate_cumulative_in_game_features(events, away_team_id, 'away', pred_min)

    # --- Interval In-Game Features (22 each) ---
    home_int_feats = []
    away_int_feats = []
    for i in range(k):
        start = pred_min - (k - i) * N
        end = pred_min - (k - i - 1) * N

        h_feat = calculate_interval_in_game_features(events, home_team_id, start, end)
        a_feat = calculate_interval_in_game_features(events, away_team_id, start, end)

        home_int_feats.append(h_feat)
        away_int_feats.append(a_feat)

    return {
        'home_cum_graph': home_cum,
        'away_cum_graph': away_cum,
        'home_interval_graphs': home_intervals,
        'away_interval_graphs': away_intervals,
        'home_cum_features': home_cum_feat,         # (1, 22)
        'away_cum_features': away_cum_feat,         # (1, 22)
        'home_interval_features': home_int_feats,   # List of k × (1, 22)
        'away_interval_features': away_int_feats,   # List of k × (1, 22)
        'label': label,
    }


def get_fibonacci_intervals(pred_min: int, k: int = 4) -> List[Tuple[int, int]]:
    """Generate Fibonacci-based interval boundaries.
    
    Fibonacci sequence for intervals: 2, 3, 5, 8, 13, 21, ...
    Intervals grow larger as we go further back in time.
    
    Args:
        pred_min: Prediction minute (e.g., 90)
        k: Number of intervals (default=4)
    
    Returns:
        List of (start_min, end_min) tuples, ordered from oldest to newest.
        Example for pred_min=90, k=4:
            [(72, 80), (80, 85), (85, 88), (88, 90)]
            Intervals: 8, 5, 3, 2 (reversed Fibonacci)
    
    Example calculation for 90 min, k=4:
        - Interval 4 (newest): 90-88 = 2 min
        - Interval 3: 88-85 = 3 min
        - Interval 2: 85-80 = 5 min
        - Interval 1 (oldest): 80-72 = 8 min
    """
    # Generate Fibonacci sequence: 2, 3, 5, 8, 13, 21, ...
    fib = [2, 3]
    while len(fib) < k:
        fib.append(fib[-1] + fib[-2])
    
    # Take first k values and reverse (newest interval = smallest)
    interval_widths = fib[:k][::-1]  # e.g., [8, 5, 3, 2] for k=4
    
    # Calculate interval boundaries from pred_min backwards
    intervals = []
    current_end = pred_min
    
    for width in interval_widths:
        start = current_end - width
        intervals.append((start, current_end))
        current_end = start
    
    # Return in chronological order (oldest first)
    return intervals[::-1]


def process_match_fibonacci(
    events: pd.DataFrame,
    players: pd.DataFrame,
    positions: pd.DataFrame,
    features_df: pd.DataFrame,
    label: int,
    pred_min: int,
    k: int = 4,
) -> Optional[Dict]:
    """Process a single match with Fibonacci-based interval widths.
    
    Unlike process_match which uses fixed N-minute intervals, this function
    uses growing intervals following the Fibonacci sequence (2, 3, 5, 8, ...).
    
    For pred_min=90 with k=4:
        - Interval 1: 72-80 (8 min) - oldest
        - Interval 2: 80-85 (5 min)
        - Interval 3: 85-88 (3 min)
        - Interval 4: 88-90 (2 min) - newest
    
    Args:
        events: All events up to pred_min
        players: Player metadata
        positions: Player positions
        features_df: Match features (single row)
        label: Match result label (0=home, 1=away, 2=draw)
        pred_min: Prediction minute
        k: Number of Fibonacci intervals (default=4)
    
    Returns:
        Dict with all graph data and features, or None if processing failed.
    """
    if features_df.empty:
        return None

    # Get Fibonacci intervals
    intervals = get_fibonacci_intervals(pred_min, k)
    
    # Validate: earliest interval must not start before minute 0
    if intervals[0][0] < 0:
        return None

    home_team_id = int(features_df.iloc[0]['home_team_id'])
    away_team_id = int(features_df.iloc[0]['away_team_id'])

    # --- Cumulative Graphs (7 node features, with rating) ---
    home_cum = create_cumulative_graph(events, players, positions, home_team_id, 'home', pred_min)
    away_cum = create_cumulative_graph(events, players, positions, away_team_id, 'away', pred_min)

    if home_cum is None or away_cum is None:
        return None

    # --- Interval Graphs (6 node features, no rating) ---
    home_intervals = []
    away_intervals = []
    
    for start, end in intervals:
        h_int = create_interval_graph(events, players, home_team_id, 'home', start, end)
        a_int = create_interval_graph(events, players, away_team_id, 'away', start, end)

        if h_int is None or a_int is None:
            return None

        home_intervals.append(h_int)
        away_intervals.append(a_int)

    # --- Cumulative In-Game Features (22) ---
    home_cum_feat = calculate_cumulative_in_game_features(events, home_team_id, 'home', pred_min)
    away_cum_feat = calculate_cumulative_in_game_features(events, away_team_id, 'away', pred_min)

    # --- Interval In-Game Features (22 each) ---
    home_int_feats = []
    away_int_feats = []
    
    for start, end in intervals:
        h_feat = calculate_interval_in_game_features(events, home_team_id, start, end)
        a_feat = calculate_interval_in_game_features(events, away_team_id, start, end)

        home_int_feats.append(h_feat)
        away_int_feats.append(a_feat)

    return {
        'home_cum_graph': home_cum,
        'away_cum_graph': away_cum,
        'home_interval_graphs': home_intervals,
        'away_interval_graphs': away_intervals,
        'home_cum_features': home_cum_feat,         # (1, 22)
        'away_cum_features': away_cum_feat,         # (1, 22)
        'home_interval_features': home_int_feats,   # List of k × (1, 22)
        'away_interval_features': away_int_feats,   # List of k × (1, 22)
        'label': label,
        'interval_config': 'fibonacci',             # Mark as Fibonacci intervals
        'intervals': intervals,                      # Store actual intervals used
    }


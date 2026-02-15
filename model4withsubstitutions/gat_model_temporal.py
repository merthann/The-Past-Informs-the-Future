"""
Temporal GAT Model for Model 4 (Sliding Window Architecture)

Architecture:
  - Separate GATEncoder for cumulative (7-dim input) and interval (6-dim input) graphs
  - Shared interval GATEncoder across all k intervals
  - Fusion layer concatenating all graph embeddings + all in-game features
  - FC classifier -> 3-class softmax (Home / Away / Draw)

Dimensions (for k=2, hidden=128):
  graph_dim  = hidden * (1 + k) * 2  = 128 * 3 * 2 = 768
  feature_dim = 22 * (1 + k) * 2     = 22 * 3 * 2  = 132
  total_input = 900 -> fusion(128) -> classifier(3)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Batch
from typing import Dict, List


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GATEncoder(nn.Module):
    """3-layer Graph Attention Network encoder.

    Architecture:
      GATConv(input -> hidden) -> ELU -> Dropout
      GATConv(hidden -> hidden) -> ELU -> Dropout
      GATConv(hidden -> hidden) -> ELU -> Dropout
      global_mean_pool -> (batch_size, hidden)
    """

    def __init__(self, input_size: int, hidden_channels: int = 128, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GATConv(input_size, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, input_size]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment vector [num_nodes] (None for single graph)

        Returns:
            Graph-level embedding [batch_size, hidden_channels]
        """
        x = self.dropout(self.elu(self.conv1(x, edge_index)))
        x = self.dropout(self.elu(self.conv2(x, edge_index)))
        x = self.dropout(self.elu(self.conv3(x, edge_index)))

        if batch is not None:
            return global_mean_pool(x, batch)
        return x.mean(dim=0, keepdim=True)


class TemporalMatchPredictor(nn.Module):
    """Model 4: Temporal Sliding Window Match Outcome Predictor.

    Processes (1 + k) temporal windows per team:
      - 1 cumulative graph (0 to t) with 7 node features
      - k interval graphs (t-kN to t-N) with 6 node features each

    Each window also has 22 in-game features per team.

    Args:
        cumulative_input_size: Node feature dimension for cumulative graph (default: 7)
        interval_input_size: Node feature dimension for interval graphs (default: 6)
        hidden_channels: GAT hidden/output dimension (default: 128)
        k: Number of historical intervals (default: 2)
        in_game_features: Number of in-game features per team per window (default: 22)
        num_classes: Output classes (default: 3)
        dropout: Dropout rate (default: 0.5)
    """

    def __init__(
        self,
        cumulative_input_size: int = 7,
        interval_input_size: int = 6,
        hidden_channels: int = 128,
        k: int = 2,
        in_game_features: int = 22,
        num_classes: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.k = k
        self.hidden_channels = hidden_channels

        # Separate GAT encoders for cumulative and interval graphs
        self.cumulative_gat = GATEncoder(cumulative_input_size, hidden_channels, dropout)
        self.interval_gat = GATEncoder(interval_input_size, hidden_channels, dropout)

        # Fusion dimensions
        # Graph: hidden_channels per graph × (1 cumulative + k intervals) × 2 (home + away)
        graph_dim = hidden_channels * (1 + k) * 2
        # Features: 22 per window × (1 cumulative + k intervals) × 2 (home + away)
        feature_dim = in_game_features * (1 + k) * 2

        total_input = graph_dim + feature_dim

        # Fusion + Classification layers
        self.fusion = nn.Linear(total_input, hidden_channels)
        self.elu = nn.ELU()
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for m in [self.fusion, self.classifier]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, batch_dict: Dict) -> torch.Tensor:
        """
        Forward pass with temporal sliding window data.

        Args:
            batch_dict: Dictionary containing:
                - home_cum_graph: Batch (cumulative home graphs)
                - away_cum_graph: Batch (cumulative away graphs)
                - home_interval_graphs: List[Batch] (k interval home graphs)
                - away_interval_graphs: List[Batch] (k interval away graphs)
                - home_cum_features: Tensor [B, 22]
                - away_cum_features: Tensor [B, 22]
                - home_interval_features: List[Tensor] (k × [B, 22])
                - away_interval_features: List[Tensor] (k × [B, 22])

        Returns:
            Raw logits [B, 3] (apply softmax for probabilities during inference)
        """
        # --- Process Cumulative Graphs ---
        hc = batch_dict['home_cum_graph']
        ac = batch_dict['away_cum_graph']

        home_cum_emb = self.cumulative_gat(hc.x, hc.edge_index, hc.batch)  # [B, hidden]
        away_cum_emb = self.cumulative_gat(ac.x, ac.edge_index, ac.batch)  # [B, hidden]

        # --- Process Interval Graphs (shared encoder) ---
        home_int_embs = []
        away_int_embs = []
        for i in range(self.k):
            hi = batch_dict['home_interval_graphs'][i]
            ai = batch_dict['away_interval_graphs'][i]
            home_int_embs.append(self.interval_gat(hi.x, hi.edge_index, hi.batch))
            away_int_embs.append(self.interval_gat(ai.x, ai.edge_index, ai.batch))

        # --- Concatenate All Graph Embeddings ---
        # Order: home_cum, away_cum, home_int_0, away_int_0, home_int_1, away_int_1, ...
        graph_parts = [home_cum_emb, away_cum_emb]
        for h_emb, a_emb in zip(home_int_embs, away_int_embs):
            graph_parts.append(h_emb)
            graph_parts.append(a_emb)

        # --- Concatenate All In-Game Features ---
        feat_parts = [
            batch_dict['home_cum_features'],
            batch_dict['away_cum_features'],
        ]
        for h_feat, a_feat in zip(
            batch_dict['home_interval_features'],
            batch_dict['away_interval_features'],
        ):
            feat_parts.append(h_feat)
            feat_parts.append(a_feat)

        # --- Fusion ---
        x = torch.cat(graph_parts + feat_parts, dim=1)
        x = self.dropout_layer(self.elu(self.fusion(x)))
        x = self.classifier(x)
        return x  # Raw logits; CrossEntropyLoss applies log_softmax internally

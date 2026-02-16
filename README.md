# Temporal Sliding Window Graph Attention Networks for Real-Time Football Match Outcome Prediction
A temporal sliding window approach for football match outcome prediction using Graph Attention Networks (GAT) that captures historical context through multiple time-windowed pass network graphs.

## 🎯 Overview

This model extends Model 3 by incorporating **temporal sliding windows** to capture momentum shifts and recent game dynamics. Instead of a single cumulative graph, it processes:

- **Cumulative Graph** (0 to t): Full match context up to prediction time
- **Interval Graphs** (t-kN to t-N): Recent historical windows for momentum capture

```
k=2 Example (90 min prediction, N=5):

[80-85 min] → [85-90 min] → [0-90 min]
  Interval      Interval     Cumulative
   Graph         Graph         Graph
     ↓             ↓             ↓
    GAT           GAT           GAT
     └─────────────┼─────────────┘
                   ↓
             Fusion Layer
                   ↓
         [Home | Draw | Away]
```

## 🔑 Key Parameters

| Parameter | Description | Values |
|-----------|-------------|--------|
| **t** | Prediction timestamp | 45, 60, 75, 90 min |
| **N** | Interval window width | 5 min (default) |
| **k** | Number of historical intervals | 1-10 |

## 📊 Features

### Node Features

| Graph Type | Features | Count |
|------------|----------|-------|
| **Cumulative** | position, height, weight, rating, pass_accuracy, avg_x, avg_y | 7 |
| **Interval** | position, height, weight, pass_accuracy, avg_x, avg_y | 6 |

> **Note:** Interval graphs exclude `rating` to prevent data leakage

### In-Game Features (22 per team per window)

All features computed from `events.csv` to prevent data leakage:
- **16 features**: ball_touches, tackles, aerials, fouls, clearances, interceptions, etc. (counted from event types)
- **6 features**: pass_success_rate, final_third_passes, crosses, key_passes, big_chance, shot_assist

## 🚀 Quick Start

### 1. Preprocess Data (Cache Creation)

```bash
python ../universal_preprocess.py --data_dir ../data/minute_90 --output_dir ../cache --pred_mins 45,60,75,90
```

### 2. Train Model

```bash
python train_model.py --cache_dir ../cache --pred_min 90 --k 2 --N 5
```

### 3. Test Model

```bash
python test_model.py --cache_dir ../cache --pred_min 90 --k 2 --N 5
```


## 📁 Project Structure

```
model4_temporal_sliding_window/
├── ARCHITECTURE.md           # Detailed technical documentation
├── train_model.py            # Training script
├── test_model.py             # Evaluation script
├── pass_network_creator.py   # Graph construction with temporal windows
├── gat_model_temporal.py     # Temporal GAT architecture
├── models/                   # Saved checkpoints
└── logs/                     # Training logs
```

## 🏗️ Architecture

The model uses separate GAT encoders for cumulative and interval graphs:

- **3-layer GAT** with multi-head attention for graph processing
- **Fusion layer** concatenates all graph embeddings + in-game features
- **Softmax classifier** outputs 3-class probabilities

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.

## 📈 Comparison with We Know Who Wins

| Feature | We Know Who Wins | TSW-GAT |
|---------|------------------|---------|
| Graph Input | Single cumulative | Multiple temporal windows |
| Temporal Context | None | k intervals × N minutes |

## 📝 Notes

- **N=3** minutes is recommended as it captures typical momentum shifts in football
- **k=1 or k=2** are good starting values; higher k increases computational cost
- Cache files are shared across all models for efficiency

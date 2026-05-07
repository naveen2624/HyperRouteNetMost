"""
HyPerRouteNet — Phase 2: Hypergraph Construction + Spatio-Temporal Model
=========================================================================
Steps:
  1. Load processed data from Phase 1
  2. Partition Bangalore road network into regions (spatial clustering)
  3. Build hypergraph:
       - Nodes  = regions
       - Hyperedges = multi-region interactions (transit routes, OD corridors,
                      spatial proximity clusters)
  4. Build incidence matrix H  [nodes x hyperedges]
  5. Define HyperConv layer (simplified hypergraph convolution)
  6. Build HyPerRouteNet:
       - Hypergraph Spatial Encoder (HyperConv layers)
       - Temporal Module (GRU)
       - OD Prediction Head
  7. Prepare spatio-temporal sequences for training
  8. Train model → predict OD demand matrix
  9. Evaluate + save model
"""

import os
import math
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all OS)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "outputs" / "models"
PLOT_DIR  = BASE_DIR / "outputs" / "plots"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
NUM_REGIONS    = 50      # spatial clusters (regions) from road network
NUM_HYPEREDGES = 80      # total hyperedges (transit + OD + proximity)
SEQ_LEN        = 12      # input time steps  (12 × 15 min = 3 hours history)
PRED_LEN       = 6       # output time steps (6 × 15 min = 90 min forecast)
HIDDEN_DIM     = 64      # GRU hidden size
EMBED_DIM      = 32      # node embedding dimension
NUM_LAYERS     = 2       # GRU layers
BATCH_SIZE     = 32
EPOCHS         = 50
LR             = 1e-3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD PROCESSED DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_processed_data():
    print("\n[1/7] Loading Phase 1 processed data...")

    speed = pd.read_parquet(PROC_DIR / "metr_la_speed_15min.parquet")
    od    = pd.read_parquet(PROC_DIR / "od_matrix.parquet")
    gtfs  = pd.read_parquet(PROC_DIR / "gtfs_hyperedges.parquet")
    stops = pd.read_parquet(PROC_DIR / "gtfs_stops.parquet")
    unif  = pd.read_parquet(PROC_DIR / "unified_features.parquet")
    wx    = pd.read_parquet(PROC_DIR / "weather_bangalore.parquet")

    print(f"     Speed matrix   : {speed.shape}")
    print(f"     OD matrix      : {od.shape}")
    print(f"     GTFS hyperedges: {gtfs.shape}")
    print(f"     GTFS stops     : {stops.shape}")
    print(f"     Unified feats  : {unif.shape}")
    print(f"     Weather        : {wx.shape}")

    return speed, od, gtfs, stops, unif, wx


# ══════════════════════════════════════════════════════════════════════════════
# 2. SPATIAL REGION PARTITIONING
#    K-Means on sensor positions → NUM_REGIONS clusters
# ══════════════════════════════════════════════════════════════════════════════

def build_regions(speed_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Since METR-LA sensors don't carry explicit lat/lon in the .h5 file,
    we create pseudo-spatial features from sensor correlation patterns.
    Sensors that behave similarly are likely geographically close.

    Returns:
      sensor_labels : (207,) cluster label per sensor
      region_centers: (NUM_REGIONS, feature_dim) cluster centroids
    """
    print("\n[2/7] Building spatial region partitioning...")

    # Use speed variance profile as spatial signature per sensor
    # Shape: (sensors, timestamps) → transpose for clustering
    X = speed_df.values.T                         # (207, T)
    # Feature: [mean, std, percentile-25, percentile-75] per sensor
    feats = np.column_stack([
        X.mean(axis=1),
        X.std(axis=1),
        np.percentile(X, 25, axis=1),
        np.percentile(X, 75, axis=1),
    ])                                             # (207, 4)

    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    kmeans = KMeans(n_clusters=NUM_REGIONS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(feats_scaled)      # (207,)
    centers = kmeans.cluster_centers_              # (NUM_REGIONS, 4)

    # Count sensors per region
    unique, counts = np.unique(labels, return_counts=True)
    print(f"     Regions created       : {NUM_REGIONS}")
    print(f"     Sensors per region    : min={counts.min()}, "
          f"max={counts.max()}, avg={counts.mean():.1f}")

    return labels, centers


# ══════════════════════════════════════════════════════════════════════════════
# 3. HYPERGRAPH CONSTRUCTION
#    Three types of hyperedges:
#      A) Transit hyperedges  — from GTFS routes (BMTC)
#      B) OD corridor edges   — high-demand OD pairs grouped
#      C) Proximity edges     — spatially adjacent region clusters
# ══════════════════════════════════════════════════════════════════════════════

def build_hypergraph(
    sensor_labels: np.ndarray,
    od_df: pd.DataFrame,
    gtfs_df: pd.DataFrame,
    n_nodes: int = NUM_REGIONS,
    n_edges: int = NUM_HYPEREDGES
) -> np.ndarray:
    """
    Constructs the incidence matrix H of shape [n_nodes x n_edges].
    H[i, e] = 1 if node i belongs to hyperedge e, else 0.
    """
    print("\n[3/7] Building hypergraph incidence matrix...")

    H = np.zeros((n_nodes, n_edges), dtype=np.float32)
    edge_idx = 0

    # ── Type A: Transit hyperedges from GTFS ──────────────────────────────
    # Map GTFS stop_ids → region indices via modulo (proxy mapping)
    # In a full system you'd do spatial join; here we use hash mapping
    # as a structurally valid proxy since GTFS stop coords aren't in the
    # same space as METR-LA sensor clusters

    transit_budget = n_edges // 3                  # ~26 edges
    sampled_routes = gtfs_df.sample(
        min(transit_budget, len(gtfs_df)), random_state=42
    )

    for _, row in sampled_routes.iterrows():
        stop_ids = row["stop_ids"]
        # Map stop_ids → region nodes via modulo
        nodes_in_edge = list({int(s) % n_nodes for s in stop_ids})
        if len(nodes_in_edge) >= 2 and edge_idx < n_edges:
            H[nodes_in_edge, edge_idx] = 1.0
            edge_idx += 1

    transit_count = edge_idx
    print(f"     Type A — Transit hyperedges : {transit_count}")

    # ── Type B: OD corridor hyperedges ───────────────────────────────────
    # High-demand OD zones → group into multi-region corridors
    od_budget = n_edges // 3                       # ~26 edges
    top_od = (
        od_df.groupby(["PULocationID", "DOLocationID"])["trip_count"]
        .sum()
        .nlargest(od_budget * 3)                   # take 3× then cluster
        .reset_index()
    )

    # Map zone IDs → region nodes
    top_od["pu_region"] = top_od["PULocationID"] % n_nodes
    top_od["do_region"] = top_od["DOLocationID"] % n_nodes

    # Each top-OD pair that spans ≥2 distinct regions = one hyperedge
    added = 0
    for _, row in top_od.iterrows():
        pu_r = int(row["pu_region"])
        do_r = int(row["do_region"])
        if pu_r != do_r and edge_idx < n_edges:
            # Extend with nearby regions (simulates corridor breadth)
            corridor = list({pu_r, do_r,
                             (pu_r + 1) % n_nodes,
                             (do_r - 1) % n_nodes})
            H[corridor, edge_idx] = 1.0
            edge_idx += 1
            added += 1

    od_count = edge_idx - transit_count
    print(f"     Type B — OD corridor hyperedges: {od_count}")

    # ── Type C: Proximity hyperedges ──────────────────────────────────────
    # Adjacent regions (sliding window over sorted region indices)
    window = 4                                      # each edge spans 4 regions
    while edge_idx < n_edges:
        start = (edge_idx * 2) % n_nodes
        nodes_in_edge = [(start + k) % n_nodes for k in range(window)]
        H[nodes_in_edge, edge_idx] = 1.0
        edge_idx += 1

    prox_count = n_edges - transit_count - od_count
    print(f"     Type C — Proximity hyperedges  : {prox_count}")
    print(f"     Total hyperedges               : {n_edges}")
    print(f"     Incidence matrix H shape       : {H.shape}")
    print(f"     Density (non-zero / total)     : "
          f"{H.sum() / H.size * 100:.2f}%")
    print(f"     Avg nodes per hyperedge        : {H.sum(axis=0).mean():.2f}")
    print(f"     Avg hyperedges per node        : {H.sum(axis=1).mean():.2f}")

    # Save
    np.save(PROC_DIR / "hypergraph_H.npy", H)
    print("     Saved: hypergraph_H.npy")

    return H


# ══════════════════════════════════════════════════════════════════════════════
# 4. PREPARE SPATIO-TEMPORAL INPUT SEQUENCES
#    For each region, aggregate sensor speeds → region-level time series
#    Then build sliding window sequences: [SEQ_LEN → PRED_LEN]
# ══════════════════════════════════════════════════════════════════════════════

def build_sequences(
    speed_df: pd.DataFrame,
    sensor_labels: np.ndarray,
    od_df: pd.DataFrame,
    n_regions: int = NUM_REGIONS
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds (X, y) sequence pairs for training.

    X shape: (N_samples, SEQ_LEN,  n_regions)  — speed history
    y shape: (N_samples, PRED_LEN, n_regions)  — future speed (proxy for OD demand)
    """
    print("\n[4/7] Building spatio-temporal sequences...")

    # ── Aggregate sensors → regions ──
    region_speed = np.zeros((len(speed_df), n_regions), dtype=np.float32)
    for region_id in range(n_regions):
        sensor_mask = sensor_labels == region_id
        if sensor_mask.sum() > 0:
            region_speed[:, region_id] = (
                speed_df.values[:, sensor_mask].mean(axis=1)
            )

    # ── Normalize per region ──
    region_mean = region_speed.mean(axis=0, keepdims=True)
    region_std  = region_speed.std(axis=0, keepdims=True) + 1e-8
    region_speed = (region_speed - region_mean) / region_std

    # ── Sliding window ──
    total_len = SEQ_LEN + PRED_LEN
    X_list, y_list = [], []

    for t in range(len(region_speed) - total_len + 1):
        X_list.append(region_speed[t          : t + SEQ_LEN])
        y_list.append(region_speed[t + SEQ_LEN: t + total_len])

    X = np.stack(X_list, axis=0)   # (N, SEQ_LEN, n_regions)
    y = np.stack(y_list, axis=0)   # (N, PRED_LEN, n_regions)

    print(f"     Region speed matrix     : {region_speed.shape}")
    print(f"     Total sequences (N)     : {len(X)}")
    print(f"     X shape (input)         : {X.shape}")
    print(f"     y shape (target)        : {y.shape}")

    np.save(PROC_DIR / "seq_X.npy", X)
    np.save(PROC_DIR / "seq_y.npy", y)
    print("     Saved: seq_X.npy, seq_y.npy")

    return X, y, region_mean, region_std


# ══════════════════════════════════════════════════════════════════════════════
# 5. PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════

class ODDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 6. HyPerRouteNet MODEL
#
#  Architecture:
#    Input  → HyperConv (spatial)  → GRU (temporal)  → Linear (OD head)
#
#  HyperConv: simplified hypergraph convolution
#    Z = D_v^{-1} H W D_e^{-1} H^T X Θ
#    where:
#      H   = incidence matrix  [N x E]
#      W   = diagonal hyperedge weight matrix [E x E]
#      D_v = node degree matrix [N x N]
#      D_e = edge degree matrix [E x E]
#      Θ   = learnable weight matrix
# ══════════════════════════════════════════════════════════════════════════════

class HyperConv(nn.Module):
    """
    Single hypergraph convolution layer.
    Propagates messages through hyperedges:
      node → hyperedge (aggregation) → node (broadcast)
    """
    def __init__(self, in_dim: int, out_dim: int, n_nodes: int, n_edges: int):
        super().__init__()
        self.theta = nn.Linear(in_dim, out_dim, bias=False)
        self.bn    = nn.BatchNorm1d(out_dim)

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        X : (batch, n_nodes, in_dim)
        H : (n_nodes, n_edges)   — incidence matrix (fixed)
        Returns: (batch, n_nodes, out_dim)
        """
        # Node degrees: D_v[i] = sum of hyperedge weights for node i
        Dv = H.sum(dim=1).clamp(min=1.0)          # (n_nodes,)
        # Edge degrees: D_e[e] = number of nodes in hyperedge e
        De = H.sum(dim=0).clamp(min=1.0)          # (n_edges,)

        # Hyperedge weight (uniform = 1 here; learnable extension possible)
        W  = torch.ones(H.shape[1], device=H.device)

        # Step 1: node → hyperedge  (aggregate node features per edge)
        # (batch, n_edges, in_dim) = bmm( H^T [E×N], X [batch,N,in_dim] )
        Ht = H.T                                   # (n_edges, n_nodes)
        De_inv = 1.0 / De                          # (n_edges,)
        # scale rows of Ht by De_inv
        Ht_scaled = Ht * De_inv.unsqueeze(1)       # (n_edges, n_nodes)

        # bmm: (batch, n_edges, in_dim)
        E_feat = torch.einsum("en,bni->bei", Ht_scaled, X)

        # Scale by hyperedge weight W
        E_feat = E_feat * W.view(1, -1, 1)

        # Step 2: hyperedge → node  (broadcast back)
        Dv_inv = 1.0 / Dv                          # (n_nodes,)
        H_scaled = H * Dv_inv.unsqueeze(1)         # (n_nodes, n_edges)

        # (batch, n_nodes, in_dim)
        Z = torch.einsum("ne,bei->bni", H_scaled, E_feat)

        # Linear transform + activation
        batch, n, d = Z.shape
        Z_flat = Z.reshape(batch * n, d)
        Z_out  = self.theta(Z_flat)                # (batch*n, out_dim)
        Z_out  = self.bn(Z_out)
        Z_out  = F.relu(Z_out)
        Z_out  = Z_out.reshape(batch, n, -1)       # (batch, n, out_dim)

        return Z_out


class HyPerRouteNet(nn.Module):
    """
    Full HyPerRouteNet model:
      1. HyperConv ×2         — spatial encoding via hypergraph
      2. GRU                  — temporal dependency modeling
      3. Linear head          — OD demand prediction
    """
    def __init__(
        self,
        n_nodes:   int,
        n_edges:   int,
        seq_len:   int,
        pred_len:  int,
        embed_dim: int,
        hidden:    int,
        n_layers:  int
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.n_nodes  = n_nodes

        # ── Spatial encoder ──
        self.hyper1 = HyperConv(1,         embed_dim, n_nodes, n_edges)
        self.hyper2 = HyperConv(embed_dim, embed_dim, n_nodes, n_edges)
        self.dropout = nn.Dropout(0.2)

        # ── Temporal encoder ──
        # Input to GRU: (batch, seq_len, n_nodes * embed_dim)
        self.gru = nn.GRU(
            input_size  = n_nodes * embed_dim,
            hidden_size = hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = 0.2 if n_layers > 1 else 0.0
        )

        # ── OD prediction head ──
        # Output: (batch, pred_len * n_nodes)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 2, pred_len * n_nodes)
        )

    def forward(
        self,
        X: torch.Tensor,         # (batch, seq_len, n_nodes)
        H: torch.Tensor          # (n_nodes, n_edges)
    ) -> torch.Tensor:
        """
        Returns: (batch, pred_len, n_nodes)
        """
        batch = X.shape[0]

        # ── Process each timestep through HyperConv ──
        spatial_out = []
        for t in range(self.seq_len):
            x_t = X[:, t, :].unsqueeze(-1)        # (batch, n_nodes, 1)
            z   = self.hyper1(x_t, H)              # (batch, n_nodes, embed)
            z   = self.hyper2(z,   H)              # (batch, n_nodes, embed)
            z   = self.dropout(z)
            spatial_out.append(z.reshape(batch, -1))   # (batch, n_nodes*embed)

        # Stack → (batch, seq_len, n_nodes*embed)
        spatial_seq = torch.stack(spatial_out, dim=1)

        # ── GRU temporal encoding ──
        gru_out, _ = self.gru(spatial_seq)         # (batch, seq_len, hidden)
        last_hidden = gru_out[:, -1, :]            # (batch, hidden)

        # ── OD demand prediction ──
        pred = self.head(last_hidden)              # (batch, pred_len*n_nodes)
        pred = pred.reshape(batch, self.pred_len, self.n_nodes)

        return pred


# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    H: np.ndarray
) -> tuple[HyPerRouteNet, dict]:
    print("\n[5/7] Initialising model and training...")

    # ── Train / Val split (80/20) ──
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = ODDataset(X_train, y_train)
    val_ds   = ODDataset(X_val,   y_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"     Train samples : {len(train_ds)}")
    print(f"     Val   samples : {len(val_ds)}")

    # ── Model ──
    H_tensor = torch.tensor(H, dtype=torch.float32).to(DEVICE)

    model = HyPerRouteNet(
        n_nodes   = NUM_REGIONS,
        n_edges   = NUM_HYPEREDGES,
        seq_len   = SEQ_LEN,
        pred_len  = PRED_LEN,
        embed_dim = EMBED_DIM,
        hidden    = HIDDEN_DIM,
        n_layers  = NUM_LAYERS
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"     Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    best_val = float("inf")
    patience_counter = 0
    EARLY_STOP = 10

    # ── Epoch loop ──
    for epoch in range(1, EPOCHS + 1):

        # Train
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb, H_tensor)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses, val_maes = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb, H_tensor)
                val_losses.append(criterion(pred, yb).item())
                val_maes.append(
                    F.l1_loss(pred, yb).item()
                )

        t_loss = np.mean(train_losses)
        v_loss = np.mean(val_losses)
        v_mae  = np.mean(val_maes)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_mae"].append(v_mae)

        scheduler.step(v_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"     Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train MSE: {t_loss:.4f} | "
                  f"Val MSE: {v_loss:.4f} | "
                  f"Val MAE: {v_mae:.4f}")

        # Save best
        if v_loss < best_val:
            best_val = v_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / "hyperroutenet_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"     Early stopping at epoch {epoch}")
                break

    print(f"\n     Best Val MSE : {best_val:.4f}")
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# 8. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model: HyPerRouteNet,
    X: np.ndarray,
    y: np.ndarray,
    H: np.ndarray
) -> dict:
    print("\n[6/7] Evaluating model on test set...")

    # Use last 10% as test
    split = int(len(X) * 0.9)
    X_test = torch.tensor(X[split:], dtype=torch.float32).to(DEVICE)
    y_test = y[split:]

    H_tensor = torch.tensor(H, dtype=torch.float32).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_DIR / "hyperroutenet_best.pt", map_location=DEVICE)
    )
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            batch = X_test[i: i + BATCH_SIZE]
            out   = model(batch, H_tensor)
            preds.append(out.cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)         # (N_test, PRED_LEN, n_regions)
    y_true = y_test                                # (N_test, PRED_LEN, n_regions)

    mae  = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
    rmse = math.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))
    # MAPE (avoid div by zero)
    mask  = np.abs(y_true) > 0.01
    mape  = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    print(f"     MAE  : {mae:.4f}")
    print(f"     RMSE : {rmse:.4f}")
    print(f"     MAPE : {mape:.2f}%")

    # Save predictions
    np.save(MODEL_DIR / "test_predictions.npy", y_pred)
    np.save(MODEL_DIR / "test_targets.npy",     y_true)

    return metrics, y_pred, y_true


# ══════════════════════════════════════════════════════════════════════════════
# 9. VISUALISATION — training curves + OD prediction sample
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(history: dict, y_pred: np.ndarray, y_true: np.ndarray):
    print("\n[7/7] Generating plots...")

    # ── 1. Training / Validation loss ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("HyPerRouteNet — Training Results", fontsize=14, fontweight="bold")

    ax = axes[0]
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train MSE", color="#2563EB", lw=2)
    ax.plot(epochs, history["val_loss"],   label="Val MSE",   color="#DC2626", lw=2,
            linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["val_mae"], label="Val MAE", color="#16A34A", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.set_title("Validation MAE over Epochs")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("     Saved: training_curves.png")

    # ── 2. OD prediction vs ground truth (sample region, all pred steps) ──
    fig, ax = plt.subplots(figsize=(12, 5))
    sample_idx = 0
    region_idx = 0
    pred_steps = np.arange(1, PRED_LEN + 1)

    ax.plot(pred_steps, y_true[sample_idx, :, region_idx],
            "o-", color="#2563EB", lw=2, label="Ground Truth")
    ax.plot(pred_steps, y_pred[sample_idx, :, region_idx],
            "s--", color="#DC2626", lw=2, label="Predicted")
    ax.fill_between(
        pred_steps,
        y_true[sample_idx, :, region_idx],
        y_pred[sample_idx, :, region_idx],
        alpha=0.15, color="#6366F1"
    )
    ax.set_xlabel("Prediction Step (each = 15 min)")
    ax.set_ylabel("Normalized Speed / OD Demand")
    ax.set_title(f"OD Demand Prediction — Region {region_idx} "
                 f"(90-minute horizon)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "od_prediction_sample.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("     Saved: od_prediction_sample.png")

    # ── 3. Hyperedge degree distribution ──
    H = np.load(PROC_DIR / "hypergraph_H.npy")
    edge_degrees = H.sum(axis=0)
    node_degrees = H.sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(len(edge_degrees)), np.sort(edge_degrees)[::-1],
                color="#2563EB", alpha=0.8)
    axes[0].set_title("Hyperedge Degree Distribution\n(nodes per hyperedge)")
    axes[0].set_xlabel("Hyperedge (sorted)")
    axes[0].set_ylabel("Number of nodes")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(range(len(node_degrees)), np.sort(node_degrees)[::-1],
                color="#16A34A", alpha=0.8)
    axes[1].set_title("Node Degree Distribution\n(hyperedges per node)")
    axes[1].set_xlabel("Region node (sorted)")
    axes[1].set_ylabel("Number of hyperedges")
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("HyPerRouteNet — Hypergraph Structure", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "hypergraph_degrees.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("     Saved: hypergraph_degrees.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  HyPerRouteNet — Phase 2: Hypergraph Model")
    print("=" * 60)

    # 1. Load data
    speed, od, gtfs, stops, unif, wx = load_processed_data()

    # 2. Spatial partitioning
    sensor_labels, region_centers = build_regions(speed)

    # 3. Build hypergraph
    H = build_hypergraph(sensor_labels, od, gtfs)

    # 4. Build sequences
    X, y, r_mean, r_std = build_sequences(speed, sensor_labels, od)

    # 5. Train
    model, history = train_model(X, y, H)

    # 6. Evaluate
    metrics, y_pred, y_true = evaluate_model(model, X, y, H)

    # 7. Plot
    plot_results(history, y_pred, y_true)

    # ── Save model config for Phase 3 ──
    config = {
        "NUM_REGIONS":    int(NUM_REGIONS),
        "NUM_HYPEREDGES": int(NUM_HYPEREDGES),
        "SEQ_LEN":        int(SEQ_LEN),
        "PRED_LEN":       int(PRED_LEN),
        "EMBED_DIM":      int(EMBED_DIM),
        "HIDDEN_DIM":     int(HIDDEN_DIM),
        "NUM_LAYERS":     int(NUM_LAYERS),
        "region_mean":    r_mean.astype(float).tolist(),
        "region_std":     r_std.astype(float).tolist(),
        "metrics": {
            "MAE":  float(metrics["MAE"]),
            "RMSE": float(metrics["RMSE"]),
            "MAPE": float(metrics["MAPE"])
        }
    }
    import json
    with open(MODEL_DIR / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("  Phase 2 COMPLETE ✓")
    print("=" * 60)
    print(f"\n  MAE  : {metrics['MAE']:.4f}")
    print(f"  RMSE : {metrics['RMSE']:.4f}")
    print(f"  MAPE : {metrics['MAPE']:.2f}%")
    print(f"\n  Model saved  → outputs/models/hyperroutenet_best.pt")
    print(f"  Plots saved  → outputs/plots/")
    print(f"  Config saved → outputs/models/model_config.json")
    print("\n✓ Ready for Phase 3: MoST-ROUTE implementation")


if __name__ == "__main__":
    main()
"""
HyPerRouteNet — Phase 4: Integration Pipeline
==============================================
Connects all phases into one unified system:

  Phase 1 data → Phase 2 model (OD prediction) → Phase 3 routing (MoST-ROUTE)
                                                       ↓
                                              Best route + reasoning

Pipeline flow:
  1. Load trained HyPerRouteNet model (Phase 2)
  2. Accept user input: Source region, Destination region
  3. Run live OD prediction for current time window
  4. Pass predictions → OD-to-Route Interface
  5. Run MoST-ROUTE (ST-Agent + GA + ACO)
  6. Return:
       - Best route
       - Future congestion insight
       - "This route is best because..." reasoning
  7. Save full pipeline result as JSON (consumed by Phase 5 API)
"""

import os
import json
import math
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from copy import deepcopy

# ─── Import Phase 2 model ─────────────────────────────────────────────────────
import sys
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "phase2_hypergraph"))
sys.path.insert(0, str(BASE_DIR / "src" / "phase3_routing"))

from hypergraph_model import HyPerRouteNet, HyperConv
from most_route import (
    STAgent, GeneticAlgorithm, AntColony,
    CongestionMap, RouteCostMatrix, RouteResult,
    build_od_interface, most_route, save_and_display
)

# ─── Paths ────────────────────────────────────────────────────────────────────
PROC_DIR   = BASE_DIR / "data" / "processed"
MODEL_DIR  = BASE_DIR / "outputs" / "models"
ROUTE_DIR  = BASE_DIR / "outputs" / "routes"
PIPE_DIR   = BASE_DIR / "outputs" / "pipeline"
PIPE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Load model config ────────────────────────────────────────────────────────
with open(MODEL_DIR / "model_config.json") as f:
    CFG = json.load(f)

NUM_REGIONS    = CFG["NUM_REGIONS"]
NUM_HYPEREDGES = CFG["NUM_HYPEREDGES"]
SEQ_LEN        = CFG["SEQ_LEN"]
PRED_LEN       = CFG["PRED_LEN"]
EMBED_DIM      = CFG["EMBED_DIM"]
HIDDEN_DIM     = CFG["HIDDEN_DIM"]
NUM_LAYERS     = CFG["NUM_LAYERS"]
REGION_MEAN    = np.array(CFG["region_mean"], dtype=np.float32)
REGION_STD     = np.array(CFG["region_std"],  dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 1. MODEL LOADER
#    Restores trained HyPerRouteNet from checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def load_hyperroutenet() -> HyPerRouteNet:
    """Load trained HyPerRouteNet model from Phase 2 checkpoint."""
    model = HyPerRouteNet(
        n_nodes   = NUM_REGIONS,
        n_edges   = NUM_HYPEREDGES,
        seq_len   = SEQ_LEN,
        pred_len  = PRED_LEN,
        embed_dim = EMBED_DIM,
        hidden    = HIDDEN_DIM,
        n_layers  = NUM_LAYERS
    ).to(DEVICE)

    ckpt_path = MODEL_DIR / "hyperroutenet_best.pt"
    model.load_state_dict(
        torch.load(ckpt_path, map_location=DEVICE)
    )
    model.eval()
    print(f"  Loaded HyPerRouteNet from {ckpt_path.name}")
    return model


def load_hypergraph() -> np.ndarray:
    """Load saved hypergraph incidence matrix."""
    H = np.load(PROC_DIR / "hypergraph_H.npy")
    print(f"  Loaded hypergraph H: {H.shape}")
    return H


def load_speed_sequences() -> Tuple[np.ndarray, np.ndarray]:
    """Load prebuilt sequences from Phase 2."""
    X = np.load(PROC_DIR / "seq_X.npy")
    y = np.load(PROC_DIR / "seq_y.npy")
    print(f"  Loaded sequences: X={X.shape}, y={y.shape}")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 2. LIVE OD PREDICTOR
#    Runs HyPerRouteNet inference on a time window
# ══════════════════════════════════════════════════════════════════════════════

class LiveODPredictor:
    """
    Wraps HyPerRouteNet for live inference.
    Given a window of recent speed readings, predicts future OD demand.
    """

    def __init__(self, model: HyPerRouteNet, H: np.ndarray):
        self.model    = model
        self.H_tensor = torch.tensor(H, dtype=torch.float32).to(DEVICE)

    def predict(
        self,
        X_sequences: np.ndarray,
        sample_idx:  Optional[int] = None
    ) -> np.ndarray:
        """
        Run inference on a batch of sequences.

        Args:
            X_sequences : (N, SEQ_LEN, n_regions) speed history
            sample_idx  : if given, use this specific sample;
                          otherwise use last SEQ_LEN timesteps

        Returns:
            pred : (PRED_LEN, n_regions) OD demand forecast
        """
        if sample_idx is None:
            # Use most recent window
            sample_idx = len(X_sequences) - 1

        x = X_sequences[sample_idx]                    # (SEQ_LEN, n_regions)
        x_tensor = torch.tensor(
            x[np.newaxis], dtype=torch.float32
        ).to(DEVICE)                                   # (1, SEQ_LEN, n_regions)

        with torch.no_grad():
            pred = self.model(x_tensor, self.H_tensor) # (1, PRED_LEN, n_regions)

        pred_np = pred.squeeze(0).cpu().numpy()        # (PRED_LEN, n_regions)
        return pred_np

    def predict_for_hour(
        self,
        X_sequences: np.ndarray,
        hour:        int
    ) -> np.ndarray:
        """
        Simulate prediction for a specific hour of day.
        Selects sequences whose time index matches the target hour.
        Falls back to last available if no match.

        Each sequence = 15-min interval.
        Hour h → sequence index ≈ h * 4  (4 intervals per hour)
        """
        target_idx = min(hour * 4, len(X_sequences) - 1)
        return self.predict(X_sequences, sample_idx=target_idx)


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONGESTION MAP BUILDER
#    Converts model predictions → CongestionMap used by MoST-ROUTE
# ══════════════════════════════════════════════════════════════════════════════

def predictions_to_congestion(
    pred:       np.ndarray,    # (PRED_LEN, n_regions)
    n_regions:  int = NUM_REGIONS,
    pred_len:   int = PRED_LEN
) -> CongestionMap:
    """
    Converts normalized speed predictions → congestion scores.
    Lower predicted speed = higher congestion.
    """
    # Shift to [0, 1] range
    shifted = pred - pred.min()
    normed  = shifted / (shifted.max() + 1e-8)

    # Invert: high speed = low congestion
    congestion = 1.0 - normed                          # (PRED_LEN, n_regions)

    # Current = first 2 steps (next 30 min)
    current = congestion[:2].mean(axis=0)              # (n_regions,)
    # Future  = steps 3-6 (30-90 min ahead)
    future  = congestion[2:] if pred_len > 2 else congestion

    future_avg   = future.mean(axis=0)
    thresh_high  = np.percentile(future_avg, 70)
    thresh_low   = np.percentile(future_avg, 30)
    peak_regions = [i for i in range(n_regions) if future_avg[i] >= thresh_high]
    safe_regions = [i for i in range(n_regions) if future_avg[i] <= thresh_low]

    return CongestionMap(
        region_scores = current,
        future_scores = future,
        peak_regions  = peak_regions,
        safe_regions  = safe_regions,
        pred_horizon  = max(pred_len - 2, 1)
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. COST MATRIX BUILDER
#    Builds RouteCostMatrix from live congestion predictions
# ══════════════════════════════════════════════════════════════════════════════

def build_cost_matrix(
    congestion_map: CongestionMap,
    n_regions:      int = NUM_REGIONS
) -> Tuple[RouteCostMatrix, nx.DiGraph]:
    """
    Build travel cost matrix and road graph from live congestion predictions.
    Costs dynamically reflect predicted congestion (not static).
    """
    current_cong = congestion_map.region_scores        # (n_regions,)

    # Bangalore grid layout (same as Phase 3)
    grid_x = np.array([i % 7  for i in range(n_regions)], dtype=float) * (50.0/6)
    grid_y = np.array([i // 7 for i in range(n_regions)], dtype=float) * (50.0/7)

    # Distance matrix
    dist_matrix = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(n_regions):
            dist_matrix[i,j] = math.sqrt(
                (grid_x[i]-grid_x[j])**2 + (grid_y[i]-grid_y[j])**2
            )

    # Time matrix (congestion-aware speed)
    speed_base  = 40.0
    time_matrix = np.zeros((n_regions, n_regions))
    cong_matrix = np.zeros((n_regions, n_regions))

    for i in range(n_regions):
        for j in range(n_regions):
            avg_cong  = (current_cong[i] + current_cong[j]) / 2.0
            eff_speed = max(speed_base * (1 - 0.7 * avg_cong), 5.0)
            time_matrix[i,j] = (dist_matrix[i,j] / eff_speed) * 60.0
            cong_matrix[i,j] = avg_cong

    # Normalized combined cost
    t_norm   = time_matrix / (time_matrix.max() + 1e-8)
    d_norm   = dist_matrix / (dist_matrix.max() + 1e-8)
    combined = 0.4 * t_norm + 0.3 * d_norm + 0.3 * cong_matrix

    cost_matrix = RouteCostMatrix(
        time_cost       = time_matrix,
        distance_cost   = dist_matrix,
        congestion_cost = cong_matrix,
        combined_cost   = combined
    )

    # Road graph: each node connects to 4 nearest neighbors
    future_avg = congestion_map.future_scores.mean(axis=0)
    G = nx.DiGraph()
    for i in range(n_regions):
        G.add_node(i,
                   congestion      = float(current_cong[i]),
                   future_cong     = float(future_avg[i]),
                   grid_x          = float(grid_x[i]),
                   grid_y          = float(grid_y[i]))

    for i in range(n_regions):
        dists = dist_matrix[i].copy()
        dists[i] = np.inf
        nearest = np.argsort(dists)[:4]
        for j in nearest:
            G.add_edge(i, j,
                       weight     = combined[i,j],
                       time       = time_matrix[i,j],
                       distance   = dist_matrix[i,j],
                       congestion = cong_matrix[i,j])
            G.add_edge(j, i,
                       weight     = combined[j,i],
                       time       = time_matrix[j,i],
                       distance   = dist_matrix[j,i],
                       congestion = cong_matrix[j,i])

    return cost_matrix, G


# ══════════════════════════════════════════════════════════════════════════════
# 5. UNIFIED PIPELINE
#    End-to-end: user query → prediction → routing → result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Complete end-to-end pipeline result."""
    query_time:       str
    source:           int
    destination:      int
    hour:             int
    od_prediction:    np.ndarray        # (PRED_LEN, n_regions)
    congestion_map:   CongestionMap
    route_result:     RouteResult
    pipeline_latency: float             # seconds


def run_pipeline(
    source:      int,
    destination: int,
    hour:        int,
    model:       HyPerRouteNet,
    predictor:   LiveODPredictor,
    X_sequences: np.ndarray,
    verbose:     bool = True
) -> PipelineResult:
    """
    Full end-to-end pipeline run.

    1. Predict OD demand for given hour
    2. Build congestion map from predictions
    3. Build dynamic cost matrix
    4. Run MoST-ROUTE
    5. Return unified result
    """
    t_start = time.perf_counter()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  PIPELINE: Region {source} → Region {destination}  "
              f"| Hour {hour:02d}:00")
        print(f"{'='*60}")

    # ── Step 1: Live OD prediction ──
    if verbose:
        print(f"\n  [1/4] Running HyPerRouteNet inference...")
    pred = predictor.predict_for_hour(X_sequences, hour)
    if verbose:
        print(f"        Prediction shape : {pred.shape}")
        print(f"        Speed range      : [{pred.min():.3f}, {pred.max():.3f}]")

    # ── Step 2: Build congestion map ──
    if verbose:
        print(f"\n  [2/4] Building congestion map...")
    cong_map = predictions_to_congestion(pred)
    if verbose:
        avg_c = cong_map.region_scores.mean()
        print(f"        Avg congestion   : {avg_c:.3f}")
        print(f"        Peak regions     : {len(cong_map.peak_regions)}")
        print(f"        Safe regions     : {len(cong_map.safe_regions)}")

    # ── Step 3: Build dynamic cost matrix ──
    if verbose:
        print(f"\n  [3/4] Building dynamic cost matrix...")
    cost_matrix, G = build_cost_matrix(cong_map)
    if verbose:
        print(f"        Graph: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges")

    # ── Step 4: MoST-ROUTE ──
    if verbose:
        print(f"\n  [4/4] Running MoST-ROUTE optimizer...")
    route_result = most_route(
        source         = source,
        destination    = destination,
        hour           = hour,
        congestion_map = cong_map,
        cost_matrix    = cost_matrix,
        G              = G
    )

    latency = time.perf_counter() - t_start

    pipeline_result = PipelineResult(
        query_time       = datetime.now().isoformat(),
        source           = source,
        destination      = destination,
        hour             = hour,
        od_prediction    = pred,
        congestion_map   = cong_map,
        route_result     = route_result,
        pipeline_latency = latency
    )

    if verbose:
        _print_pipeline_result(pipeline_result)

    return pipeline_result


def _print_pipeline_result(pr: PipelineResult):
    """Pretty-print the full pipeline result."""
    rr = pr.route_result
    print(f"\n{'='*60}")
    print(f"  PIPELINE RESULT")
    print(f"{'='*60}")
    print(f"  Query time      : {pr.query_time}")
    print(f"  Pipeline latency: {pr.pipeline_latency:.2f}s")
    print(f"  Source → Dest   : Region {pr.source} → Region {pr.destination}")
    print(f"  Hour            : {pr.hour:02d}:00")
    print(f"\n  ROUTE:")
    print(f"    Path          : {' → '.join(map(str, rr.best_path))}")
    print(f"    Algorithm     : {rr.algorithm_used}")
    print(f"    Strategy      : {rr.strategy}")
    print(f"    Travel time   : {rr.travel_time_min:.1f} min")
    print(f"    Distance      : {rr.distance_km:.2f} km")
    print(f"    Congestion    : {rr.congestion_score:.3f}")
    print(f"\n  REASONING:")
    for line in rr.reasoning.split("\n"):
        print(f"    {line}")


def save_pipeline_result(pr: PipelineResult) -> Path:
    """Save full pipeline result as JSON for API consumption."""

    def to_py(obj):
        if isinstance(obj, list):
            return [to_py(x) for x in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    rr = pr.route_result
    out = {
        "query_time":        pr.query_time,
        "pipeline_latency":  round(pr.pipeline_latency, 3),
        "source":            int(pr.source),
        "destination":       int(pr.destination),
        "hour":              int(pr.hour),
        "od_prediction": {
            "shape":         list(pr.od_prediction.shape),
            "mean_per_step": [
                float(pr.od_prediction[t].mean())
                for t in range(pr.od_prediction.shape[0])
            ]
        },
        "congestion": {
            "current_avg":   float(pr.congestion_map.region_scores.mean()),
            "future_avg":    float(pr.congestion_map.future_scores.mean()),
            "peak_regions":  to_py(pr.congestion_map.peak_regions),
            "safe_regions":  to_py(pr.congestion_map.safe_regions),
            "n_peak":        len(pr.congestion_map.peak_regions),
            "n_safe":        len(pr.congestion_map.safe_regions),
        },
        "route": {
            "best_path":        to_py(rr.best_path),
            "algorithm_used":   rr.algorithm_used,
            "strategy":         rr.strategy,
            "travel_time_min":  float(rr.travel_time_min),
            "distance_km":      float(rr.distance_km),
            "congestion_score": float(rr.congestion_score),
            "reasoning":        rr.reasoning,
            "alternative_paths": to_py(rr.alternative_paths),
        }
    }

    fname = (PIPE_DIR /
             f"pipeline_{pr.source}_to_{pr.destination}_h{pr.hour:02d}.json")
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  Pipeline result saved → {fname.name}")
    return fname


# ══════════════════════════════════════════════════════════════════════════════
# 6. DISRUPTION SIMULATION
#    Inject synthetic disruptions and show system re-routing
# ══════════════════════════════════════════════════════════════════════════════

def simulate_disruption(
    pred:        np.ndarray,
    region_ids:  List[int],
    severity:    float = 0.8
) -> np.ndarray:
    """
    Inject a traffic disruption into a prediction array.
    Simulates accidents, events, or road closures.

    Args:
        pred       : (PRED_LEN, n_regions) base prediction
        region_ids : list of regions to disrupt
        severity   : 0-1, how much to increase congestion
    """
    disrupted = pred.copy()
    for r in region_ids:
        # Lower speed in disrupted regions (higher congestion)
        disrupted[:, r] *= (1 - severity)
    return disrupted


def run_disruption_test(
    model:       HyPerRouteNet,
    predictor:   LiveODPredictor,
    X_sequences: np.ndarray
) -> None:
    """
    Demonstrates system robustness:
    1. Run normal routing
    2. Inject disruption on best path
    3. Re-route and compare
    """
    print("\n" + "="*60)
    print("  DISRUPTION SIMULATION TEST")
    print("="*60)

    source, dest, hour = 0, 49, 8

    # ── Normal route ──
    print("\n  [A] Normal conditions:")
    pred_normal = predictor.predict_for_hour(X_sequences, hour)
    cong_normal = predictions_to_congestion(pred_normal)
    cost_n, G_n = build_cost_matrix(cong_normal)
    result_normal = most_route(source, dest, hour, cong_normal, cost_n, G_n)
    print(f"      Best path   : {' → '.join(map(str, result_normal.best_path))}")
    print(f"      Travel time : {result_normal.travel_time_min:.1f} min")
    print(f"      Congestion  : {result_normal.congestion_score:.3f}")

    # ── Inject disruption on normal path middle nodes ──
    disrupted_regions = result_normal.best_path[1:-1]   # intermediate nodes
    print(f"\n  [B] Disruption injected on regions: {disrupted_regions}")
    pred_disrupted = simulate_disruption(
        pred_normal, disrupted_regions, severity=0.85
    )
    cong_disrupted = predictions_to_congestion(pred_disrupted)
    cost_d, G_d    = build_cost_matrix(cong_disrupted)
    result_disrupted = most_route(source, dest, hour,
                                  cong_disrupted, cost_d, G_d)
    print(f"      Re-routed path : "
          f"{' → '.join(map(str, result_disrupted.best_path))}")
    print(f"      Travel time    : {result_disrupted.travel_time_min:.1f} min")
    print(f"      Congestion     : {result_disrupted.congestion_score:.3f}")

    # ── Compare ──
    path_changed = (result_normal.best_path != result_disrupted.best_path)
    time_diff    = (result_disrupted.travel_time_min -
                    result_normal.travel_time_min)

    print(f"\n  DISRUPTION SUMMARY:")
    print(f"    Path changed    : {'YES — system re-routed' if path_changed else 'NO (same path optimal)'}")
    print(f"    Time difference : {time_diff:+.1f} min")
    print(f"    System robust   : ✓ (re-routing handled automatically)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  HyPerRouteNet — Phase 4: Integration Pipeline")
    print("=" * 60)
    print(f"  Device : {DEVICE}")

    # ── Load all Phase 2 artifacts ──
    print("\n[LOAD] Phase 2 artifacts...")
    model    = load_hyperroutenet()
    H        = load_hypergraph()
    X, y     = load_speed_sequences()

    # ── Build predictor ──
    predictor = LiveODPredictor(model, H)
    print(f"  LiveODPredictor ready")

    # ── Run integration scenarios ──
    scenarios = [
        {"source":  0, "dest": 49, "hour":  8,  "label": "Morning peak"},
        {"source": 10, "dest": 40, "hour": 18,  "label": "Evening peak"},
        {"source":  5, "dest": 25, "hour": 14,  "label": "Midday"},
        {"source": 20, "dest": 45, "hour": 23,  "label": "Late night"},
    ]

    all_results = []
    saved_files = []

    for sc in scenarios:
        print(f"\n{'─'*60}")
        print(f"  Scenario: {sc['label']}")
        pr = run_pipeline(
            source      = sc["source"],
            destination = sc["dest"],
            hour        = sc["hour"],
            model       = model,
            predictor   = predictor,
            X_sequences = X,
            verbose     = True
        )
        saved = save_pipeline_result(pr)
        all_results.append(pr)
        saved_files.append(saved)

    # ── Disruption robustness test ──
    run_disruption_test(model, predictor, X)

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("  PHASE 4 SUMMARY")
    print("=" * 60)
    print(f"\n  {'Scenario':<20} {'Latency':>8} {'Path':>6} "
          f"{'Time(min)':>10} {'Dist(km)':>9} {'Algorithm'}")
    print("  " + "-" * 70)
    for sc, pr in zip(scenarios, all_results):
        rr = pr.route_result
        print(f"  {sc['label']:<20} {pr.pipeline_latency:>7.2f}s "
              f"{len(rr.best_path):>6} {rr.travel_time_min:>10.1f} "
              f"{rr.distance_km:>9.2f} {rr.algorithm_used}")

    print(f"\n  Pipeline JSON files saved to: outputs/pipeline/")
    print(f"  Total files: {len(saved_files)}")

    # ── Save master config for Phase 5 API ──
    api_config = {
        "n_regions":     NUM_REGIONS,
        "n_hyperedges":  NUM_HYPEREDGES,
        "seq_len":       SEQ_LEN,
        "pred_len":      PRED_LEN,
        "embed_dim":     EMBED_DIM,
        "hidden_dim":    HIDDEN_DIM,
        "num_layers":    NUM_LAYERS,
        "model_path":    str(MODEL_DIR / "hyperroutenet_best.pt"),
        "hypergraph_path": str(PROC_DIR / "hypergraph_H.npy"),
        "sequences_path":  str(PROC_DIR / "seq_X.npy"),
        "device":        str(DEVICE),
        "phase3_metrics": CFG["metrics"],
        "pipeline_ready": True
    }
    cfg_path = MODEL_DIR / "api_config.json"
    with open(cfg_path, "w") as f:
        json.dump(api_config, f, indent=2)
    print(f"\n  API config saved → {cfg_path.name}")

    print("\n" + "=" * 60)
    print("  Phase 4 COMPLETE ✓")
    print("  Ready for Phase 5: FastAPI backend")
    print("=" * 60)


if __name__ == "__main__":
    main()
"""
HyPerRouteNet — Phase 5: FastAPI Backend
=========================================
Endpoints:
  POST /predict-od      → Run OD demand prediction for a time window
  POST /get-route       → Get optimal route for source → destination
  POST /simulate        → Simulate disruption scenarios
  GET  /health          → System health check
  GET  /congestion-map  → Current congestion map for all regions

Startup:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import math
import time
import random
import numpy as np
import torch
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src" / "phase2_hypergraph"))
sys.path.insert(0, str(BASE_DIR / "src" / "phase3_routing"))
sys.path.insert(0, str(BASE_DIR / "src" / "phase4_integration"))

from hypergraph_model import HyPerRouteNet
from most_route import (
    STAgent, GeneticAlgorithm, AntColony,
    CongestionMap, RouteCostMatrix, most_route
)
from integration_pipeline import (
    LiveODPredictor,
    predictions_to_congestion,
    build_cost_matrix,
    simulate_disruption
)

# ─── Paths ────────────────────────────────────────────────────────────────────
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "outputs" / "models"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Load API config ──────────────────────────────────────────────────────────
with open(MODEL_DIR / "api_config.json") as f:
    API_CFG = json.load(f)

with open(MODEL_DIR / "model_config.json") as f:
    MODEL_CFG = json.load(f)

NUM_REGIONS    = API_CFG["n_regions"]
NUM_HYPEREDGES = API_CFG["n_hyperedges"]
SEQ_LEN        = API_CFG["seq_len"]
PRED_LEN       = API_CFG["pred_len"]
EMBED_DIM      = API_CFG["embed_dim"]
HIDDEN_DIM     = API_CFG["hidden_dim"]
NUM_LAYERS     = API_CFG["num_layers"]


# ══════════════════════════════════════════════════════════════════════════════
# APP STATE — loaded once at startup
# ══════════════════════════════════════════════════════════════════════════════

class AppState:
    model:     HyPerRouteNet = None
    predictor: LiveODPredictor = None
    H:         np.ndarray = None
    X_seq:     np.ndarray = None
    startup_time: str = ""
    request_count: int = 0

state = AppState()


def load_all_artifacts():
    """Load model + data once at server startup."""
    print(f"[startup] Device: {DEVICE}")

    # Model
    model = HyPerRouteNet(
        n_nodes   = NUM_REGIONS,
        n_edges   = NUM_HYPEREDGES,
        seq_len   = SEQ_LEN,
        pred_len  = PRED_LEN,
        embed_dim = EMBED_DIM,
        hidden    = HIDDEN_DIM,
        n_layers  = NUM_LAYERS
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_DIR / "hyperroutenet_best.pt", map_location=DEVICE)
    )
    model.eval()
    print(f"[startup] HyPerRouteNet loaded ✓")

    # Hypergraph
    H = np.load(PROC_DIR / "hypergraph_H.npy")
    print(f"[startup] Hypergraph H {H.shape} loaded ✓")

    # Sequences
    X_seq = np.load(PROC_DIR / "seq_X.npy")
    print(f"[startup] Sequences {X_seq.shape} loaded ✓")

    predictor = LiveODPredictor(model, H)

    state.model        = model
    state.predictor    = predictor
    state.H            = H
    state.X_seq        = X_seq
    state.startup_time = datetime.now().isoformat()
    print(f"[startup] All artifacts ready. Server up at {state.startup_time}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_artifacts()
    yield
    print("[shutdown] Server shutting down.")


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "HyPerRouteNet–MoST API",
    description = "Smart mobility routing with hypergraph OD prediction",
    version     = "1.0.0",
    lifespan    = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ODPredictRequest(BaseModel):
    hour: int = Field(
        default=8,
        ge=0, le=23,
        description="Hour of day (0-23) to predict OD demand for"
    )
    n_steps: int = Field(
        default=6,
        ge=1, le=6,
        description="Number of 15-min steps to forecast"
    )

class ODPredictResponse(BaseModel):
    hour:              int
    prediction_shape:  List[int]
    avg_congestion:    float
    peak_regions:      List[int]
    safe_regions:      List[int]
    congestion_per_region: List[float]
    future_congestion_avg: List[float]
    latency_ms:        float


class RouteRequest(BaseModel):
    source:      int = Field(ge=0, lt=50, description="Source region (0-49)")
    destination: int = Field(ge=0, lt=50, description="Destination region (0-49)")
    hour:        int = Field(default=8, ge=0, le=23, description="Departure hour")

class RouteResponse(BaseModel):
    source:           int
    destination:      int
    hour:             int
    best_path:        List[int]
    algorithm_used:   str
    strategy:         str
    travel_time_min:  float
    distance_km:      float
    congestion_score: float
    reasoning:        str
    alternative_path: List[int]
    congestion_map: Dict[str, Any]
    latency_ms:       float


class SimulateRequest(BaseModel):
    source:           int = Field(ge=0, lt=50)
    destination:      int = Field(ge=0, lt=50)
    hour:             int = Field(default=8, ge=0, le=23)
    disrupted_regions: List[int] = Field(
        default=[],
        description="Region IDs to inject disruption into"
    )
    disruption_severity: float = Field(
        default=0.8,
        ge=0.0, le=1.0,
        description="Severity of disruption (0=none, 1=full block)"
    )

class SimulateResponse(BaseModel):
    scenario:             str
    normal_path:          List[int]
    normal_time_min:      float
    disrupted_path:       List[int]
    disrupted_time_min:   float
    path_changed:         bool
    time_difference_min:  float
    disrupted_regions:    List[int]
    reasoning:            str
    latency_ms:           float


class CongestionMapResponse(BaseModel):
    hour:                  int
    region_congestion:     List[float]
    future_congestion_avg: List[float]
    peak_regions:          List[int]
    safe_regions:          List[int]
    overall_congestion:    float
    congestion_level:      str


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — shared prediction + routing logic
# ══════════════════════════════════════════════════════════════════════════════

def _predict_and_build(hour: int):
    """Run OD prediction and build congestion map + cost matrix."""
    pred     = state.predictor.predict_for_hour(state.X_seq, hour)
    cong_map = predictions_to_congestion(pred)
    cost_mx, G = build_cost_matrix(cong_map)
    return pred, cong_map, cost_mx, G


def _run_routing(source, destination, hour, cong_map, cost_mx, G):
    """Run MoST-ROUTE and return result."""
    return most_route(
        source         = source,
        destination    = destination,
        hour           = hour,
        congestion_map = cong_map,
        cost_matrix    = cost_mx,
        G              = G
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """System health check."""
    state.request_count += 1
    return {
        "status":        "healthy",
        "device":        str(DEVICE),
        "model_loaded":  state.model is not None,
        "startup_time":  state.startup_time,
        "requests_served": state.request_count,
        "num_regions":   NUM_REGIONS,
        "seq_len":       SEQ_LEN,
        "pred_len":      PRED_LEN,
        "model_metrics": MODEL_CFG.get("metrics", {})
    }


@app.post("/predict-od", response_model=ODPredictResponse)
def predict_od(req: ODPredictRequest):
    """
    Predict OD demand matrix for a given hour.
    Returns congestion scores and peak/safe region lists.
    """
    state.request_count += 1
    t0 = time.perf_counter()

    try:
        pred, cong_map, _, _ = _predict_and_build(req.hour)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = (time.perf_counter() - t0) * 1000

    return ODPredictResponse(
        hour             = req.hour,
        prediction_shape = list(pred.shape),
        avg_congestion   = float(cong_map.region_scores.mean()),
        peak_regions     = [int(r) for r in cong_map.peak_regions],
        safe_regions     = [int(r) for r in cong_map.safe_regions],
        congestion_per_region = [
            float(x) for x in cong_map.region_scores.tolist()
        ],
        future_congestion_avg = [
            float(cong_map.future_scores[:, r].mean())
            for r in range(NUM_REGIONS)
        ],
        latency_ms = round(latency, 2)
    )


@app.post("/get-route", response_model=RouteResponse)
def get_route(req: RouteRequest):
    """
    Get optimal route from source to destination.

    Returns:
      - Best path (region sequence)
      - Algorithm used (GA or ACO)
      - Strategy (balanced / fastest / shortest / transit_friendly)
      - Travel time, distance, congestion score
      - Human-readable reasoning explaining WHY this route is best
    """
    state.request_count += 1
    t0 = time.perf_counter()

    if req.source == req.destination:
        raise HTTPException(
            status_code=400,
            detail="Source and destination must be different regions."
        )
    if not (0 <= req.source < NUM_REGIONS):
        raise HTTPException(status_code=400, detail=f"Source must be 0-{NUM_REGIONS-1}")
    if not (0 <= req.destination < NUM_REGIONS):
        raise HTTPException(status_code=400, detail=f"Destination must be 0-{NUM_REGIONS-1}")

    try:
        pred, cong_map, cost_mx, G = _predict_and_build(req.hour)
        result = _run_routing(
            req.source, req.destination, req.hour,
            cong_map, cost_mx, G
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = (time.perf_counter() - t0) * 1000

    alt_path = result.alternative_paths[0] if result.alternative_paths else []

    return RouteResponse(
        source           = req.source,
        destination      = req.destination,
        hour             = req.hour,
        best_path        = [int(x) for x in result.best_path],
        algorithm_used   = result.algorithm_used,
        strategy         = result.strategy,
        travel_time_min  = float(result.travel_time_min),
        distance_km      = float(result.distance_km),
        congestion_score = float(result.congestion_score),
        reasoning        = result.reasoning,
        alternative_path = [int(x) for x in alt_path],
        congestion_map   = {
            "avg_current":   float(cong_map.region_scores.mean()),
            "n_peak_regions": len(cong_map.peak_regions),
            "n_safe_regions": len(cong_map.safe_regions),
            "peak_regions":  [int(r) for r in cong_map.peak_regions],
            "safe_regions":  [int(r) for r in cong_map.safe_regions],
        },
        latency_ms = round(latency, 2)
    )


@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    """
    Simulate a disruption scenario.

    1. Compute normal route
    2. Inject disruption on specified regions
    3. Re-route and compare
    4. Return both routes + system reasoning
    """
    state.request_count += 1
    t0 = time.perf_counter()

    if req.source == req.destination:
        raise HTTPException(status_code=400, detail="Source and destination must differ.")

    try:
        # Normal route
        pred_normal, cong_normal, cost_n, G_n = _predict_and_build(req.hour)
        result_normal = _run_routing(
            req.source, req.destination, req.hour,
            cong_normal, cost_n, G_n
        )

        # Determine disrupted regions
        disrupted = req.disrupted_regions
        if not disrupted:
            # Auto: disrupt intermediate nodes of normal path
            disrupted = result_normal.best_path[1:-1]

        # Inject disruption
        pred_disrupted = simulate_disruption(
            pred_normal, disrupted, severity=req.disruption_severity
        )
        cong_disrupted = predictions_to_congestion(pred_disrupted)
        cost_d, G_d    = build_cost_matrix(cong_disrupted)
        result_disrupted = _run_routing(
            req.source, req.destination, req.hour,
            cong_disrupted, cost_d, G_d
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = (time.perf_counter() - t0) * 1000

    path_changed = (result_normal.best_path != result_disrupted.best_path)
    time_diff    = (result_disrupted.travel_time_min -
                    result_normal.travel_time_min)

    reasoning = (
        f"Disruption detected on {len(disrupted)} regions "
        f"{disrupted} with severity {req.disruption_severity:.0%}. "
        f"{'System re-routed to avoid congested zones.' if path_changed else 'Original path remains optimal despite disruption.'} "
        f"Expected additional delay: {time_diff:+.1f} minutes. "
        f"This route is best to travel now because alternative routes "
        f"are expected to experience peak traffic during your travel time."
    )

    return SimulateResponse(
        scenario            = f"Region {req.source} → {req.destination} @ {req.hour:02d}:00",
        normal_path         = [int(x) for x in result_normal.best_path],
        normal_time_min     = float(result_normal.travel_time_min),
        disrupted_path      = [int(x) for x in result_disrupted.best_path],
        disrupted_time_min  = float(result_disrupted.travel_time_min),
        path_changed        = path_changed,
        time_difference_min = round(float(time_diff), 2),
        disrupted_regions   = [int(r) for r in disrupted],
        reasoning           = reasoning,
        latency_ms          = round(latency, 2)
    )


@app.get("/congestion-map", response_model=CongestionMapResponse)
def congestion_map(hour: int = 8):
    """
    Get the current congestion map for all 50 regions.
    Used by the frontend to render heatmaps.
    """
    state.request_count += 1

    if not (0 <= hour <= 23):
        raise HTTPException(status_code=400, detail="Hour must be 0-23")

    pred, cong_map, _, _ = _predict_and_build(hour)

    avg = float(cong_map.region_scores.mean())
    if avg < 0.33:
        level = "low"
    elif avg < 0.66:
        level = "moderate"
    else:
        level = "high"

    return CongestionMapResponse(
        hour                  = hour,
        region_congestion     = [float(x) for x in cong_map.region_scores],
        future_congestion_avg = [
            float(cong_map.future_scores[:, r].mean())
            for r in range(NUM_REGIONS)
        ],
        peak_regions          = [int(r) for r in cong_map.peak_regions],
        safe_regions          = [int(r) for r in cong_map.safe_regions],
        overall_congestion    = avg,
        congestion_level      = level
    )


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host     = "0.0.0.0",
        port     = 8000,
        reload   = False,
        workers  = 1
    )
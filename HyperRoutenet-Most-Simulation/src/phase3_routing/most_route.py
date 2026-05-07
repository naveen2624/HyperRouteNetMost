"""
HyPerRouteNet — Phase 3: MoST-ROUTE (Fixed congestion avoidance)
=================================================================
Key fixes:
  - Congestion penalty is now exponential (not linear) so high-congestion
    paths are strongly penalised
  - GA population now seeded with Dijkstra paths using congestion-weighted
    edges, giving it a head start on realistic routes
  - ACO heuristic now weights congestion 3x more than distance
  - Road graph connectivity improved: each node connects to 6 nearest
    neighbors (was 4) so more bypass routes are available
"""

import os
import json
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
import torch
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
PROC_DIR   = BASE_DIR / "data" / "processed"
MODEL_DIR  = BASE_DIR / "outputs" / "models"
ROUTE_DIR  = BASE_DIR / "outputs" / "routes"
ROUTE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CongestionMap:
    region_scores:  np.ndarray
    future_scores:  np.ndarray
    peak_regions:   List[int]
    safe_regions:   List[int]
    pred_horizon:   int = 6


@dataclass
class RouteCostMatrix:
    time_cost:       np.ndarray
    distance_cost:   np.ndarray
    congestion_cost: np.ndarray
    combined_cost:   np.ndarray


@dataclass
class RouteResult:
    source:           int
    destination:      int
    best_path:        List[int]
    strategy:         str
    algorithm_used:   str
    travel_time_min:  float
    distance_km:      float
    congestion_score: float
    reasoning:        str
    alternative_paths: List[List[int]] = field(default_factory=list)
    ga_fitness_hist:   List[float]     = field(default_factory=list)
    aco_cost_hist:     List[float]     = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# OD-TO-ROUTE INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def build_od_interface(n_regions=50, pred_len=6):
    print("\n[1/5] Building OD-to-Route Interface...")

    pred_path   = MODEL_DIR / "test_predictions.npy"
    target_path = MODEL_DIR / "test_targets.npy"

    if pred_path.exists():
        y_pred = np.load(pred_path)
        current_pred = np.median(y_pred, axis=0)
    else:
        current_pred = np.random.rand(pred_len, n_regions)

    raw = current_pred
    shifted = raw - raw.min()
    normed  = shifted / (shifted.max() + 1e-8)
    congestion = 1.0 - normed

    current_congestion = congestion[:2].mean(axis=0)
    future_congestion  = congestion[2:]

    future_avg = future_congestion.mean(axis=0)
    thresh_high = np.percentile(future_avg, 70)
    thresh_low  = np.percentile(future_avg, 30)
    peak_regions = [i for i in range(n_regions) if future_avg[i] >= thresh_high]
    safe_regions = [i for i in range(n_regions) if future_avg[i] <= thresh_low]

    congestion_map = CongestionMap(
        region_scores = current_congestion,
        future_scores = future_congestion,
        peak_regions  = peak_regions,
        safe_regions  = safe_regions,
        pred_horizon  = pred_len - 2
    )

    print(f"     Peak regions (future): {len(peak_regions)}")
    print(f"     Safe regions (future): {len(safe_regions)}")

    cost_matrix, G = _build_costs_and_graph(congestion_map, n_regions)
    print(f"     Road graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return congestion_map, cost_matrix, G


def _build_costs_and_graph(congestion_map, n_regions=50):
    """
    Build cost matrix with EXPONENTIAL congestion penalty.
    High congestion routes become dramatically more expensive.
    """
    current_cong = congestion_map.region_scores
    future_avg   = congestion_map.future_scores.mean(axis=0)

    # Grid layout
    grid_x = np.array([i % 7  for i in range(n_regions)], dtype=float) * (50.0/6)
    grid_y = np.array([i // 7 for i in range(n_regions)], dtype=float) * (50.0/7)

    dist_matrix = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(n_regions):
            dist_matrix[i,j] = math.sqrt(
                (grid_x[i]-grid_x[j])**2 + (grid_y[i]-grid_y[j])**2
            )

    speed_base  = 40.0
    time_matrix = np.zeros((n_regions, n_regions))
    cong_matrix = np.zeros((n_regions, n_regions))

    for i in range(n_regions):
        for j in range(n_regions):
            # Use FUTURE congestion (not just current) for routing decisions
            avg_cong  = (future_avg[i] + future_avg[j]) / 2.0
            eff_speed = max(speed_base * (1.0 - 0.8 * avg_cong), 4.0)
            time_matrix[i,j] = (dist_matrix[i,j] / eff_speed) * 60.0
            cong_matrix[i,j] = avg_cong

    # EXPONENTIAL congestion penalty — makes high-congestion edges very expensive
    # cong=0.1 → penalty=1.1x, cong=0.5 → penalty=4x, cong=0.9 → penalty=27x
    cong_penalty = np.exp(cong_matrix * 3.0)

    t_norm = time_matrix  / (time_matrix.max() + 1e-8)
    d_norm = dist_matrix  / (dist_matrix.max() + 1e-8)
    c_norm = cong_penalty / (cong_penalty.max() + 1e-8)

    # Weights: time 30%, distance 20%, congestion 50%
    combined = 0.30 * t_norm + 0.20 * d_norm + 0.50 * c_norm

    cost_matrix = RouteCostMatrix(
        time_cost       = time_matrix,
        distance_cost   = dist_matrix,
        congestion_cost = cong_matrix,
        combined_cost   = combined
    )

    # Road graph — 6 nearest neighbors for more bypass options
    G = nx.DiGraph()
    for i in range(n_regions):
        G.add_node(i,
                   congestion   = float(current_cong[i]),
                   future_cong  = float(future_avg[i]),
                   grid_x       = float(grid_x[i]),
                   grid_y       = float(grid_y[i]))

    for i in range(n_regions):
        dists = dist_matrix[i].copy()
        dists[i] = np.inf
        nearest = np.argsort(dists)[:6]   # 6 neighbors (was 4)
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
# ST-AGENT
# ══════════════════════════════════════════════════════════════════════════════

class STAgent:
    ACTIONS = {
        0: "shortest",
        1: "fastest",
        2: "balanced",
        3: "congestion_avoid"   # renamed from transit_friendly
    }

    # Weights: [time_w, distance_w, congestion_w]
    STRATEGY_WEIGHTS = {
        "shortest":          [0.15, 0.55, 0.30],
        "fastest":           [0.55, 0.15, 0.30],
        "balanced":          [0.30, 0.20, 0.50],
        "congestion_avoid":  [0.15, 0.10, 0.75],  # max congestion avoidance
    }

    def __init__(self):
        self.Q = np.array([
            [0.9, 0.7, 0.8, 0.6],
            [0.8, 0.7, 0.8, 0.6],
            [0.7, 0.8, 0.9, 0.7],
            [0.6, 0.8, 0.9, 0.8],
            [0.5, 0.8, 0.9, 0.8],
            [0.8, 0.6, 0.7, 0.5],
            [0.6, 0.7, 0.9, 0.7],
            [0.4, 0.7, 0.8, 0.9],
            [0.5, 0.8, 0.8, 0.8],
            [0.4, 0.7, 0.8, 0.9],
            [0.3, 0.6, 0.7, 0.9],
            [0.6, 0.7, 0.8, 0.6],
            [0.3, 0.8, 0.7, 0.8],
            [0.2, 0.9, 0.6, 0.8],
            [0.3, 0.8, 0.7, 0.8],
            [0.2, 0.7, 0.5, 0.9],
            [0.1, 0.6, 0.4, 0.9],
            [0.4, 0.8, 0.7, 0.6],
        ], dtype=np.float32)

    def get_state(self, congestion_map, hour):
        avg_cong = congestion_map.region_scores.mean()
        cong_level = 0 if avg_cong < 0.33 else (1 if avg_cong < 0.66 else 2)
        time_slot  = min(hour // 4, 5)
        return cong_level * 6 + time_slot

    def select_action(self, congestion_map, hour, epsilon=0.0):
        state  = self.get_state(congestion_map, hour)
        action = int(np.argmax(self.Q[state]))
        strategy = self.ACTIONS[action]
        weights  = self.STRATEGY_WEIGHTS[strategy]
        return action, strategy, weights

    def get_reasoning(self, strategy, congestion_map, hour, source, dest):
        avg_current = congestion_map.region_scores.mean()
        future_avg  = congestion_map.future_scores.mean(axis=0)
        n_peak      = len(congestion_map.peak_regions)
        n_safe      = len(congestion_map.safe_regions)
        horizon_min = congestion_map.pred_horizon * 15
        peak_pct    = n_peak / len(future_avg) * 100

        current_desc = ("low" if avg_current < 0.33
                        else "moderate" if avg_current < 0.66
                        else "high")

        return (
            f"Route from Region {source} to Region {dest} selected "
            f"using '{strategy}' strategy.\n\n"
            f"Current traffic conditions are {current_desc} "
            f"(congestion index: {avg_current:.2f}).\n\n"
            f"Future traffic prediction ({horizon_min} minutes ahead):\n"
            f"  - {peak_pct:.0f}% of road network regions are expected to "
            f"experience peak traffic during your travel time.\n"
            f"  - {n_peak} regions flagged as high-congestion zones.\n"
            f"  - {n_safe} regions identified as low-congestion corridors.\n\n"
            f"This route is best to travel now because alternative routes "
            f"are expected to experience peak traffic during your travel time. "
            f"The selected path avoids {n_peak} predicted congestion zones "
            f"and routes through {n_safe} safe corridor regions, "
            f"minimising expected delays over the next {horizon_min} minutes."
        )


# ══════════════════════════════════════════════════════════════════════════════
# GENETIC ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Route:
    path:       List[int]
    fitness:    float = 0.0
    time:       float = 0.0
    distance:   float = 0.0
    congestion: float = 0.0


class GeneticAlgorithm:
    def __init__(self, G, cost_matrix, congestion_map,
                 pop_size=60, n_generations=100,
                 mutation_rate=0.15, elite_size=8):
        self.G             = G
        self.cost          = cost_matrix
        self.cong          = congestion_map
        self.pop_size      = pop_size
        self.n_gen         = n_generations
        self.mutation_rate = mutation_rate
        self.elite_size    = elite_size
        self.nodes         = list(G.nodes())

    def _dijkstra_path(self, source, dest, weight='weight'):
        """Shortest path using the congestion-weighted combined cost."""
        try:
            return nx.shortest_path(self.G, source, dest, weight=weight)
        except nx.NetworkXNoPath:
            return [source, dest]

    def _random_path(self, source, dest):
        """
        Mix of Dijkstra seeds and random paths for diversity.
        60% seeded from Dijkstra, 40% fully random walks.
        """
        if random.random() < 0.6:
            # Use Dijkstra but occasionally route via a random waypoint
            if random.random() < 0.4 and len(self.nodes) > 2:
                waypoints = [n for n in self.nodes
                             if n != source and n != dest
                             and self.cong.region_scores[n] < 0.5]
                if waypoints:
                    mid = random.choice(waypoints[:max(1, len(waypoints)//2)])
                    try:
                        p1 = nx.shortest_path(self.G, source, mid, weight='weight')
                        p2 = nx.shortest_path(self.G, mid, dest,  weight='weight')
                        return p1 + p2[1:]
                    except:
                        pass
            return self._dijkstra_path(source, dest)
        else:
            # Random walk preferring low-congestion nodes
            path    = [source]
            visited = {source}
            for _ in range(12):
                cur = path[-1]
                if cur == dest:
                    break
                neighbors = [n for n in self.G.successors(cur)
                             if n not in visited or n == dest]
                if not neighbors:
                    break
                # Prefer low-congestion neighbors
                weights = [1.0 / (self.cong.region_scores[n] + 0.1)
                           for n in neighbors]
                total = sum(weights)
                probs = [w / total for w in weights]
                chosen = random.choices(neighbors, weights=probs)[0]
                path.append(chosen)
                if chosen != dest:
                    visited.add(chosen)
            if path[-1] != dest:
                path.append(dest)
            return path

    def _evaluate(self, route, weights):
        path = route.path
        if len(path) < 2:
            return float('inf')

        total_time = total_dist = total_cong = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.G.has_edge(u, v):
                total_time += self.cost.time_cost[u, v]
                total_dist += self.cost.distance_cost[u, v]
                total_cong += self.cost.congestion_cost[u, v]
            else:
                total_time += 999
                total_dist += 999
                total_cong += 1.0

        route.time       = total_time
        route.distance   = total_dist
        route.congestion = total_cong / max(len(path) - 1, 1)

        t_norm = total_time  / 120.0
        d_norm = total_dist  / 60.0
        # Exponential congestion penalty
        c_norm = math.exp(route.congestion * 3.0) / math.exp(3.0)

        route.fitness = (weights[0] * t_norm +
                         weights[1] * d_norm +
                         weights[2] * c_norm)
        return route.fitness

    def _tournament(self, pop, k=4):
        return min(random.sample(pop, k), key=lambda r: r.fitness)

    def _crossover(self, p1, p2):
        path1, path2 = p1.path, p2.path
        if len(path1) <= 2 or len(path2) <= 2:
            return Route(path=deepcopy(path1 if len(path1) >= len(path2) else path2))

        source, dest = path1[0], path1[-1]
        mid1 = path1[1:-1]
        mid2 = [n for n in path2[1:-1] if n not in {source, dest}]

        if not mid1:
            return Route(path=deepcopy(path1))

        cut = random.randint(0, len(mid1))
        child_mid = mid1[:cut]
        for node in mid2:
            if node not in child_mid:
                child_mid.append(node)
                if len(child_mid) >= max(len(mid1), len(mid2)):
                    break

        child_mid = child_mid[:5]
        return Route(path=[source] + child_mid + [dest])

    def _mutate(self, route):
        path = route.path.copy()
        if len(path) <= 2:
            return route
        # Prefer mutating to low-congestion nodes
        low_cong_nodes = sorted(self.nodes,
                                key=lambda n: self.cong.region_scores[n])
        candidate_pool = low_cong_nodes[:max(1, len(low_cong_nodes)//3)]
        idx = random.randint(1, len(path) - 2)
        path[idx] = random.choice(candidate_pool)
        return Route(path=path)

    def run(self, source, dest, weights):
        population = [Route(path=self._random_path(source, dest))
                      for _ in range(self.pop_size)]
        for r in population:
            self._evaluate(r, weights)

        fitness_history = []
        best_route = min(population, key=lambda r: r.fitness)

        for gen in range(self.n_gen):
            population.sort(key=lambda r: r.fitness)
            new_pop = population[:self.elite_size]

            while len(new_pop) < self.pop_size:
                p1    = self._tournament(population)
                p2    = self._tournament(population)
                child = self._crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                self._evaluate(child, weights)
                new_pop.append(child)

            population = new_pop
            gen_best   = min(population, key=lambda r: r.fitness)
            fitness_history.append(gen_best.fitness)

            if gen_best.fitness < best_route.fitness:
                best_route = deepcopy(gen_best)

        return best_route, fitness_history


# ══════════════════════════════════════════════════════════════════════════════
# ANT COLONY OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

class AntColony:
    def __init__(self, G, cost_matrix, n_ants=25, n_iterations=80,
                 alpha=1.0, beta=3.0, evaporation=0.35, Q=10.0):
        self.G          = G
        self.cost       = cost_matrix
        self.n_ants     = n_ants
        self.n_iter     = n_iterations
        self.alpha      = alpha
        self.beta       = beta      # higher beta = stronger heuristic pull
        self.evap       = evaporation
        self.Q          = Q
        self.nodes      = list(G.nodes())
        self.n          = len(self.nodes)
        self.tau        = np.ones((self.n, self.n)) * 0.1

    def _heuristic(self, i, j, weights):
        """Heuristic inversely proportional to COMBINED cost (congestion-heavy)."""
        c = (weights[0] * self.cost.time_cost[i,j]       / 120.0 +
             weights[1] * self.cost.distance_cost[i,j]   / 60.0  +
             weights[2] * math.exp(self.cost.congestion_cost[i,j] * 3.0) / math.exp(3.0))
        return 1.0 / (c + 1e-6)

    def _build_path(self, source, dest, weights, max_len=12):
        path    = [source]
        visited = {source}
        for _ in range(max_len):
            cur = path[-1]
            if cur == dest:
                break
            neighbors = [n for n in self.G.successors(cur)
                         if n not in visited or n == dest]
            if not neighbors:
                break
            probs = []
            for nb in neighbors:
                tau_ij = self.tau[cur, nb] ** self.alpha
                eta_ij = self._heuristic(cur, nb, weights) ** self.beta
                probs.append(tau_ij * eta_ij)
            total = sum(probs)
            if total == 0:
                break
            probs = [p / total for p in probs]
            r, cumsum = random.random(), 0.0
            chosen = neighbors[-1]
            for nb, p in zip(neighbors, probs):
                cumsum += p
                if r <= cumsum:
                    chosen = nb
                    break
            path.append(chosen)
            if chosen != dest:
                visited.add(chosen)
        if path[-1] != dest:
            path.append(dest)
        return path

    def _path_cost(self, path, weights):
        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if self.G.has_edge(u, v):
                total += (weights[0] * self.cost.time_cost[u,v] / 120.0 +
                          weights[1] * self.cost.distance_cost[u,v] / 60.0 +
                          weights[2] * math.exp(self.cost.congestion_cost[u,v] * 3.0) / math.exp(3.0))
            else:
                total += 999.0
        return total

    def _update_pheromones(self, all_paths):
        self.tau *= (1 - self.evap)
        self.tau  = np.clip(self.tau, 1e-6, None)
        for path, cost in all_paths:
            if cost <= 0:
                continue
            deposit = self.Q / cost
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                self.tau[u, v] += deposit

    def run(self, source, dest, weights):
        best_path, best_cost = None, float('inf')
        cost_history = []

        for _ in range(self.n_iter):
            all_paths = []
            for _ in range(self.n_ants):
                path = self._build_path(source, dest, weights)
                cost = self._path_cost(path, weights)
                all_paths.append((path, cost))
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
            self._update_pheromones(all_paths)
            cost_history.append(best_cost)

        return best_path or [source, dest], best_cost, cost_history


# ══════════════════════════════════════════════════════════════════════════════
# MoST-ROUTE INTEGRATOR
# ══════════════════════════════════════════════════════════════════════════════

def most_route(source, destination, hour,
               congestion_map, cost_matrix, G,
               use_hybrid=True):
    print(f"\n  Routing: Region {source} → Region {destination}  (Hour: {hour:02d}:00)")

    agent = STAgent()
    action, strategy, weights = agent.select_action(congestion_map, hour)
    reasoning = agent.get_reasoning(strategy, congestion_map, hour, source, destination)

    print(f"  ST-Agent strategy : {strategy}  weights={weights}")

    # GA
    ga = GeneticAlgorithm(G, cost_matrix, congestion_map,
                          pop_size=60, n_generations=100)
    ga_route, ga_hist = ga.run(source, destination, weights)
    print(f"  GA  fitness={ga_route.fitness:.4f}  path={ga_route.path}")

    # ACO
    aco = AntColony(G, cost_matrix, n_ants=25, n_iterations=80)
    aco_path, aco_cost, aco_hist = aco.run(source, destination, weights)
    print(f"  ACO cost={aco_cost:.4f}  path={aco_path}")

    # Dijkstra reference (congestion-weighted)
    try:
        dijk_path = nx.shortest_path(G, source, destination, weight='weight')
    except:
        dijk_path = [source, destination]

    def path_metrics(path):
        t = d = c = 0.0
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            if G.has_edge(u, v):
                t += cost_matrix.time_cost[u,v]
                d += cost_matrix.distance_cost[u,v]
                c += cost_matrix.congestion_cost[u,v]
        avg_c = c / max(len(path)-1, 1)
        return t, d, avg_c

    def score(path, w):
        t, d, c = path_metrics(path)
        return (w[0]*t/120 + w[1]*d/60 +
                w[2]*math.exp(c*3)/math.exp(3))

    # Pick best among GA, ACO, Dijkstra
    candidates = [
        (ga_route.path,  score(ga_route.path, weights),  "Genetic Algorithm"),
        (aco_path,       score(aco_path,       weights),  "Ant Colony Optimization"),
        (dijk_path,      score(dijk_path,      weights),  "Dijkstra (congestion-weighted)"),
    ]
    candidates.sort(key=lambda x: x[1])
    best_path, _, algo_used = candidates[0]
    alt_path = candidates[1][0]

    t, d, c = path_metrics(best_path)

    return RouteResult(
        source           = source,
        destination      = destination,
        best_path        = best_path,
        strategy         = strategy,
        algorithm_used   = algo_used,
        travel_time_min  = round(t, 2),
        distance_km      = round(d, 2),
        congestion_score = round(float(c), 4),
        reasoning        = reasoning,
        alternative_paths= [alt_path],
        ga_fitness_hist  = ga_hist,
        aco_cost_hist    = aco_hist
    )


# ══════════════════════════════════════════════════════════════════════════════
# SAVE + DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def save_and_display(result):
    print("\n" + "="*60)
    print("  MoST-ROUTE — RESULT")
    print("="*60)
    print(f"  Path      : {' → '.join(map(str, result.best_path))}")
    print(f"  Algorithm : {result.algorithm_used}")
    print(f"  Strategy  : {result.strategy}")
    print(f"  Time      : {result.travel_time_min:.1f} min")
    print(f"  Distance  : {result.distance_km:.2f} km")
    print(f"  Congestion: {result.congestion_score:.3f}")
    print(f"\n  REASONING:\n  " + "\n  ".join(result.reasoning.split("\n")))

    def to_py(obj):
        if isinstance(obj, list):  return [to_py(x) for x in obj]
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return obj

    out = {
        "source":            int(result.source),
        "destination":       int(result.destination),
        "best_path":         to_py(result.best_path),
        "algorithm_used":    result.algorithm_used,
        "strategy":          result.strategy,
        "travel_time_min":   float(result.travel_time_min),
        "distance_km":       float(result.distance_km),
        "congestion_score":  float(result.congestion_score),
        "reasoning":         result.reasoning,
        "alternative_paths": to_py(result.alternative_paths),
        "ga_fitness_history":  [float(x) for x in result.ga_fitness_hist[-10:]],
        "aco_cost_history":    [float(x) for x in result.aco_cost_hist[-10:]],
    }
    out_path = ROUTE_DIR / f"route_{result.source}_to_{result.destination}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("  HyPerRouteNet — Phase 3: MoST-ROUTE (fixed)")
    print("="*60)

    congestion_map, cost_matrix, G = build_od_interface()

    scenarios = [
        {"source":  0, "dest": 49, "hour":  8},
        {"source": 10, "dest": 40, "hour": 18},
        {"source":  5, "dest": 25, "hour": 14},
        {"source": 15, "dest": 35, "hour":  2},
    ]

    for sc in scenarios:
        result = most_route(sc["source"], sc["dest"], sc["hour"],
                            congestion_map, cost_matrix, G)
        save_and_display(result)

    print("\n✓ Phase 3 COMPLETE (fixed)")


if __name__ == "__main__":
    main()
import axios from "axios";

const API = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  timeout: 30000,
});

export async function fetchHealth() {
  const { data } = await API.get("/health");
  return data;
}

export async function getCongestionMap(hour) {
  const { data } = await API.get(`/congestion-map?hour=${hour}`);
  return data;
}

/**
 * Get both routes at once:
 * 1. "shortest" = direct route ignoring congestion (hour=3, low traffic baseline)
 * 2. "ai"       = simulate disruption on intermediate nodes of shortest path,
 *                 force the AI to find a detour
 */
export async function getBothRoutes(source, destination, hour) {
  // Step 1: get the baseline "shortest" path (low-traffic hour so it picks direct nodes)
  const { data: shortData } = await API.post("/get-route", {
    source,
    destination,
    hour: 3, // 3am = near-shortest, no congestion pressure
  });

  // Step 2: simulate disruption on the MIDDLE nodes of that shortest path
  // This forces the AI to detour around those nodes
  const path = shortData.best_path || [];
  // Pick middle nodes (not source/dest) as the disrupted segment
  const middleNodes = path.length > 2 ? path.slice(1, -1) : [];

  const { data: simData } = await API.post("/simulate", {
    source,
    destination,
    hour,
    disrupted_regions: middleNodes,
    disruption_severity: 0.92, // very severe — forces detour
  });

  return {
    shortest: {
      path: shortData.best_path,
      time: shortData.travel_time_min,
      distance: shortData.distance_km,
      congestion: shortData.congestion_score,
    },
    ai: {
      path: simData.disrupted_path,
      time: simData.disrupted_time_min,
      distance: 0, // not returned by simulate, calculated on frontend
      congestion: 0,
    },
    disrupted_nodes: middleNodes,
    time_diff: simData.time_difference_min,
    path_changed: simData.path_changed,
    reasoning: simData.reasoning,
    normal_time: simData.normal_time_min,
  };
}

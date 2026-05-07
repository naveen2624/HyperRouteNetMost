"""
HyPerRouteNet — Phase 1: Multimodal Data Preprocessing
========================================================
Loads, cleans, and merges all datasets:
  1. OSM road network (Bangalore) via OSMnx
  2. METR-LA traffic sensor data
  3. GTFS transit feed (BMTC)
  4. NYC Yellow Taxi OD data (Jan + Feb 2025)
  5. Weather data via Open-Meteo API

Output: processed graph, hypergraph-ready structures, OD matrices
"""

import os
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import requests
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
RAW_DIR     = BASE_DIR / "data" / "raw"
PROC_DIR    = BASE_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

METR_PATH   = RAW_DIR / "metr-la.h5"
TAXI_JAN    = RAW_DIR / "yellow_tripdata_2025-01.parquet"
TAXI_FEB    = RAW_DIR / "yellow_tripdata_2025-02.parquet"
ZONE_PATH   = RAW_DIR / "taxi_zone_lookup.csv"
GTFS_DIR    = RAW_DIR / "gtfs"


# ══════════════════════════════════════════════════════════════════════════════
# 1. OSM ROAD NETWORK — Bangalore
# ══════════════════════════════════════════════════════════════════════════════

def load_osm_network(place: str = "Bangalore, Karnataka, India") -> nx.MultiDiGraph:
    """
    Downloads Bangalore road network via OSMnx.
    Returns a MultiDiGraph with nodes=intersections, edges=road segments.
    """
    print("\n[1/5] Loading OSM road network for Bangalore...")

    G = ox.graph_from_place(place, network_type="drive")

    # Add useful edge attributes
    G = ox.add_edge_speeds(G)       # infer speed limits
    G = ox.add_edge_travel_times(G) # travel time in seconds

    nodes, edges = ox.graph_to_gdfs(G)

    print(f"     Nodes (intersections) : {len(nodes)}")
    print(f"     Edges (road segments) : {len(edges)}")
    print(f"     CRS                   : {nodes.crs}")

    # Save
    nodes.to_file(PROC_DIR / "bangalore_nodes.geojson", driver="GeoJSON")
    edges.to_file(PROC_DIR / "bangalore_edges.geojson", driver="GeoJSON")
    ox.save_graphml(G, PROC_DIR / "bangalore_road_network.graphml")

    print("     Saved: bangalore_road_network.graphml")
    return G


# ══════════════════════════════════════════════════════════════════════════════
# 2. METR-LA TRAFFIC SENSOR DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_metr_la() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Loads METR-LA HDF5 file.
    Returns:
      df_speed : DataFrame [timestamps x sensors] with speed values
      df_speed_norm : normalized speed DataFrame
    """
    print("\n[2/5] Loading METR-LA traffic sensor data...")

    with h5py.File(METR_PATH, "r") as f:
        print(f"     HDF5 keys: {list(f.keys())}")

        data  = f["df"]["block0_values"][:]
        index = f["df"]["axis1"][:]   # timestamps
        cols  = f["df"]["axis0"][:]   # sensor IDs

        # Debug: inspect raw dtype so we know what we're dealing with
        print(f"     axis1 dtype : {f['df']['axis1'].dtype}")
        print(f"     axis1 sample: {index[:3]}")

    # ── Parse index robustly ──
    # METR-LA stores timestamps as byte-strings e.g. b"2012-03-01 00:00:00"
    # Decode if necessary, then parse as datetime (no unit= assumption)
    if index.dtype.kind in ("S", "O"):  # byte-string or object
        index = [i.decode("utf-8") if isinstance(i, bytes) else str(i) for i in index]
        parsed_index = pd.to_datetime(index)
    elif np.issubdtype(index.dtype, np.integer):
        sample = int(index[0])
        if sample > 1e18:   # nanoseconds (pandas internal format)
            parsed_index = pd.to_datetime(index, unit="ns")
        elif sample > 1e12:  # milliseconds
            parsed_index = pd.to_datetime(index, unit="ms")
        elif sample > 1e9:   # seconds (normal Unix)
            parsed_index = pd.to_datetime(index, unit="s")
        else:
            parsed_index = pd.to_datetime(index, unit="s")
    else:
        parsed_index = pd.to_datetime(index)

    df_speed = pd.DataFrame(
        data,
        index=parsed_index,
        columns=[str(c) for c in cols]
    )

    print(f"     Shape          : {df_speed.shape}  (timestamps x sensors)")
    print(f"     Time range     : {df_speed.index.min()} → {df_speed.index.max()}")
    print(f"     Sensors        : {df_speed.shape[1]}")
    print(f"     Missing values : {df_speed.isnull().sum().sum()}")

    # ── Clean ──
    df_speed = df_speed.interpolate(method="linear", limit=3, axis=0)
    df_speed = df_speed.ffill().bfill()
    df_speed = df_speed.clip(lower=0, upper=120)

    print(f"     After cleaning — missing: {df_speed.isnull().sum().sum()}")

    # ── Normalize ──
    scaler = MinMaxScaler()
    speed_norm = scaler.fit_transform(df_speed.values)
    df_speed_norm = pd.DataFrame(
        speed_norm, index=df_speed.index, columns=df_speed.columns
    )

    # ── Resample to 15-minute intervals ──
    df_speed_15m = df_speed.resample("15min").mean()
    print(f"     Resampled to 15-min intervals: {df_speed_15m.shape}")

    # Save
    df_speed_15m.to_parquet(PROC_DIR / "metr_la_speed_15min.parquet")
    df_speed_norm.to_parquet(PROC_DIR / "metr_la_speed_normalized.parquet")

    print("     Saved: metr_la_speed_15min.parquet")
    return df_speed_15m, df_speed_norm
# ══════════════════════════════════════════════════════════════════════════════
# 3. GTFS TRANSIT FEED (BMTC / any Indian city)
# ══════════════════════════════════════════════════════════════════════════════

def load_gtfs() -> dict[str, pd.DataFrame]:
    """
    Loads GTFS feed from the gtfs/ directory.
    Returns dict of key GTFS tables.
    """
    print("\n[3/5] Loading GTFS transit feed...")

    required = ["stops.txt", "routes.txt", "trips.txt", "stop_times.txt"]
    gtfs = {}

    for fname in required:
        fpath = GTFS_DIR / fname
        if fpath.exists():
            gtfs[fname.replace(".txt", "")] = pd.read_csv(fpath)
            print(f"     Loaded {fname:20s} — {len(gtfs[fname.replace('.txt','')]):,} rows")
        else:
            print(f"     WARNING: {fname} not found in {GTFS_DIR}")

    if "stops" not in gtfs:
        raise FileNotFoundError(f"stops.txt missing. Check {GTFS_DIR}")

    stops = gtfs["stops"]

    # ── Validate coordinates ──
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])
    stops = stops[
        (stops["stop_lat"].between(-90, 90)) &
        (stops["stop_lon"].between(-180, 180))
    ]

    print(f"     Valid stops with coordinates: {len(stops)}")

    # ── Build transit hyperedges ──
    # Each route = one hyperedge connecting all its stops
    if "stop_times" in gtfs and "trips" in gtfs:
        merged = gtfs["stop_times"].merge(
            gtfs["trips"][["trip_id", "route_id"]], on="trip_id"
        )
        # Group by route: each route → list of stop_ids (hyperedge)
        hyperedges = (
            merged.groupby("route_id")["stop_id"]
            .apply(lambda x: list(x.unique()))
            .reset_index()
            .rename(columns={"stop_id": "stop_ids"})
        )
        hyperedges["hyperedge_size"] = hyperedges["stop_ids"].apply(len)
        print(f"     Transit hyperedges (routes): {len(hyperedges)}")
        print(f"     Avg stops per route        : {hyperedges['hyperedge_size'].mean():.1f}")

        hyperedges.to_parquet(PROC_DIR / "gtfs_hyperedges.parquet")
        print("     Saved: gtfs_hyperedges.parquet")

    stops.to_parquet(PROC_DIR / "gtfs_stops.parquet")
    print("     Saved: gtfs_stops.parquet")

    return gtfs


# ══════════════════════════════════════════════════════════════════════════════
# 4. NYC YELLOW TAXI — OD MATRIX CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def load_taxi_od() -> pd.DataFrame:
    """
    Loads NYC Yellow Taxi parquet files (Jan + Feb 2025).
    Builds an Origin-Destination demand matrix aggregated by zone and hour.
    Returns: OD dataframe [pickup_zone, dropoff_zone, hour, trip_count]
    """
    print("\n[4/5] Loading NYC Yellow Taxi OD data...")

    dfs = []
    for fpath in [TAXI_JAN, TAXI_FEB]:
        if fpath.exists():
            df = pd.read_parquet(fpath, columns=[
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "PULocationID",
                "DOLocationID",
                "trip_distance",
                "fare_amount"
            ])
            dfs.append(df)
            print(f"     Loaded {fpath.name}: {len(df):,} trips")
        else:
            print(f"     WARNING: {fpath.name} not found")

    if not dfs:
        raise FileNotFoundError("No taxi parquet files found.")

    taxi = pd.concat(dfs, ignore_index=True)
    print(f"     Combined total trips: {len(taxi):,}")

    # ── Load zone lookup ──
    zones = pd.read_csv(ZONE_PATH)
    print(f"     Zone lookup loaded: {len(zones)} zones")

    # ── Clean ──
    taxi = taxi.dropna(subset=["PULocationID", "DOLocationID"])
    taxi["PULocationID"] = taxi["PULocationID"].astype(int)
    taxi["DOLocationID"] = taxi["DOLocationID"].astype(int)

    # Filter valid zones
    valid_zones = set(zones["LocationID"].astype(int))
    taxi = taxi[
        taxi["PULocationID"].isin(valid_zones) &
        taxi["DOLocationID"].isin(valid_zones)
    ]

    # Remove zero-distance trips and outlier fares
    taxi = taxi[taxi["trip_distance"] > 0.1]
    taxi = taxi[taxi["fare_amount"].between(2.5, 500)]

    # Parse timestamps
    taxi["pickup_dt"]  = pd.to_datetime(taxi["tpep_pickup_datetime"], errors="coerce")
    taxi["dropoff_dt"] = pd.to_datetime(taxi["tpep_dropoff_datetime"], errors="coerce")
    taxi = taxi.dropna(subset=["pickup_dt", "dropoff_dt"])

    # Filter to Jan–Feb 2025 only
    taxi = taxi[taxi["pickup_dt"].dt.year == 2025]
    taxi = taxi[taxi["pickup_dt"].dt.month.isin([1, 2])]

    print(f"     After cleaning: {len(taxi):,} trips remain")

    # ── Build OD matrix ──
    taxi["hour"]    = taxi["pickup_dt"].dt.hour
    taxi["weekday"] = taxi["pickup_dt"].dt.dayofweek  # 0=Mon, 6=Sun
    taxi["date"]    = taxi["pickup_dt"].dt.date

    od_matrix = (
        taxi.groupby(["PULocationID", "DOLocationID", "hour", "weekday"])
        .size()
        .reset_index(name="trip_count")
    )

    print(f"     OD pairs (zone x zone x hour x weekday): {len(od_matrix):,}")

    # ── Merge zone names ──
    od_matrix = od_matrix.merge(
        zones[["LocationID", "Zone", "Borough"]].rename(
            columns={"LocationID": "PULocationID",
                     "Zone": "PU_Zone",
                     "Borough": "PU_Borough"}
        ), on="PULocationID", how="left"
    )
    od_matrix = od_matrix.merge(
        zones[["LocationID", "Zone", "Borough"]].rename(
            columns={"LocationID": "DOLocationID",
                     "Zone": "DO_Zone",
                     "Borough": "DO_Borough"}
        ), on="DOLocationID", how="left"
    )

    # ── Peak hour statistics ──
    hourly = od_matrix.groupby("hour")["trip_count"].sum()
    peak_hour = hourly.idxmax()
    print(f"     Peak demand hour: {peak_hour}:00 "
          f"({hourly[peak_hour]:,} trips)")

    # Save
    od_matrix.to_parquet(PROC_DIR / "od_matrix.parquet")
    print("     Saved: od_matrix.parquet")

    return od_matrix


# ══════════════════════════════════════════════════════════════════════════════
# 5. WEATHER DATA — Open-Meteo API (free, no key needed)
# ══════════════════════════════════════════════════════════════════════════════

def load_weather(
    lat: float = 12.9716,    # Bangalore latitude
    lon: float = 77.5946,    # Bangalore longitude
    start: str = "2025-01-01",
    end:   str = "2025-02-28"
) -> pd.DataFrame:
    """
    Pulls historical hourly weather for Bangalore from Open-Meteo API.
    Returns: DataFrame with temperature, precipitation, wind, visibility
    """
    print("\n[5/5] Fetching weather data from Open-Meteo...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":              lat,
        "longitude":             lon,
        "start_date":            start,
        "end_date":              end,
        "hourly": [
            "temperature_2m",
            "precipitation",
            "windspeed_10m",
            "visibility",
            "weathercode"
        ],
        "timezone": "Asia/Kolkata"
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    weather = pd.DataFrame({
        "timestamp":     pd.to_datetime(data["time"]),
        "temperature_c": data["temperature_2m"],
        "precipitation": data["precipitation"],
        "wind_speed":    data["windspeed_10m"],
        "visibility":    data["visibility"],
        "weather_code":  data["weathercode"]
    })

    # ── Derive disruption flag ──
    # Flag hours likely to cause traffic disruption
    weather["is_disruption"] = (
        (weather["precipitation"] > 5.0) |   # heavy rain > 5mm/hr
        (weather["wind_speed"] > 40.0)   |   # strong wind
        (weather["visibility"] < 1000)        # poor visibility < 1km
    ).astype(int)

    print(f"     Weather records : {len(weather)}")
    print(f"     Date range      : {weather['timestamp'].min()} → "
          f"{weather['timestamp'].max()}")
    print(f"     Disruption hours: {weather['is_disruption'].sum()} "
          f"({weather['is_disruption'].mean()*100:.1f}%)")

    weather.to_parquet(PROC_DIR / "weather_bangalore.parquet")
    print("     Saved: weather_bangalore.parquet")

    return weather


# ══════════════════════════════════════════════════════════════════════════════
# 6. MERGE ALL DATASETS — Unified feature matrix
# ══════════════════════════════════════════════════════════════════════════════

def merge_all(
    df_speed: pd.DataFrame,
    od_matrix: pd.DataFrame,
    weather: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges traffic speed, OD demand, and weather into a unified
    spatio-temporal feature matrix for model training.
    """
    print("\n[6/5] Merging all datasets into unified feature matrix...")

    # ── Aggregate speed to hourly ──
    speed_hourly = df_speed.resample("1h").mean()
    speed_hourly["avg_speed"]    = speed_hourly.mean(axis=1)
    speed_hourly["min_speed"]    = speed_hourly.min(axis=1)
    speed_hourly["speed_std"]    = speed_hourly.std(axis=1)
    speed_summary = speed_hourly[["avg_speed", "min_speed", "speed_std"]].copy()
    speed_summary.index.name = "timestamp"
    speed_summary = speed_summary.reset_index()

    # ── OD aggregate per hour ──
    od_hourly = (
        od_matrix.groupby("hour")["trip_count"]
        .agg(["sum", "mean", "std"])
        .reset_index()
        .rename(columns={
            "sum":  "total_trips",
            "mean": "avg_od_demand",
            "std":  "od_demand_std"
        })
    )

    # ── Weather: round to hour ──
    weather["hour"] = weather["timestamp"].dt.hour

    # ── Merge ──
    unified = speed_summary.copy()
    unified["hour"] = unified["timestamp"].dt.hour
    unified = unified.merge(od_hourly, on="hour", how="left")
    unified = unified.merge(
        weather[["timestamp", "temperature_c", "precipitation",
                 "wind_speed", "is_disruption"]],
        on="timestamp", how="left"
    )

    # ── Fill any remaining NaN ──
    unified = unified.ffill().bfill()

    print(f"     Unified matrix shape : {unified.shape}")
    print(f"     Columns              : {list(unified.columns)}")

    unified.to_parquet(PROC_DIR / "unified_features.parquet")
    print("     Saved: unified_features.parquet")

    return unified


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  HyPerRouteNet — Phase 1: Data Preprocessing")
    print("=" * 60)

    # Step 1: Road network
    G = load_osm_network()

    # Step 2: Traffic data
    df_speed, df_speed_norm = load_metr_la()

    # Step 3: GTFS
    gtfs = load_gtfs()

    # Step 4: Taxi OD
    od_matrix = load_taxi_od()

    # Step 5: Weather
    weather = load_weather()

    # Step 6: Merge
    unified = merge_all(df_speed, od_matrix, weather)

    print("\n" + "=" * 60)
    print("  Phase 1 COMPLETE ✓")
    print(f"  All outputs saved to: {PROC_DIR}")
    print("=" * 60)
    print("\nFiles generated:")
    for f in sorted(PROC_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:40s} {size_mb:.2f} MB")

    print("\n✓ Ready for Phase 2: Hypergraph construction")


if __name__ == "__main__":
    main()
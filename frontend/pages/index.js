import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { getBothRoutes, getCongestionMap, fetchHealth } from "../lib/api";
import { getRegionName } from "../lib/regions";
import styles from "../styles/Home.module.css";

const MapView = dynamic(() => import("../components/MapView"), { ssr: false });

const HOUR_LABELS = [
  "Midnight",
  "1 AM",
  "2 AM",
  "3 AM",
  "4 AM",
  "5 AM",
  "6 AM",
  "7 AM",
  "8 AM",
  "9 AM",
  "10 AM",
  "11 AM",
  "Noon",
  "1 PM",
  "2 PM",
  "3 PM",
  "4 PM",
  "5 PM",
  "6 PM",
  "7 PM",
  "8 PM",
  "9 PM",
  "10 PM",
  "11 PM",
];

export default function Home() {
  const [source, setSource] = useState(7);
  const [dest, setDest] = useState(40);
  const [hour, setHour] = useState(8);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);
  const [cong, setCong] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => {});
    loadCong(8);
  }, []);

  async function loadCong(h) {
    try {
      setCong(await getCongestionMap(h));
    } catch {}
  }

  async function handleCompare() {
    if (source === dest) {
      setError("Source and destination must differ.");
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const data = await getBothRoutes(source, dest, hour);
      setResult(data);
      await loadCong(hour);
    } catch (e) {
      setError(
        e.response?.data?.detail ||
          "Backend not reachable. Is the server running?",
      );
    } finally {
      setLoading(false);
    }
  }

  const congLevel = cong?.congestion_level || "unknown";
  const congColor =
    { low: "#00e676", moderate: "#ffb300", high: "#ff4444" }[congLevel] ||
    "#7a9ab8";

  // Build the scenario text
  const disruptedStr =
    result?.disrupted_nodes
      ?.map((n) => `${n}(${getRegionName(n)})`)
      .join(" → ") || "";
  const timeSaved = result
    ? (result.normal_time - result.ai.time).toFixed(1)
    : null;

  return (
    <div className={styles.shell}>
      {/* Top bar */}
      <header className={styles.topbar}>
        <div className={styles.logo}>
          <span className={styles.acc}>Hyper</span>RouteNet
          <span className={styles.sub}>MoST Framework · Smart Mobility</span>
        </div>
        <div className={styles.topRight}>
          {health && (
            <>
              <Chip
                label="Device"
                val={health.device?.toUpperCase()}
                color="var(--accent)"
              />
              <Chip label="Regions" val={health.num_regions} />
              <Chip label="MAE" val={health.model_metrics?.MAE?.toFixed(3)} />
            </>
          )}
          <div
            className={styles.statusDot}
            style={{ background: health ? "#00e676" : "#ff4444" }}
          />
        </div>
      </header>

      <div className={styles.body}>
        {/* Sidebar */}
        <aside className={styles.sidebar}>
          {/* Traffic status */}
          <div className={styles.trafficBar}>
            <div
              className={styles.trafficDot}
              style={{ background: congColor }}
            />
            <div>
              <div className={styles.trafficLbl}>Traffic now</div>
              <div className={styles.trafficVal} style={{ color: congColor }}>
                {congLevel.toUpperCase()}
                {cong && ` · ${(cong.overall_congestion * 100).toFixed(0)}%`}
              </div>
            </div>
            <div className={styles.timeBig}>{HOUR_LABELS[hour]}</div>
          </div>

          {/* Controls */}
          <div className={styles.controls}>
            <label className={styles.lbl}>From</label>
            <select
              className={styles.sel}
              value={source}
              onChange={(e) => setSource(+e.target.value)}
            >
              {Array.from({ length: 50 }, (_, i) => (
                <option key={i} value={i}>
                  {i} — {getRegionName(i)}
                </option>
              ))}
            </select>

            <label className={styles.lbl}>To</label>
            <select
              className={styles.sel}
              value={dest}
              onChange={(e) => setDest(+e.target.value)}
            >
              {Array.from({ length: 50 }, (_, i) => (
                <option key={i} value={i}>
                  {i} — {getRegionName(i)}
                </option>
              ))}
            </select>

            <label className={styles.lbl}>
              Departure — {HOUR_LABELS[hour]}
            </label>
            <input
              type="range"
              min={0}
              max={23}
              value={hour}
              className={styles.slider}
              onChange={(e) => {
                setHour(+e.target.value);
                loadCong(+e.target.value);
              }}
            />

            {error && <div className={styles.err}>{error}</div>}

            <button
              className={styles.btn}
              onClick={handleCompare}
              disabled={loading}
            >
              {loading ? (
                <>
                  <Spin />
                  Calculating...
                </>
              ) : (
                "▶  Show Route Comparison"
              )}
            </button>
          </div>

          {/* Result panel */}
          {result && (
            <div className={styles.resultWrap}>
              {/* Disruption alert */}
              <div className={styles.alert}>
                <div className={styles.alertIcon}>⚠</div>
                <div>
                  <div className={styles.alertTitle}>
                    Traffic disruption detected
                  </div>
                  <div className={styles.alertSub}>
                    Congestion on segment: <b>{disruptedStr}</b>
                  </div>
                </div>
              </div>

              {/* Two route cards */}
              <div className={styles.routeCards}>
                {/* Shortest (blocked) */}
                <div
                  className={styles.routeCard}
                  style={{ borderColor: "#378ADD" }}
                >
                  <div
                    className={styles.rHead}
                    style={{ background: "rgba(55,138,221,0.1)" }}
                  >
                    <div
                      className={styles.rDot}
                      style={{ background: "#378ADD" }}
                    />
                    <span className={styles.rTitle}>Shortest path</span>
                    <span
                      className={styles.rTag}
                      style={{
                        color: "#ff9966",
                        borderColor: "rgba(255,153,102,0.4)",
                        background: "rgba(255,153,102,0.1)",
                      }}
                    >
                      CONGESTED
                    </span>
                  </div>
                  <div className={styles.pathRow}>
                    {result.shortest.path.map((n, i, arr) => (
                      <span key={i} className={styles.pathItem}>
                        <span
                          className={styles.pNode}
                          style={{
                            borderColor: "#378ADD",
                            color: "#85B7EB",
                            background: result.disrupted_nodes.includes(n)
                              ? "rgba(255,68,68,0.2)"
                              : undefined,
                            borderColor: result.disrupted_nodes.includes(n)
                              ? "#ff4444"
                              : "#378ADD",
                            color: result.disrupted_nodes.includes(n)
                              ? "#ff6666"
                              : "#85B7EB",
                          }}
                        >
                          {n}
                        </span>
                        {i < arr.length - 1 && (
                          <span
                            className={styles.arrow}
                            style={{
                              color: result.disrupted_nodes.includes(n)
                                ? "#ff4444"
                                : undefined,
                            }}
                          >
                            {result.disrupted_nodes.includes(n) ? "✕" : "→"}
                          </span>
                        )}
                      </span>
                    ))}
                  </div>
                  <div className={styles.rStats}>
                    <Stat
                      label="Est. time"
                      val={`${result.normal_time.toFixed(1)} min`}
                      color="#ff9966"
                    />
                    <Stat label="Congestion" val="HIGH" color="#ff4444" />
                    <Stat label="Status" val="SLOW" color="#ff4444" />
                  </div>
                  <div
                    className={styles.rNote}
                    style={{
                      background: "rgba(255,68,68,0.07)",
                      color: "#ff9966",
                    }}
                  >
                    ✕ Accident/congestion on nodes{" "}
                    {result.disrupted_nodes.join(", ")} — delays expected
                  </div>
                </div>

                {/* AI detour */}
                <div
                  className={styles.routeCard}
                  style={{ borderColor: "#00e676" }}
                >
                  <div
                    className={styles.rHead}
                    style={{ background: "rgba(0,230,118,0.09)" }}
                  >
                    <div
                      className={styles.rDot}
                      style={{ background: "#00e676" }}
                    />
                    <span className={styles.rTitle}>HyPerRouteNet route</span>
                    <span
                      className={styles.rTag}
                      style={{
                        color: "#00e676",
                        borderColor: "rgba(0,230,118,0.4)",
                        background: "rgba(0,230,118,0.1)",
                      }}
                    >
                      OPTIMAL
                    </span>
                  </div>
                  <div className={styles.pathRow}>
                    {result.ai.path.map((n, i, arr) => (
                      <span key={i} className={styles.pathItem}>
                        <span
                          className={styles.pNode}
                          style={{ borderColor: "#00e676", color: "#00e676" }}
                        >
                          {n}
                        </span>
                        {i < arr.length - 1 && (
                          <span className={styles.arrow}>→</span>
                        )}
                      </span>
                    ))}
                  </div>
                  <div className={styles.rStats}>
                    <Stat
                      label="Est. time"
                      val={`${result.ai.time.toFixed(1)} min`}
                      color="#00e676"
                    />
                    <Stat label="Congestion" val="LOW" color="#00e676" />
                    <Stat label="Status" val="CLEAR" color="#00e676" />
                  </div>

                  {/* Saving */}
                  <div className={styles.savingBox}>
                    <span className={styles.savingNum}>
                      {+timeSaved > 0
                        ? `⚡ ${timeSaved} min faster`
                        : "✓ Congestion-free detour"}
                    </span>
                    <span className={styles.savingDesc}>
                      by avoiding blocked nodes{" "}
                      {result.disrupted_nodes.join(", ")}
                    </span>
                  </div>
                </div>
              </div>

              {/* AI Reasoning */}
              <div className={styles.reasonBox}>
                <div className={styles.reasonLbl}>
                  <span className={styles.reasonIcon}>🧠</span> Why this route?
                </div>
                <p className={styles.reasonTxt}>
                  This route is best to travel now because the direct path
                  through {disruptedStr} is experiencing heavy congestion or
                  disruption. HyPerRouteNet predicted this{" "}
                  {result.disrupted_nodes.length > 1 ? "corridor" : "segment"}{" "}
                  would be congested during your travel window and selected an
                  alternative path that avoids the affected nodes, saving
                  approximately {Math.abs(+timeSaved)} minutes compared to the
                  congested route.
                </p>
              </div>
            </div>
          )}

          {/* Region legend */}
          <div className={styles.legend}>
            <div className={styles.legTitle}>Map legend</div>
            <LegRow
              color="#378ADD"
              type="solid"
              label="Shortest (congested) path"
            />
            <LegRow color="#00e676" type="dash" label="AI detour route" />
            <LegRow
              color="#ff4444"
              type="dot"
              label="Disrupted / accident zone"
            />
            <LegRow color="#00d4ff" type="dot" label="Source" />
            <LegRow color="#ff6666" type="dot" label="Destination" />
          </div>
        </aside>

        {/* Map */}
        <main className={styles.mapWrap}>
          <MapView
            shortestPath={result?.shortest?.path}
            aiPath={result?.ai?.path}
            disruptedNodes={result?.disrupted_nodes}
            congestion={cong}
            source={source}
            dest={dest}
            normalTime={result?.normal_time}
            aiTime={result?.ai?.time}
          />
        </main>
      </div>
    </div>
  );
}

function Chip({ label, val, color }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 1,
      }}
    >
      <span
        style={{
          fontSize: 9,
          color: "var(--text-muted)",
          fontFamily: "var(--mono)",
          textTransform: "uppercase",
          letterSpacing: 1,
        }}
      >
        {label}
      </span>
      <span
        style={{
          fontSize: 11,
          fontFamily: "var(--mono)",
          color: color || "var(--text-secondary)",
        }}
      >
        {val ?? "—"}
      </span>
    </div>
  );
}

function Stat({ label, val, color }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span
        style={{
          fontSize: 9,
          color: "var(--text-muted)",
          textTransform: "uppercase",
          letterSpacing: 1,
        }}
      >
        {label}
      </span>
      <span
        style={{
          fontSize: 13,
          fontFamily: "var(--mono)",
          fontWeight: 700,
          color,
        }}
      >
        {val}
      </span>
    </div>
  );
}

function LegRow({ color, type, label }) {
  const icon =
    type === "dot" ? (
      <div
        style={{
          width: 10,
          height: 10,
          borderRadius: "50%",
          background: color,
          flexShrink: 0,
        }}
      />
    ) : type === "solid" ? (
      <div
        style={{
          width: 20,
          height: 3,
          background: color,
          borderRadius: 2,
          flexShrink: 0,
        }}
      />
    ) : (
      <div
        style={{
          width: 20,
          height: 0,
          borderTop: `2.5px dashed ${color}`,
          flexShrink: 0,
          marginTop: 1,
        }}
      />
    );
  return (
    <div
      style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}
    >
      {icon}
      <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>
        {label}
      </span>
    </div>
  );
}

function Spin() {
  return (
    <span
      style={{
        display: "inline-block",
        width: 12,
        height: 12,
        border: "2px solid transparent",
        borderTopColor: "currentColor",
        borderRadius: "50%",
        animation: "spin .6s linear infinite",
        marginRight: 6,
      }}
    />
  );
}

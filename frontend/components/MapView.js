import { useEffect, useRef } from "react";

function congestionColor(s) {
  if (s < 0.33) return "#00e676";
  if (s < 0.66) return "#ffb300";
  return "#ff4444";
}

export default function MapView({
  shortestPath,
  aiPath,
  disruptedNodes,
  congestion,
  source,
  dest,
  normalTime,
  aiTime,
}) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const layers = useRef({
    heat: [],
    shortest: [],
    ai: [],
    pins: [],
    accident: [],
    labels: [],
  });

  // ── Init map ──────────────────────────────────────────────────────────────
  useEffect(() => {
    if (typeof window === "undefined" || mapRef.current) return;
    import("leaflet").then(({ default: L }) => {
      delete L.Icon.Default.prototype._getIconUrl;
      L.Icon.Default.mergeOptions({
        iconRetinaUrl:
          "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
        iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
        shadowUrl:
          "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
      });
      const map = L.map(containerRef.current, {
        center: [12.96, 77.6],
        zoom: 11,
        preferCanvas: false,
      });
      L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "&copy; OpenStreetMap contributors",
        maxZoom: 19,
      }).addTo(map);
      mapRef.current = map;
      setTimeout(() => map.invalidateSize(), 200);
    });
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  // ── Light congestion heatmap ──────────────────────────────────────────────
  useEffect(() => {
    if (!mapRef.current || !congestion) return;
    import("leaflet").then(({ default: L }) => {
      const map = mapRef.current;
      if (!map) return;
      layers.current.heat.forEach((l) => {
        try {
          map.removeLayer(l);
        } catch {}
      });
      layers.current.heat = [];
      const scores = congestion.region_congestion || [];
      const { REGION_COORDS } = require("../lib/regions");
      scores.forEach((score, i) => {
        const rc = REGION_COORDS[i];
        if (!rc) return;
        const color = congestionColor(score);
        const c = L.circle([rc.lat, rc.lng], {
          radius: 550 + score * 900,
          color,
          fillColor: color,
          fillOpacity: 0.05 + score * 0.13,
          weight: 0.4,
          opacity: 0.2,
          interactive: false,
        }).addTo(map);
        layers.current.heat.push(c);
      });
    });
  }, [congestion]);

  // ── All-region node number labels ─────────────────────────────────────────
  useEffect(() => {
    if (!mapRef.current || !congestion) return;
    import("leaflet").then(({ default: L }) => {
      const map = mapRef.current;
      if (!map) return;
      layers.current.labels.forEach((l) => {
        try {
          map.removeLayer(l);
        } catch {}
      });
      layers.current.labels = [];

      const { REGION_COORDS, getRegionName } = require("../lib/regions");
      const scores = congestion.region_congestion || [];

      REGION_COORDS.forEach(({ id, lat, lng }) => {
        const score = scores[id] ?? 0;
        const color = congestionColor(score);

        // Small numbered circle for every region
        const icon = L.divIcon({
          className: "",
          html: `<div style="
            width:22px; height:22px; border-radius:50%;
            background:rgba(8,12,20,0.82);
            border:1.5px solid ${color};
            display:flex; align-items:center; justify-content:center;
            font-size:9px; font-weight:700; color:${color};
            font-family:'Space Mono',monospace;
            box-shadow:0 0 6px ${color}55;
            line-height:1;
          ">${id}</div>`,
          iconSize: [22, 22],
          iconAnchor: [11, 11],
        });

        const m = L.marker([lat, lng], { icon, zIndexOffset: 100 }).addTo(map);
        m.bindPopup(`<div style="font-family:monospace;font-size:11px;min-width:140px">
          <b style="color:#00d4ff">Region ${id}</b><br/>
          ${getRegionName(id)}<br/>
          <span style="color:${color}">Congestion: ${(score * 100).toFixed(0)}%</span>
        </div>`);
        layers.current.labels.push(m);
      });
    });
  }, [congestion]);

  // ── Source / Destination pins ─────────────────────────────────────────────
  useEffect(() => {
    if (!mapRef.current) return;
    import("leaflet").then(({ default: L }) => {
      const map = mapRef.current;
      if (!map) return;
      layers.current.pins.forEach((l) => {
        try {
          map.removeLayer(l);
        } catch {}
      });
      layers.current.pins = [];
      const { REGION_COORDS, getRegionName } = require("../lib/regions");

      const makePin = (id, bg, label) => {
        const rc = REGION_COORDS[id];
        if (!rc) return;
        const icon = L.divIcon({
          className: "",
          html: `<div style="
            width:30px; height:30px; border-radius:50%;
            background:${bg}; border:3px solid #fff;
            display:flex; align-items:center; justify-content:center;
            font-size:10px; font-weight:900; color:#000;
            font-family:'Space Mono',monospace;
            box-shadow:0 0 14px ${bg}88;
          ">${id}</div>`,
          iconSize: [30, 30],
          iconAnchor: [15, 15],
        });
        const m = L.marker([rc.lat, rc.lng], { icon, zIndexOffset: 500 }).addTo(
          map,
        );
        m.bindPopup(`<div style="font-family:monospace;font-size:11px">
          <b style="color:${bg}">${label}</b><br/>Region ${id}<br/>${getRegionName(id)}
        </div>`);
        // Outer ring
        const ring = L.circle([rc.lat, rc.lng], {
          radius: 800,
          color: bg,
          fillColor: "transparent",
          weight: 2,
          opacity: 0.4,
          interactive: false,
        }).addTo(map);
        layers.current.pins.push(m, ring);
      };

      makePin(source, "#00d4ff", "▶ SOURCE");
      makePin(dest, "#ff5555", "◉ DEST");
    });
  }, [source, dest]);

  // ── Shortest path (blue solid) + accident ✕ on disrupted nodes ───────────
  useEffect(() => {
    if (!mapRef.current) return;
    import("leaflet").then(({ default: L }) => {
      const map = mapRef.current;
      if (!map) return;
      layers.current.shortest.forEach((l) => {
        try {
          map.removeLayer(l);
        } catch {}
      });
      layers.current.accident.forEach((l) => {
        try {
          map.removeLayer(l);
        } catch {}
      });
      layers.current.shortest = [];
      layers.current.accident = [];

      if (!shortestPath || shortestPath.length < 2) return;
      const { REGION_COORDS, getRegionName } = require("../lib/regions");
      const dn = new Set(disruptedNodes || []);

      // Build coords
      const coords = shortestPath
        .map((id) => {
          const rc = REGION_COORDS[id];
          return rc ? [rc.lat, rc.lng] : null;
        })
        .filter(Boolean);

      if (coords.length >= 2) {
        const glow = L.polyline(coords, {
          color: "#378ADD",
          weight: 10,
          opacity: 0.12,
          interactive: false,
        }).addTo(map);
        const line = L.polyline(coords, {
          color: "#378ADD",
          weight: 2.5,
          opacity: 0.75,
        }).addTo(map);
        line.bindPopup(`<div style="font-family:monospace;color:#85B7EB">
          <b>Shortest path</b><br/>${normalTime?.toFixed(1)} min<br/>
          <span style="color:#ff9966">⚠ Goes through congested zones</span>
        </div>`);
        layers.current.shortest.push(glow, line);
      }

      // Node labels along shortest path (show numbers on the path)
      shortestPath.forEach((id, idx) => {
        const rc = REGION_COORDS[id];
        if (!rc) return;
        if (idx === 0 || idx === shortestPath.length - 1) return; // source/dest already pinned
        const isDisrupted = dn.has(id);
        const icon = L.divIcon({
          className: "",
          html: `<div style="
            width:20px; height:20px; border-radius:50%;
            background:${isDisrupted ? "rgba(255,68,68,0.3)" : "rgba(55,138,221,0.2)"};
            border:2px solid ${isDisrupted ? "#ff4444" : "#378ADD"};
            display:flex; align-items:center; justify-content:center;
            font-size:9px; font-weight:700;
            color:${isDisrupted ? "#ff6666" : "#85B7EB"};
            font-family:'Space Mono',monospace;
          ">${id}</div>`,
          iconSize: [20, 20],
          iconAnchor: [10, 10],
        });
        const m = L.marker([rc.lat, rc.lng], { icon, zIndexOffset: 200 }).addTo(
          map,
        );
        m.bindPopup(`<div style="font-family:monospace;font-size:11px">
          Region ${id}: ${getRegionName(id)}<br/>
          ${isDisrupted ? '<b style="color:#ff4444">⚠ DISRUPTED — accident here</b>' : "Shortest path node"}
        </div>`);
        layers.current.shortest.push(m);

        // Big ✕ accident marker
        if (isDisrupted) {
          const zone = L.circle([rc.lat, rc.lng], {
            radius: 1100,
            color: "#ff4444",
            fillColor: "#ff4444",
            fillOpacity: 0.15,
            weight: 2,
            dashArray: "5 5",
            interactive: false,
          }).addTo(map);

          const axIcon = L.divIcon({
            className: "",
            html: `<div style="
              width:32px; height:32px; border-radius:50%;
              background:#ff4444; border:3px solid #fff;
              display:flex; align-items:center; justify-content:center;
              font-size:16px; font-weight:900; color:#fff;
              font-family:monospace;
              box-shadow:0 0 16px rgba(255,68,68,0.8);
              animation:pulse 1.5s ease-in-out infinite;
            ">✕</div>
            <style>@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.15)}}</style>`,
            iconSize: [32, 32],
            iconAnchor: [16, 16],
          });
          const ax = L.marker([rc.lat, rc.lng], {
            icon: axIcon,
            zIndexOffset: 600,
          }).addTo(map);
          ax.bindPopup(`<div style="font-family:monospace;font-size:12px">
            <b style="color:#ff4444">⚠ ACCIDENT / CONGESTION</b><br/>
            Region ${id}: ${getRegionName(id)}<br/>
            <span style="color:#ff9966">Avoid — heavy delays expected</span>
          </div>`);
          layers.current.accident.push(zone, ax);
        }
      });
    });
  }, [shortestPath, disruptedNodes, normalTime]);

  // ── AI detour route (green dashed) with node numbers ─────────────────────
  useEffect(() => {
    if (!mapRef.current) return;
    import("leaflet").then(({ default: L }) => {
      const map = mapRef.current;
      if (!map) return;
      layers.current.ai.forEach((l) => {
        try {
          map.removeLayer(l);
        } catch {}
      });
      layers.current.ai = [];

      if (!aiPath || aiPath.length < 2) return;
      const { REGION_COORDS, getRegionName } = require("../lib/regions");
      const shortSet = new Set(shortestPath || []);

      const coords = aiPath
        .map((id) => {
          const rc = REGION_COORDS[id];
          return rc ? [rc.lat, rc.lng] : null;
        })
        .filter(Boolean);

      if (coords.length >= 2) {
        const glow = L.polyline(coords, {
          color: "#00e676",
          weight: 14,
          opacity: 0.12,
          interactive: false,
        }).addTo(map);
        const line = L.polyline(coords, {
          color: "#00e676",
          weight: 3,
          opacity: 1,
          dashArray: "12 6",
        }).addTo(map);
        line.bindPopup(`<div style="font-family:monospace;color:#00e676">
          <b>AI optimised route</b><br/>${aiTime?.toFixed(1)} min<br/>✓ Avoids disrupted zones
        </div>`);
        layers.current.ai.push(glow, line);
      }

      // Node labels along AI path
      aiPath.forEach((id, idx) => {
        const rc = REGION_COORDS[id];
        if (!rc) return;
        if (idx === 0 || idx === aiPath.length - 1) return;
        const isDetour = !shortSet.has(id);

        const icon = L.divIcon({
          className: "",
          html: `<div style="
            width:22px; height:22px; border-radius:50%;
            background:${isDetour ? "rgba(0,230,118,0.25)" : "rgba(0,230,118,0.1)"};
            border:2px solid #00e676;
            display:flex; align-items:center; justify-content:center;
            font-size:9px; font-weight:700; color:#00e676;
            font-family:'Space Mono',monospace;
            ${isDetour ? "box-shadow:0 0 8px rgba(0,230,118,0.5);" : ""}
          ">${id}</div>`,
          iconSize: [22, 22],
          iconAnchor: [11, 11],
        });
        const m = L.marker([rc.lat, rc.lng], { icon, zIndexOffset: 300 }).addTo(
          map,
        );
        m.bindPopup(`<div style="font-family:monospace;font-size:11px">
          Region ${id}: ${getRegionName(id)}<br/>
          ${
            isDetour
              ? '<b style="color:#00e676">✓ Detour node — chosen to bypass congestion</b>'
              : "Shared with shortest path"
          }
        </div>`);
        layers.current.ai.push(m);
      });

      // Fit both paths in view
      const allCoords = [
        ...(shortestPath || [])
          .map((id) => {
            const rc = REGION_COORDS[id];
            return rc ? [rc.lat, rc.lng] : null;
          })
          .filter(Boolean),
        ...coords,
      ];
      if (allCoords.length > 0) {
        try {
          mapRef.current.fitBounds(L.latLngBounds(allCoords).pad(0.28), {
            animate: true,
          });
        } catch {}
      }
    });
  }, [aiPath, shortestPath, aiTime]);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <div ref={containerRef} style={{ width: "100%", height: "100%" }} />

      {/* Overlay legend */}
      <div
        style={{
          position: "absolute",
          top: 12,
          right: 12,
          zIndex: 1000,
          background: "rgba(8,12,20,0.92)",
          border: "1px solid #1e2d42",
          borderRadius: 8,
          padding: "10px 14px",
          fontFamily: "'Space Mono',monospace",
          backdropFilter: "blur(6px)",
          pointerEvents: "none",
          minWidth: 200,
        }}
      >
        <div
          style={{
            fontSize: 9,
            color: "#3d5a73",
            textTransform: "uppercase",
            letterSpacing: 2,
            marginBottom: 8,
          }}
        >
          Map legend
        </div>
        <LRow line solid color="#378ADD" label="Shortest (congested) path" />
        <LRow line dash color="#00e676" label="AI detour route" />
        <LRow icon="✕" color="#ff4444" label="Accident / disruption" />
        <LRow icon="●" color="#00d4ff" label="Source node" />
        <LRow icon="●" color="#ff5555" label="Destination node" />
        <LRow icon="22" color="#7a9ab8" label="Region number" />
        <div style={{ borderTop: "1px solid #1e2d42", margin: "8px 0" }} />
        <LRow dot color="#00e676" label="Low congestion" />
        <LRow dot color="#ffb300" label="Moderate congestion" />
        <LRow dot color="#ff4444" label="High congestion" />
      </div>

      <div
        style={{
          position: "absolute",
          bottom: 30,
          left: 12,
          zIndex: 1000,
          background: "rgba(8,12,20,0.78)",
          border: "1px solid #1e2d42",
          borderRadius: 4,
          padding: "3px 10px",
          fontFamily: "'Space Mono',monospace",
          fontSize: 11,
          color: "#3d5a73",
          pointerEvents: "none",
        }}
      >
        Bangalore · 50 regions · click any node for details
      </div>
    </div>
  );
}

function LRow({ line, solid, dash, dot, icon, color, label }) {
  let ind;
  if (solid)
    ind = (
      <div
        style={{
          width: 22,
          height: 3,
          background: color,
          borderRadius: 2,
          flexShrink: 0,
        }}
      />
    );
  else if (dash)
    ind = (
      <div
        style={{
          width: 22,
          height: 0,
          borderTop: `2.5px dashed ${color}`,
          marginTop: 1,
          flexShrink: 0,
        }}
      />
    );
  else if (dot)
    ind = (
      <div
        style={{
          width: 12,
          height: 12,
          borderRadius: "50%",
          background: color,
          opacity: 0.55,
          flexShrink: 0,
        }}
      />
    );
  else if (icon)
    ind = (
      <div
        style={{
          width: 20,
          height: 20,
          borderRadius: "50%",
          background: `${color}33`,
          border: `1.5px solid ${color}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: icon === "22" ? 8 : 11,
          fontWeight: 900,
          color,
          fontFamily: "monospace",
          flexShrink: 0,
        }}
      >
        {icon === "22" ? "#" : icon}
      </div>
    );

  return (
    <div
      style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}
    >
      {ind}
      <span style={{ fontSize: 11, color: "#7a9ab8" }}>{label}</span>
    </div>
  );
}

/**
 * 50 regions arranged in a clean 7-column × 8-row grid over Bangalore.
 * Layout (reading left→right, top→bottom = north→south):
 *
 *   Col:  0(W)    1       2       3(C)    4       5       6(E)
 * Row 0: 13.09  ──────── North Bangalore ────────────────────────
 * Row 1: 13.05
 * Row 2: 13.01
 * Row 3: 12.97  ──────── Central / MG Road ────────────────────
 * Row 4: 12.93
 * Row 5: 12.89
 * Row 6: 12.85
 * Row 7: 12.81  ──────── South Bangalore ────────────────────────
 *
 * Longitudes: 77.48 (west) → 77.74 (east) in 6 steps
 */

const LATS = [13.09, 13.05, 13.01, 12.97, 12.93, 12.89, 12.85, 12.81];
const LNGS = [77.48, 77.521, 77.562, 77.603, 77.644, 77.685, 77.726];

// Build 56-entry grid, take first 50
const GRID = [];
for (let row = 0; row < 8; row++) {
  for (let col = 0; col < 7; col++) {
    const id = row * 7 + col;
    if (id >= 50) break;
    GRID.push({ id, lat: LATS[row], lng: LNGS[col] });
  }
}

export const REGION_COORDS = GRID;

export const BANGALORE_CENTER = [12.955, 12.603];
export const BANGALORE_ZOOM = 11;

// Human-readable names for each region
export const REGION_NAMES = {
  // Row 0 (north-west → north-east)
  0: "Hesaraghatta",
  1: "Yelahanka West",
  2: "Yelahanka",
  3: "Jakkur",
  4: "Thanisandra",
  5: "Horamavu",
  6: "KR Puram N",
  // Row 1
  7: "Peenya Ind.",
  8: "Jalahalli",
  9: "Hebbal",
  10: "Manyata Tech",
  11: "Kalyan Nagar",
  12: "ITPL Road",
  13: "Mahadevapura",
  // Row 2
  14: "Yeshwanthpur",
  15: "Rajajinagar",
  16: "Malleswaram",
  17: "Shivajinagar",
  18: "Banaswadi",
  19: "Domlur",
  20: "KR Puram",
  // Row 3 (central)
  21: "Vijayanagar",
  22: "Basavanagudi N",
  23: "Cubbon Park",
  24: "MG Road",
  25: "Indiranagar",
  26: "Marathahalli",
  27: "Whitefield W",
  // Row 4
  28: "Kengeri",
  29: "Uttarahalli",
  30: "Lalbagh",
  31: "Koramangala",
  32: "HSR Layout N",
  33: "Bellandur",
  34: "Sarjapur N",
  // Row 5
  35: "Kanakapura N",
  36: "Banashankari",
  37: "Jayanagar",
  38: "BTM Layout",
  39: "HSR Layout",
  40: "Bommanahalli",
  41: "Sarjapur",
  // Row 6
  42: "Kanakapura",
  43: "JP Nagar",
  44: "Jayanagar S",
  45: "Madiwala",
  46: "Electronic City N",
  47: "Bellandur S",
  48: "Varthur",
  // Row 7 (south — only 2 fit)
  49: "Electronic City",
};

export function getRegionName(id) {
  return REGION_NAMES[id] ?? `R${id}`;
}

// For path display: short label
export function shortName(id) {
  const n = REGION_NAMES[id];
  if (!n) return `R${id}`;
  return n.length > 12 ? n.slice(0, 11) + "…" : n;
}

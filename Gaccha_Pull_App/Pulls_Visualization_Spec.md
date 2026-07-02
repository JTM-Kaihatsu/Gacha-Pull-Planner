
# HSR Pull Planner — Pulls Visualization Spec

**Version:** 1.0
**Status:** Proposed
**Branch:** feat/pulls-visualization

---

## Overview

A stacked bar chart showing a sample of simulation runs. Each bar is one run. Bars are stacked by pull phase (C0, W1, C1, etc.), with refunds shown as a distinct sub-segment within each phase. Success and failure runs are visually separated.

---

## The X-Axis Problem

10,000 runs cannot be rendered on an x-axis — even at 1px per bar that's 10,000px wide and illegible. Solution: **sample and sort**.

### Approach

1. The backend samples a fixed number of runs (default: **300**) during simulation and returns their per-phase detail
2. Runs are sorted for display: **successes first (ascending pulls used) → failures second (descending pulls used)**
3. This creates a natural visual shape — short successful runs on the left tapering up, then a visible "cliff" into failed runs on the right
4. The user can optionally adjust sample size (50–500) via a slider

### Why 300?

- Renders cleanly at typical screen widths (~2–3px per bar)
- Enough to show the distribution shape clearly
- Keeps the API response payload small

---

## The Y-Axis

**Pulls used** — raw count, no refund subtraction. Refunds are shown *within* each phase segment as a distinct color block, so the bar height honestly represents total pulls spent while the refund portion is visually separated.

Y-axis scale: 0 → `total_pulls` (the player's input), with a dashed horizontal line at `total_pulls` as a "budget ceiling."

---

## Bar Structure

Each bar represents one simulation run, stacked bottom-to-top in pull order:

```
┌─────────────┐  ← total pulls used
│  C1 refunds │  (amber, hatched)
│  C1 pulls   │  (violet)
│  W1 refunds │  (orange, hatched)
│  W1 pulls   │  (amber)
│  C0 refunds │  (violet, hatched)
│  C0 pulls   │  (violet)
└─────────────┘
```

### Color Scheme

| Segment | Color |
|---|---|
| Char phase (net pulls) | Violet |
| Char phase (refunds) | Violet, hatched / lighter opacity |
| Weapon phase (net pulls) | Amber |
| Weapon phase (refunds) | Amber, hatched / lighter opacity |

### Success vs Failure Distinction

| Run type | Treatment |
|---|---|
| Success | No border, normal opacity |
| Failure | Red border outline, slightly reduced opacity |

---

## Data Requirements

### New backend tracking: per-phase detail

Currently `simulate_combo_verbose` tracks `refunded_pulls` as a single float for the entire run. This needs to change to **per-phase tracking**.

Each phase in the strategy produces:
```python
{
  "banner": "char",
  "copies": 1,
  "pulls_used": 74,     # gross pulls spent in this phase
  "refunds": 8.0,       # refunds received during this phase
  "net_pulls": 66.0,    # pulls_used - refunds (for reference)
  "success": True       # did this phase complete its copies goal
}
```

Refunds are treated as **immediately recycled within the phase** — if you got 8 refund pulls while grinding C0, those 8 pulls are shown as a sub-block of the C0 segment. This supports claims like "20 of the 80 pulls for C1 were refunds."

### Sample storage

`run_simulation_verbose` needs a `viz_sample_size` parameter (default 300). During the trial loop, sampled runs store their full `phase_detail` list. Non-sampled runs skip this to keep memory usage flat.

Sampling strategy: **reservoir sampling** — randomly replace entries as trials progress so the final sample is representative of all trials, not just the first 300.

### New API response field

```json
"viz_sample": [
  {
    "success": true,
    "total_pulls_used": 187,
    "phases": [
      {"label": "C0", "banner": "char", "pulls_used": 74, "refunds": 8.0},
      {"label": "W1", "banner": "weapon",   "pulls_used": 68, "refunds": 4.4},
      {"label": "C1", "banner": "char", "pulls_used": 45, "refunds": 5.0}
    ]
  },
  ...
]
```

Phase labels are derived from the strategy: char phases are labeled C0, C1, C2… (incrementing per copy obtained); Weapon phases are labeled W1, W2… The label reflects what was *completed* in that phase, not just the banner type.

---

## Frontend Component: `PullsChart`

### Props
```jsx
<PullsChart
  vizSample={result.viz_sample}
  strategy={form.strategy}
  totalPulls={form.total_pulls}
/>
```

### Library

**Recharts** — already React-native, composable, handles stacked bars cleanly. Add via:
```bash
npm install recharts
```

No D3 needed at this stage.

### Render Logic

1. Sort `vizSample`: successes ascending by `total_pulls_used`, then failures descending
2. For each run, build a flat data object with one key per phase segment:
   ```js
   { runId: 0, success: true, C0_net: 66, C0_refund: 8, W1_net: 63.6, W1_refund: 4.4, ... }
   ```
3. Render a `<BarChart>` with one `<Bar>` per segment key, stacked
4. Color each bar by banner type; use hatching or opacity for refund segments
5. `<ReferenceLine>` at `y = totalPulls` for the budget ceiling
6. Tooltip on hover: show run index, success/fail, pulls per phase, refunds per phase

### Placement in UI

Below the stats grid, above the AI analysis block. Collapsible (like failure stats) to keep the page manageable.

---

## Implementation Order

1. **`simulation.py`** — add per-phase refund tracking; add reservoir sampling; expose `viz_sample` in `stats_summary`
2. **`main.py`** — pass `viz_sample` through in the response
3. **`frontend`** — `npm install recharts`; build `PullsChart` component; wire into `App.jsx`

---

## Open Questions

| Question | Current thinking |
|---|---|
| What if all 300 sampled runs are successes (very high success rate)? | Still useful — shows pull cost distribution among successes |
| Should refunds be capped to the phase they occurred in, or can they spill across phase boundaries? | Cap to phase — cleaner visualization, matches the "20 of 80 pulls were refunds" framing |
| Should the sample size slider be visible to the user or just a dev config? | Dev config for now, user-facing later |

---

*Last updated: 2026-06-24*

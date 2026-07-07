# Gacha Pull Simulator

A Monte Carlo planning tool for gacha games. Tell it how many pulls you have, your
current pity, and what you're trying to pull for — it runs 10,000 simulated attempts
and tells you your odds of success, what a typical failure looks like, and a blunt,
plain-English verdict on whether your goal is realistic.

Game-agnostic by design: the pity curves are fully configurable, so the same engine
models Honkai: Star Rail, Genshin Impact, Zenless Zone Zero, and similar dual-banner
(character + weapon) systems.

> **Live demo:** _TODO — add deployed URL here_
> **Screenshot:** _TODO — add a screenshot/GIF of the results view + distribution chart_
<!-- ![Gacha Pull Simulator](docs/screenshot.png) -->

---

## The product story

**Problem.** Gacha players spend real money and months of saved currency chasing
limited banners, but the systems are deliberately opaque — layered pity, 50/50 and
75/25 coin-flips, soft-pity ramps. "Do I have enough to get what I want?" is a
genuinely hard probability question, and existing tools mostly answer "here's the
average pity," which doesn't tell you your actual odds or what to do if you're short.

**Who it's for.** Players planning a specific pull goal ("2 copies of the character
plus their signature weapon") who want a confidence level and a strategy, not a
spreadsheet.

**What it does differently.**
- **Distributions, not averages.** It reports a *success rate* across 10,000 trials
  and the most common *failure states* ("you ran out after 1 of 2 characters, 30% of
  the time") — the information you actually need to decide whether to pull now or wait.
- **Strategy-aware.** Pull order matters because pity carries across a banner. The
  tool models an ordered strategy (e.g. character → weapon → character) rather than
  treating goals as independent.
- **A verdict, in plain English.** The stats are piped through an LLM prompt tuned to
  answer one question like a friend would: *doable, tight, or a stretch — and how many
  more pulls would close the gap?*

**Key product/technical decisions.**
- Chose **Monte Carlo simulation** over a closed-form probability model: the pity +
  guarantee + refund interactions are painful to express analytically but trivial to
  simulate, and simulation naturally yields the full outcome distribution.
- Made pity rates **configurable** rather than hard-coding one game — a deliberate
  scope expansion to broaden the audience beyond a single title.
- Kept the simulation core **pure and framework-free** so it's testable in isolation
  and reusable outside the web app.

**What I'd do next.** Persisted scenarios and shareable result links; per-game presets
so users don't hand-enter pity curves; replacing the LLM verdict with a cheaper
templated summary for the common cases; and frontend tests around the chart.

---

## Architecture

```
React + Vite frontend  ──POST /analyze──▶  FastAPI backend
  (form, chart, results)                     ├─ simulation.py  (Monte Carlo engine, pure)
                                             └─ analyzer.py    (LLM natural-language verdict)
```

- **`simulation.py`** — the engine. Runs 10,000 trials of an ordered pull strategy,
  modeling soft/hard pity, 50/50 & 75/25 outcomes, guarantees, and 4★ refunds. Pure
  functions, no I/O.
- **`main.py`** — FastAPI app exposing a single `POST /analyze` endpoint.
- **`analyzer.py`** — turns the aggregated stats into a short plain-English analysis
  via the OpenAI API.
- **`frontend/`** — React 19 + Vite + Tailwind UI, with a Recharts pull-distribution
  visualization.

## Tech stack

| Layer | Tools |
|-------|-------|
| Backend | Python, FastAPI, Pydantic, NumPy, pandas |
| Frontend | React 19, Vite, Tailwind CSS v4, Recharts |
| AI | OpenAI Chat Completions |
| Tooling | oxlint, pytest |

---

## Quick start

The app is two processes: the FastAPI backend (port 8000) and the Vite frontend
(port 5173), which calls the backend.

**1. Backend** — from the project root:

```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env      # required for the /analyze verdict
uvicorn main:app --reload
```

Interactive API docs are then available at http://localhost:8000/docs (Swagger UI).

**2. Frontend** — in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (http://localhost:5173) and click **Simulate**.

---

## API reference

### `POST /analyze`

Runs the simulation and returns aggregated stats plus an AI analysis.

**Request body**

```json
{
  "total_pulls": 180,
  "start_char_pity": 30,
  "start_char_guarantee": true,
  "start_weapon_pity": 10,
  "start_weapon_guarantee": false,
  "strategy": [
    { "banner": "char",   "copies": 1 },
    { "banner": "weapon", "copies": 1 },
    { "banner": "char",   "copies": 1 }
  ],
  "full_4star_chars": true,
  "char_pity_config":   { "base_rate": 0.006, "soft_pity_start": 73, "hard_pity": 90 },
  "weapon_pity_config": { "base_rate": 0.008, "soft_pity_start": 65, "hard_pity": 80 }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `total_pulls` | int | Total pulls available |
| `start_char_pity` | int | Current pity on the character banner |
| `start_char_guarantee` | bool | Next 5★ character is guaranteed the featured one (no 50/50) |
| `start_weapon_pity` | int | Current pity on the weapon banner |
| `start_weapon_guarantee` | bool | Next 5★ weapon is guaranteed the featured one (no 75/25) |
| `strategy` | array | **Ordered** phases: each `{ "banner": "char" \| "weapon", "copies": int }`. Order matters — pity carries across phases. |
| `full_4star_chars` | bool | If true, duplicate 4★ characters are treated as refunded pulls |
| `char_pity_config` | object | `base_rate`, `soft_pity_start`, `hard_pity` for the character banner |
| `weapon_pity_config` | object | Same three fields for the weapon banner |

**Response (abridged)**

```json
{
  "analysis_text": "You need to win both 50/50s...",
  "trials": 10000,
  "stats_summary": {
    "success_rate": "46.75%",
    "avg_pity_char": 74.2,
    "avg_pity_weapon": 69.8,
    "successes_char_win_rate": "89.00%",
    "successes_weapon_win_rate": "83.00%",
    "avg_leftover_pulls_on_success": 6.3,
    "avg_refund_success": 4.0,
    "failure_char_win_rate": "47.00%",
    "failure_weapon_win_rate": "42.00%",
    "avg_leftover_pulls_on_failure": 2.1,
    "avg_refund_fail": 3.0,
    "most_common_failure_state": { "chars": 1, "weapons": 0, "pct": 31.2 },
    "failure_state_distribution": [ { "chars": 1, "weapons": 0, "pct": 31.2 } ],
    "correlation_stats": { "total_successes": 4675, "total_failures": 5325 },
    "viz_sample": [ { "trial": 1, "success": true, "total_pulls_used": 172, "phases": [] } ]
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `analysis_text` | string | LLM-generated plain-English verdict |
| `trials` | int | Number of Monte Carlo trials (10,000) |
| `success_rate` | string | % of trials that met the full goal |
| `avg_pity_char` / `avg_pity_weapon` | float | Avg pity at which the 5★ hit, on successful runs |
| `successes_*_win_rate` / `failure_*_win_rate` | string | 50/50 (char) and 75/25 (weapon) win rates, split by outcome |
| `avg_leftover_pulls_on_success` / `_on_failure` | float | Pulls left over |
| `avg_refund_success` / `avg_refund_fail` | float | Avg pulls refunded from duplicate 4★s |
| `most_common_failure_state` | object\|null | Most frequent shortfall among failed runs |
| `failure_state_distribution` | array | Top failure shortfalls with percentages |
| `correlation_stats` | object | Success/failure counts broken down by contributing factors |
| `viz_sample` | array | Per-run phase breakdown sampled for the distribution chart |

Errors return `500` with a `{ "detail": "..." }` body.

---

## Testing

```bash
pip install pytest httpx
pytest
```

Covers the simulation output structure, the analyzer (mocked), and the `/analyze`
endpoint contract (mocked).

---

## Notes & limitations

- The OpenAI key lives in `.env` (gitignored) — never commit it. `/analyze` calls the
  OpenAI API, so it needs a valid key to return the verdict; the raw simulation stats
  do not.
- Pity/guarantee/refund rates model common gacha systems but are simplifications;
  tune the pity configs to match a specific game.
- No frontend test suite yet (see *What I'd do next*).

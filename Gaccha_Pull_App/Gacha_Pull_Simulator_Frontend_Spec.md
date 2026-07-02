# HSR Pull Planner — Frontend Technical Spec

**Version:** 1.1
**Status:** Proposed
**Owner:** Jeremy

---

## Objectives

| Goal | Description |
|---|---|
| Primary | Let players define a pull strategy and see simulated odds |
| Secondary | Surface AI analysis in plain language, no stat literacy required |
| Out of Scope | User accounts, saved sessions, banner schedules |

---

## Pages / Views

### P1 — Planner Form
The main input screen. Player fills in their current state and pull goal, then submits.

**Fields:**
- Total Pulls *(integer)*
- Character Pity *(0–90)*
- Character Guarantee *(toggle)*
- Weapon Pity *(0–80)*
- Weapon Guarantee *(toggle)*
- How many character copies? *(1–6, maps to E0–E5)*
- How many Weapons? *(0–5, maps to none through W5)*
- **[Conditional]** Weapon ordering question — see Strategy Logic below

### P2 — Results Panel
Shown after simulation completes. Lives on the same page — no navigation.

**Fields:**
- Success Rate *(prominent)*
- Avg Pity — Char / Weapon
- Win Rate — 50/50 and 75/25
- Leftover Pulls
- AI Analysis Text
- *[Optional v2]* Success vs Failure toggle

### P3 — Loading State
Shown while simulation runs (~2–5s). Must set expectations — this is not instant.

- Spinner or progress bar
- Copy: *"Running 10,000 simulations…"*

---

## Strategy Logic (Form → API)

The player never writes JSON. The form collects two numbers and an optional ordering choice, then builds the `strategy` array internally before calling the API.

### Step 1 — Collect inputs
- **Char copies** (`desired_chars`): 1–6
- **Weapon copies** (`desired_weapons`): 0–5

### Step 2 — Determine if ordering question is needed

Show the Weapon ordering question **only when**:
- `desired_chars > 1` **AND**
- `desired_weapons >= 1`

If `desired_chars == 1` or `desired_weapons == 0`, no ordering decision exists — build the strategy directly.

### Step 3 — Weapon Ordering Question (conditional)

> *"When do you want to pull your first Weapon?"*

Radio buttons generated dynamically based on `desired_chars`:

| Option | Meaning |
|---|---|
| After E0 | Pull Weapon before going for E1 |
| After E1 | Pull Weapon before going for E2 |
| After E2 | Pull Weapon before going for E3 |
| *(continues to second-to-last copy)* | … |

The user picks the insertion point for their **first** Weapon copy.

### Step 4 — Multi-Weapon Rule

If `desired_weapons > 1`, all copies beyond the first are **always appended to the end** of the strategy, after all character copies are obtained. The ordering question only governs where the first Weapon sits.

> *Example: 3 char copies, 3 Weapon copies, first Weapon after E0*
> → E0 → W1 → E1 → E2 → W2 → W3

### Step 5 — Strategy Array Mapping

| desired_chars | desired_weapons | Weapon after | strategy array |
|---|---|---|---|
| 1 | 0 | — | `[{char,1}]` |
| 1 | 1 | — | `[{char,1},{weapon,1}]` |
| 1 | 2 | — | `[{char,1},{weapon,1},{weapon,1}]` |
| 2 | 1 | After E0 | `[{char,1},{weapon,1},{char,1}]` |
| 2 | 1 | After E1 | `[{char,2},{weapon,1}]` |
| 3 | 1 | After E0 | `[{char,1},{weapon,1},{char,2}]` |
| 3 | 1 | After E1 | `[{char,2},{weapon,1},{char,1}]` |
| 3 | 1 | After E2 | `[{char,3},{weapon,1}]` |
| 3 | 2 | After E0 | `[{char,1},{weapon,1},{char,2},{weapon,1}]` |
| 3 | 3 | After E1 | `[{char,2},{weapon,1},{char,1},{weapon,2}]` |

**Rule:** `weapon insertion point` splits char copies into two groups. Remaining Weapon copies (`desired_weapons - 1`) are appended at the end.

---

## Key Components

| Component | Responsibility | Notes |
|---|---|---|
| `StrategyBuilder` | Renders the char/Weapon number inputs + conditional ordering question | Builds strategy array internally |
| `PityInputs` | Numeric inputs for char/weapon pity + guarantee toggles | Validate 0–90 (char), 0–80 (Weapon) |
| `StatCard` | Displays one stat with label and value | Reused across results grid |
| `AnalysisBlock` | Renders `analysis_text` from GPT | Plain text, no markdown parsing needed |
| `ApiService` | Wraps `POST /analyze`, handles errors | Single fetch call, no state library needed |

---

## Data Flow

```
User fills form
  → StrategyBuilder assembles strategy[]
  → Submit fires POST /analyze
  → FastAPI runs Monte Carlo simulation
  → GPT-4o generates analysis
  → Results rendered on page
```

---

## Proposed Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| UI | React 18 | Component model fits the conditional form logic |
| Build | Vite | Fast dev server, minimal config |
| Styling | Tailwind CSS | No overhead for a single-page tool |
| HTTP | fetch API | One endpoint, no Axios needed |
| State | useState only | No global state library required |

---

## Future: Pull Calendar Integration

The calendar feature (currently separate, not connected to FastAPI) will eventually allow players to use a projected jade count as their `total_pulls` input.

**Planned approach:**
- `total_pulls` on the API is source-agnostic — FastAPI doesn't care where the number comes from
- The form will offer two input modes:
  - *Enter pulls manually* → numeric field
  - *Use Pull Calendar* → date range / jade count field that auto-calculates and populates `total_pulls`
- When the calendar is ready to connect, only the React form layer changes — no API modifications required

---

## Risks & Considerations

| Risk | Detail |
|---|---|
| **CORS** | FastAPI must allowlist the React origin. Configured via environment variable (`ALLOWED_ORIGINS`) so local dev and production domains are handled separately without code changes. |
| **Latency** | 10,000 trials + GPT call may take 5–10s. Loading UX must set expectations or users will assume it's broken. |
| **OpenAI Key** | Must never be exposed to the browser — stays server-side in FastAPI only. |
| **Input Validation** | Pity values out of range (char > 90, Weapon > 80) should be caught client-side before hitting the API. |
| **Whale edge cases** | Multi-Weapon strategies (W2+) are supported: extra copies always append to end of strategy. Weapon input capped at 5. |

---

## Deployment Plan (Post-Frontend)

1. Deploy React to **Vercel** (free subdomain, e.g. `hsrplanner.vercel.app`)
2. Deploy FastAPI to **Railway** or **Render** (free tier, e.g. `hsrplanner-api.railway.app`)
3. Set `ALLOWED_ORIGINS` on the FastAPI host to match the Vercel subdomain
4. Purchase custom domain when ready for public launch

---

*Last updated: 2026-06-24*

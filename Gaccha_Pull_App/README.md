
# HSR Simulation API

Simulate Honkai: Star Rail banner pulls and get AI‑generated analysis using OpenAI.

## Quick Start

```bash
git clone HSR-Pull-Planner
cd hsr_sim_api
pip install -r requirements.txt   # or manually:
pip install fastapi uvicorn openai numpy python-dotenv
echo "OPENAI_API_KEY=sk-..." > .env
uvicorn main:app --reload
```

## Endpoint

### `POST /analyze`

Submit banner simulation input to receive AI analysis + detailed stats.

### Request Body (JSON)

```json
{
  "total_pulls": 180,
  "desired_chars": 2,
  "desired_lcs": 1,
  "start_char_pity": 30,
  "start_char_guarantee": true,
  "start_lc_pity": 10,
  "start_lc_guarantee": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `total_pulls` | int | How many total pulls are available |
| `desired_chars` | int | Number of rate-up characters you want |
| `desired_lcs` | int | Number of rate-up Light Cones you want |
| `start_char_pity` | int | Current pity count for character banner |
| `start_char_guarantee` | bool | If the next rate-up char is guaranteed |
| `start_lc_pity` | int | Current pity count for LC banner |
| `start_lc_guarantee` | bool | If the next rate-up LC is guaranteed |

---

### Example Response

```json
{
  "analysis_text": "Based on 50,000 simulations, your odds are decent...",
  "trials": 50000,
  "stats_summary": {
    "success_rate": 46.75,
    "avg_pity_char": 74.2,
    "avg_pity_lc": 69.8,
    "successes_char_win_rate": 0.89,
    "successes_lc_win_rate": 0.83,
    "avg_leftover_pulls_on_success": 6.3,
    "avg_refund_success": 0.0,
    "failure_char_win_rate": 0.47,
    "failure_lc_win_rate": 0.42,
    "avg_leftover_pulls_on_failure": 2.1,
    "avg_refund_fail": 0.0
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `analysis_text` | string | GPT-generated explanation of your odds and strategy |
| `trials` | int | Number of Monte Carlo simulations run |
| `stats_summary` | object | Core numerical stats from simulation |
| └ `success_rate` | float | % of trials that met or exceeded goals |
| └ `avg_pity_char` | float | Avg pity reached for characters |
| └ `avg_pity_lc` | float | Avg pity reached for lightcones |
| └ `successes_char_win_rate` | float | % of character pulls that hit the banner target (success only) |
| └ `successes_lc_win_rate` | float | Same, for lightcones |
| └ `avg_leftover_pulls_on_success` | float | Pulls remaining after success |
| └ `avg_refund_success` | float | Placeholder for future refunds |
| └ `failure_char_win_rate` | float | Rate of banner wins among character pulls (failures only) |
| └ `failure_lc_win_rate` | float | Same, for lightcones |
| └ `avg_leftover_pulls_on_failure` | float | Pulls leftover in failed runs |
| └ `avg_refund_fail` | float | Placeholder for future refunds |

---

## Testing

```bash
pip install pytest httpx
pytest
```

Tests include:
- Simulation output structure
- Analyzer behavior (mocked)
- API endpoint contract (mocked)

---

## Notes

- The OpenAI key should be stored in a `.env` file or environment variable (never exposed).
- This API is backend-only and assumes a separate frontend.

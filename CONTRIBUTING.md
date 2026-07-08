
# CONTRIBUTING.md: Developer Setup & File Structure

Welcome to the **Gacha Pull Simulator** codebase! This doc explains the structure, responsibilities of each file, and how to securely run and contribute to the backend.

---

## Project Overview

This repo simulates gacha pulls for configurable gacha banners, then generates an AI-powered analysis using OpenAI’s GPT models.

- Built with **FastAPI** for a REST API
- Uses **NumPy** for simulation
- Loads secrets from `.env`
- Returns structured + natural-language results

---

## Install Dependencies

Make sure you have **Python 3.8 or higher** and `pip` installed.

### Base install (for running the app)

```bash
pip install fastapi uvicorn openai numpy python-dotenv
```

### Development install (includes testing tools)

```bash
pip install pytest httpx
```

### Full install (base + test in one)

```bash
pip install fastapi uvicorn openai numpy python-dotenv pytest httpx
```

> Note: This project uses the latest `openai>=1.0.0` SDK. If you get an error like `cannot import name 'OpenAI'`, run:
> ```bash
> pip install --upgrade openai
> ```

---

## File Structure

| File / Folder      | Purpose |
|--------------------|---------|
| `main.py`          | FastAPI server, exposes the `/analyze` endpoint |
| `simulation.py`    | Core logic for character & weapon pull simulation (pure functions, no I/O) |
| `analyzer.py`      | Formats and sends OpenAI API requests, returns summaries |
| `config.py`        | Loads the OpenAI API key from `.env` (never hardcoded) |
| `requirements.txt` | All required packages for running or deploying the app |
| `README.md`        | Basic usage, endpoints, and response schema |
| `tests/`           | Contains Pytest-based test cases for simulation and API behavior |


---

## Secrets

Create a `.env` file at the root of the project:

```
OPENAI_API_KEY=your_openai_key_here
```

> Never commit your actual `.env` file. Ensure this file is included in gitignore

---

## Testing

```bash
pytest
```

Includes:
- `test_simulation.py` – unit tests for simulation outputs
- `test_api.py` – FastAPI integration test using mocked dependencies

---

## Endpoint Overview

### `POST /analyze`

Accepts JSON with your pull plan and returns:
- `analysis_text`: GPT-generated summary
- `trials`: number of simulations
- `stats_summary`: all key probability metrics

See `README.md` for full request/response schema.

--

## Key Concepts & Variables

| Variable | Where | What it does |
|----------|-------|---------------|
| `total_pulls` | request | How many pulls player can use |
| `desired_chars`, `desired_weapons` | request | Target number of characters/weapons |
| `start_char_pity`, `start_weapon_pity` | request | Current pity counters |
| `start_char_guarantee`, `start_weapon_guarantee` | request | True = banner is guaranteed next pull |
| `run_simulation_verbose(...)` | `simulation.py` | Runs the full simulation logic |
| `analyze_sim_result(...)` | `analyzer.py` | Sends result to OpenAI for summary |
| `get_openai_api_key()` | `config.py` | Safely retrieves the API key |

---

## Getting Started Recap

```bash
git clone Gacha-Pull-Planner
cd backend
cp .env.example .env  # then add your OpenAI key
pip install fastapi uvicorn openai numpy python-dotenv
uvicorn main:app --reload
```
You can then open Swagger and test the POST in `http://localhost:8000/docs` after running the `uvicorn main:app --reload` command

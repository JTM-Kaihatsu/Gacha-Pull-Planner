
"""main.py
FastAPI entrypoint exposing `/analyze`.
"""
import logging
from typing import List, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from simulation import run_simulation_verbose
from analyzer import analyze_sim_result
from summary import build_summary
from advisor import run_advisor, ADVISOR_TRIALS
from config import get_openai_api_key, get_allowed_origins

# Log through uvicorn's error logger so messages show in the console and Render
# logs. The AI steps are wrapped in graceful fallbacks; without this, a failure
# is invisible (the endpoint just returns status "unavailable").
logger = logging.getLogger("uvicorn.error")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

class PhaseRequest(BaseModel):
    banner: Literal["char", "weapon"]
    copies: int = Field(..., ge=1)

class PityConfig(BaseModel):
    base_rate: float = 0.006
    soft_pity_start: int = 73
    hard_pity: int = 90

class SimRequest(BaseModel):
    total_pulls: int
    start_char_pity: int
    start_char_guarantee: bool = False
    start_weapon_pity: int
    start_weapon_guarantee: bool = False
    strategy: List[PhaseRequest] = Field(..., min_length=1)
    full_4star_chars: bool = True
    enable_ai_analysis: bool = False
    char_pity_config: PityConfig = PityConfig()
    weapon_pity_config: PityConfig = PityConfig(base_rate=0.008, soft_pity_start=65, hard_pity=80)

class AdviseRequest(SimRequest):
    question: str = Field(..., min_length=1, max_length=500)

@app.post("/analyze")
def analyze(req: SimRequest):
    # The simulation is the core result — a failure here is a real 500.
    try:
        strategy = [{"banner": p.banner, "copies": p.copies} for p in req.strategy]

        stats = run_simulation_verbose(
            total_pulls=req.total_pulls,
            strategy=strategy,
            start_char_pity=req.start_char_pity,
            start_char_guarantee=req.start_char_guarantee,
            start_weapon_pity=req.start_weapon_pity,
            start_weapon_guarantee=req.start_weapon_guarantee,
            full_4star_chars=req.full_4star_chars,
            char_pity_config=req.char_pity_config.model_dump(),
            weapon_pity_config=req.weapon_pity_config.model_dump(),
        )
    except Exception as exc:
        logger.exception("simulation failed in /analyze")
        raise HTTPException(status_code=500, detail=str(exc))

    # The (paid, opt-in) LLM verdict degrades gracefully: a failure here must
    # never take down the simulation result. Report status so the UI can show
    # an honest "unavailable" message instead of silently dropping the section.
    analysis_text = None
    if not req.enable_ai_analysis:
        analysis_status = "disabled"
    else:
        try:
            analysis_text = analyze_sim_result(stats)
            analysis_status = "ok"
        except Exception as exc:
            # OpenAI rate-limit / quota-cap surfaces as HTTP 429. Log the real
            # cause (missing key, bad model, etc.) so it is not invisible.
            logger.exception("AI verdict failed")
            analysis_status = "rate_limited" if getattr(exc, "status_code", None) == 429 else "unavailable"

    return {
            # Deterministic, always-computed plain-language read (no model, no cost).
            "summary": build_summary(stats),
            "analysis_text": analysis_text,
            "analysis_status": analysis_status,
            "trials": stats["trials"],
            "stats_summary": {
                "success_rate": stats["success_rate"],
                "avg_pity_char": stats["avg_pity_char"],
                "avg_pity_weapon": stats["avg_pity_weapon"],
                "successes_char_win_rate": stats["successes_char_win_rate"],
                "successes_weapon_win_rate": stats["successes_weapon_win_rate"],
                "avg_leftover_pulls_on_success": stats["avg_leftover_pulls_on_success"],
                "avg_refund_success": stats["avg_refund_success"],
                "failure_char_win_rate": stats["failure_char_win_rate"],
                "failure_weapon_win_rate": stats["failure_weapon_win_rate"],
                "avg_leftover_pulls_on_failure": stats["avg_leftover_pulls_on_failure"],
                "avg_refund_fail": stats["avg_refund_fail"],
                "most_common_failure_state": stats["most_common_failure_state"],
                "failure_state_distribution": stats["failure_state_distribution"],
                "correlation_stats": stats["correlation_stats"],
                "viz_sample": stats["viz_sample"],
            },
        }


@app.post("/advise")
def advise(req: AdviseRequest):
    # Open-ended follow-up advisor: runs the baseline, then an agentic loop that
    # can re-run the sim to answer the question. Opt-in per question (it costs an
    # OpenAI call), and it degrades gracefully like the /analyze verdict.
    strategy = [{"banner": p.banner, "copies": p.copies} for p in req.strategy]
    baseline_params = {
        "total_pulls": req.total_pulls,
        "strategy": strategy,
        "start_char_pity": req.start_char_pity,
        "start_char_guarantee": req.start_char_guarantee,
        "start_weapon_pity": req.start_weapon_pity,
        "start_weapon_guarantee": req.start_weapon_guarantee,
        "full_4star_chars": req.full_4star_chars,
        "char_pity_config": req.char_pity_config.model_dump(),
        "weapon_pity_config": req.weapon_pity_config.model_dump(),
    }

    # The baseline run is deterministic core work; a failure here is a real 500.
    try:
        baseline_stats = run_simulation_verbose(**baseline_params, trials=ADVISOR_TRIALS)
    except Exception as exc:
        logger.exception("baseline simulation failed in /advise")
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        answer = run_advisor(baseline_params, baseline_stats, req.question)
        status = "ok"
        logger.info("advisor ok | question=%r | answer=%r", req.question[:120], (answer or "")[:200])
    except Exception as exc:
        logger.exception("advisor failed | question=%r", req.question[:120])
        answer = None
        status = "rate_limited" if getattr(exc, "status_code", None) == 429 else "unavailable"

    return {"answer": answer, "status": status}

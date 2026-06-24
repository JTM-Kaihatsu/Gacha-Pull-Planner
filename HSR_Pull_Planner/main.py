
"""main.py
FastAPI entrypoint exposing `/analyze`.
"""
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from simulation import run_simulation_verbose
from analyzer import analyze_sim_result
from config import get_openai_api_key, get_allowed_origins

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

class PhaseRequest(BaseModel):
    banner: str   # "char" or "lc"
    copies: int

class SimRequest(BaseModel):
    total_pulls: int
    start_char_pity: int
    start_char_guarantee: bool = False
    start_lc_pity: int
    start_lc_guarantee: bool = False
    strategy: List[PhaseRequest]

@app.post("/analyze")
def analyze(req: SimRequest):
    try:
        strategy = [{"banner": p.banner, "copies": p.copies} for p in req.strategy]

        stats = run_simulation_verbose(
            total_pulls=req.total_pulls,
            strategy=strategy,
            start_char_pity=req.start_char_pity,
            start_char_guarantee=req.start_char_guarantee,
            start_lc_pity=req.start_lc_pity,
            start_lc_guarantee=req.start_lc_guarantee,
        )
        analysis_text = analyze_sim_result(stats)

        return {
            "analysis_text": analysis_text,
            "trials": stats["trials"],
            "stats_summary": {
                "success_rate": stats["success_rate"],
                "avg_pity_char": stats["avg_pity_char"],
                "avg_pity_lc": stats["avg_pity_lc"],
                "successes_char_win_rate": stats["successes_char_win_rate"],
                "successes_lc_win_rate": stats["successes_lc_win_rate"],
                "avg_leftover_pulls_on_success": stats["avg_leftover_pulls_on_success"],
                "avg_refund_success": stats["avg_refund_success"],
                "failure_char_win_rate": stats["failure_char_win_rate"],
                "failure_lc_win_rate": stats["failure_lc_win_rate"],
                "avg_leftover_pulls_on_failure": stats["avg_leftover_pulls_on_failure"],
                "avg_refund_fail": stats["avg_refund_fail"],
                "most_common_failure_state": stats["most_common_failure_state"],
                "failure_state_distribution": stats["failure_state_distribution"],
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

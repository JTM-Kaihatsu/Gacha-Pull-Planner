
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
    banner: str   # "char" or "weapon"
    copies: int

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
    strategy: List[PhaseRequest]
    full_4star_chars: bool = True
    char_pity_config: PityConfig = PityConfig()
    weapon_pity_config: PityConfig = PityConfig(base_rate=0.008, soft_pity_start=65, hard_pity=80)

@app.post("/analyze")
def analyze(req: SimRequest):
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
        analysis_text = analyze_sim_result(stats)

        return {
            "analysis_text": analysis_text,
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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

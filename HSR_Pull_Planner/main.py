
"""main.py
FastAPI entrypoint exposing `/analyze`.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from simulation import run_simulation_verbose
from analyzer import analyze_sim_result
from config import get_openai_api_key

app = FastAPI()

class SimRequest(BaseModel):
    total_pulls: int
    desired_chars: int
    desired_lcs: int
    start_char_pity: int
    start_char_guarantee: bool = False
    start_lc_pity: int
    start_lc_guarantee: bool = False

@app.post("/analyze")
def analyze(req: SimRequest):
    try:
        stats = run_simulation_verbose(
            total_pulls=req.total_pulls,
            desired_chars=req.desired_chars,
            desired_lcs=req.desired_lcs,
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
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

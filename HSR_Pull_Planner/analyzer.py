"""analyzer.py
Takes simulation statistics and produces a natural‑language summary
using the OpenAI Chat Completion API.
"""
from typing import Dict
import openai
from openai import OpenAI
from config import get_openai_api_key, get_model



def describe_goal(sim_stats):
    chars = sim_stats["desired_characters"]
    lcs = sim_stats["desired_lightcones"]

    goal_label = f"E{chars - 1}S{lcs}"
    char_phrase = (
        "one copy of the limited 5★ character"
        if chars == 1 else
        f"{chars} copies of the limited 5★ character"
    )
    lc_phrase = (
        "no signature light cone"
        if lcs == 0 else
        "one signature 5★ light cone"
        if lcs == 1 else
        f"{lcs} signature 5★ light cones"
    )

    return goal_label, f"{char_phrase} and {lc_phrase}"



def analyze_sim_result(sim_stats, trials=50000, model: str = None):

    client = OpenAI(api_key=get_openai_api_key())
    model = model or get_model()

    goal_label, goal_description = describe_goal(sim_stats)

    system_prompt = (
        "You are a blunt, no-fluff Honkai Star Rail pull advisor. "
        "Give short, direct answers. No markdown, no headers, no bullet points. "
        "The reader knows how pity works — skip the basics and get to the point."
    )

    most_common = sim_stats.get("most_common_failure_state")
    if most_common:
        failure_state_line = (
            f"Most common failure state: ran out after getting {most_common['chars']} of {sim_stats['desired_characters']} "
            f"character copies and {most_common['lcs']} of {sim_stats['desired_lightcones']} LCs "
            f"({most_common['pct']}% of failures)."
        )
    else:
        failure_state_line = "No failure data available."

    cs = sim_stats.get("correlation_stats", {})
    s  = cs.get("total_successes", 0)
    f  = cs.get("total_failures", 0)

    def fmt(val, total):
        if val is None or total == 0:
            return "N/A"
        return f"{val} of {total}"

    correlation_lines = f"""
        Factor breakdown (counts out of {s} successes / {f} failures):
        - Early char pity hit (< pull {cs.get('early_pity_threshold', 70)}): {fmt(cs.get('early_char_pity_in_successes'), s)} successes | {fmt(cs.get('early_char_pity_in_failures'), f)} failures
        - Won all needed char 50/50s: {fmt(cs.get('all_5050s_won_in_successes'), s)} successes | {fmt(cs.get('all_5050s_won_in_failures'), f)} failures
        - Won all needed LC 75/25s: {fmt(cs.get('all_lc_75s_won_in_successes'), s)} successes | {fmt(cs.get('all_lc_75s_won_in_failures'), f)} failures
    """

    user_prompt = """
        Goal: {goal_label} — {goal_description}
        Pulls available: {initial_pulls} | Char pity: {start_char_pity} (guarantee: {start_char_guarantee}) | LC pity: {start_lc_pity} (guarantee: {start_lc_guarantee})

        Simulation results ({trials:,} trials):
        Success rate: {success_rate}
        Char 50/50 win rate (success): {successes_char_win_rate} | (fail): {failure_char_win_rate}
        LC 75/25 win rate (success): {successes_lc_win_rate} | (fail): {failure_lc_win_rate}
        Avg char pity on success: {avg_pity_char} | Avg LC pity on success: {avg_pity_lc}
        Avg leftover pulls — success: {avg_leftover_pulls_on_success} | fail: {avg_leftover_pulls_on_failure}
        Avg refunds — success: {avg_refund_success} | fail: {avg_refund_fail}
        {failure_state_line}
        {correlation_lines}

        Write a short, blunt analysis in plain English — 3 to 5 sentences max, no headers, no bullet points.
        Answer exactly these questions in order:
        1. How many 50/50s and 75/25s did the player need to win, and do the counts show they had to win most/all of them or was losing one survivable?
        2. Did early pity hits matter — were they noticeably more common in successful runs vs failed ones?
        3. Use the most common failure state to describe what losing typically looks like.
        4. One sentence verdict: doable, tight, or a stretch — and if tight or a stretch, how many more pulls would help?
        Do not use jargon. Write like you're texting a friend who plays the game.
        """.format(goal_label=goal_label, goal_description=goal_description,
                   failure_state_line=failure_state_line,
                   correlation_lines=correlation_lines,
                   **sim_stats)


    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ]
    )

    return response.choices[0].message.content.strip()

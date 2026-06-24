
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
        "You are a Honkai Star Rail pull simulation assistant who explains probability and gacha mechanics. "
        "Explain statistical outcomes clearly, concisely, and with emphasis on player decision-making. "
        "Assume the reader is already familiar with HSR's pity systems."
    )

    user_prompt = """
        The simulation goal was to reach {goal_label}: {goal_description}.

        The player started with:
        - Initial Pulls: {initial_pulls}
        - Character Pity: {start_char_pity} | Guarantee: {start_char_guarantee}
        - LC Pity: {start_lc_pity} | Guarantee: {start_lc_guarantee}

        Simulation ran for {trials:,} trials. Here are the results:
        - Success Rate: {success_rate}
        - Character 50/50 Wins for Successful Runs: {successes_char_win_rate}
        - Lightcone 75/25 Wins for Successful Runs: {successes_lc_win_rate}
        - Avg Leftover Pulls for Successful Runs: {avg_leftover_pulls_on_success}
        - Avg Character Pity for Successful Runs: {avg_pity_char}
        - Avg Light Cone Pity for Successful Runs: {avg_pity_lc}
        - Avg Refund for Successful Runs: {avg_refund_success}

        - Character 50/50 Wins for Failed Runs: {failure_char_win_rate}
        - Lightcone 75/25 Wins for Failed Runs: {failure_lc_win_rate}
        - Avg Leftover Pulls for Failed Runs: {avg_leftover_pulls_on_failure}
        - Avg Refund for Failed Runs: {avg_refund_fail}


        As succinctly as possible, interpret the outcome. Answer questions like: Why did the success rate end up where it did?
        Does the number of character and lightcone wins compared to the desired number mean that the user HAD to win at least certain number of 50/50s?
        How does the character win rate compare to the baseline 50% odds, and similarly for lightcone win rate comparing to the 75% odds; what does this mean?
        Is early pity wins are necessary for the successful cases? How does it compare to the failure cases?
        Consider logically how players generally will obtain a character around soft pity of 80 or a lightcone around soft pity of 70, and multiply those numbers 
        by the number of desired characters and lightcones and sum them; does that number exceed or stay within the number of pulls they have? For example, if their number of pulls is 
        just barely within that number, then you would form a hypothesis that they really have to win character 50/50 and lightcone 75/25, and if they don't, then earlies are necessary, so you would look at the 
        data holistically to see if it supports this hypothesis, and if not revise. Present the final hypothesis to the user.
        For context, character hard pity is 90, and lightcone hard pity is 80 and if there is no guarantee, there is a 50/50 or a 75/25, where it is possible the player does not 
        win the character or lightcone and the pity resets and they have to pull more.
        For context, refunds are based on receiving a 4-star character and exchanging it for another pull in the HSR store. Since 4-stars are guaranteed at 
        least every 10 pulls, a higher number of refunds must be considered in tandem with how many pulls the player used. Lower number of pulls but higher refunds 
        implies the user got lucky with obtaining 4-stars and therefore got refunds. Higher number of fulls proportionately increases the number of refunds.
        Therefore, refunds MUST be considered alongside how many leftover pulls there were and the ratio of those two between successful and unsuccessful runs must
        be calculated and compared to produce meaningful analysis.

        Lastly, provide a summary of the outcomes all considered together and if success isn't as likely, a suggestion for how many pulls may provide a buffer.
        Present this information as succinctly as possible, in plain English without using statistical jargon.
        """.format(goal_label=goal_label, goal_description=goal_description, **sim_stats) # unpack sim_stats


    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ]
    )

    return response.choices[0].message.content.strip()

"""analyzer.py
Goal-description helper shared by the advisor.

The one-shot LLM verdict that used to live here has been retired: the
deterministic "read" (summary.py) covers the factual summary, and the follow-up
advisor (advisor.py) covers open-ended AI. Only describe_goal remains.
"""


def describe_goal(sim_stats):
    chars = sim_stats["desired_characters"]
    weapons = sim_stats["desired_weapons"]

    char_label = f"C{chars - 1}" if chars > 0 else ""
    goal_label = f"{char_label}W{weapons}"
    char_phrase = (
        "no character copies"
        if chars == 0 else
        "one copy of the limited 5★ character"
        if chars == 1 else
        f"{chars} copies of the limited 5★ character"
    )
    weapon_phrase = (
        "no signature weapon"
        if weapons == 0 else
        "one signature 5★ weapon"
        if weapons == 1 else
        f"{weapons} signature 5★ weapons"
    )

    return goal_label, f"{char_phrase} and {weapon_phrase}"

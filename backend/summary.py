"""summary.py
Deterministic, model-free "read" of a simulation result.

This replaces the factual part of the old LLM verdict: a confidence label, a
margin note, and the key lever, all derived from the stats by simple thresholds.
No network call and no cost, so it is always computed and always shown. The
optional LLM verdict (analyzer.py) stays available as an opt-in extra.
"""


def _success_pct(stats):
    """Success rate as a 0-100 number. Accepts the '46.75%' string the engine
    emits, and is defensive about numeric inputs used in tests."""
    raw = stats.get("success_rate", 0)
    if isinstance(raw, str):
        try:
            return float(raw.strip().rstrip("%"))
        except ValueError:
            return 0.0
    return raw * 100 if raw <= 1 else float(raw)


def _confidence(pct):
    if pct >= 85:
        return "comfortable", "You are very likely to hit this goal."
    if pct >= 60:
        return "likely", "You are more likely than not to hit this goal."
    if pct >= 40:
        return "coin_flip", "This is close to a coin flip."
    if pct >= 20:
        return "stretch", "This is a stretch."
    return "long_shot", "This is a long shot."


def _key_lever(stats):
    """The factor that most separates successful runs from failed ones, read
    from the correlation stats. Returns None if nothing is clearly decisive."""
    cs = stats.get("correlation_stats") or {}
    ts = cs.get("total_successes", 0)
    tf = cs.get("total_failures", 0)
    if not ts or not tf:
        return None

    def discriminating_power(succ_key, fail_key):
        s, f = cs.get(succ_key), cs.get(fail_key)
        if s is None or f is None:
            return None
        return (s / ts) - (f / tf)

    candidates = []
    if stats.get("desired_characters", 0) > 0:
        p = discriminating_power("all_5050s_won_in_successes", "all_5050s_won_in_failures")
        if p is not None:
            candidates.append((p, "Winning your character 50/50s is what separates success from failure here."))
    if stats.get("desired_weapons", 0) > 0:
        p = discriminating_power("all_weapon_75s_won_in_successes", "all_weapon_75s_won_in_failures")
        if p is not None:
            candidates.append((p, "Winning the weapon 75/25 is the deciding factor here."))
    p = discriminating_power("early_char_pity_in_successes", "early_char_pity_in_failures")
    if p is not None:
        candidates.append((p, "Hitting an early pity, before soft pity, is a common thread in the runs that succeed."))

    if not candidates:
        return None
    power, phrase = max(candidates, key=lambda c: c[0])
    return phrase if power >= 0.15 else None


def _margin_note(stats, pct):
    if pct < 5:
        return None
    leftover = int(round(stats.get("avg_leftover_pulls_on_success") or 0))
    if leftover <= 0:
        return "Successful runs use almost every pull, so there is little margin."
    return f"On the runs that succeed, you finish with about {leftover} pulls to spare."


def _framing(confidence, stats):
    guaranteed = stats.get("start_char_guarantee") or stats.get("start_weapon_guarantee")
    guarantee_note = " You start on a guarantee, which removes one coin flip in your favor." if guaranteed else ""
    base = {
        "comfortable": "You are comfortably there. Adding pulls would not change much, so there is no need to overspend.",
        "likely": "You are in good shape. A lost coin flip is usually survivable here.",
        "coin_flip": "This hinges on the coin flips. Losing one costs a full pity cycle, so a few extra pulls help far less than winning the flip or entering on a guarantee.",
        "stretch": "You will likely need the coin flips to go your way. Waiting for a guarantee or a rerun may serve you better than adding a handful of pulls.",
        "long_shot": "The odds are long. Consider narrowing the goal, waiting for a guarantee, or saving for the rerun.",
    }[confidence]
    return base + guarantee_note


def build_summary(stats):
    """Return a deterministic read: a confidence key, a headline, and a short
    list of plain-language notes."""
    pct = _success_pct(stats)
    confidence, headline = _confidence(pct)
    notes = [
        _margin_note(stats, pct),
        _key_lever(stats),
        _framing(confidence, stats),
    ]
    return {
        "confidence": confidence,
        "headline": headline,
        "notes": [note for note in notes if note],
    }

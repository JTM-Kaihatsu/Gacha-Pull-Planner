"""session_state.py
Deterministic reconciliation of a mid-session narrative (what the advisor's
extraction call pulls out of a free-text question) against the baseline goal.

No OpenAI calls happen here and no math happens in the model: the extraction
call reports only raw, explicitly-stated numbers (pulls spent per banner,
4-star refunds received per banner, any additional pulls or extra copies
mentioned), and this module does every subtraction and addition. advisor.py
is the only caller, and only re-invokes the extraction model when this
module reports an inconsistency.
"""

MAX_CHARACTER_COPIES = 7  # C0-C6, matches the frontend's strategy builder
MAX_WEAPON_COPIES = 5     # W1-W5


def _net_pulls_used(extracted):
    """Sum (pulls_spent - refunds) across banners that were actually mentioned.
    Returns None if neither banner reported a pulls_spent figure."""
    total = 0
    any_given = False
    for banner in ("character", "weapon"):
        spent = extracted.get(f"{banner}_pulls_spent")
        if spent is not None:
            any_given = True
            refunds = extracted.get(f"{banner}_refunds") or 0
            total += max(spent - refunds, 0)
    return total if any_given else None


def _resolve_total_pulls(extracted, baseline_total_pulls):
    """Figure out pulls remaining from whichever combination of fields the
    question stated. `additional_pulls_stated` is an amount to add on top of
    whatever remains (e.g. "plus about 45 more from this patch"), separate
    from a direct restatement of the total. Returns (total_pulls, error)."""
    stated = extracted.get("pulls_remaining_stated")
    restated_total = extracted.get("total_pulls_restated")
    additional = extracted.get("additional_pulls_stated") or 0
    net_used = _net_pulls_used(extracted)

    if additional < 0:
        return None, f"additional_pulls_stated ({additional}) cannot be negative"

    computed = None
    base = restated_total if restated_total is not None else baseline_total_pulls
    if net_used is not None:
        computed = base - net_used + additional

    if stated is not None and computed is not None and stated != computed:
        return None, (
            f"pulls_remaining_stated ({stated}) does not reconcile with net pulls used "
            f"({net_used}) plus additional_pulls_stated ({additional}) against "
            f"total_pulls ({base}) = {computed}"
        )

    if stated is not None:
        return stated, None
    if computed is not None:
        return computed, None
    return None, (
        "could not determine pulls remaining: no pulls_remaining_stated and no "
        "per-banner pulls_spent given"
    )


def _build_breakdown(extracted, chars_remaining, weapons_remaining, total_pulls,
                      goal_complete=False, pulls_exhausted=False):
    parts = []
    for banner, label in (("character", "character"), ("weapon", "weapon")):
        if extracted.get(f"{banner}_obtained"):
            spent = extracted.get(f"{banner}_pulls_spent")
            refunds = extracted.get(f"{banner}_refunds")
            detail = ""
            if spent is not None:
                detail = f" at {spent} pity"
                if refunds:
                    detail += f" ({refunds} refunded)"
            parts.append(f"{label} secured{detail}")

    if extracted.get("character_guarantee_active") and chars_remaining >= 1:
        parts.append("character 50/50 lost, guarantee active")
    if extracted.get("weapon_guarantee_active") and weapons_remaining >= 1:
        parts.append("weapon 50/50 lost, guarantee active")

    additional_chars = extracted.get("additional_character_copies_wanted") or 0
    additional_weapons = extracted.get("additional_weapon_copies_wanted") or 0
    if additional_chars:
        parts.append(f"goal expanded by {additional_chars} extra character cop"
                      f"{'y' if additional_chars == 1 else 'ies'}")
    if additional_weapons:
        parts.append(f"goal expanded by {additional_weapons} extra weapon"
                      f"{'' if additional_weapons == 1 else 's'}")

    additional_pulls = extracted.get("additional_pulls_stated") or 0
    if additional_pulls:
        parts.append(f"plus {additional_pulls} more pulls stated")

    situation = ", ".join(parts) if parts else "no prior progress stated"

    if goal_complete:
        return f"{situation}. Goal already complete, nothing left to pull for."

    needed = []
    if chars_remaining >= 1:
        needed.append(f"{chars_remaining} character cop{'y' if chars_remaining == 1 else 'ies'}")
    if weapons_remaining >= 1:
        needed.append(f"{weapons_remaining} weapon{'s' if weapons_remaining != 1 else ''}")
    needed_str = " and ".join(needed)

    if pulls_exhausted:
        return f"{situation}. Still need {needed_str}, but no pulls remain."
    return f"{situation}. Still need {needed_str}, with {total_pulls} pulls remaining."


def reconcile(extracted, baseline_params, baseline_stats):
    """Turn the extraction model's structured output into a verified,
    ready-to-simulate state, or a specific error to feed back for one
    corrective re-extraction.

    Returns a dict. `applies` is False when the question had no session
    narrative to reconcile (a pure hypothetical), the caller should fall
    through to the existing baseline-driven flow. When `applies` is True,
    `ok` says whether reconciliation succeeded; on failure `error` names the
    specific inconsistency for the retry prompt.
    """
    if not extracted.get("has_session_context"):
        return {"applies": False}

    additional_chars = extracted.get("additional_character_copies_wanted") or 0
    additional_weapons = extracted.get("additional_weapon_copies_wanted") or 0
    desired_characters = baseline_stats["desired_characters"] + additional_chars
    desired_weapons = baseline_stats["desired_weapons"] + additional_weapons

    if desired_characters > MAX_CHARACTER_COPIES:
        return {"applies": True, "ok": False,
                "error": f"requested character copies ({desired_characters}) exceed the max of {MAX_CHARACTER_COPIES}"}
    if desired_weapons > MAX_WEAPON_COPIES:
        return {"applies": True, "ok": False,
                "error": f"requested weapon copies ({desired_weapons}) exceed the max of {MAX_WEAPON_COPIES}"}

    char_obtained = bool(extracted.get("character_obtained"))
    weapon_obtained = bool(extracted.get("weapon_obtained"))

    chars_remaining = max(desired_characters - (1 if char_obtained else 0), 0)
    weapons_remaining = max(desired_weapons - (1 if weapon_obtained else 0), 0)

    char_hard_pity = baseline_params["char_pity_config"]["hard_pity"]
    weapon_hard_pity = baseline_params["weapon_pity_config"]["hard_pity"]
    for banner, hard_pity in (("character", char_hard_pity), ("weapon", weapon_hard_pity)):
        spent = extracted.get(f"{banner}_pulls_spent")
        refunds = extracted.get(f"{banner}_refunds")
        if spent is not None and not (0 <= spent <= hard_pity):
            return {"applies": True, "ok": False,
                    "error": f"{banner}_pulls_spent ({spent}) is out of range 0-{hard_pity}"}
        if refunds is not None and spent is not None and refunds > spent:
            return {"applies": True, "ok": False,
                    "error": f"{banner}_refunds ({refunds}) exceeds {banner}_pulls_spent ({spent})"}

    total_pulls, error = _resolve_total_pulls(extracted, baseline_params["total_pulls"])
    if error:
        return {"applies": True, "ok": False, "error": error}

    if chars_remaining == 0 and weapons_remaining == 0:
        return {
            "applies": True, "ok": True, "goal_complete": True, "pulls_exhausted": False,
            "strategy": [], "total_pulls": total_pulls,
            "start_char_pity": 0, "start_char_guarantee": False,
            "start_weapon_pity": 0, "start_weapon_guarantee": False,
            "breakdown": _build_breakdown(extracted, chars_remaining, weapons_remaining,
                                           total_pulls, goal_complete=True),
        }

    if total_pulls <= 0:
        return {
            "applies": True, "ok": True, "goal_complete": False, "pulls_exhausted": True,
            "strategy": [], "total_pulls": 0,
            "start_char_pity": 0, "start_char_guarantee": False,
            "start_weapon_pity": 0, "start_weapon_guarantee": False,
            "breakdown": _build_breakdown(extracted, chars_remaining, weapons_remaining,
                                           0, pulls_exhausted=True),
        }

    # Preserve the baseline's relative pull order for whichever banners remain.
    strategy = [
        {"banner": phase["banner"],
         "copies": chars_remaining if phase["banner"] == "char" else weapons_remaining}
        for phase in baseline_params["strategy"]
        if (phase["banner"] == "char" and chars_remaining >= 1)
        or (phase["banner"] == "weapon" and weapons_remaining >= 1)
    ]
    present_banners = {p["banner"] for p in strategy}
    if chars_remaining >= 1 and "char" not in present_banners:
        strategy.append({"banner": "char", "copies": chars_remaining})
    if weapons_remaining >= 1 and "weapon" not in present_banners:
        strategy.append({"banner": "weapon", "copies": weapons_remaining})

    return {
        "applies": True, "ok": True, "goal_complete": False, "pulls_exhausted": False,
        "strategy": strategy,
        "total_pulls": total_pulls,
        "start_char_pity": 0,
        "start_char_guarantee": bool(extracted.get("character_guarantee_active")) if chars_remaining >= 1 else False,
        "start_weapon_pity": 0,
        "start_weapon_guarantee": bool(extracted.get("weapon_guarantee_active")) if weapons_remaining >= 1 else False,
        "breakdown": _build_breakdown(extracted, chars_remaining, weapons_remaining, total_pulls),
    }

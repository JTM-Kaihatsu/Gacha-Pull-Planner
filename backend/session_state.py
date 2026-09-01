"""session_state.py
Deterministic reconciliation of a mid-session narrative (an ordered sequence
of pull events the advisor's extraction call pulls out of a free-text
question) against the baseline goal.

No OpenAI calls happen here and no math happens in the model: the extraction
call reports one atomic fact per event ("this banner, this outcome, this
pity, this refund count, in this position"), and this module is the only
place that applies the actual game rules: subtract pulls spent, add refunds
back, reset pity on any outcome, track guarantees and obtained counts. It
also builds the structured "pill" data the UI renders directly, so the
displayed breakdown is assembled from the same verified numbers the
simulation itself uses, not asserted separately by a model.

advisor.py is the only caller, and only re-invokes the extraction model when
this module reports an inconsistency (capped at one retry).
"""

MAX_CHARACTER_COPIES = 7  # C0-C6, matches the frontend's strategy builder
MAX_WEAPON_COPIES = 5     # W1-W5

TOOLTIPS = {
    "start_pulls": "Starting number of pulls for run",
    "pity": "Number of pulls spent in this run",
    "refund": "Number of refunded pulls in this run",
    "run_result": "Resulting number of pulls after operations in this run",
    "final_result": "The calculated amount of pulls after this sequence of actions",
    "obtained": "Current number of characters or weapons after this run",
    "subtract_op": "Pulls were spent",
    "add_op": "Adding pulls gained",
    "equals_op": "Equals: the result of the operations to the left",
    "guarantee": "Whether or not this run is guaranteed to get the character or weapon",
    "outcome": "Whether this run resulted in a win or a loss",
    "goal": "The character and weapon copies being planned for, based on the original goal and anything added or already obtained",
    "simulated": "The actual simulated success rate using the pulls and goal above",
    "status": "The current state of your goal after the events above",
    "stated_remaining": "Pulls remaining as directly stated in your question, since not every event gave an exact pity count",
}

_BANNER_LABEL = {"character": "Character", "weapon": "Weapon"}


def _pill(kind, value, color, tooltip_key):
    return {"kind": kind, "value": value, "color": color, "tooltip": TOOLTIPS[tooltip_key]}


def _validate_events(events, char_pity_config, weapon_pity_config):
    if not isinstance(events, list):
        return "events must be a list"
    expected_order = 1
    for event in events:
        if event.get("banner_type") not in ("character", "weapon"):
            return f"event_order {event.get('event_order')}: banner_type must be 'character' or 'weapon'"
        if event.get("outcome") not in ("win", "loss"):
            return f"event_order {event.get('event_order')}: outcome must be 'win' or 'loss'"
        if event.get("event_order") != expected_order:
            return (
                f"events must be numbered sequentially starting from 1 in the order "
                f"they occurred; expected event_order {expected_order}, got {event.get('event_order')}"
            )
        expected_order += 1

        hard_pity = (char_pity_config if event["banner_type"] == "character" else weapon_pity_config)["hard_pity"]
        pity = event.get("pity_at_outcome")
        refunds = event.get("refund_count")
        # pity_at_outcome may be null: the question described an outcome
        # without giving its exact pull count (e.g. "I won with 62 to
        # spare", "I lost, and I have 81 pulls left"). That event still
        # updates obtained counts and guarantees; it just can't contribute
        # to the pulls ledger, so pulls_remaining_stated becomes required.
        if pity is not None and (not isinstance(pity, int) or not (0 <= pity <= hard_pity)):
            return f"event_order {event['event_order']}: pity_at_outcome ({pity}) is out of range 0-{hard_pity}"
        if not isinstance(refunds, int) or refunds < 0:
            return f"event_order {event['event_order']}: refund_count ({refunds}) cannot be negative"
        if pity is not None and refunds > pity:
            return f"event_order {event['event_order']}: refund_count ({refunds}) exceeds pity_at_outcome ({pity})"
    return None


def _run_label(banner, run_number, obtained, desired):
    return f"{_BANNER_LABEL[banner]} Run {run_number} (obtained {obtained} of {desired})"


def _process_events(events, running_pulls, desired_characters, desired_weapons):
    """Walk the event sequence in order, applying the deterministic game
    rules, and build one pill-line per event. Returns (lines, char_obtained,
    weapon_obtained, char_guarantee, weapon_guarantee, char_events, weapon_events,
    running_pulls, any_unknown_pity, error)."""
    counts = {"character": 0, "weapon": 0}
    guarantees = {"character": False, "weapon": False}
    run_index = {"character": 0, "weapon": 0}
    desired = {"character": desired_characters, "weapon": desired_weapons}
    lines = []
    any_unknown_pity = False

    for event in events:
        banner = event["banner_type"]
        outcome = event["outcome"]
        pity = event["pity_at_outcome"]
        refunds = event["refund_count"]

        run_index[banner] += 1

        if outcome == "win":
            counts[banner] += 1
            guarantees[banner] = False
        else:
            guarantees[banner] = True

        label = _run_label(banner, run_index[banner], counts[banner], desired[banner])

        if pity is None:
            # No exact pity given for this event: it still updates the
            # obtained count and guarantee above, but can't touch the pulls
            # ledger. The caller must fall back to a direct restatement.
            any_unknown_pity = True
            lines.append({
                "label": label,
                "pills": [
                    _pill("outcome", outcome.upper(), "green" if outcome == "win" else "red", "outcome"),
                ],
            })
            continue

        start_pulls = running_pulls
        running_pulls -= pity
        running_pulls += refunds

        if running_pulls < 0:
            return (lines, counts["character"], counts["weapon"], guarantees["character"],
                    guarantees["weapon"], run_index["character"], run_index["weapon"], running_pulls,
                    any_unknown_pity,
                    f"event_order {event['event_order']}: pulls consumed exceed the stated total budget")

        lines.append({
            "label": label,
            "pills": [
                _pill("number", start_pulls, "cyan", "start_pulls"),
                _pill("operator", "−", "magenta", "subtract_op"),
                _pill("number", pity, "cyan", "pity"),
                _pill("operator", "+", "magenta", "add_op"),
                _pill("number", refunds, "cyan", "refund"),
                _pill("operator", "=", "magenta", "equals_op"),
                _pill("result", running_pulls, "light_green", "run_result"),
                _pill("outcome", outcome.upper(), "green" if outcome == "win" else "red", "outcome"),
            ],
        })

    return (lines, counts["character"], counts["weapon"], guarantees["character"], guarantees["weapon"],
            run_index["character"], run_index["weapon"], running_pulls, any_unknown_pity, None)


def reconcile(extracted, baseline_params, baseline_stats):
    """Turn the extraction model's structured event sequence into a
    verified, ready-to-simulate state plus the structured pill data the UI
    renders, or a specific error to feed back for one corrective
    re-extraction.

    Returns a dict. `applies` is False when the question described no event
    sequence at all (a pure hypothetical or a pure strategy question), the
    caller should fall through to the existing baseline-driven flow. When
    `applies` is True, `ok` says whether reconciliation succeeded; on
    failure `error` names the specific inconsistency for the retry prompt.
    """
    if not extracted.get("has_event_sequence"):
        return {"applies": False}

    events = extracted.get("events") or []
    if not events:
        return {"applies": True, "ok": False,
                "error": "has_event_sequence was true but no events were reported"}

    error = _validate_events(events, baseline_params["char_pity_config"], baseline_params["weapon_pity_config"])
    if error:
        return {"applies": True, "ok": False, "error": error}

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

    additional_pulls = extracted.get("additional_pulls_stated") or 0
    if additional_pulls < 0:
        return {"applies": True, "ok": False, "error": f"additional_pulls_stated ({additional_pulls}) cannot be negative"}

    total_pulls_budget = extracted.get("total_pulls_restated") or baseline_params["total_pulls"]

    (event_lines, char_obtained, weapon_obtained, char_guarantee, weapon_guarantee,
     char_events, weapon_events, running_pulls, any_unknown_pity,
     error) = _process_events(events, total_pulls_budget, desired_characters, desired_weapons)
    if error:
        return {"applies": True, "ok": False, "error": error}

    lines = list(event_lines)
    stated = extracted.get("pulls_remaining_stated")

    if any_unknown_pity:
        # At least one event didn't state its exact pity, so the ledger
        # above is known-incomplete: it can't be trusted as "total consumed"
        # and there's nothing valid to cross-check. A direct restatement of
        # pulls remaining is the only way to know the true figure here.
        if stated is None:
            return {"applies": True, "ok": False, "error": (
                "at least one event did not state its exact pity, so pulls_remaining_stated "
                "must be given directly"
            )}
        pre_addition_remaining = stated
        lines.append({
            "label": "Stated Pulls Remaining",
            "pills": [_pill("result", stated, "light_green", "stated_remaining")],
        })
    else:
        # Deterministic conservation check: total consumed across every
        # event, plus whatever remains, must equal the stated starting
        # budget. This is exactly what the event walk above already
        # enforces arithmetically; cross-check it against a direct
        # restatement if the question also gave one.
        pre_addition_remaining = running_pulls
        if stated is not None and stated != pre_addition_remaining + additional_pulls:
            return {"applies": True, "ok": False, "error": (
                f"pulls_remaining_stated ({stated}) does not reconcile with the events: "
                f"{total_pulls_budget} starting pulls, {total_pulls_budget - pre_addition_remaining} net "
                f"consumed across {len(events)} event(s), leaving {pre_addition_remaining}, "
                f"plus additional_pulls_stated ({additional_pulls}) = {pre_addition_remaining + additional_pulls}"
            )}

    # pre_addition_remaining is authoritative from here on (it's either the
    # computed ledger total, or the direct restatement when any event's
    # pity was unknown); running_pulls must track it exactly even when
    # there's no additional-pulls line to reassign it below.
    running_pulls = pre_addition_remaining

    if additional_pulls:
        post_addition = pre_addition_remaining + additional_pulls
        lines.append({
            "label": "Additional Pulls Mentioned",
            "pills": [
                _pill("number", pre_addition_remaining, "cyan", "start_pulls"),
                _pill("operator", "+", "magenta", "add_op"),
                _pill("number", additional_pulls, "cyan", "refund"),
                _pill("operator", "=", "magenta", "equals_op"),
                _pill("result", post_addition, "light_green", "run_result"),
            ],
        })
        running_pulls = post_addition

    total_pulls = running_pulls
    # Mark the true final "pulls remaining" pill with the distinct tooltip
    # for the figure that actually feeds the simulation. This is the last
    # pill of *kind* "result" across all lines so far, not simply the last
    # pill of the last line: when there's no "Additional Pulls" line, the
    # last event line's last pill is its WIN/LOSS outcome, not its result.
    for line in reversed(lines):
        for i in range(len(line["pills"]) - 1, -1, -1):
            if line["pills"][i]["kind"] == "result":
                line["pills"][i] = dict(line["pills"][i], tooltip=TOOLTIPS["final_result"])
                break
        else:
            continue
        break

    chars_remaining = max(desired_characters - char_obtained, 0)
    weapons_remaining = max(desired_weapons - weapon_obtained, 0)

    goal_pill_text = _goal_text(baseline_stats, additional_chars, additional_weapons)
    goal_line = {"label": "Restated Goal", "pills": [_pill("goal", goal_pill_text, "default", "goal")]}

    if chars_remaining == 0 and weapons_remaining == 0:
        lines.append(goal_line)
        lines.append({"label": "Goal Status",
                      "pills": [_pill("status", "GOAL COMPLETE", "green", "status")]})
        return {
            "applies": True, "ok": True, "goal_complete": True, "pulls_exhausted": False,
            "strategy": [], "total_pulls": total_pulls,
            "start_char_pity": 0, "start_char_guarantee": False,
            "start_weapon_pity": 0, "start_weapon_guarantee": False,
            "lines": lines,
        }

    if total_pulls <= 0:
        lines.append(goal_line)
        lines.append({"label": "Goal Status",
                      "pills": [_pill("status", "NO PULLS REMAIN", "red", "status")]})
        return {
            "applies": True, "ok": True, "goal_complete": False, "pulls_exhausted": True,
            "strategy": [], "total_pulls": 0,
            "start_char_pity": 0, "start_char_guarantee": False,
            "start_weapon_pity": 0, "start_weapon_guarantee": False,
            "lines": lines,
        }

    # A banner nobody mentioned keeps whatever the original form stated for
    # it (nothing here contradicts that); a banner that appeared in the
    # sequence gets its pity forced to 0 (an outcome, win or loss, always
    # means a 5-star was just obtained) and its guarantee from the last
    # event on it.
    start_char_pity = 0 if char_events else baseline_params["start_char_pity"]
    start_char_guarantee = char_guarantee if char_events else baseline_params["start_char_guarantee"]
    start_weapon_pity = 0 if weapon_events else baseline_params["start_weapon_pity"]
    start_weapon_guarantee = weapon_guarantee if weapon_events else baseline_params["start_weapon_guarantee"]

    if chars_remaining >= 1:
        lines.append({
            "label": _run_label("character", char_events + 1, char_obtained, desired_characters),
            "pills": [
                _pill("number", total_pulls, "cyan", "start_pulls"),
                _pill("flag", "GUARANTEE: TRUE" if start_char_guarantee else "GUARANTEE: FALSE",
                      "green" if start_char_guarantee else "red", "guarantee"),
            ],
        })
    if weapons_remaining >= 1:
        lines.append({
            "label": _run_label("weapon", weapon_events + 1, weapon_obtained, desired_weapons),
            "pills": [
                _pill("number", total_pulls, "cyan", "start_pulls"),
                _pill("flag", "GUARANTEE: TRUE" if start_weapon_guarantee else "GUARANTEE: FALSE",
                      "green" if start_weapon_guarantee else "red", "guarantee"),
            ],
        })

    lines.append(goal_line)

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
        "start_char_pity": start_char_pity,
        "start_char_guarantee": start_char_guarantee,
        "start_weapon_pity": start_weapon_pity,
        "start_weapon_guarantee": start_weapon_guarantee,
        "lines": lines,
    }


def _goal_text(baseline_stats, additional_chars, additional_weapons):
    orig_chars = baseline_stats["desired_characters"]
    orig_weapons = baseline_stats["desired_weapons"]
    new_chars = orig_chars + additional_chars
    new_weapons = orig_weapons + additional_weapons

    def _phrase(chars, weapons):
        parts = []
        if chars:
            parts.append(f"{chars} character{'s' if chars != 1 else ''}")
        if weapons:
            parts.append(f"{weapons} weapon{'s' if weapons != 1 else ''}")
        return " and ".join(parts) if parts else "nothing"

    if additional_chars or additional_weapons:
        return f"{_phrase(orig_chars, orig_weapons)} → {_phrase(new_chars, new_weapons)}"
    return _phrase(new_chars, new_weapons)


def build_result_line(total_pulls, success_rate):
    """The deterministic pre-run's result, appended as its own pill line
    once advisor.py has actually simulated the reconciled scenario."""
    return {
        "label": "Simulated Result",
        "pills": [_pill("result", f"{total_pulls} pulls → {success_rate}", "light_green", "simulated")],
    }

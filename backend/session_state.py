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

# Refund estimation: applied only when the question doesn't state a refund
# count for an event AND the baseline was simulated with "all 4-star
# characters at max copies" checked, since that's the only condition under
# which 4-star pulls reliably convert to refunds at all. Rates are the
# empirical average refunds per pull spent on that banner under max-copy
# refunding; the dupe bonus accounts for a 5-star win on a banner where the
# featured item's first copy is already owned, since a repeat win itself
# refunds a fixed amount on top of the per-pull rate.
_REFUND_RATE = {"character": 0.1105, "weapon": 0.0578}
DUPE_WIN_REFUND_BONUS = 2


def _estimate_refunds(banner, outcome, pity, prior_copies):
    """Returns (base_estimate, dupe_bonus): base_estimate is the per-pull
    rate estimate (rounded), dupe_bonus is DUPE_WIN_REFUND_BONUS when this
    win obtains a repeat copy (at least one copy of this banner's featured
    item already owned before this event: C1+ for a character, W2+ for a
    weapon), else 0. Kept separate, rather than folded into one number, so
    the caller can render the dupe bonus as its own distinct pill."""
    base_estimate = round(_REFUND_RATE[banner] * pity)
    dupe_bonus = DUPE_WIN_REFUND_BONUS if outcome == "win" and prior_copies >= 1 else 0
    return base_estimate, dupe_bonus

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
    "goal": "The character and weapon copies still being pursued from this point onward, based on the original goal, anything added or removed, and anything already obtained above",
    "simulated": "The actual simulated success rate using the pulls and goal above",
    "agent_cycle": "An additional scenario the AI chose to explore with a real simulation, beyond the primary answer above",
    "status": "The current state of your goal after the events above",
    "stated_remaining": "Pulls remaining as directly stated in your question, since not every event gave an exact pity count",
    "banner_marker": "Which banner the guarantee flag to the right belongs to",
    "existing_pity": "Pity already built up on this banner before this run",
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
        if pity is not None:
            if not isinstance(pity, int) or pity < 0:
                return f"event_order {event['event_order']}: pity_at_outcome ({pity}) cannot be negative"
            # hard_pity bounds pity_at_outcome regardless of phrasing: an
            # absolute pity COUNTER obviously can't exceed it, but neither
            # can a pulls-SPENT count taken by itself, no single attempt can
            # ever take more pulls than hard pity allows, independent of
            # whatever pity carried over from before this conversation. This
            # is a flat, unrecoverable error either way, not something a
            # clarifying question could resolve: if the number itself is
            # already too high to be a valid pity value under ANY reading,
            # asking "did you mean it as a total including existing pity"
            # is pointless, that reading would be just as invalid. Whether a
            # pulls-spent count fits ON TOP OF that banner's actual starting
            # pity (0 unless it's the banner's first event with carryover)
            # is a separate, narrower question only _normalize_pity_carryover
            # can answer, and IS the genuine ambiguity worth clarifying.
            if pity > hard_pity:
                return f"event_order {event['event_order']}: pity_at_outcome ({pity}) is out of range 0-{hard_pity}"
        # refund_count may be null: the question simply didn't mention
        # refunds for this event. That's distinct from an explicit "no
        # refunds" (0), and is later filled in by _estimate_refunds when
        # applicable rather than assumed to be 0.
        if refunds is not None and (not isinstance(refunds, int) or refunds < 0):
            return f"event_order {event['event_order']}: refund_count ({refunds}) cannot be negative"
        if pity is not None and refunds is not None and refunds > pity:
            return f"event_order {event['event_order']}: refund_count ({refunds}) exceeds pity_at_outcome ({pity})"
    return None


def _run_label(banner, run_number, obtained, desired):
    return f"{_BANNER_LABEL[banner]} Run {run_number} (obtained {obtained} of {desired})"


def _normalize_pity_carryover(events, baseline_params):
    """Reconcile each event against whatever pity/guarantee its banner
    actually had at that point: the "Starting Situation" baseline for a
    banner's FIRST event, 0 for every event after (pity always resets on
    any outcome, win or loss). This is a no-op whenever a banner's starting
    pity is 0 and stays 0, which covers the overwhelming majority of
    questions and leaves their behavior completely unchanged.

    pity_at_outcome is ambiguous on its own: 'won at 30 pity' states the
    absolute pity COUNTER value, so the pulls actually spent THIS run is
    30 minus whatever pity already existed; 'lost after 50 pulls' already
    states a pull COUNT for this run directly. The extraction's
    pity_is_absolute flag disambiguates the two; this function is the only
    place that distinction is applied, downstream of it every event's
    pity_at_outcome is uniformly "pulls spent this run".

    Returns a dict:
      {"ok": True, "events": [...]} on success. Each banner's first event,
        if it started with pity, gets pity_at_outcome replaced with the
        actual pulls spent and a pity_carryover dict driving the
        "(ending pity - starting pity)" pill, but ONLY when that breakdown
        reveals something the question didn't already state outright (the
        absolute-pity case always does; a pulls-spent event that already
        fits within hard pity doesn't, so it stays a flat number).
      {"ok": False, "error": "..."} for a hard, unrecoverable contradiction
        (an already-guaranteed banner losing, or an absolute pity not
        exceeding what was already on the banner).
      {"ok": False, "conflicts": [...]} when one or more pulls-spent events
        claim more pulls than that banner's hard pity allows from its
        actual starting point: a genuine ambiguity ('did the 50 pulls you
        mentioned already include the pity you started with?') only the
        user can resolve, not a plain mistake to just reject.
    """
    starting_pity = {"character": baseline_params["start_char_pity"],
                      "weapon": baseline_params["start_weapon_pity"]}
    starting_guarantee = {"character": baseline_params["start_char_guarantee"],
                           "weapon": baseline_params["start_weapon_guarantee"]}
    hard_pity = {"character": baseline_params["char_pity_config"]["hard_pity"],
                 "weapon": baseline_params["weapon_pity_config"]["hard_pity"]}

    total_per_banner = {"character": 0, "weapon": 0}
    for event in events:
        total_per_banner[event["banner_type"]] += 1

    seen = {"character": False, "weapon": False}
    attempt_number = {"character": 0, "weapon": 0}
    normalized = []
    conflicts = []

    for event in events:
        banner = event["banner_type"]
        is_first = not seen[banner]
        seen[banner] = True
        attempt_number[banner] += 1

        if is_first and starting_guarantee[banner] and event["outcome"] == "loss":
            return {"ok": False, "error": (
                f"event_order {event['event_order']}: the {banner} banner already had a guaranteed "
                f"next 5-star from the starting state, so a loss on this event isn't possible"
            )}

        new_event = dict(event, pity_carryover=None)
        reported = event["pity_at_outcome"]
        # Only a banner's FIRST event can carry pity over from the starting
        # situation; every later event on it starts from a pity reset to 0.
        start = starting_pity[banner] if is_first else 0

        if reported is not None:
            is_absolute = event.get("pity_is_absolute", True)
            if is_absolute:
                # "at 30 pity": the reported number is the counter value the
                # outcome landed on; pulls spent this run is that minus the
                # pity that was already on the banner. Only meaningful when
                # there WAS carryover pity: with start == 0 the reported
                # value already equals pulls spent (and _validate_events
                # already confirmed it's within hard pity), so there's
                # nothing to subtract and no parenthetical to show.
                if start > 0:
                    ending_pity = reported
                    pulls_this_run = reported - start
                    if pulls_this_run < 1:
                        return {"ok": False, "error": (
                            f"event_order {event['event_order']}: pity_at_outcome ({reported}) must "
                            f"exceed the {start} pity already on the {banner} banner before this "
                            f"conversation"
                        )}
                    new_event["pity_at_outcome"] = pulls_this_run
                    new_event["pity_carryover"] = {
                        "ending_pity": ending_pity,
                        "starting_pity": start,
                        "ending_tooltip": "The pity the outcome was reached at, as stated in your question",
                        "minus_tooltip": (
                            f"Number of pulls spent in this run; pity reached {ending_pity}, "
                            f"already starting from {start} on this banner"
                        ),
                    }
            else:
                # "after 50 pulls": the reported number is already the pulls
                # spent this run; whatever pity the banner had at the start
                # of THIS event (its own starting pity if it's the banner's
                # first event, 0 otherwise) stacks on top to determine
                # whether the counter could plausibly have reached that
                # point without an earlier guaranteed hit. Checked
                # regardless of whether start is 0: reporting more pulls
                # than the hard pity allows is impossible either way.
                ending_pity = start + reported
                if ending_pity > hard_pity[banner]:
                    conflicts.append({
                        "event_order": event["event_order"],
                        "banner": banner,
                        "attempt_number": attempt_number[banner],
                        "total_attempts": total_per_banner[banner],
                        "reported_pulls": reported,
                        "starting_pity": start,
                        "hard_pity": hard_pity[banner],
                        "outcome": event["outcome"],
                    })
                # Otherwise it checks out: net-new pulls is just `reported`,
                # exactly as already set above, no parenthetical, showing
                # "(80 - 5)" here would only reconstruct the 50 the question
                # already gave directly, revealing nothing new.

        normalized.append(new_event)

    if conflicts:
        return {"ok": False, "conflicts": conflicts}

    return {"ok": True, "events": normalized}


def _process_events(events, running_pulls, desired_characters, desired_weapons, full_4star_chars):
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
        prior_copies = counts[banner]  # before this event's own win, if any

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

        estimated_refund = refunds is None
        dupe_bonus = 0
        if estimated_refund:
            # The question never mentioned refunds for this event. Only
            # estimate when max-copy refunding was actually in effect for
            # the baseline; otherwise there's no reliable basis for a
            # nonzero figure, so it stays 0 exactly as before.
            if full_4star_chars:
                base_refund, dupe_bonus = _estimate_refunds(banner, outcome, pity, prior_copies)
            else:
                base_refund = 0
            refunds = base_refund + dupe_bonus

        start_pulls = running_pulls
        running_pulls -= pity
        running_pulls += refunds

        if running_pulls < 0:
            return (lines, counts["character"], counts["weapon"], guarantees["character"],
                    guarantees["weapon"], run_index["character"], run_index["weapon"], running_pulls,
                    any_unknown_pity,
                    f"event_order {event['event_order']}: pulls consumed exceed the stated total budget")

        refund_pill = (
            {
                "kind": "number", "value": base_refund, "color": "light_green",
                "tooltip": (
                    "A number of refunds wasn't given and all 4-stars are obtained, so a "
                    f"formula estimating the average number of refunds for {pity} pulls has been used"
                ),
            }
            if estimated_refund and full_4star_chars else
            _pill("number", refunds, "cyan", "refund")
        )
        # A repeat copy (C1+ for a character, W2+ for a weapon) refunds a
        # fixed amount on top of the per-pull rate; shown as its own "+ 2"
        # pair rather than silently folded into the estimate above, so it
        # reads as a distinct, explainable figure instead of an
        # unexplained jump in the base rate's own number.
        dupe_pills = (
            [
                _pill("operator", "+", "magenta", "add_op"),
                {
                    "kind": "number", "value": dupe_bonus, "color": "light_green",
                    "tooltip": f"A number of refunds wasn't given, and this is an additional {banner} copy",
                },
            ]
            if dupe_bonus > 0 else []
        )

        carryover = event.get("pity_carryover")
        if carryover:
            # Break the pulls-spent figure into "(ending pity - starting
            # pity)", so the pity that was already on the banner is visible
            # rather than silently baked into one number. The group's
            # arithmetic result IS `pity` (the net pulls this run), which is
            # what the outer equation subtracts and the refund estimate uses.
            pity_pill = {
                "kind": "group",
                "pills": [
                    {"kind": "number", "value": carryover["ending_pity"], "color": "cyan",
                     "tooltip": carryover["ending_tooltip"]},
                    {"kind": "operator", "value": "−", "color": "magenta",
                     "tooltip": carryover["minus_tooltip"]},
                    {"kind": "number", "value": carryover["starting_pity"], "color": "cyan",
                     "tooltip": TOOLTIPS["existing_pity"]},
                ],
            }
        else:
            pity_pill = _pill("number", pity, "cyan", "pity")

        lines.append({
            "label": label,
            "pills": [
                _pill("number", start_pulls, "cyan", "start_pulls"),
                _pill("operator", "−", "magenta", "subtract_op"),
                pity_pill,
                _pill("operator", "+", "magenta", "add_op"),
                refund_pill,
                *dupe_pills,
                _pill("operator", "=", "magenta", "equals_op"),
                _pill("result", running_pulls, "light_green", "run_result"),
                _pill("outcome", outcome.upper(), "green" if outcome == "win" else "red", "outcome"),
            ],
        })

    return (lines, counts["character"], counts["weapon"], guarantees["character"], guarantees["weapon"],
            run_index["character"], run_index["weapon"], running_pulls, any_unknown_pity, None)


def _build_conflict_block(conflict):
    """One "did you mean total including existing pity" block: a header
    identifying which attempt on which banner (relevant once there's more
    than one event on it), the clarifying question, and the reported
    number and outcome wrapped in a red-bordered group, the same visual
    device as the pity-breakdown parenthetical elsewhere, just red instead
    of neutral grey to flag it as unresolved rather than as extra context."""
    banner_label = _BANNER_LABEL[conflict["banner"]]
    reported = conflict["reported_pulls"]
    return {
        "event_order": conflict["event_order"],
        "banner": conflict["banner"],
        "attempt_number": conflict["attempt_number"],
        "total_attempts": conflict["total_attempts"],
        # Plain data alongside the rendering fields below, for advisor.py to
        # match a clarification answer back to its conflict and build retry
        # feedback text, without digging the number back out of the pills.
        "reported_pulls": reported,
        "header": f"For {banner_label} Attempt {conflict['attempt_number']} of {conflict['total_attempts']}",
        "question": (
            f"You reported {reported} pulls spent on the {banner_label} banner, but given the "
            f"initial pity for the {banner_label} banner, this isn't possible. Did you mean in "
            f"total, including existing pity, you had spent {reported} pulls, or did you mean "
            f"something else?"
        ),
        "pills": [{
            "kind": "group",
            "color": "red",
            "pills": [
                {
                    "kind": "number", "value": reported, "color": "red",
                    "tooltip": (
                        f"Reported as pulls spent this run, but {conflict['starting_pity']} pity "
                        f"already on the {banner_label.lower()} banner would put the total past its "
                        f"hard pity of {conflict['hard_pity']}"
                    ),
                },
                _pill("outcome", conflict["outcome"].upper(),
                      "green" if conflict["outcome"] == "win" else "red", "outcome"),
            ],
        }],
    }


def _build_conflict_result(events, conflicts, baseline_params, total_pulls_budget,
                            desired_characters, desired_weapons):
    """Build the response for one or more pulls-spent conflicts: pills for
    whatever preceded the FIRST conflict (fully resolved with the same
    machinery a normal reconciliation uses, since nothing about them is
    ambiguous), plus one flagged block per conflict found anywhere in the
    sequence, so every violation gets its own clarifying question up front
    instead of being discovered one at a time across repeated retries."""
    first_conflict_order = min(c["event_order"] for c in conflicts)
    resolved_events = [e for e in events if e["event_order"] < first_conflict_order]

    lines = []
    if resolved_events:
        carryover = _normalize_pity_carryover(resolved_events, baseline_params)
        if carryover["ok"]:
            event_lines, *_, process_error = _process_events(
                carryover["events"], total_pulls_budget, desired_characters, desired_weapons,
                baseline_params["full_4star_chars"],
            )
            if not process_error:
                lines = event_lines

    return {
        "applies": True, "ok": False, "conflict": True,
        "lines": lines,
        "conflicts": [_build_conflict_block(c) for c in conflicts],
    }


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

    # additional_*_copies_wanted is a signed delta on the original goal, not
    # a total: positive adds copies beyond it, negative reduces it (e.g. the
    # user gives up on further copies of a banner after a bad outcome).
    # Floored at 0, not at whatever's already obtained in this scenario,
    # events processed further down already floor "remaining" at 0 on their
    # own, so a reduction that dips below the obtained count still resolves
    # correctly as "goal complete for that banner". Computed before
    # normalizing pity so a conflict below can still build resolved pills
    # for whatever preceded it, using the real goal.
    additional_chars = extracted.get("additional_character_copies_wanted") or 0
    additional_weapons = extracted.get("additional_weapon_copies_wanted") or 0
    desired_characters = max(baseline_stats["desired_characters"] + additional_chars, 0)
    desired_weapons = max(baseline_stats["desired_weapons"] + additional_weapons, 0)

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

    carryover = _normalize_pity_carryover(events, baseline_params)
    if not carryover["ok"]:
        if "conflicts" in carryover:
            return _build_conflict_result(
                events, carryover["conflicts"], baseline_params,
                total_pulls_budget, desired_characters, desired_weapons,
            )
        return {"applies": True, "ok": False, "error": carryover["error"]}
    events = carryover["events"]

    (event_lines, char_obtained, weapon_obtained, char_guarantee, weapon_guarantee,
     char_events, weapon_events, running_pulls, any_unknown_pity,
     error) = _process_events(events, total_pulls_budget, desired_characters, desired_weapons,
                               baseline_params["full_4star_chars"])
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

    # What's still actually being pursued from here, not the original or
    # adjusted TOTAL goal: once a copy is obtained above, it must drop out
    # of this pill, the same rule already applied to the advisor's prose
    # via remaining_goal_text elsewhere.
    goal_pill_text = remaining_goal_text(chars_remaining, weapons_remaining)
    goal_line = {"label": "Current Goal", "pills": [_pill("goal", goal_pill_text, "default", "goal")]}

    if chars_remaining == 0 and weapons_remaining == 0:
        lines.append(goal_line)
        lines.append({"label": "Goal Status",
                      "pills": [_pill("status", "GOAL COMPLETE", "green", "status")]})
        return {
            "applies": True, "ok": True, "goal_complete": True, "pulls_exhausted": False,
            "strategy": [], "total_pulls": total_pulls,
            "start_char_pity": 0, "start_char_guarantee": False,
            "start_weapon_pity": 0, "start_weapon_guarantee": False,
            "remaining_characters": 0, "remaining_weapons": 0,
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
            "remaining_characters": chars_remaining, "remaining_weapons": weapons_remaining,
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

    if chars_remaining >= 1 and weapons_remaining >= 1:
        # Both banners still need a run, but they draw from the SAME shared
        # pull pool in sequence (see the strategy build below), not two
        # independent pools. Showing separate "Run" lines that each claim
        # the full total_pulls would wrongly imply twice the actual budget;
        # one combined line makes the sharing explicit instead.
        lines.append({
            "label": (
                f"Remaining Character (obtained {char_obtained} of {desired_characters}) "
                f"and Weapon (obtained {weapon_obtained} of {desired_weapons}) Runs"
            ),
            "pills": [
                _pill("number", total_pulls, "cyan", "start_pulls"),
                _pill("banner", "CHARACTER", "default", "banner_marker"),
                _pill("flag", "GUARANTEE: TRUE" if start_char_guarantee else "GUARANTEE: FALSE",
                      "green" if start_char_guarantee else "red", "guarantee"),
                _pill("banner", "WEAPON", "default", "banner_marker"),
                _pill("flag", "GUARANTEE: TRUE" if start_weapon_guarantee else "GUARANTEE: FALSE",
                      "green" if start_weapon_guarantee else "red", "guarantee"),
            ],
        })
    elif chars_remaining >= 1:
        lines.append({
            "label": _run_label("character", char_events + 1, char_obtained, desired_characters),
            "pills": [
                _pill("number", total_pulls, "cyan", "start_pulls"),
                _pill("flag", "GUARANTEE: TRUE" if start_char_guarantee else "GUARANTEE: FALSE",
                      "green" if start_char_guarantee else "red", "guarantee"),
            ],
        })
    elif weapons_remaining >= 1:
        lines.append({
            "label": _run_label("weapon", weapon_events + 1, weapon_obtained, desired_weapons),
            "pills": [
                _pill("number", total_pulls, "cyan", "start_pulls"),
                _pill("flag", "GUARANTEE: TRUE" if start_weapon_guarantee else "GUARANTEE: FALSE",
                      "green" if start_weapon_guarantee else "red", "guarantee"),
            ],
        })

    lines.append(goal_line)

    # Preserve the baseline's relative pull order for whichever banners
    # remain. The baseline strategy can list the SAME banner in more than
    # one phase (e.g. "pull 1 character, then the weapon, then the
    # remaining 2 characters"), so a banner's full remaining count must
    # land on only the FIRST phase it still appears in; every later phase
    # for that same banner is dropped rather than getting the full
    # remaining count all over again, which would silently double (or
    # more) the actual number of copies simulated beyond what's shown as
    # the remaining goal.
    strategy = []
    seen_banners = set()
    for phase in baseline_params["strategy"]:
        banner = phase["banner"]
        if banner in seen_banners:
            continue
        remaining = chars_remaining if banner == "char" else weapons_remaining
        if remaining < 1:
            continue
        strategy.append({"banner": banner, "copies": remaining})
        seen_banners.add(banner)
    if chars_remaining >= 1 and "char" not in seen_banners:
        strategy.append({"banner": "char", "copies": chars_remaining})
    if weapons_remaining >= 1 and "weapon" not in seen_banners:
        strategy.append({"banner": "weapon", "copies": weapons_remaining})

    return {
        "applies": True, "ok": True, "goal_complete": False, "pulls_exhausted": False,
        "strategy": strategy,
        "total_pulls": total_pulls,
        "start_char_pity": start_char_pity,
        "start_char_guarantee": start_char_guarantee,
        "start_weapon_pity": start_weapon_pity,
        "start_weapon_guarantee": start_weapon_guarantee,
        "remaining_characters": chars_remaining, "remaining_weapons": weapons_remaining,
        "lines": lines,
    }


def remaining_goal_text(remaining_characters, remaining_weapons):
    """Plain-English phrase for what's actually still needed, built from
    reconcile()'s remaining_characters/remaining_weapons. This is what the
    advisor's interpretation model should be told the goal is, not the
    original full goal: once a copy is obtained above, it must stop being
    described as something that could still fail or still needs pulling."""
    parts = []
    if remaining_characters:
        parts.append(f"{remaining_characters} character{'s' if remaining_characters != 1 else ''}")
    if remaining_weapons:
        parts.append(f"{remaining_weapons} weapon{'s' if remaining_weapons != 1 else ''}")
    return " and ".join(parts) if parts else "nothing, everything above is already obtained"


def build_result_line(total_pulls, success_rate):
    """The deterministic pre-run's result, appended as its own pill line
    once advisor.py has actually simulated the reconciled scenario."""
    return {
        "label": "Simulated Result",
        "pills": [_pill("result", f"{total_pulls} pulls → {success_rate}", "light_green", "simulated")],
    }


def build_spending_tier_line(label, total_pulls, success_rate):
    """One of advisor.py's pre-computed pull-count results for the
    reconciled scenario (the current budget, or a further modest-top-up
    or guaranteed option beyond it), rendered as its own labeled pill
    line, numbered "AI Agent Simulated Result N" starting from the
    current budget itself, so each figure is a real receipt the model
    narrates, not a claim it invents in prose, and none of them read as
    a plain given the others were merely compared against."""
    return {
        "label": label,
        "pills": [_pill("result", f"{total_pulls} pulls → {success_rate}", "light_green", "simulated")],
    }


def build_agent_cycle_line(cycle_number, total_pulls, success_rate):
    """A further scenario the advisor's own agentic loop chose to explore
    beyond the guaranteed pre-run, appended once advisor.py has actually run
    it. Kept visually distinct (its own labeled line) from the pre-run's
    "Simulated Result" so any such exploration is explicit and attributable,
    not just a claim folded into the prose answer."""
    return {
        "label": f"Agent Run Cycle {cycle_number}",
        "pills": [_pill("result", f"{total_pulls} pulls → {success_rate}", "light_green", "agent_cycle")],
    }

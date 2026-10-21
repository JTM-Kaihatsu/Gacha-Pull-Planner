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

from simulation import DUPE_WIN_REFUND_BONUS

MAX_CHARACTER_COPIES = 7  # C0-C6, matches the frontend's strategy builder
MAX_WEAPON_COPIES = 5     # W1-W5

# A conditional plan ("pull the weapon first; if I win, go for the
# character; if I lose, stop") is at most this many ordered steps, a
# runaway guard against a misread question, the same idea as advisor.py's
# MAX_SCENARIOS.
MAX_PLAN_STEPS = 4
# A route reached by fewer simulated runs than this still shows its
# chance (which is measured across every run), but its average leftover
# figure rests on only that small subset, so it's flagged approximate.
MIN_ROUTE_SAMPLE = 200

# Refund estimation: applied only when the question doesn't state a refund
# count for an event AND the baseline was simulated with "all 4-star
# characters at max copies" checked, since that's the only condition under
# which 4-star pulls reliably convert to refunds at all. Rates are the
# empirical average refunds per pull spent on that banner under max-copy
# refunding; the dupe bonus (imported from simulation.py, the canonical
# source, so both stay in sync) accounts for a 5-star win on a banner where
# the featured item's first copy is already owned, since a repeat win
# itself refunds a fixed amount on top of the per-pull rate.
_REFUND_RATE = {"character": 0.1105, "weapon": 0.0578}


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
    "route_start": "Pulls going into every route of your plan, after everything above",
    "route_run_start": "Average pulls going into this run, across the simulated runs that won every run before it",
    "route_win": "Share of the simulated runs reaching this run that won it",
    "route_loss": "Share of the simulated runs reaching this run that lost the 5-star here, which is where your plan stops",
    "route_ran_out": "Share of the simulated runs reaching this run that ran out of pulls before finishing it",
    "route_capped": "Share of the simulated runs reaching this run that hit its pull or pity limit without getting every copy",
    "route_obtained": "What this route ends up with, from the runs it won",
    "route_pity_kept": "Average pity left on this banner after stopping at the limit; it carries over to your next pulls on this banner",
    "route_omitted": "Combined chance of the rarer routes not drawn individually, out of every simulated run",
    "stopped": "Pulls kept after choosing to stop pulling at this point",
    "reserve": "Pulls you're keeping in reserve, set aside before anything below is simulated",
    "reserve_op": "Pulls set aside, not spent",
    "available": "Pulls actually available to spend after setting the reserve aside",
    "restated_total": "Total pulls as stated in your question, replacing the starting number",
    "route_leftover_won": "Average pulls left after this run, across the simulated runs that won it",
    "route_leftover_lost": "Average pulls kept after stopping here, across the simulated runs that lost this run",
    "route_factor": "This run's rate, from the line above",
    "multiply_op": "Multiplied: each run only happens when the run before it was won",
    "route_chance": "Chance of this whole route, out of every simulated run",
    "route_guarantee": "Losing that 5-star means the next 5-star on this banner is guaranteed to be the featured one, whenever you pull on it again",
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


def _has_limit(step):
    return step.get("max_pulls") is not None or step.get("stop_at_pity") is not None


def active_plan(extracted):
    """The extraction's plan, only when at least one step can end early: it
    stops the plan on a loss, or it has a pull or pity limit. A plain pull
    ORDER with neither ("weapon first, then the character") is already what
    the baseline strategy plus the goal fields express, so it's ignored
    here and keeps today's behavior."""
    plan = extracted.get("plan") or []
    return plan if any(step.get("on_loss") == "stop" or _has_limit(step) for step in plan) else None


def _validate_plan(plan, hard_pity):
    if len(plan) > MAX_PLAN_STEPS:
        return f"plan has {len(plan)} steps, more than the max of {MAX_PLAN_STEPS}"
    for expected_order, step in enumerate(plan, start=1):
        if step.get("step_order") != expected_order:
            return (
                f"plan steps must be numbered sequentially starting from 1 in pulling order; "
                f"expected step_order {expected_order}, got {step.get('step_order')}"
            )
        if step.get("banner_type") not in ("character", "weapon"):
            return f"plan step {expected_order}: banner_type must be 'character' or 'weapon'"
        copies = step.get("copies")
        if not isinstance(copies, int) or copies < 1:
            return f"plan step {expected_order}: copies ({copies}) must be an integer >= 1"
        if step.get("on_loss") not in ("stop", "continue"):
            return f"plan step {expected_order}: on_loss must be 'stop' or 'continue'"
        max_pulls = step.get("max_pulls")
        if max_pulls is not None and (not isinstance(max_pulls, int) or max_pulls < 1):
            return f"plan step {expected_order}: max_pulls ({max_pulls}) must be an integer >= 1"
        stop_at_pity = step.get("stop_at_pity")
        banner_hard_pity = hard_pity[step["banner_type"]]
        if stop_at_pity is not None and (
                not isinstance(stop_at_pity, int) or not 1 <= stop_at_pity <= banner_hard_pity):
            return f"plan step {expected_order}: stop_at_pity ({stop_at_pity}) is out of range 1-{banner_hard_pity}"
        if _has_limit(step) and step.get("on_cap") not in ("stop", "continue"):
            return (
                f"plan step {expected_order} has a pull or pity limit, so on_cap must say what happens at "
                f"it: 'stop' (end the plan) or 'continue' (move on to the next step)"
            )
    return None


def _start_state_pills(total_pulls, char_guarantee, weapon_guarantee, tooltip_key="start_pulls"):
    return [
        _pill("number", total_pulls, "cyan", tooltip_key),
        _pill("banner", "CHARACTER", "default", "banner_marker"),
        _pill("flag", "GUARANTEE: TRUE" if char_guarantee else "GUARANTEE: FALSE",
              "green" if char_guarantee else "red", "guarantee"),
        _pill("banner", "WEAPON", "default", "banner_marker"),
        _pill("flag", "GUARANTEE: TRUE" if weapon_guarantee else "GUARANTEE: FALSE",
              "green" if weapon_guarantee else "red", "guarantee"),
    ]


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


def _process_events(events, running_pulls, desired_characters, desired_weapons, full_4star_chars, prior=None):
    """Walk the event sequence in order, applying the deterministic game
    rules, and build one pill-line per event. Returns (lines, char_obtained,
    weapon_obtained, char_guarantee, weapon_guarantee, char_events, weapon_events,
    running_pulls, any_unknown_pity, error, budget_conflict).

    A budget overrun (the events alone consume more than the starting
    budget) is a genuine ambiguity, not a parse failure, the user may have
    gained pulls since the initial situation, the initial number may have
    been wrong, or a reported event number may simply be wrong; it's
    reported separately from `error` so the caller can offer a clarifying
    question instead of declining outright. At most one can ever occur,
    it's a single cumulative running total, not an independent per-event
    check, so `budget_conflict` is a single dict or None, never a list.

    `prior` (a branch continuing from shared events reconciled earlier, see
    reconcile) seeds the copies already obtained and runs already shown per
    banner, so a branch's lines keep numbering on, its "obtained X of Y"
    counts what the shared part already won, and a repeat copy still earns
    its dupe refund. The returned event counts stay this call's own."""
    starting_budget = running_pulls
    prior = prior or {}
    prior_runs = prior.get("runs") or {"character": 0, "weapon": 0}
    counts = dict(prior.get("obtained") or {"character": 0, "weapon": 0})
    guarantees = {"character": False, "weapon": False}
    run_index = dict(prior_runs)
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
            budget_conflict = {
                "event_order": event["event_order"],
                "starting_budget": starting_budget,
                "consumed": starting_budget - running_pulls,
                "overrun": -running_pulls,
            }
            return (lines, counts["character"], counts["weapon"], guarantees["character"],
                    guarantees["weapon"], run_index["character"] - prior_runs["character"],
                    run_index["weapon"] - prior_runs["weapon"], running_pulls,
                    any_unknown_pity, None, budget_conflict)

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
            run_index["character"] - prior_runs["character"], run_index["weapon"] - prior_runs["weapon"],
            running_pulls, any_unknown_pity, None, None)


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
        # A stable id the frontend echoes back on the clarification answer
        # so advisor.py can match it to whichever conflict a fresh
        # re-extraction rediscovers, without the client needing to know
        # this conflict's own internal shape (banner/attempt_number here,
        # nothing at all for a budget conflict, see _build_budget_conflict_block).
        "conflict_key": f"pity:{conflict['banner']}:{conflict['attempt_number']}",
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
            event_lines, *_, process_error, process_budget_conflict = _process_events(
                carryover["events"], total_pulls_budget, desired_characters, desired_weapons,
                baseline_params["full_4star_chars"],
            )
            if not process_error and not process_budget_conflict:
                lines = event_lines

    return {
        "applies": True, "ok": False, "conflict": True,
        "lines": lines,
        "conflicts": [_build_conflict_block(c) for c in conflicts],
    }


def _build_budget_conflict_block(conflict):
    """The "you reported using more pulls than you started with" clarifying
    block: a header, the question offering the three likely explanations
    (gained pulls since the initial situation, the initial number was
    actually different, or something else), and the two conflicting
    figures wrapped in a red-bordered group, the same visual device as a
    pity conflict."""
    consumed = conflict["consumed"]
    starting_budget = conflict["starting_budget"]
    overrun = conflict["overrun"]
    # A branch's budget is what the shared events above left, not the
    # form's starting number, so its question has to say which it means.
    source = conflict.get("budget_source") or "the initial situation states you started with only"
    return {
        # At most one budget conflict can ever occur per reconciliation
        # (it's a single cumulative running total, not an independent
        # per-event check), so this key never needs a per-conflict suffix.
        "conflict_key": "budget",
        "event_order": conflict["event_order"],
        "header": "For Total Pulls Used",
        "question": (
            f"You reported a total of {consumed} pulls used across your events, but {source} "
            f"{starting_budget} pulls, {overrun} short. Did "
            f"you gain {overrun} more pulls since the initial situation, did the initial number of "
            f"pulls change, or is it something else?"
        ),
        "pills": [{
            "kind": "group",
            "color": "red",
            "pills": [
                {
                    "kind": "number", "value": consumed, "color": "red",
                    "tooltip": (
                        f"Total pulls consumed by your reported events, {overrun} more than the "
                        f"{starting_budget}-pull starting budget"
                    ),
                },
                {"kind": "operator", "value": ">", "color": "red", "tooltip": "Exceeds"},
                {
                    "kind": "number", "value": starting_budget, "color": "red",
                    "tooltip": "The starting number of pulls from the initial situation",
                },
            ],
        }],
    }


def _build_budget_conflict_result(lines, conflict):
    """Build the response for a budget-overrun conflict: the pill lines
    _process_events already built for every event before the overrun
    (nothing about them is in question), plus the one flagged block for
    the overrun itself. Unlike a pity conflict, at most one of these can
    ever occur per reconciliation."""
    return {
        "applies": True, "ok": False, "conflict": True,
        "lines": lines,
        "conflicts": [_build_budget_conflict_block(conflict)],
    }


def reconcile(extracted, baseline_params, baseline_stats, prior=None):
    """Turn the extraction model's structured event sequence into a
    verified, ready-to-simulate state plus the structured pill data the UI
    renders, or a specific error to feed back for one corrective
    re-extraction.

    Returns a dict. `applies` is False when the question described no event
    sequence at all (a pure hypothetical or a pure strategy question), the
    caller should fall through to the existing baseline-driven flow. When
    `applies` is True, `ok` says whether reconciliation succeeded; on
    failure `error` names the specific inconsistency for the retry prompt.

    Every successful result carries `end_state` (the pulls, pity,
    guarantees, copies, and runs after everything reconciled here) and
    `fact_line_count` (how many leading lines are the question's own facts,
    before the start-state and goal lines), so a branching question can
    reconcile its shared part once and then each branch FROM that state:
    pass the branch's baseline built from end_state, plus `prior` ({"runs",
    "obtained"} per banner, and optionally "budget_source", the wording a
    budget clarifying question should use for where that budget came from).
    """
    if not extracted.get("has_event_sequence"):
        return {"applies": False}

    events = extracted.get("events") or []
    # A conditional plan is the future half of the question (what to pull
    # next, and when to stop); events stay the past half. Either may be
    # empty, so a plan alone still reconciles from the starting situation.
    plan = active_plan(extracted)
    goal_only = not events and not plan and any(
        extracted.get(key) for key in (
            "additional_character_copies_wanted", "additional_weapon_copies_wanted",
            "characters_wanted_total", "weapons_wanted_total",
        )
    )
    # A pure budget what-if ("what if I had 20 more pulls", "if I keep 40
    # in reserve"): no events, plan, or goal change, just a different pool
    # of pulls, which is still a deterministic, displayable fact.
    budget_only = not events and not plan and not goal_only and any(
        extracted.get(key) for key in ("additional_pulls_stated", "total_pulls_restated", "reserve_pulls")
    )
    if not events and not goal_only and not plan and not budget_only:
        return {"applies": True, "ok": False,
                "error": "has_event_sequence was true but no events were reported"}

    error = _validate_events(events, baseline_params["char_pity_config"], baseline_params["weapon_pity_config"])
    if not error and plan:
        error = _validate_plan(plan, {"character": baseline_params["char_pity_config"]["hard_pity"],
                                      "weapon": baseline_params["weapon_pity_config"]["hard_pity"]})
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
    # An explicit replacement ("change my goal to ...") overrides the
    # baseline total outright; anything else adds to it.
    chars_total = extracted.get("characters_wanted_total")
    weapons_total = extracted.get("weapons_wanted_total")
    desired_characters = max(
        chars_total if chars_total is not None else baseline_stats["desired_characters"] + additional_chars, 0)
    desired_weapons = max(
        weapons_total if weapons_total is not None else baseline_stats["desired_weapons"] + additional_weapons, 0)

    if desired_characters > MAX_CHARACTER_COPIES:
        return {"applies": True, "ok": False,
                "error": f"requested character copies ({desired_characters}) exceed the max of {MAX_CHARACTER_COPIES}"}
    if desired_weapons > MAX_WEAPON_COPIES:
        return {"applies": True, "ok": False,
                "error": f"requested weapon copies ({desired_weapons}) exceed the max of {MAX_WEAPON_COPIES}"}

    additional_pulls = extracted.get("additional_pulls_stated") or 0
    if additional_pulls < 0:
        return {"applies": True, "ok": False, "error": f"additional_pulls_stated ({additional_pulls}) cannot be negative"}
    reserve = extracted.get("reserve_pulls") or 0
    if reserve < 0:
        return {"applies": True, "ok": False, "error": f"reserve_pulls ({reserve}) cannot be negative"}

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
     error, budget_conflict) = _process_events(events, total_pulls_budget, desired_characters, desired_weapons,
                                                baseline_params["full_4star_chars"], prior=prior)
    if budget_conflict and prior and prior.get("budget_source"):
        budget_conflict["budget_source"] = prior["budget_source"]
    if budget_conflict:
        return _build_budget_conflict_result(event_lines, budget_conflict)
    if error:
        return {"applies": True, "ok": False, "error": error}

    lines = list(event_lines)
    stated = extracted.get("pulls_remaining_stated")
    if not events and total_pulls_budget != baseline_params["total_pulls"]:
        # With events, a restated total shows up in their own running
        # arithmetic; without any, this line is the only place it would.
        lines.append({"label": "Total Pulls Stated",
                      "pills": [_pill("number", total_pulls_budget, "cyan", "restated_total")]})

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

    if reserve:
        # Setting pulls aside up front is exactly the same as never spending
        # below them: refunds land in the spendable pool either way.
        available = max(running_pulls - reserve, 0)
        lines.append({
            "label": "Pulls Kept in Reserve",
            "pills": [
                _pill("number", running_pulls, "cyan", "start_pulls"),
                _pill("operator", "−", "magenta", "reserve_op"),
                _pill("number", reserve, "cyan", "reserve"),
                _pill("operator", "=", "magenta", "equals_op"),
                _pill("result", available, "light_green", "available"),
            ],
        })
        running_pulls = available

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

    # A banner nobody mentioned keeps whatever the original form stated for
    # it (nothing here contradicts that); a banner that appeared in the
    # sequence gets its pity forced to 0 (an outcome, win or loss, always
    # means a 5-star was just obtained) and its guarantee from the last
    # event on it.
    start_char_pity = 0 if char_events else baseline_params["start_char_pity"]
    start_char_guarantee = char_guarantee if char_events else baseline_params["start_char_guarantee"]
    start_weapon_pity = 0 if weapon_events else baseline_params["start_weapon_pity"]
    start_weapon_guarantee = weapon_guarantee if weapon_events else baseline_params["start_weapon_guarantee"]

    prior_runs = (prior or {}).get("runs") or {"character": 0, "weapon": 0}
    end_state = {
        "total_pulls": total_pulls,
        "start_char_pity": start_char_pity, "start_char_guarantee": start_char_guarantee,
        "start_weapon_pity": start_weapon_pity, "start_weapon_guarantee": start_weapon_guarantee,
        "desired_characters": desired_characters, "desired_weapons": desired_weapons,
        "obtained": {"character": char_obtained, "weapon": weapon_obtained},
        "runs": {"character": prior_runs["character"] + char_events,
                 "weapon": prior_runs["weapon"] + weapon_events},
    }
    fact_line_count = len(lines)

    def _with_state(result):
        if result.get("ok"):
            result["end_state"] = end_state
            result["fact_line_count"] = fact_line_count
        return result

    if plan:
        return _with_state(_reconcile_plan(
            plan, lines, total_pulls, char_obtained, weapon_obtained,
            start_char_pity, start_char_guarantee, start_weapon_pity, start_weapon_guarantee,
        ))

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
        return _with_state({
            "applies": True, "ok": True, "goal_complete": True, "pulls_exhausted": False,
            "strategy": [], "total_pulls": total_pulls,
            "start_char_pity": 0, "start_char_guarantee": False,
            "start_weapon_pity": 0, "start_weapon_guarantee": False,
            "remaining_characters": 0, "remaining_weapons": 0,
            "lines": lines,
        })

    if total_pulls <= 0:
        lines.append(goal_line)
        lines.append({"label": "Goal Status",
                      "pills": [_pill("status", "NO PULLS REMAIN", "red", "status")]})
        return _with_state({
            "applies": True, "ok": True, "goal_complete": False, "pulls_exhausted": True,
            "strategy": [], "total_pulls": 0,
            "start_char_pity": 0, "start_char_guarantee": False,
            "start_weapon_pity": 0, "start_weapon_guarantee": False,
            "remaining_characters": chars_remaining, "remaining_weapons": weapons_remaining,
            "lines": lines,
        })

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
            "pills": _start_state_pills(total_pulls, start_char_guarantee, start_weapon_guarantee),
        })
    elif chars_remaining >= 1:
        lines.append({
            "label": _run_label("character", end_state["runs"]["character"] + 1, char_obtained, desired_characters),
            "pills": [
                _pill("number", total_pulls, "cyan", "start_pulls"),
                _pill("flag", "GUARANTEE: TRUE" if start_char_guarantee else "GUARANTEE: FALSE",
                      "green" if start_char_guarantee else "red", "guarantee"),
            ],
        })
    elif weapons_remaining >= 1:
        lines.append({
            "label": _run_label("weapon", end_state["runs"]["weapon"] + 1, weapon_obtained, desired_weapons),
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

    return _with_state({
        "applies": True, "ok": True, "goal_complete": False, "pulls_exhausted": False,
        "goal_only": goal_only,
        "budget_only": budget_only,
        "goal_text": remaining_goal_text(desired_characters, desired_weapons),
        "strategy": strategy,
        "total_pulls": total_pulls,
        "start_char_pity": start_char_pity,
        "start_char_guarantee": start_char_guarantee,
        "start_weapon_pity": start_weapon_pity,
        "start_weapon_guarantee": start_weapon_guarantee,
        "remaining_characters": chars_remaining, "remaining_weapons": weapons_remaining,
        "lines": lines,
    })


def _reconcile_plan(plan, lines, total_pulls, char_obtained, weapon_obtained,
                    start_char_pity, start_char_guarantee, start_weapon_pity, start_weapon_guarantee):
    """Finish reconciling a question with a conditional plan: everything
    before it (events, additional pulls) is already folded into `lines`
    and the start state, so this only adds the shared starting point every
    route branches from and the plan's own goal. The plan's steps ARE the
    goal from here on, the user spelled out exactly what they'll pull for,
    so the copies come from them rather than the baseline goal."""
    plan_chars = sum(s["copies"] for s in plan if s["banner_type"] == "character")
    plan_weapons = sum(s["copies"] for s in plan if s["banner_type"] == "weapon")
    if char_obtained + plan_chars > MAX_CHARACTER_COPIES:
        return {"applies": True, "ok": False, "error": (
            f"the plan's character copies ({plan_chars}) plus those already obtained ({char_obtained}) "
            f"exceed the max of {MAX_CHARACTER_COPIES}"
        )}
    if weapon_obtained + plan_weapons > MAX_WEAPON_COPIES:
        return {"applies": True, "ok": False, "error": (
            f"the plan's weapon copies ({plan_weapons}) plus those already obtained ({weapon_obtained}) "
            f"exceed the max of {MAX_WEAPON_COPIES}"
        )}

    # A pity limit is an absolute counter value, so on a banner's first
    # step it has to be above the pity that banner already starts at;
    # "stop at 75 pity" with 80 already on it can't mean anything. Later
    # steps on the same banner start from wherever each simulated run left
    # it, and simply end at once if that's already past the limit.
    start_pity = {"character": start_char_pity, "weapon": start_weapon_pity}
    seen = set()
    for step in plan:
        banner = step["banner_type"]
        limit = step.get("stop_at_pity")
        if banner not in seen and limit is not None and limit <= start_pity[banner]:
            return {"applies": True, "ok": False, "error": (
                f"plan step {step['step_order']}: stop_at_pity ({limit}) must be above the {start_pity[banner]} "
                f"pity the {banner} banner already starts this plan at"
            )}
        seen.add(banner)

    lines = list(lines)
    goal_line = {"label": "Current Goal",
                 "pills": [_pill("goal", remaining_goal_text(plan_chars, plan_weapons), "default", "goal")]}
    if total_pulls <= 0:
        lines.append(goal_line)
        lines.append({"label": "Goal Status",
                      "pills": [_pill("status", "NO PULLS REMAIN", "red", "status")]})
        return {
            "applies": True, "ok": True, "goal_complete": False, "pulls_exhausted": True,
            "strategy": [], "total_pulls": 0,
            "start_char_pity": 0, "start_char_guarantee": False,
            "start_weapon_pity": 0, "start_weapon_guarantee": False,
            "remaining_characters": plan_chars, "remaining_weapons": plan_weapons,
            "lines": lines,
        }

    lines.append({
        "label": "Starting Point for Every Route",
        "pills": _start_state_pills(total_pulls, start_char_guarantee, start_weapon_guarantee, "route_start"),
    })
    lines.append(goal_line)

    return {
        "applies": True, "ok": True, "goal_complete": False, "pulls_exhausted": False,
        "plan": True, "goal_only": False,
        "goal_text": remaining_goal_text(plan_chars, plan_weapons),
        "strategy": [
            {"banner": "char" if s["banner_type"] == "character" else "weapon",
             "copies": s["copies"], "stop_on_loss": s["on_loss"] == "stop",
             "max_pulls": s.get("max_pulls"), "cap_pity": s.get("stop_at_pity"),
             "on_cap": s.get("on_cap") if _has_limit(s) else None}
            for s in plan
        ],
        "total_pulls": total_pulls,
        "start_char_pity": start_char_pity,
        "start_char_guarantee": start_char_guarantee,
        "start_weapon_pity": start_weapon_pity,
        "start_weapon_guarantee": start_weapon_guarantee,
        "remaining_characters": plan_chars, "remaining_weapons": plan_weapons,
        "lines": lines,
    }


_PHASE_BANNER = {"char": "character", "weapon": "weapon"}


def route_step_name(phase):
    copies = phase["copies"]
    suffix = f" ({copies} copies)" if copies > 1 else ""
    return f"{_BANNER_LABEL[_PHASE_BANNER[phase['banner']]]}{suffix}"


def _pulls_pill(value, sample, tooltip_key, suffix="", color="light_green"):
    """An averaged pull count, as a whole number (pulls are whole, the
    decimals of an average would be false precision), flagged "~" with the
    sample size in its tooltip when too few simulated runs back it."""
    if value is None or sample == 0:
        return _pill("number", "N/A", "default", tooltip_key)
    # A bare count stays an int, like every other number pill.
    pill = _pill("number", f"{round(value)}{suffix}" if suffix else round(value), color, tooltip_key)
    if sample < MIN_ROUTE_SAMPLE:
        pill["value"] = f"~{pill['value']}"
        pill["tooltip"] += f" (approximate: only {sample} simulated runs got here)"
    return pill


# Display order of a step's outcomes, so routes read best case first:
# Route 1 wins every step.
_OUTCOME_RANK = {"won": 0, "capped": 1, "lost": 2, "ran_out": 3}
# A runaway guard on how many routes are drawn; any further, rarer ones
# are summarized in one line instead of each getting its own card.
MAX_ROUTES = 6


def step_limit_note(phase):
    limits = []
    if phase.get("max_pulls"):
        limits.append(f"at most {phase['max_pulls']} pulls")
    if phase.get("cap_pity"):
        limits.append(f"up to {phase['cap_pity']} pity")
    notes = []
    if phase.get("stop_on_loss"):
        notes.append("stop if lost")
    if limits:
        then = "then stop" if phase.get("on_cap") == "stop" else "then move on"
        notes.append(f"{' or '.join(limits)}, {then}")
    return "".join(f", {n}" for n in notes)


def _route_step_outcome_text(phase, outcome):
    name = route_step_name(phase)
    if outcome == "won":
        return f"{name} {'won' if phase.get('stop_on_loss') else 'obtained'}"
    if outcome == "capped":
        return f"{name} capped{', stop' if phase.get('on_cap') == 'stop' else ''}"
    return f"{name} lost, stop"


def _obtained_text(strategy, path):
    copies = {"char": 0, "weapon": 0}
    for phase, outcome in zip(strategy, path):
        if outcome == "won":
            copies[phase["banner"]] += phase["copies"]
    if not copies["char"] and not copies["weapon"]:
        return "NONE"
    return remaining_goal_text(copies["char"], copies["weapon"]).upper()


def build_route_groups(strategy, plan_result):
    """Split a simulated conditional plan (simulation.simulate_plan) into
    routes, one per way the plan can end, as pill lines. A route is a path
    through the plan's outcome tree: each step either won, hit its limit
    ("capped"), or lost and stopped the plan. Routes are ordered best case
    first, so Route 1 wins every step.

    A step's rate is conditional on reaching it, so a route's chance is the
    product of the rates along it, shown as that literal multiplication.
    The product is taken from the displayed (rounded) rates so the visible
    arithmetic always checks out; since each step's trials are exactly its
    parent's, it equals the route's share of all trials to within that
    rounding. Running out of pulls never gets a route of its own: it's
    shown as a "RAN OUT" share on the run lines it happened on, so the
    routes plus those shares account for every trial. A route no
    simulated run took never appears at all.

    Returns (routes, omitted). routes is a list of {"label", "lines",
    "chance", "avg_leftover", "sample", "ending", "obtained",
    "stopped_banner"} dicts ("ending" is "complete", "lost", "capped_stop",
    or "partial", a capped step moved on, so not every copy was won);
    omitted is None, or {"routes", "chance"} for routes beyond MAX_ROUTES,
    rarest first, that were left out."""
    nodes, leaves = plan_result["nodes"], plan_result["leaves"]

    def rate(path):
        return round(nodes[path]["count"] / nodes[path[:-1]]["count"] * 100, 2)

    def run_line(path):
        i = len(path) - 1
        phase, outcome, parent = strategy[i], path[-1], path[:-1]
        ends_plan = outcome == "lost" or (outcome == "capped" and phase.get("on_cap") == "stop")
        r = rate(path)
        outcome_pill = {
            "won": _pill("outcome", f"WIN {r:.2f}%", "green", "route_win"),
            "capped": _pill("outcome", f"CAPPED {r:.2f}%", "default", "route_capped"),
            "lost": _pill("outcome", f"LOSS {r:.2f}%", "red", "route_loss"),
        }[outcome]
        pills = [
            _pulls_pill(nodes[parent]["avg_leftover"], nodes[parent]["count"],
                        "route_start" if i == 0 else "route_run_start", color="cyan"),
            outcome_pill,
        ]
        ran_out = parent + ("ran_out",)
        if outcome != "lost" and ran_out in nodes:
            pills.append(_pill("outcome", f"RAN OUT {rate(ran_out):.2f}%", "red", "route_ran_out"))
        if outcome == "capped" and nodes[path].get("avg_pity") is not None:
            pills.append(_pill("number", f"{round(nodes[path]['avg_pity'])} PITY KEPT", "default", "route_pity_kept"))
        pills.append(_pulls_pill(
            nodes[path]["avg_leftover"], nodes[path]["count"],
            "route_leftover_lost" if ends_plan else "route_leftover_won", " KEPT" if ends_plan else " LEFT",
        ))
        return {"label": f"Run {i + 1}: {route_step_name(phase)}{step_limit_note(phase)}", "pills": pills}, r

    candidates = [path for path in leaves if path and path[-1] != "ran_out"]
    candidates.sort(key=lambda path: -leaves[path])
    kept, dropped = candidates[:MAX_ROUTES], candidates[MAX_ROUTES:]
    kept.sort(key=lambda path: [_OUTCOME_RANK[o] for o in path])

    routes = []
    for path in kept:
        lines, factors = [], []
        for i in range(len(path)):
            line, r = run_line(path[:i + 1])
            lines.append(line)
            factors.append(r)

        chance = factors[0]
        for f in factors[1:]:
            chance = chance * f / 100
        chance = round(chance, 2)
        if len(factors) == 1:
            result_pills = [_pill("result", f"{chance:.2f}%", "light_green", "route_chance")]
        else:
            result_pills = []
            for j, f in enumerate(factors):
                if j:
                    result_pills.append(_pill("operator", "×", "magenta", "multiply_op"))
                result_pills.append(_pill("number", f"{f:.2f}%", "cyan", "route_factor"))
            result_pills += [_pill("operator", "=", "magenta", "equals_op"),
                             _pill("result", f"{chance:.2f}%", "light_green", "route_chance")]

        last_phase, last = strategy[len(path) - 1], path[-1]
        if last == "lost":
            ending = "lost"
        elif last == "capped" and last_phase.get("on_cap") == "stop":
            ending = "capped_stop"
        elif len(path) == len(strategy) and all(o == "won" for o in path):
            ending = "complete"
        else:
            ending = "partial"
        stopped = ending in ("lost", "capped_stop")
        obtained = _obtained_text(strategy, path)
        result_pills.append(_pulls_pill(
            nodes[path]["avg_leftover"], leaves[path],
            "route_leftover_lost" if stopped else "route_leftover_won", " KEPT" if stopped else " LEFT",
        ))
        result_pills.append(_pill("status", f"OBTAINED: {obtained}", "green" if ending == "complete" else "default",
                                  "route_obtained"))
        stopped_banner = _PHASE_BANNER[last_phase["banner"]] if ending == "lost" else None
        if stopped_banner:
            result_pills.append(_pill("flag", f"{stopped_banner.upper()} GUARANTEE: TRUE", "green", "route_guarantee"))
        lines.append({"label": "Route Result", "pills": result_pills})

        routes.append({
            "label": ", then ".join(_route_step_outcome_text(strategy[i], o) for i, o in enumerate(path)),
            "lines": lines, "chance": chance,
            "avg_leftover": nodes[path]["avg_leftover"], "sample": leaves[path],
            "ending": ending, "obtained": obtained.lower(), "stopped_banner": stopped_banner,
            "capped_pity": [
                (_PHASE_BANNER[strategy[i]["banner"]], round(nodes[path[:i + 1]]["avg_pity"]))
                for i, o in enumerate(path) if o == "capped" and nodes[path[:i + 1]].get("avg_pity") is not None
            ],
        })

    for n, route in enumerate(routes, start=1):
        route["label"] = f"Route {n}: {route['label']}"

    omitted = None
    if dropped:
        omitted = {"routes": len(dropped),
                   "chance": round(sum(leaves[p] for p in dropped) / plan_result["trials"] * 100, 2)}
    return routes, omitted


def build_stop_line(scenario):
    """A branch that stops pulling by choice ends on what it keeps: the
    pulls left and each banner's guarantee, which carry over to whenever
    the player pulls again."""
    return {
        "label": "Stops Here",
        "pills": [
            _pill("number", f"{scenario['total_pulls']} KEPT", "light_green", "stopped"),
            *_start_state_pills(scenario["total_pulls"], scenario["start_char_guarantee"],
                                scenario["start_weapon_guarantee"])[1:],
        ],
    }


def build_omitted_routes_group(omitted):
    """The one summary card for routes beyond MAX_ROUTES, so their share of
    simulated runs is still accounted for rather than silently missing."""
    return {
        "label": "Other Routes",
        "lines": [{
            "label": f"{omitted['routes']} rarer route{'s' if omitted['routes'] != 1 else ''} not shown",
            "pills": [_pill("result", f"{omitted['chance']:.2f}%", "default", "route_omitted")],
        }],
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

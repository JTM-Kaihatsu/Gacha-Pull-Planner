"""advisor.py
Layer 3 of the advisor redesign: an optional, open-ended follow-up advisor.

This is the one place a model earns its keep. It takes a free-text question that
the deterministic presets do not cover, maps it to one or more simulation runs
via tool use, and interprets the results against the baseline. Everything factual
still comes from the simulation; the model only decides what to run and explains
the difference.

Guardrails: a capped number of tool calls per question, argument validation
before any sim runs, and a reduced trial count to keep each call fast. The caller
wraps this so a rate limit or error degrades gracefully (see main.py /advise).
"""
import json
import re

from openai import OpenAI

from analyzer import describe_goal
from config import get_openai_api_key, get_model
from session_state import (
    build_agent_cycle_line, build_result_line, build_spending_tier_line, reconcile, remaining_goal_text,
)
from simulation import run_simulation_verbose

# Fewer trials than the main endpoint: the advisor may run several sims per
# question, and it only needs directional numbers, not publication precision.
ADVISOR_TRIALS = 4000
MAX_TOOL_CALLS = 4
# Primary scenario plus at most this many additional branches, a runaway
# guard against a question the model misreads as branching far more than it
# actually does; extras beyond this are dropped, not rejected outright.
MAX_SCENARIOS = 4

# The "spend if you really want it" tier's target: comfortably likely
# without paying for a full guarantee. Below this, the F2P figure already
# covers it and there's nothing to upsell; above it just starts eating into
# the guaranteed tier's territory for diminishing return.
MODERATE_SPEND_TARGET = 0.85
# A reduced trial count for the tier search's own probing calls: it only
# needs to find roughly the right pull count, not publication precision,
# and this runs several times per answer. The pull count it lands on is
# always re-confirmed with a full ADVISOR_TRIALS run before being shown.
TIER_SEARCH_TRIALS = 800

# Every call in this module was left at the API default (1.0, its max) until
# a live-testing session caught the model citing a percentage in prose with
# no corresponding tool call behind it. A low, consistent temperature across
# every call here (extraction and interpretation alike) makes that kind of
# unprompted embellishment meaningfully less likely, in keeping with this
# module's whole "grounded, not creative" mandate.
ADVISOR_TEMPERATURE = 0.2

# Hardcoded, not an LLM call: when an event sequence can't be reconciled even
# after one corrective retry, decline plainly rather than guess. This is
# deterministic on purpose, the same lesson as everywhere else in this
# module, a model asked to relay a specific message has repeatedly failed to
# do so reliably.
PARSE_FAILURE_MESSAGE = (
    "The AI couldn't parse your situation, could you please enter it in a different way? "
    "For example: \"I pulled and won the character I wanted early (30 pity, with 5 refunds), "
    "then lost the weapon banner also early (50 pity, with 3 refunds). Is it worth it for me "
    "to keep pulling?\""
)

EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "event_order": {
            "type": "integer",
            "description": "1-based position of this event in the sequence it occurred, in order.",
        },
        "banner_type": {"type": "string", "enum": ["character", "weapon"]},
        "outcome": {"type": "string", "enum": ["win", "loss"]},
        "pity_at_outcome": {
            "type": ["integer", "null"],
            "description": "The number the question gives for this outcome: either the absolute pity COUNTER value ('at 30 pity', 'reached pity 75') or a count of pulls spent in this specific run ('after 50 pulls', 'used 42 pulls'), see pity_is_absolute for which. Null if the question did not give an exact pity/pull count for this specific event (e.g. 'I won with 62 to spare', 'I lost, and I have 81 pulls left'); in that case report pulls_remaining_stated instead so the true total can still be known.",
        },
        "pity_is_absolute": {
            "type": "boolean",
            "description": "true if pity_at_outcome is the absolute pity COUNTER value, phrasing like 'at 30 pity' or 'reached pity 75'. false if pity_at_outcome is instead a count of pulls spent in this run, phrasing like 'after 50 pulls', 'used 42 pulls', 'spent 30 pulls'. This only matters when the banner already had pity carried over from before this conversation (see the Starting Situation context); it's ignored otherwise, and ignored when pity_at_outcome is null, set it to true in either of those cases if unsure.",
        },
        "refund_count": {
            "type": ["integer", "null"],
            "description": "4-star refund pulls received back during this event, if explicitly stated (this includes an explicit '0'/'no refunds'). Null if the question simply does not mention refunds for this event at all; do not assume 0 in that case, a deterministic formula estimates it instead when applicable.",
        },
    },
    "required": ["event_order", "banner_type", "outcome", "pity_at_outcome", "pity_is_absolute", "refund_count"],
    "additionalProperties": False,
}

# Shared by the primary scenario (SESSION_STATE_SCHEMA's own top-level
# properties) and every entry of additional_scenarios below: one scenario is
# always this same shape, whether it's the only one described or one of
# several mutually exclusive branches.
_SCENARIO_FIELDS = {
    "events": {
        "type": "array",
        "description": (
            "One object per discrete pull event, in the order they occurred. Each "
            "event is atomic: which banner, whether it was won or lost, the pity "
            "count at that outcome, and any 4-star refund received. Do not sum, "
            "combine, or reorder events yourself; report exactly what is stated, "
            "one event at a time."
        ),
        "items": EVENT_SCHEMA,
    },
    "pulls_remaining_stated": {
        "type": ["integer", "null"],
        "description": "A direct, total restatement of pulls remaining, ONLY when the question's own words state a specific leftover pull count as fact (e.g. 'I have 41 pulls left', 'I'm down to 23'). Never compute this yourself by subtracting a pity count (or anything else) from the total pulls; a question that only gives an event's pity and outcome, with no separate remaining-pulls statement, must leave this null, the true remaining total is computed deterministically from the events instead. Do not use this field for an amount meant to be added on top of the remaining pulls; use additional_pulls_stated for that instead.",
    },
    "additional_pulls_stated": {
        "type": ["integer", "null"],
        "description": "An amount of pulls stated as an ADDITION on top of whatever remains, not a total (e.g. 'plus around 45 more from this patch', 'and I'll get another 20 next week'). Report the raw number only; do not add it to anything yourself.",
    },
    "total_pulls_restated": {"type": ["integer", "null"]},
    "additional_character_copies_wanted": {
        "type": ["integer", "null"],
        "description": "A signed CHANGE in character copies wanted, relative to the original goal, not a new total. Positive for extra copies wanted BEYOND the original goal (e.g. 'another character copy' = 1). Negative to reduce the goal below its original count, for example the question decides to give up on further copies of that banner after a bad outcome (e.g. 'if I lose the 50/50, I'll skip the second character copy and only go for the weapon' = -1 for that scenario; giving up on the banner entirely = negative enough to cover every copy still wanted, e.g. -2 if 2 were still wanted). Do not use this for the original goal itself, only for a stated change to it.",
    },
    "additional_weapon_copies_wanted": {
        "type": ["integer", "null"],
        "description": "Same rule as additional_character_copies_wanted (a signed change, positive to add, negative to reduce below the original goal), for the weapon banner.",
    },
}

# One further branch beyond the primary scenario: same shape as the fields
# above, plus its own label. Used only for a question that genuinely
# describes two or more mutually exclusive futures hinging on a pull that
# hasn't happened yet ("if I win the character... if I lose...").
SCENARIO_SCHEMA = {
    "type": "object",
    "properties": {
        "condition_label": {
            "type": "string",
            "description": "Short label naming the condition that leads to this scenario, e.g. 'If I win the character pull around pity 30'.",
        },
        **_SCENARIO_FIELDS,
    },
    "required": ["condition_label", *_SCENARIO_FIELDS.keys()],
    "additionalProperties": False,
}

SESSION_STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "has_event_sequence": {
            "type": "boolean",
            "description": (
                "true if the question gives one or more concrete pull outcomes to reason "
                "about: either an event that already happened this session (a banner "
                "pulled to a win or a loss, with a pity count), OR a concrete hypothetical "
                "the question itself poses as a specific outcome to test, for example 'if "
                "I win the character pull around pity 30, ...; if I lose around pity 75, "
                "...'. That second case is still a concrete outcome (a banner, a win/loss, "
                "a pity figure), just a proposed one rather than one that already happened; "
                "extract it exactly like a real event. false only for a pure hypothetical or "
                "strategy question with no concrete outcome at all, for example 'how should "
                "I split my budget between characters and weapons' or 'what if I had 20 more "
                "pulls', where there is no banner/win-or-loss/pity to extract."
            ),
        },
        **_SCENARIO_FIELDS,
        "condition_label": {
            "type": ["string", "null"],
            "description": "Short label for the condition that leads to the primary scenario above (the events/goal fields on this object). Null unless this question describes two or more mutually exclusive futures; when it does, every scenario, including this primary one, must have a label.",
        },
        "additional_scenarios": {
            "type": ["array", "null"],
            "description": (
                "Further mutually exclusive scenarios beyond the primary one above, ONLY "
                "when the question describes two or more futures hinging on the outcome of "
                "a pull that has not happened yet (e.g. 'if I win the character pull, I'll "
                "go for another copy; if I lose, I'll go straight for the weapon instead'). "
                "Each item has the exact same shape as the primary scenario's own fields, "
                "plus its own condition_label, and starts from the SAME starting point as "
                "the primary scenario (the same baseline pulls, pity, and goal), never as a "
                "continuation of it. Null or empty for a question with only one path; do "
                "not invent a branch for a question that only actually has one."
            ),
            "items": SCENARIO_SCHEMA,
        },
        "starting_situation_corrections": {
            "type": ["array", "null"],
            "description": (
                "ONLY set when a prior clarifying question told you a banner's starting pity "
                "or guarantee (not an event's own reported number) was wrong, and the user's "
                "answer corrects it, e.g. 'actually my weapon pity was only 10, not what the "
                "form says'. One entry per banner corrected: banner, corrected_pity (null if "
                "unchanged), corrected_guarantee (null if unchanged). Null otherwise, this is "
                "almost never used."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "banner": {"type": "string", "enum": ["character", "weapon"]},
                    "corrected_pity": {"type": ["integer", "null"]},
                    "corrected_guarantee": {"type": ["boolean", "null"]},
                },
                "required": ["banner", "corrected_pity", "corrected_guarantee"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "has_event_sequence", *_SCENARIO_FIELDS.keys(), "condition_label", "additional_scenarios",
        "starting_situation_corrections",
    ],
    "additionalProperties": False,
}

EXTRACTION_SYSTEM_PROMPT = (
    "You extract a sequence of discrete pull events from a gacha follow-up question. Do "
    "not do any math, do not aggregate multiple events into one, and do not decide a "
    "strategy. If the question describes one or more concrete pull outcomes, either "
    "actual pulls that already happened, OR a specific hypothetical the question itself "
    "poses as an outcome to test (e.g. 'if I win the character pull around pity 30...'), "
    "report each one as its own event: "
    "which banner, win or loss, and (if an exact number was given) the number stated for "
    "that outcome. That number is stated in one of two genuinely different ways, and "
    "pity_is_absolute records which: 'at 30 pity' or 'reached pity 75' states the absolute "
    "pity COUNTER value, pity_is_absolute is true; 'after 50 pulls', 'used 42 pulls', 'spent "
    "30 pulls' states a count of pulls spent in this specific run instead, pity_is_absolute "
    "is false. Getting this backwards silently produces the wrong pull count whenever the "
    "banner already had pity going into this conversation, so read the phrasing carefully "
    "rather than defaulting to one. For refund_count, report the exact number only if the question "
    "actually states one for that event, including an explicit 'no refunds' (0); if "
    "refunds simply are not mentioned for that event, leave refund_count null, do not "
    "assume 0, a deterministic formula estimates it separately when applicable. Number "
    "events sequentially starting from 1 in the order they occurred; never combine two events "
    "into one or sum their numbers together. An event requires an actual stated outcome, win "
    "or loss, whether real or a concrete hypothetical being tested. A stated INTENTION for "
    "what to pull next, with no outcome given at all, is not an event and must never be "
    "invented as one, for example 'I want to continue on with a second copy after the "
    "lightcone' states a plan, not a win or a loss for the lightcone, so no lightcone event "
    "should be reported for it; goal changes belong in additional_character_copies_wanted / "
    "additional_weapon_copies_wanted instead, and whatever isn't resolved by a real event is "
    "already covered by the remaining goal once reconciled. Users often describe an outcome without "
    "giving its exact pity, instead stating how many pulls they have left afterward, "
    "for example 'I won the character with 62 pulls to spare', 'I lost my first run at "
    "the character banner and I have 81 pulls left'. In that case still report the "
    "event (banner and win/loss), leave pity_at_outcome null for it, and put the stated "
    "figure in pulls_remaining_stated instead of trying to work out what the pity must "
    "have been. This only runs in the other direction: when an event's pity IS known, "
    "never invent a pulls_remaining_stated by subtracting that pity (or anything else) "
    "from the total pulls yourself, even if the question ends on a vague phrase like "
    "'with what's left' or 'from here'. pulls_remaining_stated must come only from the "
    "question's own words stating a specific leftover number as fact; if it does not, "
    "leave the field null and let the true remaining total be computed from the events. "
    "If the question has no concrete outcome at all to extract, no banner/win-or-loss/"
    "pity, just a pure strategy or what-if question like 'how should I split my budget' "
    "or 'what if I had 20 more pulls', set has_event_sequence to false and events to an "
    "empty list. Also report, only if explicitly stated: a direct restatement of pulls "
    "remaining (pulls_remaining_stated, required whenever any event's pity is null), "
    "an amount to add on top of whatever remains, not a total "
    "(additional_pulls_stated), a restated total pull budget (total_pulls_restated), "
    "and any signed change to the copies wanted for a banner "
    "(additional_character_copies_wanted / additional_weapon_copies_wanted): positive to "
    "add copies beyond the original goal, negative to reduce the goal below its original "
    "count, for example a scenario that gives up on further copies of a banner after a "
    "bad outcome ('I'll skip the second character copy and only go for the weapon' = -1). "
    "Leave a field null if the question does not state it. Most questions describe a single path: "
    "leave condition_label and additional_scenarios both null. Only when the question "
    "describes two or more mutually exclusive FUTURE scenarios hinging on the outcome of a "
    "pull that has not happened yet (e.g. 'if I win the character pull, I'll go for another "
    "copy and then the weapon; if I lose, I'll go straight for the weapon instead') do you "
    "report more than one: fill the fields above as normal for the first branch plus a short "
    "condition_label for it (e.g. 'If I win the character pull around pity 30'), and report "
    "every other branch as its own object in additional_scenarios, each with its own "
    "condition_label and its own events/goal fields. Every scenario starts from the exact "
    "same starting point, the same baseline pulls, pity, and goal; a later scenario is an "
    "alternative to the earlier ones, never a continuation of them. Do not invent a branch "
    "for a question that only actually describes one path. If (and only if) the context "
    "below includes a prior clarifying question and the user's answer to it, apply that "
    "answer to correct the specific event it was about: if they confirmed a reported number "
    "was a TOTAL including existing pity, set that event's pity_is_absolute to true using "
    "the exact same number; if they gave a different number for that event, use their "
    "corrected number instead; if they said the banner's STARTING pity or guarantee itself "
    "was wrong (not the event), report that in starting_situation_corrections instead and "
    "leave the event as originally reported. starting_situation_corrections is null on every "
    "other question."
)

RUN_SIMULATION_TOOL = {
    "type": "function",
    "function": {
        "name": "run_simulation",
        "description": (
            "Run the gacha pull Monte Carlo simulation with modified parameters and "
            "return the resulting odds. Use this to answer a what-if question by "
            "comparing the result against the baseline. Only the parameters you pass "
            "are changed; everything else keeps its baseline value."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "total_pulls": {"type": "integer", "description": "Total pulls available"},
                "start_char_pity": {"type": "integer"},
                "start_char_guarantee": {"type": "boolean"},
                "start_weapon_pity": {"type": "integer"},
                "start_weapon_guarantee": {"type": "boolean"},
                "strategy": {
                    "type": "array",
                    "description": "Ordered list of pull phases; order matters because pity carries across phases",
                    "items": {
                        "type": "object",
                        "properties": {
                            "banner": {"type": "string", "enum": ["char", "weapon"]},
                            "copies": {"type": "integer"},
                        },
                        "required": ["banner", "copies"],
                    },
                },
            },
        },
    },
}


def _condense(stats):
    """The small slice of a simulation result the model needs to reason."""
    return {
        "success_rate": stats["success_rate"],
        "total_pulls": stats["initial_pulls"],
        "desired_characters": stats["desired_characters"],
        "desired_weapons": stats["desired_weapons"],
        "avg_leftover_pulls_on_success": stats["avg_leftover_pulls_on_success"],
        "most_common_failure_state": stats["most_common_failure_state"],
    }


def _success_pct(stats):
    return float(stats["success_rate"].rstrip("%"))


def _guaranteed_pulls_for_banner(hard_pity, start_pity, start_guarantee, copies_needed):
    """Worst-case pulls to obtain `copies_needed` more of this banner's
    featured item using ONLY the hard-pity mechanic, never luck: pity
    always forces a 5-star by hard_pity pulls at the latest, and guarantee
    (already active, or triggered by a 50/50 loss) forces the very next
    5-star to be the featured one. The first copy uses this banner's actual
    current pity/guarantee; a win always resets both to 0/False, so every
    copy after the first starts completely fresh, worst case a lost 50/50
    at hard pity followed by a guaranteed win at hard pity again."""
    if copies_needed <= 0:
        return 0
    first = (hard_pity - start_pity) if start_guarantee else (hard_pity - start_pity) + hard_pity
    rest = (copies_needed - 1) * 2 * hard_pity
    return first + rest


def _guaranteed_scenario_pulls(scenario, baseline_params):
    """Total pulls that mathematically guarantee the scenario's entire
    remaining goal, both banners, worst case luck. Not a probability, hard
    pity forces this regardless of chance, which is exactly what a big
    spender wants to know: the number that removes risk entirely."""
    char_pulls = _guaranteed_pulls_for_banner(
        baseline_params["char_pity_config"]["hard_pity"],
        scenario["start_char_pity"], scenario["start_char_guarantee"],
        scenario["remaining_characters"],
    )
    weapon_pulls = _guaranteed_pulls_for_banner(
        baseline_params["weapon_pity_config"]["hard_pity"],
        scenario["start_weapon_pity"], scenario["start_weapon_guarantee"],
        scenario["remaining_weapons"],
    )
    return char_pulls + weapon_pulls


def _extra_pulls_for_target_rate(run_params, target_rate, extra_cap):
    """Smallest additional total_pulls (beyond run_params['total_pulls'])
    whose simulated success rate reaches target_rate, via a coarse search
    at a reduced trial count for speed. 0 if the baseline already clears
    it. extra_cap is a known-safe upper bound (the caller passes the extra
    pulls needed to fully guarantee the goal, which by definition clears
    any probability target below 100%), so this always terminates with a
    real answer rather than searching forever."""
    baseline_total = run_params["total_pulls"]
    target_pct = target_rate * 100

    def _rate(extra):
        stats = run_simulation_verbose(
            **{**run_params, "total_pulls": max(baseline_total + extra, 1)}, trials=TIER_SEARCH_TRIALS,
        )
        return _success_pct(stats)

    if extra_cap <= 0 or _rate(0) >= target_pct:
        return 0

    lo, hi = 0, extra_cap
    for _ in range(10):
        mid = (lo + hi) // 2
        if mid == lo:
            break
        if _rate(mid) >= target_pct:
            hi = mid
        else:
            lo = mid
    return hi


def _spending_tiers(scenario, run_params, baseline_params, primary_result):
    """Pre-compute the two further spending tiers beyond the F2P result
    already simulated: the smallest top-up that reaches a comfortably-safe
    success rate, and the pull count that mathematically guarantees the
    remaining goal outright. Everything here is either a real simulation
    result or closed-form hard-pity arithmetic, never a model guess, so the
    interpretation model can only narrate figures that are actually true.

    Returns (tiers, runs, comparisons). tiers is a list of
    {"key", "label", "total_pulls", "success_rate", "extra_pulls"} dicts
    (F2P always included; the other two omitted when they'd just repeat a
    figure already shown, e.g. F2P already clears the guarantee). runs is
    every further simulation actually executed, for the UI receipts.
    comparisons is every pairwise pull-count gap between the tiers that
    ARE present (see _tier_pull_comparisons), for the model to reason
    about relative effort with, never left to compute itself."""
    runs = []
    tiers = [{
        "key": "f2p", "label": "F2P (current budget)",
        "total_pulls": run_params["total_pulls"], "success_rate": primary_result["success_rate"],
        "extra_pulls": 0,
    }]

    guaranteed_pulls = _guaranteed_scenario_pulls(scenario, baseline_params)
    guaranteed_extra = max(guaranteed_pulls - run_params["total_pulls"], 0)

    moderate_extra = _extra_pulls_for_target_rate(run_params, MODERATE_SPEND_TARGET, guaranteed_extra)
    if moderate_extra > 0:
        moderate_result = _condense(run_simulation_verbose(
            **{**run_params, "total_pulls": run_params["total_pulls"] + moderate_extra}, trials=ADVISOR_TRIALS,
        ))
        runs.append(moderate_result)
        tiers.append({
            "key": "moderate", "label": "Spend If You Really Want It",
            "total_pulls": run_params["total_pulls"] + moderate_extra,
            "success_rate": moderate_result["success_rate"], "extra_pulls": moderate_extra,
        })

    if guaranteed_extra > 0:
        guaranteed_result = _condense(run_simulation_verbose(
            **{**run_params, "total_pulls": guaranteed_pulls}, trials=ADVISOR_TRIALS,
        ))
        runs.append(guaranteed_result)
        tiers.append({
            "key": "whale", "label": "Guaranteed (Big Spender)",
            "total_pulls": guaranteed_pulls,
            "success_rate": guaranteed_result["success_rate"], "extra_pulls": guaranteed_extra,
        })
    # guaranteed_extra <= 0 means the current F2P budget already meets or
    # exceeds the pull count that guarantees the goal outright: nothing
    # further to show, the F2P tier above already IS the guaranteed one.

    comparisons = _tier_pull_comparisons(tiers, run_params["total_pulls"])
    return tiers, runs, comparisons


def _tier_pull_comparisons(tiers, current_budget):
    """Every pairwise pull-count gap between the tiers actually present
    (F2P vs moderate, F2P vs whale, moderate vs whale, whichever of those
    exist), each expressed both as a raw pull count and as a percent of
    the CURRENT budget specifically (never a tier's own total), so the
    model has one consistent, stable reference frame throughout instead of
    a percentage that means something different in each comparison. Pure
    arithmetic on numbers _spending_tiers already computed, nothing here
    is a further simulation."""
    order = ["f2p", "moderate", "whale"]
    present = [t for k in order for t in tiers if t["key"] == k]
    comparisons = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            delta = b["total_pulls"] - a["total_pulls"]
            pct = round(delta / current_budget * 100, 1) if current_budget else 0
            comparisons.append({
                "from_label": a["label"], "to_label": b["label"],
                "delta_pulls": delta, "pct_of_current_budget": pct,
            })
    return comparisons


def _pill_text(pill):
    """One pill's value as text, recursing into a group pill's own pills
    and wrapping them in parentheses the way the UI renders them."""
    if pill.get("kind") == "group":
        return "(" + " ".join(_pill_text(p) for p in pill["pills"]) + ")"
    return str(pill["value"])


def _lines_to_text(lines):
    """Flatten the structured pill lines into plain text for the
    interpretation model's grounding context. The pills themselves (with
    colors and tooltips) are what the UI renders; this is only so the model
    has something readable to cite from."""
    parts = []
    for line in lines:
        values = " ".join(_pill_text(p) for p in line["pills"])
        parts.append(f"{line['label']}: {values}")
    return "; ".join(parts)


def _validate_strategy(strategy):
    if not isinstance(strategy, list) or not strategy:
        return "strategy must be a non-empty list of phases"
    for phase in strategy:
        if phase.get("banner") not in ("char", "weapon"):
            return "each phase banner must be 'char' or 'weapon'"
        copies = phase.get("copies")
        if not isinstance(copies, int) or copies < 1:
            return "each phase copies must be an integer >= 1"
    return None


def _run_tool(args, baseline_params, lock_start_state=False):
    """Execute the run_simulation tool: merge the model's args over the baseline,
    validate, run, and return a condensed result (or an error the model can read).

    lock_start_state=True is used once a session's real pity/guarantee state
    has been verified by reconciliation: the model may still explore a
    different total_pulls, but cannot override the actual pity, guarantee,
    or remaining copy counts it was just given, no matter what it passes.
    Copy counts are locked alongside pity/guarantee because the "restated
    goal" the model sees describes the whole session's goal (obtained plus
    remaining), and a model exploring further was found live re-deriving a
    copy count from that total rather than the already-reconciled remaining
    amount, silently re-simulating a goal from scratch instead of the real
    remaining one. Prompt-only instructions to the same effect were
    repeatedly not followed reliably; this makes it structurally impossible
    instead of asking nicely."""
    if lock_start_state:
        strategy = baseline_params["strategy"]
    else:
        strategy = args.get("strategy") or baseline_params["strategy"]
    error = _validate_strategy(strategy)
    if error:
        return {"error": error}

    total_pulls = args.get("total_pulls", baseline_params["total_pulls"])
    if not isinstance(total_pulls, int) or total_pulls < 1:
        return {"error": "total_pulls must be an integer >= 1"}

    if lock_start_state:
        start_char_pity = baseline_params["start_char_pity"]
        start_char_guarantee = baseline_params["start_char_guarantee"]
        start_weapon_pity = baseline_params["start_weapon_pity"]
        start_weapon_guarantee = baseline_params["start_weapon_guarantee"]
    else:
        start_char_pity = args.get("start_char_pity", baseline_params["start_char_pity"])
        start_char_guarantee = args.get("start_char_guarantee", baseline_params["start_char_guarantee"])
        start_weapon_pity = args.get("start_weapon_pity", baseline_params["start_weapon_pity"])
        start_weapon_guarantee = args.get("start_weapon_guarantee", baseline_params["start_weapon_guarantee"])

    stats = run_simulation_verbose(
        total_pulls=total_pulls,
        strategy=[{"banner": p["banner"], "copies": p["copies"]} for p in strategy],
        start_char_pity=start_char_pity,
        start_char_guarantee=start_char_guarantee,
        start_weapon_pity=start_weapon_pity,
        start_weapon_guarantee=start_weapon_guarantee,
        full_4star_chars=baseline_params["full_4star_chars"],
        char_pity_config=baseline_params["char_pity_config"],
        weapon_pity_config=baseline_params["weapon_pity_config"],
        trials=ADVISOR_TRIALS,
    )
    return _condense(stats)


def _extract_session_state(client, model, question, baseline_stats, error_feedback=None):
    """Ask the model to pull structured facts out of the question: no math,
    no strategy decisions, just what was explicitly stated. session_state.py
    does all the arithmetic and validation on the result."""
    goal_label, goal_description = describe_goal(baseline_stats)
    user_content = (
        f"Baseline goal: {goal_description} (goal {goal_label}), "
        f"{baseline_stats['initial_pulls']} total pulls.\n\nQuestion: {question}"
    )
    if error_feedback:
        user_content += (
            f"\n\nYour previous extraction did not reconcile: {error_feedback}. "
            "Re-read the question and correct the fields."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "session_state", "strict": True, "schema": SESSION_STATE_SCHEMA},
        },
        temperature=ADVISOR_TEMPERATURE,
    )
    try:
        return json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {"has_event_sequence": False}


def _assistant_message(msg):
    """Serialize an assistant tool-call message for the next request."""
    return {
        "role": "assistant",
        "content": msg.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ],
    }


SYSTEM_PROMPT = (
    "You are a blunt, no-fluff gacha pull advisor. The user has a baseline "
    "simulation result and is asking an open-ended follow-up. Use the run_simulation "
    "tool to actually test any what-if instead of guessing, then compare the result "
    "to the baseline and answer in short, direct sentences, never just a number followed "
    "by an offer to explore more, that tells the reader nothing they didn't already know. "
    "When given F2P / spend-if-you-really-want-it / guaranteed spending tiers, address each "
    "one given to you by name in a sentence or so and say plainly which one you'd actually "
    "recommend for someone in this spot and why, don't just list three numbers and stop. When "
    "also given the pull-count gaps between tiers (as a percent of the current budget), use them "
    "to comment on relative effort, a small percentage step is a minor top-up worth considering "
    "lightly, a step that doubles or more than doubles the budget is a real commitment and should "
    "be described as one, don't present every step as equally easy just because a number exists "
    "for it. Otherwise (a plain what-if with no tiers given) 2 to 4 short sentences is still right. "
    "Always cite the specific success rates you got from the tool (for example, at 220 pulls "
    "it is 71 percent) so your answer is grounded in the numbers. Be honest: if a tier barely "
    "helps or the odds are poor even fully spent, say so, and do not push the user to spend "
    "more than they need to; if F2P is already comfortable, say that plainly and don't manufacture "
    "urgency to spend anyway. Respect the stated goal and starting conditions: do not assume the user "
    "wants a character or weapon copy they did not include. If the goal is already a "
    "single copy, there is nothing to trim, so focus on more pulls or waiting. Pity "
    "for a banner resets to 0 the instant a 5-star of that banner's type is obtained, "
    "whether it was the desired item or a losing 50/50 result, and character and "
    "weapon pity are independent of each other. Never assume a previous pity count "
    "carries forward into a new phase after a 5-star was already obtained on that "
    "banner, and never present that as an open question or a second, higher-odds "
    "scenario: it is not a matter of interpretation. If the context states part of "
    "the original goal is already obtained, treat only that part as done and describe "
    "success or failure solely in terms of what still remains, never the original full "
    "goal. Never state a pull count, a "
    "success rate, or any other simulation-derived number in your answer unless it "
    "came directly from a run_simulation result you actually received in this "
    "conversation; if a further scenario is worth mentioning, call the tool for it "
    "first, do not estimate or reason your way to a plausible-sounding figure. "
    "No markdown, no headers, no bullet points, no em-dashes."
)


def _scenarios_from_extraction(extracted):
    """Flatten one extraction result into a list of scenario dicts, each
    shaped exactly like reconcile()'s expected input (has_event_sequence
    plus the per-scenario fields), with its own condition_label. The
    primary scenario (the extraction's own top-level fields) is always
    first; any additional_scenarios follow, capped at MAX_SCENARIOS total
    as a guard against a runaway misread. Most questions produce exactly
    one scenario here, in which case nothing downstream behaves any
    differently than before this existed."""
    primary = {"has_event_sequence": True, "condition_label": extracted.get("condition_label")}
    for key in _SCENARIO_FIELDS:
        primary[key] = extracted.get(key)
    scenarios = [primary]
    for extra in (extracted.get("additional_scenarios") or [])[:MAX_SCENARIOS - 1]:
        scenarios.append({"has_event_sequence": True, **extra})
    return scenarios


def _reconcile_scenario_list(extracted, baseline_params, baseline_stats):
    """Reconcile every scenario in one extraction result against the SAME
    baseline. Stops at the first failure rather than partially reconciling
    the rest, so a retry re-extracts the whole set fresh instead of trying
    to patch just one branch in isolation.

    A pity conflict (a banner's reported pulls don't fit its actual
    starting pity) is only surfaced as its own rich, clarifiable result
    when the question is a single scenario; a conflict inside one branch
    of a multi-scenario question falls back to the ordinary decline path
    instead; resolving one specific branch's ambiguity through the same
    human-clarification UI as the single-scenario case is not supported."""
    scenarios = _scenarios_from_extraction(extracted)
    reconciled_scenarios = []
    for scenario in scenarios:
        reconciled = reconcile(scenario, baseline_params, baseline_stats)
        if not reconciled["ok"]:
            if reconciled.get("conflict") and len(scenarios) == 1:
                return {"ok": False, "conflict": True,
                        "lines": reconciled["lines"], "conflicts": reconciled["conflicts"]}
            label = scenario.get("condition_label")
            prefix = f"{label}: " if label else ""
            error = reconciled.get("error") or "one or more reported pull counts don't fit the starting pity"
            return {"ok": False, "error": f"{prefix}{error}"}
        reconciled["condition_label"] = scenario.get("condition_label")
        reconciled_scenarios.append(reconciled)
    return {"ok": True, "scenarios": reconciled_scenarios}


_RESOLUTION_LABELS = {
    "total_includes_existing_pity": "USER CLARIFICATION: ",
    "starting_situation": "USER MODIFICATION TO INITIAL SITUATION: ",
    "prompt": "USER MODIFICATION TO PROMPT: ",
}


def _build_clarification_feedback(conflicts, clarifications):
    """Turn the user's typed answers to one or more conflict clarifying
    questions into error_feedback text for a re-extraction, matched to the
    freshly-rediscovered conflicts (not whatever the frontend echoes back)
    by banner + attempt_number, the same pairing the conflict blocks were
    built with in the first place."""
    answers = {(c["banner"], c["attempt_number"]): c["answer"] for c in clarifications if c.get("answer")}
    parts = []
    for conflict in conflicts:
        answer = answers.get((conflict["banner"], conflict["attempt_number"]))
        if not answer:
            continue
        parts.append(
            f'For the {conflict["banner"]} banner, attempt {conflict["attempt_number"]} of '
            f'{conflict["total_attempts"]} (reported as {conflict["reported_pulls"]} pulls spent), '
            f'you asked: "{conflict["question"]}" The user answered: "{answer}"'
        )
    return " ".join(parts)


def _classify_resolution(conflict, extracted, fallback_answer):
    """After a retry extraction meant to resolve one conflict, work out
    which of the three ways the user resolved it, by comparing the retry's
    output against what originally conflicted, so the right hardcoded
    label ends up on the annotation: never let the model choose its own
    label text for this, the exact prefix matters and prompt-only
    instructions to relay specific wording have repeatedly not been
    followed reliably elsewhere in this module.

    Returns (kind, description_text) or None if nothing about this
    conflict's banner/event actually changed (the retry didn't apply it)."""
    banner_label = conflict["banner"].capitalize()

    for correction in extracted.get("starting_situation_corrections") or []:
        if correction.get("banner") != conflict["banner"]:
            continue
        bits = []
        if correction.get("corrected_pity") is not None:
            bits.append(f"pity {correction['corrected_pity']}")
        if correction.get("corrected_guarantee") is not None:
            bits.append(f"guarantee {'true' if correction['corrected_guarantee'] else 'false'}")
        if bits:
            return ("starting_situation", f"{banner_label} banner starting {' and '.join(bits)}.")

    events = [e for e in (extracted.get("events") or []) if e.get("banner_type") == conflict["banner"]]
    if len(events) >= conflict["attempt_number"]:
        event = events[conflict["attempt_number"] - 1]
        if event.get("pity_is_absolute") and event.get("pity_at_outcome") == conflict["reported_pulls"]:
            return ("total_includes_existing_pity",
                    f"{conflict['reported_pulls']} total pulls were used, including existing pity.")
        if event.get("pity_at_outcome") != conflict["reported_pulls"]:
            return ("prompt", (
                f"{banner_label} banner attempt {conflict['attempt_number']} of "
                f"{conflict['total_attempts']}: {fallback_answer}"
            ))

    return None


def _apply_starting_situation_corrections(baseline_params, corrections):
    """A 'starting situation' resolution overwrites a banner's starting pity
    or guarantee for THIS reconciliation only, never the caller's own
    baseline_params dict: the user's correction applies to the run here, not
    to the overall Monte Carlo simulation and visualization above, which
    keeps whatever was actually entered in the form. Returns a new dict,
    baseline_params itself is never mutated."""
    if not corrections:
        return baseline_params
    adjusted = dict(baseline_params)
    for correction in corrections:
        prefix = {"character": "char", "weapon": "weapon"}.get(correction.get("banner"))
        if not prefix:
            continue
        if correction.get("corrected_pity") is not None:
            adjusted[f"start_{prefix}_pity"] = correction["corrected_pity"]
        if correction.get("corrected_guarantee") is not None:
            adjusted[f"start_{prefix}_guarantee"] = correction["corrected_guarantee"]
    return adjusted


def _reconcile_all_scenarios(client, model, question, baseline_params, baseline_stats, clarifications=None):
    """Extract every scenario the question describes (usually just one) and
    reconcile each deterministically against the same baseline.

    A genuine pity conflict (see reconcile()) is never auto-retried, an
    automated second guess can't resolve an ambiguity only the user can
    answer, it's returned immediately so the caller can surface it. If
    `clarifications` (answers to a previous round of conflict questions)
    are supplied, they're folded into ONE re-extraction attempt first, and
    `annotations` on the result names how each was resolved for the UI's
    labeled context line. Any other reconciliation failure still gets the
    existing one-shot auto-retry, since that failure mode is usually just
    the model misreading the question, which a second attempt can fix on
    its own.

    Returns applies=True, ok=False rather than silently discarding the
    narrative when nothing resolves it, so the caller can tell the user
    reconciliation failed instead of quietly answering from the unadjusted
    baseline as if nothing had been stated."""
    extracted = _extract_session_state(client, model, question, baseline_stats)
    if not extracted.get("has_event_sequence"):
        return {"applies": False}

    result = _reconcile_scenario_list(extracted, baseline_params, baseline_stats)
    if result["ok"]:
        return {"applies": True, **result}

    if result.get("conflict"):
        if clarifications:
            feedback = _build_clarification_feedback(result["conflicts"], clarifications)
            if feedback:
                retry_extracted = _extract_session_state(
                    client, model, question, baseline_stats, error_feedback=feedback,
                )
                if retry_extracted.get("has_event_sequence"):
                    retry_baseline_params = _apply_starting_situation_corrections(
                        baseline_params, retry_extracted.get("starting_situation_corrections"),
                    )
                    retried = _reconcile_scenario_list(retry_extracted, retry_baseline_params, baseline_stats)
                    if retried["ok"] or retried.get("conflict"):
                        annotations = []
                        for conflict in result["conflicts"]:
                            answer = next(
                                (c["answer"] for c in clarifications
                                 if c["banner"] == conflict["banner"]
                                 and c["attempt_number"] == conflict["attempt_number"]),
                                None,
                            )
                            if not answer:
                                continue
                            classified = _classify_resolution(conflict, retry_extracted, answer)
                            if classified:
                                kind, text = classified
                                annotations.append({"label": _RESOLUTION_LABELS[kind], "text": text})
                        return {"applies": True, "annotations": annotations, **retried}
        # No clarifications yet, or applying them didn't produce anything
        # usable: surface the conflict as-is rather than silently decline.
        return {"applies": True, **result}

    first_error = result["error"]
    extracted = _extract_session_state(
        client, model, question, baseline_stats, error_feedback=first_error,
    )
    if not extracted.get("has_event_sequence"):
        return {"applies": True, "ok": False, "error": first_error}

    result = _reconcile_scenario_list(extracted, baseline_params, baseline_stats)
    if result["ok"] or result.get("conflict"):
        return {"applies": True, **result}
    return {"applies": True, "ok": False, "error": result["error"]}


def _answer_branching_scenarios(client, model, question, baseline_params, scenarios, goal_label, goal_description):
    """Answer a question that reconciled into two or more mutually exclusive
    scenarios (e.g. 'if I win the first pull, ...; if I lose, ...'). Each
    scenario is simulated deterministically here, exactly like the single-
    scenario path never trusts the model with the primary number, and the
    model gets one turn to write a single answer addressing every branch
    together. No exploratory tool calls are offered here: a further what-if
    on top of one specific branch is ambiguous with more than one verified
    starting state already in play, so that stays scoped to the ordinary
    single-scenario path.

    Returns (answer_text, runs, breakdown) where breakdown is
    {"status": "ok", "groups": [{"label": ..., "lines": [...]}, ...]}, one
    group per scenario, distinct from the single-scenario "lines" shape so
    the UI can render each branch as its own visually separate section."""
    groups = []
    runs = []
    scenario_summaries = []

    for i, scenario in enumerate(scenarios):
        label = scenario.get("condition_label") or f"Scenario {i + 1}"
        lines_text = _lines_to_text(scenario["lines"])
        display_lines = scenario["lines"]

        if scenario["goal_complete"] or scenario["pulls_exhausted"]:
            status = "the goal is already complete" if scenario["goal_complete"] else "no pulls remain"
            scenario_summaries.append(
                f'"{label}": Verified state: {lines_text} ({status}; nothing to simulate for this scenario).'
            )
        else:
            run_params = {
                **baseline_params,
                "strategy": scenario["strategy"],
                "total_pulls": scenario["total_pulls"],
                "start_char_pity": scenario["start_char_pity"],
                "start_char_guarantee": scenario["start_char_guarantee"],
                "start_weapon_pity": scenario["start_weapon_pity"],
                "start_weapon_guarantee": scenario["start_weapon_guarantee"],
            }
            # Run each branch's reconciled scenario ourselves, right now,
            # same reasoning as the single-scenario path: a result that
            # already exists before the model's first turn can't be gotten
            # wrong, or attributed to the wrong branch.
            result = _condense(run_simulation_verbose(**run_params, trials=ADVISOR_TRIALS))
            runs.append(result)
            display_lines = display_lines + [build_result_line(run_params["total_pulls"], result["success_rate"])]
            remaining = remaining_goal_text(scenario["remaining_characters"], scenario["remaining_weapons"])
            scenario_summaries.append(
                f'"{label}": Verified state: {lines_text} What remains to obtain: {remaining}. A '
                f"simulation on this exact state has ALREADY been run: {run_params['total_pulls']} pulls, "
                f"success rate {result['success_rate']}, average leftover pulls on success "
                f"{result['avg_leftover_pulls_on_success']}, most common failure state "
                f"{result['most_common_failure_state']}."
            )

        groups.append({"label": label, "lines": display_lines})

    context = (
        f"Original full goal was {goal_description} (goal {goal_label}). The question describes "
        f"{len(scenarios)} mutually exclusive scenarios branching from that same starting point; "
        f"every scenario below is already verified and, where applicable, already simulated, treat "
        f"all of it as fact and do not restate it differently. Address every scenario in your answer, "
        f"describing each one only in terms of what remains for it, never the original full goal, and "
        f"cite only the exact numbers given for each scenario below; never state a number for a "
        f"scenario that was not given here. No further simulation is available for this question, do "
        f"not offer to run one.\n\n" + "\n\n".join(scenario_summaries)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nFollow-up question: {question}"},
    ]
    # No tools= param here on purpose: the API rejects tool_choice="none" when
    # tools isn't also passed, and omitting tools entirely already prevents
    # any tool call just as well, there's nothing for the model to invoke.
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=ADVISOR_TEMPERATURE,
    )
    breakdown = {"status": "ok", "groups": groups}
    return (response.choices[0].message.content or "").strip(), runs, breakdown


def run_advisor(baseline_params, baseline_stats, question, *, model=None, max_tool_calls=MAX_TOOL_CALLS,
                 clarifications=None):
    """Run the agentic follow-up loop.

    `clarifications`, when given, is the user's answers to a previous
    round of conflict clarifying questions: a list of
    {"banner", "attempt_number", "answer"} (the first two identify which
    conflict block the answer is for, exactly as returned in that block).

    Returns (answer_text, runs, breakdown) where runs is the list of condensed
    simulation results actually run (for the UI receipts), and breakdown is
    None (a pure hypothetical, no event sequence to show), {"status": "ok",
    "lines": [...], "annotations": [...]} (the structured, deterministically-
    built pill data for a single reconciled scenario; annotations is only
    present when clarifications resolved a prior conflict, one
    {"label", "text"} per conflict answered, for the UI's labeled context
    line), {"status": "ok", "groups": [{"label": ..., "lines": [...]}, ...]}
    (the question reconciled into two or more mutually exclusive scenarios,
    one group per branch, see _answer_branching_scenarios), {"status":
    "conflict", "lines": [...], "conflicts": [...]} (one or more banners'
    reported pulls don't fit their actual starting pity, a genuine
    ambiguity for the user to resolve, not a plain parse failure; answer_text
    is empty, there is nothing to say until it's resolved), or {"status":
    "error", "message": "..."} (an event sequence was described but could
    not be reconciled; see PARSE_FAILURE_MESSAGE for the answer text in
    that case)."""
    client = OpenAI(api_key=get_openai_api_key())
    model = model or get_model()

    goal_label, goal_description = describe_goal(baseline_stats)
    context = (
        f"Baseline scenario: {goal_description} (goal {goal_label}). "
        f"Total pulls: {baseline_stats['initial_pulls']}. "
        f"Character pity: {baseline_stats['start_char_pity']}, weapon pity: {baseline_stats['start_weapon_pity']}. "
        f"Baseline success rate: {baseline_stats['success_rate']}."
    )
    if baseline_stats["desired_characters"] + baseline_stats["desired_weapons"] <= 1:
        context += " This goal is already a single copy, so there is nothing to trim."

    run_params = baseline_params
    breakdown = None
    runs = []
    agent_cycle = 0
    # Locked once a session's real pity/guarantee has been reconciled and
    # verified: exploratory calls beyond that point may still vary
    # total_pulls or copies, but cannot override the actual starting state,
    # regardless of what the model passes. See _run_tool.
    lock_start_state = False

    reconciled = _reconcile_all_scenarios(
        client, model, question, baseline_params, baseline_stats, clarifications=clarifications,
    )
    if reconciled.get("applies") and reconciled.get("conflict"):
        # A genuine "does the reported pull count fit the starting pity"
        # ambiguity, not a plain decline: no prose answer to give, the
        # conflict blocks themselves ARE the answer, waiting on the user.
        breakdown = {"status": "conflict", "lines": reconciled["lines"], "conflicts": reconciled["conflicts"]}
        return "", [], breakdown
    if reconciled.get("applies") and reconciled.get("ok") and len(reconciled["scenarios"]) > 1:
        return _answer_branching_scenarios(
            client, model, question, baseline_params, reconciled["scenarios"],
            goal_label, goal_description,
        )
    if reconciled.get("applies") and reconciled.get("ok"):
        scenario = reconciled["scenarios"][0]
        lines = scenario["lines"]
        lines_text = _lines_to_text(lines)

        if scenario["goal_complete"] or scenario["pulls_exhausted"]:
            # Nothing left to simulate, so answer directly from the verified state.
            breakdown = {"status": "ok", "lines": lines}
            if reconciled.get("annotations"):
                breakdown["annotations"] = reconciled["annotations"]
            remaining = remaining_goal_text(scenario["remaining_characters"], scenario["remaining_weapons"])
            context = (
                f"Verified session state (already reconciled from the question, treat as "
                f"fact and do not restate it differently): {lines_text} "
                f"Original full goal was {goal_description} (goal {goal_label}), but what "
                f"actually still remains to obtain, after the events above, is: {remaining}. "
                f"Describe the outcome only in terms of what remains; anything not listed "
                f"there is already obtained and must not be described as still needed, at "
                f"risk, or a possible failure."
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\nFollow-up question: {question}"},
            ]
            # No tools= param here on purpose: the API rejects tool_choice="none" when
            # tools isn't also passed, and omitting tools entirely already prevents
            # any tool call just as well, there's nothing for the model to invoke.
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=ADVISOR_TEMPERATURE,
            )
            return (response.choices[0].message.content or "").strip(), [], breakdown

        run_params = {
            **baseline_params,
            "strategy": scenario["strategy"],
            "total_pulls": scenario["total_pulls"],
            "start_char_pity": scenario["start_char_pity"],
            "start_char_guarantee": scenario["start_char_guarantee"],
            "start_weapon_pity": scenario["start_weapon_pity"],
            "start_weapon_guarantee": scenario["start_weapon_guarantee"],
        }
        lock_start_state = True

        # Run the reconciled scenario ourselves, right now, rather than
        # trusting the model to call the tool with the correct parameters.
        # Prompt-only instructions repeatedly failed to stop it re-deriving
        # (and double-counting) the pull count itself; a result that already
        # exists before the model's first turn can't be gotten wrong.
        primary_result = _condense(run_simulation_verbose(**run_params, trials=ADVISOR_TRIALS))
        runs.append(primary_result)
        lines = lines + [build_result_line(run_params["total_pulls"], primary_result["success_rate"])]

        # Three spending tiers, pre-computed here rather than left to the
        # model: F2P (the primary result above, current budget as-is), a
        # comfortably-safe top-up (MODERATE_SPEND_TARGET), and the pull
        # count that mathematically guarantees the remaining goal via hard
        # pity. Every figure the model gets to cite already came from a
        # real run or closed-form arithmetic, never a guess.
        tiers, tier_runs, tier_comparisons = _spending_tiers(scenario, run_params, baseline_params, primary_result)
        runs.extend(tier_runs)
        for tier in tiers[1:]:
            lines = lines + [build_spending_tier_line(tier["label"], tier["total_pulls"], tier["success_rate"])]

        breakdown = {"status": "ok", "lines": lines}
        if reconciled.get("annotations"):
            breakdown["annotations"] = reconciled["annotations"]
        remaining = remaining_goal_text(scenario["remaining_characters"], scenario["remaining_weapons"])

        tier_facts = [
            f"F2P (spending nothing further, your current budget as reported): "
            f"{run_params['total_pulls']} pulls, success rate {primary_result['success_rate']}, "
            f"average leftover pulls on success {primary_result['avg_leftover_pulls_on_success']}, "
            f"most common failure state {primary_result['most_common_failure_state']}."
        ]
        # Live-testing found the model sometimes re-subtracting a number
        # from the question itself (e.g. "lost after 10 pulls") from the
        # F2P total, producing a lower, wrong pulls-remaining figure, even
        # though that number is already folded into the total above. Naming
        # the specific number(s) in the question, not just a generic "don't
        # recompute" instruction, is what actually stopped this in testing.
        question_numbers = sorted(set(re.findall(r"\d+", question)))
        if question_numbers:
            tier_facts.append(
                f"Numbers mentioned in your own question ({', '.join(question_numbers)}) are already "
                f"fully incorporated into the {run_params['total_pulls']} figure above; do not add or "
                f"subtract any of them from {run_params['total_pulls']} again to get a different "
                f"pulls-remaining figure, {run_params['total_pulls']} is already final."
            )
        moderate = next((t for t in tiers if t["key"] == "moderate"), None)
        whale = next((t for t in tiers if t["key"] == "whale"), None)
        if moderate:
            tier_facts.append(
                f"Spend if you really want it (worth it if this character or weapon matters to you or "
                f"your build, not required): {moderate['total_pulls']} total pulls (an extra "
                f"{moderate['extra_pulls']} beyond your current budget), success rate "
                f"{moderate['success_rate']}, comfortably likely without paying for a full guarantee."
            )
        if whale:
            tier_facts.append(
                f"Guaranteed, for a big spender or content creator who wants zero risk: "
                f"{whale['total_pulls']} total pulls (an extra {whale['extra_pulls']} beyond your "
                f"current budget), success rate {whale['success_rate']}. This is not a probability, "
                f"hard pity mathematically forces this outcome regardless of luck at that many pulls."
            )
        if not moderate and not whale:
            tier_facts.append(
                "Your current budget already meets or exceeds the pull count that guarantees this "
                "goal outright via hard pity, regardless of luck: there is nothing further worth "
                "spending on, this is as safe as it gets."
            )

        if tier_comparisons:
            comparison_facts = [
                f"{c['from_label']} to {c['to_label']}: {c['delta_pulls']} more pulls "
                f"({c['pct_of_current_budget']}% on top of your current {run_params['total_pulls']}-pull budget)."
                for c in tier_comparisons
            ]
            tier_facts.append(
                "Exact pull-count gaps between the tiers above, each also given as a percent of your "
                "current budget so you have one consistent yardstick for how big a step each one "
                "actually is: " + " ".join(comparison_facts) + " Use these to comment on relative "
                "effort, for example a gap under roughly 25 percent of the current budget is a small "
                "top-up, one over 100 percent roughly doubles or more than doubles it, that's a much "
                "bigger ask, say so plainly rather than treating every step as equally reasonable."
            )

        context = (
            f"Verified session state (already reconciled from the question, treat as fact "
            f"and do not restate it differently): {lines_text} "
            f"Original full goal was {goal_description} (goal {goal_label}), but what actually "
            f"still remains to obtain, after the events above, is: {remaining}. The figures "
            f"below were run for exactly that remaining goal, not the original one; describe "
            f"success and failure only in terms of what remains, anything not listed there is "
            f"already obtained and must not be described as still needed, at risk, or a "
            f"possible failure. "
            f"The following spending tiers have ALREADY been run or computed for you, address "
            f"the ones given below by name (F2P, spend if you really want it, guaranteed), each "
            f"in a sentence or so, using only these exact figures; never invent a tier, a pull "
            f"count, or a success rate that isn't listed here. Skip a tier only if it is genuinely "
            f"identical to one you already covered. " + " ".join(tier_facts) + " "
            f"Never state a total_pulls figure or a success rate in your answer that you did not "
            f"get from the figures above or an actual run_simulation call; if you want to explore "
            f"a further, different scenario the question specifically asks about, call the tool "
            f"for it first, deriving any different total_pulls only by adding to or subtracting "
            f"from the {run_params['total_pulls']} figure above, never by recomputing it from the "
            f"raw figures in the question yourself. Any start_char_pity, start_char_guarantee, "
            f"start_weapon_pity, start_weapon_guarantee, or strategy (copy counts) you pass on such "
            f"a call is ignored: the verified state and the remaining goal above ({remaining}) are "
            f"used regardless, so do not bother varying those. '{remaining}' already accounts for "
            f"everything obtained above; it is not a total to re-simulate from scratch."
        )
    elif reconciled.get("applies"):
        # The question described an event sequence, but it could not be
        # reliably reconciled into a consistent pull count or goal even
        # after a corrective retry. Decline entirely rather than guessing:
        # a deterministic, hardcoded message, not another model call, since
        # this is exactly the kind of instruction a model has repeatedly
        # failed to follow reliably elsewhere in this app.
        breakdown = {"status": "error", "message": "Could not parse a clear sequence of pull events from this question."}
        return PARSE_FAILURE_MESSAGE, [], breakdown

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nFollow-up question: {question}"},
    ]

    for _ in range(max_tool_calls):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=[RUN_SIMULATION_TOOL], tool_choice="auto",
            temperature=ADVISOR_TEMPERATURE,
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            return (msg.content or "").strip(), runs, breakdown

        messages.append(_assistant_message(msg))
        for tool_call in msg.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _run_tool(args, run_params, lock_start_state=lock_start_state)
            if "error" not in result:
                runs.append(result)
                # Every tool call reaching this loop is, by definition, the
                # model exploring beyond the guaranteed pre-run (which never
                # goes through here), surface it as its own labeled pill so
                # any such exploration is explicit and tracked, not just a
                # claim in prose.
                agent_cycle += 1
                cycle_line = build_agent_cycle_line(agent_cycle, result["total_pulls"], result["success_rate"])
                if breakdown is None:
                    breakdown = {"status": "ok", "lines": []}
                breakdown["lines"].append(cycle_line)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    # Tool-call budget exhausted: force a final text answer with what we have.
    # No tools= param here on purpose: the API rejects tool_choice="none" when
    # tools isn't also passed, and omitting tools entirely already prevents
    # any further tool call just as well.
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=ADVISOR_TEMPERATURE,
    )
    return (response.choices[0].message.content or "").strip(), runs, breakdown

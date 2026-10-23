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
    MIN_ROUTE_SAMPLE, active_plan, build_agent_cycle_line, build_omitted_routes_group, build_result_line,
    build_route_groups, build_stop_line, route_step_name, step_limit_note,
    build_spending_tier_line, reconcile, remaining_goal_text,
)
from simulation import run_simulation_verbose, simulate_plan

# Fewer trials than the main endpoint: the advisor may run several sims per
# question, and it only needs directional numbers, not publication precision.
ADVISOR_TRIALS = 4000
MAX_TOOL_CALLS = 4
# At most this many branches for a branching question (lettered A-D),
# a runaway guard against a question the model misreads as branching far
# more than it actually does; extras beyond this are dropped, not rejected.
MAX_SCENARIOS = 4

# The "spend if you really want it" tier's target: comfortably likely
# without paying for a full guarantee. Below this, the F2P figure already
# covers it and there's nothing to upsell; above it just starts eating into
# the guaranteed tier's territory for diminishing return.
# Above this, the current budget is already as good as it gets in
# practice; showing a further "guaranteed" pull count on top of an
# already-near-certain result would just restate the same ~100% back at a
# higher cost, not add anything worth knowing.
NEAR_CERTAIN_SUCCESS_RATE = 0.995
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
# properties) and, field by field, every entry of branches below: one scenario is
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
        "description": "A direct, total restatement of pulls remaining, ONLY when the question's own words state a specific leftover pull count as fact (e.g. 'I have 41 pulls left', 'I'm down to 23'). Never compute this yourself by subtracting a pity count (or anything else) from the total pulls; a question that only gives an event's pity and outcome, with no separate remaining-pulls statement, must leave this null, the true remaining total is computed deterministically from the events instead. Do not use this field for an amount meant to be added on top of the remaining pulls; use pull_additions for that instead.",
    },
    "pull_additions": {
        "type": ["array", "null"],
        "description": (
            "Pulls stated as an ADDITION on top of whatever remains, not a total, ONE ENTRY PER SOURCE "
            "mentioned (e.g. 'about 60 from this patch and I'll top up 40 more' is two entries). Report "
            "each raw amount exactly as stated; never add them together or convert currency yourself, "
            "code does both. Null if none."
        ),
        "items": {
            "type": "object",
            "properties": {
                "amount": {"type": "integer", "description": "The number exactly as stated."},
                "unit": {
                    "type": "string", "enum": ["pulls", "currency"],
                    "description": "'pulls' for a pull or wish count ('40 pulls', '4 ten-pulls' is 40). 'currency' for premium currency instead of pulls ('16,000 primogems', '8000 stellar jade', '3200 polychromes').",
                },
                "source": {
                    "type": "string",
                    "enum": ["savings", "patch", "event", "recurring", "purchase", "endgame", "exploration",
                             "codes", "shop", "other"],
                    "description": (
                        "Where the pulls come from: 'savings' (currency or pulls the player already has "
                        "saved up but hasn't counted yet, e.g. 'I have 16,000 primogems saved'), 'patch' (a "
                        "patch or version's rewards, new story, "
                        "quests), 'event' (limited events, login events, web events, free pulls tied to a "
                        "banner), 'recurring' (dailies, commissions, weeklies, the free battle pass track), "
                        "'purchase' (top-ups, buying currency, a monthly card or welkin, a paid battle pass), "
                        "'endgame' (endgame rewards and resets: Spiral Abyss, Memory of Chaos, Shiyu Defense "
                        "and other endgame modes), "
                        "'exploration' (exploration, chests, achievements, one-off quests), 'codes' "
                        "(livestream or redeem codes, maintenance compensation), 'shop' (starglitter, "
                        "embers or other in-game shop exchanges), 'other' (unspecified, e.g. 'I'll have "
                        "about 50 more')."
                    ),
                },
            },
            "required": ["amount", "unit", "source"],
            "additionalProperties": False,
        },
    },
    "total_pulls_restated": {"type": ["integer", "null"]},
    "reserve_pulls": {
        "type": ["integer", "null"],
        "description": "Pulls the question says to keep aside and never spend, only if stated, e.g. 'I want to keep 40 pulls for the next banner' = 40. Report the raw number only; do not subtract it from anything yourself. Null otherwise.",
    },
    "additional_character_copies_wanted": {
        "type": ["integer", "null"],
        "description": "A signed CHANGE in character copies wanted, relative to the original goal, not a new total. Positive for extra copies wanted BEYOND the original goal (e.g. 'another character copy' = 1). Negative to reduce the goal below its original count, for example the question decides to give up on further copies of that banner after a bad outcome (e.g. 'if I lose the 50/50, I'll skip the second character copy and only go for the weapon' = -1 for that scenario; giving up on the banner entirely = negative enough to cover every copy still wanted, e.g. -2 if 2 were still wanted). Do not use this for the original goal itself, only for a stated change to it.",
    },
    "additional_weapon_copies_wanted": {
        "type": ["integer", "null"],
        "description": "Same rule as additional_character_copies_wanted (a signed change, positive to add, negative to reduce below the original goal), for the weapon banner.",
    },
    "characters_wanted_total": {
        "type": ["integer", "null"],
        "description": "ONLY when the question explicitly REPLACES the goal with a stated total, for example 'what if I change my goal to just one character and no weapon', 'what if I only wanted 2 characters'. A total number of character copies wanted, not a change. Null otherwise: wording like 'if I go for C1', 'one more copy', 'also add a second copy' ADDS to the baseline goal and belongs in additional_character_copies_wanted instead, never here.",
    },
    "weapons_wanted_total": {
        "type": ["integer", "null"],
        "description": "Same rule as characters_wanted_total, for the weapon banner: only an explicit replacement of the goal with a stated total, null otherwise.",
    },
}

# One step of a conditional plan: a FUTURE pull the question intends to
# make, in order, and what happens if its 5-star is lost. Never an outcome
# (those are events); the simulation decides how each step turns out.
PLAN_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "step_order": {"type": "integer", "description": "1-based position of this step in the intended pulling order."},
        "banner_type": {"type": "string", "enum": ["character", "weapon"]},
        "copies": {"type": "integer", "description": "Copies to pull for in this step, usually 1."},
        "on_loss": {
            "type": "string", "enum": ["stop", "continue", "skip"],
            "description": "'stop' if the question says to stop pulling entirely (conserve the remaining pulls) when this step's 5-star is lost, e.g. 'if I lose, I would stop there'. 'skip' if a loss moves the plan on to a later step instead, giving up this step, e.g. 'if I lose the 50/50, I'll switch to the weapon'. 'continue' otherwise, keep pulling on this banner until the copies are obtained.",
        },
        "skip_to_banner": {
            "type": ["string", "null"], "enum": ["character", "weapon", None],
            "description": "Only when on_loss is 'skip': the banner a loss switches to, e.g. 'weapon' for 'if I lose, I'll switch to the weapon'; the plan moves on to the next step on that banner. Null means simply the next step. Null when on_loss is not 'skip'.",
        },
        "max_pulls": {
            "type": ["integer", "null"],
            "description": "A limit on pulls spent in THIS step, only if stated, e.g. 'at most 80 pulls on the character' = 80. Null otherwise.",
        },
        "stop_at_pity": {
            "type": ["integer", "null"],
            "description": "A limit stated as the banner's pity counter, only if stated, e.g. 'I'll pull up to 75 pity' = 75. Report the number as stated; never convert it to a pull count yourself. Null otherwise.",
        },
        "on_cap": {
            "type": ["string", "null"], "enum": ["stop", "continue", None],
            "description": "Only when max_pulls or stop_at_pity is set: 'continue' if, on hitting the limit without the copy, the question moves on to the next step ('then switch to the weapon'), 'stop' if it ends pulling there ('then stop'). Null when there is no limit.",
        },
    },
    "required": ["step_order", "banner_type", "copies", "on_loss", "skip_to_banner", "max_pulls", "stop_at_pity",
                 "on_cap"],
    "additionalProperties": False,
}

_PLAN_FIELD = {
    "type": ["array", "null"],
    "description": (
        "ONLY when the question lays out a pulling ORDER for the future with a way for a step "
        "to end early: a stop condition on a loss, e.g. 'should I pull the weapon first, and if "
        "I win, continue on the character banner? If I lose, I would stop there', or a pull or "
        "pity limit, e.g. 'spend at most 80 on the character, then switch to the weapon'. One step per banner in "
        "the order the question intends to pull, each with the copies wanted in that step "
        "and on_loss. These are intentions, never outcomes: do not also report them as "
        "events or as branches. Null for every other question."
    ),
    "items": PLAN_STEP_SCHEMA,
}

# One mutually exclusive future the question ASSUMES ("if I win the
# character around pity 30, ...; if I lose, ..."). It continues from the
# top-level fields, which hold everything shared by every branch (what
# already happened, pulls expected regardless), so a branch only ever
# carries what's specific to it and never repeats the shared events.
BRANCH_SCHEMA = {
    "type": "object",
    "properties": {
        "condition_label": {
            "type": "string",
            "description": "Short label naming the condition that leads to this branch, e.g. 'If I win the character pull around pity 30'.",
        },
        "assumed_events": {
            "type": "array",
            "description": (
                "The hypothetical outcome(s) this branch assumes, happening AFTER every top-level "
                "event, numbered from 1 within this branch. Same shape and rules as events. Never "
                "repeat a top-level event here."
            ),
            "items": EVENT_SCHEMA,
        },
        "pulls_remaining_stated": _SCENARIO_FIELDS["pulls_remaining_stated"],
        "pull_additions": _SCENARIO_FIELDS["pull_additions"],
        "reserve_pulls": _SCENARIO_FIELDS["reserve_pulls"],
        "additional_character_copies_wanted": _SCENARIO_FIELDS["additional_character_copies_wanted"],
        "additional_weapon_copies_wanted": _SCENARIO_FIELDS["additional_weapon_copies_wanted"],
        "characters_wanted_total": _SCENARIO_FIELDS["characters_wanted_total"],
        "weapons_wanted_total": _SCENARIO_FIELDS["weapons_wanted_total"],
        "plan": _PLAN_FIELD,
        "stops_pulling": {
            "type": "boolean",
            "description": "true when this branch says the player stops pulling entirely right after its assumed outcome(s), e.g. 'if I lose the character at 75, I'll just stop there'. false otherwise.",
        },
    },
    "required": [
        "condition_label", "assumed_events", "pulls_remaining_stated", "pull_additions",
        "reserve_pulls", "additional_character_copies_wanted", "additional_weapon_copies_wanted",
        "characters_wanted_total", "weapons_wanted_total", "plan", "stops_pulling",
    ],
    "additionalProperties": False,
}

SESSION_STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": _PLAN_FIELD,
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
        "branches": {
            "type": ["array", "null"],
            "description": (
                "ONLY when the question describes two or more mutually exclusive futures hinging on "
                "an outcome it ASSUMES for a pull that has not happened yet (e.g. 'if I win the "
                "character pull around pity 30, I'll go for another copy; if I lose around pity 75, "
                "I'll go straight for the weapon'). One branch per future, every one of them "
                "including the first. The top-level fields then hold only what is shared by every "
                "branch (what already happened, pulls expected regardless, a goal change for all), "
                "and each branch only what is specific to it. Null for a question with only one "
                "path; do not invent a branch for a question that only actually has one."
            ),
            "items": BRANCH_SCHEMA,
        },
        "reply_language": {
            "type": ["string", "null"],
            "description": "The language the question itself is written in, as its English name (for example 'English', 'Japanese'). Null if unsure.",
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
        "has_event_sequence", *_SCENARIO_FIELDS.keys(), "branches",
        "starting_situation_corrections", "reply_language", "plan",
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
    "empty list. A goal change with no events at all (for example 'what if I go for C1', 'what if I "
    "also want a second weapon copy', in any language, such as 1\u51f8 / \u51f8\u3092\u8db3\u3059) is still "
    "reported through the goal fields below even though has_event_sequence is false: wording that "
    "wants MORE copies (C1 means one more copy than C0, W2 one more than W1) is a positive "
    "additional_*_copies_wanted added to the baseline goal, never a replacement of it; only wording "
    "that explicitly changes or replaces the goal to a stated total uses characters_wanted_total / "
    "weapons_wanted_total. Also report, only if explicitly stated: a direct restatement of pulls "
    "remaining (pulls_remaining_stated, required whenever any event's pity is null), "
    "amounts to add on top of whatever remains, not a total, one entry per source (pull_additions: "
    "savings not yet counted, a patch's rewards, events, dailies or the battle pass, purchases or top-ups, endgame rewards or resets, "
    "exploration or achievements, codes or compensation, shop exchanges, or unspecified; currency "
    "amounts like '16,000 primogems' are reported as stated with unit 'currency', never converted), "
    "a restated total pull budget (total_pulls_restated), "
    "and any signed change to the copies wanted for a banner "
    "(additional_character_copies_wanted / additional_weapon_copies_wanted): positive to "
    "add copies beyond the original goal, negative to reduce the goal below its original "
    "count, for example a scenario that gives up on further copies of a banner after a "
    "bad outcome ('I'll skip the second character copy and only go for the weapon' = -1). "
    "Leave a field null if the question does not state it. Most questions describe a single path: "
    "leave branches null. Only when the question describes two or more mutually exclusive FUTURE "
    "scenarios, each hinging on an outcome the question itself ASSUMES with a pity or pull number "
    "(e.g. 'if I win the character pull around pity 30, I'll go for another copy; if I lose it around "
    "pity 75, I'll go straight for the weapon instead'), do you report branches: one per future, the "
    "first one included, each with a short condition_label (e.g. 'If I win the character pull around "
    "pity 30'), its assumed outcome(s) in assumed_events, and only what is specific to it (its own goal "
    "change, added pulls, reserve, or plan). Everything shared by every branch stays in the top-level "
    "fields and is reported exactly once there, never repeated inside a branch: anything that already "
    "happened (e.g. 'I already lost the weapon 50/50 at 40 pity') is a top-level event, and pulls "
    "expected regardless of the outcome (e.g. 'I'm getting 120 pulls this patch') are a top-level "
    "pull_additions entry. Each branch continues from the top-level state; branches are "
    "alternatives to each other, never continuations of one another. A branch can hold a plan of "
    "its own, e.g. 'if I win the character at 30 pity, I'll pull the weapon and stop if I lose it', and a "
    "branch that ends pulling right after its assumed outcome ('if I lose it at 75, I'll just stop there') "
    "sets stops_pulling true instead of a goal change. "
    "Do not invent a branch for a question that only actually describes one path. A question that instead lays out "
    "a pulling ORDER with a stop condition, for example 'should I pull on the engine/weapon "
    "banner first and if I win, then continue on the character banner? If I lose, I would "
    "stop there', is not two scenarios: report it as plan, one step per banner in pulling "
    "order (weapon first with on_loss 'stop', then character with on_loss 'continue' in that "
    "example), and leave branches null. The outcome of a "
    "planned step is never known yet, the simulation decides it, so never report a planned "
    "step as an event; anything that already happened (e.g. 'I lost the 50/50 at 30 pity') "
    "is still an event as usual, and pulls expected to be gained (e.g. 'this patch will give "
    "me around 120 pulls') are still pull_additions. A step can also end at a stated "
    "limit instead: 'spend at most 80 pulls on the character, then switch to the weapon' is "
    "max_pulls 80 with on_cap 'continue' on the character step; 'I'll pull up to 75 pity, then stop' "
    "is stop_at_pity 75 with on_cap 'stop'. A loss can also move the plan on rather than stop it: 'if I "
    "win the 50/50 I'll go for C1, then the weapon; if I lose, I'll go straight to the weapon' is three "
    "steps, character (copies 1, on_loss 'skip', skip_to_banner 'weapon'), character again for the C1 copy "
    "(copies 1), then weapon, so a win goes through C1 and a loss jumps to the weapon. A win or loss "
    "with no stated pity or pull number is never an event or a branch: the simulation decides it, so it "
    "is a plan step like these. plan is null for every question without a stop, a skip, or a "
    "limit. Pulls to keep aside and never spend ('keep 40 for the next banner') are "
    "reserve_pulls, with or without a plan. A pure budget what-if with no events, such as 'what if "
    "I had 20 more pulls' (a pull_additions entry of 20) or 'what if I had 200 pulls' "
    "(total_pulls_restated 200), is still reported through those fields even though "
    "has_event_sequence is false. If (and only if) the context "
    "below includes a prior clarifying question and the user's answer to it, apply that "
    "answer to correct the specific thing it was about. For a pity clarifying question: if "
    "they confirmed a reported number was a TOTAL including existing pity, set that event's "
    "pity_is_absolute to true using the exact same number; if they gave a different number "
    "for that event, use their corrected number instead; if they said the banner's STARTING "
    "pity or guarantee itself was wrong (not the event), report that in "
    "starting_situation_corrections instead and leave the event as originally reported. For a "
    "budget clarifying question (the events reported use more pulls than the stated starting "
    "budget): if they said they gained pulls since the initial situation, or that the initial "
    "pull count itself was different, set total_pulls_restated to whatever total makes the "
    "reported events fit (the original starting total plus however many they gained, or "
    "whatever total they directly stated); if instead they said one of the reported event "
    "numbers itself was wrong, correct that event's own pity_at_outcome or refund_count "
    "instead and leave total_pulls_restated null. starting_situation_corrections is null on "
    "every other question, and total_pulls_restated is null unless a budget clarifying "
    "question's answer, or the question itself, directly restates the total."
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
    """Pre-compute up to two further pull-count options beyond the current
    budget already simulated: a modest top-up that reaches a comfortably-
    safe success rate, and the pull count that mathematically guarantees
    the remaining goal outright. Everything here is either a real
    simulation result or closed-form hard-pity arithmetic, never a model
    guess, so the interpretation model can only narrate figures that are
    actually true. These are internal reasoning material, not "tiers" to
    be labeled or listed in the answer; see SYSTEM_PROMPT for how they're
    meant to be woven in.

    Returns (tiers, runs, comparisons). tiers is a list of
    {"key", "label", "total_pulls", "success_rate", "extra_pulls"} dicts
    (the current budget always included first; the other two omitted
    whenever they wouldn't add anything: nothing further is computed at
    all once the current budget is already effectively a sure thing, and
    the top-up option is skipped on its own whenever the current budget
    already clears it). runs is every further simulation actually
    executed, for the UI receipts. comparisons is every pairwise pull-
    count gap between the options that ARE present (see
    _tier_pull_comparisons), for the model to reason about relative
    effort with, never left to compute itself."""
    tiers = [{
        "key": "f2p", "label": "your current budget",
        "total_pulls": run_params["total_pulls"], "success_rate": primary_result["success_rate"],
        "extra_pulls": 0,
    }]

    if _success_pct(primary_result) >= NEAR_CERTAIN_SUCCESS_RATE * 100:
        # Already as good as it gets: a "guaranteed" pull count would only
        # ever restate the same ~100% back at a higher cost.
        return tiers, [], []

    runs = []
    guaranteed_pulls = _guaranteed_scenario_pulls(scenario, baseline_params)
    guaranteed_extra = max(guaranteed_pulls - run_params["total_pulls"], 0)

    moderate_extra = _extra_pulls_for_target_rate(run_params, MODERATE_SPEND_TARGET, guaranteed_extra)
    if moderate_extra > 0:
        moderate_result = _condense(run_simulation_verbose(
            **{**run_params, "total_pulls": run_params["total_pulls"] + moderate_extra}, trials=ADVISOR_TRIALS,
        ))
        runs.append(moderate_result)
        tiers.append({
            "key": "moderate", "label": "a modest top-up",
            "total_pulls": run_params["total_pulls"] + moderate_extra,
            "success_rate": moderate_result["success_rate"], "extra_pulls": moderate_extra,
        })

    if guaranteed_extra > 0:
        guaranteed_result = _condense(run_simulation_verbose(
            **{**run_params, "total_pulls": guaranteed_pulls}, trials=ADVISOR_TRIALS,
        ))
        runs.append(guaranteed_result)
        tiers.append({
            "key": "whale", "label": "the guaranteed pull count",
            "total_pulls": guaranteed_pulls,
            "success_rate": guaranteed_result["success_rate"], "extra_pulls": guaranteed_extra,
        })
    # guaranteed_extra <= 0 means the current budget already meets or
    # exceeds the pull count that guarantees the goal outright: nothing
    # further to show, the current-budget figure above already IS that.

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
    "You may be given further pull-count figures beyond the current budget (a modest top-up and/or "
    "a pull count that guarantees the goal). Never label these as tiers, levels, or address them "
    "one by one by name (do not say 'F2P', 'spend if you really want it', 'guaranteed', 'big "
    "spender', or similar labels); weave them in only where they genuinely add something, speaking "
    "to different situations implicitly instead, for example 'if you're free-to-play or price "
    "sensitive, ...', 'if this character or weapon (or an extra copy of it, for its added "
    "strength) really matters to you, ...', 'to guarantee you hit your goal no matter what, ...'. "
    "The simplest framing is often best and does not need three separate clauses: state the current "
    "odds in plain qualitative terms based on the actual percentage (roughly: under 20 percent is "
    "pretty low, 20 to 45 is a stretch, 45 to 60 is close to a coin toss, 60 to 85 is pretty good, "
    "above 85 is comfortable, at or near 100 is already about as safe as it gets) and give a plain "
    "recommendation; only mention a further pull count when it adds something useful, for example a "
    "small top-up meaningfully changing the picture is worth flagging, while a much larger step for "
    "little extra gain is worth saying isn't worth it. If nothing further was given at all (the "
    "current odds are already about as good as they can get), just say so plainly and stop, do not "
    "invent a further option to discuss. When given the pull-count gap between two such figures (as "
    "a percent of the current budget), use it to judge how big an ask that step actually is, a small "
    "percentage is a minor step, one over 100 percent roughly doubles or more than doubles the "
    "budget and is a real commitment, describe it as one rather than presenting every step as "
    "equally easy just because a number exists for it. More pulls do not always mean spending money, "
    "gacha games often let players earn more through events, achievements, or side quests too, so "
    "prefer neutral phrasing like 'getting' or 'having' more pulls over 'spending' or 'buying' them, "
    "unless the question itself is specifically about paying. Otherwise (a plain what-if with "
    "nothing further given) 2 to 4 short sentences is still right. "
    "Always cite the specific success rates you got from the tool (for example, at 220 pulls "
    "it is 71 percent) so your answer is grounded in the numbers. Be honest: if a further pull "
    "count barely helps or the odds are poor even at the highest figure given, say so, and do not "
    "push the user to get more pulls than they need to; if the current odds are already comfortable, "
    "say that plainly and don't manufacture urgency to get more anyway. Respect the stated goal and "
    "starting conditions: do not assume the user "
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


_GOAL_CHANGE_FIELDS = (
    "additional_character_copies_wanted", "additional_weapon_copies_wanted",
    "characters_wanted_total", "weapons_wanted_total",
)


def _has_goal_change(extracted):
    """True when the question changes the goal even though it reported no
    events at all ("what if I go for C1"). Such a question used to fall
    through to the model's free-form tool loop, which silently chose its
    own strategy and showed no goal anywhere; routing it through
    reconcile() with an empty event list instead makes the resulting goal
    a deterministic, displayed fact (a "Current Goal" line), the same as
    any other reconciled scenario."""
    return any(extracted.get(key) for key in _GOAL_CHANGE_FIELDS)


_BUDGET_CHANGE_FIELDS = ("pull_additions", "total_pulls_restated", "reserve_pulls")


def _extraction_applies(extracted):
    """Whether an extraction has anything to reconcile: events, a goal
    change, a conditional plan, or a budget change (each of the last three
    can stand alone, starting from the form's own situation, with no
    events at all)."""
    return bool(extracted.get("has_event_sequence") or _has_goal_change(extracted) or active_plan(extracted)
                or extracted.get("branches") or any(extracted.get(key) for key in _BUDGET_CHANGE_FIELDS))


# One letter per branch, up to MAX_SCENARIOS.
_BRANCH_LETTERS = "ABCD"

# The per-branch fields a branch may set on top of the shared state, see
# BRANCH_SCHEMA; everything else (a restated total, starting corrections)
# is only ever top-level.
_BRANCH_FIELDS = (
    "pulls_remaining_stated", "pull_additions", "reserve_pulls",
    "additional_character_copies_wanted", "additional_weapon_copies_wanted",
    "characters_wanted_total", "weapons_wanted_total",
)


def _scenarios_from_extraction(extracted):
    """Split one extraction result into reconcile() inputs: (shared,
    branches). shared is the top-level fields, the whole question for a
    single path, or only what every branch shares when there are branches
    (a plan then belongs to a branch, never the shared part). branches is
    [(condition_label, view), ...], each view holding only that branch's
    own assumed events and adjustments, capped at MAX_SCENARIOS."""
    branches = (extracted.get("branches") or [])[:MAX_SCENARIOS]
    shared = {"has_event_sequence": True, "plan": None if branches else extracted.get("plan")}
    for key in _SCENARIO_FIELDS:
        shared[key] = extracted.get(key)
    views = []
    for branch in branches:
        view = {"has_event_sequence": True, "events": branch.get("assumed_events") or [],
                "total_pulls_restated": None, "plan": branch.get("plan"),
                "stops_pulling": bool(branch.get("stops_pulling"))}
        for key in _BRANCH_FIELDS:
            view[key] = branch.get(key)
        if view["stops_pulling"]:
            # Stopping already says what happens to the goal; a goal change
            # the model also reports alongside it would only mislabel the
            # branch's own runs (e.g. "obtained 0 of 0").
            for key in _GOAL_CHANGE_FIELDS:
                view[key] = None
        views.append((branch.get("condition_label"), view))
    return shared, views


def _has_content(view):
    return bool(view.get("events") or _has_goal_change(view) or active_plan(view)
                or any(view.get(key) for key in _BUDGET_CHANGE_FIELDS + ("pulls_remaining_stated",)))


def _continue_from(state, baseline_params, baseline_stats, from_shared):
    """A branch's own baseline: exactly where the shared part left off, so
    reconcile() treats it as the starting situation, plus `prior` so its
    lines keep counting runs and copies on from the shared ones."""
    params = {
        **baseline_params,
        "total_pulls": state["total_pulls"],
        "start_char_pity": state["start_char_pity"], "start_char_guarantee": state["start_char_guarantee"],
        "start_weapon_pity": state["start_weapon_pity"], "start_weapon_guarantee": state["start_weapon_guarantee"],
    }
    stats = {**baseline_stats, "desired_characters": state["desired_characters"],
             "desired_weapons": state["desired_weapons"]}
    prior = {"runs": state["runs"], "obtained": state["obtained"],
             "guarantee_sources": state.get("guarantee_sources")}
    if from_shared:
        prior["budget_source"] = "the events shared by every scenario above leave only"
    return params, stats, prior


def _branch_conflict(conflict, index, label):
    """A conflict found inside one branch, made distinct from the same
    conflict in another branch (or the shared part) so its answer can be
    routed back to the right place, and labeled with its branch."""
    return {
        **conflict,
        "conflict_key": f"branch{index}:{conflict['conflict_key']}",
        "header": f"Scenario {_BRANCH_LETTERS[index]}, {conflict['header']}",
        "branch_index": index, "branch_label": label,
    }


def _reconcile_scenario_list(extracted, baseline_params, baseline_stats):
    """Reconcile one extraction result. A single path reconciles exactly as
    it always has. With branches, the question is a tree: the shared part
    reconciles once (its own conflicts surface exactly like a single
    path's), then each branch reconciles FROM the shared part's end state,
    so it never repeats the shared events and can't disagree with them.

    A conflict inside a branch is still a genuine ambiguity for the user,
    not a decline: every branch's conflicts are collected (each keyed and
    labeled by its branch), so they can all be answered in one round.

    Returns {"ok": True, "scenarios": [...]} for a single path, {"ok":
    True, "branched": True, "shared_lines": [...], "scenarios": [...]} for
    branches (each scenario carrying its condition_label and letter), or
    a conflict / error result."""
    shared, branches = _scenarios_from_extraction(extracted)

    if not branches:
        reconciled = reconcile(shared, baseline_params, baseline_stats)
        if not reconciled["ok"]:
            if reconciled.get("conflict"):
                return {"ok": False, "conflict": True,
                        "lines": reconciled["lines"], "conflicts": reconciled["conflicts"]}
            return {"ok": False, "error": reconciled.get("error")}
        return {"ok": True, "scenarios": [reconciled]}

    shared_lines = []
    state = {
        "total_pulls": baseline_params["total_pulls"],
        "start_char_pity": baseline_params["start_char_pity"],
        "start_char_guarantee": baseline_params["start_char_guarantee"],
        "start_weapon_pity": baseline_params["start_weapon_pity"],
        "start_weapon_guarantee": baseline_params["start_weapon_guarantee"],
        "desired_characters": baseline_stats["desired_characters"],
        "desired_weapons": baseline_stats["desired_weapons"],
        "obtained": {"character": 0, "weapon": 0}, "runs": {"character": 0, "weapon": 0},
    }
    from_shared = _has_content(shared)
    if from_shared:
        reconciled = reconcile(shared, baseline_params, baseline_stats)
        if not reconciled["ok"]:
            if reconciled.get("conflict"):
                return {"ok": False, "conflict": True,
                        "lines": reconciled["lines"], "conflicts": reconciled["conflicts"]}
            return {"ok": False, "error": f"the facts shared by every scenario: {reconciled.get('error')}"}
        shared_lines = reconciled["lines"][:reconciled["fact_line_count"]]
        state = reconciled["end_state"]

    params, stats, prior = _continue_from(state, baseline_params, baseline_stats, from_shared)
    scenarios, conflicts = [], []
    for i, (label, view) in enumerate(branches):
        if not _has_content(view):
            return {"ok": False, "error": (
                f"{label or f'branch {i + 1}'}: a branch must state the outcome it assumes (assumed_events) "
                f"or what changes in it"
            )}
        if any(e.get("pity_at_outcome") is None for e in view["events"]) and view.get("pulls_remaining_stated") is None:
            # The most common misread: "if I win the 50/50 ..., if I lose
            # ..." with no number is a plan, decided by the simulation, not
            # assumed outcomes. Saying so is what the retry needs to fix it.
            return {"ok": False, "error": (
                f"{label or f'branch {i + 1}'}: a branch assumes an outcome with no pity or pull number, so "
                f"it cannot be a branch. A win or loss with no stated number is decided by the simulation: "
                f"report the whole question as one plan instead (branches null), in pulling order, e.g. a "
                f"step with on_loss 'skip' and skip_to_banner for 'if I lose, I'll switch to the weapon', "
                f"then the steps a win continues with, then the step a loss switches to"
            )}
        reconciled = reconcile(view, params, stats, prior=prior)
        if reconciled.get("conflict"):
            conflicts.extend(_branch_conflict(c, i, label) for c in reconciled["conflicts"])
            continue
        if not reconciled["ok"]:
            return {"ok": False, "error": f"{label or f'branch {i + 1}'}: {reconciled.get('error')}"}
        reconciled["condition_label"] = label
        reconciled["letter"] = _BRANCH_LETTERS[i]
        reconciled["stops_pulling"] = view["stops_pulling"]
        scenarios.append(reconciled)

    if conflicts:
        return {"ok": False, "conflict": True, "lines": shared_lines, "conflicts": conflicts}
    return {"ok": True, "branched": True, "shared_lines": shared_lines, "scenarios": scenarios}


_RESOLUTION_LABELS = {
    "total_includes_existing_pity": "USER CLARIFICATION: ",
    "starting_situation": "USER MODIFICATION TO INITIAL SITUATION: ",
    "prompt": "USER MODIFICATION TO PROMPT: ",
}


def _build_clarification_feedback(conflicts, clarifications):
    """Turn the user's typed answers to one or more conflict clarifying
    questions into error_feedback text for a re-extraction, matched to the
    freshly-rediscovered conflicts (not whatever the frontend echoes back)
    by conflict_key, the same stable id the conflict blocks were built
    with in the first place (see session_state.py's _build_conflict_block
    and _build_budget_conflict_block)."""
    answers = {c["conflict_key"]: c["answer"] for c in clarifications if c.get("answer")}
    parts = []
    for conflict in conflicts:
        answer = answers.get(conflict["conflict_key"])
        if not answer:
            continue
        where = (
            f'In the branch "{conflict["branch_label"]}" (its assumed_events), '
            if "branch_index" in conflict else ""
        )
        if _is_budget_conflict(conflict):
            parts.append(f'{where}You asked: "{conflict["question"]}" The user answered: "{answer}"')
        else:
            parts.append(
                f'{where}For the {conflict["banner"]} banner, attempt {conflict["attempt_number"]} of '
                f'{conflict["total_attempts"]} (reported as {conflict["reported_pulls"]} pulls spent), '
                f'you asked: "{conflict["question"]}" The user answered: "{answer}"'
            )
    return " ".join(parts)


def _is_budget_conflict(conflict):
    # "budget" alone, or "branch1:budget" inside a branch.
    return conflict["conflict_key"].rsplit(":", 1)[-1] == "budget"


def _conflict_view(conflict, extracted):
    """The part of a (retry) extraction a conflict was about: the whole
    thing for a shared or single-path conflict, or just its own branch's
    assumed events for a branch conflict, which is what its attempt
    numbers count within."""
    if "branch_index" not in conflict:
        return extracted
    branches = extracted.get("branches") or []
    branch = branches[conflict["branch_index"]] if conflict["branch_index"] < len(branches) else {}
    return {"events": branch.get("assumed_events") or [], "total_pulls_restated": None,
            "starting_situation_corrections": extracted.get("starting_situation_corrections")}


def _classify_resolution(conflict, extracted, fallback_answer):
    """Dispatch to the right classifier for this conflict's kind (see
    session_state.py's _build_conflict_block for pity, and
    _build_budget_conflict_block for a budget overrun): never let the
    model choose its own label text for this, the exact prefix matters
    and prompt-only instructions to relay specific wording have
    repeatedly not been followed reliably elsewhere in this module.

    Returns (kind, description_text) or None if nothing about this
    conflict actually changed (the retry didn't apply it)."""
    extracted = _conflict_view(conflict, extracted)
    if _is_budget_conflict(conflict):
        return _classify_budget_resolution(conflict, extracted, fallback_answer)
    return _classify_pity_resolution(conflict, extracted, fallback_answer)


def _classify_pity_resolution(conflict, extracted, fallback_answer):
    """After a retry extraction meant to resolve one pity conflict, work
    out which of the three ways the user resolved it, by comparing the
    retry's output against what originally conflicted."""
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


def _classify_budget_resolution(conflict, extracted, fallback_answer):
    """After a retry extraction meant to resolve a budget-overrun
    conflict, work out which of two ways the user resolved it. "I gained
    pulls since the initial situation" and "the initial number was
    actually different" both resolve identically (a corrected total_pulls
    for this run), there's no way to tell those two framings apart after
    the fact, and no need to: either way the starting situation for this
    run is what changed, not a reported event number, so both share the
    same label. Only when the retry instead corrects a reported event's
    own number (total_pulls_restated left alone) does the "wrong prompt
    number" label apply."""
    restated = extracted.get("total_pulls_restated")
    if restated is not None:
        return ("starting_situation", f"Starting budget corrected to {restated} total pulls.")
    return ("prompt", fallback_answer)


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
    if not _extraction_applies(extracted):
        return {"applies": False}

    result = _reconcile_scenario_list(extracted, baseline_params, baseline_stats)
    if result["ok"]:
        return {"applies": True, "reply_language": extracted.get("reply_language"), **result}

    if result.get("conflict"):
        if clarifications:
            feedback = _build_clarification_feedback(result["conflicts"], clarifications)
            if feedback:
                retry_extracted = _extract_session_state(
                    client, model, question, baseline_stats, error_feedback=feedback,
                )
                if _extraction_applies(retry_extracted):
                    retry_baseline_params = _apply_starting_situation_corrections(
                        baseline_params, retry_extracted.get("starting_situation_corrections"),
                    )
                    retried = _reconcile_scenario_list(retry_extracted, retry_baseline_params, baseline_stats)
                    if retried["ok"] or retried.get("conflict"):
                        annotations = []
                        for conflict in result["conflicts"]:
                            answer = next(
                                (c["answer"] for c in clarifications
                                 if c["conflict_key"] == conflict["conflict_key"]),
                                None,
                            )
                            if not answer:
                                continue
                            classified = _classify_resolution(conflict, retry_extracted, answer)
                            if classified:
                                kind, text = classified
                                annotations.append({"label": _RESOLUTION_LABELS[kind], "text": text})
                        return {"applies": True, "annotations": annotations,
                                "reply_language": retry_extracted.get("reply_language"), **retried}
        # No clarifications yet, or applying them didn't produce anything
        # usable: surface the conflict as-is rather than silently decline.
        return {"applies": True, **result}

    first_error = result["error"]
    extracted = _extract_session_state(
        client, model, question, baseline_stats, error_feedback=first_error,
    )
    if not _extraction_applies(extracted):
        return {"applies": True, "ok": False, "error": first_error}

    result = _reconcile_scenario_list(extracted, baseline_params, baseline_stats)
    if result["ok"] or result.get("conflict"):
        return {"applies": True, "reply_language": extracted.get("reply_language"), **result}
    return {"applies": True, "ok": False, "error": result["error"]}


def _run_params(baseline_params, scenario):
    return {
        **baseline_params,
        "strategy": scenario["strategy"],
        "total_pulls": scenario["total_pulls"],
        "start_char_pity": scenario["start_char_pity"],
        "start_char_guarantee": scenario["start_char_guarantee"],
        "start_weapon_pity": scenario["start_weapon_pity"],
        "start_weapon_guarantee": scenario["start_weapon_guarantee"],
    }


def _simulate_routes(baseline_params, scenario):
    """Simulate a reconciled plan once, run by run, so each simulated run
    carries its own leftover pulls and pity into the next step, then split
    it into routes (session_state.build_route_groups). Returns (groups,
    facts): the route cards for the UI, and one grounded sentence block
    per route for the narration."""
    plan_result = simulate_plan(**_run_params(baseline_params, scenario), trials=ADVISOR_TRIALS)
    routes, omitted = build_route_groups(scenario["strategy"], plan_result)

    groups = [{"label": r["label"], "lines": r["lines"]} for r in routes]
    if omitted:
        groups.append(build_omitted_routes_group(omitted))

    facts = []
    for route in routes:
        runs_text = _lines_to_text(route["lines"])
        leftover = round(route["avg_leftover"])
        if route["ending"] == "lost":
            ending = (
                f"The plan stops here after losing the {route['stopped_banner']} 5-star, keeping {leftover} "
                f"pulls on average."
            )
        elif route["ending"] == "capped_stop":
            ending = (
                f"The plan stops at its limit, keeping {leftover} pulls on average, having obtained "
                f"{route['obtained']}."
            )
        elif route["ending"] == "partial":
            ending = (
                f"Not every step was won along the way (a limit was hit, or a loss moved the plan on), so "
                f"this route ends with {route['obtained']} (not everything planned), with {leftover} pulls "
                f"left on average."
            )
        else:
            ending = f"Every step is obtained, with {leftover} pulls left on average."
        # Every banner's guarantee where this route ends, not just one the
        # route's own loss created: a guarantee carried in from before the
        # plan (e.g. an earlier lost 50/50) was going unmentioned otherwise.
        guarantee_bits = []
        for banner, share in route["end_guarantees"].items():
            source = (
                "carried over from before this plan, since this route never pulls on it"
                if banner not in route["pulled_banners"] else "from this route's own pulls"
            )
            if share >= 1:
                guarantee_bits.append(
                    f"the {banner} banner's next 5-star IS guaranteed to be the featured one ({source})")
            elif share <= 0:
                guarantee_bits.append(f"the {banner} banner has no guarantee (its next 5-star is a "
                                      f"{'50/50' if banner == 'character' else '75/25'})")
            else:
                guarantee_bits.append(f"the {banner} banner ends guaranteed in {round(share * 100)}% of these runs")
        if guarantee_bits:
            ending += (
                f" Where this route leaves the player: {'; '.join(guarantee_bits)}. Mention every guarantee "
                f"this route leaves, not only the one a loss here created."
            )
        for banner, pity in route["capped_pity"]:
            cfg = baseline_params["char_pity_config" if banner == "character" else "weapon_pity_config"]
            ending += (
                f" Stopping at the limit keeps about {pity} pity on the {banner} banner, which carries over "
                f"to their next pulls there. Unlike a loss, hitting a limit creates no guarantee by itself."
            )
            # The most useful thing to say about a limit is often where it
            # falls on the pity curve, which the model doesn't reliably
            # work out on its own from the raw numbers, so it's stated.
            if pity >= cfg["soft_pity_start"] - 5:
                where = "just after" if pity >= cfg["soft_pity_start"] else "just before"
                ending += (
                    f" That stop falls {where} the {banner} banner's soft pity begins "
                    f"({cfg['soft_pity_start']}), the point where each further pull's odds climb steeply "
                    f"toward a certain 5-star by hard pity ({cfg['hard_pity']}), so these runs stop just as "
                    f"pulling becomes most likely to pay off; worth pointing out."
                )
        if 0 < route["sample"] < MIN_ROUTE_SAMPLE:
            ending += (
                f" This route is rare (only {route['sample']} of {plan_result['trials']} simulated runs), "
                f"so its leftover figure is approximate; say so if you cite it."
            )
        facts.append(f'"{route["label"]}": chance {route["chance"]:.2f}%. {runs_text}. {ending}')
    if omitted:
        facts.append(
            f"{omitted['routes']} further, rarer routes are not shown individually; together they make up "
            f"{omitted['chance']:.2f}% of simulated runs. Mention them only as a small remainder, if at all."
        )
    return groups, facts


def _pull_arithmetic_guard(question, lines, total):
    """Ground the narration's pull total. A generic guard alone still let
    the model add a stated "100 more pulls" a second time (240 became 340)
    in live testing; spelling out the exact sum it's already part of is
    what holds, the same lesson as naming the question's numbers on the
    single-scenario path."""
    question_numbers = sorted(set(re.findall(r"\d+", question)))
    into = f"the {total} starting pulls" if total is not None else "the verified pull figures below"
    guard = (
        f" Numbers mentioned in the question ({', '.join(question_numbers)}) are already fully "
        f"incorporated into {into}; never add or subtract any of them again."
        if question_numbers else ""
    )
    for line in lines:
        if line["label"] in ("Additional Pulls Mentioned", "Pulls Kept in Reserve"):
            pills = line["pills"]
            amounts = " and ".join(str(p["value"]) for p in pills[1:-2] if p["kind"] != "operator")
            what = "additional pulls the question mentions are" if pills[1]["value"] == "+" else "pulls kept in reserve are"
            guard += (
                f" The {amounts} {what} already accounted for: {_lines_to_text([line]).split(': ', 1)[1]}. "
                f"No simulation without them was run, so do not comment on how they change the odds."
            )
    return guard


def _banner_facts(baseline_params):
    char_cfg, weapon_cfg = baseline_params["char_pity_config"], baseline_params["weapon_pity_config"]
    return (
        f"Banner facts you may use for that: the character banner's soft pity (where the odds start "
        f"climbing steeply) begins at {char_cfg['soft_pity_start']} and its hard pity is "
        f"{char_cfg['hard_pity']}; the weapon banner's soft pity begins at {weapon_cfg['soft_pity_start']} "
        f"and its hard pity is {weapon_cfg['hard_pity']}."
    )


_ROUTE_RULES = (
    "Each route's chance is its share of all simulated runs, and a run's rate is out of the runs that "
    "reached it, which is why a route's chance is the product of its run rates."
)


def _answer_plan_routes(client, model, question, baseline_params, scenario, annotations=None, reply_language=None):
    """Answer a question that reconciled into a conditional plan ("pull the
    weapon first; if I win, go for the character; if I lose, stop"),
    simulated and split into routes (see _simulate_routes). Like the
    branching path, the model gets one tool-free turn to narrate routes
    that are already fully computed.

    Returns (answer_text, runs, breakdown) where breakdown carries both the
    shared "lines" (the facts everything starts from) and one group per
    route."""
    groups, route_facts = _simulate_routes(baseline_params, scenario)
    breakdown = {"status": "ok", "lines": scenario["lines"], "groups": groups}
    if annotations:
        breakdown["annotations"] = annotations

    total = scenario["total_pulls"]
    number_guard = _pull_arithmetic_guard(question, scenario["lines"], total)
    number_guard += f" Every route starts from exactly {total} pulls; never describe any other starting total."
    context = (
        f"Verified starting point (already reconciled from the question, treat as fact and do not "
        f"restate it differently): {_lines_to_text(scenario['lines'])} The user describes a conditional "
        f"plan, and every way it can play out has ALREADY been simulated, each simulated run carrying "
        f"its own leftover pulls into the next step. {_ROUTE_RULES} Explain every route below, each with "
        f"its chance, in plain terms (do not skip any), then give a plain recommendation about the plan "
        f"itself (for example, whether stopping after a loss or at a limit is worth it, given the pulls it "
        f"keeps and the guarantee it leaves). Cite only the exact numbers below, never state a number that "
        f"is not given here, and do not offer to run another simulation, none is available for this "
        f"question. Do not just read the numbers back: say briefly what each route means for the player, "
        f"then what the plan trades off. {_banner_facts(baseline_params)} Write plain sentences, not a "
        f"list.{number_guard}\n\n" + "\n\n".join(route_facts)
    )
    language = reply_language or "the same language as the question"
    # The raw wording ("I'm getting 100 more pulls") kept being re-read as
    # pulls on top of a total that already includes them, the same failure
    # the goal-only path hit with "C1", so the question the model answers
    # is restated from the reconciled plan. The original is still given,
    # but only for what the player cares about, never for its numbers.
    plan_text = "; then ".join(
        f"{route_step_name(phase)}{step_limit_note(phase)}" for phase in scenario["strategy"]
    )
    follow_up = (
        f"My plan, in order: {plan_text}. I have exactly {total} pulls for it. How does this plan look, "
        f"and is it a good plan? Reply in {language}."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{context}\n\nThe player's original wording, only for what they care about (every number in "
            f"it is already reflected above and must not be used): \"{question}\"\n\n"
            f"Follow-up question: {follow_up}"
        )},
    ]
    # No tools= param here on purpose: every figure is already computed,
    # and omitting tools entirely prevents a redundant re-run.
    response = client.chat.completions.create(model=model, messages=messages, temperature=ADVISOR_TEMPERATURE)
    return (response.choices[0].message.content or "").strip(), [], breakdown


def _answer_branching_scenarios(client, model, question, baseline_params, reconciled, goal_label, goal_description):
    """Answer a question that reconciled into a tree: shared facts, then two
    or more mutually exclusive branches the question ASSUMES ('if I win
    around pity 30, ...; if I lose around 75, ...'). Each branch is
    simulated deterministically here, a plan inside a branch as its own
    routes, and the model gets one tool-free turn to address them all.

    A branch is an outcome the user chose, so it has no chance of its own;
    only routes inside a branch, decided by the simulation, do.

    Returns (answer_text, runs, breakdown) where breakdown is {"status":
    "ok", "lines": shared facts, "groups": [{"label", "lines", "groups"?},
    ...]}, one group per branch, nested route groups for a branch's plan."""
    shared_lines = reconciled["shared_lines"]
    groups, runs, summaries = [], [], []

    for scenario in reconciled["scenarios"]:
        label = f"Scenario {scenario['letter']}: {scenario.get('condition_label') or 'Branch'}"
        lines_text = _lines_to_text(scenario["lines"])
        group = {"label": label, "lines": scenario["lines"]}

        if scenario["goal_complete"] or scenario["pulls_exhausted"]:
            status = "the goal is already complete" if scenario["goal_complete"] else "no pulls remain"
            summaries.append(f'"{label}": Verified state: {lines_text} ({status}; nothing to simulate for it).')
        elif scenario.get("stops_pulling"):
            # The player's own choice to stop: the start-state and goal
            # lines would describe runs that never happen, so the branch
            # ends on what it keeps instead, with nothing simulated.
            group["lines"] = scenario["lines"][:scenario["fact_line_count"]] + [build_stop_line(scenario)]
            summaries.append(
                f'"{label}": Verified state: {_lines_to_text(group["lines"])} The player chooses to stop '
                f"pulling here, keeping {scenario['total_pulls']} pulls; nothing further is simulated and "
                f"nothing more is obtained in this scenario, so this much of the goal is left unobtained: "
                f"{remaining_goal_text(scenario['remaining_characters'], scenario['remaining_weapons'])}. "
                f"The guarantees shown on its Stops Here line carry over to whenever they pull again."
            )
        elif scenario.get("plan"):
            route_groups, route_facts = _simulate_routes(baseline_params, scenario)
            group["groups"] = route_groups
            summaries.append(
                f'"{label}": Verified state: {lines_text} In this scenario the player follows a conditional '
                f"plan from {scenario['total_pulls']} pulls, ALREADY simulated into routes. {_ROUTE_RULES} "
                f"Routes:\n" + "\n".join(route_facts)
            )
        else:
            # Run each branch's reconciled scenario ourselves, right now,
            # same reasoning as the single-scenario path: a result that
            # already exists before the model's first turn can't be gotten
            # wrong, or attributed to the wrong branch.
            run_params = _run_params(baseline_params, scenario)
            result = _condense(run_simulation_verbose(**run_params, trials=ADVISOR_TRIALS))
            runs.append(result)
            group["lines"] = scenario["lines"] + [build_result_line(run_params["total_pulls"], result["success_rate"])]
            remaining = remaining_goal_text(scenario["remaining_characters"], scenario["remaining_weapons"])
            summaries.append(
                f'"{label}": Verified state: {lines_text} What remains to obtain: {remaining}. A '
                f"simulation on this exact state has ALREADY been run: {run_params['total_pulls']} pulls, "
                f"success rate {result['success_rate']}, average leftover pulls on success "
                f"{result['avg_leftover_pulls_on_success']}, most common failure state "
                f"{result['most_common_failure_state']}."
            )
        groups.append(group)

    shared_text = (
        f"Verified facts shared by every scenario (already reconciled, treat as fact): "
        f"{_lines_to_text(shared_lines)} " if shared_lines else ""
    )
    context = (
        f"Original full goal was {goal_description} (goal {goal_label}). {shared_text}The question "
        f"describes {len(groups)} mutually exclusive scenarios, each continuing from that same point and "
        f"each ASSUMING an outcome the player chose, so a scenario itself has no chance of happening to "
        f"cite; only routes inside a scenario's plan have chances. Every scenario below is already verified "
        f"and, where applicable, already simulated, treat all of it as fact and do not restate it "
        f"differently. Address every scenario in your answer, describing each one only in terms of what "
        f"remains for it, never the original full goal, and cite only the exact numbers given for each "
        f"scenario below; never state a number for a scenario that was not given here. No further "
        f"simulation is available for this question, do not offer to run one. Do not just read the "
        f"numbers back: say briefly what each scenario means for the player, and compare them where the "
        f"question asks. {_banner_facts(baseline_params)}"
        f"{_pull_arithmetic_guard(question, shared_lines + [l for g in groups for l in g['lines']], None)}"
        f"\n\n" + "\n\n".join(summaries)
    )
    language = reconciled.get("reply_language") or "the same language as the question"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nFollow-up question: {question}\n\nReply in {language}."},
    ]
    # No tools= param here on purpose: the API rejects tool_choice="none" when
    # tools isn't also passed, and omitting tools entirely already prevents
    # any tool call just as well, there's nothing for the model to invoke.
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=ADVISOR_TEMPERATURE,
    )
    breakdown = {"status": "ok", "lines": shared_lines, "groups": groups}
    if reconciled.get("annotations"):
        breakdown["annotations"] = reconciled["annotations"]
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
    one group per branch, see _answer_branching_scenarios), {"status": "ok",
    "lines": [...], "groups": [...]} (a conditional plan: the shared
    starting point as lines, then one group per route, see
    _answer_plan_routes), {"status":
    "conflict", "lines": [...], "conflicts": [...]} (one or more banners'
    reported pulls don't fit their actual starting pity, or the reported
    events together use more pulls than the stated starting budget, a
    genuine ambiguity for the user to resolve, not a plain parse failure;
    answer_text is empty, there is nothing to say until it's resolved), or {"status":
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
    if reconciled.get("applies") and reconciled.get("ok") and reconciled.get("branched"):
        return _answer_branching_scenarios(
            client, model, question, baseline_params, reconciled, goal_label, goal_description,
        )
    if reconciled.get("applies") and reconciled.get("ok") and reconciled["scenarios"][0].get("plan"):
        return _answer_plan_routes(
            client, model, question, baseline_params, reconciled["scenarios"][0],
            annotations=reconciled.get("annotations"), reply_language=reconciled.get("reply_language"),
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
        # Numbered from 1 like every other figure here, the current-budget
        # result is just as much an AI-orchestrated run as the ones after
        # it, not a separate, un-orchestrated "given" the others build on.
        lines = lines + [build_spending_tier_line("AI Agent Simulated Result 1",
                                                    run_params["total_pulls"], primary_result["success_rate"])]

        # Up to two further pull-count figures, pre-computed here rather
        # than left to the model: a comfortably-safe top-up
        # (MODERATE_SPEND_TARGET) and the pull count that mathematically
        # guarantees the remaining goal via hard pity, alongside the
        # current-budget result above. Every figure the model gets to
        # cite already came from a real run or closed-form arithmetic,
        # never a guess, and it's told to weave these in implicitly
        # rather than list them as labeled tiers (see SYSTEM_PROMPT).
        tiers, tier_runs, tier_comparisons = _spending_tiers(scenario, run_params, baseline_params, primary_result)
        runs.extend(tier_runs)
        # Generic, unbranded pill labels, numbered on from the current-
        # budget result above (so a single further figure reads "2", not
        # "1" again): what these figures ARE (a further AI-run simulation)
        # matters for the UI receipts, not what marketing tier they'd map
        # to, that framing belongs only in the prose below, softened, and
        # only when it adds something.
        for i, tier in enumerate(tiers[1:], start=2):
            label = f"AI Agent Simulated Result {i}"
            lines = lines + [build_spending_tier_line(label, tier["total_pulls"], tier["success_rate"])]

        breakdown = {"status": "ok", "lines": lines}
        if reconciled.get("annotations"):
            breakdown["annotations"] = reconciled["annotations"]
        remaining = remaining_goal_text(scenario["remaining_characters"], scenario["remaining_weapons"])

        tier_facts = [
            f"Your current budget, spending nothing further: {run_params['total_pulls']} pulls, "
            f"success rate {primary_result['success_rate']}, average leftover pulls on success "
            f"{primary_result['avg_leftover_pulls_on_success']}, most common failure state "
            f"{primary_result['most_common_failure_state']}."
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
                f"A modest additional pull count, worth mentioning only if this character or weapon "
                f"(or an extra copy of it, for its added strength) genuinely matters to the player: "
                f"{moderate['total_pulls']} total pulls (an extra {moderate['extra_pulls']} beyond the "
                f"current budget), success rate {moderate['success_rate']}, comfortably likely without "
                f"needing a full guarantee."
            )
        if whale:
            tier_facts.append(
                f"The pull count that guarantees the entire remaining goal outright, worth mentioning "
                f"only if the player wants zero risk: {whale['total_pulls']} total pulls (an extra "
                f"{whale['extra_pulls']} beyond the current budget), success rate {whale['success_rate']}. "
                f"This is not a probability, hard pity mathematically forces this outcome regardless of "
                f"luck at that many pulls."
            )
        if not moderate and not whale:
            tier_facts.append(
                "The current budget is already as good as it gets, at or near a full guarantee via hard "
                "pity, regardless of luck: there is nothing further worth pursuing, this is already as "
                "safe as it can be."
            )

        if tier_comparisons:
            comparison_facts = [
                f"{c['from_label']} to {c['to_label']}: {c['delta_pulls']} more pulls "
                f"({c['pct_of_current_budget']}% on top of the current {run_params['total_pulls']}-pull budget)."
                for c in tier_comparisons
            ]
            tier_facts.append(
                "Exact pull-count gaps between the figures above, each also given as a percent of the "
                "current budget so you have one consistent yardstick for how big a step each one "
                "actually is: " + " ".join(comparison_facts) + " Use these to comment on relative "
                "effort, for example a gap under roughly 25 percent of the current budget is a small "
                "top-up, one over 100 percent roughly doubles or more than doubles it, that's a much "
                "bigger ask, say so plainly rather than treating every step as equally reasonable."
            )

        if scenario.get("goal_only"):
            # A pure goal change: say plainly what the goal now is, in
            # words, and leave out the usual "original goal vs what
            # remains" framing, which a question wording like "C1" or
            # "1凸" kept colliding with (the model re-derived its own,
            # wrong, copy count from it instead of using the stated goal).
            context = (
                f"Verified scenario (already reconciled from the question, treat as fact, do not "
                f"re-derive it from the question's own wording such as C1/W2/1凸): the goal being "
                f"asked about is exactly {remaining}, starting from the current pity and guarantee "
                f"shown here: {lines_text} The user's baseline goal was {goal_description}, and "
                f"their wording (C1, W2, 1凸 and so on) means ADDING copies on top of that baseline, "
                f"which gives exactly {remaining}; that is what they intend, do not reinterpret it "
                f"as a smaller or different goal. All the simulations needed are ALREADY run below, "
                f"so answer now and never say you will run or check another one. Do not mention any "
                f"other goal or its odds (none are given). Every percentage you state must be "
                f"copied from the figures below. "
            )
        else:
            context = (
                f"Verified session state (already reconciled from the question, treat as fact "
                f"and do not restate it differently): {lines_text} "
                f"Original full goal was {goal_description} (goal {goal_label}), but what actually "
                f"still remains to obtain, after the events above, is: {remaining}. The figures "
                f"below were run for exactly that remaining goal, not the original one; describe "
                f"success and failure only in terms of what remains, anything not listed there is "
                f"already obtained and must not be described as still needed, at risk, or a "
                f"possible failure. "
            )
        context += (
            f"The following pull-count figures have ALREADY been run or computed for you; use only "
            f"these exact numbers, never invent a further pull count or success rate that isn't "
            f"listed here. " + " ".join(tier_facts) + " "
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

    follow_up = question
    if reconciled.get("applies") and reconciled.get("ok") and reconciled["scenarios"][0].get("goal_only"):
        # The user's own notation (C1, W2, 1凸, in any language) is what
        # the model kept re-reading as a smaller or different goal than
        # the one reconciliation already settled on, so for a pure goal
        # change it's shown the reconciled question instead of the raw
        # wording. Language-independent by construction.
        follow_up = (
            f"How likely am I to reach the goal of {reconciled['scenarios'][0]['goal_text']}, "
            f"and does getting more pulls change that? Reply in {reconciled.get('reply_language') or 'English'}."
        )
    elif reconciled.get("applies") and reconciled.get("ok") and reconciled["scenarios"][0].get("budget_only"):
        # Same lesson for a pure budget what-if: "20 more pulls" in the raw
        # wording invites adding the 20 again on top of a total that
        # already includes it, so the model sees the settled total instead.
        scenario = reconciled["scenarios"][0]
        follow_up = (
            f"How likely am I to reach the goal of {scenario['goal_text']} with {scenario['total_pulls']} pulls, "
            f"and does getting more pulls change that? Reply in {reconciled.get('reply_language') or 'English'}."
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nFollow-up question: {follow_up}"},
    ]

    if reconciled.get("applies") and reconciled.get("ok") and (
            reconciled["scenarios"][0].get("goal_only") or reconciled["scenarios"][0].get("budget_only")):
        # A pure goal-change (or budget-change) question is already fully answered by the
        # deterministic runs above; offering the tool only invited a
        # redundant re-run under a strategy the model chose itself, which
        # then showed up as an unexplained extra result.
        response = client.chat.completions.create(model=model, messages=messages, temperature=ADVISOR_TEMPERATURE)
        return (response.choices[0].message.content or "").strip(), runs, breakdown

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

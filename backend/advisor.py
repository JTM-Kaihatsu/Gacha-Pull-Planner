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

from openai import OpenAI

from analyzer import describe_goal
from config import get_openai_api_key, get_model
from session_state import build_result_line, reconcile
from simulation import run_simulation_verbose

# Fewer trials than the main endpoint: the advisor may run several sims per
# question, and it only needs directional numbers, not publication precision.
ADVISOR_TRIALS = 4000
MAX_TOOL_CALLS = 4

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
            "description": "Raw pulls spent on this banner to reach this outcome (the pity count). Null if the question did not give an exact pity/pull count for this specific event (e.g. 'I won with 62 to spare', 'I lost, and I have 81 pulls left'); in that case report pulls_remaining_stated instead so the true total can still be known.",
        },
        "refund_count": {
            "type": "integer",
            "description": "4-star refund pulls received back during this event, if stated (0 if none).",
        },
    },
    "required": ["event_order", "banner_type", "outcome", "pity_at_outcome", "refund_count"],
    "additionalProperties": False,
}

SESSION_STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "has_event_sequence": {
            "type": "boolean",
            "description": (
                "true only if the question describes one or more actual pull events that "
                "already happened this session (a banner pulled to a win or a loss, with a "
                "pity count). false for a pure hypothetical or strategy question with no "
                "event history, for example 'how should I split my budget between "
                "characters and weapons'."
            ),
        },
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
            "description": "A direct, total restatement of pulls remaining (e.g. 'I have 41 pulls left'). Do not use this for an amount meant to be added on top of the remaining pulls; use additional_pulls_stated for that instead.",
        },
        "additional_pulls_stated": {
            "type": ["integer", "null"],
            "description": "An amount of pulls stated as an ADDITION on top of whatever remains, not a total (e.g. 'plus around 45 more from this patch', 'and I'll get another 20 next week'). Report the raw number only; do not add it to anything yourself.",
        },
        "total_pulls_restated": {"type": ["integer", "null"]},
        "additional_character_copies_wanted": {
            "type": ["integer", "null"],
            "description": "Extra character copies wanted BEYOND the original goal (e.g. 'another character copy' = 1). Do not use this for the original goal itself, only for a stated expansion of it.",
        },
        "additional_weapon_copies_wanted": {
            "type": ["integer", "null"],
            "description": "Extra weapon copies wanted BEYOND the original goal, same rule as additional_character_copies_wanted.",
        },
    },
    "required": [
        "has_event_sequence", "events", "pulls_remaining_stated", "additional_pulls_stated",
        "total_pulls_restated", "additional_character_copies_wanted", "additional_weapon_copies_wanted",
    ],
    "additionalProperties": False,
}

EXTRACTION_SYSTEM_PROMPT = (
    "You extract a sequence of discrete pull events from a gacha follow-up question. Do "
    "not do any math, do not aggregate multiple events into one, and do not decide a "
    "strategy. If the question describes one or more actual pulls that already "
    "happened (a banner pulled to a win or a loss), report each one as its own event: "
    "which banner, win or loss, and (if an exact number was given) the pity count at "
    "that outcome and any 4-star refund received (0 if none stated). Number events "
    "sequentially starting from 1 in the order they occurred; never combine two events "
    "into one or sum their numbers together. Users often describe an outcome without "
    "giving its exact pity, instead stating how many pulls they have left afterward, "
    "for example 'I won the character with 62 pulls to spare', 'I lost my first run at "
    "the character banner and I have 81 pulls left'. In that case still report the "
    "event (banner and win/loss), leave pity_at_outcome null for it, and put the stated "
    "figure in pulls_remaining_stated instead of trying to work out what the pity must "
    "have been. If the question is a pure hypothetical or strategy question with no "
    "actual event history, set has_event_sequence to false and events to an empty "
    "list. Also report, only if explicitly stated: a direct restatement of pulls "
    "remaining (pulls_remaining_stated, required whenever any event's pity is null), "
    "an amount to add on top of whatever remains, not a total "
    "(additional_pulls_stated), a restated total pull budget (total_pulls_restated), "
    "and any extra copies wanted beyond the original goal "
    "(additional_character_copies_wanted / additional_weapon_copies_wanted). Leave a "
    "field null if the question does not state it."
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


def _lines_to_text(lines):
    """Flatten the structured pill lines into plain text for the
    interpretation model's grounding context. The pills themselves (with
    colors and tooltips) are what the UI renders; this is only so the model
    has something readable to cite from."""
    parts = []
    for line in lines:
        values = " ".join(str(p["value"]) for p in line["pills"])
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


def _run_tool(args, baseline_params):
    """Execute the run_simulation tool: merge the model's args over the baseline,
    validate, run, and return a condensed result (or an error the model can read)."""
    strategy = args.get("strategy") or baseline_params["strategy"]
    error = _validate_strategy(strategy)
    if error:
        return {"error": error}

    total_pulls = args.get("total_pulls", baseline_params["total_pulls"])
    if not isinstance(total_pulls, int) or total_pulls < 1:
        return {"error": "total_pulls must be an integer >= 1"}

    stats = run_simulation_verbose(
        total_pulls=total_pulls,
        strategy=[{"banner": p["banner"], "copies": p["copies"]} for p in strategy],
        start_char_pity=args.get("start_char_pity", baseline_params["start_char_pity"]),
        start_char_guarantee=args.get("start_char_guarantee", baseline_params["start_char_guarantee"]),
        start_weapon_pity=args.get("start_weapon_pity", baseline_params["start_weapon_pity"]),
        start_weapon_guarantee=args.get("start_weapon_guarantee", baseline_params["start_weapon_guarantee"]),
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
    "to the baseline and answer in 2 to 4 short sentences. Always cite the specific "
    "success rates you got from the tool (for example, at 220 pulls it is 71 percent) "
    "so your answer is grounded in the numbers. Be honest: if the change barely helps "
    "or the odds are poor, say so, and do not push the user to spend more than they "
    "need to. Respect the stated goal and starting conditions: do not assume the user "
    "wants a character or weapon copy they did not include. If the goal is already a "
    "single copy, there is nothing to trim, so focus on more pulls or waiting. Pity "
    "for a banner resets to 0 the instant a 5-star of that banner's type is obtained, "
    "whether it was the desired item or a losing 50/50 result, and character and "
    "weapon pity are independent of each other. Never assume a previous pity count "
    "carries forward into a new phase after a 5-star was already obtained on that "
    "banner, and never present that as an open question or a second, higher-odds "
    "scenario: it is not a matter of interpretation. "
    "No markdown, no headers, no bullet points, no em-dashes."
)


def _reconcile_session_state(client, model, question, baseline_params, baseline_stats):
    """Extract any real-progress narrative from the question and reconcile it
    deterministically. Retries the extraction once if session_state.reconcile
    flags an inconsistency. If it still cannot reconcile, returns
    applies=True, ok=False rather than silently discarding the narrative, so
    the caller can tell the user reconciliation failed instead of quietly
    answering from the unadjusted baseline as if nothing had been stated."""
    extracted = _extract_session_state(client, model, question, baseline_stats)
    if not extracted.get("has_event_sequence"):
        return {"applies": False}

    reconciled = reconcile(extracted, baseline_params, baseline_stats)
    if reconciled["ok"]:
        return reconciled

    first_error = reconciled["error"]
    extracted = _extract_session_state(
        client, model, question, baseline_stats, error_feedback=first_error,
    )
    if not extracted.get("has_event_sequence"):
        return {"applies": True, "ok": False, "error": first_error}

    reconciled = reconcile(extracted, baseline_params, baseline_stats)
    if reconciled["ok"]:
        return reconciled
    return {"applies": True, "ok": False, "error": reconciled["error"]}


def run_advisor(baseline_params, baseline_stats, question, *, model=None, max_tool_calls=MAX_TOOL_CALLS):
    """Run the agentic follow-up loop.

    Returns (answer_text, runs, breakdown) where runs is the list of condensed
    simulation results the model actually ran (for the UI receipts), and
    breakdown is None (a pure hypothetical, no event sequence to show),
    {"status": "ok", "lines": [...]} (the structured, deterministically-built
    pill data for the UI), or {"status": "error", "message": "..."} (an event
    sequence was described but could not be reconciled; see PARSE_FAILURE_MESSAGE
    for the answer text in that case)."""
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

    reconciled = _reconcile_session_state(client, model, question, baseline_params, baseline_stats)
    if reconciled.get("applies") and reconciled.get("ok"):
        lines = reconciled["lines"]
        lines_text = _lines_to_text(lines)

        if reconciled["goal_complete"] or reconciled["pulls_exhausted"]:
            # Nothing left to simulate, so answer directly from the verified state.
            breakdown = {"status": "ok", "lines": lines}
            context = (
                f"Verified session state (already reconciled from the question, treat as "
                f"fact and do not restate it differently): {lines_text} "
                f"Original full goal was {goal_description} (goal {goal_label})."
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\nFollow-up question: {question}"},
            ]
            response = client.chat.completions.create(model=model, messages=messages, tool_choice="none")
            return (response.choices[0].message.content or "").strip(), [], breakdown

        run_params = {
            **baseline_params,
            "strategy": reconciled["strategy"],
            "total_pulls": reconciled["total_pulls"],
            "start_char_pity": reconciled["start_char_pity"],
            "start_char_guarantee": reconciled["start_char_guarantee"],
            "start_weapon_pity": reconciled["start_weapon_pity"],
            "start_weapon_guarantee": reconciled["start_weapon_guarantee"],
        }

        # Run the reconciled scenario ourselves, right now, rather than
        # trusting the model to call the tool with the correct parameters.
        # Prompt-only instructions repeatedly failed to stop it re-deriving
        # (and double-counting) the pull count itself; a result that already
        # exists before the model's first turn can't be gotten wrong.
        primary_result = _condense(run_simulation_verbose(**run_params, trials=ADVISOR_TRIALS))
        runs.append(primary_result)
        lines = lines + [build_result_line(run_params["total_pulls"], primary_result["success_rate"])]
        breakdown = {"status": "ok", "lines": lines}
        context = (
            f"Verified session state (already reconciled from the question, treat as fact "
            f"and do not restate it differently): {lines_text} "
            f"Original full goal was {goal_description} (goal {goal_label}). "
            f"A simulation on this exact verified state has ALREADY been run for you: "
            f"{run_params['total_pulls']} pulls, success rate {primary_result['success_rate']}, "
            f"average leftover pulls on success {primary_result['avg_leftover_pulls_on_success']}, "
            f"most common failure state {primary_result['most_common_failure_state']}. Use this "
            f"as your primary answer; cite these exact numbers. Only call run_simulation again "
            f"if the question explicitly asks about a further, different scenario, and if so "
            f"derive any different total_pulls only by adding to or subtracting from the "
            f"{run_params['total_pulls']} figure above, never by recomputing it from the raw "
            f"figures in the question yourself."
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
            result = _run_tool(args, run_params)
            if "error" not in result:
                runs.append(result)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    # Tool-call budget exhausted: force a final text answer with what we have.
    response = client.chat.completions.create(
        model=model, messages=messages, tool_choice="none",
    )
    return (response.choices[0].message.content or "").strip(), runs, breakdown

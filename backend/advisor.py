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
from simulation import run_simulation_verbose

# Fewer trials than the main endpoint: the advisor may run several sims per
# question, and it only needs directional numbers, not publication precision.
ADVISOR_TRIALS = 4000
MAX_TOOL_CALLS = 4

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
    "to the baseline and answer in 2 to 4 short sentences. Be honest: if the change "
    "barely helps or the odds are poor, say so, and do not push the user to spend "
    "more than they need to. Respect the stated goal and starting conditions: do not "
    "assume the user wants a character or weapon copy they did not include. "
    "No markdown, no headers, no bullet points, no em-dashes."
)


def run_advisor(baseline_params, baseline_stats, question, *, model=None, max_tool_calls=MAX_TOOL_CALLS):
    """Run the agentic follow-up loop and return the model's answer text."""
    client = OpenAI(api_key=get_openai_api_key())
    model = model or get_model()

    goal_label, goal_description = describe_goal(baseline_stats)
    context = (
        f"Baseline scenario: {goal_description} (goal {goal_label}). "
        f"Total pulls: {baseline_stats['initial_pulls']}. "
        f"Character pity: {baseline_stats['start_char_pity']}, weapon pity: {baseline_stats['start_weapon_pity']}. "
        f"Baseline success rate: {baseline_stats['success_rate']}."
    )
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
            return (msg.content or "").strip()

        messages.append(_assistant_message(msg))
        for tool_call in msg.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _run_tool(args, baseline_params)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    # Tool-call budget exhausted: force a final text answer with what we have.
    response = client.chat.completions.create(
        model=model, messages=messages, tool_choice="none",
    )
    return (response.choices[0].message.content or "").strip()

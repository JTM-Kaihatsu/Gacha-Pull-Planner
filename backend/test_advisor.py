"""Tests for the open-ended advisor loop (advisor.py). OpenAI is fully mocked."""
import json
import types

import pytest

import advisor
from advisor import (
    run_advisor, _run_tool, _validate_strategy, PARSE_FAILURE_MESSAGE,
    _scenarios_from_extraction, MAX_SCENARIOS,
)


# --- fake OpenAI client that replays a queued list of responses -------------
def _msg(content=None, tool_calls=None):
    return types.SimpleNamespace(content=content, tool_calls=tool_calls)


def _response(message):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def _tool_call(call_id, name, arguments):
    return types.SimpleNamespace(
        id=call_id, type="function",
        function=types.SimpleNamespace(name=name, arguments=arguments),
    )


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return outer._responses.pop(0)

        self.chat = types.SimpleNamespace(completions=_Completions())


BASELINE_PARAMS = {
    "total_pulls": 120,
    "strategy": [{"banner": "char", "copies": 1}, {"banner": "weapon", "copies": 1}],
    "start_char_pity": 0, "start_char_guarantee": False,
    "start_weapon_pity": 0, "start_weapon_guarantee": False,
    "full_4star_chars": True,
    "char_pity_config": {"base_rate": 0.006, "soft_pity_start": 73, "hard_pity": 90},
    "weapon_pity_config": {"base_rate": 0.008, "soft_pity_start": 65, "hard_pity": 80},
}

BASELINE_STATS = {
    "desired_characters": 1, "desired_weapons": 1,
    "initial_pulls": 120, "start_char_pity": 0, "start_weapon_pity": 0,
    "success_rate": "46.75%",
}


def _event(order, banner, outcome, pity, refunds):
    return {"event_order": order, "banner_type": banner, "outcome": outcome,
            "pity_at_outcome": pity, "refund_count": refunds}


def _no_sequence():
    return json.dumps({
        "has_event_sequence": False, "events": [],
        "pulls_remaining_stated": None, "additional_pulls_stated": None,
        "total_pulls_restated": None,
        "additional_character_copies_wanted": None, "additional_weapon_copies_wanted": None,
    })


def _extraction_response(**overrides):
    payload = {
        "has_event_sequence": True, "events": [],
        "pulls_remaining_stated": None, "additional_pulls_stated": None,
        "total_pulls_restated": None,
        "additional_character_copies_wanted": None, "additional_weapon_copies_wanted": None,
    }
    payload.update(overrides)
    return _response(_msg(content=json.dumps(payload)))


def _scenario_fields(**overrides):
    """The fields shared by the primary scenario and every additional_scenarios
    entry (session_state.py's _SCENARIO_FIELDS), defaulted like a question
    that states nothing beyond its events."""
    base = {
        "events": [], "pulls_remaining_stated": None, "additional_pulls_stated": None,
        "total_pulls_restated": None,
        "additional_character_copies_wanted": None, "additional_weapon_copies_wanted": None,
    }
    base.update(overrides)
    return base


def _branching_extraction_response(primary_label, primary_overrides, additional_scenarios):
    payload = {
        "has_event_sequence": True,
        "condition_label": primary_label,
        "additional_scenarios": additional_scenarios,
        **_scenario_fields(**primary_overrides),
    }
    return _response(_msg(content=json.dumps(payload)))


@pytest.fixture(autouse=True)
def _patch_key(monkeypatch):
    monkeypatch.setattr(advisor, "get_openai_api_key", lambda: "sk-test")
    monkeypatch.setattr(advisor, "get_model", lambda: "test-model")


def _fake_sim(**kwargs):
    # Stand-in for run_simulation_verbose so the tool executor is fast/deterministic.
    return {
        "success_rate": "80.00%",
        "initial_pulls": kwargs["total_pulls"],
        "desired_characters": 1,
        "desired_weapons": 1,
        "avg_leftover_pulls_on_success": 10,
        "most_common_failure_state": None,
    }


def _find_line(breakdown, label):
    return next(line for line in breakdown["lines"] if line["label"] == label)


def test_runs_tool_then_answers(monkeypatch):
    tc = _tool_call("call_1", "run_simulation",
                    json.dumps({"total_pulls": 160, "strategy": BASELINE_PARAMS["strategy"]}))
    fake = _FakeClient([
        _response(_msg(content=_no_sequence())),                 # extraction: pure hypothetical
        _response(_msg(content=None, tool_calls=[tc])),          # model asks to run the sim
        _response(_msg(content="With 160 pulls you're at 80%.")),  # model answers
    ])
    monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
    monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

    answer, runs, breakdown = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "what if I had 160 pulls?")

    assert answer == "With 160 pulls you're at 80%."
    # A pure hypothetical starts with no breakdown, but any tool call the
    # model actually makes is still surfaced as its own tracked pill line,
    # not just a claim in the prose.
    assert breakdown == {"status": "ok", "lines": [
        {"label": "Agent Run Cycle 1",
         "pills": [{"kind": "result", "value": "160 pulls → 80.00%", "color": "light_green",
                    "tooltip": "An additional scenario the AI chose to explore with a real simulation, beyond the primary answer above"}]},
    ]}
    assert len(fake.calls) == 3
    # The third request carried the tool result back to the model.
    roles = [m["role"] for m in fake.calls[2]["messages"]]
    assert "tool" in roles
    # The run the model made is surfaced for the UI receipts.
    assert len(runs) == 1
    assert runs[0]["total_pulls"] == 160
    assert runs[0]["success_rate"] == "80.00%"


def test_pure_strategy_question_has_no_event_sequence(monkeypatch):
    # "assuming my budget is fixed, how should I adjust my strategy" has no
    # actual pull history in it at all, this must still work exactly like
    # any other pure hypothetical, no event-sequence machinery engaged.
    fake = _FakeClient([
        _response(_msg(content=_no_sequence())),
        _response(_msg(content="Prioritize the character copy first.")),
    ])
    monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)

    answer, runs, breakdown = run_advisor(
        BASELINE_PARAMS, BASELINE_STATS,
        "Assuming my budget is fixed, how should I adjust my character and weapon copy strategy?",
    )
    assert answer == "Prioritize the character copy first."
    assert breakdown is None
    assert runs == []


def test_answers_without_tool_call(monkeypatch):
    fake = _FakeClient([
        _response(_msg(content=_no_sequence())),
        _response(_msg(content="You're already comfortable, save your pulls.")),
    ])
    monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)

    answer, runs, breakdown = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "should I even bother?")
    assert "save your pulls" in answer
    assert breakdown is None
    assert len(fake.calls) == 2
    assert runs == []                                 # no tool call -> no receipts


def test_tool_call_budget_is_capped(monkeypatch):
    tc = _tool_call("c", "run_simulation", json.dumps({"total_pulls": 200, "strategy": BASELINE_PARAMS["strategy"]}))
    # Extraction (pure hypothetical), then the model asks for a tool on every one
    # of the 3 allowed loops; the loop then stops and makes one final forced
    # (tool_choice=none) call for the answer.
    responses = [_response(_msg(content=_no_sequence()))]
    responses += [_response(_msg(content=None, tool_calls=[tc])) for _ in range(3)]
    responses.append(_response(_msg(content="Final forced answer.")))
    fake = _FakeClient(responses)
    monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
    monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

    answer, runs, breakdown = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "keep testing", max_tool_calls=3)
    assert answer == "Final forced answer."
    assert breakdown["status"] == "ok"
    # Each of the 3 tool loops is its own tracked Agent Run Cycle line.
    assert [l["label"] for l in breakdown["lines"]] == [
        "Agent Run Cycle 1", "Agent Run Cycle 2", "Agent Run Cycle 3",
    ]
    assert len(fake.calls) == 5                       # extraction + 3 tool loops + 1 forced final
    assert "tools" not in fake.calls[-1]              # final call offers no tool, so none can be called
    assert len(runs) == 3                             # each loop ran the tool once


class TestSessionStateIntegration:
    """The scenario from the bug reports: character won at 42 pity (11
    refunded), weapon lost at 31 pity (3 refunded), guarantee active. The
    reconciled state should drop the character phase entirely (goal met)
    and simulate only the weapon, from 0 pity, with the guarantee, not the
    original full 1C/1W goal."""

    def test_reduces_goal_and_nets_refunds_before_the_model_ever_sees_it(self, monkeypatch):
        # The model calls the tool with no overrides, so whatever it actually
        # simulates comes straight from the reconciled run_params fallback.
        # Numbers mirror the original bug report exactly: 100 total pulls, 42
        # pity minus 11 refunds on the character, 31 pulls minus 3 refunds on
        # the weapon, net used is 59, so 41 pulls should remain (not
        # 100-42-31=27, which is what a naive gross subtraction would give).
        params = {**BASELINE_PARAMS, "total_pulls": 100}
        stats = {**BASELINE_STATS, "initial_pulls": 100}

        events = [_event(1, "character", "win", 42, 11), _event(2, "weapon", "loss", 31, 3)]
        tc = _tool_call("c1", "run_simulation", json.dumps({}))
        fake = _FakeClient([
            _extraction_response(events=events),
            _response(_msg(content=None, tool_calls=[tc])),
            _response(_msg(content="With the guarantee your weapon odds are strong.")),
        ])
        seen_kwargs = {}

        def _spy_sim(**kwargs):
            seen_kwargs.update(kwargs)
            return _fake_sim(**kwargs)

        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _spy_sim)

        answer, runs, breakdown = run_advisor(
            params, stats,
            "I got the character at 42 pity with 11 refunds, then used 31 pulls on the "
            "weapon with 3 refunds and lost the 50/50, how likely am I with my remaining pulls?",
        )

        assert breakdown["status"] == "ok"
        char_run = _find_line(breakdown, "Character Run 1 (obtained 1 of 1)")
        assert [p["value"] for p in char_run["pills"]] == [100, "−", 42, "+", 11, "=", 69, "WIN"]
        # The actual simulation ran on just the weapon, from scratch, guaranteed,
        # with pulls netted of refunds, not the naive gross subtraction.
        assert seen_kwargs["strategy"] == [{"banner": "weapon", "copies": 1}]
        assert seen_kwargs["total_pulls"] == 41
        assert seen_kwargs["start_weapon_pity"] == 0
        assert seen_kwargs["start_weapon_guarantee"] is True
        assert answer == "With the guarantee your weapon odds are strong."

    def test_context_states_only_the_remaining_goal_not_the_original(self, monkeypatch):
        # Live bug report: the character was already won (guaranteed), but
        # the interpretation model's prose still described the character as
        # if it could still fail, because the context only ever told it the
        # *original* combined goal. The context sent to the model must now
        # spell out what actually still remains (the weapon only) and say
        # so explicitly, not leave the model to infer it.
        params = {**BASELINE_PARAMS, "total_pulls": 100}
        stats = {**BASELINE_STATS, "initial_pulls": 100}
        events = [_event(1, "character", "win", 30, 5), _event(2, "weapon", "loss", 50, 3)]
        fake = _FakeClient([
            _extraction_response(events=events),
            _response(_msg(content="Only the weapon is left, odds are decent.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

        run_advisor(params, stats, "I won the character but lost the weapon, how's it looking?")

        context = fake.calls[-1]["messages"][1]["content"]
        assert "what actually still remains to obtain, after the events above, is: 1 weapon" in context
        assert "must not be described as still needed, at risk, or a possible failure" in context

    def test_reconciled_scenario_is_simulated_before_the_model_gets_a_turn(self, monkeypatch):
        # Fourth real bug: even with explicit instructions, the model kept
        # re-deriving (and double-counting) the pull total itself. The fix
        # is to not depend on the model at all for the primary number: run
        # the reconciled scenario in code and hand the model an answer that
        # already exists, before its first turn. Here the model never calls
        # the tool at all, so if the receipt is present, it can only have
        # come from the deterministic pre-run, not from the model.
        params = {**BASELINE_PARAMS, "total_pulls": 100}
        stats = {**BASELINE_STATS, "initial_pulls": 100}

        events = [_event(1, "character", "win", 42, 11), _event(2, "weapon", "loss", 31, 3)]
        fake = _FakeClient([
            _extraction_response(events=events),
            _response(_msg(content="With the guarantee, 41 pulls gives strong odds.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

        answer, runs, breakdown = run_advisor(
            params, stats,
            "I got the character at 42 pity with 11 refunds, then used 31 pulls on the "
            "weapon with 3 refunds and lost the 50/50, how likely am I with my remaining pulls?",
        )

        assert len(runs) == 1                     # the pre-run, with no model tool call at all
        assert runs[0]["total_pulls"] == 41
        assert answer == "With the guarantee, 41 pulls gives strong odds."
        # Only 2 model calls total: the extraction, then the answer. The
        # model was never given a chance to pick its own total_pulls.
        assert len(fake.calls) == 2
        # And the "Simulated Result" line is present, built from the
        # deterministic pre-run, not asserted by the model.
        result_line = _find_line(breakdown, "Simulated Result")
        assert result_line["pills"][0]["value"] == "41 pulls → 80.00%"

    def test_unstated_refund_is_estimated_end_to_end_when_full_4star_chars(self, monkeypatch):
        # The question never mentions refunds at all; the extraction must
        # report refund_count null (not assume 0), and session_state.py's
        # formula fills it in since the baseline was simulated with
        # full_4star_chars=True.
        params = {**BASELINE_PARAMS, "total_pulls": 100, "full_4star_chars": True}
        stats = {**BASELINE_STATS, "initial_pulls": 100}

        events = [_event(1, "character", "win", 30, None)]
        fake = _FakeClient([
            _extraction_response(events=events),
            _response(_msg(content="Odds are decent from here.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

        _, _, breakdown = run_advisor(
            params, stats, "I won the character at 30 pity, how am I looking?",
        )

        char_line = _find_line(breakdown, "Character Run 1 (obtained 1 of 1)")
        refund_pill = char_line["pills"][4]
        # 0.1105 * 30 = 3.315 -> rounds to 3
        assert refund_pill["value"] == 3
        assert refund_pill["color"] == "light_green"
        assert "formula estimating the average number of refunds for 30 pulls" in refund_pill["tooltip"]

    def test_unreconcilable_sequence_declines_with_a_hardcoded_message(self, monkeypatch):
        # Per the follow-up requirement: when the event sequence still can't
        # be reconciled after one corrective retry, decline entirely and ask
        # for alternate input, do NOT fall back to guessing on the baseline.
        # The decline message is a hardcoded constant, not another model
        # call, since a model has repeatedly failed to relay instructions
        # like this reliably elsewhere in this app.
        events = [_event(1, "weapon", "loss", 50, 0)]
        fake = _FakeClient([
            # First extraction: stated remaining conflicts with the events.
            _extraction_response(events=events, pulls_remaining_stated=999),
            # Retry still conflicts.
            _extraction_response(events=events, pulls_remaining_stated=888),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)

        answer, runs, breakdown = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "confusing question")

        assert answer == PARSE_FAILURE_MESSAGE
        assert runs == []
        assert breakdown == {"status": "error",
                              "message": "Could not parse a clear sequence of pull events from this question."}
        # No tool call and no forced-answer call happened: exactly the 2
        # extraction attempts, then the hardcoded decline, nothing more.
        assert len(fake.calls) == 2

    def test_goal_already_complete_skips_simulation(self, monkeypatch):
        events = [_event(1, "character", "win", 20, 0), _event(2, "weapon", "win", 30, 0)]
        fake = _FakeClient([
            _extraction_response(events=events),
            _response(_msg(content="You already have everything you need.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)

        answer, runs, breakdown = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "am I done?")

        assert breakdown["status"] == "ok"
        status_line = _find_line(breakdown, "Goal Status")
        assert status_line["pills"][0]["value"] == "GOAL COMPLETE"
        assert status_line["pills"][0]["color"] == "green"
        assert runs == []                              # nothing was simulated
        assert answer == "You already have everything you need."
        assert "tools" not in fake.calls[-1]  # no tool offered for this path
        context = fake.calls[-1]["messages"][1]["content"]
        assert "nothing, everything above is already obtained" in context

    def test_expands_goal_and_adds_extra_pulls_on_top_of_remaining(self, monkeypatch):
        # Second real bug report: "another character copy" (goal expansion)
        # plus "+ around 45 from this half of the patch" (an addition on top
        # of whatever remains, not a restatement of the total). 100 total,
        # 42 pity minus 11 refunds and 31 pulls minus 3 refunds already spent
        # (net 59), so remaining is 41, plus the stated 45 more = 86. The
        # goal grows to 2 characters (1 already obtained, 1 more wanted) and
        # 1 weapon (guaranteed).
        params = {**BASELINE_PARAMS, "total_pulls": 100}
        stats = {**BASELINE_STATS, "initial_pulls": 100}

        events = [_event(1, "character", "win", 42, 11), _event(2, "weapon", "loss", 31, 3)]
        tc = _tool_call("c1", "run_simulation", json.dumps({}))
        fake = _FakeClient([
            _extraction_response(
                events=events, additional_pulls_stated=45, additional_character_copies_wanted=1,
            ),
            _response(_msg(content=None, tool_calls=[tc])),
            _response(_msg(content="With the extra pulls, both goals are within reach.")),
        ])
        seen_kwargs = {}

        def _spy_sim(**kwargs):
            seen_kwargs.update(kwargs)
            return _fake_sim(**kwargs)

        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _spy_sim)

        answer, runs, breakdown = run_advisor(
            params, stats,
            "I got the character at 42 pity with 11 refunds, then used 31 pulls on the "
            "weapon with 3 refunds and lost the 50/50. How likely am I to get the "
            "lightcone and another character copy with my remaining pulls plus "
            "around 45 more from this half of the patch?",
        )

        assert breakdown["status"] == "ok"
        additional_line = _find_line(breakdown, "Additional Pulls Mentioned")
        assert additional_line["pills"][-1]["value"] == 86
        goal_line = _find_line(breakdown, "Current Goal")
        assert goal_line["pills"][0]["value"] == "1 character and 1 weapon"
        assert seen_kwargs["total_pulls"] == 86
        # 1 more character copy wanted (the one already obtained is dropped),
        # plus the still-needed weapon.
        assert {"banner": "char", "copies": 1} in seen_kwargs["strategy"]
        assert {"banner": "weapon", "copies": 1} in seen_kwargs["strategy"]
        assert seen_kwargs["start_weapon_guarantee"] is True
        assert answer == "With the extra pulls, both goals are within reach."

    def test_multiple_losses_before_a_win_end_to_end(self, monkeypatch):
        # Exactly what the old flat per-banner schema could not express at
        # all: two losses on the same banner before the eventual win.
        params = {**BASELINE_PARAMS, "total_pulls": 200}
        stats = {**BASELINE_STATS, "initial_pulls": 200}

        events = [
            _event(1, "weapon", "loss", 50, 2),
            _event(2, "weapon", "loss", 60, 1),
            _event(3, "weapon", "win", 10, 0),
        ]
        fake = _FakeClient([
            _extraction_response(events=events),
            _response(_msg(content="You got there on the third try.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

        answer, runs, breakdown = run_advisor(
            params, stats,
            "I lost the weapon 50/50 at 50 pity with 2 refunds, lost again at 60 pity with "
            "1 refund, then finally won at 10 pity with no refunds. How am I doing on the "
            "character now?",
        )

        assert breakdown["status"] == "ok"
        assert _find_line(breakdown, "Weapon Run 1 (obtained 0 of 1)")
        assert _find_line(breakdown, "Weapon Run 2 (obtained 0 of 1)")
        assert _find_line(breakdown, "Weapon Run 3 (obtained 1 of 1)")
        assert _find_line(breakdown, "Character Run 1 (obtained 0 of 1)")  # in progress, never mentioned
        assert answer == "You got there on the third try."

    def test_exploratory_call_on_reconciled_scenario_is_locked_and_tracked(self, monkeypatch):
        # Live-testing turned up two related bugs on top of the reconciled
        # scenario: the model citing a percentage with no tool call behind
        # it at all, and (when it does call the tool) inventing an
        # unrequested pity override. This exercises the fix for both: the
        # exploratory call's invented start_weapon_pity is ignored, and the
        # call itself becomes its own tracked "Agent Run Cycle" pill line
        # appended after the guaranteed "Simulated Result" line.
        params = {**BASELINE_PARAMS, "total_pulls": 100}
        stats = {**BASELINE_STATS, "initial_pulls": 100}

        events = [_event(1, "character", "win", 42, 11), _event(2, "weapon", "loss", 31, 3)]
        # The model explores a further scenario but also fabricates a pity
        # override nobody asked for.
        tc = _tool_call("c2", "run_simulation", json.dumps(
            {"total_pulls": 103, "start_weapon_pity": 53, "start_weapon_guarantee": False},
        ))
        fake = _FakeClient([
            _extraction_response(events=events),
            _response(_msg(content=None, tool_calls=[tc])),
            _response(_msg(content="Even exploring further, the odds stay similar.")),
        ])
        seen_kwargs_per_call = []

        def _spy_sim(**kwargs):
            seen_kwargs_per_call.append(kwargs)
            return _fake_sim(**kwargs)

        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _spy_sim)

        answer, runs, breakdown = run_advisor(
            params, stats,
            "I got the character at 42 pity with 11 refunds, then used 31 pulls on the "
            "weapon with 3 refunds and lost the 50/50. What if I kept going a while longer?",
        )

        # Two simulations ran: the guaranteed pre-run, then the exploratory
        # one, with its invented pity/guarantee ignored (locked).
        assert len(seen_kwargs_per_call) == 2
        exploratory_kwargs = seen_kwargs_per_call[1]
        assert exploratory_kwargs["start_weapon_pity"] == 0            # not the model's invented 53
        assert exploratory_kwargs["start_weapon_guarantee"] is True    # the real, verified guarantee
        assert exploratory_kwargs["total_pulls"] == 103                # total_pulls itself is still respected

        labels = [line["label"] for line in breakdown["lines"]]
        assert "Simulated Result" in labels
        assert "Agent Run Cycle 1" in labels
        assert labels.index("Agent Run Cycle 1") > labels.index("Simulated Result")
        assert answer == "Even exploring further, the odds stay similar."

    def test_every_call_uses_the_low_advisor_temperature(self, monkeypatch):
        events = [_event(1, "character", "win", 42, 11)]
        fake = _FakeClient([
            _extraction_response(events=events, pulls_remaining_stated=50),
            _response(_msg(content="Grounded answer.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

        run_advisor(BASELINE_PARAMS, BASELINE_STATS, "some question")

        assert len(fake.calls) == 2
        for call in fake.calls:
            assert call["temperature"] == advisor.ADVISOR_TEMPERATURE


class TestScenariosFromExtraction:
    """Pure unit tests for the helper that flattens one extraction result
    (primary scenario fields plus an optional additional_scenarios array)
    into a list of scenario dicts, each shaped like reconcile()'s input."""

    def test_no_additional_scenarios_returns_just_the_primary(self):
        extracted = {
            "condition_label": None, "additional_scenarios": None,
            **_scenario_fields(events=[_event(1, "character", "win", 30, 5)]),
        }
        scenarios = _scenarios_from_extraction(extracted)
        assert len(scenarios) == 1
        assert scenarios[0]["has_event_sequence"] is True
        assert scenarios[0]["condition_label"] is None
        assert scenarios[0]["events"] == extracted["events"]

    def test_additional_scenarios_are_appended_after_the_primary(self):
        extracted = {
            "condition_label": "If I win",
            "additional_scenarios": [
                {"condition_label": "If I lose", **_scenario_fields(events=[_event(1, "character", "loss", 75, 3)])},
            ],
            **_scenario_fields(events=[_event(1, "character", "win", 30, 5)]),
        }
        scenarios = _scenarios_from_extraction(extracted)
        assert len(scenarios) == 2
        assert scenarios[0]["condition_label"] == "If I win"
        assert scenarios[1]["condition_label"] == "If I lose"
        assert scenarios[1]["has_event_sequence"] is True
        assert scenarios[1]["events"][0]["outcome"] == "loss"

    def test_scenarios_are_capped_at_max_scenarios(self):
        extras = [
            {"condition_label": f"Branch {i}", **_scenario_fields()}
            for i in range(MAX_SCENARIOS + 5)
        ]
        extracted = {
            "condition_label": "Primary", "additional_scenarios": extras,
            **_scenario_fields(),
        }
        scenarios = _scenarios_from_extraction(extracted)
        assert len(scenarios) == MAX_SCENARIOS


class TestBranchingScenarios:
    """The question describes two or more mutually exclusive futures ('if I
    win... if I lose...'), reconciled from the same starting point."""

    def _fake_sim_by_pulls(self, **kwargs):
        # Encode total_pulls into the rate so a test can tell which branch's
        # simulation produced which number.
        return {
            "success_rate": f"{kwargs['total_pulls']}.00%",
            "initial_pulls": kwargs["total_pulls"],
            "desired_characters": 1,
            "desired_weapons": 1,
            "avg_leftover_pulls_on_success": 10,
            "most_common_failure_state": None,
        }

    def test_two_branches_each_simulated_and_grouped_separately(self, monkeypatch):
        params = {**BASELINE_PARAMS, "total_pulls": 100}
        stats = {**BASELINE_STATS, "initial_pulls": 100}

        fake = _FakeClient([
            _branching_extraction_response(
                "If I win the character pull around pity 30",
                {"events": [_event(1, "character", "win", 30, 5)]},
                [{
                    "condition_label": "If I lose the character pull around pity 75",
                    **_scenario_fields(events=[_event(1, "character", "loss", 75, 3)]),
                }],
            ),
            _response(_msg(content="Win branch is strong, lose branch is a stretch.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", self._fake_sim_by_pulls)

        answer, runs, breakdown = run_advisor(
            params, stats,
            "If I win the character pull around pity 30, what are my odds for the weapon? "
            "If I lose around pity 75, what are my odds instead?",
        )

        assert answer == "Win branch is strong, lose branch is a stretch."
        assert breakdown["status"] == "ok"
        assert "groups" in breakdown and "lines" not in breakdown
        labels = [g["label"] for g in breakdown["groups"]]
        assert labels == [
            "If I win the character pull around pity 30",
            "If I lose the character pull around pity 75",
        ]

        # Win branch: 100 - 30 + 5 = 75 pulls remain, weapon only.
        win_group = breakdown["groups"][0]
        win_result = next(l for l in win_group["lines"] if l["label"] == "Simulated Result")
        assert win_result["pills"][0]["value"] == "75 pulls → 75.00%"

        # Lose branch: 100 - 75 + 3 = 28 pulls remain, character (guaranteed) + weapon.
        lose_group = breakdown["groups"][1]
        lose_result = next(l for l in lose_group["lines"] if l["label"] == "Simulated Result")
        assert lose_result["pills"][0]["value"] == "28 pulls → 28.00%"

        # Both branches were actually simulated deterministically, not left to the model.
        assert len(runs) == 2
        assert {r["total_pulls"] for r in runs} == {75, 28}

        # Only 2 model calls total: the extraction, then one consolidated
        # answer. No exploratory tool loop for a branching question.
        assert len(fake.calls) == 2
        assert "tools" not in fake.calls[-1]

    def test_branch_that_is_already_complete_is_not_simulated(self, monkeypatch):
        params = {**BASELINE_PARAMS, "total_pulls": 100}
        stats = {**BASELINE_STATS, "initial_pulls": 100}

        seen_calls = []

        def _spy_sim(**kwargs):
            seen_calls.append(kwargs["total_pulls"])
            return self._fake_sim_by_pulls(**kwargs)

        fake = _FakeClient([
            _branching_extraction_response(
                "If I win the character",
                {"events": [_event(1, "character", "win", 20, 0), _event(2, "weapon", "win", 30, 0)]},
                [{
                    "condition_label": "If I lose the character",
                    **_scenario_fields(events=[_event(1, "character", "loss", 90, 0)]),
                }],
            ),
            _response(_msg(content="First branch is already done, second is rough.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _spy_sim)

        answer, runs, breakdown = run_advisor(
            params, stats, "If I win everything on my first tries, am I done? If I lose the character, then what?",
        )

        win_group = breakdown["groups"][0]
        status_line = next(l for l in win_group["lines"] if l["label"] == "Goal Status")
        assert status_line["pills"][0]["value"] == "GOAL COMPLETE"
        # No "Simulated Result" line for the already-complete branch.
        assert not any(l["label"] == "Simulated Result" for l in win_group["lines"])

        # Only the second (still-incomplete) branch was actually simulated:
        # 100 - 90 + 0 refunds = 10 pulls remain for it.
        assert seen_calls == [10]
        assert len(runs) == 1

    def test_scenario_failure_includes_its_label_in_the_error(self, monkeypatch):
        # The second branch's pulls_remaining_stated conflicts with its own
        # events; the label must be identifiable in the surfaced error so a
        # multi-branch failure isn't ambiguous about which branch broke.
        fake = _FakeClient([
            _branching_extraction_response(
                "If I win",
                {"events": [_event(1, "character", "win", 20, 0)]},
                [{
                    "condition_label": "If I lose",
                    **_scenario_fields(events=[_event(1, "character", "loss", 30, 0)], pulls_remaining_stated=999),
                }],
            ),
            # Retry also fails the same way.
            _branching_extraction_response(
                "If I win",
                {"events": [_event(1, "character", "win", 20, 0)]},
                [{
                    "condition_label": "If I lose",
                    **_scenario_fields(events=[_event(1, "character", "loss", 30, 0)], pulls_remaining_stated=999),
                }],
            ),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)

        answer, runs, breakdown = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "confusing branching question")

        assert answer == PARSE_FAILURE_MESSAGE
        assert breakdown["status"] == "error"


class TestToolExecutor:
    def test_valid_run(self, monkeypatch):
        monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)
        result = _run_tool({"total_pulls": 200}, BASELINE_PARAMS)
        assert result["success_rate"] == "80.00%"
        assert result["total_pulls"] == 200          # baseline strategy reused

    def test_invalid_strategy_returns_error(self):
        result = _run_tool({"strategy": [{"banner": "relic", "copies": 1}]}, BASELINE_PARAMS)
        assert "error" in result

    def test_invalid_total_pulls_returns_error(self):
        result = _run_tool({"total_pulls": 0}, BASELINE_PARAMS)
        assert "error" in result

    def test_lock_start_state_ignores_model_supplied_pity_and_guarantee(self, monkeypatch):
        seen_kwargs = {}

        def _spy_sim(**kwargs):
            seen_kwargs.update(kwargs)
            return _fake_sim(**kwargs)

        monkeypatch.setattr(advisor, "run_simulation_verbose", _spy_sim)
        _run_tool(
            {"total_pulls": 103, "start_weapon_pity": 53, "start_weapon_guarantee": False},
            BASELINE_PARAMS, lock_start_state=True,
        )
        # The model's invented pity/guarantee are ignored entirely; the
        # verified baseline values are used regardless of what was passed.
        assert seen_kwargs["start_weapon_pity"] == BASELINE_PARAMS["start_weapon_pity"]
        assert seen_kwargs["start_weapon_guarantee"] == BASELINE_PARAMS["start_weapon_guarantee"]
        # total_pulls is still respected: only pity/guarantee are locked.
        assert seen_kwargs["total_pulls"] == 103

    def test_without_lock_start_state_overrides_are_respected(self, monkeypatch):
        seen_kwargs = {}

        def _spy_sim(**kwargs):
            seen_kwargs.update(kwargs)
            return _fake_sim(**kwargs)

        monkeypatch.setattr(advisor, "run_simulation_verbose", _spy_sim)
        _run_tool({"start_weapon_pity": 53}, BASELINE_PARAMS, lock_start_state=False)
        assert seen_kwargs["start_weapon_pity"] == 53

    def test_lock_start_state_also_ignores_model_supplied_strategy(self, monkeypatch):
        # Live bug report: given a "restated goal" of e.g. 2 characters
        # (1 obtained + 1 remaining), an exploratory call re-derived that as
        # "2 fresh copies", re-simulating a goal from scratch instead of the
        # already-reconciled remaining amount, and produced a contradictory,
        # much lower number that the answer then treated as authoritative.
        # Copy counts must be locked exactly like pity/guarantee.
        seen_kwargs = {}

        def _spy_sim(**kwargs):
            seen_kwargs.update(kwargs)
            return _fake_sim(**kwargs)

        monkeypatch.setattr(advisor, "run_simulation_verbose", _spy_sim)
        _run_tool(
            {"strategy": [{"banner": "char", "copies": 2}, {"banner": "weapon", "copies": 1}]},
            BASELINE_PARAMS, lock_start_state=True,
        )
        assert seen_kwargs["strategy"] == BASELINE_PARAMS["strategy"]


class TestValidateStrategy:
    def test_ok(self):
        assert _validate_strategy([{"banner": "char", "copies": 2}]) is None

    def test_empty(self):
        assert _validate_strategy([]) is not None

    def test_bad_copies(self):
        assert _validate_strategy([{"banner": "char", "copies": 0}]) is not None

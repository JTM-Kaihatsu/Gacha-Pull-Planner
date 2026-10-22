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
    """The top-level fields of a scenario (advisor.py's _SCENARIO_FIELDS),
    defaulted like a question that states nothing beyond its events."""
    base = {
        "events": [], "pulls_remaining_stated": None, "additional_pulls_stated": None,
        "total_pulls_restated": None,
        "additional_character_copies_wanted": None, "additional_weapon_copies_wanted": None,
    }
    base.update(overrides)
    return base


def _branch(label, assumed_events, **overrides):
    branch = {
        "condition_label": label, "assumed_events": assumed_events,
        "pulls_remaining_stated": None, "additional_pulls_stated": None, "reserve_pulls": None,
        "additional_character_copies_wanted": None, "additional_weapon_copies_wanted": None,
        "characters_wanted_total": None, "weapons_wanted_total": None, "plan": None,
        "stops_pulling": False,
    }
    branch.update(overrides)
    return branch


def _branching_extraction_response(shared_overrides, branches):
    payload = {"has_event_sequence": True, "branches": branches, **_scenario_fields(**shared_overrides)}
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
        # The spending tiers (see TestSpendingTiers) are their own further
        # simulations layered on top of the pre-run this test is about;
        # neutralized here so len(runs) still isolates just the pre-run.
        monkeypatch.setattr(advisor, "_spending_tiers", lambda *a, **k: ([], [], []))

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
        # And the "AI Agent Simulated Result 1" line is present, built
        # from the deterministic pre-run, not asserted by the model.
        result_line = _find_line(breakdown, "AI Agent Simulated Result 1")
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
        # appended after the guaranteed "AI Agent Simulated Result 1" line.
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
        # Neutralized so the spending tiers' own further simulations (see
        # TestSpendingTiers) don't shift the indices this test relies on.
        monkeypatch.setattr(advisor, "_spending_tiers", lambda *a, **k: ([], [], []))

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
        assert "AI Agent Simulated Result 1" in labels
        assert "Agent Run Cycle 1" in labels
        assert labels.index("Agent Run Cycle 1") > labels.index("AI Agent Simulated Result 1")
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


def _ps_event(order, banner, outcome, pulls_spent, refunds=None):
    """An event reported as pulls SPENT this run ('after 60 pulls'), not an
    absolute pity counter value. pity_is_absolute must be explicit here
    (unlike _event, which relies on the True default) since these tests
    are specifically about the pulls-spent branch of pity carryover."""
    return {"event_order": order, "banner_type": banner, "outcome": outcome,
            "pity_at_outcome": pulls_spent, "pity_is_absolute": False, "refund_count": refunds}


def _abs_event(order, banner, outcome, pity, refunds=None):
    return {"event_order": order, "banner_type": banner, "outcome": outcome,
            "pity_at_outcome": pity, "pity_is_absolute": True, "refund_count": refunds}


class TestScenariosFromExtraction:
    """One extraction splits into the shared part and its branches."""

    PLAN = [{"step_order": 1, "banner_type": "weapon", "copies": 1, "on_loss": "stop",
             "max_pulls": None, "stop_at_pity": None, "on_cap": None}]

    def test_no_branches_returns_the_whole_question_as_shared(self):
        extracted = {"has_event_sequence": True, "branches": None, "plan": self.PLAN,
                     **_scenario_fields(events=[_event(1, "character", "win", 30, 5)])}
        shared, branches = _scenarios_from_extraction(extracted)
        assert branches == []
        assert shared["events"] == [_event(1, "character", "win", 30, 5)]
        assert shared["plan"] == self.PLAN

    def test_branches_keep_only_their_own_assumed_outcomes(self):
        extracted = {
            "has_event_sequence": True, "plan": self.PLAN,
            **_scenario_fields(events=[_event(1, "weapon", "loss", 40, 2)], additional_pulls_stated=120),
            "branches": [
                _branch("If I win", [_event(1, "character", "win", 30, 3)]),
                _branch("If I lose", [_event(1, "character", "loss", 75, 8)], plan=self.PLAN),
            ],
        }
        shared, branches = _scenarios_from_extraction(extracted)
        # A plan only ever belongs to a branch once there are branches.
        assert shared["plan"] is None
        assert shared["events"] == [_event(1, "weapon", "loss", 40, 2)]
        assert [label for label, _ in branches] == ["If I win", "If I lose"]
        assert branches[0][1]["events"] == [_event(1, "character", "win", 30, 3)]
        assert branches[0][1]["additional_pulls_stated"] is None
        assert branches[1][1]["plan"] == self.PLAN

    def test_branches_are_capped_at_max_scenarios(self):
        extracted = {"has_event_sequence": True, "plan": None, **_scenario_fields(),
                     "branches": [_branch(f"Branch {i}", []) for i in range(MAX_SCENARIOS + 5)]}
        _, branches = _scenarios_from_extraction(extracted)
        assert len(branches) == MAX_SCENARIOS


class TestBranchTree:
    """Shared facts reconcile once; each branch continues from them."""

    def test_shared_events_show_once_and_each_branch_continues_from_them(self):
        extracted = {
            "has_event_sequence": True, "plan": None,
            # Already happened: lost the weapon at 40 pity, 2 refunds.
            **_scenario_fields(events=[_event(1, "weapon", "loss", 40, 2)]),
            "branches": [
                _branch("If I win the character at 30", [_event(1, "character", "win", 30, 3)]),
                _branch("If I lose the character at 60", [_event(1, "character", "loss", 60, 7)]),
            ],
        }
        result = advisor._reconcile_scenario_list(extracted, BASELINE_PARAMS, BASELINE_STATS)

        assert result["ok"] and result["branched"]
        assert [l["label"] for l in result["shared_lines"]] == ["Weapon Run 1 (obtained 0 of 1)"]
        win, lose = result["scenarios"]
        assert (win["letter"], lose["letter"]) == ("A", "B")
        # 120 - 40 + 2 = 82 shared, then each branch from 82, never repeating it.
        assert [p["value"] for p in win["lines"][0]["pills"]][:7] == [82, "−", 30, "+", 3, "=", 55]
        assert [p["value"] for p in lose["lines"][0]["pills"]][:7] == [82, "−", 60, "+", 7, "=", 29]
        assert not any(l["label"].startswith("Weapon Run 1") for l in win["lines"] + lose["lines"])
        # The shared loss left the weapon guaranteed in both branches.
        assert win["start_weapon_guarantee"] and lose["start_weapon_guarantee"]
        assert lose["start_char_guarantee"] and not win["start_char_guarantee"]
        assert win["remaining_characters"] == 0 and win["remaining_weapons"] == 1

    def test_runs_and_copies_keep_counting_on_from_the_shared_part(self):
        # Already won C0 at 30 (3 refunds); a branch assumes the C1 copy at 50.
        stats = {**BASELINE_STATS, "desired_characters": 2}
        extracted = {
            "has_event_sequence": True, "plan": None,
            **_scenario_fields(events=[_event(1, "character", "win", 30, 3)]),
            "branches": [
                _branch("If I win C1 at 50", [_event(1, "character", "win", 50, None)]),
                _branch("If I lose at 50", [_event(1, "character", "loss", 50, None)]),
            ],
        }
        result = advisor._reconcile_scenario_list(extracted, BASELINE_PARAMS, stats)
        win = result["scenarios"][0]
        assert win["lines"][0]["label"] == "Character Run 2 (obtained 2 of 2)"
        # A repeat copy still gets its dupe refund pill on top of the estimate.
        assert [p["value"] for p in win["lines"][0]["pills"]][:9] == [93, "−", 50, "+", 6, "+", 2, "=", 51]
        assert win["remaining_characters"] == 0

    def test_a_conflict_inside_a_branch_is_keyed_and_labeled_by_branch(self):
        params = {**BASELINE_PARAMS, "start_weapon_pity": 22}
        extracted = {
            "has_event_sequence": True, "plan": None, **_scenario_fields(),
            "branches": [
                _branch("If I win the weapon after 60 pulls", [_ps_event(1, "weapon", "win", 60)]),
                _branch("If I lose the weapon after 30 pulls", [_ps_event(1, "weapon", "loss", 30)]),
            ],
        }
        result = advisor._reconcile_scenario_list(extracted, params, BASELINE_STATS)
        assert result["conflict"]
        (conflict,) = result["conflicts"]
        assert conflict["conflict_key"] == "branch0:pity:weapon:1"
        assert conflict["header"] == "Scenario A, For Weapon Attempt 1 of 1"
        assert conflict["branch_label"] == "If I win the weapon after 60 pulls"

    def test_a_budget_conflict_in_a_branch_names_the_shared_budget(self):
        extracted = {
            "has_event_sequence": True, "plan": None,
            **_scenario_fields(events=[_event(1, "weapon", "win", 70, 0)]),
            "branches": [_branch("If I lose the character at 80", [_event(1, "character", "loss", 80, 0)]),
                         _branch("If I win the character at 20", [_event(1, "character", "win", 20, 0)])],
        }
        result = advisor._reconcile_scenario_list(extracted, BASELINE_PARAMS, BASELINE_STATS)
        (conflict,) = result["conflicts"]
        assert conflict["conflict_key"] == "branch0:budget"
        assert "the events shared by every scenario above leave only 50 pulls" in conflict["question"]
        assert [l["label"] for l in result["lines"]] == ["Weapon Run 1 (obtained 1 of 1)"]

    def test_a_conflict_in_the_shared_part_stays_unprefixed(self):
        params = {**BASELINE_PARAMS, "start_weapon_pity": 22}
        extracted = {
            "has_event_sequence": True, "plan": None,
            **_scenario_fields(events=[_ps_event(1, "weapon", "win", 60)]),
            "branches": [_branch("If I win", [_event(1, "character", "win", 30, 3)]),
                         _branch("If I lose", [_event(1, "character", "loss", 30, 3)])],
        }
        result = advisor._reconcile_scenario_list(extracted, params, BASELINE_STATS)
        assert [c["conflict_key"] for c in result["conflicts"]] == ["pity:weapon:1"]


class TestBranchingScenarios:
    """The question describes two or more mutually exclusive futures it
    assumes ('if I win... if I lose...'), continuing from shared facts."""

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
        fake = _FakeClient([
            _branching_extraction_response({}, [
                _branch("If I win the character pull around pity 30", [_event(1, "character", "win", 30, 5)]),
                _branch("If I lose the character pull around pity 75", [_event(1, "character", "loss", 75, 3)]),
            ]),
            _response(_msg(content="Win branch is strong, lose branch is a stretch.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", self._fake_sim_by_pulls)

        answer, runs, breakdown = run_advisor(
            BASELINE_PARAMS, BASELINE_STATS,
            "If I win the character pull around pity 30, what are my odds for the weapon? "
            "If I lose around pity 75, what are my odds instead?",
        )

        assert answer == "Win branch is strong, lose branch is a stretch."
        assert breakdown["status"] == "ok"
        assert breakdown["lines"] == []
        assert [g["label"] for g in breakdown["groups"]] == [
            "Scenario A: If I win the character pull around pity 30",
            "Scenario B: If I lose the character pull around pity 75",
        ]

        # Win branch: 120 - 30 + 5 = 95 pulls remain, weapon only.
        win_result = next(l for l in breakdown["groups"][0]["lines"] if l["label"] == "Simulated Result")
        assert win_result["pills"][0]["value"] == "95 pulls → 95.00%"
        # Lose branch: 120 - 75 + 3 = 48 pulls remain, character (guaranteed) + weapon.
        lose_result = next(l for l in breakdown["groups"][1]["lines"] if l["label"] == "Simulated Result")
        assert lose_result["pills"][0]["value"] == "48 pulls → 48.00%"

        assert {r["total_pulls"] for r in runs} == {95, 48}
        # The extraction, then one consolidated answer, no tool loop.
        assert len(fake.calls) == 2
        assert "tools" not in fake.calls[-1]
        assert "has no chance of happening to cite" in fake.calls[-1]["messages"][-1]["content"]

    def test_branch_that_is_already_complete_is_not_simulated(self, monkeypatch):
        seen_calls = []

        def _spy_sim(**kwargs):
            seen_calls.append(kwargs["total_pulls"])
            return self._fake_sim_by_pulls(**kwargs)

        fake = _FakeClient([
            _branching_extraction_response({}, [
                _branch("If I win the character",
                        [_event(1, "character", "win", 20, 0), _event(2, "weapon", "win", 30, 0)]),
                _branch("If I lose the character", [_event(1, "character", "loss", 90, 0)]),
            ]),
            _response(_msg(content="First branch is already done, second is rough.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _spy_sim)

        _, runs, breakdown = run_advisor(
            BASELINE_PARAMS, BASELINE_STATS,
            "If I win everything on my first tries, am I done? If I lose the character, then what?",
        )

        win_group = breakdown["groups"][0]
        status_line = next(l for l in win_group["lines"] if l["label"] == "Goal Status")
        assert status_line["pills"][0]["value"] == "GOAL COMPLETE"
        assert not any(l["label"] == "Simulated Result" for l in win_group["lines"])
        # Only the still-incomplete branch ran: 120 - 90 + 0 = 30.
        assert seen_calls == [30]
        assert len(runs) == 1

    def test_a_branch_with_a_plan_nests_its_routes(self, monkeypatch):
        plan = [{"step_order": 1, "banner_type": "weapon", "copies": 1, "on_loss": "stop",
                 "max_pulls": None, "stop_at_pity": None, "on_cap": None}]
        plan_calls = []

        def _fake_plan(**kwargs):
            plan_calls.append(kwargs)
            node = lambda count, left: {"count": count, "avg_leftover": left}
            return {"trials": 4000, "nodes": {(): node(4000, kwargs["total_pulls"]),
                                              ("won",): node(3000, 40.0), ("lost",): node(1000, 42.0)},
                    "leaves": {("won",): 3000, ("lost",): 1000}}

        fake = _FakeClient([
            _branching_extraction_response({}, [
                _branch("If I win the character at 30", [_event(1, "character", "win", 30, 3)], plan=plan),
                _branch("If I lose the character at 75", [_event(1, "character", "loss", 75, 8)]),
            ]),
            _response(_msg(content="Both branches explained.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "simulate_plan", _fake_plan)
        monkeypatch.setattr(advisor, "run_simulation_verbose", self._fake_sim_by_pulls)

        _, _, breakdown = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "branch with a plan")

        win_group, lose_group = breakdown["groups"]
        assert [g["label"] for g in win_group["groups"]] == [
            "Route 1: Weapon won", "Route 2: Weapon lost, stop",
        ]
        assert plan_calls[0]["total_pulls"] == 93   # 120 - 30 + 3
        assert "groups" not in lose_group
        prompt = fake.calls[-1]["messages"][-1]["content"]
        assert '"Route 1: Weapon won": chance 75.00%' in prompt

    def test_scenario_failure_falls_back_to_the_parse_failure_message(self, monkeypatch):
        # The second branch's pulls_remaining_stated conflicts with its own
        # events, on both the first try and the retry.
        bad = lambda: _branching_extraction_response({}, [
            _branch("If I win", [_event(1, "character", "win", 20, 0)]),
            _branch("If I lose", [_event(1, "character", "loss", 30, 0)], pulls_remaining_stated=999),
        ])
        fake = _FakeClient([bad(), bad()])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)

        answer, runs, breakdown = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "confusing branching question")

        assert answer == PARSE_FAILURE_MESSAGE
        assert breakdown["status"] == "error"
        # The retry was told which branch broke.
        assert "If I lose:" in fake.calls[1]["messages"][-1]["content"]


def _tier_fake_sim(guaranteed_at, **overrides):
    """A run_simulation_verbose stand-in whose success rate scales linearly
    with total_pulls up to guaranteed_at (where it hits ~100%), mirroring
    the real hard-pity guarantee this test simulation itself models: fast,
    deterministic, and internally consistent with _guaranteed_pulls_for_banner
    so a test can predict exactly where the moderate/whale tiers should land."""
    def _fake(**kw):
        pct = min(99.99, kw["total_pulls"] / guaranteed_at * 100)
        base = {
            "success_rate": f"{pct:.2f}%", "initial_pulls": kw["total_pulls"],
            "desired_characters": 0, "desired_weapons": 1,
            "avg_leftover_pulls_on_success": 10, "most_common_failure_state": None,
        }
        base.update(overrides)
        return base
    return _fake


class TestSpendingTiers:
    """The three pre-computed spending tiers (F2P / spend-if-you-really-
    want-it / guaranteed) advisor.py hands the interpretation model instead
    of letting it invent its own follow-up numbers."""

    def test_guaranteed_pulls_for_banner_without_existing_guarantee(self):
        # Worst case: lose the 50/50 at hard pity, then a guaranteed win
        # takes up to another hard pity's worth of pulls.
        assert advisor._guaranteed_pulls_for_banner(80, 0, False, 1) == 160
        assert advisor._guaranteed_pulls_for_banner(80, 20, False, 1) == 60 + 80

    def test_guaranteed_pulls_for_banner_with_existing_guarantee(self):
        # Already guaranteed: no 50/50 risk at all, just reach hard pity.
        assert advisor._guaranteed_pulls_for_banner(90, 30, True, 1) == 60

    def test_guaranteed_pulls_for_banner_multiple_copies(self):
        # The first copy uses the real starting state; a win always resets
        # both pity and guarantee, so every copy after it starts fresh.
        assert advisor._guaranteed_pulls_for_banner(80, 0, False, 2) == 160 + 160

    def test_guaranteed_pulls_for_banner_zero_copies_needed(self):
        assert advisor._guaranteed_pulls_for_banner(80, 50, False, 0) == 0

    def test_guaranteed_scenario_pulls_sums_both_banners_independently(self):
        scenario = {
            "start_char_pity": 0, "start_char_guarantee": False, "remaining_characters": 1,
            "start_weapon_pity": 0, "start_weapon_guarantee": True, "remaining_weapons": 1,
        }
        # Character (not guaranteed): (90-0)+90 = 180. Weapon (guaranteed): 80-0 = 80.
        assert advisor._guaranteed_scenario_pulls(scenario, BASELINE_PARAMS) == 260

    def test_extra_pulls_for_target_rate_is_zero_when_baseline_already_clears_it(self, monkeypatch):
        monkeypatch.setattr(advisor, "run_simulation_verbose",
                             lambda **kw: {"success_rate": "90.00%"})
        run_params = {**BASELINE_PARAMS, "total_pulls": 100}
        assert advisor._extra_pulls_for_target_rate(run_params, 0.85, extra_cap=200) == 0

    def test_extra_pulls_for_target_rate_is_zero_when_cap_is_not_positive(self, monkeypatch):
        calls = []
        monkeypatch.setattr(advisor, "run_simulation_verbose",
                             lambda **kw: calls.append(kw["total_pulls"]) or {"success_rate": "10.00%"})
        run_params = {**BASELINE_PARAMS, "total_pulls": 100}
        assert advisor._extra_pulls_for_target_rate(run_params, 0.85, extra_cap=0) == 0
        assert calls == []   # short-circuits: a non-positive cap can't be searched at all

    def test_extra_pulls_for_target_rate_converges_on_the_true_crossing_point(self, monkeypatch):
        # Linear model: success rate equals total_pulls directly (capped at
        # 100), so 85% requires exactly 85 total pulls, 15 more than the
        # 70-pull baseline. The search should land on exactly that.
        monkeypatch.setattr(advisor, "run_simulation_verbose",
                             lambda **kw: {"success_rate": f"{min(99.99, kw['total_pulls']):.2f}%"})
        run_params = {**BASELINE_PARAMS, "total_pulls": 70}
        assert advisor._extra_pulls_for_target_rate(run_params, 0.85, extra_cap=100) == 15

    def test_spending_tiers_includes_moderate_and_whale_when_baseline_is_low(self, monkeypatch):
        # Weapon only, no existing pity/guarantee, one copy needed:
        # guaranteed pulls = (80-0)+80 = 160 via hard pity.
        monkeypatch.setattr(advisor, "run_simulation_verbose", _tier_fake_sim(160))
        scenario = {
            "start_char_pity": 0, "start_char_guarantee": False, "remaining_characters": 0,
            "start_weapon_pity": 0, "start_weapon_guarantee": False, "remaining_weapons": 1,
        }
        run_params = {**BASELINE_PARAMS, "total_pulls": 50, "strategy": [{"banner": "weapon", "copies": 1}]}
        primary_result = {"success_rate": "31.25%"}

        tiers, runs, comparisons = advisor._spending_tiers(scenario, run_params, BASELINE_PARAMS, primary_result)

        assert [t["key"] for t in tiers] == ["f2p", "moderate", "whale"]
        assert tiers[0]["total_pulls"] == 50 and tiers[0]["extra_pulls"] == 0
        # 85% of 160 = 136 total pulls, 86 more than the 50-pull baseline.
        assert tiers[1]["total_pulls"] == 136 and tiers[1]["extra_pulls"] == 86
        assert tiers[2]["total_pulls"] == 160 and tiers[2]["extra_pulls"] == 110
        assert len(runs) == 2   # moderate confirmation + guaranteed confirmation

        # All three pairwise gaps, each as a percent of the 50-pull current budget.
        assert len(comparisons) == 3
        f2p_to_mod, f2p_to_whale, mod_to_whale = comparisons
        assert f2p_to_mod["delta_pulls"] == 86 and f2p_to_mod["pct_of_current_budget"] == 172.0
        assert f2p_to_whale["delta_pulls"] == 110 and f2p_to_whale["pct_of_current_budget"] == 220.0
        assert mod_to_whale["delta_pulls"] == 24 and mod_to_whale["pct_of_current_budget"] == 48.0

    def test_single_further_pill_label_continues_the_numbering_from_the_first(self, monkeypatch):
        # Current odds already comfortable (93.75%, above the 85% moderate
        # target but short of near-certain), so only the guaranteed figure
        # is computed on top of the current-budget result; its pill label
        # should read "2", continuing the numbering, never restarting at 1.
        monkeypatch.setattr(advisor, "run_simulation_verbose", _tier_fake_sim(160))
        scenario = {
            "start_char_pity": 0, "start_char_guarantee": False, "remaining_characters": 0,
            "start_weapon_pity": 0, "start_weapon_guarantee": False, "remaining_weapons": 1,
        }
        run_params = {**BASELINE_PARAMS, "total_pulls": 150, "strategy": [{"banner": "weapon", "copies": 1}]}
        primary_result = {"success_rate": "93.75%"}
        tiers, _runs, _comparisons = advisor._spending_tiers(scenario, run_params, BASELINE_PARAMS, primary_result)
        assert [t["key"] for t in tiers] == ["f2p", "whale"]

        # Same scenario end-to-end through run_advisor: starting budget 160,
        # 2 weapon copies desired, 1 already WON after 10 pulls (a win, not
        # a loss, so guarantee resets to False for the still-remaining
        # copy, matching the "not guaranteed" 160-pull figure above).
        params = {**BASELINE_PARAMS, "total_pulls": 160, "strategy": [{"banner": "weapon", "copies": 2}]}
        stats = {**BASELINE_STATS, "initial_pulls": 160, "desired_characters": 0, "desired_weapons": 2}
        events = [_event(1, "weapon", "win", 10, 0)]
        fake = _FakeClient([
            _extraction_response(events=events),
            _response(_msg(content="ok")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)

        answer, runs, breakdown = run_advisor(params, stats, "I won the weapon after 10 pulls, how am I doing?")

        labels = [line["label"] for line in breakdown["lines"]]
        assert "AI Agent Simulated Result 1" in labels
        assert "AI Agent Simulated Result 2" in labels
        assert "AI Agent Simulated Result 3" not in labels   # no moderate figure to number

    def test_spending_tiers_skips_moderate_and_whale_when_f2p_already_guarantees(self, monkeypatch):
        monkeypatch.setattr(advisor, "run_simulation_verbose", _tier_fake_sim(160))
        scenario = {
            "start_char_pity": 0, "start_char_guarantee": False, "remaining_characters": 0,
            "start_weapon_pity": 0, "start_weapon_guarantee": False, "remaining_weapons": 1,
        }
        run_params = {**BASELINE_PARAMS, "total_pulls": 200, "strategy": [{"banner": "weapon", "copies": 1}]}
        primary_result = {"success_rate": "99.99%"}

        tiers, runs, comparisons = advisor._spending_tiers(scenario, run_params, BASELINE_PARAMS, primary_result)

        assert [t["key"] for t in tiers] == ["f2p"]
        assert runs == []
        assert comparisons == []   # nothing to compare with only one tier present

    def test_spending_tiers_skips_everything_once_the_current_budget_is_already_near_certain(self, monkeypatch):
        # Live bug report: the theoretical worst-case "guaranteed" pull
        # count (via hard pity) can be well above what the current budget
        # actually needs in practice, since real average-case luck beats
        # the pessimistic worst case. If the current budget's OWN
        # simulated success rate is already near-certain, nothing further
        # should even be computed, regardless of how the closed-form
        # guarantee bound compares to the current budget.
        monkeypatch.setattr(advisor, "run_simulation_verbose", _tier_fake_sim(160))
        scenario = {
            "start_char_pity": 0, "start_char_guarantee": False, "remaining_characters": 0,
            "start_weapon_pity": 0, "start_weapon_guarantee": False, "remaining_weapons": 1,
        }
        # guaranteed_pulls would be 160 (see _guaranteed_pulls_for_banner),
        # comfortably above this 100-pull budget, so the OLD guaranteed_extra
        # check alone would not have skipped anything here.
        run_params = {**BASELINE_PARAMS, "total_pulls": 100, "strategy": [{"banner": "weapon", "copies": 1}]}
        primary_result = {"success_rate": "99.80%"}

        tiers, runs, comparisons = advisor._spending_tiers(scenario, run_params, BASELINE_PARAMS, primary_result)

        assert [t["key"] for t in tiers] == ["f2p"]
        assert runs == []
        assert comparisons == []

    def test_run_advisor_surfaces_tier_lines_and_context_to_the_model(self, monkeypatch):
        # End-to-end: the further pull-count lines land in the breakdown
        # for the UI under generic, unbranded labels, and the model's
        # final call is told the underlying figures so it can't just
        # regurgitate the current-budget number and punt.
        params = {**BASELINE_PARAMS, "total_pulls": 50,
                  "strategy": [{"banner": "weapon", "copies": 1}]}
        stats = {**BASELINE_STATS, "initial_pulls": 50, "desired_characters": 0, "desired_weapons": 1}
        events = [_event(1, "weapon", "loss", 10, 0)]

        fake = _FakeClient([
            _extraction_response(events=events),
            _response(_msg(content="Answer covering the current odds and further options.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _tier_fake_sim(160))

        answer, runs, breakdown = run_advisor(
            params, stats, "I lost the weapon after 10 pulls, how am I doing?",
        )

        assert answer == "Answer covering the current odds and further options."
        labels = [line["label"] for line in breakdown["lines"]]
        # Generic, numbered pill labels, no "tier" branding in the UI: the
        # current-budget result is "1", same numbering as everything after
        # it, not a separate un-orchestrated given the others compare against.
        assert "AI Agent Simulated Result 1" in labels
        assert "AI Agent Simulated Result 2" in labels
        assert "AI Agent Simulated Result 3" in labels
        # The current-budget pre-run plus the two further confirmations,
        # all tracked receipts.
        assert len(runs) == 3

        final_call_context = fake.calls[-1]["messages"][1]["content"]
        assert "Your current budget, spending nothing further" in final_call_context
        assert "A modest additional pull count" in final_call_context
        assert "The pull count that guarantees the entire remaining goal outright" in final_call_context
        # The model is told NOT to label these as tiers in its own answer.
        assert "Never label these as tiers" in advisor.SYSTEM_PROMPT
        # The pairwise pull-count gaps, as a percent of the current budget,
        # are handed over too, not left for the model to compute itself.
        assert "pull-count gaps between the figures" in final_call_context
        assert "% on top of the current 40-pull budget" in final_call_context

    def test_context_names_numbers_from_the_question_to_prevent_recomputation(self, monkeypatch):
        # Live-testing found the model sometimes re-subtracting a number
        # from the question itself (e.g. "lost after 10 pulls") from the
        # already-final F2P total, producing a wrong, lower figure. The
        # fix is naming that exact number in the context so it can't be
        # mistaken for something still needing to be applied.
        params = {**BASELINE_PARAMS, "total_pulls": 50,
                  "strategy": [{"banner": "weapon", "copies": 1}]}
        stats = {**BASELINE_STATS, "initial_pulls": 50, "desired_characters": 0, "desired_weapons": 1}
        events = [_event(1, "weapon", "loss", 10, 0)]

        fake = _FakeClient([
            _extraction_response(events=events),
            _response(_msg(content="ok")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _tier_fake_sim(160))

        run_advisor(params, stats, "I lost the weapon after 10 pulls, how am I doing with what's left?")

        final_call_context = fake.calls[-1]["messages"][1]["content"]
        assert "Numbers mentioned in your own question (10)" in final_call_context
        assert "already fully incorporated into the 40 figure above" in final_call_context

    def test_tier_pull_comparisons_covers_every_pair_present(self):
        tiers = [
            {"key": "f2p", "label": "F2P", "total_pulls": 50},
            {"key": "moderate", "label": "Moderate", "total_pulls": 136},
            {"key": "whale", "label": "Whale", "total_pulls": 160},
        ]
        comparisons = advisor._tier_pull_comparisons(tiers, current_budget=50)
        assert [(c["from_label"], c["to_label"]) for c in comparisons] == [
            ("F2P", "Moderate"), ("F2P", "Whale"), ("Moderate", "Whale"),
        ]
        assert [c["delta_pulls"] for c in comparisons] == [86, 110, 24]
        assert [c["pct_of_current_budget"] for c in comparisons] == [172.0, 220.0, 48.0]

    def test_tier_pull_comparisons_empty_when_only_f2p_is_present(self):
        assert advisor._tier_pull_comparisons([{"key": "f2p", "label": "F2P", "total_pulls": 50}], 50) == []


class TestPityConflictClarification:
    """A pulls-spent event whose implied ending pity exceeds the banner's
    hard pity is a genuine ambiguity (did the reported count already
    include existing pity?), not a plain parse failure: it's surfaced as a
    conflict for the user to resolve, and clarifications resolve it via one
    retry extraction rather than an automatic guess."""

    def test_conflict_is_returned_immediately_without_auto_retry(self, monkeypatch):
        # Weapon starts at 22 pity (hard pity 80); "won after 60 pulls" would
        # put it at 82, past hard pity, so this can't be pulls-spent as-is.
        event = _ps_event(1, "weapon", "win", 60)
        fake = _FakeClient([_extraction_response(events=[event])])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)

        params = {**BASELINE_PARAMS, "start_weapon_pity": 22}
        stats = {**BASELINE_STATS, "start_weapon_pity": 22}

        answer, runs, breakdown = run_advisor(
            params, stats, "I won the weapon after 60 pulls, how am I doing?",
        )

        assert answer == ""
        assert runs == []
        assert breakdown["status"] == "conflict"
        assert len(breakdown["conflicts"]) == 1
        conflict = breakdown["conflicts"][0]
        assert conflict["header"] == "For Weapon Attempt 1 of 1"
        assert conflict["banner"] == "weapon"
        assert "60 pulls spent" in conflict["question"] or "60 pulls" in conflict["question"]
        # A genuine ambiguity only a human can resolve: no automatic retry.
        assert len(fake.calls) == 1

    def test_clarification_confirming_the_total_includes_existing_pity(self, monkeypatch):
        conflict_event = _ps_event(1, "weapon", "win", 60)
        resolved_event = _abs_event(1, "weapon", "win", 60)
        fake = _FakeClient([
            _extraction_response(events=[conflict_event]),
            _extraction_response(events=[resolved_event]),
            _response(_msg(content="With the guarantee your weapon odds look fine.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

        params = {**BASELINE_PARAMS, "start_weapon_pity": 22}
        stats = {**BASELINE_STATS, "start_weapon_pity": 22}

        answer, runs, breakdown = run_advisor(
            params, stats, "I won the weapon after 60 pulls, how am I doing?",
            clarifications=[{"conflict_key": "pity:weapon:1",
                              "answer": "60 total including existing pity"}],
        )

        assert breakdown["status"] == "ok"
        assert breakdown["annotations"] == [
            {"label": "USER CLARIFICATION: ",
             "text": "60 total pulls were used, including existing pity."},
        ]
        assert answer == "With the guarantee your weapon odds look fine."
        assert len(fake.calls) == 3   # extraction, retry, final answer

    def test_clarification_correcting_the_starting_situation_overrides_pity_for_this_run_only(self, monkeypatch):
        # Same event reported unchanged; the retry instead corrects the
        # banner's starting pity itself (22 -> 10), which must actually be
        # applied before re-reconciling: 10 + 60 = 70 is under the 80 hard
        # pity, so this should resolve, not conflict again.
        conflict_event = _ps_event(1, "weapon", "win", 60)
        fake = _FakeClient([
            _extraction_response(events=[conflict_event]),
            _extraction_response(
                events=[conflict_event],
                starting_situation_corrections=[
                    {"banner": "weapon", "corrected_pity": 10, "corrected_guarantee": None},
                ],
            ),
            _response(_msg(content="Looks solid with the corrected pity.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

        original_baseline_params = {**BASELINE_PARAMS, "start_weapon_pity": 22}
        params = dict(original_baseline_params)
        stats = {**BASELINE_STATS, "start_weapon_pity": 22}

        answer, runs, breakdown = run_advisor(
            params, stats, "I won the weapon after 60 pulls, how am I doing?",
            clarifications=[{"conflict_key": "pity:weapon:1",
                              "answer": "actually my weapon pity was only 10, not 22"}],
        )

        assert breakdown["status"] == "ok"
        assert breakdown["annotations"] == [
            {"label": "USER MODIFICATION TO INITIAL SITUATION: ",
             "text": "Weapon banner starting pity 10."},
        ]
        assert answer == "Looks solid with the corrected pity."
        # The correction applied only to this run, never mutating the dict
        # the caller passed in (the overall simulation/visualization above
        # must keep the pity actually entered in the form).
        assert params == original_baseline_params

    def test_clarification_correcting_the_prompt_number_resolves_the_conflict(self, monkeypatch):
        conflict_event = _ps_event(1, "weapon", "win", 60)
        corrected_event = _ps_event(1, "weapon", "win", 40)
        fake = _FakeClient([
            _extraction_response(events=[conflict_event]),
            _extraction_response(events=[corrected_event]),
            _response(_msg(content="40 pulls looks fine.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

        params = {**BASELINE_PARAMS, "start_weapon_pity": 22}
        stats = {**BASELINE_STATS, "start_weapon_pity": 22}

        answer, runs, breakdown = run_advisor(
            params, stats, "I won the weapon after 60 pulls, how am I doing?",
            clarifications=[{"conflict_key": "pity:weapon:1",
                              "answer": "sorry, I meant 40 pulls, not 60"}],
        )

        assert breakdown["status"] == "ok"
        assert breakdown["annotations"] == [
            {"label": "USER MODIFICATION TO PROMPT: ",
             "text": "Weapon banner attempt 1 of 1: sorry, I meant 40 pulls, not 60"},
        ]
        assert answer == "40 pulls looks fine."

    def test_multiple_conflicts_each_get_their_own_annotation_in_order(self, monkeypatch):
        char_conflict = _ps_event(1, "character", "win", 85)
        weapon_conflict = _ps_event(2, "weapon", "loss", 60)
        char_resolved = _abs_event(1, "character", "win", 85)
        weapon_resolved = _abs_event(2, "weapon", "loss", 60)

        fake = _FakeClient([
            _extraction_response(events=[char_conflict, weapon_conflict]),
            _extraction_response(events=[char_resolved, weapon_resolved]),
            _response(_msg(content="Both banners check out now.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

        params = {**BASELINE_PARAMS, "start_char_pity": 20, "start_weapon_pity": 22}
        stats = {**BASELINE_STATS, "start_char_pity": 20, "start_weapon_pity": 22}

        answer, runs, breakdown = run_advisor(
            params, stats,
            "I won the character after 85 pulls and lost the weapon after 60 pulls, how am I doing?",
            clarifications=[
                {"conflict_key": "pity:character:1", "answer": "85 total including existing pity"},
                {"conflict_key": "pity:weapon:1", "answer": "60 total including existing pity"},
            ],
        )

        assert breakdown["status"] == "ok"
        assert breakdown["annotations"] == [
            {"label": "USER CLARIFICATION: ",
             "text": "85 total pulls were used, including existing pity."},
            {"label": "USER CLARIFICATION: ",
             "text": "60 total pulls were used, including existing pity."},
        ]

    def test_a_branch_conflict_is_clarified_and_resolved(self, monkeypatch):
        # Branch A's assumed weapon win "after 60 pulls" can't fit on top of
        # the 22 pity already there; it's asked about, not declined.
        branches = lambda event: [
            _branch("If I win the weapon pull", [event]),
            _branch("If I lose the weapon pull", [_ps_event(1, "weapon", "loss", 30)]),
        ]
        conflicted = lambda: _branching_extraction_response({}, branches(_ps_event(1, "weapon", "win", 60)))
        fake = _FakeClient([
            conflicted(),                       # first Ask
            conflicted(),                       # resubmit: fresh extraction rediscovers it
            _branching_extraction_response({}, branches(_abs_event(1, "weapon", "win", 60))),  # retry
            _response(_msg(content="Both branches look fine.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

        params = {**BASELINE_PARAMS, "start_weapon_pity": 22}
        stats = {**BASELINE_STATS, "start_weapon_pity": 22}
        question = "If I win the weapon pull after 60 pulls, what next? If I lose after 30, what then?"

        answer, _, breakdown = run_advisor(params, stats, question)
        assert answer == "" and breakdown["status"] == "conflict"
        assert [c["conflict_key"] for c in breakdown["conflicts"]] == ["branch0:pity:weapon:1"]

        answer, _, breakdown = run_advisor(
            params, stats, question,
            clarifications=[{"conflict_key": "branch0:pity:weapon:1", "answer": "60 total including pity"}],
        )
        assert breakdown["status"] == "ok" and len(breakdown["groups"]) == 2
        assert breakdown["annotations"] == [
            {"label": "USER CLARIFICATION: ", "text": "60 total pulls were used, including existing pity."},
        ]
        retry_prompt = fake.calls[2]["messages"][-1]["content"]
        assert 'In the branch "If I win the weapon pull"' in retry_prompt


class TestBudgetConflictClarification:
    """The reported events together use more pulls than the stated
    starting budget: a genuine ambiguity (gained pulls since the initial
    situation? the initial number was wrong? something else?), not a
    parse failure, so it's surfaced as a conflict to clarify."""

    def test_budget_conflict_is_returned_immediately_without_auto_retry(self, monkeypatch):
        # 50-pull budget, but the reported loss alone spends 70.
        event = _ps_event(1, "weapon", "loss", 70, refunds=0)
        fake = _FakeClient([_extraction_response(events=[event])])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)

        params = {**BASELINE_PARAMS, "total_pulls": 50}
        stats = {**BASELINE_STATS, "initial_pulls": 50}

        answer, runs, breakdown = run_advisor(
            params, stats, "I lost the weapon after 70 pulls, how am I doing?",
        )

        assert answer == ""
        assert runs == []
        assert breakdown["status"] == "conflict"
        assert len(breakdown["conflicts"]) == 1
        conflict = breakdown["conflicts"][0]
        assert conflict["conflict_key"] == "budget"
        assert conflict["header"] == "For Total Pulls Used"
        assert "You reported a total of 70 pulls used" in conflict["question"]
        assert "started with only 50 pulls, 20 short" in conflict["question"]
        # A genuine ambiguity only a human can resolve: no automatic retry.
        assert len(fake.calls) == 1

    def test_clarification_correcting_the_starting_budget_resolves_the_conflict(self, monkeypatch):
        # Same event reported unchanged; the retry restates the total
        # budget itself (covers both "I gained pulls" and "the initial
        # number was wrong", they resolve identically).
        conflict_event = _ps_event(1, "weapon", "loss", 70, refunds=0)
        fake = _FakeClient([
            _extraction_response(events=[conflict_event]),
            _extraction_response(events=[conflict_event], total_pulls_restated=150),
            _response(_msg(content="With the extra pulls you're in a solid spot.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

        params = {**BASELINE_PARAMS, "total_pulls": 50}
        stats = {**BASELINE_STATS, "initial_pulls": 50}

        answer, runs, breakdown = run_advisor(
            params, stats, "I lost the weapon after 70 pulls, how am I doing?",
            clarifications=[{"conflict_key": "budget",
                              "answer": "I gained 100 pulls from an event since then"}],
        )

        assert breakdown["status"] == "ok"
        assert breakdown["annotations"] == [
            {"label": "USER MODIFICATION TO INITIAL SITUATION: ",
             "text": "Starting budget corrected to 150 total pulls."},
        ]
        assert answer == "With the extra pulls you're in a solid spot."

    def test_clarification_correcting_the_prompt_number_resolves_the_budget_conflict(self, monkeypatch):
        conflict_event = _ps_event(1, "weapon", "loss", 70, refunds=0)
        corrected_event = _ps_event(1, "weapon", "loss", 30, refunds=0)
        fake = _FakeClient([
            _extraction_response(events=[conflict_event]),
            _extraction_response(events=[corrected_event]),
            _response(_msg(content="30 pulls fits your budget fine.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

        params = {**BASELINE_PARAMS, "total_pulls": 50}
        stats = {**BASELINE_STATS, "initial_pulls": 50}

        answer, runs, breakdown = run_advisor(
            params, stats, "I lost the weapon after 70 pulls, how am I doing?",
            clarifications=[{"conflict_key": "budget",
                              "answer": "sorry, I meant 30 pulls, not 70"}],
        )

        assert breakdown["status"] == "ok"
        assert breakdown["annotations"] == [
            {"label": "USER MODIFICATION TO PROMPT: ",
             "text": "sorry, I meant 30 pulls, not 70"},
        ]
        assert answer == "30 pulls fits your budget fine."


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


class TestGoalOnlyHypotheticals:
    def test_goal_change_without_events_is_reconciled_and_displayed(self, monkeypatch):
        # No events (has_event_sequence false) but an added character copy:
        # must go through reconciliation, not the model's free-form tool loop.
        fake = _FakeClient([
            _response(_msg(content=json.dumps({
                "has_event_sequence": False, "events": [],
                "pulls_remaining_stated": None, "additional_pulls_stated": None,
                "total_pulls_restated": None,
                "additional_character_copies_wanted": 1, "additional_weapon_copies_wanted": None,
            }))),
            _response(_msg(content="Adding a copy lowers your odds.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)
        monkeypatch.setattr(advisor, "_spending_tiers", lambda *a, **k: ([], [], []))

        answer, runs, breakdown = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "what if I go for C1?")

        goal = _find_line(breakdown, "Current Goal")
        assert goal["pills"][0]["value"] == "2 characters and 1 weapon"
        assert "AI Agent Simulated Result 1" in [l["label"] for l in breakdown["lines"]]
        assert answer == "Adding a copy lowers your odds."
        assert len(fake.calls) == 2   # extraction + answer, no tool loop


class TestConditionalPlanRouting:
    """A pulling order with a stop condition ('weapon first; if I lose, stop')
    is simulated run by run and answered as routes, never as invented
    win/lose scenarios and never through the free-form tool loop."""

    PARAMS = {**BASELINE_PARAMS, "total_pulls": 140}
    STATS = {**BASELINE_STATS, "initial_pulls": 140}
    QUESTION = (
        "I lost the 50/50 at 30 pity. Should I pull on the engine banner first and if I win, then "
        "continue on the character banner? If I lose, I would stop there. I'm expecting around 120 pulls."
    )
    PLAN = [
        {"step_order": 1, "banner_type": "weapon", "copies": 1, "on_loss": "stop"},
        {"step_order": 2, "banner_type": "character", "copies": 1, "on_loss": "continue"},
    ]

    def _fake_plan(self, **kwargs):
        self.plan_calls.append(kwargs)
        node = lambda count, left, char=None, weapon=None: {
            "count": count, "avg_leftover": left,
            **({"guaranteed": {"char": char, "weapon": weapon}} if char is not None else {}),
        }
        return {"trials": 4000, "nodes": {
            (): node(4000, kwargs["total_pulls"]),
            ("won",): node(3000, 182.0), ("lost",): node(1000, 183.0, 1.0, 1.0),
            ("won", "won"): node(3000, 128.0, 0.0, 0.0),
        }, "leaves": {("won", "won"): 3000, ("lost",): 1000}}

    def _run(self, monkeypatch, extraction):
        fake = _FakeClient([extraction, _response(_msg(content="Pull the weapon first."))])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        self.plan_calls = []
        monkeypatch.setattr(advisor, "simulate_plan", self._fake_plan)
        monkeypatch.setattr(advisor, "run_simulation_verbose",
                            lambda **_: pytest.fail("a plan must not run the ordinary simulation"))
        return fake, run_advisor(self.PARAMS, self.STATS, self.QUESTION)

    def test_events_then_plan_answer_as_routes(self, monkeypatch):
        fake, (answer, runs, breakdown) = self._run(monkeypatch, _extraction_response(
            events=[{**_event(1, "character", "loss", 30, None), "pity_is_absolute": True}],
            additional_pulls_stated=120, plan=self.PLAN, reply_language="English",
        ))

        assert answer == "Pull the weapon first."
        assert [l["label"] for l in breakdown["lines"]][-2:] == ["Starting Point for Every Route", "Current Goal"]
        assert [g["label"] for g in breakdown["groups"]] == [
            "Route 1: Weapon won, then Character obtained", "Route 2: Weapon lost, stop",
        ]
        # The chained run started from the reconciled state: 140 - 30 + 3 + 120.
        (call,) = self.plan_calls
        assert call["total_pulls"] == 233 and call["start_char_guarantee"] is True
        assert call["strategy"][0] == {"banner": "weapon", "copies": 1, "stop_on_loss": True,
                                       "max_pulls": None, "cap_pity": None, "on_cap": None,
                                       "skip_to": None}

        # One extraction, one tool-free narration grounded in the routes.
        assert len(fake.calls) == 2 and "tools" not in fake.calls[-1]
        prompt = fake.calls[-1]["messages"][-1]["content"]
        assert '"Route 1: Weapon won, then Character obtained": chance 75.00%' in prompt
        assert "keeping 183 pulls on average" in prompt
        # Route 2 states BOTH guarantees it leaves, including the character
        # one carried in from the earlier lost 50/50, not just the new one.
        route2 = prompt.split('"Route 2:')[1]
        assert ("the character banner's next 5-star IS guaranteed to be the featured one (carried over "
                "from before this plan") in route2
        assert "the weapon banner's next 5-star IS guaranteed to be the featured one (from this route's" in route2
        assert "(120, 30, 50)" in prompt
        # The model answers a question restated from the reconciled plan.
        assert prompt.endswith(
            "Follow-up question: My plan, in order: Weapon, stop if lost; then Character. I have exactly "
            "233 pulls for it. How does this plan look, and is it a good plan? Reply in English."
        )

    def test_a_plan_with_no_events_still_applies(self, monkeypatch):
        _, (_, _, breakdown) = self._run(monkeypatch, _extraction_response(
            has_event_sequence=False, plan=self.PLAN,
        ))
        assert len(breakdown["groups"]) == 2
        assert self.plan_calls[0]["total_pulls"] == 140


class TestBudgetOnlyWhatIfs:
    """'What if I had 20 more pulls': no events, but still a deterministic,
    displayed total, answered without the free-form tool loop."""

    def test_more_pulls_is_reconciled_shown_and_restated(self, monkeypatch):
        seen = []

        def _spy_sim(**kwargs):
            seen.append(kwargs["total_pulls"])
            return _fake_sim(**kwargs)

        fake = _FakeClient([
            _extraction_response(has_event_sequence=False, additional_pulls_stated=20, reply_language="English"),
            _response(_msg(content="At 140 pulls you're at 80%.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _spy_sim)
        monkeypatch.setattr(advisor, "_spending_tiers",
                            lambda scenario, run_params, *_: ([{"key": "f2p", "total_pulls": run_params["total_pulls"]}], [], []))

        answer, runs, breakdown = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "what if I had 20 more pulls?")

        assert answer == "At 140 pulls you're at 80%."
        assert seen == [140]
        line = _find_line(breakdown, "Additional Pulls Mentioned")
        assert [p["value"] for p in line["pills"]] == [120, "+", 20, "=", 140]
        # No tool loop, and the model sees the settled total, not "20 more".
        assert len(fake.calls) == 2 and "tools" not in fake.calls[-1]
        follow_up = fake.calls[-1]["messages"][-1]["content"].split("Follow-up question: ")[1]
        assert follow_up.startswith("How likely am I to reach the goal of 1 character and 1 weapon with 140 pulls")


class TestStoppingBranch:
    def test_a_branch_that_stops_ends_on_what_it_keeps(self, monkeypatch):
        fake = _FakeClient([
            _branching_extraction_response({}, [
                _branch("If I win the character at 30", [_event(1, "character", "win", 30, 3)]),
                _branch("If I lose the character at 75", [_event(1, "character", "loss", 75, 8)],
                        stops_pulling=True, characters_wanted_total=0),
            ]),
            _response(_msg(content="Stopping keeps your guarantee.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        seen = []

        def _spy_sim(**kwargs):
            seen.append(kwargs["total_pulls"])
            return _fake_sim(**kwargs)

        monkeypatch.setattr(advisor, "run_simulation_verbose", _spy_sim)

        _, _, breakdown = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "if I lose I'll just stop there")

        stop_group = breakdown["groups"][1]
        # The goal change reported alongside stopping is ignored, not "0 of 0".
        assert [l["label"] for l in stop_group["lines"]] == ["Character Run 1 (obtained 0 of 1)", "Stops Here"]
        assert [p["value"] for p in stop_group["lines"][-1]["pills"]] == [
            "53 KEPT", "CHARACTER", "0 pity", "GUARANTEE: TRUE", "WEAPON", "0 pity", "GUARANTEE: FALSE",
        ]
        assert stop_group["lines"][-1]["pills"][3]["tooltip"].startswith(
            "Guaranteed: Character Run 1 above lost its 50/50")
        # Only the branch that keeps pulling was simulated: 120 - 30 + 3 = 93.
        assert seen == [93]
        assert "chooses to stop pulling here, keeping 53 pulls" in fake.calls[-1]["messages"][-1]["content"]


class TestUnnumberedBranchesAreAPlan:
    def test_an_outcome_with_no_number_is_sent_back_as_a_plan(self):
        extracted = {
            "has_event_sequence": True, "plan": None, **_scenario_fields(),
            "branches": [
                _branch("If I win the 50/50", [_event(1, "character", "win", None, None)],
                        additional_character_copies_wanted=1),
                _branch("If I lose the 50/50", [_event(1, "character", "loss", None, None)]),
            ],
        }
        result = advisor._reconcile_scenario_list(extracted, BASELINE_PARAMS, BASELINE_STATS)
        assert not result["ok"]
        assert "report the whole question as one plan instead" in result["error"]
        assert "skip_to_banner" in result["error"]

"""Tests for the open-ended advisor loop (advisor.py). OpenAI is fully mocked."""
import json
import types

import pytest

import advisor
from advisor import run_advisor, _run_tool, _validate_strategy


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


def _no_session_context():
    return json.dumps({
        "has_session_context": False,
        "character_obtained": None, "character_pulls_spent": None, "character_refunds": None,
        "character_guarantee_active": None,
        "weapon_obtained": None, "weapon_pulls_spent": None, "weapon_refunds": None,
        "weapon_guarantee_active": None,
        "pulls_remaining_stated": None, "additional_pulls_stated": None,
        "total_pulls_restated": None,
        "additional_character_copies_wanted": None, "additional_weapon_copies_wanted": None,
    })


def _extraction_response(**overrides):
    payload = {
        "has_session_context": True,
        "character_obtained": None, "character_pulls_spent": None, "character_refunds": None,
        "character_guarantee_active": None,
        "weapon_obtained": None, "weapon_pulls_spent": None, "weapon_refunds": None,
        "weapon_guarantee_active": None,
        "pulls_remaining_stated": None, "additional_pulls_stated": None,
        "total_pulls_restated": None,
        "additional_character_copies_wanted": None, "additional_weapon_copies_wanted": None,
    }
    payload.update(overrides)
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


def test_runs_tool_then_answers(monkeypatch):
    tc = _tool_call("call_1", "run_simulation",
                    json.dumps({"total_pulls": 160, "strategy": BASELINE_PARAMS["strategy"]}))
    fake = _FakeClient([
        _response(_msg(content=_no_session_context())),          # extraction: pure hypothetical
        _response(_msg(content=None, tool_calls=[tc])),          # model asks to run the sim
        _response(_msg(content="With 160 pulls you're at 80%.")),  # model answers
    ])
    monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
    monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

    answer, runs, breakdown = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "what if I had 160 pulls?")

    assert answer == "With 160 pulls you're at 80%."
    assert breakdown is None
    assert len(fake.calls) == 3
    # The third request carried the tool result back to the model.
    roles = [m["role"] for m in fake.calls[2]["messages"]]
    assert "tool" in roles
    # The run the model made is surfaced for the UI receipts.
    assert len(runs) == 1
    assert runs[0]["total_pulls"] == 160
    assert runs[0]["success_rate"] == "80.00%"


def test_answers_without_tool_call(monkeypatch):
    fake = _FakeClient([
        _response(_msg(content=_no_session_context())),
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
    responses = [_response(_msg(content=_no_session_context()))]
    responses += [_response(_msg(content=None, tool_calls=[tc])) for _ in range(3)]
    responses.append(_response(_msg(content="Final forced answer.")))
    fake = _FakeClient(responses)
    monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
    monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

    answer, runs, breakdown = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "keep testing", max_tool_calls=3)
    assert answer == "Final forced answer."
    assert breakdown is None
    assert len(fake.calls) == 5                       # extraction + 3 tool loops + 1 forced final
    assert fake.calls[-1]["tool_choice"] == "none"    # final call forbids more tools
    assert len(runs) == 3                             # each loop ran the tool once


class TestSessionStateIntegration:
    """The scenario from the bug report: character already secured, weapon
    lost the 50/50 (guarantee active), 41 pulls stated as remaining. The
    reconciled state should drop the character phase entirely and simulate
    only the weapon, from 0 pity, with the guarantee, not the original
    full 1C/1W goal."""

    def test_reduces_goal_and_nets_refunds_before_the_model_ever_sees_it(self, monkeypatch):
        # The model calls the tool with no overrides, so whatever it actually
        # simulates comes straight from the reconciled run_params fallback.
        # Numbers mirror the original bug report exactly: 100 total pulls, 42
        # pity minus 11 refunds on the character, 31 pulls minus 3 refunds on
        # the weapon, net used is 59, so 41 pulls should remain (not
        # 100-42-31=27, which is what a naive gross subtraction would give).
        params = {**BASELINE_PARAMS, "total_pulls": 100}
        stats = {**BASELINE_STATS, "initial_pulls": 100}

        tc = _tool_call("c1", "run_simulation", json.dumps({}))
        fake = _FakeClient([
            _extraction_response(
                character_obtained=True, character_pulls_spent=42, character_refunds=11,
                weapon_obtained=False, weapon_pulls_spent=31, weapon_refunds=3,
                weapon_guarantee_active=True,
            ),
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

        assert breakdown is not None
        assert "42 pity" in breakdown
        assert "41 pulls remaining" in breakdown
        # The actual simulation ran on just the weapon, from scratch, guaranteed,
        # with pulls netted of refunds, not the naive gross subtraction.
        assert seen_kwargs["strategy"] == [{"banner": "weapon", "copies": 1}]
        assert seen_kwargs["total_pulls"] == 41
        assert seen_kwargs["start_weapon_pity"] == 0
        assert seen_kwargs["start_weapon_guarantee"] is True
        assert answer == "With the guarantee your weapon odds are strong."

    def test_retries_extraction_once_then_surfaces_the_failure(self, monkeypatch):
        # A reconciliation that never resolves must NOT silently answer from
        # the unadjusted baseline as if nothing had been stated, the user
        # needs to see that their described progress could not be verified.
        tc = _tool_call("c1", "run_simulation", json.dumps({}))
        fake = _FakeClient([
            # First extraction: pulls figures conflict, reconcile rejects it.
            _extraction_response(
                character_obtained=True, weapon_obtained=False,
                pulls_remaining_stated=41, weapon_pulls_spent=50, total_pulls_restated=120,
            ),
            # Retry still doesn't resolve it (still conflicting).
            _extraction_response(
                character_obtained=True, weapon_obtained=False,
                pulls_remaining_stated=41, weapon_pulls_spent=60, total_pulls_restated=120,
            ),
            _response(_msg(content=None, tool_calls=[tc])),
            _response(_msg(content="Could not verify your progress, using the baseline goal.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
        monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

        answer, runs, breakdown = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "confusing question")

        assert breakdown is not None
        assert "Could not verify" in breakdown        # the caveat is visible, not silent
        assert len(fake.calls) == 4                    # 2 extractions + 1 tool loop + 1 answer
        assert answer == "Could not verify your progress, using the baseline goal."
        # It still ran the actual simulation on the unadjusted baseline, since
        # there's no reliable reconciled state to use instead.
        assert len(runs) == 1

    def test_goal_already_complete_skips_simulation(self, monkeypatch):
        fake = _FakeClient([
            _extraction_response(
                character_obtained=True, weapon_obtained=True, pulls_remaining_stated=20,
            ),
            _response(_msg(content="You already have everything you need.")),
        ])
        monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)

        answer, runs, breakdown = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "am I done?")

        assert "complete" in breakdown
        assert runs == []                              # nothing was simulated
        assert answer == "You already have everything you need."
        assert fake.calls[-1]["tool_choice"] == "none"  # no tool offered for this path

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

        tc = _tool_call("c1", "run_simulation", json.dumps({}))
        fake = _FakeClient([
            _extraction_response(
                character_obtained=True, character_pulls_spent=42, character_refunds=11,
                weapon_obtained=False, weapon_pulls_spent=31, weapon_refunds=3,
                weapon_guarantee_active=True,
                additional_pulls_stated=45,
                additional_character_copies_wanted=1,
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

        assert breakdown is not None
        assert "86 pulls remaining" in breakdown  # 41 net remaining + 45 additional
        assert seen_kwargs["total_pulls"] == 86
        # 1 more character copy wanted (the one already obtained is dropped),
        # plus the still-needed weapon.
        assert {"banner": "char", "copies": 1} in seen_kwargs["strategy"]
        assert {"banner": "weapon", "copies": 1} in seen_kwargs["strategy"]
        assert seen_kwargs["start_weapon_guarantee"] is True
        assert answer == "With the extra pulls, both goals are within reach."


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


class TestValidateStrategy:
    def test_ok(self):
        assert _validate_strategy([{"banner": "char", "copies": 2}]) is None

    def test_empty(self):
        assert _validate_strategy([]) is not None

    def test_bad_copies(self):
        assert _validate_strategy([{"banner": "char", "copies": 0}]) is not None

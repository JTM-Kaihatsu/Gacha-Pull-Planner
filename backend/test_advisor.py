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
        _response(_msg(content=None, tool_calls=[tc])),          # model asks to run the sim
        _response(_msg(content="With 160 pulls you're at 80%.")),  # model answers
    ])
    monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
    monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

    answer = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "what if I had 160 pulls?")

    assert answer == "With 160 pulls you're at 80%."
    assert len(fake.calls) == 2
    # The second request carried the tool result back to the model.
    roles = [m["role"] for m in fake.calls[1]["messages"]]
    assert "tool" in roles


def test_answers_without_tool_call(monkeypatch):
    fake = _FakeClient([_response(_msg(content="You're already comfortable, save your pulls."))])
    monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)

    answer = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "should I even bother?")
    assert "save your pulls" in answer
    assert len(fake.calls) == 1


def test_tool_call_budget_is_capped(monkeypatch):
    tc = _tool_call("c", "run_simulation", json.dumps({"total_pulls": 200, "strategy": BASELINE_PARAMS["strategy"]}))
    # Model asks for a tool on every one of the 3 allowed loops; the loop then
    # stops and makes one final forced (tool_choice=none) call for the answer.
    responses = [_response(_msg(content=None, tool_calls=[tc])) for _ in range(3)]
    responses.append(_response(_msg(content="Final forced answer.")))
    fake = _FakeClient(responses)
    monkeypatch.setattr(advisor, "OpenAI", lambda **_: fake)
    monkeypatch.setattr(advisor, "run_simulation_verbose", _fake_sim)

    answer = run_advisor(BASELINE_PARAMS, BASELINE_STATS, "keep testing", max_tool_calls=3)
    assert answer == "Final forced answer."
    assert len(fake.calls) == 4                       # 3 tool loops + 1 forced final
    assert fake.calls[-1]["tool_choice"] == "none"    # final call forbids more tools


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

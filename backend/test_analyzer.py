"""Tests for analyzer.py. The OpenAI client is fully mocked — no network calls."""
import copy

import pytest

import analyzer
from analyzer import describe_goal, analyze_sim_result


# ---------------------------------------------------------------------------
# describe_goal — pure
# ---------------------------------------------------------------------------
class TestDescribeGoal:
    def test_single_char_single_weapon(self):
        label, desc = describe_goal({"desired_characters": 1, "desired_weapons": 1})
        assert label == "C0W1"
        assert "one copy of the limited 5★ character" in desc
        assert "one signature 5★ weapon" in desc

    def test_multi_char_no_weapon(self):
        label, desc = describe_goal({"desired_characters": 3, "desired_weapons": 0})
        assert label == "C2W0"
        assert "3 copies" in desc
        assert "no signature weapon" in desc

    def test_multi_weapon(self):
        label, desc = describe_goal({"desired_characters": 1, "desired_weapons": 2})
        assert label == "C0W2"
        assert "2 signature 5★ weapons" in desc

    def test_no_characters_only_weapons(self):
        # Regression: 0 characters must not become "C-1" or "0 copies of the character".
        label, desc = describe_goal({"desired_characters": 0, "desired_weapons": 2})
        assert label == "W2"
        assert "no character copies" in desc
        assert "2 signature 5★ weapons" in desc
        assert "C-1" not in label and "0 copies" not in desc


# ---------------------------------------------------------------------------
# analyze_sim_result — mocked client
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_openai(monkeypatch):
    """Patch analyzer.OpenAI + api-key lookup. Returns a list of the kwargs each
    chat.completions.create call received, for prompt inspection."""
    calls = []

    class _Msg:
        content = "  MOCK VERDICT  "

    class _Choice:
        message = _Msg()

    class _Completion:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return _Completion()

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = _Chat()

    monkeypatch.setattr(analyzer, "OpenAI", _FakeClient)
    monkeypatch.setattr(analyzer, "get_openai_api_key", lambda: "sk-test")
    return calls


class TestAnalyzeSimResult:
    def test_returns_stripped_content_and_calls_once(self, fake_openai, sample_stats):
        result = analyze_sim_result(sample_stats, model="test-model")
        assert result == "MOCK VERDICT"        # stripped
        assert len(fake_openai) == 1

    def test_prompt_formats_against_real_stats_keys(self, fake_openai, sample_stats):
        # Exercises the double-.format() / brace-escaping path against the exact
        # key set the engine emits. A missing/renamed key would raise here.
        analyze_sim_result(sample_stats, model="test-model")
        messages = fake_openai[0]["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert sample_stats["success_rate"] in user_msg   # values interpolated
        assert "{" not in user_msg                          # no leftover placeholders

    def test_no_failure_branch(self, fake_openai, sample_stats):
        stats = copy.deepcopy(sample_stats)
        stats["most_common_failure_state"] = None
        result = analyze_sim_result(stats, model="test-model")
        assert result == "MOCK VERDICT"
        user_msg = next(
            m["content"] for m in fake_openai[0]["messages"] if m["role"] == "user"
        )
        assert "100%" in user_msg or "No failures" in user_msg

    def test_guarantee_note_included_when_set(self, fake_openai, sample_stats):
        stats = copy.deepcopy(sample_stats)
        stats["start_char_guarantee"] = True
        analyze_sim_result(stats, model="test-model")
        user_msg = next(
            m["content"] for m in fake_openai[0]["messages"] if m["role"] == "user"
        )
        assert "guarantee" in user_msg.lower()

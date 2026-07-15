"""Tests for analyzer.py (describe_goal). The one-shot LLM verdict was retired."""
from analyzer import describe_goal


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

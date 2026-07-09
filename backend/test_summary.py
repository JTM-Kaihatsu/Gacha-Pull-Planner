"""Tests for the deterministic read in summary.py."""
from summary import build_summary


def _base(**overrides):
    stats = {
        "success_rate": "50.00%",
        "desired_characters": 1,
        "desired_weapons": 0,
        "avg_leftover_pulls_on_success": 5,
        "start_char_guarantee": False,
        "start_weapon_guarantee": False,
        "correlation_stats": {
            "total_successes": 100, "total_failures": 100,
            "all_5050s_won_in_successes": 55, "all_5050s_won_in_failures": 45,
            "early_char_pity_in_successes": 30, "early_char_pity_in_failures": 28,
        },
    }
    stats.update(overrides)
    return stats


class TestConfidence:
    def test_comfortable(self):
        assert build_summary(_base(success_rate="92.00%"))["confidence"] == "comfortable"

    def test_coin_flip(self):
        assert build_summary(_base(success_rate="50.00%"))["confidence"] == "coin_flip"

    def test_stretch(self):
        assert build_summary(_base(success_rate="25.00%"))["confidence"] == "stretch"

    def test_long_shot(self):
        assert build_summary(_base(success_rate="8.00%"))["confidence"] == "long_shot"

    def test_shape_always_has_headline_and_notes(self):
        out = build_summary(_base())
        assert set(out) == {"confidence", "headline", "notes"}
        assert out["headline"]
        assert len(out["notes"]) >= 1          # framing is always present


class TestKeyLever:
    def test_5050_flagged_when_decisive(self):
        stats = _base(correlation_stats={
            "total_successes": 100, "total_failures": 100,
            "all_5050s_won_in_successes": 90, "all_5050s_won_in_failures": 20,
            "early_char_pity_in_successes": 30, "early_char_pity_in_failures": 25,
        })
        notes = " ".join(build_summary(stats)["notes"])
        assert "50/50" in notes

    def test_no_lever_when_no_failures(self):
        # 100% success -> total_failures 0 -> no lever computed, but still valid.
        stats = _base(success_rate="100.00%", correlation_stats={
            "total_successes": 100, "total_failures": 0,
        })
        out = build_summary(stats)
        assert out["confidence"] == "comfortable"
        assert len(out["notes"]) >= 1

    def test_no_lever_when_factors_are_weak(self):
        # Success and failure look the same -> nothing decisive -> no lever note.
        stats = _base(correlation_stats={
            "total_successes": 100, "total_failures": 100,
            "all_5050s_won_in_successes": 50, "all_5050s_won_in_failures": 50,
            "early_char_pity_in_successes": 20, "early_char_pity_in_failures": 20,
        })
        notes = " ".join(build_summary(stats)["notes"])
        assert "50/50" not in notes


class TestFraming:
    def test_comfortable_discourages_overspending(self):
        notes = " ".join(build_summary(_base(success_rate="95.00%"))["notes"]).lower()
        assert "overspend" in notes

    def test_guarantee_note_appended(self):
        notes = " ".join(build_summary(_base(start_char_guarantee=True))["notes"]).lower()
        assert "guarantee" in notes


class TestRobustness:
    def test_accepts_numeric_success_rate(self):
        # Test/mocked stats sometimes use a float; must not crash.
        out = build_summary(_base(success_rate=1.0))
        assert out["confidence"] == "comfortable"

    def test_tolerates_missing_correlation_stats(self):
        out = build_summary({"success_rate": "40.00%"})
        assert out["confidence"] == "coin_flip"
        assert out["notes"]

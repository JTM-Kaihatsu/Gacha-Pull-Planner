"""Tests for session_state.py: pure functions, no mocking needed."""
from session_state import reconcile

BASELINE_PARAMS = {
    "total_pulls": 100,
    "strategy": [{"banner": "char", "copies": 1}, {"banner": "weapon", "copies": 1}],
    "start_char_pity": 0, "start_char_guarantee": False,
    "start_weapon_pity": 0, "start_weapon_guarantee": False,
    "full_4star_chars": True,
    "char_pity_config": {"base_rate": 0.006, "soft_pity_start": 73, "hard_pity": 90},
    "weapon_pity_config": {"base_rate": 0.008, "soft_pity_start": 65, "hard_pity": 80},
}

BASELINE_STATS = {
    "desired_characters": 1, "desired_weapons": 1,
    "initial_pulls": 100, "start_char_pity": 0, "start_weapon_pity": 0,
    "success_rate": "50.00%",
}


def _blank(**overrides):
    base = {
        "has_session_context": True,
        "character_obtained": None, "character_pity_at_obtain": None,
        "character_guarantee_active": None,
        "weapon_obtained": None, "weapon_pity_at_obtain": None,
        "weapon_guarantee_active": None,
        "pulls_used_so_far": None, "pulls_remaining_stated": None,
        "total_pulls_restated": None,
    }
    base.update(overrides)
    return base


def test_applies_false_when_no_session_context():
    result = reconcile(_blank(has_session_context=False), BASELINE_PARAMS, BASELINE_STATS)
    assert result == {"applies": False}


def test_drops_satisfied_banner_and_defaults_pity_to_zero():
    extracted = _blank(
        character_obtained=True, character_pity_at_obtain=42,
        weapon_obtained=False, weapon_guarantee_active=True,
        pulls_remaining_stated=41,
    )
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)

    assert result["ok"] is True
    assert result["goal_complete"] is False
    assert result["pulls_exhausted"] is False
    assert result["strategy"] == [{"banner": "weapon", "copies": 1}]
    assert result["total_pulls"] == 41
    assert result["start_char_pity"] == 0
    assert result["start_char_guarantee"] is False
    assert result["start_weapon_pity"] == 0
    assert result["start_weapon_guarantee"] is True
    assert "42 pity" in result["breakdown"]
    assert "guarantee active" in result["breakdown"]
    assert "41 pulls remaining" in result["breakdown"]


def test_computes_remaining_pulls_from_pulls_used_so_far():
    extracted = _blank(character_obtained=True, weapon_obtained=False, pulls_used_so_far=59)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is True
    assert result["total_pulls"] == 100 - 59


def test_conflicting_pull_counts_return_error():
    extracted = _blank(
        character_obtained=True, weapon_obtained=False,
        pulls_remaining_stated=41, pulls_used_so_far=50, total_pulls_restated=100,
    )
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["applies"] is True
    assert result["ok"] is False
    assert "does not reconcile" in result["error"]


def test_missing_pull_info_returns_error():
    extracted = _blank(character_obtained=True, weapon_obtained=False)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is False
    assert "could not determine pulls remaining" in result["error"]


def test_goal_already_complete():
    extracted = _blank(character_obtained=True, weapon_obtained=True, pulls_remaining_stated=10)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is True
    assert result["goal_complete"] is True
    assert result["strategy"] == []
    assert "complete" in result["breakdown"]


def test_pulls_exhausted():
    extracted = _blank(character_obtained=True, weapon_obtained=False, pulls_remaining_stated=0)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is True
    assert result["pulls_exhausted"] is True
    assert result["total_pulls"] == 0
    assert "no pulls remain" in result["breakdown"]


def test_pity_out_of_range_returns_error():
    extracted = _blank(character_obtained=False, weapon_obtained=True, weapon_pity_at_obtain=999,
                        pulls_remaining_stated=10)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is False
    assert "out of range" in result["error"]

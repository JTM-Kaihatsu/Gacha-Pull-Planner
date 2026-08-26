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
        "character_obtained": None, "character_pulls_spent": None, "character_refunds": None,
        "character_guarantee_active": None,
        "weapon_obtained": None, "weapon_pulls_spent": None, "weapon_refunds": None,
        "weapon_guarantee_active": None,
        "pulls_remaining_stated": None, "additional_pulls_stated": None,
        "total_pulls_restated": None,
        "additional_character_copies_wanted": None, "additional_weapon_copies_wanted": None,
    }
    base.update(overrides)
    return base


def test_applies_false_when_no_session_context():
    result = reconcile(_blank(has_session_context=False), BASELINE_PARAMS, BASELINE_STATS)
    assert result == {"applies": False}


def test_drops_satisfied_banner_and_defaults_pity_to_zero():
    # No weapon pulls stated yet (the question is purely about the character
    # phase), so pulls remaining is computed from the character spend alone:
    # 100 - 42 = 58.
    extracted = _blank(
        character_obtained=True, character_pulls_spent=42,
        weapon_obtained=False, weapon_guarantee_active=True,
    )
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)

    assert result["ok"] is True
    assert result["goal_complete"] is False
    assert result["pulls_exhausted"] is False
    assert result["strategy"] == [{"banner": "weapon", "copies": 1}]
    assert result["total_pulls"] == 58
    assert result["start_char_pity"] == 0
    assert result["start_char_guarantee"] is False
    assert result["start_weapon_pity"] == 0
    assert result["start_weapon_guarantee"] is True
    assert "42 pity" in result["breakdown"]
    assert "guarantee active" in result["breakdown"]
    assert "58 total pulls remaining" in result["breakdown"]


def test_computes_remaining_pulls_net_of_refunds():
    # The bug this schema exists to prevent: 42 pity minus 11 refunds on the
    # character, 31 pulls minus 3 refunds on the weapon, out of 100 total.
    # Net used = 31 + 28 = 59, so 41 should remain, not 100 - 42 - 31 = 27.
    extracted = _blank(
        character_obtained=True, character_pulls_spent=42, character_refunds=11,
        weapon_obtained=False, weapon_pulls_spent=31, weapon_refunds=3,
        weapon_guarantee_active=True,
    )
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is True
    assert result["total_pulls"] == 41


def test_refunds_exceeding_pulls_spent_return_error():
    extracted = _blank(character_obtained=True, weapon_obtained=False,
                        weapon_pulls_spent=10, weapon_refunds=15, pulls_remaining_stated=50)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is False
    assert "exceeds" in result["error"]


def test_conflicting_pull_counts_return_error():
    extracted = _blank(
        character_obtained=True, weapon_obtained=False,
        pulls_remaining_stated=41, weapon_pulls_spent=50, total_pulls_restated=100,
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


def test_pulls_spent_out_of_range_returns_error():
    extracted = _blank(character_obtained=False, weapon_obtained=True, weapon_pulls_spent=999,
                        pulls_remaining_stated=10)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is False
    assert "out of range" in result["error"]


def test_additional_pulls_stated_adds_on_top_of_computed_remaining():
    # The second bug report: "plus around 45 more" is an addition on top of
    # whatever remains, not a restatement of the total. Net used = 59 out of
    # 100, so remaining is 41, plus the stated 45 more = 86.
    extracted = _blank(
        character_obtained=True, character_pulls_spent=42, character_refunds=11,
        weapon_obtained=False, weapon_pulls_spent=31, weapon_refunds=3,
        weapon_guarantee_active=True, additional_pulls_stated=45,
    )
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is True
    assert result["total_pulls"] == 86
    assert "86 total pulls remaining" in result["breakdown"]
    # The note that this already includes the addition guards against the
    # interpretation model adding it a second time on top of this figure.
    assert "already includes the 45 additional pulls" in result["breakdown"]


def test_additional_copies_wanted_expands_the_goal():
    # "another character copy" on top of the original 1-character goal means
    # the total desired grows to 2; 1 is already obtained, so 1 remains.
    extracted = _blank(
        character_obtained=True,
        weapon_obtained=False, weapon_guarantee_active=True,
        additional_character_copies_wanted=1, pulls_remaining_stated=41,
    )
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is True
    assert {"banner": "char", "copies": 1} in result["strategy"]
    assert {"banner": "weapon", "copies": 1} in result["strategy"]
    assert "1 extra character copy" in result["breakdown"]


def test_requested_copies_exceeding_max_return_error():
    extracted = _blank(
        character_obtained=False, weapon_obtained=False,
        additional_character_copies_wanted=10, pulls_remaining_stated=50,
    )
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is False
    assert "exceed the max" in result["error"]


def test_negative_additional_pulls_returns_error():
    extracted = _blank(character_obtained=True, weapon_obtained=False,
                        additional_pulls_stated=-5, pulls_remaining_stated=10)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is False
    assert "cannot be negative" in result["error"]

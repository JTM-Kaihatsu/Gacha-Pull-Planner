"""Tests for session_state.py: pure functions, no mocking needed."""
from session_state import TOOLTIPS, build_result_line, reconcile

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


def _event(order, banner, outcome, pity, refunds):
    return {"event_order": order, "banner_type": banner, "outcome": outcome,
            "pity_at_outcome": pity, "refund_count": refunds}


def _blank(**overrides):
    base = {
        "has_event_sequence": True,
        "events": [],
        "pulls_remaining_stated": None, "additional_pulls_stated": None,
        "total_pulls_restated": None,
        "additional_character_copies_wanted": None, "additional_weapon_copies_wanted": None,
    }
    base.update(overrides)
    return base


def test_applies_false_when_no_event_sequence():
    result = reconcile(_blank(has_event_sequence=False), BASELINE_PARAMS, BASELINE_STATS)
    assert result == {"applies": False}


def test_reproduces_the_original_bug_report_exactly():
    # 100 pulls, character won at 42 pity (11 refunded), weapon lost at 31
    # pity (3 refunded), goal expanded by 1 extra character, plus 45 stated
    # as an addition. Net used = 31 + 28 = 59, so 41 remain, +45 = 86.
    events = [
        _event(1, "character", "win", 42, 11),
        _event(2, "weapon", "loss", 31, 3),
    ]
    extracted = _blank(events=events, additional_pulls_stated=45, additional_character_copies_wanted=1)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)

    assert result["ok"] is True
    assert result["total_pulls"] == 86
    assert result["start_char_pity"] == 0          # character had an event, forced to 0
    assert result["start_char_guarantee"] is False  # won, no guarantee
    assert result["start_weapon_pity"] == 0
    assert result["start_weapon_guarantee"] is True  # lost, guarantee active
    assert {"banner": "char", "copies": 1} in result["strategy"]   # 1 more wanted, 1 already obtained
    assert {"banner": "weapon", "copies": 1} in result["strategy"]

    labels = [line["label"] for line in result["lines"]]
    assert "Character Run 1 (obtained 1)" in labels
    assert "Weapon Run 1 (obtained 0)" in labels
    assert "Additional Pulls Mentioned" in labels
    assert "Character Run 2 (obtained 1)" in labels   # the current, in-progress attempt
    assert "Weapon Run 2 (obtained 0)" in labels
    assert "Restated Goal" in labels

    char_run_1 = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 1)")
    values = [p["value"] for p in char_run_1["pills"]]
    assert values == [100, "−", 42, "+", 11, 69, "WIN"]
    assert char_run_1["pills"][-1]["color"] == "green"

    weapon_run_1 = next(l for l in result["lines"] if l["label"] == "Weapon Run 1 (obtained 0)")
    assert weapon_run_1["pills"][-1]["value"] == "LOSS"
    assert weapon_run_1["pills"][-1]["color"] == "red"
    assert [p["value"] for p in weapon_run_1["pills"][:6]] == [69, "−", 31, "+", 3, 41]

    goal_line = next(l for l in result["lines"] if l["label"] == "Restated Goal")
    assert goal_line["pills"][0]["value"] == "1 character and 1 weapon → 2 characters and 1 weapon"

    char_run_2 = next(l for l in result["lines"] if l["label"] == "Character Run 2 (obtained 1)")
    assert char_run_2["pills"][0]["value"] == 86
    assert char_run_2["pills"][1]["value"] == "GUARANTEE: FALSE"
    assert char_run_2["pills"][1]["color"] == "red"

    weapon_run_2 = next(l for l in result["lines"] if l["label"] == "Weapon Run 2 (obtained 0)")
    assert weapon_run_2["pills"][1]["value"] == "GUARANTEE: TRUE"
    assert weapon_run_2["pills"][1]["color"] == "green"


def test_unmentioned_banner_preserves_original_form_state():
    # The question only describes a character event; the weapon banner was
    # never mentioned at all, so its pity/guarantee should come from
    # whatever the original form stated, not be forced to 0/False.
    params = {**BASELINE_PARAMS, "start_weapon_pity": 37, "start_weapon_guarantee": True}
    extracted = _blank(events=[_event(1, "character", "win", 20, 0)], pulls_remaining_stated=80)
    result = reconcile(extracted, params, BASELINE_STATS)
    assert result["ok"] is True
    assert result["start_weapon_pity"] == 37
    assert result["start_weapon_guarantee"] is True


def test_multiple_losses_before_a_win_on_the_same_banner():
    # This is exactly what the old flat per-banner schema could not express:
    # two losses (each still a real pity-reset outcome) before the win.
    events = [
        _event(1, "weapon", "loss", 50, 2),
        _event(2, "weapon", "loss", 60, 1),
        _event(3, "weapon", "win", 10, 0),
    ]
    extracted = _blank(events=events, total_pulls_restated=200)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is True
    # 200 - (50-2) - (60-1) - (10-0) = 200 - 48 - 59 - 10 = 83
    assert result["total_pulls"] == 83
    # Weapon goal (1) is now satisfied; character was never mentioned, so it
    # still needs its own in-progress run, and the goal is not complete.
    assert result["goal_complete"] is False
    labels = [line["label"] for line in result["lines"]]
    assert "Weapon Run 1 (obtained 0)" in labels
    assert "Weapon Run 2 (obtained 0)" in labels
    assert "Weapon Run 3 (obtained 1)" in labels
    assert "Character Run 1 (obtained 0)" in labels   # in-progress, character never mentioned
    assert not any(label.startswith("Weapon Run 4") for label in labels)  # weapon goal met


def test_event_order_must_be_sequential():
    events = [_event(2, "character", "win", 20, 0)]
    result = reconcile(_blank(events=events, pulls_remaining_stated=80), BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is False
    assert "sequentially" in result["error"]


def test_pity_out_of_range_returns_error():
    events = [_event(1, "weapon", "win", 999, 0)]
    result = reconcile(_blank(events=events, pulls_remaining_stated=80), BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is False
    assert "out of range" in result["error"]


def test_refund_exceeding_pity_returns_error():
    events = [_event(1, "weapon", "loss", 10, 15)]
    result = reconcile(_blank(events=events, pulls_remaining_stated=80), BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is False
    assert "exceeds" in result["error"]


def test_events_consuming_more_than_budget_returns_error():
    # Each event is individually valid (within the weapon's 0-80 hard pity),
    # but together they consume more than the 100-pull budget.
    events = [_event(1, "weapon", "loss", 70, 0), _event(2, "weapon", "loss", 70, 0)]
    result = reconcile(_blank(events=events), BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is False
    assert "exceed the stated total budget" in result["error"]


def test_conflicting_pulls_remaining_stated_returns_error():
    events = [_event(1, "character", "win", 42, 11)]
    extracted = _blank(events=events, pulls_remaining_stated=999)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is False
    assert "does not reconcile" in result["error"]


def test_requested_copies_exceeding_max_return_error():
    events = [_event(1, "character", "win", 20, 0)]
    extracted = _blank(events=events, additional_character_copies_wanted=10, pulls_remaining_stated=80)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is False
    assert "exceed the max" in result["error"]


def test_empty_events_with_has_event_sequence_true_returns_error():
    result = reconcile(_blank(events=[]), BASELINE_PARAMS, BASELINE_STATS)
    assert result["applies"] is True
    assert result["ok"] is False
    assert "no events were reported" in result["error"]


def test_goal_already_complete_via_win_events():
    events = [_event(1, "character", "win", 20, 0), _event(2, "weapon", "win", 30, 0)]
    extracted = _blank(events=events, pulls_remaining_stated=50)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is True
    assert result["goal_complete"] is True
    assert result["strategy"] == []
    labels = [line["label"] for line in result["lines"]]
    assert "Restated Goal" in labels
    # no in-progress "current run" lines once the goal is complete
    assert not any("Run 2" in label for label in labels)
    status = next(l for l in result["lines"] if l["label"] == "Goal Status")
    assert status["pills"][0]["value"] == "GOAL COMPLETE"
    assert status["pills"][0]["color"] == "green"


def test_pulls_exhausted_when_events_consume_everything():
    # Restate the budget down to exactly the character's max pity so a
    # single loss there consumes the whole budget without going negative.
    events = [_event(1, "character", "loss", 90, 0)]
    extracted = _blank(events=events, total_pulls_restated=90)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is True
    assert result["pulls_exhausted"] is True
    assert result["total_pulls"] == 0
    status = next(l for l in result["lines"] if l["label"] == "Goal Status")
    assert status["pills"][0]["value"] == "NO PULLS REMAIN"
    assert status["pills"][0]["color"] == "red"


def test_build_result_line():
    line = build_result_line(86, "16.70%")
    assert line["label"] == "Simulated Result"
    assert line["pills"][0]["value"] == "86 pulls → 16.70%"
    assert line["pills"][0]["color"] == "light_green"


def test_final_result_tooltip_targets_the_result_pill_not_the_outcome_pill():
    # Regression: when there's no "Additional Pulls" line, the last event
    # line's last pill is its WIN/LOSS outcome, not its result. The "final
    # result" tooltip must land on the result pill (second to last), and
    # must never overwrite the outcome pill's own tooltip.
    events = [_event(1, "character", "win", 60, 5), _event(2, "weapon", "loss", 70, 10)]
    extracted = _blank(events=events, total_pulls_restated=200)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is True

    weapon_line = next(l for l in result["lines"] if l["label"] == "Weapon Run 1 (obtained 0)")
    outcome_pill = weapon_line["pills"][-1]
    result_pill = weapon_line["pills"][-2]
    assert outcome_pill["kind"] == "outcome"
    assert outcome_pill["tooltip"] == TOOLTIPS["outcome"]
    assert result_pill["kind"] == "result"
    assert result_pill["tooltip"] == TOOLTIPS["final_result"]


class TestUnstatedPity:
    """Users often describe an outcome without an exact pity/pull count,
    e.g. "I won with 62 to spare" or "I lost, and I have 81 left". These
    events carry pity_at_outcome=None; pulls_remaining_stated becomes the
    only valid source of truth for the current total in that case."""

    def test_single_win_with_pulls_to_spare_instead_of_pity(self):
        # "I won the character 50/50 with 62 pulls to spare"
        events = [_event(1, "character", "win", None, 0)]
        extracted = _blank(events=events, pulls_remaining_stated=62)
        result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)

        assert result["ok"] is True
        assert result["total_pulls"] == 62
        char_line = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 1)")
        assert char_line["pills"] == [
            {"kind": "outcome", "value": "WIN", "color": "green", "tooltip": TOOLTIPS["outcome"]},
        ]
        stated_line = next(l for l in result["lines"] if l["label"] == "Stated Pulls Remaining")
        assert stated_line["pills"][0]["value"] == 62
        assert stated_line["pills"][0]["tooltip"] == TOOLTIPS["final_result"]  # overwritten, it's the final figure

    def test_loss_with_pulls_left_instead_of_pity(self):
        # "I lost the first run at the character banner and I have 81 pulls left"
        events = [_event(1, "character", "loss", None, 0)]
        extracted = _blank(events=events, pulls_remaining_stated=81)
        result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)

        assert result["ok"] is True
        assert result["total_pulls"] == 81
        assert result["start_char_guarantee"] is True   # a loss is a loss, known regardless of pity
        char_line = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 0)")
        assert char_line["pills"][0]["value"] == "LOSS"

    def test_unstated_pity_without_a_direct_restatement_returns_error(self):
        events = [_event(1, "character", "win", None, 0)]
        result = reconcile(_blank(events=events), BASELINE_PARAMS, BASELINE_STATS)
        assert result["ok"] is False
        assert "pulls_remaining_stated" in result["error"]

    def test_mixed_known_and_unknown_pity_events(self):
        # One event with full numbers, one without: the known event still
        # contributes to the ledger for context, but the final total comes
        # from the direct restatement, not from partially-summed math.
        events = [
            _event(1, "character", "win", 42, 11),
            _event(2, "weapon", "loss", None, 0),
        ]
        extracted = _blank(events=events, pulls_remaining_stated=50)
        result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
        assert result["ok"] is True
        assert result["total_pulls"] == 50
        char_line = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 1)")
        assert [p["value"] for p in char_line["pills"]] == [100, "−", 42, "+", 11, 69, "WIN"]

    def test_additional_pulls_still_apply_on_top_of_stated_remaining(self):
        events = [_event(1, "character", "win", None, 0)]
        extracted = _blank(events=events, pulls_remaining_stated=62, additional_pulls_stated=20)
        result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
        assert result["ok"] is True
        assert result["total_pulls"] == 82

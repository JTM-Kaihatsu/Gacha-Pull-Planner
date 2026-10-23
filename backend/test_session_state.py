"""Tests for session_state.py: pure functions, no mocking needed."""
from session_state import (
    MIN_ROUTE_SAMPLE, TOOLTIPS, build_result_line, build_route_groups, reconcile, remaining_goal_text,
    _estimate_refunds,
)

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


def _event(order, banner, outcome, pity, refunds, pity_is_absolute=True):
    return {"event_order": order, "banner_type": banner, "outcome": outcome,
            "pity_at_outcome": pity, "refund_count": refunds, "pity_is_absolute": pity_is_absolute}


def _add(amount, source="other", unit="pulls"):
    """One pull_additions entry, as the extraction reports it."""
    return {"amount": amount, "unit": unit, "source": source}


def _blank(**overrides):
    base = {
        "has_event_sequence": True,
        "events": [],
        "pulls_remaining_stated": None, "pull_additions": None,
        "total_pulls_restated": None,
        "additional_character_copies_wanted": None, "additional_weapon_copies_wanted": None,
    }
    base.update(overrides)
    return base


def _row(line):
    """A pill row's values; a group pill (the '(ending - starting)' pity
    breakdown) becomes a nested list of its own inner values."""
    return [
        [sp["value"] for sp in p["pills"]] if p.get("kind") == "group" else p["value"]
        for p in line["pills"]
    ]


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
    extracted = _blank(events=events, pull_additions=[_add(45)], additional_character_copies_wanted=1)
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
    assert "Character Run 1 (obtained 1 of 2)" in labels
    assert "Weapon Run 1 (obtained 0 of 1)" in labels
    assert "Additional Pulls Mentioned" in labels
    # Both banners still need a run: one combined line, not two separate
    # ones each wrongly implying its own independent pull budget.
    assert "Remaining Character (obtained 1 of 2) and Weapon (obtained 0 of 1) Runs" in labels
    assert "Current Goal" in labels

    char_run_1 = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 1 of 2)")
    values = [p["value"] for p in char_run_1["pills"]]
    assert values == [100, "−", 42, "+", 11, "=", 69, "WIN"]
    assert char_run_1["pills"][-1]["color"] == "green"
    assert char_run_1["pills"][5]["color"] == "magenta"  # the "=" operator

    weapon_run_1 = next(l for l in result["lines"] if l["label"] == "Weapon Run 1 (obtained 0 of 1)")
    assert weapon_run_1["pills"][-1]["value"] == "LOSS"
    assert weapon_run_1["pills"][-1]["color"] == "red"
    assert [p["value"] for p in weapon_run_1["pills"][:7]] == [69, "−", 31, "+", 3, "=", 41]

    # "Current Goal" reflects what's still actually needed (1 more of each),
    # not the original or expanded total (2 characters and 1 weapon).
    goal_line = next(l for l in result["lines"] if l["label"] == "Current Goal")
    assert goal_line["pills"][0]["value"] == "1 character and 1 weapon"

    remaining_line = next(
        l for l in result["lines"]
        if l["label"] == "Remaining Character (obtained 1 of 2) and Weapon (obtained 0 of 1) Runs"
    )
    assert remaining_line["pills"][0]["value"] == 86           # shared starting pool, shown once
    assert remaining_line["pills"][1]["value"] == "CHARACTER"
    assert remaining_line["pills"][2]["value"] == "GUARANTEE: FALSE"
    assert remaining_line["pills"][2]["color"] == "red"
    assert remaining_line["pills"][3]["value"] == "WEAPON"
    assert remaining_line["pills"][4]["value"] == "GUARANTEE: TRUE"
    assert remaining_line["pills"][4]["color"] == "green"

    # 1 of the 2 wanted characters obtained, 0 of 1 weapons: 1 character and
    # 1 weapon still actually remain, not the original 2 characters/1 weapon.
    assert result["remaining_characters"] == 1
    assert result["remaining_weapons"] == 1


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


def test_both_banners_remaining_merge_into_one_shared_pool_line():
    # Both the character and weapon goals are still open after this event:
    # they draw from the same remaining pull pool in sequence, so this must
    # be one combined line, not two lines each wrongly implying its own
    # independent total_pulls budget.
    stats = {**BASELINE_STATS, "desired_characters": 2}
    events = [_event(1, "character", "win", 20, 0)]
    extracted = _blank(events=events, pulls_remaining_stated=80)
    result = reconcile(extracted, BASELINE_PARAMS, stats)
    assert result["ok"] is True
    assert result["remaining_characters"] == 1
    assert result["remaining_weapons"] == 1

    labels = [l["label"] for l in result["lines"]]
    assert "Remaining Character (obtained 1 of 2) and Weapon (obtained 0 of 1) Runs" in labels
    assert not any(l.startswith("Character Run 2") or l == "Weapon Run 1 (obtained 0 of 1)" for l in labels)

    line = next(l for l in result["lines"]
                if l["label"] == "Remaining Character (obtained 1 of 2) and Weapon (obtained 0 of 1) Runs")
    assert [p["value"] for p in line["pills"]] == [
        80, "CHARACTER", "GUARANTEE: FALSE", "WEAPON", "GUARANTEE: FALSE",
    ]
    assert line["pills"][2]["color"] == "red"    # character: won, no guarantee
    assert line["pills"][4]["color"] == "red"    # weapon: never mentioned, baseline default False


def test_only_one_banner_remaining_keeps_the_single_run_label():
    # Only the weapon is still needed (character goal already met): no
    # sharing ambiguity with just one banner left, so this stays the plain
    # single-banner "Run" line, unmerged.
    events = [_event(1, "character", "win", 20, 0)]
    extracted = _blank(events=events, pulls_remaining_stated=80)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is True
    assert result["remaining_characters"] == 0
    assert result["remaining_weapons"] == 1
    labels = [l["label"] for l in result["lines"]]
    assert "Weapon Run 1 (obtained 0 of 1)" in labels
    assert not any("Remaining" in l for l in labels)


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
    assert "Weapon Run 1 (obtained 0 of 1)" in labels
    assert "Weapon Run 2 (obtained 0 of 1)" in labels
    assert "Weapon Run 3 (obtained 1 of 1)" in labels
    assert "Character Run 1 (obtained 0 of 1)" in labels   # in-progress, character never mentioned
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


def test_events_consuming_more_than_budget_becomes_a_conflict_not_a_decline():
    # Each event is individually valid (within the weapon's 0-80 hard pity),
    # but together they consume more than the 100-pull budget: a genuine
    # ambiguity (gained pulls since the initial situation? the initial
    # number was wrong? something else?), not a plain parse failure, so
    # this surfaces as a conflict to clarify, not a decline.
    events = [_event(1, "weapon", "loss", 70, 0), _event(2, "weapon", "loss", 70, 0)]
    result = reconcile(_blank(events=events), BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is False
    assert result["conflict"] is True
    assert len(result["conflicts"]) == 1
    block = result["conflicts"][0]
    assert block["conflict_key"] == "budget"
    assert block["header"] == "For Total Pulls Used"
    assert "You reported a total of 140 pulls used" in block["question"]
    assert "started with only 100 pulls, 40 short" in block["question"]
    assert "Did you gain 40 more pulls" in block["question"]
    group = block["pills"][0]
    assert group["kind"] == "group" and group["color"] == "red"
    assert [p["value"] for p in group["pills"]] == [140, ">", 100]
    # The first event (individually fine) still resolves normally.
    first_line = next(l for l in result["lines"] if l["label"] == "Weapon Run 1 (obtained 0 of 1)")
    assert _row(first_line) == [100, "−", 70, "+", 0, "=", 30, "LOSS"]


def test_budget_conflict_on_the_first_event_has_no_resolved_lines():
    # The overrun happens on the very first event: nothing precedes it,
    # so lines is empty, matching a pity conflict's own first-event case.
    params = {**BASELINE_PARAMS, "total_pulls": 50}
    events = [_event(1, "weapon", "loss", 70, 0)]
    result = reconcile(_blank(events=events), params, BASELINE_STATS)
    assert result["ok"] is False
    assert result["conflict"] is True
    assert result["lines"] == []
    assert result["conflicts"][0]["conflict_key"] == "budget"


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


def test_negative_delta_reduces_goal_below_original():
    # "I'll skip the character entirely and just go for the weapon" after a
    # loss: additional_character_copies_wanted is negative enough to zero
    # out a 2-character goal. The weapon is untouched and still needed.
    stats = {**BASELINE_STATS, "desired_characters": 2}
    events = [_event(1, "character", "loss", 30, 0)]
    extracted = _blank(events=events, additional_character_copies_wanted=-2, pulls_remaining_stated=70)
    result = reconcile(extracted, BASELINE_PARAMS, stats)
    assert result["ok"] is True
    assert result["remaining_characters"] == 0
    assert result["remaining_weapons"] == 1
    assert {"banner": "char", "copies": 1} not in result["strategy"]
    assert {"banner": "weapon", "copies": 1} in result["strategy"]
    goal_line = next(l for l in result["lines"] if l["label"] == "Current Goal")
    assert goal_line["pills"][0]["value"] == "1 weapon"


def test_negative_delta_reducing_to_one_is_satisfied_by_the_first_copy():
    # "I'll skip the second character copy": reduces a 2-character goal to
    # 1, and the copy already won in this scenario satisfies that reduced
    # target, the character goal is complete even though the original goal
    # was 2.
    stats = {**BASELINE_STATS, "desired_characters": 2}
    events = [_event(1, "character", "win", 20, 0)]
    extracted = _blank(events=events, additional_character_copies_wanted=-1, pulls_remaining_stated=80)
    result = reconcile(extracted, BASELINE_PARAMS, stats)
    assert result["ok"] is True
    assert result["remaining_characters"] == 0
    assert result["remaining_weapons"] == 1


def test_negative_delta_floors_at_zero_not_negative():
    # An over-large reduction must clamp at 0, not go negative and leak
    # into the goal text or the ledger.
    events = [_event(1, "character", "loss", 30, 0)]
    extracted = _blank(events=events, additional_character_copies_wanted=-99, pulls_remaining_stated=70)
    result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is True
    assert result["remaining_characters"] == 0
    goal_line = next(l for l in result["lines"] if l["label"] == "Current Goal")
    assert "-" not in str(goal_line["pills"][0]["value"])


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
    assert "Current Goal" in labels
    # no in-progress "current run" lines once the goal is complete
    assert not any("Run 2" in label for label in labels)
    status = next(l for l in result["lines"] if l["label"] == "Goal Status")
    assert status["pills"][0]["value"] == "GOAL COMPLETE"
    assert status["pills"][0]["color"] == "green"
    assert result["remaining_characters"] == 0
    assert result["remaining_weapons"] == 0


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
    # The character was lost, not obtained, and the weapon was never
    # mentioned: both are still fully wanted despite the budget running out.
    assert result["remaining_characters"] == 1
    assert result["remaining_weapons"] == 1


def test_won_character_then_lost_weapon_leaves_only_weapon_remaining():
    # The exact scenario from the live bug report: character won outright,
    # weapon lost (guarantee active). Only the weapon should be reported as
    # still remaining, the character must not show up as still needed.
    events = [
        _event(1, "character", "win", 30, 5),
        _event(2, "weapon", "loss", 50, 3),
    ]
    result = reconcile(_blank(events=events), BASELINE_PARAMS, BASELINE_STATS)
    assert result["ok"] is True
    assert result["goal_complete"] is False
    assert result["remaining_characters"] == 0
    assert result["remaining_weapons"] == 1
    assert {"banner": "char", "copies": 1} not in result["strategy"]
    assert {"banner": "weapon", "copies": 1} in result["strategy"]


def test_interleaved_pull_order_does_not_duplicate_remaining_copies():
    # Live bug report: a pull order like "1 character, then the weapon,
    # then the remaining 2 characters" lists the "char" banner in TWO
    # separate baseline strategy phases. After 2 of 3 character copies are
    # won, remaining_characters correctly reads 1, but the strategy used
    # to actually run the follow-up simulation was assigning the FULL
    # remaining count to EVERY matching phase, silently doubling it to 2
    # characters simulated instead of 1, and gutting the reported success
    # rate to well under half of the true figure.
    params = {**BASELINE_PARAMS, "total_pulls": 300, "start_char_pity": 15, "start_weapon_pity": 8,
              "strategy": [{"banner": "char", "copies": 1}, {"banner": "weapon", "copies": 1},
                           {"banner": "char", "copies": 2}]}
    stats = {**BASELINE_STATS, "desired_characters": 3, "initial_pulls": 300,
             "start_char_pity": 15, "start_weapon_pity": 8}
    events = [
        _event(1, "character", "win", 42, None, pity_is_absolute=True),
        _event(2, "character", "win", 60, None, pity_is_absolute=False),
        _event(3, "weapon", "loss", 15, None, pity_is_absolute=True),
    ]
    result = reconcile(_blank(events=events), params, stats)
    assert result["ok"] is True
    assert result["remaining_characters"] == 1
    assert result["remaining_weapons"] == 1
    # Exactly one phase per remaining banner, each with the true remaining
    # count, not one entry per original phase that banner appeared in.
    assert result["strategy"] == [{"banner": "char", "copies": 1}, {"banner": "weapon", "copies": 1}]


def test_remaining_goal_text():
    assert remaining_goal_text(0, 0) == "nothing, everything above is already obtained"
    assert remaining_goal_text(1, 0) == "1 character"
    assert remaining_goal_text(0, 1) == "1 weapon"
    assert remaining_goal_text(2, 3) == "2 characters and 3 weapons"


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

    weapon_line = next(l for l in result["lines"] if l["label"] == "Weapon Run 1 (obtained 0 of 1)")
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
        char_line = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 1 of 1)")
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
        char_line = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 0 of 1)")
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
        char_line = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 1 of 1)")
        assert [p["value"] for p in char_line["pills"]] == [100, "−", 42, "+", 11, "=", 69, "WIN"]

    def test_additional_pulls_still_apply_on_top_of_stated_remaining(self):
        events = [_event(1, "character", "win", None, 0)]
        extracted = _blank(events=events, pulls_remaining_stated=62, pull_additions=[_add(20)])
        result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
        assert result["ok"] is True
        assert result["total_pulls"] == 82


class TestEstimateRefunds:
    def test_character_formula(self):
        # 0.1105 * 30 = 3.315 -> rounds to 3, no dupe bonus (a loss).
        assert _estimate_refunds("character", "loss", 30, prior_copies=0) == (3, 0)

    def test_weapon_formula(self):
        # 0.0578 * 50 = 2.89 -> rounds to 3, no dupe bonus (a loss).
        assert _estimate_refunds("weapon", "loss", 50, prior_copies=0) == (3, 0)

    def test_dupe_win_adds_bonus(self):
        # A win when a copy of that banner's featured item is already owned
        # returns the +2 dupe bonus separately from the per-pull formula.
        # 0.1105 * 30 = 3.315 -> 3, dupe bonus 2, kept apart, not summed.
        assert _estimate_refunds("character", "win", 30, prior_copies=1) == (3, 2)

    def test_dupe_bonus_only_applies_to_wins(self):
        # A loss never obtains a 5-star, so no dupe applies regardless of
        # how many copies were already owned.
        assert _estimate_refunds("character", "loss", 30, prior_copies=1) == (3, 0)

    def test_first_copy_win_gets_no_dupe_bonus(self):
        # prior_copies == 0: this win is for the first copy, not a repeat.
        assert _estimate_refunds("character", "win", 30, prior_copies=0) == (3, 0)


class TestPityCarryover:
    """The banner's starting pity/guarantee (the 'Starting Situation') must
    be accounted for on the FIRST event per banner. Every existing test
    elsewhere in this file uses BASELINE_PARAMS's default 0 starting pity
    and False starting guarantee, for which all of this is a deliberate
    no-op, that's what confirms zero regression for the common case."""

    def test_absolute_pity_breaks_into_ending_minus_starting_parenthetical(self):
        # "won at 30 pity" with 20 pity already on the banner: the pity
        # figure becomes "(30 - 20)", resolving to 10 pulls actually spent
        # this run, which is what the outer equation subtracts.
        params = {**BASELINE_PARAMS, "start_char_pity": 20}
        events = [_event(1, "character", "win", 30, 0, pity_is_absolute=True)]
        result = reconcile(_blank(events=events), params, BASELINE_STATS)
        assert result["ok"] is True
        char_line = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 1 of 1)")
        # 100 - (30 - 20) + 0 = 90
        assert _row(char_line) == [100, "−", [30, "−", 20], "+", 0, "=", 90, "WIN"]
        group = char_line["pills"][2]
        assert group["kind"] == "group"
        assert "pity reached 30, already starting from 20" in group["pills"][1]["tooltip"]
        assert group["pills"][2]["tooltip"] == TOOLTIPS["existing_pity"]
        assert "as stated in your question" in group["pills"][0]["tooltip"]

    def test_pulls_spent_phrasing_within_bounds_stays_a_flat_number(self):
        # "lost after 50 pulls" with 10 pity already on the banner checks
        # out fine (10 + 50 = 60 <= 80 hard pity): no parenthetical, since
        # "(60 - 10)" would only reconstruct the 50 the question already
        # gave directly, revealing nothing new. The ledger is unaffected.
        params = {**BASELINE_PARAMS, "start_weapon_pity": 10}
        events = [_event(1, "weapon", "loss", 50, 3, pity_is_absolute=False)]
        result = reconcile(_blank(events=events, pulls_remaining_stated=53), params, BASELINE_STATS)
        assert result["ok"] is True
        weapon_line = next(l for l in result["lines"] if l["label"] == "Weapon Run 1 (obtained 0 of 1)")
        assert _row(weapon_line) == [100, "−", 50, "+", 3, "=", 53, "LOSS"]
        assert weapon_line["pills"][2]["kind"] != "group"

    def test_refund_estimate_uses_the_normalized_pulls_not_the_raw_pity(self):
        # Refund estimation must use the ACTUAL pulls spent (10), not the
        # raw stated pity (30), or it would badly overestimate refunds.
        params = {**BASELINE_PARAMS, "start_char_pity": 20}
        events = [_event(1, "character", "win", 30, None, pity_is_absolute=True)]
        result = reconcile(_blank(events=events), params, BASELINE_STATS)
        assert result["ok"] is True
        char_line = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 1 of 1)")
        # 0.1105 * 10 = 1.105 -> 1, not 0.1105 * 30 = 3.315 -> 3
        assert char_line["pills"][4]["value"] == 1

    def test_only_the_first_event_on_a_banner_gets_the_carryover_adjustment(self):
        # Pity resets to 0 after any outcome, so the SECOND event on a
        # banner never needs starting-pity subtraction, only the first.
        params = {**BASELINE_PARAMS, "start_char_pity": 20}
        events = [
            _event(1, "character", "loss", 30, 0, pity_is_absolute=True),   # 30-20=10 spent
            _event(2, "character", "win", 15, 0, pity_is_absolute=True),    # reset to 0, 15 spent directly
        ]
        result = reconcile(_blank(events=events, additional_character_copies_wanted=1), params, BASELINE_STATS)
        assert result["ok"] is True
        first = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 0 of 2)")
        second = next(l for l in result["lines"] if l["label"] == "Character Run 2 (obtained 1 of 2)")
        assert _row(first)[:3] == [100, "−", [30, "−", 20]]   # first: parenthetical
        assert _row(second)[:3] == [90, "−", 15]              # second: flat, pity reset to 0

    def test_absolute_pity_not_exceeding_starting_pity_is_an_error(self):
        # "at 20 pity" when the banner already started at pity 30 is
        # nonsensical: pity only goes up, so this must fail rather than
        # silently produce a negative or zero pull count.
        params = {**BASELINE_PARAMS, "start_char_pity": 30}
        events = [_event(1, "character", "win", 20, 0, pity_is_absolute=True)]
        result = reconcile(_blank(events=events), params, BASELINE_STATS)
        assert result["ok"] is False
        assert "must exceed" in result["error"]

    def test_pulls_spent_exceeding_hard_pity_becomes_a_conflict_not_a_decline(self):
        # 75 pity already on the weapon banner (hard pity 80) plus 10 more
        # pulls spent would reach 85, past the hard cap: a genuine "did you
        # mean total including existing pity" ambiguity, not a plain
        # mistake, so this surfaces as a conflict to clarify, not a decline.
        params = {**BASELINE_PARAMS, "start_weapon_pity": 75}
        events = [_event(1, "weapon", "loss", 10, 0, pity_is_absolute=False)]
        result = reconcile(_blank(events=events), params, BASELINE_STATS)
        assert result["ok"] is False
        assert result["conflict"] is True
        assert result["lines"] == []  # nothing precedes the conflict
        assert len(result["conflicts"]) == 1
        block = result["conflicts"][0]
        assert block["banner"] == "weapon"
        assert block["header"] == "For Weapon Attempt 1 of 1"
        assert "You reported 10 pulls spent on the Weapon banner" in block["question"]
        assert "Did you mean in total, including existing pity, you had spent 10 pulls" in block["question"]
        group = block["pills"][0]
        assert group["kind"] == "group" and group["color"] == "red"
        assert [p["value"] for p in group["pills"]] == [10, "LOSS"]
        assert group["pills"][0]["color"] == "red"
        assert group["pills"][1]["color"] == "red"

    def test_conflict_preserves_resolved_pills_for_events_before_it(self):
        # A character win is fully resolvable and comes first; the weapon
        # loss after it is the conflict. The character's pills must still
        # render normally, only the weapon gets flagged.
        params = {**BASELINE_PARAMS, "start_weapon_pity": 75}
        events = [
            _event(1, "character", "win", 20, 0),
            _event(2, "weapon", "loss", 10, 0, pity_is_absolute=False),
        ]
        result = reconcile(_blank(events=events), params, BASELINE_STATS)
        assert result["ok"] is False
        assert result["conflict"] is True
        char_line = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 1 of 1)")
        assert _row(char_line) == [100, "−", 20, "+", 0, "=", 80, "WIN"]
        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0]["banner"] == "weapon"

    def test_multiple_conflicts_across_both_banners_are_all_reported_at_once(self):
        # Both the character's and the weapon's first events are
        # individually impossible; both must be surfaced together as two
        # separate clarifying questions, not just the first one found.
        params = {**BASELINE_PARAMS, "start_char_pity": 85, "start_weapon_pity": 75}
        events = [
            _event(1, "character", "win", 10, 0, pity_is_absolute=False),   # 85+10=95 > 90
            _event(2, "weapon", "loss", 10, 0, pity_is_absolute=False),     # 75+10=85 > 80
        ]
        result = reconcile(_blank(events=events), params, BASELINE_STATS)
        assert result["ok"] is False
        assert len(result["conflicts"]) == 2
        assert {c["banner"] for c in result["conflicts"]} == {"character", "weapon"}
        assert result["conflicts"][0]["header"] == "For Character Attempt 1 of 1"
        assert result["conflicts"][1]["header"] == "For Weapon Attempt 1 of 1"

    def test_pulls_spent_exceeding_hard_pity_by_itself_is_a_flat_error_not_a_conflict(self):
        # 95 pulls spent, hard pity 90: this is impossible under EITHER
        # reading, no single attempt can ever take more pulls than hard
        # pity allows, existing carryover pity or not. Asking "did you mean
        # it as a total including existing pity" would be pointless here,
        # that reading (ending pity 95) is just as invalid; this must be a
        # flat, unrecoverable error, never routed through the conflict UI.
        # A non-first event always starts from 0 (a reset), so this is also
        # the only way a later attempt in a sequence can ever be invalid.
        events = [
            _event(1, "character", "win", 20, 0),
            _event(2, "character", "loss", 95, 0, pity_is_absolute=False),  # 95 alone > 90
            _event(3, "character", "win", 15, 0),
        ]
        extracted = _blank(events=events, additional_character_copies_wanted=2)
        result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
        assert result["ok"] is False
        assert "conflicts" not in result
        assert "out of range 0-90" in result["error"]

    def test_conflict_attempt_numbering_reflects_total_attempts_but_stays_on_the_first(self):
        # A conflict can only ever land on a banner's FIRST event: pity
        # resets to 0 after every outcome, so only the first event can ever
        # carry nonzero starting pity to be ambiguous about. Three character
        # attempts total, the first one conflicting: the header must say
        # "1 of 3", reflecting the total while still naming the first.
        params = {**BASELINE_PARAMS, "start_char_pity": 85}
        events = [
            _event(1, "character", "loss", 10, 0, pity_is_absolute=False),  # 85+10=95 > 90, 10 <= 90
            _event(2, "character", "loss", 20, 0),
            _event(3, "character", "win", 15, 0),
        ]
        extracted = _blank(events=events, additional_character_copies_wanted=2)
        result = reconcile(extracted, params, BASELINE_STATS)
        assert result["ok"] is False
        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0]["header"] == "For Character Attempt 1 of 3"
        assert result["conflicts"][0]["question"].startswith("You reported 10 pulls spent")

    def test_loss_on_an_already_guaranteed_banner_is_an_error(self):
        # A guaranteed next 5-star cannot lose the 50/50; a "loss" reported
        # for the first event on an already-guaranteed banner is impossible.
        params = {**BASELINE_PARAMS, "start_char_guarantee": True}
        events = [_event(1, "character", "loss", 20, 0)]
        result = reconcile(_blank(events=events), params, BASELINE_STATS)
        assert result["ok"] is False
        assert "already had a guaranteed" in result["error"]

    def test_loss_on_an_already_guaranteed_weapon_banner_is_an_error(self):
        # Same contradiction, weapon side: the check must not be
        # character-only.
        params = {**BASELINE_PARAMS, "start_weapon_guarantee": True}
        events = [_event(1, "weapon", "loss", 15, 0)]
        result = reconcile(_blank(events=events), params, BASELINE_STATS)
        assert result["ok"] is False
        assert "already had a guaranteed" in result["error"]

    def test_win_on_an_already_guaranteed_banner_is_not_an_error(self):
        # A guaranteed win is exactly what's expected, not a contradiction;
        # only a LOSS on an already-guaranteed banner is impossible.
        params = {**BASELINE_PARAMS, "start_char_guarantee": True}
        events = [_event(1, "character", "win", 20, 0)]
        result = reconcile(_blank(events=events), params, BASELINE_STATS)
        assert result["ok"] is True

    def test_guarantee_and_pity_carryover_compose_correctly_together(self):
        # Both dimensions of the starting state apply to the same first
        # event at once: already guaranteed AND already at pity 20, wins
        # at absolute pity 30, i.e. 10 pulls spent this run.
        params = {**BASELINE_PARAMS, "start_char_pity": 20, "start_char_guarantee": True}
        events = [_event(1, "character", "win", 30, 0, pity_is_absolute=True)]
        result = reconcile(_blank(events=events), params, BASELINE_STATS)
        assert result["ok"] is True
        char_line = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 1 of 1)")
        assert _row(char_line)[:3] == [100, "−", [30, "−", 20]]

    def test_guarantee_contradiction_check_only_applies_to_the_first_event(self):
        # The banner started guaranteed and the first event is a
        # consistent WIN (consuming the guarantee); the SECOND event on
        # that banner is then a LOSS, which is fine, it's evaluated against
        # the state after event 1 (guarantee now false), not the original
        # starting guarantee.
        params = {**BASELINE_PARAMS, "start_char_guarantee": True}
        events = [
            _event(1, "character", "win", 20, 0),
            _event(2, "character", "loss", 15, 0),
        ]
        result = reconcile(_blank(events=events, additional_character_copies_wanted=1), params, BASELINE_STATS)
        assert result["ok"] is True

    def test_absolute_pity_exactly_one_more_than_starting_pity_is_valid(self):
        # Boundary: exactly 1 pull spent (30 starting, win stated at pity
        # 31) must succeed, not be rejected as "not exceeding".
        params = {**BASELINE_PARAMS, "start_char_pity": 30}
        events = [_event(1, "character", "win", 31, 0, pity_is_absolute=True)]
        result = reconcile(_blank(events=events), params, BASELINE_STATS)
        assert result["ok"] is True
        char_line = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 1 of 1)")
        assert _row(char_line)[:3] == [100, "−", [31, "−", 30]]

    def test_pulls_spent_landing_exactly_on_hard_pity_is_valid(self):
        # Boundary: landing exactly AT the hard pity cap is the guaranteed-
        # hit point itself, valid; only exceeding it is impossible.
        params = {**BASELINE_PARAMS, "start_weapon_pity": 70}
        events = [_event(1, "weapon", "loss", 10, 0, pity_is_absolute=False)]  # 70 + 10 = 80 == hard_pity
        result = reconcile(_blank(events=events), params, BASELINE_STATS)
        assert result["ok"] is True

    def test_zero_starting_pity_is_a_complete_no_op(self):
        # Explicit regression guard: with 0 starting pity (the default),
        # behavior must be byte-identical to before this feature existed,
        # a flat pity pill, no parenthetical.
        events = [_event(1, "character", "win", 30, 0, pity_is_absolute=True)]
        result = reconcile(_blank(events=events), BASELINE_PARAMS, BASELINE_STATS)
        assert result["ok"] is True
        char_line = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 1 of 1)")
        assert _row(char_line) == [100, "−", 30, "+", 0, "=", 70, "WIN"]
        assert char_line["pills"][2]["kind"] != "group"
        assert char_line["pills"][2]["tooltip"] == TOOLTIPS["pity"]


class TestRefundEstimationInReconcile:
    def test_unstated_refund_with_full_4star_chars_is_estimated_and_flagged(self):
        events = [_event(1, "character", "win", 30, None)]
        result = reconcile(_blank(events=events), BASELINE_PARAMS, BASELINE_STATS)
        assert result["ok"] is True
        char_line = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 1 of 1)")
        refund_pill = char_line["pills"][4]
        # 0.1105 * 30 = 3.315 -> 3
        assert refund_pill["value"] == 3
        assert refund_pill["color"] == "light_green"   # matches the "calculated" result styling
        assert "formula estimating the average number of refunds for 30 pulls" in refund_pill["tooltip"]
        # The ledger itself used the estimated figure: 100 - 30 + 3 = 73
        assert char_line["pills"][6]["value"] == 73

    def test_unstated_refund_without_full_4star_chars_defaults_to_zero(self):
        params = {**BASELINE_PARAMS, "full_4star_chars": False}
        events = [_event(1, "character", "win", 30, None)]
        result = reconcile(_blank(events=events), params, BASELINE_STATS)
        assert result["ok"] is True
        char_line = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 1 of 1)")
        refund_pill = char_line["pills"][4]
        assert refund_pill["value"] == 0
        assert refund_pill["color"] == "cyan"   # not flagged as estimated, same as an explicit 0
        assert refund_pill["tooltip"] == TOOLTIPS["refund"]

    def test_explicit_zero_refund_is_not_treated_as_estimated(self):
        # An explicit 0 is a stated fact, not an absence of information;
        # it must render exactly like any other stated refund count.
        events = [_event(1, "character", "win", 30, 0)]
        result = reconcile(_blank(events=events), BASELINE_PARAMS, BASELINE_STATS)
        assert result["ok"] is True
        char_line = next(l for l in result["lines"] if l["label"] == "Character Run 1 (obtained 1 of 1)")
        refund_pill = char_line["pills"][4]
        assert refund_pill["value"] == 0
        assert refund_pill["color"] == "cyan"
        assert refund_pill["tooltip"] == TOOLTIPS["refund"]

    def test_second_copy_win_with_unstated_refund_gets_dupe_bonus(self):
        # First copy stated explicitly (0 refunds), second copy's refund
        # unstated: the estimate for the second event must include the +2
        # dupe bonus since a copy of this banner was already obtained, and
        # the bonus renders as its own "+ 2" pair, not folded into the
        # base rate's own number.
        events = [
            _event(1, "character", "win", 20, 0),
            _event(2, "character", "win", 30, None),
        ]
        extracted = _blank(events=events, additional_character_copies_wanted=1)
        result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
        assert result["ok"] is True
        second_line = next(l for l in result["lines"] if l["label"] == "Character Run 2 (obtained 2 of 2)")
        # 0.1105 * 30 = 3.315 -> 3, base rate pill, unchanged by the dupe bonus.
        base_refund_pill = second_line["pills"][4]
        assert base_refund_pill["value"] == 3
        assert base_refund_pill["color"] == "light_green"
        # The +2 dupe bonus is its own separate "+" and number pill pair.
        assert second_line["pills"][5] == {"kind": "operator", "value": "+", "color": "magenta",
                                            "tooltip": TOOLTIPS["add_op"]}
        dupe_pill = second_line["pills"][6]
        assert dupe_pill["value"] == 2
        assert dupe_pill["color"] == "light_green"
        assert dupe_pill["tooltip"] == "A number of refunds wasn't given, and this is an additional character copy"
        # The ledger itself used the combined 3 + 2 = 5 total refund: first
        # copy 100 - 20 + 0 = 80, second copy 80 - 30 + 5 = 55.
        assert second_line["pills"][7] == {"kind": "operator", "value": "=", "color": "magenta",
                                            "tooltip": TOOLTIPS["equals_op"]}
        assert second_line["pills"][8]["value"] == 55

    def test_second_weapon_copy_win_with_unstated_refund_gets_dupe_bonus(self):
        # Same mechanic on the weapon banner (W2+, not just C1+ on
        # characters): the dupe bonus and its wording must say "weapon".
        events = [
            _event(1, "weapon", "win", 20, 0),
            _event(2, "weapon", "win", 40, None),
        ]
        extracted = _blank(events=events, additional_weapon_copies_wanted=1)
        result = reconcile(extracted, BASELINE_PARAMS, BASELINE_STATS)
        assert result["ok"] is True
        second_line = next(l for l in result["lines"] if l["label"] == "Weapon Run 2 (obtained 2 of 2)")
        # 0.0578 * 40 = 2.312 -> 2, base rate pill, unchanged by the dupe bonus.
        assert second_line["pills"][4]["value"] == 2
        assert second_line["pills"][5] == {"kind": "operator", "value": "+", "color": "magenta",
                                            "tooltip": TOOLTIPS["add_op"]}
        dupe_pill = second_line["pills"][6]
        assert dupe_pill["value"] == 2
        assert dupe_pill["tooltip"] == "A number of refunds wasn't given, and this is an additional weapon copy"
        # 100 - 20 + 0 = 80, then 80 - 40 + 4 = 44.
        assert second_line["pills"][8]["value"] == 44


class TestGoalOnlyHypotheticals:
    """A goal change with no events at all ("what if I go for C1") must
    reconcile to a displayed, deterministic goal, additive to the baseline
    by default, replacing it only when explicitly stated."""

    def test_extra_copy_adds_to_the_baseline_goal_and_shows_it(self):
        result = reconcile(_blank(events=[], additional_character_copies_wanted=1),
                           BASELINE_PARAMS, BASELINE_STATS)
        assert result["ok"] is True
        assert result["remaining_characters"] == 2 and result["remaining_weapons"] == 1
        goal = next(l for l in result["lines"] if l["label"] == "Current Goal")
        assert goal["pills"][0]["value"] == "2 characters and 1 weapon"
        assert result["total_pulls"] == BASELINE_PARAMS["total_pulls"]

    def test_explicit_replacement_overrides_the_baseline_goal(self):
        result = reconcile(_blank(events=[], characters_wanted_total=1, weapons_wanted_total=0),
                           BASELINE_PARAMS, BASELINE_STATS)
        assert result["ok"] is True
        assert result["remaining_characters"] == 1 and result["remaining_weapons"] == 0

    def test_no_events_and_no_goal_change_is_still_an_error(self):
        result = reconcile(_blank(events=[]), BASELINE_PARAMS, BASELINE_STATS)
        assert result["ok"] is False


def _step(order, banner, on_loss, copies=1, max_pulls=None, stop_at_pity=None, on_cap=None, skip_to_banner=None):
    return {"step_order": order, "banner_type": banner, "copies": copies, "on_loss": on_loss,
            "max_pulls": max_pulls, "stop_at_pity": stop_at_pity, "on_cap": on_cap,
            "skip_to_banner": skip_to_banner}


class TestConditionalPlan:
    """'Pull the weapon first; if I win, continue to the character; if I
    lose, stop': the past (events, added pulls) reconciles exactly as
    before, and the plan becomes the strategy every route branches from."""

    PARAMS = {**BASELINE_PARAMS, "total_pulls": 140}
    STATS = {**BASELINE_STATS, "initial_pulls": 140}
    PLAN = [_step(1, "weapon", "stop"), _step(2, "character", "continue")]

    def test_reported_question_reconciles_events_then_the_plan(self):
        # "I lost the 50/50 at 30 pity ... should I pull on the engine
        # banner first and if I win, continue on the character banner? If I
        # lose, I would stop there. I'm expecting around 120 pulls."
        result = reconcile(_blank(
            events=[_event(1, "character", "loss", 30, None)],
            pull_additions=[_add(120)], plan=self.PLAN,
        ), self.PARAMS, self.STATS)

        assert result["ok"] and result["plan"]
        labels = [line["label"] for line in result["lines"]]
        assert labels == ["Character Run 1 (obtained 0 of 1)", "Additional Pulls Mentioned",
                          "Starting Point for Every Route", "Current Goal"]
        assert _row(result["lines"][0]) == [140, "−", 30, "+", 3, "=", 113, "LOSS"]
        assert _row(result["lines"][1]) == [113, "+", 120, "=", 233]
        assert _row(result["lines"][2]) == [233, "CHARACTER", "0 pity", "GUARANTEE: TRUE",
                                            "WEAPON", "0 pity", "GUARANTEE: FALSE"]
        # Each guarantee says where it came from.
        char_flag, weapon_flag = result["lines"][2]["pills"][3], result["lines"][2]["pills"][6]
        assert char_flag["tooltip"].startswith("Guaranteed: Character Run 1 above lost its 50/50")
        assert weapon_flag["tooltip"] == "Carried over unchanged from the Starting Situation"
        assert _row(result["lines"][3]) == ["1 character and 1 weapon"]
        assert result["total_pulls"] == 233
        assert result["start_char_guarantee"] is True
        assert result["strategy"] == [
            {"banner": "weapon", "copies": 1, "stop_on_loss": True, "max_pulls": None, "cap_pity": None,
             "on_cap": None, "skip_to": None},
            {"banner": "char", "copies": 1, "stop_on_loss": False, "max_pulls": None, "cap_pity": None,
             "on_cap": None, "skip_to": None},
        ]

    def test_a_plan_alone_starts_from_the_form_situation(self):
        result = reconcile(_blank(plan=self.PLAN), self.PARAMS, self.STATS)
        assert result["ok"] and result["plan"]
        assert result["total_pulls"] == 140

    def test_a_pull_order_without_any_stop_is_ignored(self):
        plan = [_step(1, "weapon", "continue"), _step(2, "character", "continue")]
        result = reconcile(_blank(events=[_event(1, "character", "loss", 30, 3)], plan=plan),
                           self.PARAMS, self.STATS)
        assert result["ok"] and not result.get("plan")
        assert result["strategy"] == [{"banner": "char", "copies": 1}, {"banner": "weapon", "copies": 1}]

    def test_out_of_order_steps_return_error(self):
        plan = [_step(1, "weapon", "stop"), _step(3, "character", "continue")]
        result = reconcile(_blank(plan=plan), self.PARAMS, self.STATS)
        assert not result["ok"] and "step_order" in result["error"]

    def test_plan_copies_beyond_the_max_return_error(self):
        plan = [_step(1, "weapon", "stop", copies=6)]
        result = reconcile(_blank(plan=plan), self.PARAMS, self.STATS)
        assert not result["ok"] and "exceed the max" in result["error"]


class TestPlanLimits:
    PARAMS = {**BASELINE_PARAMS, "total_pulls": 233}
    STATS = {**BASELINE_STATS, "initial_pulls": 233}

    def test_a_capped_step_alone_makes_a_plan(self):
        # "Spend at most 80 on the character, then switch to the weapon."
        plan = [_step(1, "character", "continue", max_pulls=80, on_cap="continue"),
                _step(2, "weapon", "continue")]
        result = reconcile(_blank(plan=plan), self.PARAMS, self.STATS)
        assert result["ok"] and result["plan"]
        assert result["strategy"][0] == {"banner": "char", "copies": 1, "stop_on_loss": False,
                                         "max_pulls": 80, "cap_pity": None, "on_cap": "continue",
                                         "skip_to": None}

    def test_a_limit_without_on_cap_returns_error(self):
        plan = [_step(1, "character", "continue", max_pulls=80)]
        result = reconcile(_blank(plan=plan), self.PARAMS, self.STATS)
        assert not result["ok"] and "on_cap" in result["error"]

    def test_stop_at_pity_beyond_hard_pity_returns_error(self):
        plan = [_step(1, "character", "continue", stop_at_pity=95, on_cap="stop")]
        result = reconcile(_blank(plan=plan), self.PARAMS, self.STATS)
        assert not result["ok"] and "out of range 1-90" in result["error"]

    def test_stop_at_pity_must_be_above_the_banner_starting_pity(self):
        params = {**self.PARAMS, "start_char_pity": 76}
        plan = [_step(1, "character", "continue", stop_at_pity=75, on_cap="stop")]
        result = reconcile(_blank(plan=plan), params, self.STATS)
        assert not result["ok"] and "must be above the 76 pity" in result["error"]

    def test_stop_at_pity_passes_through_as_the_banner_counter(self):
        params = {**self.PARAMS, "start_char_pity": 20}
        plan = [_step(1, "character", "continue", stop_at_pity=75, on_cap="stop")]
        result = reconcile(_blank(plan=plan), params, self.STATS)
        assert result["strategy"][0]["cap_pity"] == 75
        assert result["start_char_pity"] == 20


class TestBudgetWhatIfs:
    def test_more_pulls_alone_reconciles_to_a_shown_total(self):
        result = reconcile(_blank(pull_additions=[_add(20)]), BASELINE_PARAMS, BASELINE_STATS)
        assert result["ok"] and result["budget_only"] and not result["goal_only"]
        assert _row(result["lines"][0]) == [100, "+", 20, "=", 120]
        assert result["total_pulls"] == 120

    def test_a_restated_total_alone_gets_its_own_line(self):
        result = reconcile(_blank(total_pulls_restated=200), BASELINE_PARAMS, BASELINE_STATS)
        assert result["budget_only"]
        assert result["lines"][0]["label"] == "Total Pulls Stated" and _row(result["lines"][0]) == [200]
        assert result["total_pulls"] == 200

    def test_a_reserve_is_set_aside_after_everything_else(self):
        result = reconcile(_blank(events=[_event(1, "character", "win", 30, 3)],
                                  pull_additions=[_add(20)], reserve_pulls=40),
                           BASELINE_PARAMS, BASELINE_STATS)
        assert not result["budget_only"]
        reserve_line = next(l for l in result["lines"] if l["label"] == "Pulls Kept in Reserve")
        assert _row(reserve_line) == [93, "−", 40, "=", 53]
        assert result["total_pulls"] == 53
        assert reserve_line["pills"][-1]["tooltip"] == TOOLTIPS["final_result"]

    def test_a_reserve_covering_everything_leaves_no_pulls(self):
        result = reconcile(_blank(reserve_pulls=150), BASELINE_PARAMS, BASELINE_STATS)
        assert result["ok"] and result["pulls_exhausted"]


class TestBuildRouteGroups:
    STRATEGY = [{"banner": "weapon", "copies": 1, "stop_on_loss": True},
                {"banner": "char", "copies": 1, "stop_on_loss": False}]

    @staticmethod
    def _result(nodes, leaves, trials=4000):
        """nodes: {path: (count, avg_leftover)}; the root is added here."""
        built = {(): {"count": trials, "avg_leftover": 233}}
        built.update({p: {"count": c, "avg_leftover": left} for p, (c, left) in nodes.items()})
        return {"trials": trials, "nodes": built, "leaves": leaves}

    def test_win_route_multiplies_and_loss_route_keeps_pulls(self):
        result = self._result(
            {("won",): (3000, 182.4), ("lost",): (1000, 183.1),
             ("won", "won"): (2970, 128.6), ("won", "ran_out"): (30, 0.2)},
            {("won", "won"): 2970, ("lost",): 1000, ("won", "ran_out"): 30},
        )
        # The character started guaranteed (an earlier lost 50/50): the win
        # route uses it up, the loss route never touches it.
        result["nodes"][("won", "won")]["guaranteed"] = {"char": 0.0, "weapon": 0.0}
        result["nodes"][("lost",)]["guaranteed"] = {"char": 1.0, "weapon": 1.0}
        routes, omitted = build_route_groups(self.STRATEGY, result)
        assert omitted is None
        assert [r["label"] for r in routes] == [
            "Route 1: Weapon won, then Character obtained",
            "Route 2: Weapon lost, stop",
        ]
        win, loss = routes
        assert [_row(l) for l in win["lines"]] == [
            [233, "WIN 75.00%", "182 LEFT"],
            [182, "WIN 99.00%", "RAN OUT 1.00%", "129 LEFT"],
            ["75.00%", "×", "99.00%", "=", "74.25%", "129 LEFT", "OBTAINED: 1 CHARACTER AND 1 WEAPON",
             "CHARACTER GUARANTEE: FALSE", "WEAPON GUARANTEE: FALSE"],
        ]
        assert win["chance"] == 74.25 and win["ending"] == "complete"
        assert [_row(l) for l in loss["lines"]] == [
            [233, "LOSS 25.00%", "183 KEPT"],
            ["25.00%", "183 KEPT", "OBTAINED: NONE", "CHARACTER GUARANTEE: TRUE", "WEAPON GUARANTEE: TRUE"],
        ]
        assert loss["chance"] == 25.0 and loss["ending"] == "lost" and loss["stopped_banner"] == "weapon"
        assert loss["end_guarantees"] == {"character": 1.0, "weapon": 1.0}
        assert loss["pulled_banners"] == ["weapon"]

    def test_a_mixed_guarantee_shows_its_share_of_runs(self):
        result = self._result({("won",): (4000, 180), ("won", "won"): (4000, 120)}, {("won", "won"): 4000})
        result["nodes"][("won", "won")]["guaranteed"] = {"char": 0.37, "weapon": 0.0}
        routes, _ = build_route_groups(self.STRATEGY, result)
        assert _row(routes[0]["lines"][-1])[-2:] == ["CHARACTER GUARANTEE: 37% OF RUNS", "WEAPON GUARANTEE: FALSE"]

    def test_a_capped_step_branches_into_its_own_route(self):
        strategy = [{"banner": "char", "copies": 1, "max_pulls": 80, "on_cap": "continue"},
                    {"banner": "weapon", "copies": 1}]
        routes, _ = build_route_groups(strategy, self._result(
            {("won",): (2000, 180), ("capped",): (2000, 162),
             ("won", "won"): (2000, 116), ("capped", "won"): (2000, 98)},
            {("won", "won"): 2000, ("capped", "won"): 2000},
        ))
        assert [r["label"] for r in routes] == [
            "Route 1: Character obtained, then Weapon obtained",
            "Route 2: Character capped, then Weapon obtained",
        ]
        capped = routes[1]
        assert capped["lines"][0]["label"] == "Run 1: Character, at most 80 pulls, then move on"
        assert _row(capped["lines"][0]) == [233, "CAPPED 50.00%", "162 LEFT"]
        # With kept pity known, the capped line shows what carries over.
        nodes = self._result({("won",): (2000, 180), ("capped",): (2000, 162),
                              ("won", "won"): (2000, 116), ("capped", "won"): (2000, 98)},
                             {("won", "won"): 2000, ("capped", "won"): 2000})
        nodes["nodes"][("capped",)]["avg_pity"] = 61.7
        routes, _ = build_route_groups(strategy, nodes)
        assert _row(routes[1]["lines"][0]) == [233, "CAPPED 50.00%", "62 PITY KEPT", "162 LEFT"]
        assert routes[1]["capped_pity"] == [("character", 62)]
        assert _row(capped["lines"][-1])[-1] == "OBTAINED: 1 WEAPON"
        assert capped["ending"] == "partial"

    def test_a_stop_route_no_run_took_is_omitted(self):
        routes, _ = build_route_groups(self.STRATEGY, self._result(
            {("won",): (4000, 180), ("won", "won"): (4000, 120)}, {("won", "won"): 4000},
        ))
        assert [r["label"] for r in routes] == ["Route 1: Weapon won, then Character obtained"]

    def test_a_rare_route_flags_its_leftover_as_approximate(self):
        rare = MIN_ROUTE_SAMPLE - 1
        routes, _ = build_route_groups(self.STRATEGY, self._result(
            {("won",): (4000 - rare, 180), ("lost",): (rare, 200), ("won", "won"): (4000 - rare, 120)},
            {("won", "won"): 4000 - rare, ("lost",): rare},
        ))
        kept = routes[1]["lines"][-1]["pills"][1]
        assert kept["value"] == "~200 KEPT"
        assert f"only {rare} simulated runs" in kept["tooltip"]

    def test_routes_beyond_the_max_are_summarized_not_drawn(self):
        strategy = [{"banner": "char", "copies": 1, "max_pulls": 50, "on_cap": "continue", "stop_on_loss": True},
                    {"banner": "weapon", "copies": 1, "max_pulls": 50, "on_cap": "continue", "stop_on_loss": True}]
        # 7 leaves: won/capped first, each followed by won/capped/lost, plus
        # a first-step loss. The rarest (400-run) leaf beyond 6 is dropped.
        nodes = {("won",): (1300, 150), ("capped",): (1300, 150), ("lost",): (1400, 150)}
        leaves = {("lost",): 1400}
        for first in ("won", "capped"):
            for second, count in (("won", 450), ("capped", 400), ("lost", 450)):
                nodes[(first, second)] = (count, 100)
                leaves[(first, second)] = count
        routes, omitted = build_route_groups(strategy, self._result(nodes, leaves))
        assert len(routes) == 6
        assert omitted == {"routes": 1, "chance": 10.0}


class TestSkipOnLoss:
    """'If I win the 50/50 I'll go for C1, then the weapon; if I lose, I'll
    go straight to the weapon': a loss moves the plan on to a named step."""

    STRATEGY = [{"banner": "char", "copies": 1, "skip_to": 2},
                {"banner": "char", "copies": 1},
                {"banner": "weapon", "copies": 1}]

    def test_skip_finds_the_next_step_on_the_named_banner(self):
        # The model names the banner; the C1 character step in between is
        # passed over because code, not the model, finds the weapon step.
        plan = [_step(1, "character", "skip", skip_to_banner="weapon"), _step(2, "character", "continue"),
                _step(3, "weapon", "continue")]
        result = reconcile(_blank(plan=plan), BASELINE_PARAMS, BASELINE_STATS)
        assert result["ok"] and [p["skip_to"] for p in result["strategy"]] == [2, None, None]
        nxt = reconcile(_blank(plan=[_step(1, "character", "skip"), _step(2, "weapon", "continue")]),
                        BASELINE_PARAMS, BASELINE_STATS)
        assert nxt["strategy"][0]["skip_to"] == 1

    def test_skip_must_point_at_a_later_step(self):
        for plan, why in (([_step(1, "character", "skip")], "this is the last step"),
                          ([_step(1, "character", "skip", skip_to_banner="weapon"), _step(2, "character", "continue")],
                           "no later step is on the weapon banner")):
            result = reconcile(_blank(plan=plan), BASELINE_PARAMS, BASELINE_STATS)
            assert not result["ok"] and why in result["error"]

    def test_a_switched_route_skips_the_passed_over_step(self):
        built = {(): {"count": 4000, "avg_leftover": 260}}
        for path, (count, left) in {
            ("won",): (2000, 205), ("won", "won"): (2000, 126), ("won", "won", "won"): (2000, 77),
            ("switched",): (2000, 205), ("switched", "skipped"): (2000, 205),
            ("switched", "skipped", "won"): (2000, 142),
        }.items():
            built[path] = {"count": count, "avg_leftover": left}
        built[("switched", "skipped", "won")]["guaranteed"] = {"char": 1.0, "weapon": 0.0}
        routes, _ = build_route_groups(self.STRATEGY, {
            "trials": 4000, "nodes": built,
            "leaves": {("won", "won", "won"): 2000, ("switched", "skipped", "won"): 2000},
        })
        assert [r["label"] for r in routes] == [
            "Route 1: Character won, then Character obtained, then Weapon obtained",
            "Route 2: Character lost, moved on, then Weapon obtained",
        ]
        switched = routes[1]
        assert [l["label"] for l in switched["lines"]] == [
            "Run 1: Character, if lost, skip to Run 3", "Run 3: Weapon", "Route Result",
        ]
        assert _row(switched["lines"][0]) == [260, "LOSS 50.00%", "205 LEFT"]
        assert _row(switched["lines"][-1]) == [
            "50.00%", "×", "100.00%", "=", "50.00%", "142 LEFT", "OBTAINED: 1 WEAPON",
            "CHARACTER GUARANTEE: TRUE", "WEAPON GUARANTEE: FALSE",
        ]
        assert switched["ending"] == "partial" and switched["end_guarantees"]["character"] == 1.0


class TestPullAdditions:
    """Pulls stated as additions, one entry per source, summed and
    converted by code in view, each pill saying where it came from."""

    @staticmethod
    def _line(result, label):
        return next(l for l in result["lines"] if l["label"] == label)

    def test_an_addition_names_its_source_not_a_refund(self):
        result = reconcile(_blank(pull_additions=[_add(120, "patch")]), BASELINE_PARAMS, BASELINE_STATS)
        line = self._line(result, "Additional Pulls Mentioned")
        assert _row(line) == [100, "+", 120, "=", 220]
        assert line["pills"][2]["tooltip"] == (
            "From this patch's rewards (new story, quests, exploration), as stated in your question")

    def test_several_sources_are_summed_in_view(self):
        # "About 60 from the patch, and I'll top up 40 more."
        result = reconcile(_blank(pull_additions=[_add(60, "patch"), _add(40, "purchase")]),
                           BASELINE_PARAMS, BASELINE_STATS)
        line = self._line(result, "Additional Pulls Mentioned")
        assert _row(line) == [100, "+", 60, "+", 40, "=", 200]
        assert line["pills"][4]["tooltip"].startswith("Bought with real money")
        assert result["total_pulls"] == 200

    def test_currency_converts_at_the_shown_rate(self):
        result = reconcile(_blank(pull_additions=[_add(16000, "savings", "currency"), _add(20, "event")]),
                           BASELINE_PARAMS, BASELINE_STATS)
        conversion = self._line(result, "Currency Converted to Pulls")
        assert _row(conversion) == [16000, "÷", 160, "=", 100]
        assert "Assumed cost of one pull: 160" in conversion["pills"][2]["tooltip"]
        additions = self._line(result, "Additional Pulls Mentioned")
        assert _row(additions) == [100, "+", 100, "+", 20, "=", 220]
        assert additions["pills"][2]["tooltip"] == (
            "From currency or pulls you already have saved, converted from currency above")

    def test_currency_rounds_down_to_whole_pulls(self):
        result = reconcile(_blank(pull_additions=[_add(1000, "purchase", "currency")]),
                           BASELINE_PARAMS, BASELINE_STATS)
        conversion = self._line(result, "Currency Converted to Pulls")
        assert _row(conversion) == [1000, "÷", 160, "=", 6]
        assert "40 currency left over" in conversion["pills"][-1]["tooltip"]
        assert result["total_pulls"] == 106

    def test_an_unknown_source_or_bad_amount_returns_error(self):
        for addition in (_add(20, "gift"), _add(0, "patch"), _add(20, "patch", "gems")):
            result = reconcile(_blank(pull_additions=[addition]), BASELINE_PARAMS, BASELINE_STATS)
            assert not result["ok"] and "pull_additions" in result["error"]

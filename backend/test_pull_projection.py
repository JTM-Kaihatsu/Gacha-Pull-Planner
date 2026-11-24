"""Tests for pull_projection.py: pure date arithmetic, no mocking needed."""
from datetime import date

from pull_projection import ProjectionSettings, generate_pull_projection, projection_events

# A Monday, and a patch that starts on it.
START = date(2026, 1, 5)
ANCHOR = date(2026, 1, 5)


def _only(**overrides):
    """Settings with every income source off, so one can be tested alone."""
    base = dict(
        daily_training_jade=0, express_pass=False, weekly_endgame_pulls=0,
        weekly_simulated_universe_pulls=0, hoyolab_checkin_jade=0, embers_exchange_pulls=0,
        patch_compensation_jade=0, patch_free_pulls=0, trial_jade_each=0, nameless_honor=False,
    )
    base.update(overrides)
    return ProjectionSettings(**base)


def _total(start, end, settings, anchor=ANCHOR):
    return generate_pull_projection(0, start, end, anchor, settings)["total_pulls"]


def _dates(start, end, settings, source, anchor=ANCHOR):
    return [d for d, label, _ in projection_events(start, end, anchor, settings) if label.startswith(source)]


class TestDailyIncome:
    def test_each_day_is_counted_exactly_once(self):
        # The old code paid daily income both as weekly rows and between
        # events; 20 days must be 20 days' worth.
        total = _total(START, date(2026, 1, 25), _only(daily_training_jade=60, express_pass=True,
                                                         express_pass_renewal_jade=0))
        assert total == round(20 * 150 / 160, 2)

    def test_no_express_pass_means_training_only_and_no_renewals(self):
        settings = _only(daily_training_jade=60, express_pass=False)
        assert _total(START, date(2026, 3, 5), settings) == round(59 * 60 / 160, 2)
        assert _dates(START, date(2026, 3, 5), settings, "Express Supply Pass Renewal") == []

    def test_renewals_every_30_days_from_the_first_one(self):
        settings = _only(express_pass=True, express_pass_daily_jade=0, express_pass_start=date(2026, 1, 20))
        assert _dates(START, date(2026, 3, 31), settings, "Express Supply Pass Renewal") == [
            date(2026, 1, 20), date(2026, 2, 19), date(2026, 3, 21),
        ]


class TestWeeklyAverages:
    def test_endgame_average_is_adjustable_and_credited_on_reset_day(self):
        # A player clearing about half the endgame each cycle.
        settings = _only(weekly_endgame_pulls=1.25)
        assert _dates(START, date(2026, 1, 26), settings, "Endgame") == [
            date(2026, 1, 12), date(2026, 1, 19), date(2026, 1, 26),
        ]
        assert _total(START, date(2026, 1, 26), settings) == 3.75

    def test_simulated_universe_completion_is_adjustable(self):
        assert _total(START, date(2026, 1, 19), _only(weekly_simulated_universe_pulls=1.0)) == 2.0

    def test_the_reset_weekday_is_adjustable(self):
        settings = _only(weekly_endgame_pulls=1, weekly_reset_weekday=2)  # Wednesday
        assert _dates(START, date(2026, 1, 15), settings, "Endgame") == [date(2026, 1, 7), date(2026, 1, 14)]


class TestPatchCycle:
    def test_patch_income_follows_the_patch_length_from_the_anchor(self):
        settings = _only(patch_compensation_jade=600, patch_length_days=42)
        # The anchor (start date) itself is excluded: already collected.
        assert _dates(START, date(2026, 4, 30), settings, "New Patch Compensation") == [
            date(2026, 2, 16), date(2026, 3, 30),
        ]

    def test_an_anchor_in_the_past_or_future_lays_out_the_same_cycle(self):
        settings = _only(patch_compensation_jade=600)
        for anchor in (date(2025, 1, 6), date(2027, 3, 1)):
            past_or_future = _dates(START, date(2026, 4, 30), settings, "New Patch", anchor=anchor)
            assert past_or_future and all((d - anchor).days % 42 == 0 for d in past_or_future)
            assert all((b - a).days == 42 for a, b in zip(past_or_future, past_or_future[1:]))

    def test_a_different_patch_length_moves_everything_tied_to_it(self):
        settings = _only(patch_compensation_jade=600, patch_length_days=35)
        assert _dates(START, date(2026, 3, 31), settings, "New Patch") == [date(2026, 2, 9), date(2026, 3, 16)]

    def test_character_trials_follow_the_banner_phase_length(self):
        settings = _only(trial_characters_per_phase=2, trial_jade_each=20, banner_phase_days=21)
        assert _dates(START, date(2026, 2, 28), settings, "Character Trials") == [
            date(2026, 1, 26), date(2026, 2, 16),
        ]
        assert _total(START, date(2026, 2, 28), settings) == round(2 * 40 / 160, 2)

    def test_nameless_honor_keys_off_the_patch_anchor(self):
        settings = _only(nameless_honor=True)
        rewards = [(d, label) for d, label, _ in projection_events(START, date(2026, 2, 20), ANCHOR, settings)]
        assert rewards == [
            (date(2026, 1, 12), "Nameless Honor Reward (1 Pull)"),
            (date(2026, 1, 22), "Nameless Honor Reward (2 Pulls)"),
            (date(2026, 1, 26), "Nameless Honor Reward (1 Pull)"),
            (date(2026, 2, 16), "Nameless Honor Purchase Bonus"),
        ]

    def test_nameless_honor_off_adds_nothing(self):
        assert _total(START, date(2026, 6, 1), _only(nameless_honor=False)) == 0


class TestMonthlyAndOneOff:
    def test_checkins_and_embers_land_on_their_days(self):
        settings = _only(hoyolab_checkin_jade=20, embers_exchange_pulls=5)
        events = projection_events(START, date(2026, 2, 5), ANCHOR, settings)
        assert [(d, label) for d, label, _ in events] == [
            (date(2026, 1, 12), "HoYoLAB Check-in"),
            (date(2026, 1, 19), "HoYoLAB Check-in"),
            (date(2026, 2, 1), "Embers Exchange Reset"),
            (date(2026, 2, 4), "HoYoLAB Check-in"),
        ]

    def test_extra_events_count_only_inside_the_window(self):
        settings = _only(extra_events=[(date(2026, 1, 20), "Anniversary", 1600),
                                       (date(2026, 9, 1), "Too late", 1600)])
        assert _total(START, date(2026, 2, 1), settings) == 10.0


class TestProjection:
    def test_rows_accumulate_from_current_pulls_and_round_down_at_the_end(self):
        result = generate_pull_projection(20, START, date(2026, 1, 12), ANCHOR, _only(weekly_endgame_pulls=2.5))
        assert result["rows"][0] == {"date": START, "source": "Starting Pulls", "pulls": 0.0, "cumulative_pulls": 20.0}
        assert result["rows"][-1]["cumulative_pulls"] == 22.5
        assert result["total_pulls"] == 22.5 and result["projected_pulls"] == 22

    def test_defaults_run_end_to_end(self):
        result = generate_pull_projection(0, START, date(2026, 3, 1), ANCHOR)
        assert "error" not in result and result["projected_pulls"] > 0
        cumulative = [r["cumulative_pulls"] for r in result["rows"]]
        assert cumulative == sorted(cumulative)

    def test_invalid_input_returns_an_error(self):
        assert "error" in generate_pull_projection(0, START, START, ANCHOR)
        assert "error" in generate_pull_projection(0, START, date(2026, 2, 1), None)
        assert "error" in generate_pull_projection(0, START, date(2026, 2, 1), ANCHOR,
                                                   ProjectionSettings(patch_length_days=0))

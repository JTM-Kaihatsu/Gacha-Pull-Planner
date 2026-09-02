"""Tests for the pure Monte Carlo engine in simulation.py."""
import pytest

from simulation import (
    CHAR_PITY_DEFAULTS,
    WEAPON_PITY_DEFAULTS,
    VIZ_SAMPLE_SIZE,
    banner_probability,
    simulate_combo_verbose,
    run_simulation_verbose,
    _build_copy_labels,
)


# ---------------------------------------------------------------------------
# banner_probability — pure, exact math
# ---------------------------------------------------------------------------
class TestBannerProbability:
    def test_below_soft_pity_is_base_rate(self):
        assert banner_probability(1, **CHAR_PITY_DEFAULTS) == 0.006
        assert banner_probability(72, **CHAR_PITY_DEFAULTS) == 0.006

    def test_at_soft_pity_start_ramps_by_one_increment(self):
        # pity == soft_pity_start enters the ramp: base + 1 * increment
        base, soft, hard = 0.006, 73, 90
        increment = (1.0 - base) / (hard - soft)
        assert banner_probability(73, base, soft, hard) == pytest.approx(base + increment)

    def test_reaches_one_at_hard_pity_minus_one(self):
        assert banner_probability(89, **CHAR_PITY_DEFAULTS) == pytest.approx(1.0)

    def test_at_and_above_hard_pity_is_one(self):
        assert banner_probability(90, **CHAR_PITY_DEFAULTS) == 1.0
        assert banner_probability(200, **CHAR_PITY_DEFAULTS) == 1.0

    def test_never_exceeds_one_and_is_monotonic(self):
        prev = -1.0
        for pity in range(0, 95):
            p = banner_probability(pity, **CHAR_PITY_DEFAULTS)
            assert 0.0 <= p <= 1.0
            assert p >= prev
            prev = p

    def test_weapon_defaults_boundaries(self):
        assert banner_probability(64, **WEAPON_PITY_DEFAULTS) == 0.008
        assert banner_probability(80, **WEAPON_PITY_DEFAULTS) == 1.0


# ---------------------------------------------------------------------------
# simulate_combo_verbose — single-run behavior
# ---------------------------------------------------------------------------
class TestSimulateCombo:
    def test_guaranteed_single_char_win_is_deterministic(self, always_hit_config):
        success, used, refunded, leftover, meta = simulate_combo_verbose(
            total_pulls=50,
            strategy=[{"banner": "char", "copies": 1}],
            start_char_guarantee=True,
            char_pity_config=always_hit_config,
        )
        assert success is True
        assert used == 1                      # first pull hits, guarantee wins it
        assert refunded == 0.0                # 5★ every pull -> 4★ branch never runs
        assert leftover == 49
        assert meta["chars_obtained"] == 1
        assert meta["char_50_50_wins"] == 1

    def test_guaranteed_loss_uses_all_pulls(self, never_hit_config):
        success, used, refunded, leftover, meta = simulate_combo_verbose(
            total_pulls=40,
            strategy=[{"banner": "char", "copies": 1}],
            simulate_4star=False,
            char_pity_config=never_hit_config,
        )
        assert success is False
        assert used == 40
        assert leftover == 0
        assert meta["chars_obtained"] == 0

    def test_zero_pulls_fails_nonempty_goal(self, always_hit_config):
        success, used, refunded, leftover, meta = simulate_combo_verbose(
            total_pulls=0,
            strategy=[{"banner": "char", "copies": 1}],
            start_char_guarantee=True,
            char_pity_config=always_hit_config,
        )
        assert success is False
        assert used == 0

    def test_empty_strategy_is_trivially_successful(self):
        # Low-level engine treats an empty goal as trivially met (0 >= 0). The API
        # layer rejects empty strategies before reaching here (see test_api.py).
        success, used, refunded, leftover, meta = simulate_combo_verbose(
            total_pulls=10, strategy=[]
        )
        assert success is True
        assert used == 0
        assert meta["desired_chars"] == 0
        assert meta["desired_weapons"] == 0

    def test_char_refund_toggle(self, never_hit_config):
        # Force 4★ every 10 pulls (hard 4★ pity) with no 5★ ever.
        with_refund = simulate_combo_verbose(
            total_pulls=30,
            strategy=[{"banner": "char", "copies": 1}],
            full_4star_chars=True,
            char_pity_config=never_hit_config,
        )
        without_refund = simulate_combo_verbose(
            total_pulls=30,
            strategy=[{"banner": "char", "copies": 1}],
            full_4star_chars=False,
            char_pity_config=never_hit_config,
        )
        assert with_refund[2] > 0            # refunded pulls > 0 when full-4★ toggle on
        assert without_refund[2] == 0.0      # no refunds when off

    def test_phases_share_pull_budget(self, never_hit_config, always_hit_config):
        # An earlier phase that never hits drains the shared budget, leaving
        # nothing for the later phase (order matters).
        _, used, _, _, meta = simulate_combo_verbose(
            total_pulls=20,
            strategy=[
                {"banner": "char", "copies": 1},
                {"banner": "weapon", "copies": 1},
            ],
            start_weapon_guarantee=True,
            simulate_4star=False,
            char_pity_config=never_hit_config,
            weapon_pity_config=always_hit_config,
        )
        copies = meta["copy_detail"]
        assert [c["banner"] for c in copies] == ["char", "weapon"]
        assert copies[0]["pulls_used"] == 20   # char phase eats the whole budget
        assert copies[1]["pulls_used"] == 0    # nothing left for the weapon phase
        assert used == 20

    def test_sequential_phases_both_won(self, always_hit_config):
        # Both banners guaranteed + always-hit -> each phase wins in one pull.
        success, used, _, _, meta = simulate_combo_verbose(
            total_pulls=20,
            strategy=[
                {"banner": "char", "copies": 1},
                {"banner": "weapon", "copies": 1},
            ],
            start_char_guarantee=True,
            start_weapon_guarantee=True,
            char_pity_config=always_hit_config,
            weapon_pity_config=always_hit_config,
        )
        assert success is True
        assert used == 2
        assert meta["chars_obtained"] == 1 and meta["weapons_obtained"] == 1
        assert [c["pulls_used"] for c in meta["copy_detail"]] == [1, 1]

    def test_multi_copy_phase_splits_into_one_entry_per_copy(self, always_hit_config):
        # A 2-copy character phase used to collapse into one aggregated
        # phase_detail entry ("C0-C1: N pulls combined"); it must now report
        # each copy's own pulls separately. Only the first copy's 50/50 is
        # guaranteed (the guarantee flag clears after a win), so only its
        # pull count is deterministic; the second copy's isn't, but the
        # split into two entries summing to the total used is.
        _, used, _, _, meta = simulate_combo_verbose(
            total_pulls=20,
            strategy=[{"banner": "char", "copies": 2}],
            start_char_guarantee=True,
            char_pity_config=always_hit_config,
        )
        assert meta["chars_obtained"] == 2
        copies = meta["copy_detail"]
        assert [c["banner"] for c in copies] == ["char", "char"]
        assert copies[0]["pulls_used"] == 1   # first copy's 50/50 was guaranteed
        assert sum(c["pulls_used"] for c in copies) == used

    def test_incomplete_run_pads_remaining_copies_with_zero(self, never_hit_config):
        # 3 character copies wanted but only 5 pulls available and the
        # banner never hits: the whole budget goes into the first copy
        # attempt, and the 2nd/3rd copies (never reached) must still appear
        # as zero-pull entries so every trial has the same number of
        # per-copy columns for the chart, regardless of outcome.
        _, used, _, _, meta = simulate_combo_verbose(
            total_pulls=5,
            strategy=[{"banner": "char", "copies": 3}],
            simulate_4star=False,
            char_pity_config=never_hit_config,
        )
        assert used == 5
        copies = meta["copy_detail"]
        assert len(copies) == 3
        assert [c["pulls_used"] for c in copies] == [5, 0, 0]

    def test_lost_5050_failure_flag_false_when_no_5star(self, never_hit_config):
        _, _, _, _, meta = simulate_combo_verbose(
            total_pulls=5,
            strategy=[{"banner": "char", "copies": 1}],
            simulate_4star=False,
            char_pity_config=never_hit_config,
        )
        assert meta["lost_char_50_50_failure"] == 0

    def test_return_contract(self, always_hit_config):
        result = simulate_combo_verbose(
            total_pulls=10,
            strategy=[{"banner": "char", "copies": 1}],
            start_char_guarantee=True,
            char_pity_config=always_hit_config,
        )
        success, used, refunded, leftover, meta = result
        assert isinstance(success, bool)
        assert isinstance(used, int)
        assert isinstance(refunded, float)
        for key in ("chars_obtained", "weapons_obtained", "copy_detail",
                    "desired_chars", "desired_weapons", "final_pity_char"):
            assert key in meta


# ---------------------------------------------------------------------------
# _build_copy_labels — pure, exact strings
# ---------------------------------------------------------------------------
class TestCopyLabels:
    def test_single_copies(self):
        assert _build_copy_labels([{"banner": "char", "copies": 1}]) == ["C0"]
        assert _build_copy_labels([{"banner": "weapon", "copies": 1}]) == ["W1"]

    def test_multi_copy_phase_gets_one_label_per_copy(self):
        # No more collapsed ranges ("C0-C2"): one label per individual copy,
        # matching copy_detail now having one entry per copy.
        assert _build_copy_labels([{"banner": "char", "copies": 3}]) == ["C0", "C1", "C2"]
        assert _build_copy_labels([{"banner": "weapon", "copies": 3}]) == ["W1", "W2", "W3"]

    def test_mixed_ordered_strategy_increments(self):
        strategy = [
            {"banner": "char", "copies": 1},
            {"banner": "weapon", "copies": 1},
            {"banner": "char", "copies": 1},
        ]
        assert _build_copy_labels(strategy) == ["C0", "W1", "C1"]

    def test_empty_strategy(self):
        assert _build_copy_labels([]) == []


# ---------------------------------------------------------------------------
# run_simulation_verbose — aggregation + response contract
# ---------------------------------------------------------------------------
# The exact keys main.py reads out of stats_summary; guards against FE↔BE drift.
API_KEYS = {
    "success_rate", "avg_pity_char", "avg_pity_weapon",
    "successes_char_win_rate", "successes_weapon_win_rate",
    "avg_leftover_pulls_on_success", "avg_refund_success",
    "failure_char_win_rate", "failure_weapon_win_rate",
    "avg_leftover_pulls_on_failure", "avg_refund_fail",
    "most_common_failure_state", "failure_state_distribution",
    "correlation_stats", "viz_sample", "trials",
    "desired_characters", "desired_weapons",
}


class TestRunSimulation:
    def test_returns_full_api_contract(self):
        out = run_simulation_verbose(
            total_pulls=120,
            strategy=[{"banner": "char", "copies": 1}, {"banner": "weapon", "copies": 1}],
            trials=200,
        )
        assert API_KEYS.issubset(out.keys())

    def test_success_rate_is_percentage_string(self):
        out = run_simulation_verbose(
            total_pulls=120,
            strategy=[{"banner": "char", "copies": 1}],
            trials=100,
        )
        assert isinstance(out["success_rate"], str)
        assert out["success_rate"].endswith("%")

    def test_viz_sample_capped_at_sample_size(self):
        out = run_simulation_verbose(
            total_pulls=80,
            strategy=[{"banner": "char", "copies": 1}],
            trials=VIZ_SAMPLE_SIZE + 500,
        )
        assert len(out["viz_sample"]) == VIZ_SAMPLE_SIZE

    def test_viz_sample_labels_match_copy_labels(self):
        strategy = [{"banner": "char", "copies": 1}, {"banner": "weapon", "copies": 1}]
        out = run_simulation_verbose(total_pulls=120, strategy=strategy, trials=50)
        expected = _build_copy_labels(strategy)
        for entry in out["viz_sample"]:
            assert [c["label"] for c in entry["copies"]] == expected

    def test_hundred_percent_success_has_no_failures(self, always_hit_config):
        out = run_simulation_verbose(
            total_pulls=50,
            strategy=[{"banner": "char", "copies": 1}],
            start_char_guarantee=True,
            char_pity_config=always_hit_config,
            trials=100,
        )
        assert out["success_rate"] == "100.00%"
        assert out["most_common_failure_state"] is None
        assert out["failure_state_distribution"] == []
        assert out["correlation_stats"]["total_failures"] == 0

    def test_zero_percent_success_degrades_gracefully(self, never_hit_config):
        out = run_simulation_verbose(
            total_pulls=30,
            strategy=[{"banner": "char", "copies": 1}],
            char_pity_config=never_hit_config,
            trials=100,
        )
        assert out["success_rate"] == "0.00%"
        assert out["avg_pity_char"] is None          # no successful runs
        assert out["most_common_failure_state"] is not None
        assert out["correlation_stats"]["total_successes"] == 0

    def test_failure_state_shape(self, never_hit_config):
        out = run_simulation_verbose(
            total_pulls=30,
            strategy=[{"banner": "char", "copies": 1}, {"banner": "weapon", "copies": 1}],
            char_pity_config=never_hit_config,
            weapon_pity_config=never_hit_config,
            trials=50,
        )
        for state in out["failure_state_distribution"]:
            assert set(state) == {"chars", "weapons", "pct"}
            assert 0 <= state["pct"] <= 100

    def test_trials_count_respected(self):
        out = run_simulation_verbose(
            total_pulls=50, strategy=[{"banner": "char", "copies": 1}], trials=321
        )
        assert out["trials"] == 321


class TestWinRateFormatting:
    def test_weapon_win_rate_is_na_when_no_weapon_goal(self, always_hit_config):
        # No weapon phase -> zero weapon encounters -> 'N/A' (not a literal 'nan%').
        out = run_simulation_verbose(
            total_pulls=50,
            strategy=[{"banner": "char", "copies": 1}],
            start_char_guarantee=True,
            char_pity_config=always_hit_config,
            trials=20,
        )
        assert out["successes_weapon_win_rate"] == "N/A"

    def test_failure_win_rates_are_na_on_full_success(self, always_hit_config):
        # 100% success -> no failed runs -> failure win rates are 'N/A', never 'nan%'.
        out = run_simulation_verbose(
            total_pulls=50,
            strategy=[{"banner": "char", "copies": 1}],
            start_char_guarantee=True,
            char_pity_config=always_hit_config,
            trials=20,
        )
        assert out["failure_char_win_rate"] == "N/A"
        assert "nan" not in out["successes_char_win_rate"].lower()

    def test_win_rate_is_percentage_when_encounters_exist(self, always_hit_config):
        out = run_simulation_verbose(
            total_pulls=50,
            strategy=[{"banner": "char", "copies": 1}],
            start_char_guarantee=True,
            char_pity_config=always_hit_config,
            trials=20,
        )
        assert out["successes_char_win_rate"].endswith("%")
        assert out["successes_char_win_rate"] != "N/A"

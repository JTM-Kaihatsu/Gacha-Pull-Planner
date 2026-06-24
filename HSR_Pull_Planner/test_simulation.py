import simulation
import numpy as np

def test_structure():
    np.random.seed(0)

    result = simulation.run_simulation_verbose(
        total_pulls=50,
        strategy=[{"banner": "char", "copies": 1}],
        start_char_pity=0,
        start_char_guarantee=False,
        start_lc_pity=0,
        start_lc_guarantee=False,
        trials=100,
    )

    # Ensure result is a dictionary
    assert isinstance(result, dict)

    # These are the keys we expect in the returned stats_summary
    expected_keys = {
        "success_rate",
        "avg_pity_char",
        "avg_pity_lc",
        "successes_char_win_rate",
        "successes_lc_win_rate",
        "avg_leftover_pulls_on_success",
        "avg_refund_success",
        "failure_char_win_rate",
        "failure_lc_win_rate",
        "avg_leftover_pulls_on_failure",
        "avg_refund_fail"
    }

    # Assert that all expected keys are present
    assert expected_keys.issubset(result.keys())

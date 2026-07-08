"""Shared pytest fixtures for the backend test suite.

Key idea: most engine tests are made deterministic with *boundary pity configs*
rather than only RNG seeding:

- ALWAYS_HIT_CONFIG (base_rate 1.0): every pull is a 5★, so with a starting
  guarantee a single copy is won in exactly one pull, with no 4★ branch reached.
- NEVER_HIT_CONFIG (base_rate 0.0, pity ceiling above any realistic pull count):
  no 5★ ever hits, so a goal can never be met.

An autouse fixture seeds both RNGs (numpy + stdlib random) for the statistical
tests that do rely on randomness.
"""
import random

import numpy as np
import pytest

# Every pull hits a 5★ (base_rate 1.0 saturates banner_probability at 1.0 everywhere).
ALWAYS_HIT_CONFIG = {"base_rate": 1.0, "soft_pity_start": 73, "hard_pity": 90}

# No 5★ ever hits, as long as pity stays under soft_pity_start (10_000 >> any test).
NEVER_HIT_CONFIG = {"base_rate": 0.0, "soft_pity_start": 10_000, "hard_pity": 10_001}


@pytest.fixture(autouse=True)
def _seed_rngs():
    """Seed both RNGs before every test for reproducibility."""
    np.random.seed(0)
    random.seed(0)


@pytest.fixture
def always_hit_config():
    return dict(ALWAYS_HIT_CONFIG)


@pytest.fixture
def never_hit_config():
    return dict(NEVER_HIT_CONFIG)


@pytest.fixture
def sample_stats():
    """A realistic full stats_summary from a small real simulation run.

    Used to exercise the analyzer's prompt formatting against the *actual* set
    of keys the engine emits (catches contract drift)."""
    from simulation import run_simulation_verbose

    return run_simulation_verbose(
        total_pulls=120,
        strategy=[
            {"banner": "char", "copies": 1},
            {"banner": "weapon", "copies": 1},
        ],
        start_char_pity=0,
        start_char_guarantee=False,
        start_weapon_pity=0,
        start_weapon_guarantee=False,
        trials=200,
    )


@pytest.fixture
def base_payload():
    """A valid /analyze request body (AI disabled by default)."""
    return {
        "total_pulls": 120,
        "start_char_pity": 0,
        "start_char_guarantee": False,
        "start_weapon_pity": 0,
        "start_weapon_guarantee": False,
        "strategy": [
            {"banner": "char", "copies": 1},
            {"banner": "weapon", "copies": 1},
        ],
    }

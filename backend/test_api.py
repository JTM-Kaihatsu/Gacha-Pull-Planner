import main
from fastapi.testclient import TestClient

def test_endpoint(monkeypatch):
    # ------------------------------------------------------------------
    # 1. Mocked flat dict result now includes 'trials'
    # ------------------------------------------------------------------
    dummy_stats = {
        "trials": 1,
        "success_rate": 1.0,
        "avg_pity_char": 73.0,
        "avg_pity_weapon": 73.0,
        "successes_char_win_rate": 1.0,
        "successes_weapon_win_rate": 1.0,
        "avg_leftover_pulls_on_success": 0,
        "avg_refund_success": 0,
        "failure_char_win_rate": 0.0,
        "failure_weapon_win_rate": 0.0,
        "avg_leftover_pulls_on_failure": 0,
        "avg_refund_fail": 0,
        "most_common_failure_state": None,
        "failure_state_distribution": [],
        "correlation_stats": {},
        "viz_sample": [],
    }

    # ------------------------------------------------------------------
    # 2. Patch the functions used in the API flow
    # ------------------------------------------------------------------
    monkeypatch.setattr(main, "run_simulation_verbose", lambda **_: dummy_stats)

    # ------------------------------------------------------------------
    # 3. Call the /analyze endpoint with valid JSON
    # ------------------------------------------------------------------
    client = TestClient(main.app)

    payload = {
        "total_pulls": 50,
        "start_char_pity": 0,
        "start_char_guarantee": False,
        "start_weapon_pity": 0,
        "start_weapon_guarantee": False,
        "strategy": [
            {"banner": "char", "copies": 1},
            {"banner": "weapon",   "copies": 1},
            {"banner": "char", "copies": 2},
        ],
    }

    response = client.post("/analyze", json=payload)

    # ------------------------------------------------------------------
    # 4. Assertions
    # ------------------------------------------------------------------
    assert response.status_code == 200
    data = response.json()

    assert set(data["summary"]) == {"confidence", "headline", "notes"}  # deterministic read always present
    assert data["trials"] == dummy_stats["trials"]
    for key, value in dummy_stats.items():
        if key != "trials":
            assert data["stats_summary"][key] == value


def _valid_payload():
    return {
        "total_pulls": 50,
        "start_char_pity": 0,
        "start_char_guarantee": False,
        "start_weapon_pity": 0,
        "start_weapon_guarantee": False,
        "strategy": [{"banner": "char", "copies": 1}],
    }


def test_missing_required_field_returns_422():
    client = TestClient(main.app)
    payload = _valid_payload()
    del payload["total_pulls"]
    assert client.post("/analyze", json=payload).status_code == 422


def test_bad_strategy_item_returns_422():
    client = TestClient(main.app)
    payload = _valid_payload()
    payload["strategy"] = [{"banner": "char"}]  # missing 'copies'
    assert client.post("/analyze", json=payload).status_code == 422


def test_empty_strategy_returns_422():
    client = TestClient(main.app)
    payload = _valid_payload()
    payload["strategy"] = []
    assert client.post("/analyze", json=payload).status_code == 422


def test_zero_copies_returns_422():
    client = TestClient(main.app)
    payload = _valid_payload()
    payload["strategy"] = [{"banner": "char", "copies": 0}]
    assert client.post("/analyze", json=payload).status_code == 422


def test_invalid_banner_returns_422():
    client = TestClient(main.app)
    payload = _valid_payload()
    payload["strategy"] = [{"banner": "relic", "copies": 1}]
    assert client.post("/analyze", json=payload).status_code == 422


_DUMMY_STATS = {
    "trials": 1, "success_rate": "0.00%", "avg_pity_char": None, "avg_pity_weapon": None,
    "successes_char_win_rate": "0%", "successes_weapon_win_rate": "0%",
    "avg_leftover_pulls_on_success": 0, "avg_refund_success": 0,
    "failure_char_win_rate": "0%", "failure_weapon_win_rate": "0%",
    "avg_leftover_pulls_on_failure": 0, "avg_refund_fail": 0,
    "most_common_failure_state": None, "failure_state_distribution": [],
    "correlation_stats": {}, "viz_sample": [],
}


def test_sim_failure_still_500(monkeypatch):
    """A real simulation failure (not the AI step) is still a 500."""
    def _boom(**_):
        raise RuntimeError("sim broke")

    monkeypatch.setattr(main, "run_simulation_verbose", _boom)
    client = TestClient(main.app)
    response = client.post("/analyze", json=_valid_payload())
    assert response.status_code == 500
    assert "sim broke" in response.json()["detail"]


def test_cors_allows_configured_origin(monkeypatch):
    dummy_stats = {
        "trials": 1, "success_rate": "0.00%", "avg_pity_char": None, "avg_pity_weapon": None,
        "successes_char_win_rate": "0%", "successes_weapon_win_rate": "0%",
        "avg_leftover_pulls_on_success": 0, "avg_refund_success": 0,
        "failure_char_win_rate": "0%", "failure_weapon_win_rate": "0%",
        "avg_leftover_pulls_on_failure": 0, "avg_refund_fail": 0,
        "most_common_failure_state": None, "failure_state_distribution": [],
        "correlation_stats": {}, "viz_sample": [],
    }
    monkeypatch.setattr(main, "run_simulation_verbose", lambda **_: dummy_stats)

    client = TestClient(main.app)
    response = client.post(
        "/analyze", json=_valid_payload(),
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


# --- /advise endpoint -------------------------------------------------------
def _advise_payload(**overrides):
    payload = _valid_payload()
    payload["question"] = "what if I had 40 more pulls?"
    payload.update(overrides)
    return payload


def test_advise_returns_answer_and_runs(monkeypatch):
    fake_runs = [{"total_pulls": 90, "desired_characters": 1, "desired_weapons": 1, "success_rate": "70.00%"}]
    monkeypatch.setattr(main, "run_simulation_verbose", lambda **_: _DUMMY_STATS | {"initial_pulls": 50})
    monkeypatch.setattr(main, "run_advisor", lambda *a, **k: ("You would reach about 70 percent.", fake_runs))

    client = TestClient(main.app)
    response = client.post("/advise", json=_advise_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "You would reach about 70 percent."
    assert data["status"] == "ok"
    assert data["runs"] == fake_runs


def test_advise_missing_question_returns_422():
    client = TestClient(main.app)
    payload = _valid_payload()  # no "question"
    assert TestClient(main.app).post("/advise", json=payload).status_code == 422


def test_advise_blank_question_returns_422():
    client = TestClient(main.app)
    assert client.post("/advise", json=_advise_payload(question="")).status_code == 422


def test_advise_degrades_gracefully(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("openai down")

    monkeypatch.setattr(main, "run_simulation_verbose", lambda **_: _DUMMY_STATS | {"initial_pulls": 50})
    monkeypatch.setattr(main, "run_advisor", _boom)

    client = TestClient(main.app)
    response = client.post("/advise", json=_advise_payload())
    assert response.status_code == 200          # not a 500
    assert response.json()["answer"] is None
    assert response.json()["status"] == "unavailable"


def test_advise_rate_limit_status(monkeypatch):
    class _RateLimited(Exception):
        status_code = 429

    def _boom(*a, **k):
        raise _RateLimited("slow down")

    monkeypatch.setattr(main, "run_simulation_verbose", lambda **_: _DUMMY_STATS | {"initial_pulls": 50})
    monkeypatch.setattr(main, "run_advisor", _boom)

    client = TestClient(main.app)
    response = client.post("/advise", json=_advise_payload())
    assert response.status_code == 200
    assert response.json()["status"] == "rate_limited"

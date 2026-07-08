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
    monkeypatch.setattr(main, "analyze_sim_result", lambda *_, **__: "analysis")
    monkeypatch.setattr(main, "get_openai_api_key", lambda: "sk-test")

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
        "enable_ai_analysis": True,
    }

    response = client.post("/analyze", json=payload)
    print(response.json())


    # ------------------------------------------------------------------
    # 4. Assertions
    # ------------------------------------------------------------------
    assert response.status_code == 200
    data = response.json()

    assert data["analysis_text"] == "analysis"
    assert data["analysis_status"] == "ok"
    assert data["trials"] == dummy_stats["trials"]
    for key, value in dummy_stats.items():
        if key != "trials":
            assert data["stats_summary"][key] == value


def test_ai_analysis_disabled_by_default(monkeypatch):
    """When enable_ai_analysis is omitted (default False), the analyzer is not
    called and analysis_text comes back null."""
    dummy_stats = {
        "trials": 1, "success_rate": 1.0, "avg_pity_char": 73.0, "avg_pity_weapon": 73.0,
        "successes_char_win_rate": 1.0, "successes_weapon_win_rate": 1.0,
        "avg_leftover_pulls_on_success": 0, "avg_refund_success": 0,
        "failure_char_win_rate": 0.0, "failure_weapon_win_rate": 0.0,
        "avg_leftover_pulls_on_failure": 0, "avg_refund_fail": 0,
        "most_common_failure_state": None, "failure_state_distribution": [],
        "correlation_stats": {}, "viz_sample": [],
    }

    called = {"analyzer": False}

    def _fail_if_called(*_, **__):
        called["analyzer"] = True
        return "should not appear"

    monkeypatch.setattr(main, "run_simulation_verbose", lambda **_: dummy_stats)
    monkeypatch.setattr(main, "analyze_sim_result", _fail_if_called)
    monkeypatch.setattr(main, "get_openai_api_key", lambda: "sk-test")

    client = TestClient(main.app)
    payload = {
        "total_pulls": 50,
        "start_char_pity": 0,
        "start_char_guarantee": False,
        "start_weapon_pity": 0,
        "start_weapon_guarantee": False,
        "strategy": [{"banner": "char", "copies": 1}],
    }

    response = client.post("/analyze", json=payload)

    assert response.status_code == 200
    assert response.json()["analysis_text"] is None
    assert response.json()["analysis_status"] == "disabled"
    assert called["analyzer"] is False


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


def test_ai_failure_degrades_gracefully(monkeypatch):
    """An AI failure must NOT 500 the whole request — the simulation result is
    still returned, with analysis_text null and status 'unavailable'."""
    def _boom(*_, **__):
        raise RuntimeError("openai down")

    monkeypatch.setattr(main, "run_simulation_verbose", lambda **_: _DUMMY_STATS)
    monkeypatch.setattr(main, "analyze_sim_result", _boom)

    client = TestClient(main.app)
    payload = _valid_payload()
    payload["enable_ai_analysis"] = True

    response = client.post("/analyze", json=payload)
    assert response.status_code == 200                       # not 500
    data = response.json()
    assert data["analysis_text"] is None
    assert data["analysis_status"] == "unavailable"
    assert data["stats_summary"]["success_rate"] == "0.00%"  # sim result intact


def test_ai_rate_limit_reports_rate_limited_status(monkeypatch):
    """A 429 from the AI provider maps to the 'rate_limited' status."""
    class _RateLimited(Exception):
        status_code = 429

    def _boom(*_, **__):
        raise _RateLimited("rate limit exceeded")

    monkeypatch.setattr(main, "run_simulation_verbose", lambda **_: _DUMMY_STATS)
    monkeypatch.setattr(main, "analyze_sim_result", _boom)

    client = TestClient(main.app)
    payload = _valid_payload()
    payload["enable_ai_analysis"] = True

    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    assert response.json()["analysis_status"] == "rate_limited"
    assert response.json()["analysis_text"] is None


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

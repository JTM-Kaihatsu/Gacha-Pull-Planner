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
        "avg_pity_lc": 73.0,
        "successes_char_win_rate": 1.0,
        "successes_lc_win_rate": 1.0,
        "avg_leftover_pulls_on_success": 0,
        "avg_refund_success": 0,
        "failure_char_win_rate": 0.0,
        "failure_lc_win_rate": 0.0,
        "avg_leftover_pulls_on_failure": 0,
        "avg_refund_fail": 0
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
        "start_lc_pity": 0,
        "start_lc_guarantee": False,
        "strategy": [
            {"banner": "char", "copies": 1},
            {"banner": "lc",   "copies": 1},
            {"banner": "char", "copies": 2},
        ]
    }

    response = client.post("/analyze", json=payload)
    print(response.json())


    # ------------------------------------------------------------------
    # 4. Assertions
    # ------------------------------------------------------------------
    assert response.status_code == 200
    data = response.json()

    assert data["analysis_text"] == "analysis"
    assert data["trials"] == dummy_stats["trials"]
    for key, value in dummy_stats.items():
        if key != "trials": 
            assert data["stats_summary"][key] == value

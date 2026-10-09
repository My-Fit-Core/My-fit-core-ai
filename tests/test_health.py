from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_analyze_mock_matches_schema():
    res = client.post("/v0/analyze", json={"job_id": "test", "image_url": "https://example.com/a.jpg"})
    assert res.status_code == 200
    body = res.json()
    assert body["job_id"] == "test"
    assert body["items"][0]["category"]

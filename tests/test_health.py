from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


_VALID_STATUSES = {"loaded", "not_loaded", "unavailable"}


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["api_version"]
    assert body["models"]["detector"] in _VALID_STATUSES
    assert body["models"]["tagger"] in _VALID_STATUSES


def test_health_models_field_exists():
    """ML 패키지 유무와 관계없이 models 필드가 있어야 한다."""
    body = client.get("/health").json()
    assert "models" in body
    assert "detector" in body["models"]
    assert "tagger" in body["models"]


def test_analyze_accepts_job():
    body = {
        "job_id": "test",
        "image_url": "https://example.com/a.jpg",
        "callback_url": "https://example.com/cb",
        "items": [{"client_item_key": "k1", "upload_url": "https://example.com/k1.png"}],
    }
    res = client.post("/v1/analyze", json=body)
    assert res.status_code == 200
    assert res.json() == {"job_id": "test", "job_status": "queued"}

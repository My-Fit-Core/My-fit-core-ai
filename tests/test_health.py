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


def test_analyze_accepts_job(monkeypatch):
    """접수 응답을 바로 주고 분석은 뒤에서 돈다(여기서는 실제 분석을 막는다)."""
    import app.main

    ran = []
    monkeypatch.setattr(app.main, "_run_job", lambda req: ran.append(req.job_id))
    body = {
        "job_id": "test",
        "image_url": "https://example.com/a.jpg",
        "callback_url": "https://example.com/cb",
        "items": [{"client_item_key": "k1", "upload_url": "https://example.com/k1.png"}],
    }
    res = client.post("/v1/analyze", json=body)
    assert res.status_code == 200
    assert res.json() == {"job_id": "test", "job_status": "queued"}
    assert ran == ["test"]


def test_secret_header_required_when_set(monkeypatch):
    monkeypatch.setenv("AI_SERVER_SECRET", "s3cret")
    res = client.post("/v1/recommend", json={"weather": {}, "main_style_id": 1, "wardrobe": []})
    assert res.status_code == 401
    res = client.post(
        "/v1/recommend",
        json={"weather": {}, "main_style_id": 1, "wardrobe": []},
        headers={"X-AI-Server-Secret": "s3cret"},
    )
    assert res.status_code != 401


def test_recommend_v1_example_request():
    """예시 요청(docs/api-v1-examples.json)이 /v1/recommend와 /swap을 통과한다."""
    import json
    from pathlib import Path

    examples = json.loads(
        (Path(__file__).resolve().parents[1] / "docs/api-v1-examples.json").read_text(encoding="utf-8")
    )
    req = examples["recommend_request"]
    res = client.post("/v1/recommend", json=req)
    assert res.status_code == 200, res.text
    sets = res.json()["sets"]
    assert sets and {"i1", "i2"} <= set(sets[0]["item_ids"])  # 상의+하의 뼈대

    swap = client.post("/v1/recommend/swap", json={**req, "outfit_item_ids": ["i1", "i2"], "slot": "bottom"})
    # 하의가 하나뿐이라 교체 후보가 없다
    assert swap.status_code == 422

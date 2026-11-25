"""분석 파이프라인(app/pipeline/analyze.py)과 콜백 서명(app/callback.py).

모델·네트워크 없이 가짜 검출기·태거·업로드로 확인한다.
"""

import io
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from PIL import Image

from app.callback import SIGNATURE_HEADER, TIMESTAMP_HEADER, send_result, sign, verify
from app.pipeline.analyze import UploadError, cutout_path, run_analysis
from app.pipeline.tagging import TagResult
from app.schemas import AnalyzeRequest, AnalyzeResult, UploadTarget

W, H = 200, 100


def _jpeg(size=(W, H)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(buf, format="JPEG")
    return buf.getvalue()


def _req(n=2, category=None) -> AnalyzeRequest:
    return AnalyzeRequest(
        job_id="j1",
        image_url="https://s/raw.jpg",
        callback_url="https://b/cb",
        items=[
            {"client_item_key": f"k{i}", "upload_url": f"https://s/k{i}.png?sig=x"} for i in range(n)
        ],
        category=category,
    )


def _det(x1, cls="short_sleeved_shirt", conf=0.9, y2=90):
    mask = np.zeros((H, W), dtype=bool)
    mask[10:y2, x1 : x1 + 50] = True
    return SimpleNamespace(bbox=(x1, 10.0, x1 + 50.0, float(y2)), conf=conf, df2_class=cls, mask=mask)


def _tag(image, bbox, mask, category=None):
    return TagResult(
        category=category or "top",
        subcategory="tshirt",
        category_candidates=[{"category": category or "top", "subcategory": "tshirt", "score": 0.8}],
        style_scores=[{"style_id": 1, "score": 0.4}],
        pattern="solid",
        fit="regular",
        off_shoulder=False,
        sleeve="long",  # 검출기 클래스가 있으면 이 값 대신 그걸 쓴다
        material="cotton_jersey",
        details=[],
        embedding=[1.0, 0.0],
    )


def _colors(image, mask):
    return [("#785028", 1.0)]


class Fake:
    """가짜 다운로드·업로드·검출기. 호출 기록을 남긴다."""

    def __init__(self, detections, fail_upload_keys=(), image=None):
        self.detections = detections
        self.fail = set(fail_upload_keys)
        self.uploads: dict[str, bytes] = {}
        self.max_items = None
        self.image = image if image is not None else _jpeg()

    def fetch(self, url):
        return self.image

    def upload(self, url, png):
        if any(k in url for k in self.fail):
            raise UploadError("503")
        self.uploads[url] = png

    def detect(self, image, max_items):
        self.max_items = max_items
        return self.detections[:max_items]


def _run(req, fake):
    return run_analysis(req, fake.fetch, fake.upload, detect=fake.detect, tag=_tag, colors=_colors)


def test_succeeded_two_items():
    fake = Fake([_det(0), _det(100, "trousers", 0.8)])
    res = _run(_req(2), fake)
    assert res.result_status == "succeeded" and not res.errors
    a, b = res.items
    assert (a.client_item_key, a.category, a.sleeve) == ("k0", "top", "short")  # 검출기 소매 우선
    assert (b.client_item_key, b.category) == ("k1", "bottom")
    assert b.sleeve == "long"  # 바지 클래스에는 소매가 없어 태깅 값
    assert a.bbox == [0.0, 0.1, 0.25, 0.9]
    assert a.cutout_path == "https://s/k0.png"  # path가 없으면 쿼리 뺀 업로드 URL
    assert res.embedding_model == "fashion-clip-vitb32"
    png = Image.open(io.BytesIO(fake.uploads["https://s/k0.png?sig=x"]))
    assert png.mode == "RGBA" and png.size == (50, 80)
    AnalyzeResult.model_validate(json.loads(res.model_dump_json()))


def test_detections_capped_by_upload_targets():
    fake = Fake([_det(0), _det(50), _det(100)])
    res = _run(_req(2), fake)
    assert len(res.items) == 2 and [i.bbox[0] for i in res.items] == [0.0, 0.25]


def test_duplicate_detection_dropped():
    """같은 하의를 두 번 잡으면(마스크가 거의 겹침) 신뢰도 높은 하나만 남긴다."""
    fake = Fake([_det(0), _det(100, "trousers", 0.8), _det(105, "skirt", 0.5)])
    res = _run(_req(3), fake)
    assert [i.category for i in res.items] == ["top", "bottom"]


def test_bottom_cut_at_frame_edge_dropped():
    """아래가 사진 밖으로 잘린 하의는 빼고 상의만 본다."""
    fake = Fake([_det(0), _det(100, "trousers", 0.8, y2=H)])
    res = _run(_req(2), fake)
    assert [i.category for i in res.items] == ["top"]


def test_only_cut_bottom_means_no_garment():
    res = _run(_req(1), Fake([_det(100, "trousers", 0.8, y2=H)]))
    assert res.errors[0].code == "NO_GARMENT_DETECTED"


def test_partial_failed_on_upload_error():
    res = _run(_req(2), Fake([_det(0), _det(100)], fail_upload_keys=["k1"]))
    assert res.result_status == "partial_failed"
    assert [i.client_item_key for i in res.items] == ["k0"]
    assert res.errors[0].code == "STORAGE_UNAVAILABLE" and res.errors[0].client_item_key == "k1"


def test_failed_when_no_garment():
    res = _run(_req(2), Fake([]))
    assert res.result_status == "failed" and not res.items
    assert res.errors[0].code == "NO_GARMENT_DETECTED" and res.errors[0].client_item_key is None


def test_failed_when_download_fails():
    fake = Fake([_det(0)])
    fake.fetch = lambda url: (_ for _ in ()).throw(httpx.ConnectError("down"))
    res = _run(_req(1), fake)
    assert res.errors[0].code == "STORAGE_UNAVAILABLE" and res.result_status == "failed"


def test_failed_on_invalid_image():
    res = _run(_req(1), Fake([_det(0)], image=b"not an image"))
    assert res.errors[0].code == "ANALYSIS_FAILED"


def test_direct_registration_without_detection_uses_full_image():
    """신발처럼 검출기에 없는 종류: 사진 전체를 한 벌로, 사용자가 고른 대분류로."""
    fake = Fake([])
    res = _run(_req(1, category="shoes"), fake)
    assert res.result_status == "succeeded"
    item = res.items[0]
    assert item.category == "shoes" and item.bbox == [0.0, 0.0, 1.0, 1.0]


def test_direct_registration_uses_matching_detection():
    """업로드 대상이 1개여도, 신뢰도 1위가 다른 대분류면 그 다음 검출을 본다."""
    fake = Fake([_det(0, "trousers", 0.95), _det(100, "short_sleeved_shirt", 0.6)])
    res = _run(_req(1, category="top"), fake)
    assert len(res.items) == 1 and res.items[0].bbox[0] == 0.5  # 상의 검출 영역


def test_model_error_on_item_is_retryable_code():
    def broken_tag(*a, **k):
        raise RuntimeError("CUDA out of memory")

    fake = Fake([_det(0)])
    res = run_analysis(_req(1), fake.fetch, fake.upload, detect=fake.detect, tag=broken_tag, colors=_colors)
    assert res.result_status == "failed" and res.errors[0].code == "MODEL_UNAVAILABLE"


def test_tagger_version_matches():
    from app.pipeline.analyze import TAGGER_VERSION
    from app.pipeline.tagging import MODEL_VERSION

    assert TAGGER_VERSION == MODEL_VERSION


def test_cutout_path_prefers_backend_path():
    t = UploadTarget(client_item_key="k", upload_url="https://s/x.png?sig=1", path="cutout/x.png")
    assert cutout_path(t) == "cutout/x.png"


# ── 콜백 ─────────────────────────────────────────────────────────


def test_sign_verify_roundtrip():
    body = b'{"job_id":"j1"}'
    headers = sign(body, "key", 1_000)
    assert headers[SIGNATURE_HEADER].startswith("sha256=")
    assert verify(body, headers, "key", now=1_100)
    assert not verify(body + b" ", headers, "key", now=1_100)  # 본문이 바뀌면 실패
    assert not verify(body, headers, "other", now=1_100)
    assert not verify(body, headers, "key", now=1_000 + 301)  # 5분 초과


def _result():
    return AnalyzeResult(job_id="j1", result_status="failed", model_version="m")


def test_send_result_signs_and_retries(monkeypatch):
    monkeypatch.setenv("AI_CALLBACK_HMAC_SECRET", "key")
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        return httpx.Response(500 if len(seen) < 3 else 200)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert send_result("https://b/cb", _result(), client, sleep=lambda s: None)
    assert len(seen) == 3
    last = seen[-1]
    headers = {TIMESTAMP_HEADER: last.headers[TIMESTAMP_HEADER], SIGNATURE_HEADER: last.headers[SIGNATURE_HEADER]}
    assert verify(last.content, headers, "key", now=int(last.headers[TIMESTAMP_HEADER]))


def test_send_result_invalid_url_does_not_raise(monkeypatch):
    monkeypatch.delenv("AI_CALLBACK_HMAC_SECRET", raising=False)
    with httpx.Client() as client:
        assert not send_result("not a url", _result(), client, sleep=lambda s: None)


def test_send_result_gives_up(monkeypatch):
    monkeypatch.delenv("AI_CALLBACK_HMAC_SECRET", raising=False)
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))) as client:
        assert not send_result("https://b/cb", _result(), client, sleep=lambda s: None)


@pytest.mark.parametrize("cls", ["vest", "sling_dress", "long_sleeved_outwear"])
def test_detector_classes_map_to_known_categories(cls):
    import yaml

    from app.pipeline.analyze import DETECTOR_CLASSES, ROOT

    cats = yaml.safe_load((ROOT / "configs/categories.yaml").read_text(encoding="utf-8"))["categories"]
    assert DETECTOR_CLASSES[cls]["category"] in cats

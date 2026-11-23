import os
import sys
import time
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")
from PIL import Image

from app.pipeline import detection
from app.pipeline.image_input import validate_image

DF2_CLASSES = {
    "short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear",
    "long_sleeved_outwear", "vest", "sling", "shorts", "trousers", "skirt",
    "short_sleeved_dress", "long_sleeved_dress", "vest_dress", "sling_dress",
}  # fmt: skip


def _ring(size: int = 20) -> "np.ndarray":
    mask = np.zeros((size, size), dtype=bool)
    mask[4:16, 4:16] = True
    mask[8:12, 8:12] = False  # 가운데 구멍
    return mask


def test_fill_holes_fills_center():
    out = detection.fill_holes(_ring())
    assert out.dtype == bool
    assert out[8:12, 8:12].all()
    assert out.sum() == 12 * 12


def test_fill_holes_empty_mask():
    out = detection.fill_holes(np.zeros((5, 5), dtype=bool))
    assert out.dtype == bool and not out.any()


def test_to_detections_sorts_and_limits():
    names = {0: "trousers", 1: "skirt", 2: "vest"}
    boxes = [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]]
    masks = np.stack([_ring()] * 3)
    out = detection.to_detections(boxes, [0.3, 0.9, 0.5], [0, 1, 2], masks, names, 2)
    assert [d.df2_class for d in out] == ["skirt", "vest"]
    assert [d.conf for d in out] == [0.9, 0.5]
    assert out[0].bbox == (1.0, 1.0, 2.0, 2.0)
    assert out[0].mask[8:12, 8:12].all()  # 구멍 메움 적용


def test_to_detections_no_masks():
    assert detection.to_detections([], [], [], None, {}, 6) == []


def test_load_detector_once(monkeypatch):
    calls = []

    class FakeYOLO:
        def __init__(self, path):
            calls.append(path)

    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO)
    )
    detection.load_detector.cache_clear()
    try:
        first = detection.load_detector(Path("fake.pt"))
        assert detection.load_detector(Path("fake.pt")) is first
        assert calls == ["fake.pt"]
    finally:
        detection.load_detector.cache_clear()


# 통합 테스트: 가중치가 있을 때만. 실제 사진은 DETECTOR_SAMPLE_IMAGE로 지정(커밋 금지).
needs_weights = pytest.mark.skipif(
    not detection.WEIGHTS_PATH.exists(),
    reason="가중치 없음 (scripts/download_weights.py)",
)
SAMPLE = os.getenv("DETECTOR_SAMPLE_IMAGE")


@needs_weights
def test_detect_blank_image_returns_empty():
    pytest.importorskip("ultralytics")
    assert detection.detect(Image.new("RGB", (640, 640), "white")) == []


@needs_weights
@pytest.mark.skipif(not SAMPLE, reason="DETECTOR_SAMPLE_IMAGE 없음")
def test_detect_sample_image():
    pytest.importorskip("ultralytics")
    image = validate_image(Path(SAMPLE).read_bytes()).image
    items = detection.detect(image)
    assert 1 <= len(items) <= 6
    for d in items:
        assert d.mask.shape == (image.height, image.width)
        assert d.mask.dtype == bool
        assert d.df2_class in DF2_CLASSES
        x1, y1, x2, y2 = d.bbox
        assert 0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height
    assert len(detection.detect(image, max_items=1)) == 1

    start = time.perf_counter()
    detection.detect(image)
    print(f"\nwarm detect: {time.perf_counter() - start:.3f}s, items={len(items)}")

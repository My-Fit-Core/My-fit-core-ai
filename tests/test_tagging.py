"""app/pipeline/tagging.py 단위 테스트.

crop_item / _softmax / TagResult 는 모델 없이 실행.
tag_item 통합 테스트는 open_clip 없으면 skip.
"""

import pytest

np = pytest.importorskip("numpy")
from PIL import Image

from app.pipeline.tagging import (
    _PATTERN_PROMPTS,
    MODEL_VERSION,
    TagResult,
    _softmax,
    crop_item,
)

# ── crop_item ────────────────────────────────────────────────────


def _full_mask(w: int, h: int) -> "np.ndarray":
    return np.ones((h, w), dtype=bool)


def test_crop_item_size():
    img = Image.new("RGB", (100, 100), (200, 100, 50))
    mask = _full_mask(100, 100)
    crop = crop_item(img, (10, 20, 70, 80), mask)
    assert crop.size == (60, 60)


def test_crop_item_outside_is_white():
    """마스크 밖 픽셀은 흰색."""
    img = Image.new("RGB", (100, 100), (200, 100, 50))
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True  # 중앙 20×20만 마스크
    crop = crop_item(img, (0, 0, 100, 100), mask)
    arr = np.asarray(crop)
    # 모서리(0,0)는 마스크 밖 → 흰색
    assert arr[0, 0].tolist() == [255, 255, 255]


def test_crop_item_inside_preserves_color():
    """마스크 안 픽셀은 원본 색 유지."""
    img = Image.new("RGB", (100, 100), (200, 100, 50))
    mask = _full_mask(100, 100)
    crop = crop_item(img, (10, 10, 90, 90), mask)
    arr = np.asarray(crop)
    assert arr[0, 0].tolist() == [200, 100, 50]


def test_crop_item_bbox_float():
    """bbox가 float여도 올바르게 반올림한다."""
    img = Image.new("RGB", (100, 100), "red")
    mask = _full_mask(100, 100)
    crop = crop_item(img, (10.4, 20.6, 70.2, 80.9), mask)
    assert crop.size == (60, 60)  # round(70.2)-round(10.4)=60, round(80.9)-round(20.6)=60


# ── _softmax ─────────────────────────────────────────────────────


def test_softmax_sums_to_one():
    result = _softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(result.sum() - 1.0) < 1e-6


def test_softmax_argmax_preserved():
    arr = np.array([0.1, 5.0, 0.2])
    assert np.argmax(_softmax(arr)) == 1


def test_softmax_uniform():
    result = _softmax(np.zeros(5))
    assert all(abs(v - 0.2) < 1e-6 for v in result)


# ── TagResult ────────────────────────────────────────────────────


def test_tag_result_default_model_version():
    r = TagResult(
        aesthetic_scores={"feminine": 0.5, "minimal": 0.5},
        category_candidates=[{"category": "top", "subcategory": "tshirt", "score": 0.9}],
        pattern="solid",
        embedding=[0.1] * 512,
    )
    assert r.model_version == MODEL_VERSION


def test_pattern_labels_complete():
    assert set(_PATTERN_PROMPTS.keys()) == {
        "solid", "stripe", "check", "dot", "floral", "animal", "graphic", "other"
    }


# ── tag_item 통합 (open_clip 필요) ───────────────────────────────


def test_tag_item_integration():
    pytest.importorskip("open_clip", reason="open_clip 없으면 skip")
    pytest.importorskip("torch", reason="torch 없으면 skip")

    from app.pipeline.tagging import tag_item

    img = Image.new("RGB", (200, 300), (200, 150, 100))
    mask = np.ones((300, 200), dtype=bool)
    bbox = (0.0, 0.0, 200.0, 300.0)

    result = tag_item(img, bbox, mask)

    assert set(result.aesthetic_scores.keys()) == {
        "feminine", "street", "minimal", "vintage_layered", "casual"
    }
    assert abs(sum(result.aesthetic_scores.values()) - 1.0) < 1e-5
    assert len(result.category_candidates) == 3
    for c in result.category_candidates:
        assert "category" in c and "subcategory" in c and "score" in c
    assert result.pattern in set(_PATTERN_PROMPTS.keys())
    assert len(result.embedding) > 0
    assert result.model_version == MODEL_VERSION

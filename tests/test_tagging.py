"""app/pipeline/tagging.py 단위 테스트.

crop_item / _softmax / TagResult 는 모델 없이 실행.
tag_item 통합 테스트는 transformers·torch 없으면 skip.
"""

import pytest

np = pytest.importorskip("numpy")
import yaml
from PIL import Image

from app.pipeline import tagging
from app.pipeline.tagging import (
    MODEL_VERSION,
    ROOT,
    TagResult,
    _attributes,
    _get_device,
    _Scorer,
    _softmax,
    _style_scores,
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
        category="top",
        subcategory="tshirt",
        category_candidates=[{"category": "top", "subcategory": "tshirt", "score": 0.9}],
        style_scores=[{"style_id": 1, "score": 0.3}],
        pattern="solid",
        fit=None,
        off_shoulder=None,
        sleeve=None,
        material=None,
        details=[],
        embedding=[0.1] * 512,
    )
    assert r.model_version == MODEL_VERSION


# ── 설정 일관성 ──────────────────────────────────────────────────


def _cfg(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_attribute_prompts_match_value_lists():
    attrs = _cfg("attributes.yaml")
    t = attrs["tagging"]
    assert list(t["pattern"]["prompts"]) == attrs["patterns"]
    assert list(t["material"]["prompts"]) == attrs["materials"]
    assert set(t["details"]["prompts"]) == set(attrs["details"])
    assert set(t["details"]["verified"]) <= set(attrs["details"])
    assert list(t["sleeve"]["prompts"]) == attrs["sleeve"]
    for prompts in t["fit"]["prompts"].values():
        assert list(prompts) == attrs["fit"]


def test_styles_match_style_ids():
    styles = {s["style_id"]: s["key"] for s in _cfg("styles.yaml")["styles"]}
    assert styles == _cfg("style_ids.yaml")["styles"]


# ── 판정 로직 (가짜 문장 임베딩, 모델 없이) ──────────────────────


@pytest.fixture
def fake_texts(monkeypatch):
    """문장마다 고정 벡터를 준다. register(text, vec)로 원하는 문장의 방향을 정한다."""
    table: dict[str, np.ndarray] = {}

    def register(text, vec):
        v = np.asarray(vec, dtype=float)
        table[text] = v / np.linalg.norm(v)

    def feats(texts):
        return np.stack([table.get(t, np.array([0.0, 0.0, 1.0])) for t in texts])

    monkeypatch.setattr(tagging, "_text_feats", feats)
    return register


def test_style_scores_independent(fake_texts):
    cfg = {
        "independent_score": {"temperature": 1.0, "neutral_prompts": ["neutral"]},
        "styles": [
            {"style_id": 1, "key": "a", "prompts": ["style a"]},
            {"style_id": 2, "key": "b", "prompts": ["style b"]},
        ],
    }
    fake_texts("neutral", [0, 0, 1])
    fake_texts("style a", [1, 0, 0])
    fake_texts("style b", [1, 0.2, 0])
    img = np.array([[1.0, 0.0, 0.0]])
    scores = {s["style_id"]: s["score"] for s in _style_scores(_Scorer(img, 10.0), cfg)}
    assert all(0 <= v <= 1 for v in scores.values())
    assert scores[1] > 0.5 and scores[2] > 0.5  # 둘 다 기준보다 가깝다
    assert scores[1] > scores[2]
    assert abs(sum(scores.values()) - 1.0) > 0.1  # 합 1이 아니다


def _attr_cfg():
    return _cfg("attributes.yaml")["tagging"]


def test_attributes_skip_by_category(fake_texts):
    img = np.array([[0.0, 0.0, 1.0]])  # 모든 문장과 같은 거리 → 확률이 고르게 나뉨
    out = _attributes(_Scorer(img, 100.0), _attr_cfg(), "shoes", "boots")
    assert out["fit"] is None and out["off_shoulder"] is None and out["sleeve"] is None
    assert out["details"] == []  # verified가 비어 있다


def test_sleeve_skipped_for_sleeveless(fake_texts):
    cfg = _attr_cfg()
    short, long_ = cfg["sleeve"]["prompts"]["short"], cfg["sleeve"]["prompts"]["long"]
    fake_texts(short, [1, 0, 0])
    fake_texts(long_, [0, 1, 0])
    img = np.array([[1.0, 0.0, 0.0]])
    assert _attributes(_Scorer(img, 100.0), cfg, "top", "tshirt")["sleeve"] == "short"
    assert _attributes(_Scorer(img, 100.0), cfg, "top", "sleeveless")["sleeve"] is None


def test_off_shoulder_thresholds(fake_texts):
    cfg = _attr_cfg()
    yes, no = cfg["off_shoulder"]["prompts"]
    fake_texts(yes, [1, 0, 0])
    fake_texts(no, [0, 1, 0])
    assert _attributes(_Scorer(np.array([[1.0, 0, 0]]), 100.0), cfg, "top", "knit")["off_shoulder"] is True
    assert _attributes(_Scorer(np.array([[0, 1.0, 0]]), 100.0), cfg, "top", "knit")["off_shoulder"] is False
    tie = np.array([[1.0, 1.0, 0]]) / np.sqrt(2)  # 반반 → 모름
    assert _attributes(_Scorer(tie, 100.0), cfg, "top", "knit")["off_shoulder"] is None


def test_tag_item_rejects_unknown_category():
    """모델을 불러오기 전에 잘못된 대분류를 거절한다."""
    from app.pipeline.tagging import tag_item

    img = Image.new("RGB", (10, 10))
    with pytest.raises(ValueError):
        tag_item(img, (0, 0, 10, 10), np.ones((10, 10), bool), category="foo")


def test_choose_returns_none_below_min_prob(fake_texts):
    img = np.array([[0.0, 0.0, 1.0]])
    assert _Scorer(img, 100.0).choose({"x": "p1", "y": "p2"}, 0.6) is None  # 0.5 < 0.6
    fake_texts("p2", [1, 0, 0])  # p1은 기본 벡터(이미지와 같은 방향)
    assert _Scorer(img, 100.0).choose({"x": "p1", "y": "p2"}, 0.6) == "x"


def test_get_device_returns_valid():
    """torch 없으면 skip, 있으면 유효한 device 문자열 반환."""
    pytest.importorskip("torch", reason="torch 없으면 skip")
    device = _get_device()
    assert device in ("cpu", "cuda", "mps")


# ── tag_item 통합 (transformers·torch 필요, 첫 실행 시 모델 다운로드) ──


def test_tag_item_integration():
    pytest.importorskip("transformers", reason="transformers 없으면 skip")
    pytest.importorskip("torch", reason="torch 없으면 skip")

    from app.pipeline.tagging import tag_item

    img = Image.new("RGB", (200, 300), (200, 150, 100))
    mask = np.ones((300, 200), dtype=bool)
    bbox = (0.0, 0.0, 200.0, 300.0)

    result = tag_item(img, bbox, mask)

    styles = yaml.safe_load((ROOT / "configs/styles.yaml").read_text(encoding="utf-8"))["styles"]
    assert [s["style_id"] for s in result.style_scores] == [s["style_id"] for s in styles]
    assert all(0 <= s["score"] <= 1 for s in result.style_scores)
    assert len(result.category_candidates) == 3
    assert result.category == result.category_candidates[0]["category"]
    patterns = yaml.safe_load((ROOT / "configs/attributes.yaml").read_text(encoding="utf-8"))["patterns"]
    assert result.pattern in patterns
    assert result.details == []
    assert len(result.embedding) == 512
    assert abs(np.linalg.norm(result.embedding) - 1.0) < 1e-4
    assert result.model_version == MODEL_VERSION

    shoe = tag_item(img, bbox, mask, category="shoes")
    assert shoe.category == "shoes"
    assert all(c["category"] == "shoes" for c in shoe.category_candidates)
    assert shoe.fit is None and shoe.off_shoulder is None


def test_subcategory_prompts_match_categories():
    subs = {s for v in _cfg("categories.yaml")["categories"].values() for s in v}
    assert set(_attr_cfg()["subcategory"]["prompts"]) <= subs


def test_ai_skip_subcategories_never_candidates(fake_texts):
    from app.pipeline.tagging import _category_candidates

    cats = {"top": ["tshirt", "inner_tshirt", "knit"]}
    fake_texts("inner", [1, 0, 0])
    img = np.array([[1.0, 0.0, 0.0]])
    prompts = {"inner_tshirt": ["inner"]}
    got = _category_candidates(_Scorer(img, 100.0), cats, "top", prompts, skip=["inner_tshirt"])
    assert "inner_tshirt" not in [c["subcategory"] for c in got]
    assert _category_candidates(_Scorer(img, 100.0), cats, "top", prompts)[0]["subcategory"] == "inner_tshirt"

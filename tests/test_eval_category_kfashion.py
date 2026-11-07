import numpy as np
import pytest
from PIL import Image

from scripts.eval_category_kfashion import CATEGORY_PART, PARTS, crop, summarize


def test_categories_cover_kfashion_21_and_known_parts():
    assert len(CATEGORY_PART) == 21
    assert set(CATEGORY_PART.values()) == set(PARTS)


def test_crop_clamps_box_and_whitens_outside_mask():
    image = Image.new("RGB", (4, 4), "black")
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    out = crop(image, [-1, -1, 2, 2], mask)
    assert out.size == (2, 2)
    assert out.getpixel((0, 0)) == (0, 0, 0)
    assert out.getpixel((1, 1)) == (255, 255, 255)


def _row(part, category, yolo, pred_part, pred_cat):
    row = {"part": part, "category": category, "yolo_part": yolo}
    for c in ("box", "masked"):
        row[f"part4_{c}"] = pred_part
        row[f"part_from_category_{c}"] = CATEGORY_PART[pred_cat]
        row[f"category_{c}"] = pred_cat
        row[f"top3_{c}"] = [pred_cat, category]
    return row


def test_summarize_rates_by_part():
    rows = [
        _row("아우터", "재킷", "상의", "아우터", "재킷"),
        _row("아우터", "코트", "상의", "상의", "셔츠"),
        _row("상의", "티셔츠", "상의", "상의", "티셔츠"),
    ]
    result = summarize(rows, missed=2)
    assert result["judged"] == 3 and result["missed_by_detector"] == 2
    outer = result["by_group"]["아우터"]
    assert outer["n"] == 2
    assert outer["part_yolo"] == 0
    assert outer["part4_box"] == pytest.approx(0.5)
    assert outer["part_from_category_masked"] == pytest.approx(0.5)
    assert outer["category_box"] == pytest.approx(0.5)
    assert outer["category_top3_box"] == pytest.approx(1)
    assert result["by_group"]["전체"]["part_yolo"] == pytest.approx(1 / 3)
    assert "원피스" not in result["by_group"]

import pytest

from scripts.eval_segmentation_kfashion import (
    parts_from_label,
    polygon_points,
    rect_to_xyxy,
    summarize,
)


def test_rect_to_xyxy():
    rect = {"X좌표": 10.5, "Y좌표": 20, "가로": 30, "세로": 40}
    assert rect_to_xyxy(rect) == [10.5, 20, 40.5, 60]


def test_polygon_points_orders_numerically_and_drops_unpaired():
    polygon = {
        "X좌표10": 3,
        "Y좌표10": 30,
        "X좌표2": 2,
        "Y좌표2": 20,
        "X좌표1": 1,
        "Y좌표1": 10,
        "X좌표11": 9,
    }
    assert polygon_points(polygon) == [(1, 10), (2, 20), (3, 30)]


def test_parts_from_label_keeps_only_parts_with_box_and_polygon():
    triangle = {
        "X좌표1": 0,
        "Y좌표1": 0,
        "X좌표2": 9,
        "Y좌표2": 0,
        "X좌표3": 0,
        "Y좌표3": 9,
    }
    rect = {"X좌표": 0, "Y좌표": 0, "가로": 9, "세로": 9}
    doc = {
        "데이터셋 정보": {
            "데이터셋 상세설명": {
                "렉트좌표": {
                    "상의": [rect],
                    "하의": [rect],
                    "아우터": [{}],
                    "원피스": [{}],
                },
                "폴리곤좌표": {
                    "상의": [triangle],
                    "하의": [{}],
                    "아우터": [triangle],
                    "원피스": [{}],
                },
            }
        }
    }
    assert [p["part"] for p in parts_from_label(doc)] == ["상의"]


def test_assign_masks_matches_by_overlap_when_a_mask_is_missing():
    np = pytest.importorskip("numpy")
    from scripts.eval_segmentation_kfashion import assign_masks, box_mask, mask_iou

    second = box_mask([5, 5, 10, 10], (10, 10))
    boxes = [[0, 0, 4, 4], [5, 5, 10, 10]]

    assigned = assign_masks([second], boxes)

    assert assigned[0] is None
    assert assigned[1] is second
    assert mask_iou(second, second) == 1.0
    assert mask_iou(second, np.zeros((10, 10), dtype=bool)) == 0.0


def test_postprocess_fills_holes_and_drops_fragments():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from scripts.eval_segmentation_kfashion import (
        VARIANTS,
        count_components,
        postprocess_mask,
    )

    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    mask[40:50, 40:50] = False  # 구멍
    mask[2:6, 90:94] = True  # 떨어진 조각

    assert count_components(mask) == 2

    result = postprocess_mask(mask, **VARIANTS["all"])
    assert result[45, 45] and result[50, 50]
    assert not result[3, 91]

    largest = postprocess_mask(mask, largest=True)
    assert not largest[3, 91] and not largest[45, 45]

    filled = postprocess_mask(mask, fill=True)
    assert filled[45, 45] and filled[3, 91]

    assert (postprocess_mask(mask) == mask).all()
    assert not postprocess_mask(np.zeros((10, 10), dtype=bool), largest=True).any()


def test_summarize_counts_low_iou_and_missing():
    rows = [
        {"part": "상의", "iou_raw": 0.8, "iou": 0.9, "iou_fill": 0.7, "missing": False},
        {"part": "하의", "iou_raw": 0.0, "iou": 0.0, "iou_fill": 0.0, "missing": True},
    ]
    result = summarize(rows, [1.0, 3.0], [0.1, 0.3])
    assert result["mean_iou"] == pytest.approx(0.45)
    assert result["mean_iou_raw"] == pytest.approx(0.4)
    assert result["improved_by_postprocess"] == 1
    assert result["worsened_by_postprocess"] == 0
    assert result["variants"] == {
        "fill": {"mean_iou": pytest.approx(0.35), "improved": 0, "worsened": 1}
    }
    assert result["mean_postprocess_seconds_per_image"] == pytest.approx(0.2)
    assert result["iou_under_0.5"] == pytest.approx(0.5)
    assert result["missing_masks"] == 1
    assert result["mean_seconds_per_image"] == pytest.approx(2.0)
    assert result["by_part"]["상의"] == {"n": 1, "mean_iou": 0.9}

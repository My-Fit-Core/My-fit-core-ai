import pytest
import yaml

from scripts.eval_aesthetics_on_kfashion import (
    axis_value,
    calibrate,
    mapped_hit_rate,
    pair_confusion,
    softmax,
    summarize,
    top1_counts,
)
from scripts.eval_kfashion_styles import ROOT, Record


def test_summarize_means_lift_and_top_counts():
    records = [
        Record(1, "A", None, (1, 1)),
        Record(2, "A", None, (2, 1)),
        Record(3, "B", None, (3, 1)),
    ]
    rows = [[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]]

    result = summarize(records, rows, ["A", "B"], 2)

    assert result["overall"] == pytest.approx([0.5, 0.5])
    assert result["by_style"]["A"]["n"] == 2
    assert result["by_style"]["A"]["mean"] == pytest.approx([0.7, 0.3])
    assert result["by_style"]["A"]["lift"] == pytest.approx([0.2, -0.2])
    assert result["by_style"]["A"]["top"] == [2, 0]
    assert result["by_style"]["B"]["top"] == [0, 1]


def test_aesthetic_mapping_uses_known_styles():
    load = lambda name: yaml.safe_load(
        (ROOT / "configs" / name).read_text(encoding="utf-8")
    )
    known = {s["name"] for s in load("kfashion_styles.yaml")["styles"]}
    for a in load("aesthetics.yaml")["aesthetics"]:
        mapping = a["kfashion"]
        assert mapping["styles"] or mapping.get("note"), a[
            "key"
        ]  # 연결 없음은 사유 필수
        for field in ("styles", "secondary", "excluded"):
            assert set(mapping.get(field, [])) <= known, (a["key"], field)


def test_calibrate_removes_constant_bias():
    rows = [[0.6, 0.4], [0.8, 0.2]]
    result = calibrate(rows, [0.7, 0.3])
    assert all(sum(r) == pytest.approx(1) for r in result)
    # 늘 높던 첫 항목이 1위를 덜 차지한다
    assert top1_counts(rows, 2) == [2, 0]
    assert top1_counts(result, 2) == [1, 1]


def test_mapped_hit_rate_axis_value_softmax():
    records = [
        Record(1, "A", None, (1, 1)),
        Record(2, "A", None, (2, 1)),
        Record(3, "B", None, (3, 1)),
    ]
    rows = [[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]]
    assert mapped_hit_rate(records, rows, [{"A"}, {"B"}]) == [0.5, 1.0]
    assert axis_value([0.0, 0.0]) == pytest.approx(0.5)
    assert axis_value([0.0, 2.0]) > 0.5
    assert softmax([1.0, 1.0]) == pytest.approx([0.5, 0.5])


def test_prompt_sets_and_axes_are_complete():
    config = yaml.safe_load(
        (ROOT / "configs/aesthetics.yaml").read_text(encoding="utf-8")
    )
    for a in config["aesthetics"]:
        assert a["prompts"]["outfit"], a["key"]
    for section in ("calibration", "calibration_item"):
        prior = config[section]["prior"]
        assert set(prior) == {a["key"] for a in config["aesthetics"]}, section
        assert sum(prior.values()) == pytest.approx(1, abs=0.01), section
    for axis in config["axes"]:
        assert len(axis["prompts"]) >= 2, axis["key"]
        assert all(len(pair) == 2 for pair in axis["prompts"]), axis["key"]


def test_hit_rate_skips_unmapped_and_pair_confusion():
    records = [
        Record(1, "A", None, (1, 1)),
        Record(2, "A", None, (2, 1)),
        Record(3, "B", None, (3, 1)),
    ]
    # 추구미 0(연결 A), 1(연결 없음), 2(연결 없음)
    rows = [[0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.1, 0.3, 0.6]]
    mapped = [{"A"}, set(), set()]
    assert mapped_hit_rate(records, rows, mapped) == [0.5, None, None]
    c = pair_confusion(records, rows, mapped, 1, 0)
    assert c["a_top1_on_b_styles"] == 0.5  # A 사진 중 1위가 추구미 1인 비율
    assert c["b_top1_on_a_styles"] is None  # 추구미 1은 연결 없음
    assert c["top1_either"] == 2
    assert c["top2_is_other"] == 1  # 두 사진 모두 2위가 상대

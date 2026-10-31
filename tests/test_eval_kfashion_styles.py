import pytest

from scripts.eval_kfashion_styles import Record, dedupe, evaluate, sample_per_style


def rec(image_id, main, sub=None, content=None):
    return Record(image_id, main, sub, content or (image_id, 1))


def test_dedupe_keeps_one_when_labels_match():
    kept, stats = dedupe(
        [rec(2, "A", "B", (9, 9)), rec(1, "A", "B", (9, 9)), rec(3, "A")]
    )
    assert sorted(r.image_id for r in kept) == [1, 3]
    assert stats == {"dup_groups": 1, "dup_groups_conflicting": 0, "images_removed": 1}


@pytest.mark.parametrize(
    "other",
    [rec(2, "B", None, (9, 9)), rec(2, "A", "B", (9, 9)), rec(2, None, None, (9, 9))],
    ids=["main_differs", "sub_differs", "no_style"],
)
def test_dedupe_drops_group_when_labels_differ(other):
    kept, stats = dedupe([rec(1, "A", None, (9, 9)), other, rec(3, "A")])
    assert [r.image_id for r in kept] == [3]
    assert stats == {"dup_groups": 1, "dup_groups_conflicting": 1, "images_removed": 2}


def test_sample_is_reproducible_balanced_and_skips_unlabeled():
    records = (
        [rec(i, "A") for i in range(20)]
        + [rec(i, "B") for i in range(20, 40)]
        + [rec(99, None)]
    )
    first = sample_per_style(records, ["A", "B"], 5, seed=42)
    assert first == sample_per_style(list(reversed(records)), ["A", "B"], 5, seed=42)
    assert first != sample_per_style(records, ["A", "B"], 5, seed=43)
    assert sum(r.main == "A" for r in first) == sum(r.main == "B" for r in first) == 5
    assert all(r.main for r in first)


def test_sample_stops_when_style_has_too_few_images():
    with pytest.raises(SystemExit, match="B"):
        sample_per_style(
            [rec(1, "A"), rec(2, "A"), rec(3, "B")], ["A", "B"], 2, seed=42
        )


def test_evaluate_counts_sub_style_and_balances_per_style():
    records = [
        rec(1, "A", "B"),
        rec(2, "A"),
        rec(3, "B"),
        rec(4, "B"),
        rec(5, "B"),
        rec(6, "B"),
    ]
    result = evaluate(records, ["B", "A", "B", "A", "A", "A"], ["A", "B", "C"])

    assert result["per_style"]["A"] == {
        "n": 2,
        "accuracy": 1.0,
        "main_only_accuracy": 0.5,
    }
    assert result["per_style"]["B"] == {
        "n": 4,
        "accuracy": 0.25,
        "main_only_accuracy": 0.25,
    }
    assert result["balanced_accuracy"] == pytest.approx((1.0 + 0.25) / 2)
    assert result["balanced_main_only_accuracy"] == pytest.approx((0.5 + 0.25) / 2)
    assert result["confusion_matrix"] == [[1, 1, 0], [3, 1, 0], [0, 0, 0]]
    assert result["chance_main_only"] == pytest.approx(1 / 3)
    assert result["chance_with_sub"] == pytest.approx(7 / 6 / 3)

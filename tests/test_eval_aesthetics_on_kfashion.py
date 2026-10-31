import pytest

from scripts.eval_aesthetics_on_kfashion import summarize
from scripts.eval_kfashion_styles import Record


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

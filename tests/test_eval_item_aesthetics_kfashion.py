import re

import pytest
import yaml

from scripts.eval_item_aesthetics_kfashion import style_logits, top1_by_category
from scripts.eval_kfashion_styles import ROOT


def test_style_logits_averages_prompts_and_scales():
    sims = [0.2, 0.4, 0.1]
    assert style_logits(sims, [[0, 1], [2]], 100) == pytest.approx([30, 10])


def test_top1_by_category_counts_and_handles_missing_category():
    items = [{"category": "청바지"}, {"category": "청바지"}, {"category": None}]
    rows = [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]
    assert top1_by_category(items, rows, 2) == {"청바지": [1, 1], "(없음)": [0, 1]}


def test_item_prompts_filled_and_casual_street_items_do_not_overlap():
    aesthetics = {
        a["key"]: a
        for a in yaml.safe_load(
            (ROOT / "configs/aesthetics.yaml").read_text(encoding="utf-8")
        )["aesthetics"]
    }
    for key, a in aesthetics.items():
        assert 3 <= len(a["prompts"]["item"]) <= 4, key
    # 캐주얼과 스트리트는 같은 옷 종류를 쓰지 않는다(issue #44, #35).
    garments = ["t-shirt", "hoodie", "sweatshirt", "jeans", "cargo", "knit", "shirt"]

    def used(key):
        text = " ".join(aesthetics[key]["prompts"]["item"])
        # "t-shirt"·"sweatshirt" 안의 "shirt"는 셔츠로 세지 않는다.
        return {g for g in garments if re.search(rf"(?<![\w-]){g}\b", text)}

    assert not used("casual") & used("street")

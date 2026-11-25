"""scripts/eval_style_prompts_cached.py의 지표 계산 (모델 없이)."""

from types import SimpleNamespace

import numpy as np

from scripts.eval_style_prompts_cached import auc, evaluate, sample


def test_auc_basic():
    assert auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auc(np.array([1.0]), np.array([2.0])) == 0.0
    assert auc(np.array([1.0]), np.array([1.0])) == 0.5  # 동점


def test_sample_excludes_and_reports_short():
    recs = [SimpleNamespace(image_id=i, main="a" if i < 5 else "b") for i in range(10)]
    out, short = sample(recs, ["a", "b"], 3, seed=0, exclude_ids={0, 1, 2})
    assert not {r.image_id for r in out} & {0, 1, 2}
    assert short == {"a": 2}  # a는 3,4만 남음


def test_evaluate_hit_and_auc():
    recs = [SimpleNamespace(main=m) for m in ["x", "x", "y", "y"]]
    aes = [{"key": "A", "name": "A", "kfashion": {"styles": ["x"]}}, {"key": "B", "name": "B", "kfashion": {"styles": ["y"]}}]
    logits = np.array([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
    per, summ = evaluate(recs, logits, aes)
    assert per[0]["hit"] == 1.0 and per[1]["hit"] == 0.5
    assert per[0]["top1"] == 3
    assert summ["chance"] == 0.5

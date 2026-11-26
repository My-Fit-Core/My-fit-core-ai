"""scripts/eval_reference_styles.py 지표·분할 (모델 없이)."""

import numpy as np

from scripts.eval_reference_styles import KEYS, evaluate, prototype_scores, split_mask


def test_split_mask_halves_each_subtag():
    rows = [{"file": f"{i}", "sub_tag": "a" if i < 4 else "b"} for i in range(7)]
    m = split_mask(rows)
    assert m[:4].sum() == 2 and m[4:].sum() == 2  # 4장 → 2, 3장 → 2


def test_prototype_excludes_self():
    emb = np.eye(3)[[0, 0, 1]].astype(float)
    truth = np.zeros((3, len(KEYS)), bool)
    truth[:, 0] = [True, True, False]
    s = prototype_scores(emb, truth)
    assert s[0, 0] == 1.0  # 자기 자신이 아닌 1번 사진과 같은 방향
    assert s[2, 0] == 0.0


def test_evaluate_multilabel_hit():
    truth = np.zeros((2, len(KEYS)), bool)
    truth[0, [0, 1]] = True  # 두 추구미 정답
    truth[1, 2] = True
    scores = np.zeros((2, len(KEYS)))
    scores[0, 1] = 1  # 1위가 정답 중 하나
    scores[1, 0] = 1  # 틀림
    r = evaluate(scores, truth)
    per = {p["key"]: p for p in r["per"]}
    assert per[KEYS[0]]["hit"] == 1.0 and per[KEYS[1]]["hit"] == 1.0 and per[KEYS[2]]["hit"] == 0.0

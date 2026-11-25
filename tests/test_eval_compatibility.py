import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from scripts.eval_compatibility_kfashion import (
    auc,
    fitb_accuracy,
    group_split,
    near_dup_groups,
    pair_features,
    sample_negatives,
)


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_fitb_and_auc_rank_true_pair_first():
    rng = np.random.default_rng(0)
    top = _unit(rng.normal(size=(40, 8)))
    bottom = top.copy()  # 정답 하의 = 상의와 같은 방향 → cosine이 항상 최대

    def cos(ti, bi):
        return np.sum(top[ti] * bottom[bi], axis=1)

    idx = np.arange(40)
    acc, n = fitb_accuracy(cos, idx, 9, np.random.default_rng(1))
    assert acc == 1.0 and n == 40
    negs = sample_negatives(idx, 4, np.random.default_rng(2))
    assert all(a != b for a, b in negs)
    ni, nj = np.array([a for a, _ in negs]), np.array([b for _, b in negs])
    assert auc(cos(idx, idx), cos(ni, nj)) == 1.0


def test_hard_negatives_stay_in_style():
    styles = np.array(["a", "a", "a", "b", "b"])
    negs = sample_negatives(np.arange(5), 3, np.random.default_rng(0), styles)
    assert all(styles[i] == styles[j] and i != j for i, j in negs)


def test_fitb_skips_when_pool_too_small():
    styles = np.array(["a", "a", "b"])
    _, n = fitb_accuracy(
        lambda t, b: np.zeros(len(t)), np.arange(3), 9, np.random.default_rng(0), styles
    )
    assert n == 0


def test_near_dup_groups_and_split_no_leak():
    base = _unit(np.eye(6)[:, :6].astype(float))
    top = np.vstack([base, base[:2]])  # 6, 7번 = 0, 1번 복제
    bottom = top.copy()
    groups = near_dup_groups(top, bottom, 0.95)
    assert groups[6] == groups[0] and groups[7] == groups[1]
    assert len(np.unique(groups)) == 6
    for seed in range(5):
        train, test = group_split(groups, 0.3, seed)
        assert not set(groups[train]) & set(groups[test])
        assert len(train) + len(test) == 8


def test_pair_features_shape():
    a = np.ones((3, 4))
    assert pair_features(a, a).shape == (3, 16)

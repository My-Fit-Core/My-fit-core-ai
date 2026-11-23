import re

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from PIL import Image

from app.pipeline.color import extract_colors


def _full(img: Image.Image) -> "np.ndarray":
    return np.ones((img.height, img.width), dtype=bool)


def test_solid():
    img = Image.new("RGB", (64, 64), (255, 0, 0))
    assert extract_colors(img, _full(img)) == [("#FF0000", 1.0)]


def test_outside_mask_ignored():
    img = Image.new("RGB", (64, 64), "white")
    img.paste((20, 30, 90), (16, 16, 48, 48))
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    assert extract_colors(img, mask) == [("#141E5A", 1.0)]


def _stripes(w: int = 100, h: int = 100) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):  # 10px 주기 중 7px 검정, 3px 흰색
        arr[:, x] = (0, 0, 0) if x % 10 < 7 else (255, 255, 255)
    return Image.fromarray(arr)


def test_stripes_ratio_and_format():
    img = _stripes()
    out = extract_colors(img, _full(img))
    assert [c for c, _ in out] == ["#000000", "#FFFFFF"]
    assert out[0][1] == pytest.approx(0.7, abs=0.02)
    assert abs(sum(r for _, r in out) - 1) < 1e-6
    assert all(re.fullmatch(r"#[0-9A-F]{6}", c) for c, _ in out)


def test_more_colors_than_k():
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    for i, c in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        arr[i * 10 : (i + 1) * 10] = c
    img = Image.fromarray(arr)
    out = extract_colors(img, _full(img), k=3)
    assert len(out) == 3
    assert [r for _, r in out] == sorted((r for _, r in out), reverse=True)


def test_empty_mask():
    img = Image.new("RGB", (10, 10))
    assert extract_colors(img, np.zeros((10, 10), dtype=bool)) == []


def test_mask_shape_mismatch():
    with pytest.raises(ValueError):
        extract_colors(Image.new("RGB", (10, 10)), np.ones((5, 5), dtype=bool))


def test_large_image_sampled_and_deterministic():
    img = _stripes(2000, 2000)
    mask = _full(img)
    first = extract_colors(img, mask)
    assert first == extract_colors(img, mask)
    assert [c for c, _ in first] == ["#000000", "#FFFFFF"]
    assert first[0][1] == pytest.approx(0.7, abs=0.02)

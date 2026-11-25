import io

import pytest
from PIL import Image

from app.pipeline.image_input import (
    ALLOWED_FORMATS,
    HARD_MAX_SIDE,
    MAX_BYTES,
    ImageValidationError,
    validate_image,
)


def _encode(img: Image.Image, fmt: str, **kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def _reason(data: bytes) -> str:
    with pytest.raises(ImageValidationError) as e:
        validate_image(data)
    return e.value.reason


def test_valid_jpeg():
    out = validate_image(_encode(Image.new("RGB", (1024, 768), "red"), "JPEG"))
    assert out.image.mode == "RGB"
    assert out.image.size == (1024, 768)
    assert out.warnings == []


def test_png_rgba_converted_to_rgb():
    img = Image.new("RGBA", (600, 600), (0, 0, 255, 128))
    out = validate_image(_encode(img, "PNG"))
    assert out.image.mode == "RGB"


def test_valid_webp():
    out = validate_image(_encode(Image.new("RGB", (700, 600), "green"), "WEBP"))
    assert out.image.mode == "RGB"
    assert out.image.size == (700, 600)
    assert out.warnings == []


def test_mpo_jpeg_accepted():
    frames = [Image.new("RGB", (600, 600), c) for c in ("red", "blue")]
    data = _encode(frames[0], "MPO", save_all=True, append_images=frames[1:])
    assert Image.open(io.BytesIO(data)).format == "MPO"
    assert validate_image(data).image.size == (600, 600)


def test_config_loaded():
    assert ALLOWED_FORMATS == {"JPEG", "PNG", "WEBP"}
    assert MAX_BYTES == 20 * 1024 * 1024


@pytest.mark.parametrize("fmt", ["GIF", "BMP"])
def test_unsupported_format(fmt):
    assert _reason(_encode(Image.new("RGB", (600, 600)), fmt)) == "unsupported_format"


def test_not_an_image():
    assert _reason(b"not an image") == "corrupt_image"


def test_truncated_jpeg():
    data = _encode(Image.effect_noise((800, 800), 50).convert("RGB"), "JPEG")
    assert _reason(data[: len(data) // 2]) == "corrupt_image"


def test_file_too_large():
    assert _reason(b"\0" * (MAX_BYTES + 1)) == "file_too_large"


def test_long_side_resized_not_rejected():
    """2048px보다 크면 거절하지 않고 비율을 유지해 줄인다(안건 6)."""
    ok = validate_image(_encode(Image.new("RGB", (2048, 600)), "PNG"))
    assert ok.image.size == (2048, 600)
    big = validate_image(_encode(Image.new("RGB", (4096, 1200)), "PNG"))
    assert big.image.size == (2048, 600)


def test_hard_side_limit():
    too_long = _encode(Image.new("RGB", (HARD_MAX_SIDE + 1, 10)), "PNG")
    assert _reason(too_long) == "long_side_exceeded"


def test_short_side_warning():
    out = validate_image(_encode(Image.new("RGB", (800, 300)), "JPEG"))
    assert out.warnings == ["short_side_below_512"]


def test_exif_rotation_applied_and_stripped():
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: 시계 방향 90도 회전 필요
    exif[0x010F] = "TestCam"  # Make
    out = validate_image(_encode(Image.new("RGB", (100, 50)), "JPEG", exif=exif))
    assert out.image.size == (50, 100)
    assert "exif" not in out.image.info
    assert len(out.image.getexif()) == 0
    # 다시 저장해도 EXIF가 따라가지 않는다
    assert "exif" not in Image.open(io.BytesIO(_encode(out.image, "JPEG"))).info

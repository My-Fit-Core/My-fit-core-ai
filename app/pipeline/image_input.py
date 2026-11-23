"""입력 이미지 검증 (02-specs.md §3 입력 이미지 규칙).

프론트가 이미 처리했더라도 AI 서버에서 한 번 더 확인한다.
reason/warning 문자열은 모듈 내부 값이다. 응답 오류 코드 형식은 API v1(T6)에서 확정한다.
"""

import io
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image, ImageOps, UnidentifiedImageError

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "image_input.yaml"
_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

ALLOWED_FORMATS = set(_config["allowed_formats"])
MAX_BYTES = int(_config["max_mb"] * 1024 * 1024)
MAX_LONG_SIDE = 2048
MIN_SHORT_SIDE = 512


class ImageValidationError(ValueError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class ValidatedImage:
    image: Image.Image  # RGB, EXIF 회전 적용, EXIF 제거
    warnings: list[str] = field(default_factory=list)


def validate_image(data: bytes) -> ValidatedImage:
    if len(data) > MAX_BYTES:
        raise ImageValidationError("file_too_large", f"{len(data)} bytes > {MAX_BYTES}")
    try:
        img = Image.open(io.BytesIO(data))
        # 휴대폰 JPEG 중 Multi-Picture 정보가 있으면 Pillow는 MPO로 읽는다
        fmt = "JPEG" if img.format == "MPO" else img.format
        if fmt not in ALLOWED_FORMATS:
            raise ImageValidationError("unsupported_format", f"format {img.format}")
        # 픽셀 디코딩 전에 헤더 크기로 거절한다 (디컴프레션 폭탄 방어)
        if max(img.size) > MAX_LONG_SIDE:
            raise ImageValidationError("long_side_exceeded", f"size {img.size}")
        img.load()
        img = ImageOps.exif_transpose(img)
        # ponytail: 알파는 버리기만 한다(투명 픽셀 아래 RGB가 그대로 드러남). 흰 배경 합성 여부는 미정
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ImageValidationError("corrupt_image", str(e)) from e

    img.info.pop("exif", None)
    img.getexif().clear()
    warnings = ["short_side_below_512"] if min(img.size) < MIN_SHORT_SIDE else []
    return ValidatedImage(image=img, warnings=warnings)

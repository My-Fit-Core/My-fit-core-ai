"""분석 파이프라인: 사진 받기 → 검증 → 검출·누끼 → 업로드 → 태깅·색 → 결과 (F-201~203, #101).

입출력은 API v1 스키마(app/schemas.py)다. 네트워크(사진 다운로드, 누끼 업로드)와 모델은
인자로 받아 테스트에서 바꿔 끼울 수 있게 했다. 결과 전달(콜백)은 app/callback.py.
"""

import io
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import yaml
from PIL import Image

from app.pipeline.image_input import ImageValidationError, validate_image
from app.schemas import (
    MAX_ITEMS,
    AnalyzedItem,
    AnalyzeRequest,
    AnalyzeResult,
    CategoryCandidate,
    ColorInfo,
    ErrorCode,
    ErrorDetail,
    StyleScore,
    UploadTarget,
)

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DETECTOR_CLASSES: dict[str, dict] = yaml.safe_load(
    (ROOT / "configs/categories.yaml").read_text(encoding="utf-8")
)["detector_classes"]
DETECTOR_VERSION = "deepfashion2-yolov8s-seg"
TAGGER_VERSION = "fashion-clip-vitb32"  # app.pipeline.tagging.MODEL_VERSION과 같아야 한다(테스트가 확인)

# 같은 옷을 두 번 검출한 것으로 보는 기준: 마스크가 작은 쪽의 이 비율 이상 겹치면(같은 대분류) 하나만 남긴다
DUPLICATE_OVERLAP = 0.6
# 아래가 잘린 하의·원피스: 마스크가 사진 맨 아래 이 비율(높이 기준) 안에 닿으면 잘린 것으로 보고 뺀다.
# 입은 사진에서 하의가 프레임 밖으로 나가면 기장·핏을 알 수 없어서다(사용자 의견: 종아리까지 보여야 함).
# ponytail: 프레임 끝에 닿는지로만 본다. 종아리 위치를 직접 보려면 자세 추정 모델이 필요하다
BOTTOM_EDGE_RATIO = 0.01
TRUNCATION_CHECKED = {"bottom", "dress"}

Fetch = Callable[[str], bytes]
Upload = Callable[[str, bytes], None]


@dataclass
class Region:
    """누끼를 딸 영역 하나. bbox는 픽셀 xyxy, mask는 이미지 크기 bool."""

    bbox: tuple[float, float, float, float]
    mask: np.ndarray
    category: str | None  # 검출기 클래스나 사용자가 정한 대분류
    sleeve: str | None = None  # 검출기 클래스에서 알 수 있을 때만


class JobError(Exception):
    """사진 전체가 실패한 경우. 결과는 failed가 된다."""

    def __init__(self, code: ErrorCode, message: str):
        super().__init__(message)
        self.code = code


def cutout_png(image: Image.Image, region: Region) -> bytes:
    """마스크 밖을 투명하게 하고 bbox로 자른 RGBA PNG."""
    rgba = image.convert("RGBA")
    alpha = Image.fromarray((region.mask.astype(np.uint8) * 255), mode="L")
    rgba.putalpha(alpha)
    x1, y1, x2, y2 = (round(v) for v in region.bbox)
    buf = io.BytesIO()
    rgba.crop((x1, y1, x2, y2)).save(buf, format="PNG")
    return buf.getvalue()


def cutout_path(target: UploadTarget) -> str:
    """백엔드가 path를 주면 그대로, 아니면 upload_url에서 쿼리(서명)를 뺀 값."""
    if target.path:
        return target.path
    parts = urlsplit(target.upload_url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def normalize_bbox(bbox, size: tuple[int, int]) -> list[float]:
    w, h = size
    x1, y1, x2, y2 = bbox
    return [
        min(max(v, 0.0), 1.0) for v in (x1 / w, y1 / h, x2 / w, y2 / h)
    ]


def find_regions(image: Image.Image, req: AnalyzeRequest, detect) -> list[Region]:
    """검출기로 옷 영역을 찾는다. 신뢰도 높은 순, 업로드 대상 개수까지만.

    사용자가 대분류를 정한 직접 등록(req.category)은 한 벌로 본다. 그 대분류의 옷이 검출되면
    그 마스크로 누끼를 따고, 아니면(신발·가방 등 검출기에 없는 종류) 사진 전체를 쓴다.
    ponytail: 검출기에 없는 종류는 배경을 지우지 못한다. 누끼 모델(#41)을 붙이면 여기서 바꾼다.
    """
    full = Region(
        bbox=(0.0, 0.0, float(image.width), float(image.height)),
        mask=np.ones((image.height, image.width), dtype=bool),
        category=req.category,
    )
    # 중복·잘린 옷을 거른 뒤에 업로드 대상 개수로 자르므로 상한까지 받는다(신뢰도 높은 순)
    detections = detect(image, max_items=MAX_ITEMS)
    regions = []
    for d in detections:
        cls = DETECTOR_CLASSES.get(d.df2_class, {})
        regions.append(Region(d.bbox, d.mask, cls.get("category"), cls.get("sleeve")))
    regions = drop_duplicates(regions)

    if req.category:
        same = [r for r in regions if r.category == req.category]
        return [same[0] if same else full]
    regions = [r for r in regions if not is_cut_at_bottom(r, image.height)]
    if not regions:
        raise JobError(ErrorCode.NO_GARMENT_DETECTED, "옷을 찾지 못했습니다")
    return regions[: len(req.items)]


def drop_duplicates(regions: list[Region]) -> list[Region]:
    """같은 옷을 두 번 잡은 검출을 뺀다. 앞(신뢰도 높은 것)을 남긴다."""
    kept: list[Region] = []
    for r in regions:
        area = r.mask.sum()
        dup = any(
            k.category == r.category
            and area > 0
            and (k.mask & r.mask).sum() / min(area, k.mask.sum() or 1) >= DUPLICATE_OVERLAP
            for k in kept
        )
        if not dup:
            kept.append(r)
    return kept


def is_cut_at_bottom(region: Region, height: int) -> bool:
    """하의·원피스의 마스크가 사진 맨 아래에 닿으면(프레임 밖으로 이어지면) True."""
    if region.category not in TRUNCATION_CHECKED:
        return False
    rows = max(1, round(height * BOTTOM_EDGE_RATIO))
    return bool(region.mask[-rows:].any())


def _analyze_item(image, region: Region, target: UploadTarget, upload: Upload, tag, colors):
    upload(target.upload_url, cutout_png(image, region))
    t = tag(image, region.bbox, region.mask, category=region.category)
    return AnalyzedItem(
        client_item_key=target.client_item_key,
        cutout_path=cutout_path(target),
        bbox=normalize_bbox(region.bbox, image.size),
        category=t.category,
        subcategory=t.subcategory,
        category_candidates=[CategoryCandidate(**c) for c in t.category_candidates],
        colors=[ColorInfo(hex=h, ratio=r) for h, r in colors(image, region.mask)],
        pattern=t.pattern,
        style_scores=[StyleScore(**s) for s in t.style_scores],
        fit=t.fit,
        off_shoulder=t.off_shoulder,
        # 검출기 클래스(short/long sleeved)가 있으면 그걸 쓴다
        sleeve=region.sleeve or t.sleeve,
        material=t.material,
        details=t.details,
        embedding=t.embedding,
    )


def run_analysis(
    req: AnalyzeRequest,
    fetch: Fetch,
    upload: Upload,
    detect=None,
    tag=None,
    colors=None,
) -> AnalyzeResult:
    """분석 한 건. 예외를 밖으로 던지지 않고 항상 결과(succeeded/partial_failed/failed)를 돌려준다."""
    model_version = f"{TAGGER_VERSION}+{DETECTOR_VERSION}"

    def failed(code: ErrorCode, message: str) -> AnalyzeResult:
        return AnalyzeResult(
            job_id=req.job_id,
            result_status="failed",
            model_version=model_version,
            errors=[ErrorDetail(code=code, message=message)],
        )

    try:
        if detect is None or tag is None or colors is None:
            # ML 패키지는 실제 실행 때만 불러온다(테스트·기본 설치에서는 가짜를 넘긴다)
            try:
                from app.pipeline.color import extract_colors
                from app.pipeline.detection import detect as _detect
                from app.pipeline.tagging import tag_item
            except ImportError as e:
                raise JobError(ErrorCode.MODEL_UNAVAILABLE, f"ML 패키지 없음: {e}") from e
            detect, tag, colors = detect or _detect, tag or tag_item, colors or extract_colors
        try:
            data = fetch(req.image_url)
        except Exception as e:  # 네트워크·스토리지 오류는 종류가 많다
            raise JobError(ErrorCode.STORAGE_UNAVAILABLE, f"사진 다운로드 실패: {e}") from e
        try:
            image = validate_image(data).image
        except ImageValidationError as e:
            raise JobError(ErrorCode.ANALYSIS_FAILED, f"사진 규격 오류({e.reason})") from e
        try:
            regions = find_regions(image, req, detect)
        except (ImportError, OSError, RuntimeError) as e:
            raise JobError(ErrorCode.MODEL_UNAVAILABLE, f"검출 모델 오류: {e}") from e
    except JobError as e:
        return failed(e.code, str(e))
    except Exception as e:
        log.exception("analysis failed: job_id=%s", req.job_id)
        return failed(ErrorCode.INTERNAL_ERROR, str(e))

    items, errors = [], []
    for region, target in zip(regions, req.items):
        try:
            items.append(_analyze_item(image, region, target, upload, tag, colors))
        except UploadError as e:
            errors.append(_item_error(ErrorCode.STORAGE_UNAVAILABLE, f"누끼 업로드 실패: {e}", target))
        except (ImportError, OSError, RuntimeError) as e:
            # 모델 로드·실행 문제는 재시도하면 될 수 있다
            errors.append(_item_error(ErrorCode.MODEL_UNAVAILABLE, f"태깅 모델 오류: {e}", target))
        except Exception as e:
            log.exception("item failed: job_id=%s key=%s", req.job_id, target.client_item_key)
            errors.append(_item_error(ErrorCode.ANALYSIS_FAILED, str(e), target))

    status = "succeeded" if not errors else "partial_failed" if items else "failed"
    return AnalyzeResult(
        job_id=req.job_id,
        result_status=status,
        model_version=model_version,
        embedding_model=TAGGER_VERSION if items else None,
        items=items,
        errors=errors,
    )


class UploadError(Exception):
    """누끼 업로드 실패. upload 함수가 던진다."""


def _item_error(code: ErrorCode, message: str, target: UploadTarget) -> ErrorDetail:
    return ErrorDetail(code=code, message=message, client_item_key=target.client_item_key)

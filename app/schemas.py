"""백엔드 ↔ AI 서버 API 계약.

변경 시 API_VERSION을 올리고 docs/reports에 Report를 남긴다 (AGENTS.md 참고).
v1: 1차 멘토링 후 결정(261125-2258-01)과 추구미 style_id 제안(261126 Report, issue #114)을 반영한 초안.
예시 JSON은 docs/api-v1-examples.json이며 tests/test_schemas.py가 이 스키마와 맞는지 확인한다.

값 목록(카테고리·패턴·소재·디테일 등)은 문자열로 주고받고 목록은 configs/에 둔다.
백엔드가 enum으로 받을지는 미정이라, 값을 추가해도 계약이 깨지지 않게 문자열로 둔다.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

API_VERSION = "v1"
MAX_ITEMS = 6  # 사진 1장당 옷 수 상한 = 누끼 업로드 URL 개수 상한


class ErrorCode(str, Enum):
    # 재시도하면 될 수 있음
    WORKER_TIMEOUT = "WORKER_TIMEOUT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    CALLBACK_TIMEOUT = "CALLBACK_TIMEOUT"
    # 재시도해도 같은 결과 (이미지 규격 위반 포함)
    NO_GARMENT_DETECTED = "NO_GARMENT_DETECTED"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str
    client_item_key: str | None = Field(
        default=None, description="옷 한 벌의 오류면 그 옷의 키, 사진 전체 오류면 null"
    )


class StyleScore(BaseModel):
    """추구미(스타일) 하나의 점수. 스타일마다 따로 매긴 0~1 값이며 합이 1이 아니다."""

    style_id: int = Field(description="백엔드 style 테이블의 id (메인 스타일만)")
    score: float = Field(ge=0, le=1)


# ── 분석 (비동기, 결과는 콜백) ────────────────────────────────────────────────


class UploadTarget(BaseModel):
    client_item_key: str = Field(description="옷 키. 누끼 업로드 URL과 1:1 (부여 방식은 백엔드 확인 대기)")
    upload_url: str = Field(description="누끼 PNG를 PUT으로 올릴 signed URL")
    path: str | None = Field(
        default=None,
        description="업로드 후 저장 경로. 주면 결과의 cutout_path에 그대로 넣는다. 없으면 upload_url에서 쿼리를 뺀 값",
    )


class AnalyzeRequest(BaseModel):
    job_id: str
    image_url: str = Field(description="원본 이미지 signed URL")
    callback_url: str = Field(description="결과를 POST할 URL (HMAC 서명 헤더 포함)")
    items: list[UploadTarget] = Field(
        min_length=1,
        max_length=MAX_ITEMS,
        description="누끼 업로드 대상. 검출된 옷을 신뢰도 높은 순으로 앞에서부터 배정한다",
    )
    category: str | None = Field(
        default=None,
        description="사용자가 직접 고른 대분류(신발·가방 등 직접 등록). 있으면 검출 없이 사진 전체를 한 벌로 본다",
    )


class AnalyzeAccepted(BaseModel):
    """POST /v1/analyze의 즉시 응답. 결과는 callback_url로 보낸다."""

    job_id: str
    job_status: Literal["queued", "running", "done", "failed"]


class ColorInfo(BaseModel):
    hex: str
    ratio: float = Field(ge=0, le=1)


class CategoryCandidate(BaseModel):
    category: str
    subcategory: str
    score: float = Field(ge=0, le=1)


class AnalyzedItem(BaseModel):
    client_item_key: str
    cutout_path: str = Field(description="누끼 PNG를 올린 경로")
    bbox: list[float] = Field(
        description="회전 보정된 이미지 기준 정규화 좌표 [x_min, y_min, x_max, y_max], 0~1",
        min_length=4,
        max_length=4,
    )
    category: str = Field(description="대분류 (configs/categories.yaml)")
    subcategory: str | None = Field(default=None, description="소분류 1순위")
    category_candidates: list[CategoryCandidate] = Field(
        default=[], description="소분류 상위 3개 후보 (등록 화면에서 사용자가 고름)"
    )
    colors: list[ColorInfo] = []
    pattern: str | None = Field(default=None, description="configs/attributes.yaml patterns")
    style_scores: list[StyleScore] = []
    # 숨은 속성: 화면에 표시하지 않고 추천에만 쓴다. 판정이 불확실하면 null(모름)
    fit: Literal["oversized", "regular", "slim"] | None = None
    off_shoulder: bool | None = Field(default=None, description="오프숄더·원숄더 여부")
    sleeve: Literal["short", "long"] | None = Field(
        default=None, description="반소매·긴소매. 민소매는 소분류 sleeveless"
    )
    material: str | None = Field(default=None, description="configs/attributes.yaml materials")
    details: list[str] = Field(
        default=[],
        description="정확도가 확인된 디테일만 채운다. 빈 목록은 '없음'이 아니라 '확인된 것 없음'",
    )
    embedding: list[float] | None = Field(
        default=None, description="FashionCLIP 이미지 임베딩 (512차원, L2 정규화)"
    )


class AnalyzeResult(BaseModel):
    """callback_url로 보내는 결과."""

    job_id: str
    result_status: Literal["succeeded", "partial_failed", "failed"]
    model_version: str
    embedding_model: str | None = Field(
        default=None, description="임베딩 모델 식별값. 값이 다른 임베딩끼리는 비교하지 않는다"
    )
    items: list[AnalyzedItem] = []
    errors: list[ErrorDetail] = []


# ── 추천 (동기) ──────────────────────────────────────────────────────────────


class WeatherInput(BaseModel):
    temperature: float | None = Field(default=None, description="기온 (°C)")
    feels_like: float | None = Field(default=None, description="체감온도 (°C), 선택")
    precipitation_type: str | None = Field(default=None, description="none / rain / snow")
    humidity: float | None = Field(default=None, ge=0, le=100)
    wind_speed: float | None = Field(default=None, ge=0, description="m/s")
    precipitation_prob: float | None = Field(default=None, ge=0, le=100)


class WardrobeItemInput(BaseModel):
    """백엔드가 저장해 둔 분석 결과 + 사용자 설정을 옷마다 담아 보낸다."""

    item_id: str
    category: str = Field(description="top / bottom / dress / outerwear / shoes / bag / accessory")
    subcategory: str | None = None
    style_scores: list[StyleScore] = []
    fit: Literal["oversized", "regular", "slim"] | None = None
    off_shoulder: bool | None = None
    sleeve: Literal["short", "long"] | None = None
    material: str | None = None
    details: list[str] = []
    warmth: Literal["thin", "normal", "warm"] | None = Field(
        default=None, description="사용자가 등록 화면에서 고른 따뜻함. 없으면 소분류 기본값"
    )
    colors: list[ColorInfo] = []
    pattern: str | None = None
    embedding: list[float] | None = None


class RecommendRequest(BaseModel):
    weather: WeatherInput
    main_style_id: int = Field(description="메인 추구미 style_id")
    sub_style_ids: list[int] = Field(default=[], max_length=2, description="보조 추구미 style_id")
    wardrobe: list[WardrobeItemInput]


class OutfitSetOutput(BaseModel):
    item_ids: list[str]
    score: float


class RecommendResult(BaseModel):
    sets: list[OutfitSetOutput]
    reasons: list[str] = Field(
        default=[], description="추천이 부족하거나 규칙을 완화한 이유 코드 (예: fewer_than_requested)"
    )


class SwapRequest(RecommendRequest):
    outfit_item_ids: list[str] = Field(description="현재 코디 아이템 ID 목록")
    slot: str = Field(description="교체할 슬롯의 대분류 category")


class SwapResult(BaseModel):
    candidates: list[str] = Field(description="교체 후보 item_id 목록 (점수 높은 순)")

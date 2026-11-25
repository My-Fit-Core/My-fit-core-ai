"""백엔드 ↔ AI 서버 API 계약.

변경 시 API_VERSION을 올리고 docs/reports에 Report를 남긴다 (AGENTS.md 참고).
현재 v0: 백엔드와 미합의 초안.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

API_VERSION = "v0"


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
    client_item_key: str | None = None


class AnalyzeRequest(BaseModel):
    job_id: str
    image_url: str = Field(description="원본 이미지 signed URL")
    callback_url: str | None = Field(default=None, description="결과 전달 방식 TBD")
    max_items: int = Field(default=6, ge=1, le=6)


class ColorInfo(BaseModel):
    hex: str
    ratio: float = Field(ge=0, le=1)


class DetectedItem(BaseModel):
    bbox: list[float] = Field(description="정규화 좌표 [x_min, y_min, x_max, y_max], 0~1 (plan §4)", min_length=4, max_length=4)
    mask_path: str | None = Field(default=None, description="누끼 PNG 저장 경로")
    category: str
    category_conf: float = Field(ge=0, le=1)
    colors: list[ColorInfo] = []
    pattern: str | None = None
    aesthetic_scores: dict[str, float] = {}
    embedding: list[float] | None = None


class AnalyzeResult(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "failed"]
    model_version: str
    items: list[DetectedItem] = []
    errors: list[ErrorDetail] = []


# ── 추천 ────────────────────────────────────────────────────────────────────


class WeatherInput(BaseModel):
    temperature: float | None = Field(default=None, description="기온 (°C)")
    precipitation_type: str | None = Field(default=None, description="none / rain / snow")
    humidity: float | None = Field(default=None, ge=0, le=100)
    wind_speed: float | None = Field(default=None, ge=0, description="m/s")
    precipitation_prob: float | None = Field(default=None, ge=0, le=100)


class WardrobeItemInput(BaseModel):
    item_id: str
    category: str = Field(description="top / bottom / dress / outerwear / shoes / bag")
    subcategory: str | None = None
    aesthetic_scores: dict[str, float] = {}
    embedding: list[float] | None = None


class RecommendRequest(BaseModel):
    weather: WeatherInput
    aesthetics: list[str] = Field(default=[], description="메인 추구미 먼저, 보조 추구미 이후")
    schedule: str | None = None
    wardrobe: list[WardrobeItemInput]


class OutfitSetOutput(BaseModel):
    item_ids: list[str]
    score: float


class RecommendResult(BaseModel):
    sets: list[OutfitSetOutput]


class SwapRequest(BaseModel):
    weather: WeatherInput
    aesthetics: list[str] = []
    schedule: str | None = None
    wardrobe: list[WardrobeItemInput]
    outfit_item_ids: list[str] = Field(description="현재 코디 아이템 ID 목록")
    slot: str = Field(description="교체할 슬롯의 대분류 category")


class SwapResult(BaseModel):
    candidates: list[str] = Field(description="교체 후보 item_id 목록 (점수 높은 순)")

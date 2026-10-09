"""백엔드 ↔ AI 서버 API 계약.

변경 시 API_VERSION을 올리고 docs/reports에 Report를 남긴다 (AGENTS.md 참고).
현재 v0: 백엔드와 미합의 초안.
"""

from typing import Literal

from pydantic import BaseModel, Field

API_VERSION = "v0"


class AnalyzeRequest(BaseModel):
    job_id: str
    image_url: str = Field(description="원본 이미지 signed URL")
    callback_url: str | None = Field(default=None, description="결과 전달 방식 TBD")
    max_items: int = Field(default=6, ge=1, le=6)


class ColorInfo(BaseModel):
    hex: str
    ratio: float = Field(ge=0, le=1)


class DetectedItem(BaseModel):
    bbox: list[int] = Field(description="[x1, y1, x2, y2]", min_length=4, max_length=4)
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
    error: str | None = None

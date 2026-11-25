"""AI 서버 엔트리포인트.

분석(/v1/analyze)은 아직 접수 응답만 돌려준다(실제 파이프라인은 #101).
추천은 recommend/scoring.py를 쓴다.
"""

from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException

# ML 패키지(cv2, torch 등)가 없는 환경(기본 requirements.txt)에서도 서버가 뜨도록 lazy import
try:
    from app.pipeline import detection
    from app.pipeline import tagging as _tagging

    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

from app.schemas import (
    API_VERSION,
    AnalyzeAccepted,
    AnalyzeRequest,
    OutfitSetOutput,
    RecommendRequest,
    RecommendResult,
    SwapRequest,
    SwapResult,
)
from recommend.scoring import Item
from recommend.scoring import Weather as ScoringWeather
from recommend.scoring import recommend as _scoring_recommend
from recommend.scoring import swap as _scoring_swap

MODEL_VERSION = "mock-0"

app = FastAPI(title="MyFit AI Server", version=API_VERSION)


def _model_status(loader) -> str:
    """functools.cache 로드 여부. 모델을 강제로 로드하지 않는다."""
    return "loaded" if loader.cache_info().currsize > 0 else "not_loaded"


@app.get("/health")
def health() -> dict:
    models = (
        {
            "detector": _model_status(detection.load_detector),
            "tagger": _model_status(_tagging.load_tagger),
        }
        if _ML_AVAILABLE
        else {"detector": "unavailable", "tagger": "unavailable"}
    )
    return {
        "status": "ok",
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "models": models,
    }


@app.post(f"/{API_VERSION}/analyze", response_model=AnalyzeAccepted)
def analyze(req: AnalyzeRequest) -> AnalyzeAccepted:
    # TODO(#101): 실제 파이프라인 + 콜백으로 교체. 결과 형식은 docs/api-v1-examples.json
    return AnalyzeAccepted(job_id=req.job_id, job_status="queued")


# scoring.py는 내부 추구미 키(예: "feminine")로 동작한다. API의 style_id를 여기서 바꾼다.
_STYLE_KEYS: dict[int, str] = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "configs/style_ids.yaml").read_text(encoding="utf-8")
)["styles"]


def _style_key(style_id: int) -> str:
    return _STYLE_KEYS.get(style_id, str(style_id))


def _to_scoring_items(wardrobe: list) -> list[Item]:
    return [
        Item(
            w.item_id,
            w.category,
            w.subcategory or "",
            {_style_key(s.style_id): s.score for s in w.style_scores} or None,
        )
        for w in wardrobe
    ]


def _to_scoring_weather(w) -> ScoringWeather:
    return ScoringWeather(
        temp_c=w.temperature,
        feels_like_c=w.feels_like,
        precip_type=w.precipitation_type,
        humidity=w.humidity,
        wind_ms=w.wind_speed,
        precip_prob=w.precipitation_prob,
    )


@app.post(f"/{API_VERSION}/recommend", response_model=RecommendResult)
def recommend(req: RecommendRequest) -> RecommendResult:
    items = _to_scoring_items(req.wardrobe)
    weather = _to_scoring_weather(req.weather)
    result = _scoring_recommend(items, weather, _style_key(req.main_style_id))
    if not result.outfits:
        raise HTTPException(
            status_code=422,
            detail="옷장에 추천 가능한 조합이 없습니다. 상의+하의 또는 원피스를 등록해 주세요.",
        )
    return RecommendResult(
        sets=[OutfitSetOutput(item_ids=list(o.items.values()), score=o.score) for o in result.outfits],
        reasons=result.reasons,
    )


@app.post(f"/{API_VERSION}/recommend/swap", response_model=SwapResult)
def recommend_swap(req: SwapRequest) -> SwapResult:
    items = _to_scoring_items(req.wardrobe)
    weather = _to_scoring_weather(req.weather)
    result = _scoring_swap(
        items, weather, _style_key(req.main_style_id), req.outfit_item_ids, req.slot
    )
    if not result.candidates:
        raise HTTPException(
            status_code=422,
            detail=f"교체 가능한 '{req.slot}' 아이템이 옷장에 없습니다.",
        )
    return SwapResult(candidates=[c.item_id for c in result.candidates])

"""AI 서버 엔트리포인트.

지금은 백엔드가 계약대로 연동을 시작할 수 있도록 목업 응답을 돌려준다.
실제 파이프라인은 app/pipeline/ 에 구현 후 연결한다.
"""

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
    AnalyzeRequest,
    AnalyzeResult,
    ColorInfo,
    DetectedItem,
    OutfitSetOutput,
    RecommendRequest,
    RecommendResult,
    SwapRequest,
    SwapResult,
)
from recommend.engine import (
    WardrobeItem,
    build_outfit_sets,
    check_minimum_condition,
    swap_slot,
)

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


@app.post(f"/{API_VERSION}/analyze", response_model=AnalyzeResult)
def analyze(req: AnalyzeRequest) -> AnalyzeResult:
    # TODO(vision-pipeline-poc): 실제 파이프라인으로 교체
    return AnalyzeResult(
        job_id=req.job_id,
        status="done",
        model_version=MODEL_VERSION,
        items=[
            DetectedItem(
                bbox=[0.08, 0.12, 0.84, 0.88],
                category="top.knit",
                category_conf=0.93,
                colors=[ColorInfo(hex="#2B2B2B", ratio=0.71)],
                pattern="solid",
                aesthetic_scores={
                    "feminine": 0.08,
                    "street": 0.12,
                    "minimal": 0.46,
                    "vintage_layered": 0.12,
                    "casual": 0.22,
                },
            )
        ],
    )


def _to_engine_items(wardrobe: list) -> list[WardrobeItem]:
    return [
        WardrobeItem(
            item_id=w.item_id,
            category=w.category,
            aesthetic_scores=w.aesthetic_scores,
            embedding=w.embedding,
        )
        for w in wardrobe
    ]


@app.post(f"/{API_VERSION}/recommend", response_model=RecommendResult)
def recommend(req: RecommendRequest) -> RecommendResult:
    items = _to_engine_items(req.wardrobe)
    if not check_minimum_condition(items):
        raise HTTPException(
            status_code=422,
            detail="옷장에 추천 가능한 조합이 없습니다. 상의+하의 또는 원피스를 등록해 주세요.",
        )
    sets = build_outfit_sets(items, temperature=req.weather.temperature)
    return RecommendResult(
        sets=[OutfitSetOutput(item_ids=s.item_ids, score=s.score) for s in sets]
    )


@app.post(f"/{API_VERSION}/recommend/swap", response_model=SwapResult)
def recommend_swap(req: SwapRequest) -> SwapResult:
    items = _to_engine_items(req.wardrobe)
    candidates = swap_slot(req.outfit_item_ids, req.slot, items)
    if not candidates:
        raise HTTPException(
            status_code=422,
            detail=f"교체 가능한 '{req.slot}' 아이템이 옷장에 없습니다.",
        )
    return SwapResult(candidates=candidates)

"""AI 서버 엔트리포인트.

지금은 백엔드가 계약대로 연동을 시작할 수 있도록 목업 응답을 돌려준다.
실제 파이프라인은 app/pipeline/ 에 구현 후 연결한다.
"""

from fastapi import FastAPI

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
                bbox=[40, 60, 420, 520],
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

"""AI 서버 엔트리포인트.

분석(/v1/analyze)은 접수 응답을 바로 주고, 분석은 뒤에서 돌려 결과를 callback_url로 보낸다(#101).
추천은 recommend/scoring.py를 쓴다.

인증: 백엔드 → AI 서버 요청은 공유 비밀키 헤더(X-AI-Server-Secret, 이름은 제안)로 확인한다.
환경변수 AI_SERVER_SECRET이 비어 있으면 확인하지 않는다(개발용).
"""

import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import yaml
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException

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

@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not os.getenv("AI_SERVER_SECRET"):
        # 비어 있으면 누구나 image_url·callback_url을 정해 이 서버가 그 주소로 요청하게 할 수 있다(SSRF)
        log.warning("AI_SERVER_SECRET이 비어 있어 인증 없이 열려 있다. 배포 환경에서는 반드시 설정할 것")
    # 모델은 서버 시작 때 한 번 불러온다(02-specs §8). 첫 요청이 모델 로드 시간(CPU에서 1~2분)을 기다리지 않게.
    # TestClient(app)를 with 없이 쓰면 이 단계를 건너뛴다.
    if _ML_AVAILABLE and os.getenv("PRELOAD_MODELS", "1") == "1":
        try:
            detection.load_detector()
            _tagging.load_tagger()
        except Exception:
            log.exception("모델 미리 불러오기 실패: 첫 분석 요청에서 다시 시도한다")
    yield


app = FastAPI(title="MyFit AI Server", version=API_VERSION, lifespan=lifespan)
log = logging.getLogger(__name__)
SECRET_HEADER = "X-AI-Server-Secret"
DOWNLOAD_TIMEOUT_S = 15.0


def require_secret(secret: str | None = Header(default=None, alias=SECRET_HEADER)) -> None:
    expected = os.getenv("AI_SERVER_SECRET", "")
    if expected and not hmac.compare_digest((secret or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="인증 실패")


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


def _run_job(req: AnalyzeRequest) -> None:
    """뒤에서 분석하고 결과를 콜백으로 보낸다. 예외를 밖으로 던지지 않는다."""
    from app.callback import send_result
    from app.pipeline.analyze import UploadError, run_analysis

    with httpx.Client(timeout=DOWNLOAD_TIMEOUT_S) as client:

        def fetch(url: str) -> bytes:
            res = client.get(url)
            res.raise_for_status()
            return res.content

        def upload(url: str, png: bytes) -> None:
            try:
                client.put(url, content=png, headers={"Content-Type": "image/png"}).raise_for_status()
            except httpx.HTTPError as e:
                raise UploadError(str(e)) from e

        result = run_analysis(req, fetch, upload)
        send_result(req.callback_url, result, client)


@app.post(
    f"/{API_VERSION}/analyze",
    response_model=AnalyzeAccepted,
    dependencies=[Depends(require_secret)],
)
def analyze(req: AnalyzeRequest, background: BackgroundTasks) -> AnalyzeAccepted:
    # ponytail: 작업 큐 없이 프로세스 안에서 돈다. 서버가 재시작되면 진행 중이던 작업은 사라진다
    background.add_task(_run_job, req)
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
            sleeve=w.sleeve,
            warmth=w.warmth,
            material=w.material,
            off_shoulder=w.off_shoulder,
            fit=w.fit,
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


@app.post(
    f"/{API_VERSION}/recommend",
    response_model=RecommendResult,
    dependencies=[Depends(require_secret)],
)
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


@app.post(
    f"/{API_VERSION}/recommend/swap",
    response_model=SwapResult,
    dependencies=[Depends(require_secret)],
)
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

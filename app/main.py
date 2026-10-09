"""AI 서버 엔트리포인트.

지금은 백엔드가 계약대로 연동을 시작할 수 있도록 목업 응답을 돌려준다.
실제 파이프라인은 app/pipeline/ 에 구현 후 연결한다.
"""

from fastapi import FastAPI

from app.schemas import API_VERSION, AnalyzeRequest, AnalyzeResult, ColorInfo, DetectedItem

MODEL_VERSION = "mock-0"

app = FastAPI(title="MyFit AI Server", version=API_VERSION)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "api_version": API_VERSION, "model_version": MODEL_VERSION}


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
                aesthetic_scores={"dark_academia": 0.82, "clean_girl": 0.31},
            )
        ],
    )

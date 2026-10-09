# 02. 기술 제약과 구현 규칙

> `TBD` 표시는 아직 팀 합의 전인 항목입니다. 에이전트는 임의로 정하지 말고 TODO의 "열린 질문"에 남깁니다.

## 1. 시스템 구성

```
[앱] ──업로드──▶ [Supabase Storage]
  │
  └──▶ [백엔드 (Supabase + Vercel)] ──비동기 요청(job)──▶ [AI 서버 (이 레포)]
                     ▲                                       │
                     └──────────── 결과 콜백 또는 polling ──────┘
```

- AI 서버는 GPU가 필요한 추론만 담당한다. 인증(JWT), 날씨 수집·캐시, DB 저장은 백엔드 책임.
- 서빙 플랫폼: `TBD` (후보: Modal, RunPod Serverless, 직접 GPU 서버). FastAPI + Docker 이미지 하나로 어디든 옮길 수 있게 유지한다.
- 결과 전달 방식(콜백 vs polling): `TBD`

## 2. 기술 스택

| 영역 | 선택 | 비고 |
| --- | --- | --- |
| 언어 | Python 3.11 | PyTorch 호환성 기준 |
| 서빙 | FastAPI + Uvicorn | |
| 검출 | Grounding DINO / YOLO(DeepFashion2) | PoC 비교 후 확정 `TBD` |
| 분할·누끼 | SAM 2 또는 MobileSAM, 필요 시 BiRefNet 후처리 | PoC 후 확정 `TBD` |
| 태깅·임베딩 | FashionCLIP (zero-shot) | 색상은 마스크 픽셀 k-means |
| 추천 | 규칙 기반 점수 함수 (`recommend/`) | GPU 불필요 |
| LLM 코멘트 (P1) | 경량 LLM API | 모델 `TBD` |
| 벡터 저장 | Supabase pgvector (백엔드 DB) | 임베딩 차원은 모델 확정 후 기록 |

## 3. API 계약 (백엔드 ↔ AI 서버)

현재 버전: **v0 (초안, 백엔드와 미합의)**. 실제 정의는 `app/schemas.py`가 기준이며, 이 섹션과 다르면 어긋남을 보고한다.

### POST `/v0/analyze` — 다중 의류 분석 요청

요청
```json
{
  "job_id": "uuid",
  "image_url": "원본 이미지 signed URL",
  "callback_url": "선택, TBD",
  "max_items": 6
}
```

응답(결과)
```json
{
  "job_id": "uuid",
  "status": "done | failed",
  "model_version": "v0.1",
  "items": [
    {
      "bbox": [0, 0, 100, 100],
      "mask_path": "누끼 PNG 저장 경로",
      "category": "top.knit",
      "category_conf": 0.93,
      "colors": [{"hex": "#2B2B2B", "ratio": 0.71}],
      "pattern": "solid",
      "aesthetic_scores": {"dark_academia": 0.82},
      "embedding": [0.0]
    }
  ],
  "error": null
}
```

### 추천 로직 호출 방식 — `TBD`
추천(F-302)은 백엔드 7시 배치에서 실행된다. AI 서버 API로 노출할지, 백엔드에 로직을 포함할지 회의에서 확정.

## 4. 데이터 기준 (configs/aesthetics.yaml)

- 15대 추구미 목록: `TBD` (현재 확정: Clean Girl, Dark Academia, Balletcore, Cottagecore)
- 카테고리 체계: 상의/하의/아우터/신발/잡화 + 세부 분류 `TBD`. 백엔드 ERD enum과 반드시 동일하게 유지.
- 주 타겟이 1020대 여성이므로 원피스, 스커트 세부, 가방·액세서리를 세분화한다.

## 5. 성능 기준

| 항목 | 기준 | 상태 |
| --- | --- | --- |
| 다중 의류 처리 시간 | 6벌 기준 2초(승인 조건) / 1.5초(유저플로우·AI 트랙) 혼재 | `TBD` — 하나로 확정, 콜드 스타트 포함 여부 명시 |
| 카테고리 분류 정확도 | 90% 이상 | 테스트셋·정답 체계 확정 후 측정 |
| 매칭률(%) 산정 | 코사인 유사도 → 0~100% 보정 방식 | `TBD` (P1) |

## 6. 추천 입력값

백엔드 날씨 snapshot 기준으로 사용한다. 현재 백엔드가 검증하는 값: 온도, 습도, 강수확률, 강수형태.
기획안에 있는 체감온도·미세먼지 제공 여부: `TBD`

추천 엔진이 지원해야 할 것
- 세트 3종 생성 (서로 다양성 확보)
- 특정 슬롯(예: 신발) 하나의 대체 후보 랭킹 — F-304가 P0
- TPO(출근, 데이트, 캐주얼 외출 등) 반영

## 7. 테스트 데이터

- 팀원이 직접 촬영한 Flat Lay 사진 20~30장 (2~6벌, 바닥/침대)
- 저장 위치: `TBD` (팀 공유 스토리지). 레포에는 커밋하지 않는다.

## 8. 코드 규칙

- 포맷터/린터: ruff
- 타입 힌트 사용, 요청/응답은 pydantic 모델로만 주고받는다.
- 모델은 서버 시작 시 1회 로드, 요청마다 로드하지 않는다.
- 비밀값은 환경변수로만 읽는다.

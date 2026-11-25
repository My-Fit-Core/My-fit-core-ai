# 02. 기술 제약과 구현 규칙

> `TBD` 표시는 아직 팀 합의 전인 항목입니다. 에이전트는 임의로 정하지 말고 TODO의 "열린 질문"에 남깁니다.
> `[확인]` 표시는 회의 답변을 해석한 것이라 사람 확인이 필요한 항목입니다. 안건 번호는 10/12 회의 노션 페이지 기준입니다.

## 1. 시스템 구성

```
[앱] ──업로드──▶ [Supabase Storage]
  │
  └──▶ [백엔드 (Supabase + Vercel)] ──분석 요청(비동기, job)──▶ [AI 서버 (이 레포)]
                     ▲          └──────추천 요청(동기)──────▶       │
                     └──────────────── 분석 결과 콜백 ───────────────┘
```

- AI 서버는 GPU가 필요한 추론과 추천 계산을 담당한다. 인증(JWT), 날씨 수집·캐시, DB 저장은 백엔드 책임.
- 서빙 플랫폼: `TBD` (후보: Modal, RunPod Serverless, 직접 GPU 서버). FastAPI + Docker 이미지 하나로 어디든 옮길 수 있게 유지한다.
- 결과 전달 방식: 분석은 비동기로 처리하고 결과를 백엔드에 **콜백**으로 보낸다 `[확인]`(10/12 회의 안건 4). 추천은 동기 응답(안건 2, 4).
- 콜백: 분석 요청마다 `callback_url`을 받는다. AI 서버 → 백엔드 콜백은 HMAC 서명을 붙이고, 키는 환경변수 `AI_CALLBACK_HMAC_SECRET`이다(11/25 백엔드 답변). 서명 대상·헤더 이름·시간 허용 범위는 `TBD`(제안: `X-AI-Timestamp` + `X-AI-Signature: sha256=<hex>`, 서명 대상 `timestamp + "." + 본문`, 5분).
- 백엔드 → AI 서버 인증: 공유 비밀키 헤더(11/25 백엔드 동의). 헤더 이름은 `TBD`(제안: `X-AI-Server-Secret`).

## 2. 기술 스택

| 영역 | 선택 | 비고 |
| --- | --- | --- |
| 언어 | Python 3.11 | PyTorch 호환성 기준 |
| 서빙 | FastAPI + Uvicorn | |
| 검출 | YOLO(DeepFashion2, `deepfashion2_yolov8s-seg.pt`) | 기본으로 결정. 부위 판정도 YOLO 클래스만 쓴다(`261108-0315-01-category-poc.md`). Grounding DINO는 보류. 근거는 `261108-0305-01-detection-segmentation-poc.md` |
| 분할·누끼 | YOLO 자체 마스크 + 구멍 메우기 | 기본으로 결정. SAM 2.1 tiny 추가 여부는 Flat Lay 사진에서 경계 품질 비교 후 결정 `TBD` |
| 태깅·임베딩 | FashionCLIP (zero-shot) | 입력은 마스크 밖을 흰색으로 지우고 박스로 자른 사진(`261108-0315-01-category-poc.md`). 색상은 마스크 픽셀 k-means |
| 추천 | 규칙 기반 점수 함수 (`recommend/`) | GPU 불필요. AI 서버에서 실행, 동기 응답(10/12 회의 안건 2, 4) |
| LLM 코멘트 (P1) | 경량 LLM API | 모델 `TBD` |
| 벡터 저장 | Supabase pgvector (백엔드 DB) | 분석 결과에 옷별 임베딩과 `embedding_model`을 담고, 백엔드가 등록한 옷과 연결해 저장한다(11/25 백엔드 답변). FashionCLIP ViT-B/32, 512차원, L2 정규화, 코사인 유사도(= 내적). 백엔드 확인 대기 |

- 패키지 버전: `requirements.txt`는 현재 버전으로 고정한다(10/12 회의 안건 10).

## 3. API 계약 (백엔드 ↔ AI 서버)

현재 버전: **v1 (초안, issue #114)**. 실제 정의는 `app/schemas.py`가 기준이고, 예시는 `docs/api-v1-examples.json`이다(테스트가 둘이 맞는지 확인). 이 섹션과 다르면 어긋남을 보고한다.

v1 요약
- `POST /v1/analyze`: 요청 `{job_id, image_url, callback_url, items[{client_item_key, upload_url, path?}], category?}` → 즉시 `{job_id, job_status}`. 결과(`AnalyzeResult`)는 `callback_url`로 보낸다.
  - `items`는 누끼 업로드 대상 1~6개. 검출된 옷을 신뢰도 높은 순으로 앞에서부터 배정한다.
  - `category`가 있으면(신발·가방 직접 등록) 검출 없이 사진 전체를 한 벌로 본다.
- 결과 옷마다: `client_item_key`, `cutout_path`, `bbox`(0~1), `category`, `subcategory`, `category_candidates`(상위 3), `colors`, `pattern`, `style_scores[{style_id, score}]`, 숨은 속성(`fit`, `off_shoulder`, `sleeve`, `material`, `details`), `embedding`.
  - `style_scores`는 스타일마다 따로 매긴 0~1 점수이며 합이 1이 아니다. `style_id`는 백엔드 style 테이블의 메인 스타일 id다.
  - 숨은 속성은 화면에 표시하지 않고 추천에만 쓴다. 불확실하면 `null`, `details`의 빈 목록은 "확인된 것 없음"이다. 값 목록은 `configs/attributes.yaml`.
- 결과 상태 `result_status`: `succeeded` / `partial_failed` / `failed`(아래 v1 합의 사항).
- `POST /v1/recommend`, `/v1/recommend/swap`: `{weather, main_style_id, sub_style_ids, wardrobe[...]}`. 옷장 아이템에는 분석 결과(소분류·스타일 점수·숨은 속성)와 사용자가 고른 `warmth`(선택)를 담는다.
- 값 목록(카테고리·패턴·소재·디테일)은 문자열로 주고받는다. enum으로 받을지는 백엔드 확인 대기.

### v1 합의 사항 (10/12 회의, 스키마 반영은 T6)

분석 요청·결과
- 카테고리는 대분류 `category`와 소분류 `subcategory` 두 필드로 나눈다. 점 표기(`top.knit`)는 쓰지 않는다(안건 7). 값 목록은 §4.
- 누끼 업로드: 백엔드가 원본 사진 URL과 누끼 업로드 URL을 발급해 요청에 담는다. AI 서버는 받은 URL로 누끼 PNG를 올리고, 결과에는 올린 파일 경로를 넣는다. 누끼 업로드 URL 개수가 사진 1장의 옷 수 상한이다(안건 8). 필드 이름은 v1 스키마(`items[].upload_url`, 결과 `cutout_path`)로 정했다(백엔드 확인 대기).
- 옷마다 `client_item_key`를 둔다. v1은 요청 `items[]`에 업로드 URL과 묶어 받는 것으로 가정했다. 부여 방식은 백엔드 확인 대기(T3).
- 결과 상태(안건 7)
  - `succeeded`: items ≥1, errors 0
  - `partial_failed`: items ≥1, errors ≥1. 실패한 옷은 errors에 `client_item_key`와 함께 넣는다
  - `failed`: items 0, errors ≥1
- 작업 상태: AI 서버는 대기·진행중·완료·실패 4개를 쓴다. 백엔드는 대기·진행·취소를 따로 관리한다(안건 7).
- 결과는 `items`와 `errors`를 함께 담는다. 에러 형식 `[확인]`(안건 8):
  ```json
  {"code": "NO_GARMENT_DETECTED", "message": "...", "client_item_key": null}
  ```
  사진 전체 오류는 `client_item_key`가 `null`이다. 재시도 여부는 백엔드가 판단한다.
- 에러 코드(백엔드 임시안) `[확인]`(안건 8)

  | 코드 | 재시도 | 쓰는 상황 |
  | --- | --- | --- |
  | `WORKER_TIMEOUT` | 가능 | 처리 시간 초과 |
  | `MODEL_UNAVAILABLE` | 가능 | 모델 로드 실패 |
  | `STORAGE_UNAVAILABLE` | 가능 | 사진 다운로드·누끼 업로드 실패 |
  | `INTERNAL_ERROR` | 가능 | 예상 못한 오류 |
  | `CALLBACK_TIMEOUT` | 가능 | 콜백 전달 실패 |
  | `NO_GARMENT_DETECTED` | 무의미 | 검출 0벌 |
  | `ANALYSIS_FAILED` | 무의미 | 분석 실패. 이미지 규격 위반도 우선 여기로 보낸다 `[확인]`. 전용 코드 필요 여부는 `TBD` |

- 임베딩은 분석 결과에 옷마다 담는다(11/25 백엔드 답변, §2).

### 입력 이미지 규칙

- 형식: JPEG/PNG/WebP. HEIC는 프론트에서 JPEG로 변환한다(10/12 회의 안건 9).
- 크기: 최대 20MB(안건 9). 긴 변 최대 2048px은 현재 구현값이며 유지 여부는 `TBD`(plan §8). 짧은 변이 512px 미만이면 안내한다.
- 형식·용량 값은 `configs/image_input.yaml`에서 읽는다(안건 9). 검증 코드는 `app/pipeline/image_input.py`.
- 회전 보정과 EXIF 제거는 프론트에서 업로드 전에 하고, AI 서버도 한 번 더 확인한다(안건 9).

### 좌표 규칙

- `bbox`는 회전 보정된 이미지 기준 정규화 좌표(0~1)이며 `[x_min, y_min, x_max, y_max]` 순서다. 왼쪽 위가 (0, 0)이다(안건 9).
- v1 스키마(`app/schemas.py`)는 이 규칙대로 `bbox`를 0~1 float로 둔다.

### 추천 API (10/12 회의 안건 2, 3, 4)

추천(F-302)은 백엔드 7시 배치에서 AI 서버 API를 호출해 실행한다.
- `POST /v1/recommend`: 코디 세트 3종. `POST /v1/recommend/swap`: 슬롯 1개 대체 후보 랭킹(F-304).
- 동기 응답이며 `job_id`가 없다.
- 백엔드가 날씨 + 추구미 + 옷장 목록을 한 요청에 담아 보낸다. AI 서버는 날씨를 조회하지 않고 DB에도 접근하지 않는다.
- 요청·응답 필드는 v1 스키마(§3 v1 요약)를 따른다. 옷장 아이템의 임베딩은 선택 필드다.

## 4. 데이터 기준 (configs/aesthetics.yaml, configs/categories.yaml)

- 사용자에게 보이는 추구미: 7종 확대 제안 중(미니멀·캐주얼·러블리/페미닌·걸리시 스트릿·Y2K/빈티지·오피스/클래식·보헤미안, `configs/style_ids.yaml`). 합의 전 현재 구현은 기본 스타일 5개 (페미닌, 스트리트, 미니멀, 빈티지 레이어드, 캐주얼). 2026-11-09 기획 합의로 Y2K를 스트리트로 바꿨다(issue #44).
  - 캐주얼은 테마 없는 기본템(무지, 기본핏), 스트리트는 오버핏·그래픽·로고·카고·캡이다.
  - 우리 스트리트 ≠ K-Fashion 스트리트. K-Fashion 스트리트는 일상복에 가까워 캐주얼에 연결하고, 우리 스트리트는 K-Fashion 힙합·스포티에 연결한다(잠정, `261109-1840-01-base-styles.md`).
  - Y2K 같은 마이크로 트렌드는 2차 로드맵으로 미룬다. K-Fashion에 대응 스타일이 없어 이미지를 직접 모아야 한다.
- 추구미 점수
  - 코디 사진: FashionCLIP 프롬프트 세트 `outfit`의 점수를 `calibration.prior`(고정값)로 나눠 다시 합 1로 보정한다. 근거는 `docs/reports/261108-0230-01-aesthetic-display.md`.
  - v1부터 API 점수는 `style_scores`(스타일마다 따로 0~1, 합 1 아님)로 바뀐다(11/25 결정, 계산 방식은 태깅 수정에서). 아래는 v0 방식 기록이다.
  - 옷 한 벌(v0 `aesthetic_scores`): 프롬프트 세트 `item`을 쓰고 보정하지 않는다. `calibration_item.prior`는 기록만 한다. 근거는 `docs/reports/261110-0117-01-item-prompts.md`.
  - 옷 종류 쏠림(예: 청바지 → 캐주얼, 후드티 → 스트리트)은 MVP에서 그대로 둔다.
- 카테고리 체계: 대분류 `category`와 소분류 `subcategory` 두 필드(10/12 회의 안건 7). 백엔드 ERD enum과 반드시 동일하게 유지. 목록은 `configs/categories.yaml`에 둔다(현재 `category-taxonomy.md` 안 A 초안, 확정 아님).
  - 대분류 6종(백엔드 제안, 안건 7): `outerwear`, `top`, `bottom`, `shoes`, `bag`, `accessory`
  - 원피스(`dress`)를 대분류에 추가해 7종(11/25 안건 2-1 결정).
  - 소분류 목록: 안건 초안 그대로 확정(11/25 안건 2-2), `configs/categories.yaml`. 이후 #115에서 이너 소분류가 추가돼 백엔드·프론트 공유 필요.
  - MVP 자동 검출은 의류(`top`, `bottom`, `outerwear`, `dress`)만(11/25 안건 2-4 A). 신발·가방·액세서리는 사용자가 카테고리를 골라 등록한다.
- 주 타겟이 1020대 여성이므로 원피스, 스커트 세부, 가방·액세서리를 세분화한다.

## 5. 성능 기준

| 항목 | 기준 | 상태 |
| --- | --- | --- |
| AI 추론 시간 | 사진 1장(옷 최대 6벌) 기준 2초 목표, 1.5초는 최적화 목표 | 확정(10/12 회의 안건 5). 서버 기동·이미지 다운로드/업로드 시간은 따로 측정 |
| 사용자 체감 시간 (업로드→결과 확인) | 추론 시간과 별도 지표(안건 5) | 목표치는 테스트 후 결정 `TBD` |
| 카테고리 분류 정확도 | 90% 이상 | 테스트셋·정답 체계 확정 후 측정 |
| 매칭률(%) 산정 | 코사인 유사도 → 0~100% 보정 방식 | `TBD` (P1) |

## 6. 추천 입력값

백엔드 날씨 snapshot 기준으로 사용한다(10/12 회의 안건 3).
- 받는 값: 기온, 강수 형태, 습도, 풍속, 강수확률
- 체감온도·미세먼지: 백엔드가 추가할 예정인 선택 값. 없으면 있는 값만으로 계산하고 결과에 표시한다.

추천 엔진이 지원해야 할 것
- 세트 3종 생성 (서로 다양성 확보)
- 특정 슬롯(예: 신발) 하나의 대체 후보 랭킹 — F-304가 P0
- TPO(출근, 데이트 등)는 1차 MVP에서 제외하고 2차에서 다룬다(안건 6).

## 7. 테스트 데이터

- 팀원이 직접 촬영한 Flat Lay 사진 20~30장 (2~6벌, 바닥/침대)
- 저장 위치: `TBD` (팀 공유 스토리지). 레포에는 커밋하지 않는다.

## 8. 코드 규칙

- 포맷터/린터: ruff
- 타입 힌트 사용, 요청/응답은 pydantic 모델로만 주고받는다.
- 모델은 서버 시작 시 1회 로드, 요청마다 로드하지 않는다.
- 비밀값은 환경변수로만 읽는다.

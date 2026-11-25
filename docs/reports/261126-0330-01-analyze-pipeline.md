# 분석 파이프라인 — 검출·누끼 업로드·태깅·콜백 연결

- 일시: 2026-11-26 03:30
- 작성: AI 파트 (gaeun1961, Claude 작성)
- 관련 TODO: `docs/todo/analyze-endpoint.md`
- 관련 기능 ID: F-201, F-202, F-203
- 관련 이슈: #101 (선행: 스키마 v1 PR #116, 태깅 v1 PR #118)

## 요약

`POST /v1/analyze`를 실제로 동작하게 했다. 요청을 받으면 바로 접수 응답을 주고, 뒤에서 사진 다운로드 → 검증 → YOLO 검출·마스크 → 누끼 PNG 업로드 → 태깅·색 추출을 거쳐 결과를 `callback_url`로 HMAC 서명과 함께 보낸다. 백엔드 → AI 서버 요청은 공유 비밀키 헤더로 확인한다. 서명·헤더 세부 규칙은 백엔드 확인 전 제안값이다.

## 변경 내역

- `app/pipeline/analyze.py`(새 파일): `run_analysis(req, fetch, upload, ...)` — 예외를 던지지 않고 항상 `succeeded`/`partial_failed`/`failed` 결과를 돌려준다
  - 검출된 옷을 신뢰도 순으로 업로드 대상(`items`) 개수까지만 쓴다
  - 대분류는 검출기 클래스(DeepFashion2)로 정하고, 태깅은 그 대분류 안에서 소분류를 고른다. 소매도 클래스 이름(short/long sleeved)에 있으면 그 값을 쓴다
  - 직접 등록(`category`): 같은 대분류가 검출되면 그 마스크, 아니면(신발·가방) 사진 전체를 한 벌로 본다
  - 누끼: 마스크 밖 투명 RGBA PNG, bbox로 자름. 결과 `cutout_path`는 요청의 `path`, 없으면 업로드 URL에서 쿼리를 뺀 값
- `app/callback.py`(새 파일): HMAC-SHA256 서명(`X-AI-Timestamp`, `X-AI-Signature: sha256=<hex>`, 서명 대상 `timestamp.본문`), 실패하면 1·2초 뒤 2번 더 보내고 로그만 남김. 백엔드 검증 규칙과 같은 `verify()`도 둠
- `app/main.py`: 서버 시작 때 모델 미리 불러오기, `/v1/analyze`가 `BackgroundTasks`로 분석 실행, `X-AI-Server-Secret` 헤더 확인(분석·추천·교체 모두, `AI_SERVER_SECRET`이 비면 확인 안 함)
- `app/pipeline/image_input.py`: 긴 변 2048px 초과를 **거절하지 않고 줄인다**(안건 6: 프론트가 줄이고 AI 서버 리사이즈는 대비용으로 유지). 8192px 초과만 디코딩 전 거절
- `configs/categories.yaml`: 검출기 클래스 → 대분류·소매 매핑(`detector_classes`)
- `.env.example`: `AI_CALLBACK_HMAC_SECRET` 추가, 쓰지 않는 Supabase 키 제거
- 테스트: `tests/test_analyze.py`(가짜 검출기·태거·업로드로 성공·부분 실패·검출 0·다운로드 실패·규격 오류·직접 등록·콜백 서명과 재시도), `tests/test_health.py`(접수 응답, 비밀키), `tests/test_image_input.py`(줄이기)

## 오류 코드 매핑

| 상황 | 코드 | 범위 |
| --- | --- | --- |
| 사진 다운로드 실패 | `STORAGE_UNAVAILABLE` | 사진 전체 |
| 사진 규격 위반(형식·용량·손상) | `ANALYSIS_FAILED` | 사진 전체 |
| ML 패키지 없음, 검출기 로드·실행 오류 | `MODEL_UNAVAILABLE` | 사진 전체 |
| 옷 0벌 | `NO_GARMENT_DETECTED` | 사진 전체 |
| 그 밖의 예외 | `INTERNAL_ERROR` | 사진 전체 |
| 누끼 업로드 실패 | `STORAGE_UNAVAILABLE` | 옷 한 벌(`client_item_key`) |
| 태깅 모델 로드·실행 오류 | `MODEL_UNAVAILABLE` | 옷 한 벌 |
| 그 밖의 옷 한 벌 처리 오류 | `ANALYSIS_FAILED` | 옷 한 벌 |
| 콜백 전달 실패 | (보낼 곳이 없어 로그만) | — |

## 결과·측정치

- pytest: `tests/test_analyze.py`·`test_health.py`·`test_image_input.py`·`test_schemas.py`·`test_recommend.py` 75 passed, ruff 통과. 이 테스트들은 가짜 검출기·태거를 쓰는 단위 테스트이고, **실제 모델로 파이프라인 전체를 도는 자동 테스트는 아직 없다**(아래 수동 실행만)
- 실제 모델 실행(K-Fashion 사진 3장, CPU, 업로드는 메모리): 3장 모두 `succeeded`, 누끼 5개 업로드(2벌·2벌·1벌)

| 사진 | 시간 | 결과(대분류, 소분류, 소매, 핏, 오프숄더, 소재, 1위 style_id) |
| --- | --- | --- |
| 스트리트 | 124.5초 | 상의 blouse short oversized False chiffon_satin 2 / 하의 jeans – regular – denim 2 |
| 로맨틱 | 4.3초 | 상의 blouse long oversized False chiffon_satin 3 / 하의 jeans – regular – denim 2 |
| 클래식 | 7.7초 | 원피스 mini_dress long oversized True wool_tweed 5 |

  - 첫 사진 124.5초는 검출·태깅 모델을 처음 불러온 시간이 들어 있다. 그래서 서버 시작 때 모델을 미리 불러오게 했다(`lifespan`, `PRELOAD_MODELS=1` 기본).
  - 두 번째부터 4~8초는 CPU, 그리고 다른 측정(7종 평가)이 같은 CPU를 쓰는 중이라 느리다. **2초 목표 판단에 쓸 숫자가 아니다.**
  - 값은 정답과 비교하지 않았다(미검증). 클래식 원피스가 오프숄더 `True`, 1위 스타일이 Y2K/빈티지(5)로 나온 것은 맞는지 모른다

## 주요 결정

| 결정 | 근거 | 대안과 버린 이유 |
| --- | --- | --- |
| 대분류는 검출기 클래스로 | 카테고리 PoC(261108-0315)에서 YOLO 클래스로 부위를 정하기로 함 | 태깅 문장만으로: 상의·아우터 혼동 |
| 프로세스 안 `BackgroundTasks` | 가장 단순, 배포 방식(#102) 미정 | 작업 큐(Redis 등): 배포 방식이 정해진 뒤 |
| 2048px 초과는 줄이기 | 안건 6 결정(AI 리사이즈 유지) | 거절: 기존 구현이 결정과 달랐음 |
| 콜백 재시도 3번 후 포기 | MVP. 실패 사실은 백엔드가 작업 시간 초과로 알 수 있음 | 영속 재시도: 큐가 필요 |

## 알려진 한계

- **신발·가방 누끼는 배경을 지우지 못한다.** 검출기(DeepFashion2)에 없는 종류라 사진 전체를 올린다. 누끼 모델(#41, SAM 등)을 붙일 때 바꾼다 `[결정: 사용자]`
- 서버가 재시작되면 진행 중이던 분석은 사라진다(콜백이 안 감)
- **`AI_SERVER_SECRET`이 비어 있으면 인증 없이 열린다.** 이때는 누구나 `image_url`·`callback_url`을 정해 이 서버가 그 주소로 요청하게 할 수 있다(SSRF). 시작할 때 경고 로그를 남기며, 배포 환경에서는 반드시 설정한다. 비어 있으면 서버가 뜨지 않게 할지는 배포(#102) 때 정한다 `[결정: 사용자]`
- 6벌 2초 목표는 GPU에서 다시 재야 한다(지금은 CPU)

## 백엔드 확인이 필요한 것

- 헤더 이름(`X-AI-Server-Secret`), 콜백 서명 규칙(서명 대상·헤더·5분)
- 업로드는 `PUT`, `Content-Type: image/png`로 보낸다. signed URL 방식이 맞는지
- 검출 수가 업로드 URL보다 적으면 남는 URL은 쓰지 않는다

## 후속 작업

- T1 Flat Lay 사진으로 검출·누끼 품질과 시간 측정(#99)
- 신발·가방 배경 제거 방법

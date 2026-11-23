# YOLO 검출+마스크 모듈 (T8)

- 일시: 2026-11-23 15:44
- 작성: AI 파트 (Claude Code)
- 관련 TODO: `docs/todo/yolo-detection.md`
- 관련 기능 ID: F-201, F-202
- 관련 issue: #53

## 요약

`app/pipeline/detection.py`에 `detect(image, max_items=6)`를 만들었다. YOLO(DeepFashion2) seg 모델로 신뢰도 순 상위 `max_items`개의 bbox(픽셀 xyxy)·신뢰도·DeepFashion2 클래스·구멍 메운 bool 마스크를 돌려준다. 가중치는 `scripts/download_weights.py`로 받고(sha256 확인), 모델은 프로세스당 한 번만 로드한다.

## 변경 내역

- `app/pipeline/detection.py`: `Detection`, `load_detector`(캐시), `fill_holes`, `to_detections`(순수 함수), `detect`
- `scripts/download_weights.py`: Hugging Face `Bingsu/adetailer`에서 받기, 이미 있으면 건너뜀, sha256 불일치면 삭제
- `tests/test_detection.py`: 단위 테스트 5개(가중치 없이 실행), 통합 테스트 2개(가중치·샘플 사진이 있을 때만)
- `.env.example`: `DETECTOR_WEIGHTS`

## 결과·측정치

- `pytest`(51 passed, 2 skipped: 가중치 없는 기본 환경), `ruff check` 통과.
- 통합 테스트: 로컬 가중치 + K-Fashion Validation 1장(800x532, 스트리트)을 `DETECTOR_WEIGHTS`, `DETECTOR_SAMPLE_IMAGE`로 지정해 7개 모두 통과. 결과 2벌(`trousers` 0.93, `short_sleeved_shirt` 0.91), 흰 단색 이미지는 0벌.
- 시간(참고용, Windows CPU): 워밍 후 YOLO 호출만 0.6~2.1초로 실행마다 크게 흔들렸다. 후처리(구멍 메우기)는 차이가 거의 없었다. PoC의 장당 0.45초와 다른 건 측정 시 CPU 부하 차이로 보이며, "6벌 2초" 기준 측정은 GPU에서 T13으로 한다.
- 다운로드 스크립트: scratchpad 경로로 실제 다운로드·해시 확인·재실행 시 건너뜀까지 확인. sha256은 Hugging Face LFS 값과 로컬 PoC 가중치 해시가 같았다.

## 주요 결정

| 결정 | 근거 | 대안과 버린 이유 |
| --- | --- | --- |
| `main.py` lifespan 연결은 하지 않음 | 가중치·ultralytics 없는 환경(현재 Dockerfile, `/health` 테스트)이 깨지고, 로드 실패 처리는 T12 완료 기준 | 지금 연결: T12 범위 침범 |
| 1회 로드는 `functools.cache` | T12가 lifespan에서 한 번 불러 미리 데우면 된다 | 전역 변수·클래스: 같은 효과에 코드만 늘어남 |
| bbox는 픽셀 xyxy | T9·T10이 크롭에 바로 쓴다. 정규화는 API 경계(T6/T12) | 지금 정규화: 스키마 변경과 같이 해야 함 |
| 부위는 DeepFashion2 클래스 이름 그대로 | 서비스 카테고리 매핑은 T2 미정 | PoC `DF2_TO_PART` 재사용: 카테고리 체계를 임의로 정하는 셈 |
| 검출 0벌은 빈 목록 | 오류 코드 형식은 T6 미정 | 예외 발생: 호출부(T12)에서 정하는 게 맞음 |
| 후처리는 구멍 메우기만, PoC 코드를 옮겨 적음 | PoC에서 셋 다 적용 시 IoU 하락. `app`이 `scripts`를 import하지 않게 | `scripts/` import: 실험 코드에 서버가 묶임 |
| 가중치 sha256 확인 | `.pt`는 pickle이라 로드 시 코드 실행 가능 | 확인 생략: 공급망 위험 |

## 후속 작업

- T9: `Detection.mask`로 누끼 PNG 생성
- T10: 마스크 픽셀 k-means 색상
- T12: lifespan에서 `load_detector()` 호출, 실패 시 `/health` 반영, 0벌·저신뢰 오류 코드
- T13: GPU에서 6벌 워밍 추론 시간 측정

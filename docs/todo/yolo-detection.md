# YOLO 검출+마스크 모듈 (T8)

- 상태: 완료
- 담당: AI 파트
- 관련 기능 ID: F-201, F-202
- 시작일: 2026-11-23
- 완료 Report: `261123-1544-01-yolo-detection.md`

## 목표

T7을 통과한 RGB 이미지에서 YOLO(DeepFashion2)로 옷의 bbox·마스크·부위 클래스를 뽑아, T9(누끼)·T10(색)·T11(태깅)이 그대로 쓸 수 있게 한다.

## 완료 조건

- [x] 가중치 다운로드 스크립트(sha256 확인, 이미 있으면 건너뜀), 가중치는 gitignore
- [x] 프로세스당 1회 로드(`load_detector`, `functools.cache`)
- [x] 마스크 구멍 메우기
- [x] 샘플 이미지에서 bbox·마스크·부위 클래스 반환, `max_items` 반영

## 맥락

- `docs/tasks.md` T8, 이슈 #53. 근거 `261108-0305-01-detection-segmentation-poc.md`.

## 진행 메모

- 2026-11-23: `app/pipeline/detection.py`, `scripts/download_weights.py`, `tests/test_detection.py` 작성. K-Fashion 1장으로 통합 테스트 확인.

## 열린 질문

- [ ] 검출 0벌·저신뢰를 어떤 오류 코드로 돌려줄지 → API v1(T6), 연결은 T12. 지금 `detect()`는 빈 목록만 반환
- [ ] DeepFashion2 13종 → 서비스 카테고리 매핑 → 카테고리 체계(T2) 확정 후
- [ ] `CONF`(0.25)는 K-Fashion PoC 값. Flat Lay(T1)로 다시 정할지 → T5·T13
- [ ] DeepFashion2 가중치는 비상업 라이선스. 공개 서비스 전 교체 또는 허용 확인 (plan §2)

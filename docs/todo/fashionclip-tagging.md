# FashionCLIP zero-shot 태깅 (T11)

- 상태: 진행중
- 담당: AI 파트
- 관련 기능 ID: F-203
- 시작일: 2026-11-24
- 완료 Report: (완료 시 채움)

## 목표

검출된 옷 한 벌마다 추구미 점수·카테고리 후보·패턴·임베딩을 FashionCLIP zero-shot으로 뽑는다. T12가 `/v0/analyze` 응답에 넣을 수 있게 준비한다.

## 완료 조건

- [x] `app/pipeline/tagging.py` — `crop_item`, `load_tagger`, `tag_item`, `TagResult`
- [x] `tests/test_tagging.py` — 단위 테스트(모델 불필요), 통합 테스트(open_clip skip)
- [ ] ruff / pytest 통과 확인
- [ ] 00-todo-list.md 상태 갱신

## 맥락

- `docs/tasks.md` T11. 선행: T8 완료, T2(소분류) 미확정.
- 추구미 점수는 `prompts.item` (보정 없음, plan §4).
- 카테고리 소분류는 `configs/categories.yaml` 초안 사용 (T2 확정 전).
- 패턴 값 목록은 안건 7 AI 제안 초안 (T3에서 백엔드 ERD enum과 맞출 것).
- `crop_item`은 T10(패턴 미구현 부분)을 흡수한다 — FashionCLIP으로 패턴을 같이 판정하므로 별도 `pattern.py`는 만들지 않는다.

## 진행 메모

- 2026-11-24: `app/pipeline/tagging.py`, `tests/test_tagging.py` 작성. 카테고리 소분류는 TBD 상태로 configs 초안을 그대로 사용.

## 열린 질문

- [ ] 소분류 목록 확정 (T2) — 확정되면 `configs/categories.yaml`만 수정
- [ ] 패턴 값 목록 백엔드 ERD 동의 (T3, 안건 7)
- [ ] 임베딩 차원 (`model_version` 기록 후 백엔드에 전달, T3 안건 4)

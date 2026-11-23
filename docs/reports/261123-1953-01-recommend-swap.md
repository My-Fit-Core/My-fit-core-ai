# 슬롯 대체 후보 랭킹 함수 (T16)

- 일시: 2026-11-23 19:53
- 작성: AI 파트 (Claude Code)
- 관련 TODO: `docs/todo/recommend-swap.md`
- 관련 기능 ID: F-304
- 관련 issue: #67

## 요약

`recommend/scoring.py`에 `swap(wardrobe, weather, aesthetic_key, outfit_item_ids, slot)`를 추가했다. 코디의 한 슬롯(대분류)을 옷장의 다른 옷으로 바꿨을 때의 코디 점수를 T14 `recommend()`와 **같은 점수 함수**로 매겨 내림차순으로 돌려준다. 후보가 없거나 슬롯이 맞지 않으면 빈 목록과 사유를 돌려준다. 엔드포인트·스키마는 만들지 않았다(T15·T6 이후).

## 변경 내역

- `recommend/scoring.py`
  - 점수 계산(실효 기온, 목표 따뜻함, 비 감점, 추구미 평균, 가중합)을 `_Scorer`로 묶어 `recommend()`와 `swap()`이 공유. `recommend()` 동작은 그대로(T14 테스트 17개 그대로 통과)
  - `swap()`, `SwapCandidate(item_id, outfit)`, `SwapResult(candidates, reasons, missing_weather, target_warmth)`
- `tests/test_recommend.py`: swap 테스트 10개

## 동작

1. 설정에 없는 대분류면 `unknown_slot`.
2. `outfit_item_ids` 중 옷장에 없는 옷은 빼고 계속 계산하며 `unknown_outfit_item`을 붙인다.
3. 바꿀 수 있는 슬롯은 코디에 있는 슬롯과 같은 뼈대의 슬롯(상의만 있으면 하의도), 아우터, 선택 슬롯(신발·가방·액세서리)이다. 그 밖(예: 상의+하의 코디에 `dress`)은 `slot_not_in_outfit`.
4. 후보는 그 슬롯의 옷 중 지금 코디에 없는 옷. 없으면 `no_candidates`.
5. 후보마다 "다른 슬롯은 그대로 두고 그 슬롯만 바꾼 코디"를 `recommend()`와 같은 점수로 매겨 내림차순(동점은 item id 순).
6. 날씨 결측 처리(`weather_unavailable`, `missing_weather`)와 설정에 없는 소분류 처리(`skipped_unknown_subcategory`)는 `recommend()`와 같다.

## 결과·측정치

- `pytest` 85 passed, 2 skipped(가중치 필요한 통합 테스트), `ruff check` 통과.
- swap 테스트: 같은 슬롯 후보만·현재 옷 제외·다른 슬롯 유지, 추운 날 아우터 교체 시 코트·패딩 1위, 비 오는 날 샌들 감점, 후보 없음(`no_candidates`: 가방 1개뿐 / 아우터 없는 옷장), `unknown_slot`, `slot_not_in_outfit`, `unknown_outfit_item`, 날씨 결측, recommend 1위 코디보다 높은 대체가 없음(같은 점수 함수 확인).

## 설정 분리

- swap은 새 규칙 값을 만들지 않았다. 슬롯 구성(`bases`, `outer_slot`, `optional_slots`), 따뜻함, 날씨 기준, 가중치는 모두 `configs/recommend.yaml`·`configs/categories.yaml`에서 읽는다. 규칙 숫자는 T14와 같이 초안 v0이고 T17에서 조정한다.

## 주요 결정

| 결정 | 근거 | 대안과 버린 이유 |
| --- | --- | --- |
| 점수 함수를 `_Scorer`로 묶어 공유 | 추천과 교체가 다른 점수를 쓰면 "추천은 A가 1위인데 교체 후보에선 B가 위" 같은 어긋남이 생긴다 | swap 전용 점수: 규칙이 두 벌 |
| 같은 뼈대 슬롯은 허용, 다른 뼈대 슬롯은 `slot_not_in_outfit` | 상의+하의 코디에서 dress로 바꾸면 상·하의를 같이 빼야 해 "슬롯 하나 교체"가 아님 | dress 교체 시 상·하의 자동 제거: 규칙 추가 필요, 요청 범위 밖 |
| 코디에 없는 옷 id는 빼고 계속 | 백엔드 옷장과 코디가 잠깐 어긋나도 후보를 줄 수 있게. 사유로 알림 | 바로 실패: 사용자 경험 나쁨 |
| 후보는 전부 반환 | 몇 개를 보여 줄지 기준이 없음 | 상위 k개: 숫자를 새로 정해야 함(열린 질문) |
| 엔드포인트·pydantic 없음 | 사용자 지시, `schemas.py`는 T6 | - |

## 후속 작업

- T15에서 `POST /v0/recommend`와 함께 `POST /v0/recommend/swap` 엔드포인트 연결(`tasks.md` T15에 메모)
- 열린 질문: 아우터 "빼기" 선택지, 후보 개수 (`docs/todo/recommend-swap.md`)

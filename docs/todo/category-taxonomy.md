# 카테고리 체계 확정 (T2)

- 상태: 대기 (사람 결정 필요)
- 담당: AI 파트 + 백엔드
- 관련 기능 ID: F-203
- 시작일: 2026-11-23
- 완료 Report:

## 목표

대분류·소분류 목록과 MVP 검출 범위를 정해 `configs/categories.yaml`과 백엔드 ERD enum을 맞춘다. 이게 정해져야 T6(스키마 v1), T11(태깅), T14(추천)가 시작된다.

## 완료 조건

- [ ] `dress` 대분류 추가 여부 결정
- [ ] 소분류 목록 결정, 백엔드에 전달
- [ ] MVP 검출 범위 결정, `plan.md` §3에 기록
- [ ] `configs/categories.yaml`(T14에서 안 A 초안으로 만듦)을 확정 목록으로 수정, 백엔드 enum과 일치 확인. 소분류가 바뀌면 `configs/recommend.yaml` `warmth`도 함께 고친다

## 맥락

- 확정: `category`(대분류)와 `subcategory`(소분류) 두 필드, 대분류 6종 `outerwear`, `top`, `bottom`, `shoes`, `bag`, `accessory`(백엔드 제안, 10/12 회의 안건 7).
- 기존 `configs/aesthetics.yaml`의 `categories` 블록(`outer`, bag 포함 `accessory`)은 T14에서 지우고 안 A 초안을 `configs/categories.yaml`로 옮겼다(확정 아님).
- 현재 검출기(DeepFashion2 YOLO)의 클래스는 13종 의류뿐이다(가중치에서 확인): `short_sleeved_shirt`, `long_sleeved_shirt`, `short_sleeved_outwear`, `long_sleeved_outwear`, `vest`, `sling`, `shorts`, `trousers`, `skirt`, `short_sleeved_dress`, `long_sleeved_dress`, `vest_dress`, `sling_dress`. 신발·가방·액세서리 클래스는 없다.
- 세부 카테고리는 FashionCLIP 1위 정확도가 약 52%, 상위 3개가 약 75%라 상위 3개 후보를 보여 주고 사용자가 고르는 방식이다(`261108-0315-01-category-poc.md`).

## 초안 (결정 아님, 회의·백엔드 검토용)

### 1. `dress` 대분류

| 안 | 내용 | 장점 | 단점 |
| --- | --- | --- | --- |
| A `[제안]` | 대분류에 `dress` 추가 (7종) | 1020 여성 타겟에서 원피스 비중이 크다. 원피스 한 벌이 상의+하의 두 슬롯을 채우는 규칙을 추천(T14)에서 바로 쓸 수 있다. DeepFashion2에 dress 4종이 있어 검출과 그대로 대응 | 백엔드 enum 추가 필요 |
| B | `top`의 소분류로 원피스를 둔다 | 백엔드 6종 그대로 | 추천에서 "하의가 필요 없는 상의"를 따로 처리해야 해서 규칙이 꼬인다 |

### 2. 소분류 초안 (안 A 기준)

키는 snake_case, 기존 `configs/aesthetics.yaml` 초안을 백엔드 6종 이름에 맞춰 옮겼다.

| 대분류 | 소분류 초안 | 비고 |
| --- | --- | --- |
| `top` | `tshirt`, `shirt`, `blouse`, `knit`, `hoodie`, `sweatshirt`, `sleeveless` | `sweatshirt`(맨투맨) 추가 제안 |
| `bottom` | `jeans`, `slacks`, `shorts`, `mini_skirt`, `midi_skirt`, `long_skirt` | `midi_skirt`는 11/27 추가 |
| `dress` | `mini_dress`, `long_dress` | 안 B면 `top` 아래로 |
| `outerwear` | `jacket`, `blazer`, `coat`, `cardigan`, `padding` | 기존 `outer` → `outerwear` |
| `shoes` | `sneakers`, `flats`, `loafers`, `heels`, `boots`, `sandals` | MVP 검출 범위에 따라 수동 등록만 |
| `bag` | `shoulder_bag`, `tote_bag`, `backpack`, `crossbody_bag` | 기존 `accessory`의 `bag`을 분리. 세부는 새 제안 |
| `accessory` | `hat`, `scarf`, `jewelry` | |

### 3. DeepFashion2 클래스 → 대분류 매핑 초안

| DeepFashion2 | 대분류 | 근거 |
| --- | --- | --- |
| `short_sleeved_shirt`, `long_sleeved_shirt`, `vest`, `sling` | `top` | `vest`는 민소매 상의(`261108-0305-01-detection-segmentation-poc.md`) |
| `short_sleeved_outwear`, `long_sleeved_outwear` | `outerwear` | 부위 판정은 YOLO 단독. 아우터 정확도는 61%라 등록 시 사용자 확인 단계에서 고친다(`261108-0315-01-category-poc.md`, 조합 규칙은 전체 정확도가 떨어져 버림) |
| `shorts`, `trousers`, `skirt` | `bottom` | |
| `short_sleeved_dress`, `long_sleeved_dress`, `vest_dress`, `sling_dress` | `dress` (안 B면 `top`) | |

### 4. MVP 검출 범위

| 안 | 내용 | 장점 | 단점 |
| --- | --- | --- | --- |
| A `[제안]` | 자동 검출은 의류(`top`, `bottom`, `outerwear`, `dress`)만. `shoes`, `bag`, `accessory`는 사진 한 장 등록 시 사용자가 직접 고르거나 단독 사진으로 등록 | 지금 검출기 그대로 쓸 수 있어 추가 작업이 가장 적다(일정은 추정) | "사진 한 장으로 전부 등록"이 의류에 한정됨 |
| B | 신발·가방용 검출기 추가(다른 공개 가중치 또는 Grounding DINO 같은 open-vocabulary 검출) | 기획 그대로 | 모델 조사·평가·속도 측정이 추가로 필요. 검출기가 늘면 2초 목표에 부담이 될 수 있다(미측정) |
| C | 단독 사진(옷 한 벌만 찍은 사진)은 검출 없이 전체를 한 벌로 보고 FashionCLIP으로 대분류까지 판정 | A와 같이 쓰면 신발·가방도 자동 태깅 가능 | 누끼는 별도 처리 필요, 판정 정확도 미측정 |

## 진행 메모

- 2026-11-23: 초안 작성(이슈 #60). 결정은 사람이 한다.

## 열린 질문

- [ ] `dress` 대분류 추가 (안 A/B)
- [ ] 소분류 목록 확정 (위 초안 수정 여부)
- [ ] MVP 검출 범위 (안 A/B/C)
- [ ] 카테고리 정확도 90% 승인 조건을 대분류 기준으로 볼지, 소분류 기준으로 볼지

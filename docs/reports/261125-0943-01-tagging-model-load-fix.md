# FashionCLIP 태깅 모델 로드 404·점수 스케일 누락 수정

- 일시: 2026-11-25 09:43
- 작성: AI 파트 (Claude Code)
- 관련 TODO: `docs/todo/fashionclip-tagging.md`
- 관련 기능 ID: F-203
- 관련 issue: #87 (PR #88)

## 요약

PR #72로 들어온 `app/pipeline/tagging.py`는 단위 테스트는 통과했지만 실제 FashionCLIP 모델을 불러오지 못했다(404). 불러와지더라도 점수에 CLIP 스케일을 곱하지 않아 추구미·카테고리 점수가 거의 균등하게 나왔다. 실험 스크립트와 같은 로더·같은 계산으로 맞췄고, 통합 테스트가 실제 모델로 통과한다.

## 변경 내역

- `app/pipeline/tagging.py`
  - 모델 로드: `open_clip`(`hf-hub:patrickjohncyh/fashion-clip`) → `transformers` `CLIPModel`/`CLIPProcessor`. 이 Hugging Face 저장소는 transformers 형식이라 `open_clip_config.json`이 없어 404가 났다. 실험 스크립트(`scripts/eval_*.py`)는 처음부터 transformers로 불러왔다.
  - 점수: softmax 전에 `logit_scale.exp()`(약 99.8)를 곱한다. 추구미는 "scale × 프롬프트별 유사도 평균 → softmax"로 `scripts/eval_item_aesthetics_kfashion.py`의 `style_logits`와 같다. 카테고리 후보에도 같은 scale을 적용했다. 패턴은 argmax만 써서 결과가 바뀌지 않는다.
  - 설정 yaml을 `encoding="utf-8"`로 읽는다(Windows 기본 인코딩에서 한글 설정이 깨질 수 있음).
  - `load_tagger()` 반환값이 `(model, preprocess, tokenizer)` → `(model, processor)`. 다른 호출부는 `/health`의 `cache_info()`뿐이라 영향 없음.
- `tests/test_tagging.py`: 통합 테스트 skip 조건을 `open_clip` → `transformers` 기준으로.
- `requirements-ml.txt`: 쓰는 곳이 없어진 `open_clip_torch` 제거.

## 결과·측정치

- `pytest` 97 passed, 2 skipped(검출 가중치·샘플 사진 없음), `ruff check` 통과. `test_tag_item_integration`이 이 환경에서 처음으로 실제 모델로 실행·통과.
- 수정 전 점수: 추구미 5개가 모두 약 0.2(scale 없이 코사인 유사도에 바로 softmax).
- 실제 사진 확인(K-Fashion Validation 1장, T8 검출 2벌, Windows CPU):

  | 옷 | 추구미 1위 | 카테고리 후보 1~3위 | 패턴 |
  | --- | --- | --- | --- |
  | 바지(`trousers`) | casual 1.0 | jeans 1.0, slacks 0.0, padding 0.0 | other |
  | 셔츠(`short_sleeved_shirt`) | casual 0.97 | blouse 0.47, tshirt 0.27, shirt 0.26 | graphic |

- 임베딩 512차원. 시간: 첫 호출 12초(모델 로드 포함), 이후 옷당 약 1초(CPU). 6벌이면 태깅만 약 6초라 "6벌 2초"는 GPU에서 다시 재야 한다.
- 사진 1장으로 동작만 확인했다. 정확도는 미검증이다.

## 주요 결정

| 결정 | 근거 | 대안과 버린 이유 |
| --- | --- | --- |
| transformers 로더로 교체 | 이 저장소의 공식 형식이고, 실험 결과(`261110-0117` 등)가 모두 이 로더로 나왔다 | open_clip용 변환 체크포인트 찾기: 출처 확인 부담, 실험과 수치가 달라질 위험 |
| scale × 유사도 평균 → softmax | 실험 Report와 같은 계산이어야 실험에서 정한 프롬프트·판단이 서비스에서도 유지됨 | scale 없이 softmax: 점수가 균등해 의미 없음 |
| `open_clip_torch` 의존성 제거 | 쓰는 곳 없음 | 유지: 설치만 무거워짐 |

## 배운 점

- 단위 테스트만 있고 통합 테스트가 skip되는 환경에서는 "모델을 실제로 한 번도 안 불러 본" 코드가 머지될 수 있다. 모델을 쓰는 모듈은 PR 전에 실제 로드를 한 번 돌려 본다.
- scale 누락처럼 에러 없이 조용히 틀리는 버그는 실험 스크립트와 계산을 대조해야 잡힌다.

## 후속 작업

- 추구미 점수 쏠림(청바지·셔츠 모두 casual 0.97 이상)이 실제 사진에서도 큰지 T1 사진으로 확인, 대학원생 자문 때 함께 논의
- 프롬프트 텍스트 임베딩은 매 호출 다시 계산한다. 속도가 문제면 프롬프트 임베딩을 캐시한다(T13 측정 후)
- GPU(학교 서버)에서 옷당 시간 측정(T13)

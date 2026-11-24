# 마스크 픽셀 k-means 색상 추출 (T10 색상)

- 일시: 2026-11-23 16:41
- 작성: AI 파트 (Claude Code)
- 관련 TODO: `docs/todo/color-pattern.md`
- 관련 기능 ID: F-203
- 관련 issue: #62

## 요약

`app/pipeline/color.py`에 `extract_colors(image, mask)`를 만들었다. 마스크 안 픽셀만 k-means(k=3)로 묶어 `("#RRGGBB", ratio)` 목록을 비율 내림차순, 합 1로 돌려준다. T10의 나머지 절반인 패턴은 방법과 값 목록이 정해지지 않아 구현하지 않고 열린 질문으로 남겼다.

## 변경 내역

- `app/pipeline/color.py`: `extract_colors(image, mask, k=3, max_pixels=10_000, seed=0)`
  - 마스크 픽셀이 1만 개를 넘으면 seed 고정 무작위 표본 1만 개만 쓴다
  - 클러스터 수는 `min(k, 서로 다른 색 수)` (무지일 때 sklearn 경고 방지)
  - 빈 마스크는 `[]`, 마스크 크기가 이미지와 다르면 `ValueError`
- `tests/test_color.py`: 합성 이미지 테스트 7개(단색, 마스크 밖 무시, 7:3 줄무늬 비율·형식, 색 수 > k, 빈 마스크, 크기 불일치, 2000×2000 표본·결정성)

## 결과·측정치

- `pytest` 58 passed, 2 skipped(가중치 필요한 통합 테스트), `ruff check` 통과.
- 실제 사진 확인(K-Fashion Validation 1장, T8 검출 2벌, Windows CPU):

  | 옷 | 대표색 (ratio) |
  | --- | --- |
  | `trousers` | `#1F4053` 0.69, `#B1B5C1` 0.16, `#50718D` 0.16 |
  | `short_sleeved_shirt` | `#E7E2E8` 0.74, `#BEA191` 0.13, `#5E3F28` 0.13 |

  남색 바지와 흰 셔츠가 1위 색으로 나왔다. 2·3위에는 마스크 경계에 걸린 피부·배경 색이 섞인 것으로 보인다(눈으로만 확인, 미측정).
- 시간(같은 2벌, 표본 1만 개, 3회 중 최소): `n_init=4` 94~114ms, `n_init=1` 6~8ms이고 결과 색·비율은 같았다. 첫 호출은 sklearn 준비 시간이 더해져 0.2~0.5초였다. CPU 부하에 따라 흔들려 참고용이다.

## 주요 결정

| 결정 | 근거 | 대안과 버린 이유 |
| --- | --- | --- |
| 패턴은 이번에 구현하지 않음 | 방법·값 목록이 문서에 없고 백엔드 enum과도 맞춰야 함(임의 결정 금지) | 통계 임계값으로 solid/patterned 판정: 값 목록과 임계값을 임의로 정하게 됨 |
| `n_init=1` | 실제 사진 2벌에서 `n_init=4`와 같은 결과, 시간 약 1/10. 6벌 2초 예산 | `n_init=4`: 6벌이면 0.6초 이상 |
| k=3, 작은 색 버리지 않음 | 근거 문서 없는 기본값. `ponytail:` 주석으로 표시 | 임계값으로 버리기: 기준 없음 |
| 반환은 tuple 목록 | 스키마 `ColorInfo`가 T6에서 바뀔 수 있어 변환은 T12에서 | 지금 `ColorInfo` 반환: 스키마에 묶임 |

## 후속 작업

- 패턴: 방법(T11에 합치는 안 제안)과 값 목록 결정 후 구현. `docs/todo/color-pattern.md` 열린 질문
- 마스크 경계의 피부·배경 색 섞임이 Flat Lay 사진에서도 문제인지 T1 사진으로 확인
- T12에서 `extract_colors` 결과를 `ColorInfo`로 변환

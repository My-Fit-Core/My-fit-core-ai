# 색상·패턴 추출 (T10)

- 상태: 진행중 (색상 완료, 패턴 결정 대기)
- 담당: AI 파트
- 관련 기능 ID: F-203
- 시작일: 2026-11-23
- 완료 Report: `261123-1641-01-color-extraction.md` (색상)

## 목표

검출된 옷(T8 `Detection.mask`)마다 대표색 `colors[{hex, ratio}]`와 `pattern`을 뽑아 T12가 응답에 넣을 수 있게 한다.

## 완료 조건

- [x] 마스크 픽셀 k-means 색상 `colors[{hex, ratio}]` (이슈 #62)
- [ ] `pattern` 반환 — 방법·값 목록 결정 후
- [x] 무지/무늬 합성 이미지 테스트 (색상)

## 맥락

- `docs/tasks.md` T10. 색상 방식은 `02-specs.md` §2 "색상은 마스크 픽셀 k-means".
- 패턴은 `02-specs.md`·`plan.md`에 방법이 없다. 단서는 `vision-pipeline-poc.md`의 "FashionCLIP zero-shot 카테고리·패턴"뿐이고, v0 스키마 예시 값은 `"solid"` 하나다.

## 진행 메모

- 2026-11-23: `app/pipeline/color.py` 작성. 패턴은 미정이라 구현하지 않음.

## 열린 질문

- [ ] 패턴 판정 방법: (a) T11에서 FashionCLIP zero-shot으로 같이 판정(크롭·모델 공유, AI 제안) vs (b) 이미지 통계(색 분포·엣지 밀도) 기반. (b)는 임계값을 검증할 데이터가 필요하다
- [ ] `pattern` 값 목록(예: 무지·스트라이프·체크·도트·플로럴·그래픽 등) — 백엔드 ERD enum과 맞춰야 해서 T2·T3에서 함께 정한다
- [ ] 대표색 개수(k=3)와 비율이 아주 작은 색을 버릴지 — 앱 화면에 몇 개를 보여 줄지 기준이 있으면 맞춘다

# 00. TODO 목록

> 작업을 시작하거나 끝낼 때 이 표의 상태를 갱신합니다. 상세 맥락은 각 파일에 있습니다.
> 상태: 대기 / 진행중 / 보류 / 완료

## P0 — MVP 블로커 (순서대로 처리)

| 상태 | 작업 | 이슈 | 관련 기능 | 파일 | 완료 Report |
| --- | --- | --- | --- | --- | --- |
| 완료 | 레포 초기 세팅 | — | — | — | `261009-1840-01-repo-initial-setup.md` |
| 진행중 | 회의 확정 사항 정리 (추구미·카테고리·지연시간·추천 위치 등) | — | 전체 | `meeting-decisions-1013.md` | |
| 진행중 | 백엔드 API 계약 v1 — 인증·임베딩·콜백 미확정 항목 완료 | #98 | F-201~203 | `api-contract-v1.md` | |
| 대기 | 카테고리 소분류 확정 및 categories.yaml 반영 (T2) | #97 | F-203 | `category-taxonomy.md` | |
| 진행중 | Vision 파이프라인 PoC (검출+누끼+태깅) | — | F-201, F-202, F-203 | `vision-pipeline-poc.md` | |
| 진행중 | FashionCLIP zero-shot 태깅 (T11, 소분류 T2 대기) | — | F-203 | `fashionclip-tagging.md` | `261125-0943-01-tagging-model-load-fix.md` |
| 대기 | /v0/analyze 통합 엔드포인트 (검출·누끼·태깅 연결) | #101 | F-201~203 | `analyze-endpoint.md` | |
| 진행중 | 테스트셋 구축 및 정확도/오진율 평가 스크립트 (261125 피드백 필수) | #99 | F-201~203, F-302, F-402 | `testset-and-eval.md` | |

## P0 — MVP 필수 (병렬 가능)

| 상태 | 작업 | 이슈 | 관련 기능 | 파일 | 완료 Report |
| --- | --- | --- | --- | --- | --- |
| 진행중 | 추천 엔진 v0 (엔드포인트 완료, 점수 고도화 대기) | — | F-302, F-304 | `recommend-engine-v0.md` | `261123-1652-01-recommend-scoring-v0.md` |
| 대기 | 추구미별 온보딩 이미지 선정 및 Y2K 데이터 소스 확보 | #100 | F-201, F-203, F-302 | `data-and-target-review.md` | |
| 대기 | 서빙 플랫폼 선정 및 Docker 배포 테스트 | #102 | 서빙 | `serving-platform.md` | |

## P1 — 오픈 베타 전

| 상태 | 작업 | 이슈 | 관련 기능 | 파일 | 완료 Report |
| --- | --- | --- | --- | --- | --- |
| 대기 | LLM 스타일리스트 코멘트 — 모델 선정 + 프로토타입 | #103 | F-303 | `llm-stylist-comment.md` | |
| 대기 | 내 옷장으로 따라입기 유사도 검색 — 대안 조사 + 구현 | #104 | F-402 | `followalong-similarity.md` | |
| 대기 | 추천 점수 고도화 — 임베딩 유사도 + 가중치 | #106 | F-302 | `recommend-scoring-v1.md` | |

## 조사/실험

| 상태 | 작업 | 이슈 | 관련 기능 | 파일 | 완료 Report |
| --- | --- | --- | --- | --- | --- |
| 진행중 | 상의·하의 어울림 점수(compatibility) 실험 | #91 | F-302 | — | |
| 대기 | 오진율 개선 — 슬롯별 피드백 루프 설계 조사 | #105 | F-302 | `feedback-loop-design.md` | |
| 대기 | 누끼 모델 후보 비교 (MobileSAM, SAM 2 tiny) | #41 | F-202 | — | |
| 보류 | FashionCLIP 임베딩 위에 linear probe 학습 검토 | #34 | F-203 | — | |

## 완료

| 상태 | 작업 | 완료 Report |
| --- | --- | --- |
| 완료 | 입력 이미지 검증 모듈 (T7) | `261123-1535-01-input-image-validation.md` |
| 완료 | YOLO 검출+마스크 모듈 (T8) | `261123-1544-01-yolo-detection.md` |
| 완료 | 02-specs.md를 10/12 회의 결정에 맞게 갱신 (T24) | `261123-1624-01-specs-1012-decisions.md` |
| 완료 | 색상·패턴 추출 (T10) | `261123-1641-01-color-extraction.md` |
| 완료 | "15대 추구미"·TPO 표현 정리 (T22) | `261123-1643-01-docs-wording-cleanup.md` |
| 완료 | 슬롯 대체 후보 랭킹 함수 (T16) | `261123-1953-01-recommend-swap.md` |
| 완료 | K-Fashion 데이터 준비와 FashionCLIP 스타일 평가 | `261031-2152-01-kfashion-style-eval.md` |
| 완료 | 추구미 설명형 프롬프트, 비율 보정 | `261108-0230-01-aesthetic-display.md` |
| 완료 | 추구미를 기본 스타일 5개로 재정의 | `261109-1840-01-base-styles.md` |
| 완료 | 옷 한 벌용 추구미 프롬프트(prompts.item) 작성과 측정 | `261110-0117-01-item-prompts.md` |

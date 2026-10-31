# K-Fashion 데이터 준비와 FashionCLIP 스타일 평가

- 상태: 완료
- 담당: AI 파트
- 관련 기능 ID: F-203, F-302
- 시작일: 2026-10-31
- 완료 Report: `261031-2152-01-kfashion-style-eval.md`

## 목표

AI Hub K-Fashion Validation 세트로 FashionCLIP이 일반 스타일을 구분하는지 확인하고, 기획의 추구미 5개를 K-Fashion 스타일과 연결할 근거를 만든다.

## 완료 조건

- [x] Validation 라벨과 원천데이터를 받아 무결성 확인
- [x] 라벨 구조와 스타일 분포를 `docs/04-kfashion-validation.md`에 정리
- [x] 실험 노트북을 로컬(CPU)에서 `SMOKE_TEST`로 확인
- [x] 스타일별 정확도, 균형 정확도, 혼동 행렬 산출

## 맥락

학습에는 쓰지 않는다. AI Hub 약관상 재배포 금지라 데이터와 이미지는 레포와 문서에 넣지 않는다.

## 진행 메모

- 2026-10-31: aihubshell이 Git Bash에서 한글 파일명 조각을 지워 0바이트 zip을 남기는 문제와, 연결이 끊겨도 성공처럼 출력하는 문제를 확인했다. `docs/04-kfashion-validation.md`에 대응 방법을 적었다.

## 열린 질문

> 사람의 확인이 필요한 것. 답이 나오면 답과 날짜를 함께 적는다.

- [x] 추구미 5개를 K-Fashion 스타일 어디에 연결할지 → 2026-11-01 잠정 연결 (`261101-0002-03-aesthetic-style-mapping.md`). 미니멀과 Y2K는 근거가 약함
- [x] 스타일별 특징을 적은 프롬프트로 바꾸면 결과가 나아지는지 → 2026-11-01 균형 정확도 19.7%에서 23.0%로 소폭 개선, 예측 일치 21%로 불안정 (`261101-0002-02-described-style-prompts.md`)

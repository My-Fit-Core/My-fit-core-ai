# 백엔드와 분석 API 계약 합의

- 상태: 대기
- 담당: AI 파트 + 백엔드
- 관련 기능 ID: F-201, F-202, F-203
- 시작일:
- 완료 Report:

## 목표

`02-specs.md`의 v0 초안을 백엔드와 검토해 v1로 확정하고, 백엔드가 AI 서버 완성 전에도 목업 응답으로 개발할 수 있게 한다.

## 완료 조건

- [ ] 요청/응답 필드, 에러 형식, job 상태값 합의
- [ ] `app/schemas.py`를 v1으로 갱신 (에러 enum 포함)
- [ ] 목업 응답 예시 JSON을 백엔드에 공유
- [ ] 노션 API 설계서에 링크

## 확정된 사항

**에러 코드 7종** (안건 5 — 배윤선 확인 2026-11-24)

| 재시도 가능 | `WORKER_TIMEOUT`, `MODEL_UNAVAILABLE`, `STORAGE_UNAVAILABLE`, `INTERNAL_ERROR`, `CALLBACK_TIMEOUT` |
| 재시도 불가 | `NO_GARMENT_DETECTED`, `ANALYSIS_FAILED` |

- 에러 형식: `{"code": "...", "message": "...", "client_item_key": "옷 키 또는 null"}`
- 이미지 규격 위반(형식·용량·손상)은 `ANALYSIS_FAILED`로 처리 (전용 코드 없음)
- 재시도 가능 여부 판단은 백엔드가 코드로 처리

**입력 사진 규격** (안건 6 — 배윤선 확인 2026-11-24)

- FE가 종횡비 유지 채로 긴 변 최대 2048px 리사이징 후 업로드
- AI 서버 리사이징 로직은 예외 대비로 유지
- ⚠️ FE 구현 완료 후 AI 검출·누끼 품질 최종 검증 필요

## 맥락

백엔드는 Supabase + Vercel, AI는 별도 GPU 서버에서 비동기 실행 예정. 레포가 분리되어 있어 스키마 어긋남이 가장 큰 리스크.

## 열린 질문

- [x] 누끼 PNG 업로드 주체 → 백엔드가 발급한 URL로 AI 서버가 업로드
- [ ] AI 서버 인증 방식 (공유 시크릿 헤더 `X-AI-Server-Secret` 제안 — BE 확인 필요)
- [x] 임베딩 전달 → 분석 결과에 포함, 추천 요청 시 옷장 목록에 담아 전달 (안건 4 AI 제안)

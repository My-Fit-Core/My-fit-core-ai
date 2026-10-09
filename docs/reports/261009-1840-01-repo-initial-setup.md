# 레포 초기 세팅

- 일시: 2026-10-09 18:40
- 작성: AI 파트
- 관련 TODO: -
- 관련 기능 ID: -

## 요약

AI 서버 레포의 문서 구조(docs/todo/reports), 에이전트 공통 진입점(AGENTS.md), FastAPI 서버 뼈대와 API 스키마 초안을 만들었다.

## 변경 내역

- `AGENTS.md`, `CLAUDE.md`: 사람·AI 공통 작업 규칙
- `docs/01~03`: 폴더 책임, 기술 제약·API 계약 초안, 제품 요약
- `docs/todo`, `docs/reports`: 템플릿과 초기 TODO 3건
- `app/`: FastAPI 헬스체크와 `/v0/analyze` 목업 엔드포인트, pydantic 스키마
- `configs/aesthetics.yaml`: 추구미·카테고리 임시 목록
- `.gitignore`, `.env.example`, `requirements*.txt`, `Dockerfile`

## 주요 결정

| 결정 | 근거 | 대안과 버린 이유 |
| --- | --- | --- |
| 백엔드와 레포 분리 | 배포 환경(Vercel vs GPU)·의존성·배포 주기가 다름 | 모노레포: 배포 설정 복잡 |
| 규칙은 AGENTS.md 하나에, CLAUDE.md는 포인터 | 팀원이 서로 다른 AI 도구를 써도 같은 규칙을 읽게 | 도구별 규칙 파일: 내용이 갈라짐 |
| 가중치·데이터는 레포 밖 | 용량, 히스토리 오염 방지 | Git LFS: 무료 용량 한계 |
| 모델 의존성을 `requirements-ml.txt`로 분리 | 목업 서버는 GPU 없이 바로 실행 가능하게 | 단일 파일: 로컬 설치가 무거움 |

## 후속 작업

- `docs/todo/meeting-decisions-1013.md`
- `docs/todo/api-contract-v1.md`
- `docs/todo/vision-pipeline-poc.md`

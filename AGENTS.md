# AGENTS.md — Working Agreement

> 이 레포에서 작업하는 **사람과 AI 에이전트 모두**가 가장 먼저 읽는 문서입니다.
> Claude Code, Cursor, Codex, Copilot 등 어떤 도구를 쓰든 이 문서의 규칙을 따릅니다.

## 이 레포는

MyFit:Core(추구미 기반 코디 추천 앱)의 **AI 서버 레포**입니다.
다중 의류 검출·누끼·태깅, 날씨 × 추구미 코디 추천, '내 옷장으로 따라입기' 유사도 검색을 담당합니다.
백엔드(Supabase + Vercel)는 별도 레포이며, 이 레포는 백엔드가 비동기로 호출하는 추론 API를 제공합니다.

## 작업 시작 전 읽는 순서

1. `docs/03-product-plan.md` — 무엇을, 왜 만드는가 (제품 목적·범위·승인 조건)
2. `docs/02-specs.md` — 어떻게 만들어야 하는가 (기술 제약·API 계약·구현 규칙)
3. `docs/01-folder-architecture.md` — 어디에 두는가 (폴더 구조와 책임)
4. `docs/todo/00-todo-list.md` — 지금 무엇을 해야 하는가
5. 관련 작업의 `docs/todo/<slug>.md` 와 최근 `docs/reports/` — 맥락과 이전 결정

## 작업 흐름

1. **시작**: `docs/todo/00-todo-list.md`에서 작업을 찾는다. 없으면 `docs/todo/_template.md`로 새 TODO를 만들고 목록에 추가한다.
2. **진행**: TODO 상태를 `진행중`으로 바꾸고, 작업 중 생긴 질문·막힌 점은 해당 TODO 파일에 적는다.
3. **완료**: `docs/reports/_template.md`로 Report를 작성하고, TODO 상태를 `완료`로 바꾼 뒤 Report 파일명을 연결한다.
4. **결정이 생기면**: 구현 방향을 바꾸는 결정은 Report의 "주요 결정" 섹션에 근거와 함께 남긴다. 규칙으로 굳어질 결정이면 `02-specs.md`에도 반영한다.

## 반드시 지킬 것

- **확정되지 않은 사항(문서에 `TBD`로 표시된 것)을 임의로 결정하지 않는다.** 필요한 경우 TODO 파일의 "열린 질문"에 남기고 사람에게 확인한다.
- **백엔드와의 API 스키마(`app/schemas.py`, `02-specs.md`의 API 계약)를 바꿀 때**는 버전을 올리고 Report를 남긴다. 백엔드가 이 스키마에 의존한다.
- **커밋 금지**: 모델 가중치(`*.pt`, `*.pth`, `*.onnx`, `*.safetensors`), 테스트 이미지·데이터셋, `.env`, 노트북 출력.
- 문서는 한국어로 작성한다. 코드 식별자와 커밋 메시지 prefix는 영어.
- 문서 내용과 코드가 어긋나면 임의로 한쪽에 맞추지 말고 어긋남을 보고한다.

## 커밋·브랜치 규칙

- `main`: 배포 브랜치. `develop`에서 PR로만 병합한다.
- `develop`: 개발 통합 브랜치. 작업 브랜치는 여기서 만들고 여기로 PR한다.
- 작업 브랜치: `feat/<slug>`, `fix/<slug>`, `docs/<slug>`, `exp/<slug>`(실험)
- 커밋 메시지: `feat: ...`, `fix: ...`, `docs: ...`, `exp: ...`, `chore: ...`

## 파일명 규칙

- TODO 상세: `docs/todo/<kebab-case-slug>.md`
- Report: `docs/reports/YYMMDD-HHMM-NN-<slug>.md` (예: `261009-1840-01-repo-initial-setup.md`)

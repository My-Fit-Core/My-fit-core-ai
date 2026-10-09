# 01. 폴더 구조와 책임

새 파일을 만들기 전에 아래 표에서 위치를 확인합니다. 어디에도 맞지 않으면 임의로 폴더를 만들지 말고 TODO에 질문으로 남깁니다.

```
myfit-ai/
├── AGENTS.md                 # 작업 규칙·진입점 (사람 + AI 공통)
├── CLAUDE.md                 # Claude Code용 포인터 → AGENTS.md
├── README.md                 # 실행 방법, 빠른 시작
├── docs/
│   ├── 01-folder-architecture.md
│   ├── 02-specs.md
│   ├── 03-product-plan.md
│   ├── todo/                 # 해야 할 일, 후속·보류 작업
│   └── reports/              # 완료한 일, 주요 결정
├── app/                      # FastAPI 서빙 (백엔드가 호출하는 추론 API)
│   ├── main.py               # 엔드포인트 정의
│   ├── schemas.py            # 백엔드와의 요청/응답 계약 (변경 시 버전업)
│   └── pipeline/             # 검출 → 분할/누끼 → 태깅 파이프라인
├── recommend/                # 날씨 × 추구미 × TPO 코디 추천 로직 (GPU 불필요)
├── configs/
│   └── aesthetics.yaml       # 15대 추구미 정의, CLIP 프롬프트, 카테고리 체계
├── notebooks/                # PoC·실험 (출력 지우고 커밋)
├── scripts/                  # 평가, 가중치 다운로드 등 일회성/유틸 스크립트
├── tests/
├── Dockerfile
├── requirements.txt          # 서빙 기본 의존성
├── requirements-ml.txt       # 모델 의존성 (torch 등, GPU 환경)
└── .env.example
```

## 디렉터리별 책임

| 경로 | 책임 | 두지 않는 것 |
| --- | --- | --- |
| `app/` | HTTP 입출력, 요청 검증, 파이프라인 호출 | 모델 로직 자체, 추천 규칙 |
| `app/pipeline/` | 이미지 → 아이템 리스트 변환 (검출·누끼·태깅·임베딩) | HTTP 코드 |
| `recommend/` | 옷장 아이템 + 날씨 + 추구미 + TPO → 코디 후보 점수화 | 모델 추론 (임베딩·점수는 입력으로 받음) |
| `configs/` | 사람이 합의해서 바꾸는 값 (추구미, 카테고리, 임계값) | 코드 |
| `notebooks/` | 실험, 결과 확인 | 서빙에서 import되는 코드 |
| `scripts/` | 평가 스크립트, 가중치 다운로드 | 서빙 로직 |
| `docs/todo/` | 진행 전·진행 중 작업의 맥락과 상태 | 완료 기록 |
| `docs/reports/` | 완료 결과, 변경 내역, 주요 결정 | 앞으로 할 일 |

## 레포 밖에 두는 것

| 대상 | 위치 |
| --- | --- |
| 모델 가중치 | Hugging Face Hub 등에서 실행 시 다운로드 (`weights/`는 gitignore) |
| 테스트 사진·데이터셋 | 팀 공유 스토리지 (경로만 `02-specs.md`에 기록), 로컬은 `data/` (gitignore) |
| API 키 | `.env` (gitignore), 키 이름은 `.env.example`에 기록 |

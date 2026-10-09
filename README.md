# MyFit AI Server

MyFit:Core의 AI 서버입니다. 다중 의류 검출·누끼·태깅, 코디 추천, 따라입기 유사도 검색을 담당합니다.

**처음 오셨다면 [AGENTS.md](AGENTS.md)부터 읽어주세요.** 사람과 AI 에이전트 모두 같은 규칙을 따릅니다.

## 빠른 시작 (목업 서버, GPU 불필요)

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

- 헬스체크: http://localhost:8000/health
- API 문서: http://localhost:8000/docs

모델 실험(PoC)은 GPU 환경(Colab 등)에서 `pip install -r requirements-ml.txt` 후 진행합니다.

## 노트북 커밋 전 설정 (최초 1회)

```bash
pip install nbstripout
nbstripout --install
```

## 문서

| 문서 | 내용 |
| --- | --- |
| [docs/03-product-plan.md](docs/03-product-plan.md) | 무엇을, 왜 |
| [docs/02-specs.md](docs/02-specs.md) | 기술 제약, 백엔드와의 API 계약 |
| [docs/01-folder-architecture.md](docs/01-folder-architecture.md) | 폴더 구조와 책임 |
| [docs/todo/00-todo-list.md](docs/todo/00-todo-list.md) | 할 일 |
| [docs/reports/](docs/reports/) | 완료한 일, 주요 결정 |

## 관련 레포

- 백엔드: (링크 추가 예정)

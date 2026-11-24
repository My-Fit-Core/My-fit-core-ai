# 입력 이미지 검증 모듈 (T7)

- 일시: 2026-11-23 15:35
- 작성: AI 파트 (Claude Code)
- 관련 TODO: `docs/todo/input-image-validation.md`
- 관련 기능 ID: F-201
- 관련 issue: #52

## 요약

`app/pipeline/image_input.py`에 `validate_image(data: bytes)`를 만들었다. 용량 → 형식 → 긴 변 → 디코딩 순서로 검사하고, EXIF 회전을 적용한 뒤 EXIF를 지운 RGB 이미지와 경고 목록을 돌려준다. 형식·용량은 `configs/image_input.yaml`에서 읽으며, 10/12 회의 안건 9에 따라 JPEG/PNG/WebP, 20MB로 정했다.

## 변경 내역

- `app/pipeline/image_input.py`: `validate_image`, `ValidatedImage`, `ImageValidationError(reason)`
- `configs/image_input.yaml`: `allowed_formats`, `max_mb`
- `tests/test_image_input.py`: 합성 이미지 테스트 13개(정상 JPEG/PNG/WebP·MPO, GIF·BMP 거절, 비이미지·잘린 JPEG, 용량 초과, 긴 변 2048/2049, 짧은 변 경고, EXIF Orientation=6 회전·제거)
- `requirements.txt`: `pillow==13.0.0` 추가(서버 기본 환경에서도 필요). `requirements-ml.txt`에서는 제거(`-r requirements.txt`로 포함됨)
- `docs/02-specs.md` §3, `docs/plan.md` §4: 형식 WebP 추가, 용량 5MB → 20MB, 설정 파일 위치
- `docs/plan.md`, `docs/tasks.md`: 레포에 처음 커밋(이전에는 작업 트리에만 있었다)

## 결과·측정치

- `pytest`, `ruff check` 통과. 이미지 파일은 커밋하지 않고 테스트 안에서 PIL로 만든다.

## 주요 결정

| 결정 | 근거 | 대안과 버린 이유 |
| --- | --- | --- |
| 형식 JPEG/PNG/WebP, 최대 20MB | 10/12 회의 안건 9 (2026-11-23 사용자 지시) | 기존 JPEG/PNG, 5MB: 회의 결정으로 변경 |
| 형식·용량은 `configs/image_input.yaml` | 기준이 회의로 바뀌므로 코드 수정 없이 바꿀 수 있게 | 환경변수: 다른 설정(`aesthetics.yaml`)과 같은 위치가 찾기 쉽다 |
| 긴 변은 디코딩 전에 헤더 크기로 거절 | 디컴프레션 폭탄 방어, 불필요한 디코딩 생략 | 디코딩 후 검사: 메모리 낭비 |
| 긴 변 2048px 초과는 실패 | `plan.md` §7 "규격 위반은 failed" | 리사이즈 후 통과: 정해진 적 없어 열린 질문으로 남김 |
| `reason` 문자열은 모듈 내부 값 | 응답 오류 코드 형식은 API v1(T6) 몫 | 지금 스키마에 반영: T6 범위라 하지 않음 |
| 투명 영역은 단순 `convert("RGB")`(알파만 버림) | 흰 배경 합성 여부 미정 | 열린 질문으로 남김 |
| MPO(Multi-Picture JPEG)는 JPEG로 취급 | 휴대폰 JPEG를 Pillow가 MPO로 읽어 정상 사진이 거절됨(리뷰 지적) | 설정에 MPO 추가: 설정을 보는 사람이 JPEG와 같은 것인지 알기 어렵다 |

## 후속 작업

- T8 검출 모듈은 `validate_image`를 통과한 RGB 이미지를 입력으로 받는다.
- T6에서 오류·경고 코드 형식 확정 후 `reason` 문자열을 맞춘다.
- `meeting-decisions-1013.md`에 10/12 안건 9(WebP·20MB) 반영 필요(사람).

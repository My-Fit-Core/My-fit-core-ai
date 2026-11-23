# 입력 이미지 검증 모듈 (T7)

- 상태: 완료
- 담당: AI 파트
- 관련 기능 ID: F-201
- 시작일: 2026-11-23
- 완료 Report: `261123-1535-01-input-image-validation.md`

## 목표

업로드 이미지가 `02-specs.md` §3 입력 이미지 규칙을 지키는지 AI 서버에서 다시 확인하고, 회전 보정·EXIF 제거된 RGB 이미지를 다음 단계(T8 검출)에 넘긴다.

## 완료 조건

- [x] 형식(JPEG/PNG/WebP), 용량(20MB), 긴 변 2048px 검사, 손상 파일 거절
- [x] EXIF 회전 보정과 제거, 짧은 변 512px 미만 경고
- [x] 형식·용량은 `configs/image_input.yaml` 설정값으로 분리
- [x] 정상/규격 위반/회전 이미지 단위 테스트(합성 이미지)

## 맥락

- `docs/tasks.md` T7, 이슈 #52.
- 2026-11-23 사용자 지시로 규격 변경: JPEG/PNG → JPEG/PNG/WebP, 5MB → 20MB (10/12 회의 안건 9).

## 진행 메모

- 2026-11-23: `app/pipeline/image_input.py`, `tests/test_image_input.py` 작성.

## 열린 질문

- [ ] 오류·경고 코드(`file_too_large`, `unsupported_format`, `long_side_exceeded`, `corrupt_image`, `short_side_below_512`)의 응답 형식과 이름 → API v1(T6)에서 확정
- [ ] 짧은 변 경고를 응답 어디에 실을지 (`AnalyzeResult`에 warnings 필드 없음) → T6·T12
- [ ] PNG/WebP 투명 영역은 RGB로 바꿀 때 알파만 버려 투명 픽셀 아래 RGB 값이 그대로 드러난다. 흰 배경 합성이 필요한지
- [ ] 긴 변 2048px 초과는 실패 처리(`plan.md` §7). 서버에서 리사이즈해 통과시킬지
- [ ] 10/12 회의 안건 9(WebP·20MB)가 `meeting-decisions-1013.md`에는 없다. 회의 기록에 반영 필요

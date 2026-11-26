# inner_tshirt 소분류 정리 안건

- 상태: 부분 완료
- 담당: 추천 담당(#115) + 사용자
- 관련 기능 ID: F-203 (태깅), #115 (추천 이너 슬롯)
- 시작일: 2026-11-27
- 완료 Report: -

## 목표

`inner_tshirt` 소분류를 어떻게 처리할지 팀 결정을 내린다.
결정에 따라 `configs/attributes.yaml`의 `tagging.subcategory.ai_skip`과 `configs/categories.yaml`을 수정한다.

## 맥락

- 추천에서 이너 소분류는 상의 뼈대로 쓰이지 않는다(#115). 즉 AI가 `inner_tshirt`로 판정한 옷은 메인 상의 추천에서 빠진다.
- 레퍼런스 사진과 오늘(11/27) 테스트에서 **슬림핏 반팔 티셔츠가 `inner_tshirt`(0.89)로 판정**되는 사례 확인.
  - 같은 흰 반팔이라도 오버핏은 `tshirt`(0.81), 슬림은 `inner_tshirt`(0.89)로 갈림.
  - 사용자가 이너로 입을 의도 없이 등록한 슬림 티셔츠가 추천에서 빠질 수 있다.
- tshirt ↔ shirt 구분은 현재 프롬프트로 충분히 구분됨(오늘 테스트 확인).
- 관련 기록: `docs/reports/261126-1200-01-handoff-1126.md` §열린 질문, `docs/reports/261126-0300-01-tagging-v1.md` §추가

## 선택지

### A. inner_tshirt 완전 제거
- `configs/categories.yaml`에서 `inner_tshirt` 삭제
- 반팔·긴팔 이너 → 모두 `tshirt`로 흡수
- 나시 이너 → `sleeveless_inner` 유지
- 장점: 단순. 슬림 티셔츠 오분류 없음
- 단점: 얇고 비치는 소재의 레이어드용 반팔을 구분하는 신호가 없어짐

### B. inner_tshirt = 나시·민소매만 (sleeveless_inner와 통합)
- 반팔 이너는 `tshirt`로 → `inner_tshirt` 사실상 불필요
- `sleeveless_inner`가 나시 이너를 커버하므로 `inner_tshirt` 삭제와 결과가 같음
- 사실상 A와 동일

### C. ai_skip에 inner_tshirt 추가 (현상 유지 + AI 판정만 끔)
- `tagging.subcategory.ai_skip: [inner_tshirt]`
- AI는 inner_tshirt를 고르지 않고, 팀원이 수동 등록 화면에서 처리
- 장점: 코드 변경 최소. 추천 로직 그대로
- 단점: 등록 화면에서 사용자/팀원이 직접 처리해야 함

## 진행 메모

- 2026-11-27: `ai_skip: [inner_tshirt]` 적용 (HandoA01). 슬림 반팔 → tshirt 0.997로 판정 확인.

## 열린 질문

> 사람의 확인이 필요한 것. 답이 나오면 답과 날짜를 함께 적는다.

- [x] inner_tshirt AI 판정 → ai_skip으로 결정 (HandoA01, 11/27)
- [ ] 레이어드 추천 방향 확정 시 inner_tshirt 소분류 삭제 여부 재검토 (기획 답변 대기)
- [ ] `sleeveless_inner`는 유지하는지? (나시가 이너 슬롯에 들어가는 케이스)

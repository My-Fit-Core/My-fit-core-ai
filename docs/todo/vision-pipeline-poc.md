# Vision 파이프라인 PoC

- 상태: 진행중
- 담당: AI 파트
- 관련 기능 ID: F-201, F-202, F-203
- 시작일: 2026-11-02
- 완료 Report:

## 목표

사전학습 모델 조합만으로 Flat Lay 사진에서 옷을 분리·누끼·태깅할 수 있는지, 정확도와 처리 시간을 실측한다.

## 완료 조건

- [ ] 검출: Grounding DINO와 YOLO(DeepFashion2) 비교
- [ ] 분할: SAM 2 / MobileSAM에 bbox 프롬프트, 경계 품질 확인
- [ ] 태깅: FashionCLIP zero-shot 카테고리·패턴·추구미 점수, 색상은 k-means
- [ ] 테스트 사진 20~30장 기준 카테고리 정확도, 6벌 처리 시간 측정
- [ ] 결과를 Report로 남기고 모델 선택 결정 기록

## 맥락

- 실행 환경: PoC는 로컬 CPU로 측정하고, 처리 시간 목표 검증은 GPU에서 한다.
- 노트북은 `notebooks/`에 두고 출력은 지우고 커밋.

## 진행 메모

- 2026-11-08: K-Fashion 50장으로 YOLO(DeepFashion2) 검출과 SAM 누끼 비교(`261108-0305-01-detection-segmentation-poc.md`). 기본은 YOLO 마스크만, SAM 2.1 tiny는 Flat Lay에서 비교 후 결정, Grounding DINO 보류.

## 열린 질문

- [ ] 15대 추구미·카테고리 체계 확정 전에는 임시 목록으로 진행해도 되는지
- [ ] 옷이 겹친 사진도 테스트셋에 포함할지

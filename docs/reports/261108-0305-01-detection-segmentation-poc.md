# 검출(YOLO DeepFashion2)과 누끼(SAM) K-Fashion 비교

- 일시: 2026-11-08 03:05
- 작성: AI 파트
- 관련 TODO: `docs/todo/vision-pipeline-poc.md`
- 관련 기능 ID: F-201, F-202
- 관련 issue: #41

## 요약

K-Fashion 착용 사진 50장으로 YOLO(DeepFashion2) 검출과, 검출 박스를 SAM에 준 누끼를 정답 박스를 줬을 때와 비교했다. YOLO는 옷 부위의 93%를 찾았고, 찾은 부위만 보면 검출 박스로 해도 누끼 품질이 거의 떨어지지 않았다. YOLO 자체 마스크도 SAM과 IoU가 비슷하고 SAM의 추가 시간(장당 2~4.6초)이 들지 않아, 기본 파이프라인은 YOLO 마스크만 쓰기로 했다.

## 변경 내역

- `scripts/eval_segmentation_kfashion.py`
  - 대표 IoU(`mean_iou`)의 후처리를 셋 다 적용에서 구멍 메우기만으로 변경(`DEFAULT_POSTPROCESS`). 셋 다 적용하면 후처리 전보다 낮아졌다(v3: 0.749 < 0.779).
  - 기본 모델을 MobileSAM, SAM 2.1 tiny로 변경. s/b/l은 `--models`로만 실행
  - `--detector`: YOLO로 검출을 평가하고, 검출 박스로도 SAM을 실행. YOLO 자체 마스크도 같은 기준으로 평가
- `tests/test_eval_segmentation_kfashion.py`: 박스 IoU, 박스 매칭, 기본 후처리, DeepFashion2 부위 연결 테스트
- `docs/02-specs.md`: 검출·분할 모델 선택 반영
- `docs/todo/vision-pipeline-poc.md`: 상태, 시작일, 실행 환경

## 결과·측정치

조건: K-Fashion Validation에서 옷이 2벌 이상 라벨된 이미지 50장(시드 42), 부위 104개(상의 40, 하의 50, 아우터 14), Windows CPU. 정답은 라벨의 렉트좌표(박스)와 폴리곤좌표(마스크)다. 신발과 잡화는 라벨이 없어 평가하지 않는다. 결과 폴더는 `outputs/segmentation_poc_v4/`.

### 검출 (YOLO, conf 0.25)

검출 박스가 정답 박스와 IoU 0.5 이상이면 찾은 것으로 본다. 부위는 DeepFashion2 13종을 K-Fashion 부위로 연결해 비교한다.

| 지표 | 전체 | 상의 | 하의 | 아우터 |
| --- | ---: | ---: | ---: | ---: |
| 재현율 | 93.3% (97/104) | 93% | 94% | 93% |
| 부위까지 맞힌 비율 | 83.7% | 85% | 94% | **43%** |

- 찾은 박스의 박스 IoU 평균 0.85, 정답과 짝이 안 된 검출은 이미지당 0.24개
- 장당 0.45초(중앙값 0.44, 최대 0.74)
- 아우터는 위치는 찾지만 부위를 상의로 판정하는 경우가 많다.

### 누끼 (마스크 IoU, 구멍 메우기 적용)

| 방식 | 정답 박스 | 검출 박스 | 검출 박스, 찾은 부위만 | 장당 시간 (검출 제외) |
| --- | ---: | ---: | ---: | ---: |
| SAM 2.1 tiny | 0.784 | 0.725 | 0.777 | 4.6초 |
| MobileSAM | 0.739 | 0.713 | 0.764 | 2.2초 |
| YOLO 자체 마스크 | | 0.717 | 0.769 | 0초 (검출에 포함) |

- 검출 박스로 바꿨을 때의 하락은 대부분 놓친 7개(IoU 0으로 계산) 때문이다. 찾은 부위만 보면 SAM 2.1 tiny는 0.784에서 0.777로 거의 같다.
- SAM 2.1 tiny 정답 박스 결과(0.784)는 v3의 구멍 메우기 결과와 같아 재구성 후에도 재현된다.
- 착용 사진 기준이다. 서비스 입력인 Flat Lay 사진에서는 다시 확인해야 한다.

## 주요 결정

| 결정 | 근거 | 대안과 버린 이유 |
| --- | --- | --- |
| 검출은 YOLO(DeepFashion2) | 재현율 93%, 장당 0.45초(CPU) | Grounding DINO: 보류. YOLO 결과가 충분해 Flat Lay 확인 뒤 필요하면 비교 |
| 기본 누끼는 YOLO 자체 마스크 + 구멍 메우기 | 찾은 부위 IoU 0.769로 SAM 2.1 tiny(0.777)와 비슷하고, SAM의 추가 시간이 없다 | SAM 2.1 tiny: Flat Lay 사진에서 경계 품질을 비교한 뒤 결정. MobileSAM: tiny보다 IoU가 낮음 |
| 대표 IoU의 후처리는 구멍 메우기만 | 셋 다 적용하면 후처리 전보다 낮아짐 | 셋 다 적용: 가장 큰 덩어리만 남기기와 경계 다듬기가 IoU를 낮춤 |
| DeepFashion2 `vest`는 상의로 연결 | DeepFashion2 vest는 민소매 상의다 | 아우터: K-Fashion 베스트(아우터)와 이름만 같음 |

### 가중치 출처와 라이선스

- 검출: `deepfashion2_yolov8s-seg.pt`, Hugging Face [`Bingsu/adetailer`](https://huggingface.co/Bingsu/adetailer)에서 받음. `weights/`에 두고 커밋하지 않는다.
- DeepFashion2 데이터셋은 비상업 연구용 라이선스다. 이 가중치로 학습된 모델은 학교 프로젝트에는 쓸 수 있지만 상업 서비스에는 쓰지 않는다.
- SAM 2.1, MobileSAM 가중치는 Ultralytics 배포본(`weights/`)을 썼다.

## 후속 작업

- YOLO 박스로 잘라낸 옷을 FashionCLIP으로 카테고리 판정(아우터/상의 구분) — 다음 실험
- Flat Lay 테스트 사진(`data/segmentation/test/`)으로 YOLO 마스크와 SAM 2.1 tiny 경계 품질 비교
- 놓친 부위가 Flat Lay에서도 많으면 Grounding DINO 비교

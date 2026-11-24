"""YOLO(DeepFashion2) 다중 의류 검출 + 자체 마스크 (F-201, F-202).

근거: docs/reports/261108-0305-01-detection-segmentation-poc.md
- 검출과 누끼 마스크는 YOLO seg 하나로, 후처리는 구멍 메우기만 한다.
- 부위는 DeepFashion2 13종 이름 그대로 돌려준다. 서비스 카테고리 매핑은 T2 결정 후.
- 검출 0벌이면 빈 목록. 오류 코드로 바꾸는 건 호출부(T12) 책임이다.
"""

import functools
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
# 빈 값(.env.example 복사)도 기본 경로로
WEIGHTS_PATH = Path(
    os.getenv("DETECTOR_WEIGHTS") or ROOT / "weights" / "deepfashion2_yolov8s-seg.pt"
)
# ponytail: PoC(K-Fashion) 값. Flat Lay로 튜닝하게 되면(T5, T13) configs/로 옮긴다
CONF = 0.25


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # 픽셀 xyxy. 정규화는 API 경계(T6/T12)에서
    conf: float
    df2_class: str  # DeepFashion2 클래스 이름
    mask: np.ndarray  # bool (H, W), 입력 이미지 크기, 구멍 메운 뒤


@functools.cache
def load_detector(weights: Path = WEIGHTS_PATH):
    """프로세스당 1회만 로드한다. 서버 기동 시 미리 부르는 연결은 T12.

    ultralytics가 MPS(Apple Silicon)·CUDA·CPU를 자동 감지한다 (#37).
    """
    from ultralytics import YOLO

    return YOLO(str(weights))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """바깥 윤곽선 안쪽을 모두 채운다 (PoC postprocess_mask의 fill과 같은 방식)."""
    out = mask.astype("uint8")
    if not out.any():
        return mask.astype(bool)
    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, dtype="uint8")
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled.astype(bool)


def to_detections(
    boxes, confs, class_ids, masks, names, max_items: int
) -> list[Detection]:
    """YOLO 결과 배열을 신뢰도 순 상위 max_items개의 Detection으로 바꾼다."""
    if masks is None:
        return []
    order = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)[:max_items]
    return [
        Detection(
            bbox=tuple(float(v) for v in boxes[i]),
            conf=float(confs[i]),
            df2_class=names[int(class_ids[i])],
            mask=fill_holes(np.asarray(masks[i]).astype(bool)),
        )
        for i in order
    ]


def detect(image: Image.Image, max_items: int = 6) -> list[Detection]:
    """회전 보정된 RGB 이미지(app.pipeline.image_input.validate_image 결과)를 받는다."""
    result = load_detector()(image, conf=CONF, retina_masks=True, verbose=False)[0]
    return to_detections(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
        None if result.masks is None else result.masks.data.cpu().numpy(),
        result.names,
        max_items,
    )

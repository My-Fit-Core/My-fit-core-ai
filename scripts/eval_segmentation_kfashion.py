"""누끼 모델 후보를 K-Fashion Validation 이미지로 비교한다.

라벨의 렉트좌표를 박스 프롬프트로 주고, 폴리곤좌표를 정답 삼아 마스크 IoU와 처리 시간을 잰다.
검출 모델 없이 정답 박스를 쓰므로 검출 시간과 검출 오류는 결과에 포함되지 않는다.
K-Fashion은 모델 착용 사진이라 Flat Lay 조건과 다르다. 결과는 outputs/segmentation_poc/에 저장한다.

    python scripts/eval_segmentation_kfashion.py
"""

import argparse
import csv
import io
import json
import random
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_kfashion_styles import (
    ROOT,
    ZIP_NAME_ENCODING,
    dedupe,
    find_zip,
    load_records,
)

# 기본은 후보 2개. s/b/l은 tiny와 IoU가 비슷하고 느려서(v2 측정) --models로만 돌린다.
MODELS = ["mobile_sam.pt", "sam2.1_t.pt"]
PARTS = ["상의", "하의", "아우터", "원피스"]
SMOOTH_RATIO = 0.005  # 경계 다듬기 크기. 이미지 짧은 변 대비 비율


def rect_to_xyxy(rect):
    x, y = rect["X좌표"], rect["Y좌표"]
    return [x, y, x + rect["가로"], y + rect["세로"]]


def polygon_points(polygon):
    """{X좌표1, Y좌표1, ...}를 번호 순서의 점 목록으로 바꾼다. 짝이 없는 번호는 뺀다."""
    indexes = sorted(
        int(key[3:])
        for key in polygon
        if key.startswith("X좌표") and f"Y좌표{key[3:]}" in polygon
    )
    return [(polygon[f"X좌표{i}"], polygon[f"Y좌표{i}"]) for i in indexes]


def parts_from_label(doc):
    """박스와 3점 이상의 폴리곤이 모두 있는 부위만 돌려준다."""
    detail = doc["데이터셋 정보"]["데이터셋 상세설명"]
    parts = []
    for part in PARTS:
        rect = detail["렉트좌표"].get(part, [{}])[0]
        points = polygon_points(detail["폴리곤좌표"].get(part, [{}])[0])
        if rect and len(points) >= 3:
            category = detail.get("라벨링", {}).get(part, [{}])[0].get("카테고리")
            parts.append(
                {
                    "part": part,
                    "box": rect_to_xyxy(rect),
                    "points": points,
                    "category": category,
                }
            )
    return parts


def mask_iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def box_mask(box, shape):
    import numpy as np

    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = (round(v) for v in box)
    mask[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)] = True
    return mask


def assign_masks(masks, boxes):
    """마스크를 박스에 대응시킨다. 빈 마스크가 빠져 수가 다르면 박스와 많이 겹치는 순서로 맞춘다."""
    if len(masks) == len(boxes):
        return list(masks)
    assigned = [None] * len(boxes)
    pairs = sorted(
        (
            (mask_iou(mask, box_mask(box, mask.shape)), m, b)
            for m, mask in enumerate(masks)
            for b, box in enumerate(boxes)
        ),
        key=lambda pair: -pair[0],
    )
    used = set()
    for _, m, b in pairs:
        if m not in used and assigned[b] is None:
            assigned[b] = masks[m]
            used.add(m)
    return assigned


# 후처리 단계 조합. 어느 단계가 IoU를 바꾸는지 따로 보려고 하나씩도 잰다.
VARIANTS = {
    "largest": {"largest": True},
    "fill": {"fill": True},
    "smooth": {"smooth": True},
    "all": {"largest": True, "fill": True, "smooth": True},
}
# 대표 IoU(mean_iou)에 쓰는 후처리. 셋 다 적용하면 후처리 전보다 낮아져(v3: 0.749 < 0.779) 구멍 메우기만 쓴다.
DEFAULT_POSTPROCESS = "fill"


def postprocess_mask(mask, largest=False, fill=False, smooth=False):
    """가장 큰 덩어리만 남기기, 구멍 메우기, 경계 다듬기를 고른 순서대로 적용한다."""
    import cv2
    import numpy as np

    out = mask.astype("uint8")
    if not out.any():
        return mask
    if largest:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        out = (labels == 1 + stats[1:, cv2.CC_STAT_AREA].argmax()).astype("uint8")
    if fill:
        contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros(mask.shape, dtype="uint8")
        cv2.drawContours(out, contours, -1, 1, cv2.FILLED)
    if smooth:
        size = max(3, round(min(mask.shape) * SMOOTH_RATIO) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        opened = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
        out = opened if opened.any() else out
    return out.astype(bool)


def count_components(mask):
    import cv2

    return cv2.connectedComponents(mask.astype("uint8"), connectivity=8)[0] - 1


def summarize(rows, seconds, post_seconds=()):
    ious = [r["iou"] for r in rows]
    by_part = {}
    for part in PARTS:
        values = [r["iou"] for r in rows if r["part"] == part]
        if values:
            by_part[part] = {"n": len(values), "mean_iou": mean(values)}
    return {
        "parts": len(rows),
        "mean_iou": mean(ious),
        "mean_iou_raw": mean(r["iou_raw"] for r in rows),
        "median_iou": median(ious),
        "iou_under_0.5": sum(i < 0.5 for i in ious) / len(ious),
        "iou_under_0.7": sum(i < 0.7 for i in ious) / len(ious),
        "improved_by_postprocess": sum(r["iou"] > r["iou_raw"] for r in rows),
        "worsened_by_postprocess": sum(r["iou"] < r["iou_raw"] for r in rows),
        "missing_masks": sum(r["missing"] for r in rows),
        "variants": {
            v: {
                "mean_iou": mean(r[f"iou_{v}"] for r in rows),
                "improved": sum(r[f"iou_{v}"] > r["iou_raw"] for r in rows),
                "worsened": sum(r[f"iou_{v}"] < r["iou_raw"] for r in rows),
            }
            for v in VARIANTS
            if f"iou_{v}" in rows[0]
        },
        "mean_postprocess_seconds_per_image": mean(post_seconds)
        if post_seconds
        else 0.0,
        "mean_seconds_per_image": mean(seconds),
        "median_seconds_per_image": median(seconds),
        "max_seconds_per_image": max(seconds),
        "by_part": by_part,
    }


def pick_samples(data_dir, n, seed, min_parts=2, per_style=None):
    """옷이 min_parts벌 이상 라벨된 이미지를 n장 고른다. 내용이 같은 이미지는 평가 기준대로 처리한다.

    per_style을 주면 n 대신 주 스타일마다 per_style장씩 고른다.
    """
    labels_zip = find_zip(data_dir / "labels")
    images_zip = find_zip(data_dir / "images")
    kept, _ = dedupe(load_records(labels_zip, images_zip))
    candidates = sorted((r for r in kept if r.main), key=lambda r: r.image_id)
    random.Random(seed).shuffle(candidates)

    samples = []
    with zipfile.ZipFile(labels_zip, metadata_encoding=ZIP_NAME_ENCODING) as labels:
        members = {
            Path(i.filename).stem: i
            for i in labels.infolist()
            if i.filename.endswith(".json")
        }
        counts = Counter()
        for record in candidates:
            if per_style and counts[record.main] >= per_style:
                continue
            doc = json.loads(labels.read(members[str(record.image_id)]).decode("utf-8"))
            parts = parts_from_label(doc)
            if len(parts) >= min_parts:
                size = (
                    doc["이미지 정보"]["이미지 너비"],
                    doc["이미지 정보"]["이미지 높이"],
                )
                samples.append({"record": record, "parts": parts, "size": size})
                counts[record.main] += 1
                if not per_style and len(samples) == n:
                    break
    return images_zip, samples


# DeepFashion2 13종 → K-Fashion 부위. DeepFashion2 vest는 민소매 상의라 상의로 둔다(K-Fashion 베스트는 아우터).
DF2_TO_PART = {
    "short_sleeved_shirt": "상의",
    "long_sleeved_shirt": "상의",
    "vest": "상의",
    "sling": "상의",
    "short_sleeved_outwear": "아우터",
    "long_sleeved_outwear": "아우터",
    "shorts": "하의",
    "trousers": "하의",
    "skirt": "하의",
    "short_sleeved_dress": "원피스",
    "long_sleeved_dress": "원피스",
    "vest_dress": "원피스",
    "sling_dress": "원피스",
}
MATCH_IOU = 0.5  # 검출 박스를 정답 박스로 인정하는 최소 박스 IoU


def box_iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(truth, preds, threshold=MATCH_IOU):
    """정답 박스마다 겹침이 큰 순서로 예측 박스를 하나씩 짝짓는다. 부위는 보지 않는다."""
    pairs = sorted(
        (
            (box_iou(t, p), i, j)
            for i, t in enumerate(truth)
            for j, p in enumerate(preds)
        ),
        reverse=True,
    )
    matched, used = [None] * len(truth), set()
    for iou, i, j in pairs:
        if iou < threshold:
            break
        if matched[i] is None and j not in used:
            matched[i] = j
            used.add(j)
    return matched


def load_image(archive, sample):
    """이미지를 읽는다. 크기가 라벨과 다르면 좌표를 믿을 수 없으므로 None."""
    from PIL import Image

    image = Image.open(io.BytesIO(archive.read(sample["record"].member))).convert("RGB")
    return image if image.size == sample["size"] else None


def truth_mask(points, size):
    import numpy as np
    from PIL import Image, ImageDraw

    truth = Image.new("1", size)
    ImageDraw.Draw(truth).polygon(points, fill=1)
    return np.array(truth)


def mask_row(model, image_id, part, raw, truth):
    """원본 마스크와 후처리별 마스크의 IoU. 마스크가 없으면(놓침) 0."""
    row = {
        "model": model,
        "image_id": image_id,
        "part": part,
        "iou_raw": 0.0 if raw is None else mask_iou(raw, truth),
        "missing": raw is None,
        "components": 0 if raw is None else count_components(raw),
    }
    for v, steps in VARIANTS.items():
        row[f"iou_{v}"] = (
            0.0 if raw is None else mask_iou(postprocess_mask(raw, **steps), truth)
        )
    row["iou"] = row[f"iou_{DEFAULT_POSTPROCESS}"]
    return row


def detect(weights, images_zip, samples, device, conf):
    """YOLO 검출 결과(박스, 부위, 자체 마스크)와 장당 시간. 크기가 다른 이미지는 None."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    kwargs = {"conf": conf, "device": device, "retina_masks": True, "verbose": False}
    detections, seconds = [], []
    with zipfile.ZipFile(images_zip, metadata_encoding=ZIP_NAME_ENCODING) as archive:
        for sample in samples:
            image = load_image(archive, sample)
            if image is None:
                detections.append(None)
                continue
            if not seconds:  # 첫 호출의 준비 시간은 뺀다.
                model(image, **kwargs)
            start = time.perf_counter()
            result = model(image, **kwargs)[0]
            seconds.append(time.perf_counter() - start)
            detections.append(
                {
                    "boxes": result.boxes.xyxy.cpu().tolist(),
                    "parts": [
                        DF2_TO_PART[result.names[int(c)]]
                        for c in result.boxes.cls.tolist()
                    ],
                    "masks": []
                    if result.masks is None
                    else list(result.masks.data.cpu().numpy().astype(bool)),
                }
            )
    return detections, seconds


def detection_summary(samples, detections, seconds):
    """재현율(박스 IoU 0.5 이상으로 찾은 정답 비율), 부위까지 맞힌 비율, 찾은 박스의 IoU."""
    rows, extra = [], []
    for sample, det in zip(samples, detections):
        if det is None:
            continue
        matched = match_boxes([p["box"] for p in sample["parts"]], det["boxes"])
        for part, j in zip(sample["parts"], matched):
            rows.append(
                {
                    "part": part["part"],
                    "found": j is not None,
                    "part_correct": j is not None and det["parts"][j] == part["part"],
                    "box_iou": 0.0
                    if j is None
                    else box_iou(part["box"], det["boxes"][j]),
                }
            )
        extra.append(len(det["boxes"]) - sum(j is not None for j in matched))
    found = [r for r in rows if r["found"]]
    by_part = {}
    for part in PARTS:
        values = [r for r in rows if r["part"] == part]
        if values:
            by_part[part] = {
                "n": len(values),
                "recall": sum(r["found"] for r in values) / len(values),
                "recall_with_part": sum(r["part_correct"] for r in values)
                / len(values),
            }
    return {
        "parts": len(rows),
        "recall": len(found) / len(rows),
        "recall_with_part": sum(r["part_correct"] for r in rows) / len(rows),
        "mean_box_iou_found": mean(r["box_iou"] for r in found) if found else 0.0,
        "unmatched_detections_per_image": mean(extra),
        "mean_seconds_per_image": mean(seconds),
        "median_seconds_per_image": median(seconds),
        "max_seconds_per_image": max(seconds),
        "by_part": by_part,
    }


def yolo_mask_rows(samples, detections):
    """YOLO 자체 마스크를 정답 폴리곤과 비교한다. SAM 없이도 충분한지 보려는 비교."""
    rows = []
    for sample, det in zip(samples, detections):
        if det is None:
            continue
        matched = match_boxes([p["box"] for p in sample["parts"]], det["boxes"])
        for part, j in zip(sample["parts"], matched):
            raw = None if j is None or not det["masks"] else det["masks"][j]
            truth = truth_mask(part["points"], sample["size"])
            rows.append(
                mask_row(
                    "yolo_mask", sample["record"].image_id, part["part"], raw, truth
                )
            )
    return rows


def run_model(
    name,
    weights_dir,
    images_zip,
    samples,
    device,
    out_dir,
    save_cutouts,
    detections=None,
):
    """SAM에 박스를 주고 정답 폴리곤과 비교한다. detections가 있으면 정답 박스 대신 검출 박스를 쓴다."""
    from PIL import Image
    from ultralytics import SAM

    start = time.perf_counter()
    model = SAM(str(weights_dir / name))
    load_seconds = time.perf_counter() - start

    label = name if detections is None else f"{name}+det"
    rows, seconds, post_seconds, skipped = [], [], [], 0
    cutout_dir = out_dir / "cutouts" / label.replace(".pt", "")
    with zipfile.ZipFile(images_zip, metadata_encoding=ZIP_NAME_ENCODING) as archive:
        for index, sample in enumerate(samples):
            record = sample["record"]
            image = load_image(archive, sample)
            if image is None:
                skipped += 1
                continue
            truth_boxes = [p["box"] for p in sample["parts"]]
            if detections is None:
                boxes, matched = truth_boxes, list(range(len(truth_boxes)))
            else:
                boxes = detections[index]["boxes"]
                matched = match_boxes(truth_boxes, boxes)

            masks = []
            if boxes:
                if not seconds:  # 첫 호출의 준비 시간은 워밍된 추론 시간에서 뺀다.
                    model(image, bboxes=boxes, device=device, verbose=False)
                start = time.perf_counter()
                result = model(image, bboxes=boxes, device=device, verbose=False)[0]
                seconds.append(time.perf_counter() - start)
                if result.masks is not None:
                    masks = list(result.masks.data.cpu().numpy().astype(bool))
            assigned = assign_masks(masks, boxes) if boxes else []
            start = time.perf_counter()
            for m in assigned:
                if m is not None:
                    postprocess_mask(m, **VARIANTS[DEFAULT_POSTPROCESS])
            post_seconds.append(time.perf_counter() - start)

            for part, j in zip(sample["parts"], matched):
                raw = None if j is None else assigned[j]
                truth = truth_mask(part["points"], image.size)
                rows.append(mask_row(label, record.image_id, part["part"], raw, truth))
                if raw is not None and index < save_cutouts:
                    cutout_dir.mkdir(parents=True, exist_ok=True)
                    stem = f"{record.image_id}_{part['part']}"
                    mask = postprocess_mask(raw, **VARIANTS[DEFAULT_POSTPROCESS])
                    for suffix, alpha in (("", mask), ("_raw", raw)):
                        cutout = image.copy()
                        cutout.putalpha(Image.fromarray(alpha.astype("uint8") * 255))
                        cutout.save(cutout_dir / f"{stem}{suffix}.png")

    summary = summarize(rows, seconds, post_seconds)
    summary["load_seconds"] = load_seconds
    summary["skipped_images"] = skipped
    summary["parameters"] = sum(p.numel() for p in model.model.parameters())
    return rows, summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data/kfashion/validation"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/segmentation_poc"
    )
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--images", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-cutouts", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument(
        "--detector",
        type=Path,
        help="YOLO 가중치. 주면 검출을 평가하고, 검출 박스로도 SAM을 돌린다.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    import torch

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device={device}")
    images_zip, samples = pick_samples(args.data_dir, args.images, args.seed)
    args.weights_dir.mkdir(exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows, summaries, detections = [], {}, None
    payload = {"device": device, "seed": args.seed, "images": len(samples)}
    if args.detector:
        detections, det_seconds = detect(
            args.detector, images_zip, samples, device, args.conf
        )
        payload["detector"] = {
            "weights": args.detector.name,
            "conf": args.conf,
            "match_iou": MATCH_IOU,
            **detection_summary(samples, detections, det_seconds),
        }
        print("detector", json.dumps(payload["detector"], ensure_ascii=False))
        rows = yolo_mask_rows(samples, detections)
        summaries["yolo_mask"] = summarize(rows, det_seconds)
        all_rows += rows
        print("yolo_mask", json.dumps(summaries["yolo_mask"], ensure_ascii=False))

    for name in args.models:
        for source in [None] if detections is None else [None, detections]:
            rows, summary = run_model(
                name,
                args.weights_dir,
                images_zip,
                samples,
                device,
                args.output_dir,
                args.save_cutouts,
                source,
            )
            summaries[rows[0]["model"]] = summary
            all_rows += rows
            print(rows[0]["model"], json.dumps(summary, ensure_ascii=False))
    payload["models"] = summaries
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with open(
        args.output_dir / "parts.csv", "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"결과: {args.output_dir}")


if __name__ == "__main__":
    main()

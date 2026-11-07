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

MODELS = ["sam2.1_t.pt", "sam2.1_s.pt", "sam2.1_b.pt", "sam2.1_l.pt"]
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
            parts.append({"part": part, "box": rect_to_xyxy(rect), "points": points})
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


def pick_samples(data_dir, n, seed):
    """옷이 2벌 이상 라벨된 이미지를 n장 고른다. 내용이 같은 이미지는 평가 기준대로 처리한다."""
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
        for record in candidates:
            doc = json.loads(labels.read(members[str(record.image_id)]).decode("utf-8"))
            parts = parts_from_label(doc)
            if len(parts) >= 2:
                size = (
                    doc["이미지 정보"]["이미지 너비"],
                    doc["이미지 정보"]["이미지 높이"],
                )
                samples.append({"record": record, "parts": parts, "size": size})
                if len(samples) == n:
                    break
    return images_zip, samples


def run_model(name, weights_dir, images_zip, samples, device, out_dir, save_cutouts):
    import numpy as np
    from PIL import Image, ImageDraw
    from ultralytics import SAM

    start = time.perf_counter()
    model = SAM(str(weights_dir / name))
    load_seconds = time.perf_counter() - start

    rows, seconds, post_seconds, skipped = [], [], [], 0
    cutout_dir = out_dir / "cutouts" / Path(name).stem
    with zipfile.ZipFile(images_zip, metadata_encoding=ZIP_NAME_ENCODING) as archive:
        for index, sample in enumerate(samples):
            record = sample["record"]
            image = Image.open(io.BytesIO(archive.read(record.member))).convert("RGB")
            if image.size != sample["size"]:
                skipped += 1
                continue
            boxes = [p["box"] for p in sample["parts"]]
            if index == 0:  # 첫 호출의 준비 시간은 워밍된 추론 시간에서 뺀다.
                model(image, bboxes=boxes, device=device, verbose=False)
            start = time.perf_counter()
            result = model(image, bboxes=boxes, device=device, verbose=False)[0]
            seconds.append(time.perf_counter() - start)

            masks = (
                []
                if result.masks is None
                else result.masks.data.cpu().numpy().astype(bool)
            )
            assigned = assign_masks(list(masks), boxes)
            start = time.perf_counter()
            cleaned = [
                None if m is None else postprocess_mask(m, **VARIANTS["all"])
                for m in assigned
            ]
            post_seconds.append(time.perf_counter() - start)
            for part, raw, mask in zip(sample["parts"], assigned, cleaned):
                truth = Image.new("1", image.size)
                ImageDraw.Draw(truth).polygon(part["points"], fill=1)
                truth = np.array(truth)
                row = {
                    "model": name,
                    "image_id": record.image_id,
                    "part": part["part"],
                    "iou_raw": 0.0 if raw is None else mask_iou(raw, truth),
                    "iou": 0.0 if mask is None else mask_iou(mask, truth),
                    "missing": mask is None,
                    "components": 0 if raw is None else count_components(raw),
                }
                for v, steps in VARIANTS.items():
                    row[f"iou_{v}"] = (
                        0.0
                        if raw is None
                        else mask_iou(postprocess_mask(raw, **steps), truth)
                    )
                rows.append(row)
                if mask is not None and index < save_cutouts:
                    cutout_dir.mkdir(parents=True, exist_ok=True)
                    stem = f"{record.image_id}_{part['part']}"
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

    all_rows, summaries = [], {}
    for name in args.models:
        rows, summaries[name] = run_model(
            name,
            args.weights_dir,
            images_zip,
            samples,
            device,
            args.output_dir,
            args.save_cutouts,
        )
        all_rows += rows
        print(name, json.dumps(summaries[name], ensure_ascii=False))

    payload = {
        "device": device,
        "seed": args.seed,
        "images": len(samples),
        "models": summaries,
    }
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

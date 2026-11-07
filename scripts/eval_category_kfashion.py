"""YOLO가 찾은 옷을 잘라 FashionCLIP zero-shot으로 부위와 카테고리를 판정하고, K-Fashion 라벨과 비교한다.

검출·누끼 평가(eval_segmentation_kfashion.py)와 같은 50장을 쓴다. 잘라내기는 두 가지를 비교한다.
- box: 검출 박스로 자른 사진(배경과 안에 입은 옷이 함께 보임)
- masked: YOLO 마스크 밖을 흰색으로 지운 뒤 박스로 자른 사진(서비스 누끼 입력에 가까움)

부위 판정도 두 가지를 비교한다.
- part4: 부위 4개 문장(상의/하의/아우터/원피스) 중 가장 가까운 것
- from_category: 카테고리 21개 중 가장 가까운 것의 부위

정답 박스와 짝이 안 된(검출이 놓친) 부위는 판정하지 않고 수만 센다.

    python scripts/eval_category_kfashion.py
"""

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_kfashion_styles import MODEL_ID, ROOT, ZIP_NAME_ENCODING
from scripts.eval_segmentation_kfashion import (
    PARTS,
    detect,
    load_image,
    match_boxes,
    pick_samples,
)

# K-Fashion 카테고리 21종(docs/04-kfashion-validation.md)과 프롬프트에 넣을 영어 표현. 작성자가 고른 표현이다.
CATEGORIES = {
    "상의": {
        "티셔츠": "a t-shirt",
        "블라우스": "a blouse",
        "니트웨어": "a knit sweater",
        "셔츠": "a button-up shirt",
        "탑": "a sleeveless top",
        "후드티": "a hoodie",
        "브라탑": "a bra top",
    },
    "하의": {
        "팬츠": "trousers",
        "청바지": "jeans",
        "스커트": "a skirt",
        "조거팬츠": "jogger pants",
        "래깅스": "leggings",
    },
    "아우터": {
        "재킷": "a jacket",
        "가디건": "a cardigan",
        "코트": "a coat",
        "베스트": "a sleeveless vest worn over a top",
        "점퍼": "a casual zip-up jumper jacket",
        "패딩": "a padded puffer jacket",
        "짚업": "a zip-up sweatshirt jacket",
    },
    "원피스": {
        "드레스": "a dress",
        "점프수트": "a jumpsuit",
    },
}
PART_PROMPTS = {
    "상의": "a top such as a shirt, blouse or t-shirt",
    "하의": "bottoms such as pants, jeans or a skirt",
    "아우터": "outerwear such as a jacket or coat",
    "원피스": "a one-piece dress or jumpsuit",
}
TEMPLATE = "a photo of {}"
CROPS = ["box", "masked"]
CATEGORY_PART = {c: part for part, cats in CATEGORIES.items() for c in cats}


def crop(image, box, mask=None):
    """박스로 자른다. 마스크가 있으면 마스크 밖을 흰색으로 지운 뒤 자른다."""
    from PIL import Image

    if mask is not None:
        white = Image.new("RGB", image.size, "white")
        image = Image.composite(
            image, white, Image.fromarray(mask.astype("uint8") * 255)
        )
    w, h = image.size
    x1, y1, x2, y2 = box
    return image.crop(
        (max(int(x1), 0), max(int(y1), 0), min(int(x2), w), min(int(y2), h))
    )


def summarize(rows, missed):
    """부위 정확도(YOLO, part4, from_category)와 카테고리 정확도를 정답 부위별로 낸다."""

    def rate(values, key, truth="part"):
        return sum(r[key] == r[truth] for r in values) / len(values) if values else 0.0

    groups = {"전체": rows, **{p: [r for r in rows if r["part"] == p] for p in PARTS}}
    result = {"judged": len(rows), "missed_by_detector": missed, "by_group": {}}
    for name, values in groups.items():
        if not values:
            continue
        g = {"n": len(values), "part_yolo": rate(values, "yolo_part")}
        for c in CROPS:
            g[f"part4_{c}"] = rate(values, f"part4_{c}")
            g[f"part_from_category_{c}"] = rate(values, f"part_from_category_{c}")
            g[f"category_{c}"] = rate(values, f"category_{c}", "category")
            g[f"category_top3_{c}"] = sum(
                r["category"] in r[f"top3_{c}"] for r in values
            ) / len(values)
        result["by_group"][name] = g
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data/kfashion/validation"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/category_poc"
    )
    parser.add_argument(
        "--detector", type=Path, default=ROOT / "weights/deepfashion2_yolov8s-seg.pt"
    )
    parser.add_argument("--images", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    images_zip, samples = pick_samples(args.data_dir, args.images, args.seed)
    detections, _ = detect(args.detector, images_zip, samples, device, args.conf)

    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    categories = list(CATEGORY_PART)
    texts = [TEMPLATE.format(CATEGORIES[CATEGORY_PART[c]][c]) for c in categories]
    texts += [TEMPLATE.format(PART_PROMPTS[p]) for p in PARTS]
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        text = model.get_text_features(**inputs).pooler_output  # 투영된 특징
        text = text / text.norm(dim=-1, keepdim=True)

    def scores(image):
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            feat = model.get_image_features(**inputs).pooler_output
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return (feat @ text.T)[0].cpu().tolist()

    rows, missed = [], 0
    with zipfile.ZipFile(images_zip, metadata_encoding=ZIP_NAME_ENCODING) as archive:
        for sample, det in zip(samples, detections):
            if det is None:
                continue
            image = load_image(archive, sample)
            matched = match_boxes([p["box"] for p in sample["parts"]], det["boxes"])
            for part, j in zip(sample["parts"], matched):
                if j is None:
                    missed += 1
                    continue
                row = {
                    "image_id": sample["record"].image_id,
                    "part": part["part"],
                    "category": part["category"],
                    "yolo_part": det["parts"][j],
                }
                mask = det["masks"][j] if det["masks"] else None
                for c, m in (("box", None), ("masked", mask)):
                    s = scores(crop(image, det["boxes"][j], m))
                    cat_scores, part_scores = s[: len(categories)], s[len(categories) :]
                    ranked = sorted(
                        range(len(categories)), key=lambda i: -cat_scores[i]
                    )
                    row[f"category_{c}"] = categories[ranked[0]]
                    row[f"top3_{c}"] = [categories[i] for i in ranked[:3]]
                    row[f"part_from_category_{c}"] = CATEGORY_PART[
                        categories[ranked[0]]
                    ]
                    row[f"part4_{c}"] = PARTS[part_scores.index(max(part_scores))]
                rows.append(row)

    result = summarize(rows, missed)
    result["run"] = {
        "model": MODEL_ID,
        "detector": args.detector.name,
        "images": len(samples),
        "seed": args.seed,
        "conf": args.conf,
        "template": TEMPLATE,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with open(
        args.output_dir / "parts.csv", "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"판정 {result['judged']}개, 검출이 놓친 부위 {missed}개")
    for name, g in result["by_group"].items():
        print(
            name, json.dumps({k: round(v, 3) for k, v in g.items()}, ensure_ascii=False)
        )
    print(f"결과: {args.output_dir}")


if __name__ == "__main__":
    main()

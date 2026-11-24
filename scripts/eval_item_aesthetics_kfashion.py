"""K-Fashion 사진에서 YOLO 마스크로 잘라낸 옷 한 벌마다 추구미 점수를 내고, 코디용(outfit)과 옷용(item) 프롬프트를 비교한다.

정답은 원본 사진의 주 스타일이다. 추구미 i의 연결 스타일(configs/aesthetics.yaml의 kfashion.styles)에 속한
사진에서 나온 옷이면 i가 맞는 답이다. 옷 자르기는 카테고리 PoC와 같다(검출된 옷 중 정답 박스와 짝지어진 것,
마스크 밖을 흰색으로 지운 사진). 옷 종류 쏠림을 보려고 K-Fashion 카테고리별 1위 추구미 분포도 낸다.

    python scripts/eval_item_aesthetics_kfashion.py --seed 42
"""

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_aesthetics_on_kfashion import (
    argmax,
    calibrate,
    mapped_hit_rate,
    softmax,
    summarize,
    top1_counts,
)
from scripts.eval_category_kfashion import crop
from scripts.eval_kfashion_styles import MODEL_ID, ROOT, ZIP_NAME_ENCODING
from scripts.eval_segmentation_kfashion import (
    detect,
    load_image,
    match_boxes,
    pick_samples,
)

PROMPT_SETS = ["outfit", "item"]


def style_logits(similarities, prompt_ids, scale):
    """프롬프트별 코사인 유사도를 추구미별 평균 logit으로 묶는다(eval_kfashion_styles.predict와 같은 방식)."""
    return [scale * sum(similarities[i] for i in ids) / len(ids) for ids in prompt_ids]


def top1_by_category(items, rows, n_classes):
    """K-Fashion 카테고리별로 1위 추구미 횟수를 센다. 카테고리가 비어 있으면 '(없음)'."""
    table = {}
    for item, row in zip(items, rows):
        counts = table.setdefault(item["category"] or "(없음)", [0] * n_classes)
        counts[argmax(row)] += 1
    return dict(sorted(table.items(), key=lambda kv: -sum(kv[1])))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data/kfashion/validation"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--detector", type=Path, default=ROOT / "weights/deepfashion2_yolov8s-seg.pt"
    )
    parser.add_argument("--per-style", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    output_dir = args.output_dir or ROOT / f"outputs/item_aesthetics/seed{args.seed}"

    import torch
    from transformers import CLIPModel, CLIPProcessor

    styles = [
        s["name"]
        for s in yaml.safe_load(
            (ROOT / "configs/kfashion_styles.yaml").read_text(encoding="utf-8")
        )["styles"]
    ]
    aesthetics = yaml.safe_load(
        (ROOT / "configs/aesthetics.yaml").read_text(encoding="utf-8")
    )["aesthetics"]
    names = [a["name"] for a in aesthetics]
    mapped = [set(a["kfashion"]["styles"]) for a in aesthetics]
    n = len(names)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    images_zip, samples = pick_samples(
        args.data_dir, None, args.seed, min_parts=1, per_style=args.per_style
    )
    detections, _ = detect(args.detector, images_zip, samples, device, args.conf)

    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    texts, prompt_ids = [], {}
    for ps in PROMPT_SETS:
        prompt_ids[ps] = []
        for a in aesthetics:
            prompts = a["prompts"][ps]
            prompt_ids[ps].append(list(range(len(texts), len(texts) + len(prompts))))
            texts += prompts
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        text = model.get_text_features(**inputs).pooler_output
        text = text / text.norm(dim=-1, keepdim=True)
        scale = model.logit_scale.exp().item()

    items, logits = [], {ps: [] for ps in PROMPT_SETS}
    missed = 0
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
                mask = det["masks"][j] if det["masks"] else None
                with torch.no_grad():
                    pixels = processor(
                        images=crop(image, det["boxes"][j], mask), return_tensors="pt"
                    ).to(device)
                    feat = model.get_image_features(**pixels).pooler_output
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    sims = (feat @ text.T)[0].cpu().tolist()
                items.append(
                    {
                        "record": sample["record"],
                        "part": part["part"],
                        "category": part["category"],
                    }
                )
                for ps in PROMPT_SETS:
                    logits[ps].append(style_logits(sims, prompt_ids[ps], scale))
            print(f"옷 {len(items)}벌", end="\r")

    records = [it["record"] for it in items]
    report = {
        "run": {
            "model": MODEL_ID,
            "detector": args.detector.name,
            "seed": args.seed,
            "per_style": args.per_style,
            "images": len(samples),
            "items": len(items),
            "missed_by_detector": missed,
        },
        "aesthetics": names,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for ps in PROMPT_SETS:
        rows = [softmax(r) for r in logits[ps]]
        result = summarize(records, rows, styles, n)
        calibrated = calibrate(rows, result["overall"])
        result.update(
            {
                "top1": top1_counts(rows, n),
                "top1_calibrated": top1_counts(calibrated, n),
                "hit_rate": mapped_hit_rate(records, rows, mapped),
                "hit_rate_calibrated": mapped_hit_rate(records, calibrated, mapped),
                "top1_by_category": top1_by_category(items, rows, n),
                "top1_by_category_calibrated": top1_by_category(items, calibrated, n),
            }
        )
        report[ps] = result
        with open(
            output_dir / f"items_{ps}.csv", "w", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                ["image_id", "style", "part", "category", *names]
                + [f"{x}_보정" for x in names]
            )
            for it, row, cal in zip(items, rows, calibrated):
                writer.writerow(
                    [
                        it["record"].image_id,
                        it["record"].main,
                        it["part"],
                        it["category"],
                    ]
                    + [f"{s:.4f}" for s in row + cal]
                )

        hr = [x for x in result["hit_rate"] if x is not None]
        hc = [x for x in result["hit_rate_calibrated"] if x is not None]
        print(f"\n[{ps}] 옷 {len(items)}벌 (놓침 {missed})\t" + "\t".join(names))
        print("전체 평균\t" + "\t".join(f"{x:.3f}" for x in result["overall"]))
        print("1위 횟수\t" + "\t".join(map(str, result["top1"])))
        print("1위 횟수(보정)\t" + "\t".join(map(str, result["top1_calibrated"])))
        print(
            "적중률\t"
            + "\t".join(f"{x:.2f}" for x in result["hit_rate"])
            + f"\t평균 {sum(hr) / len(hr):.3f}"
        )
        print(
            "적중률(보정)\t"
            + "\t".join(f"{x:.2f}" for x in result["hit_rate_calibrated"])
            + f"\t평균 {sum(hc) / len(hc):.3f}"
        )

    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n결과: {output_dir}")


if __name__ == "__main__":
    main()

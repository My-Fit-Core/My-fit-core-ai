"""같은 옷 종류에서 색(밝기·채도)에 따라 추구미가 갈리는지 본다(issue #50).

issue #35 측정과 같은 표본(23종 × 20장)에서 YOLO 마스크 안 픽셀의 LAB 밝기(L, 0~100)와 채도(C = sqrt(a² + b²))
평균을 낸다. 카테고리마다 밝기 3분위(연/중/진)로 나눠 두 분포를 비교한다.

- (a) FashionCLIP item 1위: issue #35 결과(items_item.csv, 보정 전)를 이미지 식별자·부위로 이어 붙인다.
- (b) 정답 라벨: 원본 사진의 주 스타일을 기본 스타일 연결(kfashion.styles)로 바꾼 것. 연결이 없으면 "연결 없음".

분위별 분포가 갈리는지는 카이제곱 독립성 검정으로 본다. 칸당 수가 적어 참고용이다.

    python scripts/eval_color_aesthetics_kfashion.py --seed 42
"""

import argparse
import csv
import json
import statistics
import sys
import zipfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_kfashion_styles import ROOT, ZIP_NAME_ENCODING
from scripts.eval_segmentation_kfashion import (
    detect,
    load_image,
    match_boxes,
    pick_samples,
)

CATEGORIES = ["청바지", "티셔츠"]
BANDS = ["연", "중", "진"]
UNMAPPED = "연결 없음"


def lab_stats(image, mask):
    """마스크 안 픽셀의 LAB 밝기와 채도 평균."""
    import cv2
    import numpy as np

    rgb = np.asarray(image, dtype="float32") / 255
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[mask]
    return float(lab[:, 0].mean()), float(np.hypot(lab[:, 1], lab[:, 2]).mean())


def bands_by_lightness(values):
    """밝기 3분위. 밝은 1/3이 '연', 어두운 1/3이 '진'. 경계값은 [진|중, 중|연]."""
    low, high = statistics.quantiles(values, n=3)
    return ["연" if v > high else "진" if v <= low else "중" for v in values], [
        low,
        high,
    ]


def crosstab(bands, labels, columns):
    return {
        b: [sum(x == b and y == c for x, y in zip(bands, labels)) for c in columns]
        for b in BANDS
    }


def chi2_p(table):
    """카이제곱 독립성 검정 p값. 합이 0인 열은 뺀다."""
    from scipy.stats import chi2_contingency

    rows = list(table.values())
    keep = [j for j in range(len(rows[0])) if sum(r[j] for r in rows)]
    return float(chi2_contingency([[r[j] for j in keep] for r in rows]).pvalue)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data/kfashion/validation"
    )
    parser.add_argument(
        "--detector", type=Path, default=ROOT / "weights/deepfashion2_yolov8s-seg.pt"
    )
    parser.add_argument("--per-style", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--scores-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    scores_dir = args.scores_dir or ROOT / f"outputs/item_aesthetics/seed{args.seed}"
    output_dir = args.output_dir or ROOT / f"outputs/color_aesthetics/seed{args.seed}"

    aesthetics = yaml.safe_load(
        (ROOT / "configs/aesthetics.yaml").read_text(encoding="utf-8")
    )["aesthetics"]
    names = [a["name"] for a in aesthetics]
    style_to_aesthetic = {
        s: a["name"] for a in aesthetics for s in a["kfashion"]["styles"]
    }

    with open(scores_dir / "items_item.csv", encoding="utf-8-sig") as f:
        top1 = {
            (int(r["image_id"]), r["part"]): max(names, key=lambda x: float(r[x]))
            for r in csv.DictReader(f)
        }

    images_zip, samples = pick_samples(
        args.data_dir, None, args.seed, min_parts=1, per_style=args.per_style
    )
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detections, _ = detect(args.detector, images_zip, samples, device, args.conf)

    items = []
    with zipfile.ZipFile(images_zip, metadata_encoding=ZIP_NAME_ENCODING) as archive:
        for sample, det in zip(samples, detections):
            if det is None or not det["masks"]:
                continue
            wanted = [p for p in sample["parts"] if p["category"] in CATEGORIES]
            if not wanted:
                continue
            image = load_image(archive, sample)
            matched = match_boxes([p["box"] for p in sample["parts"]], det["boxes"])
            for part, j in zip(sample["parts"], matched):
                if j is None or part not in wanted:
                    continue
                key = (sample["record"].image_id, part["part"])
                if key not in top1:  # issue #35 결과와 같은 옷만 쓴다
                    continue
                lightness, chroma = lab_stats(image, det["masks"][j])
                items.append(
                    {
                        "image_id": key[0],
                        "style": sample["record"].main,
                        "category": part["category"],
                        "L": lightness,
                        "C": chroma,
                        "clip_top1": top1[key],
                        "label": style_to_aesthetic.get(
                            sample["record"].main, UNMAPPED
                        ),
                    }
                )

    report = {"seed": args.seed, "categories": {}}
    for category in CATEGORIES:
        group = [it for it in items if it["category"] == category]
        bands, cuts = bands_by_lightness([it["L"] for it in group])
        for it, b in zip(group, bands):
            it["band"] = b
        clip = crosstab(bands, [it["clip_top1"] for it in group], names)
        label = crosstab(bands, [it["label"] for it in group], [*names, UNMAPPED])
        report["categories"][category] = {
            "n": len(group),
            "lightness_cuts": cuts,
            "mean_L": {
                b: statistics.mean(it["L"] for it in group if it["band"] == b)
                for b in BANDS
            },
            "mean_C": {
                b: statistics.mean(it["C"] for it in group if it["band"] == b)
                for b in BANDS
            },
            "clip_top1": clip,
            "label": label,
            "chi2_p_clip": chi2_p(clip),
            "chi2_p_label": chi2_p(label),
        }
        r = report["categories"][category]
        print(f"\n[{category}] n={len(group)}, 밝기 경계 {cuts[0]:.1f} / {cuts[1]:.1f}")
        print(
            "분위\tn\tL\tC\t| CLIP 1위: "
            + " ".join(names)
            + "\t| 정답: "
            + " ".join([*names, UNMAPPED])
        )
        for b in BANDS:
            print(
                f"{b}\t{sum(clip[b])}\t{r['mean_L'][b]:.1f}\t{r['mean_C'][b]:.1f}\t| "
                + " ".join(map(str, clip[b]))
                + "\t| "
                + " ".join(map(str, label[b]))
            )
        print(f"카이제곱 p: CLIP {r['chi2_p_clip']:.3f}, 정답 {r['chi2_p_label']:.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with open(output_dir / "items.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)
    print(f"\n결과: {output_dir}")


if __name__ == "__main__":
    main()

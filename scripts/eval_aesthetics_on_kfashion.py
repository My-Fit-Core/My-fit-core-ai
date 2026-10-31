"""K-Fashion 스타일 평가와 같은 표본을 추구미 프롬프트로 점수 내어 스타일별 평균을 구한다.

추구미 5개를 K-Fashion 스타일과 연결할 근거를 만든다. 점수는 이미지마다 합이 1인 상대값이다.
"차이"는 스타일 평균에서 전체 평균을 뺀 값으로, 특정 추구미로 예측이 쏠리는 영향을 덜어낸다.

    python scripts/eval_aesthetics_on_kfashion.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_kfashion_styles import (
    ROOT,
    dedupe,
    find_zip,
    load_records,
    predict,
    sample_per_style,
)


def summarize(records, score_rows, styles, n_classes):
    """스타일별 평균 점수, 전체 평균, 평균과의 차이, 1위 횟수를 계산한다."""
    overall = [
        sum(row[c] for row in score_rows) / len(score_rows) for c in range(n_classes)
    ]
    by_style = {}
    for style in styles:
        rows = [row for r, row in zip(records, score_rows) if r.main == style]
        mean = [sum(row[c] for row in rows) / len(rows) for c in range(n_classes)]
        top = [0] * n_classes
        for row in rows:
            top[row.index(max(row))] += 1
        by_style[style] = {
            "n": len(rows),
            "mean": mean,
            "lift": [m - o for m, o in zip(mean, overall)],
            "top": top,
        }
    return {"overall": overall, "by_style": by_style}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data/kfashion/validation"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/aesthetics_on_kfashion"
    )
    parser.add_argument("--prompt-set", default="outfit")
    parser.add_argument("--per-style", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    style_config = yaml.safe_load(
        (ROOT / "configs/kfashion_styles.yaml").read_text(encoding="utf-8")
    )["styles"]
    styles = [s["name"] for s in style_config]
    aesthetics = yaml.safe_load(
        (ROOT / "configs/aesthetics.yaml").read_text(encoding="utf-8")
    )["aesthetics"]
    config = [
        {"key": a["key"], "name": a["name"], "prompts": a["prompts"][args.prompt_set]}
        for a in aesthetics
    ]
    names = [c["name"] for c in config]

    images_zip = find_zip(args.data_dir / "images")
    kept, _ = dedupe(load_records(find_zip(args.data_dir / "labels"), images_zip))
    sampled = sample_per_style(kept, styles, args.per_style, args.seed)
    score_rows = predict(sampled, images_zip, config, args.batch_size)
    result = summarize(sampled, score_rows, styles, len(names))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "aesthetics": names,
        "seed": args.seed,
        "per_style": args.per_style,
        **result,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with open(
        args.output_dir / "scores.csv", "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "main", "sub", *names])
        for r, row in zip(sampled, score_rows):
            writer.writerow(
                [r.image_id, r.main, r.sub or "", *(f"{s:.4f}" for s in row)]
            )

    print("\n스타일별 평균 점수\n스타일\t" + "\t".join(names))
    for style in styles:
        print(
            style
            + "\t"
            + "\t".join(f"{m:.3f}" for m in result["by_style"][style]["mean"])
        )
    print("전체\t" + "\t".join(f"{m:.3f}" for m in result["overall"]))
    print(f"\n결과: {args.output_dir}")


if __name__ == "__main__":
    main()

"""K-Fashion 스타일 평가와 같은 방식의 표본을 추구미 프롬프트로 점수 내어 스타일별 평균을 구한다.

추구미 5개를 K-Fashion 스타일과 연결할 근거를 만든다. 점수는 이미지마다 합이 1인 상대값이다.
"차이"는 스타일 평균에서 전체 평균을 뺀 값으로, 특정 추구미로 예측이 쏠리는 영향을 덜어낸다.
"보정 점수"는 점수를 전체 평균으로 나눠 다시 합 1로 만든 값이다(issue #40).
반대 개념 축(configs/aesthetics.yaml의 axes)도 같은 표본으로 점수를 낸다(issue #42).

    python scripts/eval_aesthetics_on_kfashion.py [--prompt-sets outfit] [--seed 7]
"""

import argparse
import csv
import json
import math
import statistics
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


def softmax(row):
    top = max(row)
    exps = [math.exp(x - top) for x in row]
    return [e / sum(exps) for e in exps]


def calibrate(rows, prior):
    """점수를 전체 평균(prior)으로 나눠 다시 합 1로 만든다. 늘 높게 나오는 항목의 쏠림을 덜어낸다."""
    out = []
    for row in rows:
        scaled = [p / q for p, q in zip(row, prior)]
        out.append([x / sum(scaled) for x in scaled])
    return out


def argmax(row):
    return row.index(max(row))


def top1_counts(rows, n_classes):
    counts = [0] * n_classes
    for row in rows:
        counts[argmax(row)] += 1
    return counts


def mapped_hit_rate(records, rows, mapped):
    """연결 스타일 적중률. 주 스타일이 추구미 i에 연결된 이미지 중 1위가 i인 비율.

    연결 스타일이 없는 추구미는 None(계산 제외).
    """
    rates = []
    for i, styles in enumerate(mapped):
        if not styles:
            rates.append(None)
            continue
        picked = [row for r, row in zip(records, rows) if r.main in styles]
        rates.append(sum(argmax(row) == i for row in picked) / len(picked))
    return rates


# 혼동을 따로 보는 추구미 쌍(key). 무지·기본템 쪽이 서로 겹칠 수 있다(issue #44).
CONFUSION_PAIRS = [("casual", "street"), ("casual", "minimal")]


def pair_confusion(records, rows, mapped, a, b):
    """추구미 a와 b의 혼동.

    - a_top1_on_b_styles: b에 연결된 스타일 사진 중 1위가 a인 비율(b에 연결이 없으면 None)
    - b_top1_on_a_styles: 위의 반대
    - top2_is_other: 1위가 a나 b인 사진 중 2위가 나머지 하나인 비율
    """

    def top1_share(styles, target):
        picked = [row for r, row in zip(records, rows) if r.main in styles]
        if not picked:
            return None
        return sum(argmax(row) == target for row in picked) / len(picked)

    either = [row for row in rows if argmax(row) in (a, b)]
    other = {a: b, b: a}
    second = [sorted(range(len(row)), key=lambda c: -row[c])[1] for row in either]
    return {
        "a_top1_on_b_styles": top1_share(mapped[b], a),
        "b_top1_on_a_styles": top1_share(mapped[a], b),
        "top1_either": len(either),
        "top2_is_other": sum(s == other[argmax(row)] for s, row in zip(second, either))
        / len(either)
        if either
        else None,
    }


def fmt(x):
    return "-" if x is None else f"{x:.2f}"


def axis_value(logits):
    """[low, high] logit에서 high 쪽 확률."""
    return 1 / (1 + math.exp(logits[0] - logits[1]))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data/kfashion/validation"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt-sets", nargs="+", default=["outfit"])
    parser.add_argument("--per-style", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    output_dir = (
        args.output_dir or ROOT / f"outputs/aesthetics_on_kfashion/seed{args.seed}"
    )

    styles = [
        s["name"]
        for s in yaml.safe_load(
            (ROOT / "configs/kfashion_styles.yaml").read_text(encoding="utf-8")
        )["styles"]
    ]
    config = yaml.safe_load(
        (ROOT / "configs/aesthetics.yaml").read_text(encoding="utf-8")
    )
    aesthetics, axes = config["aesthetics"], config["axes"]
    names = [a["name"] for a in aesthetics]
    keys = [a["key"] for a in aesthetics]
    mapped = [set(a["kfashion"]["styles"]) for a in aesthetics]

    # 모든 프롬프트 묶음을 한 번에 logit으로 받고, 묶음별로 나눠 softmax한다.
    groups = [
        {"prompts": a["prompts"][ps]} for ps in args.prompt_sets for a in aesthetics
    ]
    for axis in axes:
        for low, high in axis["prompts"]:
            groups += [{"prompts": [low]}, {"prompts": [high]}]

    images_zip = find_zip(args.data_dir / "images")
    kept, _ = dedupe(load_records(find_zip(args.data_dir / "labels"), images_zip))
    sampled = sample_per_style(kept, styles, args.per_style, args.seed)
    logit_rows = predict(sampled, images_zip, groups, args.batch_size, softmax=False)

    n = len(names)
    report = {"seed": args.seed, "per_style": args.per_style, "aesthetics": names}
    top1_by_set = []
    for k, ps in enumerate(args.prompt_sets):
        rows = [softmax(row[k * n : (k + 1) * n]) for row in logit_rows]
        result = summarize(sampled, rows, styles, n)
        calibrated = calibrate(rows, result["overall"])
        top1_by_set.append(([argmax(r) for r in rows], [argmax(r) for r in calibrated]))
        result["top1"] = top1_counts(rows, n)
        result["top1_calibrated"] = top1_counts(calibrated, n)
        result["hit_rate"] = mapped_hit_rate(sampled, rows, mapped)
        result["hit_rate_calibrated"] = mapped_hit_rate(sampled, calibrated, mapped)
        result["confusion"] = {
            f"{a}_{b}": {
                "raw": pair_confusion(
                    sampled, rows, mapped, keys.index(a), keys.index(b)
                ),
                "calibrated": pair_confusion(
                    sampled, calibrated, mapped, keys.index(a), keys.index(b)
                ),
            }
            for a, b in CONFUSION_PAIRS
        }
        report[ps] = result

        out = output_dir / ps
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "scores.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["image_id", "main", "sub", *names, *(f"{x}_보정" for x in names)]
            )
            for r, row, cal in zip(sampled, rows, calibrated):
                writer.writerow(
                    [r.image_id, r.main, r.sub or "", *(f"{s:.4f}" for s in row + cal)]
                )

        print(f"\n[{ps}]\t" + "\t".join(names))
        print("1위 횟수\t" + "\t".join(map(str, result["top1"])))
        print("1위 횟수(보정)\t" + "\t".join(map(str, result["top1_calibrated"])))
        print("적중률\t" + "\t".join(map(fmt, result["hit_rate"])))
        print("적중률(보정)\t" + "\t".join(map(fmt, result["hit_rate_calibrated"])))
        print("전체 평균\t" + "\t".join(f"{x:.3f}" for x in result["overall"]))
        for pair, values in result["confusion"].items():
            for kind, c in values.items():
                print(f"혼동 {pair} ({kind}): {c}")

    if len(top1_by_set) == 2:
        (a_raw, a_cal), (b_raw, b_cal) = top1_by_set
        report["agreement"] = {
            "raw": sum(x == y for x, y in zip(a_raw, b_raw)) / len(sampled),
            "calibrated": sum(x == y for x, y in zip(a_cal, b_cal)) / len(sampled),
        }
        print(f"\n두 세트 1위 일치: {report['agreement']}")

    col = len(args.prompt_sets) * n
    report["axes"] = {}
    for axis in axes:
        wordings = []
        for _ in axis["prompts"]:
            wordings.append([axis_value(row[col : col + 2]) for row in logit_rows])
            col += 2
        values = [statistics.mean(v) for v in zip(*wordings)]
        by_style = {
            s: statistics.mean(v for r, v in zip(sampled, values) if r.main == s)
            for s in styles
        }
        ranked = sorted(by_style, key=by_style.get)
        report["axes"][axis["key"]] = {
            "wording_correlation": statistics.correlation(wordings[0], wordings[1]),
            "mean": statistics.mean(values),
            "by_style": by_style,
        }
        print(
            f"\n{axis['low']}↔{axis['high']}: 문장 간 상관 "
            f"{report['axes'][axis['key']]['wording_correlation']:.2f}"
        )
        for side, picked in (
            (axis["low"], ranked[:3]),
            (axis["high"], ranked[::-1][:3]),
        ):
            print(
                f"  {side} 쪽: " + ", ".join(f"{s} {by_style[s]:.2f}" for s in picked)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n결과: {output_dir}")


if __name__ == "__main__":
    main()

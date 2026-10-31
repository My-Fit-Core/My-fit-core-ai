"""K-Fashion Validation으로 FashionCLIP의 일반 스타일 구분 성능을 평가한다.

라벨과 이미지는 zip에서 바로 읽는다. 파일 12만 개를 하나씩 여는 것보다 빠르고,
cp949 파일 이름이 풀리면서 깨질 수 있는 OS에서도 같게 동작한다.

    python scripts/eval_kfashion_styles.py [--prompt-set name_only|described]

평가 기준은 docs/04-kfashion-validation.md 참고. 결과는 outputs/kfashion_eval/<프롬프트 세트>/에 저장한다.
"""

import argparse
import csv
import io
import json
import random
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "patrickjohncyh/fashion-clip"
ZIP_NAME_ENCODING = "cp949"


@dataclass(frozen=True)
class Record:
    image_id: int
    main: str | None
    sub: str | None
    content: tuple[int, int]  # 이미지 (CRC, 크기). 같으면 내용이 같은 이미지로 본다.
    member: str = ""


def dedupe(records):
    """내용이 같은 묶음은 1장만 남긴다. 묶음 안에 스타일 라벨이 다르면 묶음 전체를 뺀다."""
    groups = defaultdict(list)
    for r in records:
        groups[r.content].append(r)
    kept = []
    stats = {"dup_groups": 0, "dup_groups_conflicting": 0, "images_removed": 0}
    for members in groups.values():
        if len(members) == 1:
            kept.append(members[0])
            continue
        stats["dup_groups"] += 1
        if len({(m.main, m.sub) for m in members}) > 1:
            stats["dup_groups_conflicting"] += 1
            stats["images_removed"] += len(members)
            continue
        kept.append(min(members, key=lambda m: m.image_id))
        stats["images_removed"] += len(members) - 1
    return kept, stats


def sample_per_style(records, styles, per_style, seed):
    """주 스타일별로 같은 수를 뽑는다. 스타일 없음(기타)은 styles에 없으므로 빠진다."""
    rng = random.Random(seed)
    sampled = []
    for style in styles:
        candidates = sorted(
            (r for r in records if r.main == style), key=lambda r: r.image_id
        )
        if len(candidates) < per_style:
            raise SystemExit(
                f"{style}: 후보 {len(candidates)}장으로 {per_style}장을 뽑을 수 없습니다."
            )
        sampled += rng.sample(candidates, per_style)
    return sampled


def evaluate(records, preds, styles):
    """정답은 주 스타일 또는 서브스타일. 주 스타일만 맞힌 비율은 strict로 따로 센다."""
    index = {s: i for i, s in enumerate(styles)}
    matrix = [[0] * len(styles) for _ in styles]
    totals = dict.fromkeys(styles, 0)
    lenient = dict.fromkeys(styles, 0)
    strict = dict.fromkeys(styles, 0)
    for r, pred in zip(records, preds):
        matrix[index[r.main]][index[pred]] += 1
        totals[r.main] += 1
        lenient[r.main] += pred in (r.main, r.sub)
        strict[r.main] += pred == r.main
    present = [s for s in styles if totals[s]]
    per_style = {
        s: {
            "n": totals[s],
            "accuracy": lenient[s] / totals[s],
            "main_only_accuracy": strict[s] / totals[s],
        }
        for s in present
    }
    return {
        "balanced_accuracy": sum(v["accuracy"] for v in per_style.values())
        / len(present),
        "balanced_main_only_accuracy": sum(
            v["main_only_accuracy"] for v in per_style.values()
        )
        / len(present),
        "chance_main_only": 1 / len(styles),
        "chance_with_sub": sum(1 + bool(r.sub) for r in records)
        / len(records)
        / len(styles),
        "per_style": per_style,
        "confusion_matrix": matrix,
    }


def find_zip(directory):
    zips = sorted(directory.rglob("*.zip"))
    if len(zips) != 1:
        raise SystemExit(
            f"{directory}에서 zip을 1개 찾아야 하는데 {len(zips)}개입니다. docs/04-kfashion-validation.md 참고."
        )
    return zips[0]


def load_records(labels_zip, images_zip):
    with zipfile.ZipFile(images_zip, metadata_encoding=ZIP_NAME_ENCODING) as images:
        by_id = {Path(i.filename).stem: i for i in images.infolist() if not i.is_dir()}
    records = []
    with zipfile.ZipFile(labels_zip, metadata_encoding=ZIP_NAME_ENCODING) as labels:
        for info in labels.infolist():
            if not info.filename.endswith(".json"):
                continue
            doc = json.loads(labels.read(info).decode("utf-8"))
            style = doc["데이터셋 정보"]["데이터셋 상세설명"]["라벨링"]["스타일"][0]
            image = by_id[Path(info.filename).stem]
            records.append(
                Record(
                    image_id=doc["이미지 정보"]["이미지 식별자"],
                    main=style.get("스타일"),
                    sub=style.get("서브스타일"),
                    content=(image.CRC, image.file_size),
                    member=image.filename,
                )
            )
    return records


def predict(records, images_zip, config, batch_size):
    import torch
    from PIL import Image, ImageOps
    from transformers import CLIPModel, CLIPProcessor

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device={device}")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    prompts, prompt_ids = [], []
    for style in config:
        prompt_ids.append(
            list(range(len(prompts), len(prompts) + len(style["prompts"])))
        )
        prompts.extend(style["prompts"])

    rows = []
    with (
        zipfile.ZipFile(images_zip, metadata_encoding=ZIP_NAME_ENCODING) as archive,
        torch.no_grad(),
    ):
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            images = [
                ImageOps.exif_transpose(
                    Image.open(io.BytesIO(archive.read(r.member)))
                ).convert("RGB")
                for r in batch
            ]
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            logits = model(**inputs).logits_per_image
            per_style = torch.stack(
                [logits[:, ids].mean(dim=1) for ids in prompt_ids], dim=1
            )
            rows += per_style.softmax(dim=1).cpu().tolist()
            print(f"{start + len(batch)}/{len(records)}")
    return rows


def save_results(out_dir, records, score_rows, preds, result, config):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [s["name"] for s in config]
    keys = [s["key"] for s in config]
    (out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_id", "main", "sub", "pred", "correct", "main_only_correct", *names]
        )
        for r, pred, row in zip(records, preds, score_rows):
            writer.writerow(
                [
                    r.image_id,
                    r.main,
                    r.sub or "",
                    pred,
                    pred in (r.main, r.sub),
                    pred == r.main,
                    *(f"{s:.4f}" for s in row),
                ]
            )
    with open(
        out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["main \\ pred", *names])
        for name, row in zip(names, result["confusion_matrix"]):
            writer.writerow([name, *row])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 차트 라벨은 기본 글꼴에서 한글이 깨지므로 영어 key를 쓴다.
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(result["confusion_matrix"], cmap="Blues", vmin=0)
    ax.set_xticks(range(len(keys)), keys, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(keys)), keys, fontsize=8)
    ax.set_xlabel("predicted style")
    ax.set_ylabel("main style (label)")
    ax.set_title("FashionCLIP on K-Fashion validation")
    fig.colorbar(image, ax=ax, label="images", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)


def print_summary(result):
    print(
        f"\n균형 정확도 (주 또는 서브 인정): {result['balanced_accuracy']:.3f}  무작위 기준 {result['chance_with_sub']:.3f}"
    )
    print(
        f"균형 정확도 (주 스타일만):     {result['balanced_main_only_accuracy']:.3f}  무작위 기준 {result['chance_main_only']:.3f}"
    )
    print("\n스타일\tn\t주 또는 서브\t주 스타일만")
    for style, v in sorted(
        result["per_style"].items(), key=lambda kv: -kv[1]["accuracy"]
    ):
        print(f"{style}\t{v['n']}\t{v['accuracy']:.2f}\t{v['main_only_accuracy']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data/kfashion/validation"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--prompt-set", default="name_only", choices=["name_only", "described"]
    )
    parser.add_argument("--per-style", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config = yaml.safe_load(
        (ROOT / "configs/kfashion_styles.yaml").read_text(encoding="utf-8")
    )["styles"]
    config = [{**s, "prompts": s["prompts"][args.prompt_set]} for s in config]
    output_dir = args.output_dir or ROOT / "outputs/kfashion_eval" / args.prompt_set
    styles = [s["name"] for s in config]
    images_zip = find_zip(args.data_dir / "images")

    all_records = load_records(find_zip(args.data_dir / "labels"), images_zip)
    unknown = {r.main for r in all_records if r.main} - set(styles)
    if unknown:
        raise SystemExit(f"설정에 없는 스타일: {sorted(unknown)}")
    kept, stats = dedupe(all_records)
    candidates = {s: sum(r.main == s for r in kept) for s in styles}
    print(f"전체 {len(all_records)}장, 중복 처리 후 {len(kept)}장, {stats}")
    print(f"가장 적은 후보: {min(candidates.items(), key=lambda kv: kv[1])}")

    sampled = sample_per_style(kept, styles, args.per_style, args.seed)
    score_rows = predict(sampled, images_zip, config, args.batch_size)
    preds = [styles[row.index(max(row))] for row in score_rows]
    result = evaluate(sampled, preds, styles)
    result["run"] = {
        "model": MODEL_ID,
        "prompt_set": args.prompt_set,
        "seed": args.seed,
        "per_style": args.per_style,
        "total_images": len(all_records),
        "dedupe": stats,
    }
    save_results(output_dir, sampled, score_rows, preds, result, config)
    print_summary(result)
    print(f"\n결과: {output_dir}")


if __name__ == "__main__":
    main()

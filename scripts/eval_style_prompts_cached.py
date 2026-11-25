"""추구미 문장을 빠르게 비교하는 평가 (issue #111).

사진 임베딩을 한 번만 계산해 저장하고(outputs/style_prompt_cache/), 문장만 바꿔 다시 잰다.
eval_aesthetics_on_kfashion.py와 같은 표본·같은 적중률 정의를 쓴다. 추가로 추구미별 AUC를 낸다.

- 적중률: 연결 스타일 사진 중 1위가 그 추구미인 비율(추구미끼리 softmax 1위)
- 보정 적중률: 점수를 표본 전체 평균으로 나눠 다시 1위를 고른 적중률(issue #40)
- AUC: 추구미 하나만 보고, 연결 스타일 사진의 점수가 나머지 사진보다 높은지(0.5 = 우연).
  다른 추구미 점수와 상관없어 쏠림의 영향을 받지 않는다. 독립 점수(태깅 v1)와 같은 성격의 지표다.

문장 다듬기용과 최종 확인용 사진을 나눈다. --exclude-seeds로 다른 시드 표본의 사진을 빼고 뽑는다.

    python scripts/eval_style_prompts_cached.py --config configs/aesthetics_7.yaml --seed 42
    python scripts/eval_style_prompts_cached.py --config configs/aesthetics_7.yaml --seed 123 --exclude-seeds 42
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_kfashion_styles import (
    MODEL_ID,
    ROOT,
    dedupe,
    find_zip,
    load_records,
)

CACHE_DIR = ROOT / "outputs/style_prompt_cache"


def sample(kept, styles, per_style, seed, exclude_ids):
    """eval_kfashion_styles.sample_per_style와 같은 방식. 제외 후 모자라면 남은 만큼만 쓴다."""
    rng = random.Random(seed)
    out, short = [], {}
    for style in styles:
        cands = sorted(
            (r for r in kept if r.main == style and r.image_id not in exclude_ids),
            key=lambda r: r.image_id,
        )
        k = min(per_style, len(cands))
        if k < per_style:
            short[style] = k
        out += rng.sample(cands, k)
    return out, short


def load_model():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(MODEL_ID).eval()
    return model, CLIPProcessor.from_pretrained(MODEL_ID), torch


def _feat(out):
    return getattr(out, "pooler_output", out)


def image_embeddings(records, images_path, cache: Path, batch_size=32):
    """L2 정규화 이미지 임베딩 (N, 512). 캐시가 있으면 그대로 쓴다."""
    ids = [r.image_id for r in records]
    if cache.exists():
        data = np.load(cache)
        if data["ids"].tolist() == ids:
            return data["emb"]
    import io

    from PIL import Image, ImageOps

    from scripts.eval_kfashion_styles import open_images

    model, processor, torch = load_model()
    embs = []
    with open_images(images_path) as archive, torch.no_grad():
        for s in range(0, len(records), batch_size):
            batch = records[s : s + batch_size]
            imgs = [
                ImageOps.exif_transpose(Image.open(io.BytesIO(archive.read(r.member)))).convert("RGB")
                for r in batch
            ]
            f = _feat(model.get_image_features(**processor(images=imgs, return_tensors="pt"))).float()
            embs.append((f / f.norm(dim=-1, keepdim=True)).numpy())
            print(f"{s + len(batch)}/{len(records)}", flush=True)
    emb = np.concatenate(embs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.array(ids), emb=emb)
    return emb


def text_logits(img_emb, prompt_groups):
    """추구미별 문장 평균 logit (N, K). logit = logit_scale × 코사인 유사도."""
    model, processor, torch = load_model()
    with torch.no_grad():
        scale = float(model.logit_scale.exp())
        cols = []
        for prompts in prompt_groups:
            inp = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
            t = _feat(model.get_text_features(**inp)).float()
            t = (t / t.norm(dim=-1, keepdim=True)).numpy()
            cols.append((scale * img_emb @ t.T).mean(axis=1))
    return np.stack(cols, axis=1)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney 방식 AUC (동점은 0.5)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float)
    # 동점 평균 순위
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    r_pos = ranks[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2
    return float(r_pos / (len(pos) * len(neg)))


def evaluate(records, logits, aesthetics):
    mains = np.array([r.main for r in records])
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    cal = probs / probs.mean(axis=0)
    top, top_cal = probs.argmax(axis=1), cal.argmax(axis=1)
    out = []
    for k, a in enumerate(aesthetics):
        mask = np.isin(mains, a["kfashion"]["styles"])
        out.append({
            "key": a["key"],
            "name": a["name"],
            "n": int(mask.sum()),
            "hit": float((top[mask] == k).mean()),
            "hit_cal": float((top_cal[mask] == k).mean()),
            "auc": auc(logits[mask, k], logits[~mask, k]),
            "top1": int((top == k).sum()),
        })
    n = len(aesthetics)
    summary = {
        "hit_mean": float(np.mean([o["hit"] for o in out])),
        "hit_cal_mean": float(np.mean([o["hit_cal"] for o in out])),
        "auc_mean": float(np.nanmean([o["auc"] for o in out])),
        "chance": 1 / n,
    }
    return out, summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/kfashion/validation")
    p.add_argument("--config", type=Path, default=ROOT / "configs/aesthetics.yaml")
    p.add_argument("--prompt-set", default="outfit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-style", type=int, default=50)
    p.add_argument("--exclude-seeds", type=int, nargs="*", default=[])
    p.add_argument("--json", type=Path, help="결과를 저장할 파일")
    p.add_argument("--confusion", action="store_true", help="추구미별로 연결 스타일 사진의 1위 분포를 출력")
    args = p.parse_args()

    styles = [s["name"] for s in yaml.safe_load((ROOT / "configs/kfashion_styles.yaml").read_text(encoding="utf-8"))["styles"]]
    aesthetics = yaml.safe_load(args.config.read_text(encoding="utf-8"))["aesthetics"]

    images = find_zip(args.data_dir / "images", allow_dir=True)
    kept, _ = dedupe(load_records(find_zip(args.data_dir / "labels"), images))

    exclude = set()
    for s in args.exclude_seeds:
        ex, _ = sample(kept, styles, args.per_style, s, set())
        exclude |= {r.image_id for r in ex}
    records, short = sample(kept, styles, args.per_style, args.seed, exclude)
    tag = f"seed{args.seed}_n{args.per_style}" + (f"_ex{'-'.join(map(str, args.exclude_seeds))}" if exclude else "")
    emb = image_embeddings(records, images, CACHE_DIR / f"{tag}.npz")

    logits = text_logits(emb, [a["prompts"][args.prompt_set] for a in aesthetics])
    per, summ = evaluate(records, logits, aesthetics)
    print(f"[{args.config.name} {tag}] 사진 {len(records)}장" + (f", 모자란 스타일 {short}" if short else ""))
    print("추구미\t적중률\t보정\tAUC\t1위 수")
    for o in per:
        print(f"{o['name']}\t{o['hit']:.2f}\t{o['hit_cal']:.2f}\t{o['auc']:.3f}\t{o['top1']}")
    print(
        f"평균\t{summ['hit_mean']:.3f}\t{summ['hit_cal_mean']:.3f}\t{summ['auc_mean']:.3f}\t"
        f"(우연 {summ['chance']:.3f}, 배수 {summ['hit_mean'] / summ['chance']:.2f} / 보정 {summ['hit_cal_mean'] / summ['chance']:.2f})"
    )
    if args.confusion:
        mains = np.array([r.main for r in records])
        top = logits.argmax(axis=1)
        for k, a in enumerate(aesthetics):
            mask = np.isin(mains, a["kfashion"]["styles"])
            counts = np.bincount(top[mask], minlength=len(aesthetics))
            order = counts.argsort()[::-1][:3]
            print(f"  {a['name']} 사진 → " + ", ".join(f"{aesthetics[i]['name']} {counts[i]}" for i in order))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"config": args.config.name, "tag": tag, "n": len(records), "short": short, "per": per, "summary": summ}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

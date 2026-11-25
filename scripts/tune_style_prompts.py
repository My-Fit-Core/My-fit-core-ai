"""추구미 문장 후보 중에서 조합을 골라 적중률을 높인다 (issue #111, 실험용).

- 사진 임베딩은 eval_style_prompts_cached.py가 만든 캐시를 쓴다(문장만 바꾸므로 빠르다).
- 스타일마다 후보 문장 묶음(pool)에서 k개를 고른다. 한 스타일씩 돌아가며 가장 좋은 조합으로 바꾸기를
  몇 바퀴 반복한다(좌표 상승). 목표는 문장 다듬기용 사진(--tune-seed)의 추구미별 적중률 평균이다.
- 고른 문장은 확인용 사진(--check-seed, 다듬기용 사진을 뺀 표본)에서 따로 잰다. 확인용 숫자만 보고한다.
- 연결 스타일(K-Fashion)은 바꾸지 않는다. 문장만 고른다.

    python scripts/tune_style_prompts.py --config configs/aesthetics_7v4.yaml --pool pool.json \\
        --tune-seed 42 --check-seed 123 --out configs/aesthetics_7_tuned.yaml
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_kfashion_styles import (
    ROOT,
    dedupe,
    find_zip,
    load_records,
)
from scripts.eval_style_prompts_cached import (
    CACHE_DIR,
    evaluate,
    image_embeddings,
    load_model,
    sample,
)


def encode_texts(texts):
    model, processor, torch = load_model()
    with torch.no_grad():
        scale = float(model.logit_scale.exp())
        inp = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        f = getattr(model.get_text_features(**inp), "pooler_output", None)
        f = model.get_text_features(**inp) if f is None else f
        f = f.float()
        return scale, (f / f.norm(dim=-1, keepdim=True)).numpy()


def hit_mean(sims, choice, mains, mapped):
    """choice: 스타일마다 고른 문장 인덱스 목록. sims: (N, 전체 문장)."""
    logits = np.stack([sims[:, idx].mean(axis=1) for idx in choice], axis=1)
    top = logits.argmax(axis=1)
    rates = []
    for k, styles in enumerate(mapped):
        m = np.isin(mains, styles)
        rates.append((top[m] == k).mean())
    return float(np.mean(rates)), logits


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/kfashion/validation")
    p.add_argument("--config", type=Path, required=True, help="시작 설정(연결 스타일·이름)")
    p.add_argument("--pool", type=Path, required=True, help="스타일 key → 후보 문장 목록 JSON")
    p.add_argument("--k", type=int, default=4, help="스타일마다 고를 문장 수")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--tune-seed", type=int, default=42)
    p.add_argument("--check-seed", type=int, default=123)
    p.add_argument("--per-style", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    styles = [s["name"] for s in yaml.safe_load((ROOT / "configs/kfashion_styles.yaml").read_text(encoding="utf-8"))["styles"]]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    aes = cfg["aesthetics"]
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    for a in aes:  # 지금 문장도 후보에 넣는다
        pool[a["key"]] = list(dict.fromkeys(a["prompts"]["outfit"] + pool.get(a["key"], [])))

    images = find_zip(args.data_dir / "images", allow_dir=True)
    kept, _ = dedupe(load_records(find_zip(args.data_dir / "labels"), images))
    tune_recs, _ = sample(kept, styles, args.per_style, args.tune_seed, set())
    ex = {r.image_id for r in tune_recs}
    check_recs, short = sample(kept, styles, args.per_style, args.check_seed, ex)
    tune_emb = image_embeddings(tune_recs, images, CACHE_DIR / f"seed{args.tune_seed}_n{args.per_style}.npz")
    check_emb = image_embeddings(
        check_recs, images, CACHE_DIR / f"seed{args.check_seed}_n{args.per_style}_ex{args.tune_seed}.npz"
    )

    texts, ranges = [], []
    for a in aes:
        ranges.append(list(range(len(texts), len(texts) + len(pool[a["key"]]))))
        texts += pool[a["key"]]
    scale, t = encode_texts(texts)
    tune_sims, check_sims = scale * tune_emb @ t.T, scale * check_emb @ t.T
    tune_mains = np.array([r.main for r in tune_recs])
    mapped = [a["kfashion"]["styles"] for a in aes]

    # 시작: 지금 문장(각 pool의 앞부분)
    choice = [r[: len(a["prompts"]["outfit"])] for r, a in zip(ranges, aes)]
    best, _ = hit_mean(tune_sims, choice, tune_mains, mapped)
    print(f"시작 다듬기용 적중률 평균 {best:.3f}")
    for rnd in range(args.rounds):
        improved = False
        for k, r in enumerate(ranges):
            for combo in itertools.combinations(r, min(args.k, len(r))):
                trial = choice[:k] + [list(combo)] + choice[k + 1 :]
                score, _ = hit_mean(tune_sims, trial, tune_mains, mapped)
                if score > best + 1e-9:
                    best, choice, improved = score, trial, True
        print(f"{rnd + 1}바퀴: 다듬기용 {best:.3f}")
        if not improved:
            break

    for a, idx in zip(aes, choice):
        a["prompts"]["outfit"] = [texts[i] for i in idx]
    head = "".join(line for line in args.config.read_text(encoding="utf-8").splitlines(True) if line.startswith("#"))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(head + f"# 문장은 scripts/tune_style_prompts.py로 고름(다듬기 시드 {args.tune_seed}, 후보 {args.pool.name})\n\n")
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, width=200)

    for name, sims, recs in (("다듬기용", tune_sims, tune_recs), ("확인용", check_sims, check_recs)):
        logits = np.stack([sims[:, idx].mean(axis=1) for idx in choice], axis=1)
        per, summ = evaluate(recs, logits, aes)
        print(f"[{name}] " + " ".join(f"{o['name']} {o['hit']:.2f}" for o in per))
        print(f"  평균 {summ['hit_mean']:.3f} 보정 {summ['hit_cal_mean']:.3f} AUC {summ['auc_mean']:.3f}")
    if short:
        print("확인용에서 모자란 스타일:", short)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()

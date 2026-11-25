"""옷 한 벌용 추구미 문장 고르기와 독립 점수 offset 정하기 (#99, 실험용).

eval_item_attributes_kfashion.py가 만든 옷 임베딩 캐시를 쓴다. 다듬기용(--tune)으로 문장을 고르고
확인용(--check, 다듬기용 사진을 뺀 표본)에서만 결과를 보고한다. 방법은 tune_style_prompts.py와 같다.
offset은 다듬기용에서 "옷별 1위 스타일 점수의 중앙값이 0.5"가 되는 값으로 정하고, 확인용 분포를 함께 본다.

    python scripts/tune_item_style_prompts.py --tune outputs/item_attributes/seed42 \\
        --check outputs/item_attributes/seed7_ex42 --out configs/styles_tuned.yaml
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_kfashion_styles import ROOT
from scripts.eval_style_prompts_cached import auc
from scripts.tune_style_prompts import encode_texts


def load(d: Path):
    meta = [m for m in json.loads((d / "items_meta.json").read_text(encoding="utf-8")) if m["matched"]]
    emb = np.load(d / "cache.npz")["emb"]
    return emb[[m["emb"] for m in meta]], np.array([m["style"] for m in meta])


def logits_for(sims, choice):
    return np.stack([sims[:, idx].mean(axis=1) for idx in choice], axis=1)


def hit_mean(logits, styles, mapped):
    top = logits.argmax(axis=1)
    return float(np.mean([(top[np.isin(styles, m)] == k).mean() for k, m in enumerate(mapped)]))


RESULTS: dict = {}


def report(name, logits, styles, mapped, keys):
    top = logits.argmax(axis=1)
    per = []
    for k, m in enumerate(mapped):
        mask = np.isin(styles, m)
        per.append((keys[k], float((top[mask] == k).mean()), auc(logits[mask, k], logits[~mask, k]), int((top == k).sum())))
    print(f"[{name}] 적중률 평균 {np.mean([p[1] for p in per]):.3f}  AUC 평균 {np.mean([p[2] for p in per]):.3f}")
    print("   " + "  ".join(f"{k} {h:.2f}/{a:.3f}/{t}" for k, h, a, t in per))
    RESULTS[name] = {"hit_mean": float(np.mean([p[1] for p in per])), "auc_mean": float(np.mean([p[2] for p in per])),
                     "per": [{"key": k, "hit": h, "auc": a, "top1": t} for k, h, a, t in per]}
    return per


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tune", type=Path, required=True)
    ap.add_argument("--check", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=ROOT / "configs/style_item_prompt_pool.json")
    ap.add_argument("--styles", type=Path, default=ROOT / "configs/styles.yaml")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path, help="결과 수치를 저장할 파일")
    args = ap.parse_args()

    styles_cfg = yaml.safe_load(args.styles.read_text(encoding="utf-8"))
    a7 = {a["key"]: a["kfashion"]["styles"] for a in yaml.safe_load((ROOT / "configs/aesthetics_7.yaml").read_text(encoding="utf-8"))["aesthetics"]}
    keys = [s["key"] for s in styles_cfg["styles"]]
    mapped = [a7[k] for k in keys]
    pool = json.loads(args.pool.read_text(encoding="utf-8"))

    texts, ranges = [], []
    for s in styles_cfg["styles"]:
        cand = list(dict.fromkeys(s["prompts"] + pool.get(s["key"], [])))
        ranges.append(list(range(len(texts), len(texts) + len(cand))))
        texts += cand
    neutral = styles_cfg["independent_score"]["neutral_prompts"]
    scale, t = encode_texts(texts + neutral)
    t_style, t_neutral = t[: len(texts)], t[len(texts):]

    tune_emb, tune_styles = load(args.tune)
    check_emb, check_styles = load(args.check)
    tune_sims, check_sims = scale * tune_emb @ t_style.T, scale * check_emb @ t_style.T

    choice = [r[: len(s["prompts"])] for r, s in zip(ranges, styles_cfg["styles"])]
    report("지금 문장 / 다듬기용", logits_for(tune_sims, choice), tune_styles, mapped, keys)
    report("지금 문장 / 확인용", logits_for(check_sims, choice), check_styles, mapped, keys)
    best = hit_mean(logits_for(tune_sims, choice), tune_styles, mapped)
    for rnd in range(args.rounds):
        improved = False
        for k, r in enumerate(ranges):
            for combo in itertools.combinations(r, min(args.k, len(r))):
                trial = choice[:k] + [list(combo)] + choice[k + 1:]
                sc = hit_mean(logits_for(tune_sims, trial), tune_styles, mapped)
                if sc > best + 1e-9:
                    best, choice, improved = sc, trial, True
        print(f"{rnd + 1}바퀴 다듬기용 {best:.3f}")
        if not improved:
            break
    report("고른 문장 / 다듬기용", logits_for(tune_sims, choice), tune_styles, mapped, keys)
    report("고른 문장 / 확인용", logits_for(check_sims, choice), check_styles, mapped, keys)

    # offset: 다듬기용에서 옷별 1위 점수 중앙값이 0.5가 되게
    temp = styles_cfg["independent_score"]["temperature"]

    def diffs(emb, sims):
        neu = (scale * emb @ t_neutral.T).mean(axis=1, keepdims=True)
        return temp * (logits_for(sims, choice) - neu)

    d_tune, d_check = diffs(tune_emb, tune_sims), diffs(check_emb, check_sims)
    offset = float(-np.median(d_tune.max(axis=1)))
    sig = lambda x: 1 / (1 + np.exp(-x))
    for name, d in (("다듬기용", d_tune), ("확인용", d_check)):
        s = sig(d + offset)
        print(f"offset {offset:.2f} {name}: 1위 점수 중앙값 {np.median(s.max(axis=1)):.2f}, 전체 평균 {s.mean():.2f}, 0.5 넘는 점수 비율 {(s > 0.5).mean():.2f}")
        RESULTS[f"offset/{name}"] = {"offset": offset, "top_median": float(np.median(s.max(axis=1))),
                                     "mean": float(s.mean()), "share_over_0.5": float((s > 0.5).mean())}

    for s, idx in zip(styles_cfg["styles"], choice):
        s["prompts"] = [texts[i] for i in idx]
    styles_cfg["independent_score"]["offset"] = round(offset, 2)
    head = "".join(line for line in args.styles.read_text(encoding="utf-8").splitlines(True) if line.startswith("#"))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(head + f"# [실험] 문장·offset은 scripts/tune_item_style_prompts.py로 고름(다듬기용 {args.tune.name}, 확인용 {args.check.name})\n\n")
        yaml.safe_dump(styles_cfg, f, allow_unicode=True, sort_keys=False, width=200)
    if args.json:
        args.json.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()

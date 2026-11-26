"""레퍼런스 사진으로 추구미 7종 코디 문장 고르기 (#111, 실험용).

다듬기: 레퍼런스 사진 절반(하위 태그별로 반씩) + K-Fashion 다듬기용(시드 42)
확인: 레퍼런스 나머지 절반 + K-Fashion 확인용(시드 123, 시드 42 사진 제외)
목표: 다듬기용 레퍼런스 적중률 평균 + K-Fashion 적중률 평균의 합(요즘 사진만 맞추고 K-Fashion이 무너지지 않게)
방법: 스타일마다 문장 후보(configs/style_prompt_pool.json)에서 4개 조합, 한 스타일씩 돌아가며 바꾸기

    python scripts/tune_reference_prompts.py --start configs/aesthetics_7.yaml --out configs/aesthetics_7_ref.yaml
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_reference_styles import (
    KEYS,
    REF,
    ROOT,
    embeddings,
    evaluate,
    load_reference,
    split_mask,
)
from scripts.tune_style_prompts import encode_texts

KF = ROOT / "outputs/style_prompt_cache"


def kf_set(tag, mapping, cfg_styles):
    """K-Fashion 캐시 임베딩과 7종 정답(연결 스타일)."""
    import json as _j

    emb = np.load(KF / f"{tag}.npz")
    ids, e = emb["ids"], emb["emb"]
    meta = _j.loads((KF / f"{tag}_mains.json").read_text(encoding="utf-8"))
    mains = np.array([meta[str(i)] for i in ids])
    truth = np.stack([np.isin(mains, mapping[k]) for k in KEYS], 1)
    keep = truth.any(1)
    return e[keep], truth[keep]


def hit(scores, truth):
    top = scores.argmax(1)
    return float(np.mean([truth[truth[:, k], top[truth[:, k]]].mean() for k in range(len(KEYS)) if truth[:, k].any()]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=Path, default=ROOT / "configs/aesthetics_7.yaml")
    ap.add_argument("--pool", type=Path, default=ROOT / "configs/style_prompt_pool.json")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metric", choices=["hit", "auc"], default="hit", help="목표 지표(auc는 쏠림에 덜 민감)")
    ap.add_argument("--json", type=Path, default=ROOT / "outputs/reference_eval/tune_result.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.start.read_text(encoding="utf-8"))
    aes = {a["key"]: a for a in cfg["aesthetics"]}
    mapping = {k: aes[k]["kfashion"]["styles"] for k in KEYS}
    pool = json.loads(args.pool.read_text(encoding="utf-8"))

    rows, rtruth = load_reference()
    remb = embeddings(rows, REF, ROOT / "outputs/reference_eval/cache.npz")
    tune_mask = split_mask(rows)
    k_tune_e, k_tune_t = kf_set("seed42_n50", mapping, aes)
    k_chk_e, k_chk_t = kf_set("seed123_n50_ex42", mapping, aes)

    texts, ranges = [], []
    for k in KEYS:
        cand = list(dict.fromkeys(aes[k]["prompts"]["outfit"] + pool.get(k, [])))
        ranges.append(list(range(len(texts), len(texts) + len(cand))))
        texts += cand
    scale, t = encode_texts(texts)
    sims = {name: scale * e @ t.T for name, e in
            {"r": remb, "kt": k_tune_e, "kc": k_chk_e}.items()}

    def scores(name, choice, mask=None):
        s = np.stack([sims[name][:, idx].mean(1) for idx in choice], 1)
        return s if mask is None else s[mask]

    def objective(choice):
        if args.metric == "auc":
            return (evaluate(scores("r", choice), rtruth, tune_mask)["auc_mean"]
                    + evaluate(scores("kt", choice), k_tune_t)["auc_mean"])
        return hit(scores("r", choice, tune_mask), rtruth[tune_mask]) + hit(scores("kt", choice), k_tune_t)

    choice = [r[: len(aes[k]["prompts"]["outfit"])] for r, k in zip(ranges, KEYS)]
    result = {}

    def report(label, ch):
        out = {
            "ref_tune": evaluate(scores("r", ch), rtruth, tune_mask),
            "ref_check": evaluate(scores("r", ch), rtruth, ~tune_mask),
            "kf_check": evaluate(scores("kc", ch), k_chk_t),
        }
        result[label] = out
        print(f"[{label}] 레퍼런스 다듬기 {out['ref_tune']['hit_mean']:.3f}/{out['ref_tune']['auc_mean']:.3f}  "
              f"레퍼런스 확인 {out['ref_check']['hit_mean']:.3f}/{out['ref_check']['auc_mean']:.3f}  "
              f"K-Fashion 확인 {out['kf_check']['hit_mean']:.3f}/{out['kf_check']['auc_mean']:.3f}  (적중률/AUC)")

    report("시작 문장", choice)
    best = objective(choice)
    for rnd in range(args.rounds):
        improved = False
        for i, r in enumerate(ranges):
            for combo in itertools.combinations(r, min(args.k, len(r))):
                trial = choice[:i] + [list(combo)] + choice[i + 1:]
                s = objective(trial)
                if s > best + 1e-9:
                    best, choice, improved = s, trial, True
        print(f"{rnd + 1}바퀴 목표 {best:.3f}")
        if not improved:
            break
    report("고른 문장", choice)

    for k, idx in zip(KEYS, choice):
        aes[k]["prompts"]["outfit"] = [texts[i] for i in idx]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# [실험, 서비스에 쓰지 않음] 레퍼런스 사진 + K-Fashion으로 고른 추구미 7종 코디 문장 (#111)\n"
                f"# scripts/tune_reference_prompts.py, 시작 {args.start.name}. 결과 {args.json}\n\n")
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, width=200)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

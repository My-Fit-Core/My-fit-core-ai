"""요즘 스타일 레퍼런스 사진(data/reference_styles/, 레포 밖)으로 추구미 7종을 잰다 (#99, #111).

정답은 labels.csv의 7종 칸(1이면 그 추구미, 여러 개 가능). 핀터레스트 검색어 기준이라 섞였을 수 있다(미검증).
사진 전체(코디)를 한 장으로 본다.

지표
- 1위 적중률: 1위 추구미가 정답 칸 중 하나면 맞음. 추구미별로 "그 추구미가 정답인 사진" 중 맞은 비율, 그 평균
- AUC: 추구미 하나만 보고 정답 사진의 점수가 다른 사진보다 높은지(0.5 = 우연)

방법
- prompts: 설정 파일의 문장(outfit 또는 item)과 비교(지금 방식)
- prototype: 레퍼런스 사진의 평균 임베딩과 비교(예시 사진 방식). 자기 자신이 들어가지 않게 나머지 사진만으로
  평균을 낸다(leave-one-out). 방법 비교용이며 서비스에 넣으려면 평균 임베딩을 저장해야 한다(결정 필요)

    python scripts/eval_reference_styles.py --configs configs/aesthetics_7.yaml configs/aesthetics_7_tuned.yaml
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_style_prompts_cached import auc

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data/reference_styles"
KEYS = ["minimal", "casual", "feminine", "street", "y2k_vintage", "office_classic", "bohemian"]


def load_reference(ref_dir: Path = REF):
    with open(ref_dir / "labels.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if any(r.get(k) == "1" for k in KEYS)]
    truth = np.array([[r[k] == "1" for k in KEYS] for r in rows])
    return rows, truth


def embeddings(rows, ref_dir: Path, cache: Path):
    files = [r["file"] for r in rows]
    if cache.exists():
        d = np.load(cache)
        if d["files"].tolist() == files:
            return d["emb"]
    from PIL import Image

    from app.pipeline import tagging

    model, processor = tagging.load_tagger()
    emb = np.concatenate(
        [tagging._encode_image(model, processor, Image.open(ref_dir / "images" / f).convert("RGB")) for f in files]
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, files=np.array(files), emb=emb)
    return emb


def prompt_scores(emb, cfg_path: Path, prompt_set: str):
    from scripts.tune_style_prompts import encode_texts

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    aes = {a["key"]: a for a in (cfg.get("aesthetics") or cfg["styles"])}
    cols = []
    for k in KEYS:
        p = aes[k]["prompts"]
        texts = p[prompt_set] if isinstance(p, dict) else p
        scale, t = encode_texts(texts)
        cols.append((scale * emb @ t.T).mean(axis=1))
    return np.stack(cols, axis=1)


def prototype_scores(emb, truth, train_mask=None):
    """스타일별 평균 임베딩과의 코사인 유사도. train_mask가 없으면 leave-one-out."""
    n = len(emb)
    out = np.zeros((n, len(KEYS)))
    for k in range(len(KEYS)):
        for i in range(n):
            use = truth[:, k].copy()
            if train_mask is not None:
                use &= train_mask
            use[i] = False  # 자기 자신은 빼기
            if not use.any():
                out[i, k] = -1
                continue
            c = emb[use].mean(axis=0)
            out[i, k] = emb[i] @ c / np.linalg.norm(c)
    return out


def evaluate(scores, truth, mask=None):
    if mask is not None:
        scores, truth = scores[mask], truth[mask]
    top = scores.argmax(axis=1)
    per = []
    for k, key in enumerate(KEYS):
        pos = truth[:, k]
        per.append({
            "key": key,
            "n": int(pos.sum()),
            "hit": float(truth[pos, top[pos]].mean()) if pos.any() else None,
            "auc": auc(scores[pos, k], scores[~pos, k]) if pos.any() and (~pos).any() else None,
            "top1": int((top == k).sum()),
        })
    return {
        "per": per,
        "hit_mean": float(np.mean([p["hit"] for p in per if p["hit"] is not None])),
        "auc_mean": float(np.mean([p["auc"] for p in per if p["auc"] is not None])),
        "n": len(truth),
    }


def configs_five(emb, truth):
    order = ["feminine", "street", "minimal", "vintage_layered", "casual"]
    idx = {k: i for i, k in enumerate(KEYS)}
    t5 = np.stack([truth[:, idx["feminine"]], truth[:, idx["street"]], truth[:, idx["minimal"]],
                   truth[:, idx["y2k_vintage"]] | truth[:, idx["bohemian"]], truth[:, idx["casual"]]], 1)
    keep = t5.any(1)
    from scripts.tune_style_prompts import encode_texts

    cfg = {a["key"]: a for a in yaml.safe_load((ROOT / "configs/aesthetics.yaml").read_text(encoding="utf-8"))["aesthetics"]}
    cols = []
    for k in order:
        scale, t = encode_texts(cfg[k]["prompts"]["outfit"])
        cols.append((scale * emb[keep] @ t.T).mean(axis=1))
    s, tt = np.stack(cols, 1), t5[keep]
    top = s.argmax(1)
    per = [{"key": k, "n": int(tt[:, j].sum()), "hit": float(tt[tt[:, j], top[tt[:, j]]].mean()),
            "auc": auc(s[tt[:, j], j], s[~tt[:, j], j]), "top1": int((top == j).sum())} for j, k in enumerate(order)]
    return {"per": per, "hit_mean": float(np.mean([p["hit"] for p in per])),
            "auc_mean": float(np.mean([p["auc"] for p in per])), "n": int(keep.sum())}


def split_mask(rows, seed=0):
    """하위 태그별로 반씩 나눠 다듬기용(True)/확인용(False)."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r.get("sub_tag") or r["file"], []).append(i)
    mask = np.zeros(len(rows), bool)
    for idx in groups.values():
        idx = list(rng.permutation(idx))
        mask[idx[: (len(idx) + 1) // 2]] = True
    return mask


def show(name, r):
    print(f"[{name}] 사진 {r['n']}장  1위 적중률 평균 {r['hit_mean']:.3f}  AUC 평균 {r['auc_mean']:.3f}")
    print("   " + "  ".join(f"{p['key']} {p['hit']:.2f}/{p['auc']:.3f}/{p['top1']}" for p in r["per"] if p["hit"] is not None))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["configs/aesthetics_7.yaml", "configs/aesthetics_7_tuned.yaml"])
    ap.add_argument("--item-configs", nargs="*", default=["configs/styles.yaml"])
    ap.add_argument("--json", type=Path, default=ROOT / "outputs/reference_eval/summary.json")
    args = ap.parse_args()

    rows, truth = load_reference()
    emb = embeddings(rows, REF, ROOT / "outputs/reference_eval/cache.npz")
    tune = split_mask(rows)
    rep = {"n": len(rows), "per_style_n": truth.sum(axis=0).tolist(), "results": {}}
    for c in args.configs:
        r = evaluate(prompt_scores(emb, ROOT / c, "outfit"), truth)
        rep["results"][f"prompts:{c}"] = r
        show(c, r)
    for c in args.item_configs:
        r = evaluate(prompt_scores(emb, ROOT / c, "item"), truth)
        rep["results"][f"prompts:{c}"] = r
        show(c + " (옷 한 벌 문장)", r)
    r = evaluate(prototype_scores(emb, truth), truth)
    rep["results"]["prototype:leave-one-out"] = r
    show("예시 사진 평균(leave-one-out)", r)
    r = evaluate(prototype_scores(emb, truth, tune), truth, ~tune)
    rep["results"]["prototype:tune->check"] = r
    show("예시 사진 평균(다듬기용 반 → 확인용 반)", r)
    # 5종 비교: 빈티지 레이어드 = Y2K/빈티지 또는 보헤미안, 5종에 대응이 없는 오피스만 있는 사진은 뺀다
    five = configs_five(emb, truth)
    rep["results"]["prompts:configs/aesthetics.yaml (5종)"] = five
    show("5종 configs/aesthetics.yaml", five)
    same = truth[:, [i for i, k in enumerate(KEYS) if k != "office_classic"]].any(1)  # 5종과 같은 사진
    for c in args.configs:
        r = evaluate(prompt_scores(emb, ROOT / c, "outfit"), truth, same)
        rep["results"][f"prompts:{c} (5종과 같은 사진)"] = r
        show(c + " (5종과 같은 사진)", r)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

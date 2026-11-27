"""레퍼런스 사진에서 잘라낸 옷 한 벌로 서비스 추구미 문장(configs/styles.yaml)을 고른다 (#99, #111).

- 레퍼런스 사진마다 분석 파이프라인과 같은 방식(검출 → 중복·잘린 옷 제외 → 마스크 밖 흰색 크롭)으로 옷을 잘라낸다.
  옷의 정답은 그 사진의 7종 칸(1이면 그 추구미). 옷 하나가 코디 전체 스타일을 대표하지는 않으므로 낮게 나오는 지표다
- 다듬기: 레퍼런스 옷(사진 단위로 하위 태그별 반) + K-Fashion 옷 다듬기용(outputs/item_attributes/seed42)
- 확인: 레퍼런스 나머지 반 + K-Fashion 옷 확인용(seed7_ex42)
- 목표: 다듬기용 AUC 평균의 합. 후보: configs/style_item_prompt_pool.json
- offset: 고른 문장으로 다듬기용 K-Fashion 옷의 1위 점수 중앙값이 0.5가 되게 다시 구한다

    python scripts/tune_item_reference.py --out outputs/reference_eval/styles_ref.yaml
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
    evaluate,
    load_reference,
    split_mask,
)
from scripts.tune_style_prompts import encode_texts


def reference_items(rows, cache: Path):
    """레퍼런스 사진의 옷 임베딩과 사진 번호. 캐시가 있으면 다시 쓴다."""
    if cache.exists():
        d = np.load(cache)
        return d["emb"], d["photo"]
    from PIL import Image

    from app.pipeline import detection, tagging
    from app.pipeline.analyze import JobError, find_regions
    from app.schemas import AnalyzeRequest

    model, processor = tagging.load_tagger()
    embs, photo = [], []
    req = AnalyzeRequest(job_id="x", image_url="x", callback_url="x",
                         items=[{"client_item_key": f"k{i}", "upload_url": "m"} for i in range(6)])
    for i, r in enumerate(rows):
        img = Image.open(REF / "images" / r["file"]).convert("RGB")
        try:
            regions = find_regions(img, req, detection.detect)
        except JobError as e:  # 옷을 못 찾은 사진(콜라주 등)은 건너뛴다
            print(f"건너뜀 {r['file']}: {e}")
            continue
        for reg in regions:
            embs.append(tagging._encode_image(model, processor, tagging.crop_item(img, reg.bbox, reg.mask))[0])
            photo.append(i)
        print(f"{i + 1}/{len(rows)}", flush=True)
    emb, photo = np.stack(embs), np.array(photo)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, emb=emb, photo=photo)
    return emb, photo


def kf_items(d: Path, mapping):
    meta = [m for m in json.loads((d / "items_meta.json").read_text(encoding="utf-8")) if m["matched"]]
    emb = np.load(d / "cache.npz")["emb"][[m["emb"] for m in meta]]
    styles = np.array([m["style"] for m in meta])
    truth = np.stack([np.isin(styles, mapping[k]) for k in KEYS], 1)
    return emb, truth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--styles", type=Path, default=ROOT / "configs/styles.yaml")
    ap.add_argument("--pool", type=Path, default=ROOT / "configs/style_item_prompt_pool.json")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=ROOT / "outputs/reference_eval/item_tune_result.json")
    args = ap.parse_args()

    styles_cfg = yaml.safe_load(args.styles.read_text(encoding="utf-8"))
    by_key = {s["key"]: s for s in styles_cfg["styles"]}
    a7 = {a["key"]: a["kfashion"]["styles"] for a in yaml.safe_load((ROOT / "configs/aesthetics_7.yaml").read_text(encoding="utf-8"))["aesthetics"]}
    pool = json.loads(args.pool.read_text(encoding="utf-8"))

    rows, rtruth = load_reference()
    r_emb, r_photo = reference_items(rows, ROOT / "outputs/reference_eval/item_cache.npz")
    r_truth = rtruth[r_photo]
    photo_tune = split_mask(rows)
    r_tune = photo_tune[r_photo]
    kt_emb, kt_truth = kf_items(ROOT / "outputs/item_attributes/seed42", a7)
    kc_emb, kc_truth = kf_items(ROOT / "outputs/item_attributes/seed7_ex42", a7)

    texts, ranges = [], []
    for k in KEYS:
        cand = list(dict.fromkeys(by_key[k]["prompts"] + pool.get(k, [])))
        ranges.append(list(range(len(texts), len(texts) + len(cand))))
        texts += cand
    neutral = styles_cfg["independent_score"]["neutral_prompts"]
    scale, t = encode_texts(texts + neutral)
    t_s, t_n = t[: len(texts)], t[len(texts):]
    sims = {"r": scale * r_emb @ t_s.T, "kt": scale * kt_emb @ t_s.T, "kc": scale * kc_emb @ t_s.T}

    def sc(name, ch):
        return np.stack([sims[name][:, idx].mean(1) for idx in ch], 1)

    def objective(ch):
        return evaluate(sc("r", ch), r_truth, r_tune)["auc_mean"] + evaluate(sc("kt", ch), kt_truth)["auc_mean"]

    result = {"n_ref_items": len(r_emb), "n_ref_photos_with_items": len(set(r_photo.tolist()))}

    def report(label, ch):
        out = {"ref_tune": evaluate(sc("r", ch), r_truth, r_tune), "ref_check": evaluate(sc("r", ch), r_truth, ~r_tune),
               "kf_check": evaluate(sc("kc", ch), kc_truth)}
        result[label] = out
        print(f"[{label}] 레퍼런스 다듬기 {out['ref_tune']['hit_mean']:.3f}/{out['ref_tune']['auc_mean']:.3f}  "
              f"레퍼런스 확인 {out['ref_check']['hit_mean']:.3f}/{out['ref_check']['auc_mean']:.3f}  "
              f"K-Fashion 확인 {out['kf_check']['hit_mean']:.3f}/{out['kf_check']['auc_mean']:.3f}")

    choice = [r[: len(by_key[k]["prompts"])] for r, k in zip(ranges, KEYS)]
    print(f"레퍼런스 옷 {len(r_emb)}벌(사진 {result['n_ref_photos_with_items']}장)")
    report("지금 문장", choice)
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

    # offset 다시 구하기(다듬기용 K-Fashion 옷 1위 점수 중앙값 0.5)
    temp = styles_cfg["independent_score"]["temperature"]
    diff = temp * (sc("kt", choice) - (scale * kt_emb @ t_n.T).mean(1, keepdims=True))
    offset = float(-np.median(diff.max(1)))
    result["offset"] = offset
    print(f"offset {offset:.2f}")

    for k, idx in zip(KEYS, choice):
        by_key[k]["prompts"] = [texts[i] for i in idx]
    styles_cfg["independent_score"]["offset"] = round(offset, 2)
    head = "".join(line for line in args.styles.read_text(encoding="utf-8").splitlines(True) if line.startswith("#"))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(head + "# 문장·offset: scripts/tune_item_reference.py (요즘 레퍼런스 옷 + K-Fashion 옷, 11/27)\n\n")
        yaml.safe_dump(styles_cfg, f, allow_unicode=True, sort_keys=False, width=200)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

"""옷 한 벌 단위 정확도: 대분류·소분류·숨은 속성·스타일 점수 (#99, K-Fashion 정답 라벨).

K-Fashion 착용 사진에서 YOLO(DeepFashion2)로 옷을 찾고, 정답 박스와 짝지어진 옷만 잘라
(마스크 밖 흰색) 서비스 태깅 코드(app/pipeline/tagging.py)의 판정을 K-Fashion 라벨과 비교한다.
바닥에 펼친 사진(T1)이 아니라 입은 사진이라 서비스 입력과 다르다. 사진이 오면 같은 지표로 다시 잰다.

    python scripts/eval_item_attributes_kfashion.py --per-style 20 --seed 42

결과: outputs/item_attributes/seed<seed>/summary.json, items.csv
옷 임베딩은 outputs/item_attributes/seed<seed>/cache.npz에 저장해 다시 쓴다.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_category_kfashion import crop
from scripts.eval_kfashion_styles import ROOT, open_images
from scripts.eval_segmentation_kfashion import (
    detect,
    load_image,
    match_boxes,
    pick_samples,
)
from scripts.eval_style_prompts_cached import auc

PART_TO_CATEGORY = {"상의": "top", "하의": "bottom", "아우터": "outerwear", "원피스": "dress"}
DF2_TO_CATEGORY = {
    "short_sleeved_shirt": "top", "long_sleeved_shirt": "top", "vest": "top", "sling": "top",
    "short_sleeved_outwear": "outerwear", "long_sleeved_outwear": "outerwear",
    "shorts": "bottom", "trousers": "bottom", "skirt": "bottom",
    "short_sleeved_dress": "dress", "long_sleeved_dress": "dress", "vest_dress": "dress", "sling_dress": "dress",
}

# K-Fashion 라벨 → 우리 값. 없는 값(None)은 정답이 없는 것으로 보고 뺀다.
SLEEVE = {"반팔": "short", "캡": "short", "긴팔": "long", "7부소매": "long"}
FIT = {
    "top": {"오버사이즈": "oversized", "루즈": "oversized", "노멀": "regular", "타이트": "slim"},
    "outerwear": {"오버사이즈": "oversized", "루즈": "oversized", "노멀": "regular", "타이트": "slim"},
    "dress": {"오버사이즈": "oversized", "루즈": "oversized", "노멀": "regular", "타이트": "slim"},
    "bottom": {"와이드": "oversized", "벨보텀": "oversized", "루즈": "oversized", "노멀": "regular", "스키니": "slim"},
}
MATERIAL = {
    "데님": "denim", "니트": "knit", "헤어 니트": "knit", "저지": "cotton_jersey", "가죽": "leather",
    "스웨이드": "suede", "퍼": "fur", "무스탕": "fur", "울/캐시미어": "wool_tweed", "트위드": "wool_tweed",
    "린넨": "linen", "시폰": "chiffon_satin", "실크": "chiffon_satin", "새틴": "chiffon_satin",
    "패딩": "nylon_padding",
}
PATTERN = {
    "무지": "solid", "스트라이프": "stripe", "체크": "check", "깅엄": "check", "하운즈\xa0투스": "check",
    "아가일": "check", "도트": "dot", "플로럴": "floral", "호피": "animal", "지브라": "animal",
    "뱀피": "animal", "그래픽": "graphic", "레터링": "graphic",
}
DETAIL = {"리본": "ribbon", "프릴": "frill", "러플": "frill", "레이스": "lace", "스터드": "studs",
          "프린지": "fringe", "퍼트리밍": "fur_trim"}
# K-Fashion 카테고리 → 우리 소분류(대응이 하나로 정해지는 것만)
SUBCATEGORY = {"블라우스": "blouse", "티셔츠": "tshirt", "니트웨어": "knit", "셔츠": "shirt", "후드티": "hoodie",
               "청바지": "jeans", "가디건": "cardigan", "코트": "coat", "패딩": "padding"}


def as_list(v):
    return v if isinstance(v, list) else ([] if v in (None, "") else [v])


def item_truth(label: dict, category: str) -> dict:
    t = {"kf_category": label.get("카테고리")}
    t["sleeve"] = SLEEVE.get(label.get("소매기장"))
    t["fit"] = FIT[category].get(label.get("핏"))
    neck = label.get("넥라인")
    t["off_shoulder"] = None if not neck else neck in ("오프숄더", "원숄더")
    # 우븐(일반 직물)·레이스·메시 등 우리 목록에 없는 소재는 "other"(평범한 원단)로 본다
    mats = {MATERIAL.get(m, "other") for m in as_list(label.get("소재"))}
    t["material"] = sorted(mats)  # 여러 개일 수 있다. 판정이 그중 하나면 맞음
    t["material_labeled"] = bool(as_list(label.get("소재")))
    prints = as_list(label.get("프린트"))
    t["pattern"] = sorted({PATTERN.get(p, "other") for p in prints}) if prints else []
    t["details_labeled"] = "디테일" in label
    t["details"] = sorted({DETAIL[d] for d in as_list(label.get("디테일")) if d in DETAIL})
    t["subcategory"] = SUBCATEGORY.get(label.get("카테고리"))
    if category == "bottom" and label.get("카테고리") == "스커트":
        t["subcategory"] = {"미니": "mini_skirt", "롱": "long_skirt", "맥시": "long_skirt"}.get(label.get("기장"))
    if category == "dress" and label.get("카테고리") == "드레스":
        t["subcategory"] = {"미니": "mini_dress", "니렝스": "mini_dress", "미디": "long_dress", "롱": "long_dress", "맥시": "long_dress"}.get(label.get("기장"))
    return t


def collect(args):
    """옷 목록과 임베딩. 캐시가 있으면 다시 쓴다."""
    name = f"seed{args.seed}" + (f"_ex{args.exclude_seed}" if args.exclude_seed is not None else "")
    name += f"_neck{args.neckline_test}" if args.neckline_test else ""
    out_dir = ROOT / f"outputs/item_attributes/{name}"
    cache = out_dir / "cache.npz"
    meta_path = out_dir / "items_meta.json"
    if cache.exists() and meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8")), np.load(cache)["emb"], out_dir

    import torch

    from app.pipeline import tagging

    exclude = set()
    if args.exclude_seed is not None:
        ex_meta = ROOT / f"outputs/item_attributes/seed{args.exclude_seed}/items_meta.json"
        exclude = {m["image_id"] for m in json.loads(ex_meta.read_text(encoding="utf-8"))}
    include, n, per_style = None, None, args.per_style
    if args.neckline_test:
        pos, neg = neckline_ids(args.data_dir)
        rng = np.random.default_rng(args.seed)
        include = set(rng.choice(sorted(pos), args.neckline_test, replace=False).tolist())
        include |= set(rng.choice(sorted(neg), args.neckline_test, replace=False).tolist())
        n, per_style = len(include), None
    images, samples = pick_samples(
        args.data_dir, n, args.seed, min_parts=1, per_style=per_style, exclude_ids=exclude,
        include_ids=include,
    )
    detections, _ = detect(args.detector, images, samples, "cpu", 0.25)
    model, processor = tagging.load_tagger()
    labels_by_id = load_part_labels(args.data_dir, {s["record"].image_id for s in samples})
    meta, embs = [], []
    with open_images(images) as archive, torch.no_grad():
        for sample, det in zip(samples, detections):
            if det is None:
                continue
            image = load_image(archive, sample)
            truth_boxes = [p["box"] for p in sample["parts"]]
            matched = match_boxes(truth_boxes, det["boxes"])
            labels = labels_by_id.get(sample["record"].image_id, {})
            for part, j in zip(sample["parts"], matched):
                cat = PART_TO_CATEGORY[part["part"]]
                row = {"image_id": sample["record"].image_id, "style": sample["record"].main, "part": part["part"],
                       "category": cat, "matched": j is not None}
                if j is None:
                    meta.append(row)
                    continue
                df2 = det["df2_classes"][j]
                row |= {"pred_category": DF2_TO_CATEGORY[df2], "df2": df2,
                        **item_truth(labels.get(part["part"], {}), cat)}
                img = crop(image, det["boxes"][j], det["masks"][j] if det["masks"] else None)
                embs.append(tagging._encode_image(model, processor, img)[0])
                row["emb"] = len(embs) - 1
                meta.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = np.stack(embs)
    np.savez(cache, emb=emb)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta, emb, out_dir


def neckline_ids(data_dir):
    """상의·원피스 넥라인이 오프숄더·원숄더인 사진(pos)과, 넥라인 라벨은 있지만 아닌 사진(neg)."""
    import zipfile

    from scripts.eval_kfashion_styles import ZIP_NAME_ENCODING, find_zip

    pos, neg = set(), set()
    with zipfile.ZipFile(find_zip(data_dir / "labels"), metadata_encoding=ZIP_NAME_ENCODING) as z:
        for info in z.infolist():
            if not info.filename.endswith(".json"):
                continue
            doc = json.loads(z.read(info).decode("utf-8"))
            lab = doc["데이터셋 정보"]["데이터셋 상세설명"]["라벨링"]
            necks = [(lab.get(p) or [{}])[0].get("넥라인") for p in ("상의", "원피스")]
            necks = [x for x in necks if x]
            if not necks:
                continue
            iid = doc["이미지 정보"]["이미지 식별자"]
            (pos if any(x in ("오프숄더", "원숄더") for x in necks) else neg).add(iid)
    return pos, neg


def load_part_labels(data_dir, ids):
    import zipfile

    from scripts.eval_kfashion_styles import ZIP_NAME_ENCODING, find_zip

    out = {}
    with zipfile.ZipFile(find_zip(data_dir / "labels"), metadata_encoding=ZIP_NAME_ENCODING) as z:
        for info in z.infolist():
            if not info.filename.endswith(".json"):
                continue
            doc = json.loads(z.read(info).decode("utf-8"))
            iid = doc["이미지 정보"]["이미지 식별자"]
            if iid in ids:
                lab = doc["데이터셋 정보"]["데이터셋 상세설명"]["라벨링"]
                out[iid] = {p: (lab.get(p) or [{}])[0] for p in PART_TO_CATEGORY}
    return out


def rate(pairs):
    pairs = [(p, t) for p, t in pairs if t is not None]
    return {"n": len(pairs), "acc": (sum(p == t for p, t in pairs) / len(pairs)) if pairs else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/kfashion/validation")
    ap.add_argument("--detector", type=Path, default=ROOT / "weights/deepfashion2_yolov8s-seg.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-style", type=int, default=20)
    ap.add_argument("--exclude-seed", type=int, help="이 시드 표본의 사진을 빼고 뽑는다(확인용)")
    ap.add_argument(
        "--neckline-test", type=int,
        help="오프숄더 재현율용: 상의·원피스 넥라인이 오프숄더·원숄더인 사진 N장과 아닌 사진 N장만 뽑는다",
    )
    ap.add_argument("--name-prompts", action="store_true", help="소분류를 이름 문장(예전 방식)으로 판정해 비교")
    ap.add_argument("--style-configs", nargs="+", default=["configs/aesthetics.yaml", "configs/styles.yaml"])
    args = ap.parse_args()

    from app.pipeline import tagging

    meta, emb, out_dir = collect(args)
    # 정답은 매번 라벨에서 다시 만든다(매핑을 고쳐도 캐시를 다시 만들 필요가 없게)
    labels = load_part_labels(args.data_dir, {m["image_id"] for m in meta})
    for m in meta:
        if m["matched"]:
            m.update(item_truth(labels.get(m["image_id"], {}).get(m["part"], {}), m["category"]))
    model, _ = tagging.load_tagger()
    scale = float(model.logit_scale.detach().exp())
    _, categories, attr_cfg = tagging._load_configs()
    rep: dict = {"seed": args.seed, "per_style": args.per_style}

    parts = [m for m in meta]
    matched = [m for m in parts if m["matched"]]
    rep["detection"] = {"truth_items": len(parts), "matched": len(matched), "match_rate": len(matched) / len(parts)}
    rep["category"] = {"n": len(matched), "acc": sum(m["pred_category"] == m["category"] for m in matched) / len(matched),
                       "confusion": Counter(f"{m['category']}→{m['pred_category']}" for m in matched if m["pred_category"] != m["category"]).most_common(6)}

    rows, sub_top1, sub_top3 = [], [], []
    attr_pairs = defaultdict(list)
    for m in matched:
        sc = tagging._Scorer(emb[m["emb"]][None, :], scale)
        cat = m["pred_category"]
        sub_cfg = {} if args.name_prompts else attr_cfg.get("subcategory", {})
        cands = tagging._category_candidates(sc, categories, cat, sub_cfg.get("prompts"), sub_cfg.get("ai_skip", ()))
        a = tagging._attributes(sc, attr_cfg, cat, cands[0]["subcategory"])
        sleeve = {"short_sleeved_shirt": "short", "long_sleeved_shirt": "long", "short_sleeved_outwear": "short",
                  "long_sleeved_outwear": "long", "short_sleeved_dress": "short", "long_sleeved_dress": "long"}.get(m["df2"])
        final_sleeve = sleeve or a["sleeve"]
        if cat == m["category"] and m["subcategory"]:
            subs = [c["subcategory"] for c in cands]
            sub_top1.append(subs[0] == m["subcategory"])
            sub_top3.append(m["subcategory"] in subs)
        attr_pairs["sleeve"].append((final_sleeve, m["sleeve"], a["sleeve"]))
        attr_pairs["fit"].append((a["fit"], m["fit"]))
        attr_pairs["off_shoulder"].append((a["off_shoulder"], m["off_shoulder"]))
        if m["material_labeled"]:
            attr_pairs["material"].append((a["material"], m["material"]))
        if m["pattern"]:
            attr_pairs["pattern"].append((a["pattern"], m["pattern"]))
        rows.append({**{k: v for k, v in m.items() if k != "emb"}, "pred_sub": cands[0]["subcategory"],
                     "pred_sleeve": final_sleeve, "pred_fit": a["fit"], "pred_off": a["off_shoulder"],
                     "pred_material": a["material"], "pred_pattern": a["pattern"]})

    rep["subcategory"] = {"n": len(sub_top1), "top1": float(np.mean(sub_top1)) if sub_top1 else None,
                          "top3": float(np.mean(sub_top3)) if sub_top3 else None}

    def judged(pairs):
        """판정한 것(None 아님)의 정확도와 판정 비율."""
        have = [(p, t) for p, t, *_ in pairs if t is not None]
        dec = [(p, t) for p, t in have if p is not None]
        hit = sum((p in t) if isinstance(t, list) else (p == t) for p, t in dec)
        return {"labeled": len(have), "decided": len(dec), "coverage": len(dec) / len(have) if have else None,
                "acc_when_decided": hit / len(dec) if dec else None}

    rep["attributes"] = {k: judged(v) for k, v in attr_pairs.items()}
    rep["attributes"]["sleeve_tagger_only"] = judged([(t_only, t) for _, t, t_only in attr_pairs["sleeve"]])
    rep["attributes"]["off_shoulder"]["truth_true"] = sum(1 for _, t in attr_pairs["off_shoulder"] if t)
    pred_true = [t for p, t in attr_pairs["off_shoulder"] if p and t is not None]
    rep["attributes"]["off_shoulder"]["precision_true"] = (sum(pred_true) / len(pred_true)) if pred_true else None
    rep["attributes"]["off_shoulder"]["recall_true"] = (
        sum(1 for p, t in attr_pairs["off_shoulder"] if t and p) / max(1, sum(1 for _, t in attr_pairs["off_shoulder"] if t))
    )

    # 오프숄더 기준값별 정밀도·재현율(상의·원피스, 넥라인 라벨이 있는 옷)
    off_cfg = attr_cfg["off_shoulder"]
    off_items = [m for m in matched if m["pred_category"] in off_cfg["categories"] and m["off_shoulder"] is not None]
    if off_items:
        probs = np.array([tagging._Scorer(emb[m["emb"]][None, :], scale).yes_prob(off_cfg["prompts"]) for m in off_items])
        truth = np.array([m["off_shoulder"] for m in off_items])
        table = {"auc": auc(probs[truth], probs[~truth]) if truth.any() and (~truth).any() else None}
        for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            pred = probs >= th
            tp = int((pred & truth).sum())
            table[f"th{th}"] = {"precision": tp / pred.sum() if pred.sum() else None,
                                "recall": tp / truth.sum() if truth.sum() else None}
        rep["attributes"]["off_shoulder_thresholds"] = table

    # 디테일: 디테일 라벨이 있는 옷만, 디테일마다 AUC와 기준값별 정밀도·재현율
    det_cfg = attr_cfg["details"]
    det_items = [m for m in matched if m.get("details_labeled")]
    rep["details"] = {}
    for d, pair in det_cfg["prompts"].items():
        probs = np.array([tagging._Scorer(emb[m["emb"]][None, :], scale).yes_prob(pair) for m in det_items])
        truth = np.array([d in m["details"] for m in det_items])
        entry = {"positives": int(truth.sum()), "n": len(det_items),
                 "auc": auc(probs[truth], probs[~truth]) if truth.any() and (~truth).any() else None}
        for th in (0.5, 0.7, 0.8, 0.9):
            pred = probs >= th
            tp = int((pred & truth).sum())
            entry[f"th{th}"] = {"precision": tp / pred.sum() if pred.sum() else None, "recall": tp / truth.sum() if truth.sum() else None}
        rep["details"][d] = entry

    # 스타일 점수(옷 한 벌): 사진의 주 스타일이 연결 스타일인 옷을 양성으로 본 AUC와 1위 적중률
    rep["styles"] = {}
    for cfg_path in args.style_configs:
        cfg = yaml.safe_load((ROOT / cfg_path).read_text(encoding="utf-8"))
        aes = cfg.get("aesthetics") or cfg["styles"]
        mapping = {a["key"]: a.get("kfashion", {}).get("styles") for a in aes}
        if any(v is None for v in mapping.values()):  # styles.yaml에는 연결이 없어 aesthetics_7.yaml에서 가져온다
            a7 = {a["key"]: a["kfashion"]["styles"] for a in yaml.safe_load((ROOT / "configs/aesthetics_7.yaml").read_text(encoding="utf-8"))["aesthetics"]}
            mapping = {k: v or a7[k] for k, v in mapping.items()}
        prompts = [(a["prompts"]["item"] if isinstance(a["prompts"], dict) else a["prompts"]) for a in aes]
        logits = np.stack([np.stack([tagging._Scorer(emb[m["emb"]][None, :], scale).logits(p).mean() for m in matched]) for p in prompts], axis=1)
        styles_arr = np.array([m["style"] for m in matched])
        top = logits.argmax(axis=1)
        per = []
        for k, a in enumerate(aes):
            mask = np.isin(styles_arr, mapping[a["key"]])
            per.append({"key": a["key"], "n": int(mask.sum()), "hit": float((top[mask] == k).mean()) if mask.any() else None,
                        "auc": auc(logits[mask, k], logits[~mask, k]) if mask.any() else None, "top1": int((top == k).sum())})
        rep["styles"][cfg_path] = {"per": per, "hit_mean": float(np.mean([p["hit"] for p in per])),
                                   "auc_mean": float(np.mean([p["auc"] for p in per])), "chance": 1 / len(aes)}

    (out_dir / "summary.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    with open(out_dir / "items.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()

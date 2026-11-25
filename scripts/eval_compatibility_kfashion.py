"""상의·하의 어울림 점수(compatibility) 실험 (T25, issue #91).

K-Fashion 착용 사진은 실제로 함께 입은 코디다. 같은 사진의 상의·하의를 positive,
다른 사진의 하의를 붙인 쌍을 negative로 두고, FashionCLIP 이미지 임베딩 위에서
(a) cosine 유사도(학습 없음)와 (b) 로지스틱 회귀([a, b, a*b, |a-b|])를 비교한다.

- 표본: 라벨 zip + 사진(zip 또는 풀어 둔 jpg). 바이트가 같은 사진은 1장만
- 크롭: 라벨 폴리곤 마스크 밖을 흰색으로 지운 박스 크롭(app.pipeline.tagging.crop_item, 서비스 입력과 같은 형식)
- 분할: 쌍을 만들기 전에 이미지 단위로, 거의 같은 사진(같은 인물·촬영 추정) 묶음은 한쪽에만
- 평가: AUC, FITB@10(정답 하의 1 + 다른 하의 9 중 1위), random / hard(같은 K-Fashion 주 스타일 안의 negative)
- 대조 실험(--mode): graywb·colorwb(옷 픽셀 기준 화이트밸런스 — 옷 색까지 지워짐), gray(흑백),
  color(CLIP 없이 색 히스토그램), colorwbimg(사진 전체 기준 화이트밸런스 뒤 색 히스토그램)로
  "같은 사진" 지름길이 얼마나 되는지 본다. 분할은 raw 임베딩 기준 묶음으로 모드끼리 같다.
- 재배포 금지: 임베딩·결과는 outputs/compatibility/에만 저장(gitignore). Report에는 집계 숫자만.

    python scripts/eval_compatibility_kfashion.py --data-dir <K-Fashion validation 폴더> [--detector <yolo 가중치>]
"""

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_kfashion_styles import ROOT, ZIP_NAME_ENCODING

TOP, BOTTOM, OUTER = "상의", "하의", "아우터"


# ── 순수 함수 (tests/test_eval_compatibility.py) ──────────────────────────


def near_dup_groups(
    top: np.ndarray, bottom: np.ndarray, threshold: float
) -> np.ndarray:
    """상의 cosine과 하의 cosine이 둘 다 threshold 이상인 사진끼리 묶는다(union-find). 묶음 id 배열."""
    n = len(top)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    both = (top @ top.T >= threshold) & (bottom @ bottom.T >= threshold)
    for i, j in zip(*np.nonzero(np.triu(both, 1))):
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])


def group_split(
    groups: np.ndarray, test_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """묶음 단위로 train/test 인덱스를 나눈다."""
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    n_test = max(1, round(len(uniq) * test_ratio))
    test_groups = set(uniq[:n_test].tolist())
    is_test = np.array([g in test_groups for g in groups])
    return np.nonzero(~is_test)[0], np.nonzero(is_test)[0]


def pair_features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([a, b, a * b, np.abs(a - b)], axis=1)


def sample_negatives(
    idx: np.ndarray, k: int, rng, styles=None
) -> list[tuple[int, int]]:
    """idx 안에서 (상의 i, 다른 사진 j의 하의) 쌍을 i마다 k개. styles를 주면 같은 스타일 안에서만(hard)."""
    pairs = []
    for i in idx:
        pool = idx[idx != i]
        if styles is not None:
            pool = pool[styles[pool] == styles[i]]
        if len(pool) == 0:
            continue
        for j in rng.choice(pool, size=min(k, len(pool)), replace=False):
            pairs.append((int(i), int(j)))
    return pairs


def fitb_accuracy(
    score_fn, idx: np.ndarray, k: int, rng, styles=None
) -> tuple[float, int]:
    """상의마다 정답 하의 1 + 다른 사진 하의 k개 중 정답이 1위인 비율. (정확도, 문항 수)."""
    hits = total = 0
    for i in idx:
        pool = idx[idx != i]
        if styles is not None:
            pool = pool[styles[pool] == styles[i]]
        if len(pool) < k:
            continue
        cands = np.concatenate([[i], rng.choice(pool, size=k, replace=False)])
        scores = score_fn(np.full(len(cands), i), cands)
        hits += int(np.argmax(scores) == 0)
        total += 1
    return (hits / total if total else float("nan")), total


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(roc_auc_score(y, np.concatenate([pos, neg])))


# ── 데이터·임베딩 ────────────────────────────────────────────────────────


def pick_top_bottom(data_dir: Path, n: int, seed: int) -> list[dict]:
    """상의·하의 박스와 폴리곤이 모두 있는 사진 n장. 바이트가 같은 사진은 1장만.

    라벨은 labels/ 아래 zip, 사진은 images/ 아래 zip 또는 풀어 둔 jpg를 읽는다.
    """
    import hashlib
    import random

    from scripts.eval_kfashion_styles import find_zip
    from scripts.eval_segmentation_kfashion import parts_from_label

    image_zips = sorted((data_dir / "images").rglob("*.zip"))
    if image_zips:
        with zipfile.ZipFile(image_zips[0], metadata_encoding=ZIP_NAME_ENCODING) as z:
            source = {
                Path(i.filename).stem: (image_zips[0], i.filename)
                for i in z.infolist()
                if not i.is_dir()
            }
    else:
        source = {f.stem: (None, f) for f in (data_dir / "images").rglob("*.jpg")}
    print(f"사진 {len(source)}장 찾음")

    labels_zip = find_zip(data_dir / "labels")
    with zipfile.ZipFile(labels_zip, metadata_encoding=ZIP_NAME_ENCODING) as labels:
        members = sorted(
            (i for i in labels.infolist() if i.filename.endswith(".json")),
            key=lambda i: i.filename,
        )
        random.Random(seed).shuffle(members)
        samples, seen = [], set()
        for info in members:
            if len(samples) == n:
                break
            stem = Path(info.filename).stem
            if stem not in source:
                continue
            doc = json.loads(labels.read(info).decode("utf-8"))
            parts = parts_from_label(doc)
            if not {p["part"] for p in parts} >= {TOP, BOTTOM}:
                continue
            digest = hashlib.sha1(_read_image_bytes(source[stem])).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            style = doc["데이터셋 정보"]["데이터셋 상세설명"]["라벨링"]["스타일"][0]
            info_ = doc["이미지 정보"]
            samples.append(
                {
                    "image_id": info_["이미지 식별자"],
                    "style": style.get("스타일"),
                    "source": source[stem],
                    "parts": parts,
                    "size": (info_["이미지 너비"], info_["이미지 높이"]),
                }
            )
    return samples


def _read_image_bytes(source) -> bytes:
    zip_path, member = source
    if zip_path is None:
        return Path(member).read_bytes()
    with zipfile.ZipFile(zip_path, metadata_encoding=ZIP_NAME_ENCODING) as z:
        return z.read(member)


def _load(sample):
    """사진을 읽는다. 크기가 라벨과 다르면 좌표를 믿을 수 없으므로 None."""
    import io

    from PIL import Image

    raw = _read_image_bytes(sample["source"])
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return image if image.size == sample["size"] else None


def embed_batch(model, processor, crops) -> np.ndarray:
    import torch

    from app.pipeline.tagging import _features, _get_device

    with torch.no_grad():
        inputs = processor(images=crops, return_tensors="pt").to(_get_device())
        feats = _features(model.get_image_features(**inputs)).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


# ── 대조 실험: "같은 사진"(조명·색감) 지름길 확인 ─────────────────────────
# raw: 원래 크롭 / graywb: 옷 픽셀 기준 gray-world — 조명 색뿐 아니라 옷 자체 색까지 지워짐(너무 강한 대조)
# gray: 흑백(색 정보 제거) / color: CLIP 없이 옷 픽셀 HSV 히스토그램만
# colorwb: 옷 픽셀 기준 gray-world 뒤 HSV 히스토그램 — 옷 자체 색까지 지워져 조명만 빼는 대조로는 너무 강함
# colorwbimg: 사진 전체 기준 gray-world(조명 추정) 뒤 HSV 히스토그램 — 옷끼리의 색 차이는 남음
MODES = ("raw", "graywb", "gray", "color", "colorwb", "colorwbimg")


def gray_world(image, mask: np.ndarray):
    """마스크 안 픽셀의 채널 평균이 같아지도록 채널별로 곱한다(조명 색 보정)."""
    from PIL import Image

    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    m = mask.astype(bool)
    if not m.any():
        return image
    means = arr[m].mean(axis=0)
    scale = means.mean() / np.maximum(means, 1e-6)
    return Image.fromarray(np.clip(arr * scale, 0, 255).astype(np.uint8))


def color_hist(image, mask: np.ndarray, bins=(8, 4, 4)) -> np.ndarray:  # bins [초안]
    """마스크 안 픽셀의 HSV 3D 히스토그램(합 1 뒤 L2 정규화). 마스크가 비면 0벡터."""
    hsv = np.asarray(image.convert("HSV"))[mask.astype(bool)]
    if len(hsv) == 0:
        return np.zeros(int(np.prod(bins)), dtype=np.float32)
    hist, _ = np.histogramdd(hsv, bins=bins, range=[(0, 256)] * 3)
    v = hist.ravel().astype(np.float32)
    v /= v.sum()
    return v / np.linalg.norm(v)


def build_embeddings(samples, cache: Path, mode: str = "raw") -> dict:
    """사진별 상의·하의·아우터 특징. cache에 이미 있는 사진은 건너뛴다."""
    from app.pipeline.tagging import crop_item, load_tagger
    from scripts.eval_segmentation_kfashion import truth_mask

    data = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        data = z["data"].item()
    todo = [s for s in samples if s["image_id"] not in data]
    print(f"[{mode}] 캐시 {len(data)}장, 새로 뽑을 사진 {len(todo)}장")
    if not todo:
        return data

    color_only = mode in ("color", "colorwb", "colorwbimg")
    model, processor = (None, None) if color_only else load_tagger()
    start = time.perf_counter()
    for n, s in enumerate(todo, 1):
        image = _load(s)
        if image is not None:
            parts = [p for p in s["parts"] if p["part"] in (TOP, BOTTOM, OUTER)]
            masks = [truth_mask(p["points"], s["size"]) for p in parts]
            if color_only:
                if mode == "colorwbimg":  # 사진 전체 픽셀로 조명을 한 번 추정
                    full = np.ones(masks[0].shape, dtype=bool)
                    balanced = gray_world(image, full)
                embs = [
                    color_hist(
                        balanced
                        if mode == "colorwbimg"
                        else gray_world(image, m)
                        if mode == "colorwb"
                        else image,
                        m,
                    )
                    for m in masks
                ]
            else:
                crops = []
                for p, m in zip(parts, masks):
                    src = gray_world(image, m) if mode == "graywb" else image
                    crop = crop_item(src, p["box"], m)
                    crops.append(
                        crop.convert("L").convert("RGB") if mode == "gray" else crop
                    )
                embs = embed_batch(model, processor, crops)
            data[s["image_id"]] = {
                "style": s["style"],
                **{p["part"]: e for p, e in zip(parts, embs)},
            }
        if n % 100 == 0 or n == len(todo):
            np.savez(cache, data=np.array(data, dtype=object))
            rate = (time.perf_counter() - start) / n
            print(f"{n}/{len(todo)}장, 장당 {rate:.2f}초", flush=True)
    return data


def detector_embeddings(samples, test_ids: set, weights: Path, limit: int) -> dict:
    """테스트 사진을 YOLO 검출 크롭으로 다시 임베딩한다(서비스 입력과의 차이 확인)."""
    from app.pipeline import detection
    from app.pipeline.tagging import crop_item, load_tagger
    from scripts.eval_segmentation_kfashion import DF2_TO_PART, match_boxes

    detection.load_detector.cache_clear()
    detector = detection.load_detector(weights)
    model, processor = load_tagger()
    out = {}
    for s in samples:
        if len(out) >= limit:
            break
        if s["image_id"] not in test_ids:
            continue
        image = _load(s)
        if image is None:
            continue
        r = detector(image, conf=detection.CONF, retina_masks=True, verbose=False)[0]
        if r.masks is None:
            continue
        dets = detection.to_detections(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().tolist(),
            r.boxes.cls.cpu().tolist(),
            r.masks.data.cpu().numpy(),
            r.names,
            10,
        )
        truth = {p["part"]: p["box"] for p in s["parts"] if p["part"] in (TOP, BOTTOM)}
        parts = list(truth)
        matched = match_boxes([truth[p] for p in parts], [list(d.bbox) for d in dets])
        if any(m is None for m in matched):
            continue  # 상의·하의를 둘 다 찾은 사진만
        crops = [crop_item(image, dets[m].bbox, dets[m].mask) for m in matched]
        embs = embed_batch(model, processor, crops)
        out[s["image_id"]] = {
            **dict(zip(parts, embs)),
            "df2_part_ok": [
                DF2_TO_PART.get(dets[m].df2_class) == p for m, p in zip(matched, parts)
            ],
        }
    return out


# ── 평가 ─────────────────────────────────────────────────────────────────


def evaluate(top, bottom, styles, train, test, seed, neg_ratio, fitb_k):
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    tr_pos = [(int(i), int(i)) for i in train]
    tr_neg = sample_negatives(train, neg_ratio, rng)
    pairs = tr_pos + tr_neg
    x = pair_features(top[[a for a, _ in pairs]], bottom[[b for _, b in pairs]])
    y = np.array([1] * len(tr_pos) + [0] * len(tr_neg))
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(x, y)

    def cos(ti, bi):
        return np.sum(top[ti] * bottom[bi], axis=1)

    def lr_score(ti, bi):
        return lr.predict_proba(pair_features(top[ti], bottom[bi]))[:, 1]

    result = {
        "train_images": len(train),
        "test_images": len(test),
        "train_pairs": len(pairs),
    }
    for neg_kind, sty in (("random", None), ("hard", styles)):
        negs = sample_negatives(test, neg_ratio, np.random.default_rng(seed + 1), sty)
        ni = np.array([a for a, _ in negs])
        nj = np.array([b for _, b in negs])
        for name, fn in (("cosine", cos), ("lr", lr_score)):
            result[f"{name}_auc_{neg_kind}"] = auc(fn(test, test), fn(ni, nj))
            acc, n_q = fitb_accuracy(
                fn, test, fitb_k, np.random.default_rng(seed + 2), sty
            )
            result[f"{name}_fitb_{neg_kind}"] = acc
            result[f"fitb_{neg_kind}_questions"] = n_q
        result[f"test_neg_pairs_{neg_kind}"] = len(negs)
    return result, lr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data/kfashion/validation"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/compatibility"
    )
    # [초안] 표본·임계값은 근거 없는 실험 초안
    parser.add_argument("--images", type=int, default=2000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[42, 7])
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--neg-ratio", type=int, default=4)
    parser.add_argument("--fitb-k", type=int, default=9)
    parser.add_argument("--dup-threshold", type=float, default=0.95)
    parser.add_argument("--detector", type=Path, default=None)
    parser.add_argument("--detector-images", type=int, default=300)
    parser.add_argument("--mode", choices=MODES, default="raw")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    samples = pick_top_bottom(args.data_dir, args.images, args.sample_seed)
    print(f"상의+하의 사진 {len(samples)}장")
    suffix = "" if args.mode == "raw" else f"_{args.mode}"
    data = build_embeddings(
        samples, args.output_dir / f"embeddings{suffix}.npz", args.mode
    )
    embed_seconds = time.perf_counter() - t0

    ids = [s["image_id"] for s in samples if s["image_id"] in data]
    top = np.stack([data[i][TOP] for i in ids])
    bottom = np.stack([data[i][BOTTOM] for i in ids])
    styles = np.array(
        [data[i]["style"] or "(없음)" for i in ids]
    )  # 스타일 라벨 없는 사진
    # 거의 같은 사진 묶음은 모드와 상관없이 원래(raw) CLIP 임베딩으로 정한다(분할을 모드끼리 같게)
    raw = data
    if args.mode != "raw":
        raw_cache = args.output_dir / "embeddings.npz"
        assert raw_cache.exists(), "raw 모드를 먼저 실행해야 한다"
        raw = np.load(raw_cache, allow_pickle=True)["data"].item()
        assert all(i in raw for i in ids)
    groups = near_dup_groups(
        np.stack([raw[i][TOP] for i in ids]),
        np.stack([raw[i][BOTTOM] for i in ids]),
        args.dup_threshold,
    )
    n_groups = len(np.unique(groups))

    summary = {
        "images": len(ids),
        "pairs_top_bottom": len(ids),
        "near_dup_groups": n_groups,
        "images_in_multi_groups": int(sum(np.bincount(groups)[groups] > 1)),
        "styles": {k: int(v) for k, v in zip(*np.unique(styles, return_counts=True))},
        "pick_and_embed_seconds_this_run": round(
            embed_seconds, 1
        ),  # 캐시 사용 시 추출 시간 제외
        "chance": {"auc": 0.5, "fitb": 1 / (args.fitb_k + 1)},
        "splits": {},
    }

    last = None
    for seed in args.split_seeds:
        train, test = group_split(groups, args.test_ratio, seed)
        assert not set(train) & set(test)
        assert not set(groups[train]) & set(groups[test]), (
            "near-dup 묶음이 train/test에 걸침"
        )
        res, lr = evaluate(
            top, bottom, styles, train, test, seed, args.neg_ratio, args.fitb_k
        )
        summary["splits"][str(seed)] = res
        last = (train, test, lr)
        print(json.dumps({"split_seed": seed, **res}, ensure_ascii=False))

    # 아우터 쌍: cosine만 (표본이 적어 학습하지 않음)
    outer_ids = [i for i in ids if OUTER in data[i]]
    if len(outer_ids) >= 20:
        rng = np.random.default_rng(0)
        o = np.stack([data[i][OUTER] for i in outer_ids])
        for part in (TOP, BOTTOM):
            p = np.stack([data[i][part] for i in outer_ids])
            idx = np.arange(len(outer_ids))
            negs = sample_negatives(idx, args.neg_ratio, rng)
            pos = np.sum(o * p, axis=1)
            neg = np.array([o[a] @ p[b] for a, b in negs])
            summary[f"outer_{part}_cosine_auc"] = auc(pos, neg)
        summary["outer_images"] = len(outer_ids)

    # 서비스 입력(YOLO 검출 크롭)으로 바꿨을 때: 마지막 분할의 test 사진
    if args.detector and last and args.mode == "raw":
        train, test, lr = last
        test_ids = {ids[i] for i in test}
        det = detector_embeddings(
            samples, test_ids, args.detector, args.detector_images
        )
        d_ids = [i for i in ids if i in det]
        if len(d_ids) >= 20:
            dt = np.stack([det[i][TOP] for i in d_ids])
            db = np.stack([det[i][BOTTOM] for i in d_ids])
            idx = np.arange(len(d_ids))
            negs = sample_negatives(idx, args.neg_ratio, np.random.default_rng(1))
            ni, nj = np.array([a for a, _ in negs]), np.array([b for _, b in negs])
            label_pos = {ids[i]: i for i in test}
            lt = top[[label_pos[i] for i in d_ids]]
            lb = bottom[[label_pos[i] for i in d_ids]]
            summary["detector"] = {
                "images": len(d_ids),
                "part_ok_ratio": float(
                    np.mean([ok for i in d_ids for ok in det[i]["df2_part_ok"]])
                ),
                "cosine_auc_random": auc(
                    np.sum(dt * db, 1), np.sum(dt[ni] * db[nj], 1)
                ),
                "lr_auc_random": auc(
                    lr.predict_proba(pair_features(dt, db))[:, 1],
                    lr.predict_proba(pair_features(dt[ni], db[nj]))[:, 1],
                ),
                "label_crop_lr_auc_random_same_images": auc(
                    lr.predict_proba(pair_features(lt, lb))[:, 1],
                    lr.predict_proba(pair_features(lt[ni], lb[nj]))[:, 1],
                ),
                "crop_cosine_label_vs_detector_mean": float(
                    np.mean(np.concatenate([np.sum(lt * dt, 1), np.sum(lb * db, 1)]))
                ),
            }
            print(json.dumps({"detector": summary["detector"]}, ensure_ascii=False))

    summary["mode"] = args.mode
    out = args.output_dir / f"summary{suffix}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()

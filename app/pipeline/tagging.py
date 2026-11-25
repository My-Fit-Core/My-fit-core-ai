"""FashionCLIP zero-shot 태깅 v1 (F-203, issue #117).

옷 한 벌(흰 배경 박스 크롭)에서 다음을 낸다. 학습 없이 문장 비교로만 판정한다.
- 소분류 후보 상위 3 (configs/categories.yaml)
- 스타일 점수: 스타일마다 따로 0~1, 합 1 아님 (configs/styles.yaml)
- 패턴, 숨은 속성 fit·off_shoulder·sleeve·material·details (configs/attributes.yaml의 tagging)
- 임베딩 (512차원, L2 정규화)
판정이 불확실한 속성은 None(모름)으로 둔다. 기준값은 초안이다.
"""

import functools
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

# 이 저장소는 transformers 형식이다(open_clip hf-hub 설정 파일 없음, #87). 실험 스크립트와 같은 로더를 쓴다
MODEL_NAME = "patrickjohncyh/fashion-clip"
MODEL_VERSION = "fashion-clip-vitb32"


def _read(name: str) -> dict:
    with open(ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.cache
def _load_configs() -> tuple[dict, dict, dict]:
    """(styles.yaml, categories, attributes.yaml의 tagging) — 파일 변경 시 프로세스 재시작."""
    return _read("styles.yaml"), _read("categories.yaml")["categories"], _read("attributes.yaml")["tagging"]


@functools.cache
def _get_device() -> str:
    """MPS(Apple Silicon) → CUDA → CPU 순으로 선택."""
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@functools.cache
def load_tagger():
    """프로세스당 1회 로드. 서버 기동 시 미리 호출하는 연결은 T12."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(MODEL_NAME).to(_get_device()).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def _features(out):
    """transformers 버전에 따라 get_*_features가 텐서 또는 출력 객체를 돌려준다."""
    return getattr(out, "pooler_output", out)


def _encode_texts(model, processor, texts: list[str]) -> np.ndarray:
    """L2 정규화된 텍스트 임베딩, shape (N, D)."""
    import torch

    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        inputs = inputs.to(_get_device())
        feats = _features(model.get_text_features(**inputs)).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def _encode_image(model, processor, image: Image.Image) -> np.ndarray:
    """L2 정규화된 이미지 임베딩, shape (1, D)."""
    import torch

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(_get_device())
        feat = _features(model.get_image_features(**inputs)).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()


def _softmax(arr: np.ndarray) -> np.ndarray:
    exp = np.exp(arr - arr.max())
    return exp / exp.sum()


def crop_item(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    mask: np.ndarray,
) -> Image.Image:
    """마스크 밖을 흰색으로 채우고 bbox(픽셀 xyxy)로 크롭한다."""
    arr = np.asarray(image.convert("RGB")).copy()
    arr[~mask.astype(bool)] = 255
    x1, y1, x2, y2 = (round(v) for v in bbox)
    return Image.fromarray(arr).crop((x1, y1, x2, y2))


@functools.cache
def _text_feats(texts: tuple[str, ...]) -> np.ndarray:
    """문장 임베딩은 옷마다 같으므로 한 번만 계산한다."""
    model, processor = load_tagger()
    return _encode_texts(model, processor, list(texts))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class TagResult:
    category: str
    subcategory: str | None
    category_candidates: list[dict]  # [{category, subcategory, score}, ...] 상위 3
    style_scores: list[dict]  # [{style_id, score}], 스타일마다 0~1, 합 1 아님
    pattern: str | None
    fit: str | None
    off_shoulder: bool | None
    sleeve: str | None
    material: str | None
    details: list[str]
    embedding: list[float]  # FashionCLIP 이미지 임베딩
    model_version: str = field(default=MODEL_VERSION)


class _Scorer:
    """이미지 하나에 대한 문장별 logit(= logit_scale × 코사인 유사도)."""

    def __init__(self, img_feat: np.ndarray, scale: float):
        self.img = img_feat
        self.scale = scale

    def logits(self, texts: list[str]) -> np.ndarray:
        return self.scale * (self.img @ _text_feats(tuple(texts)).T).reshape(-1)

    def choose(self, prompts: dict[str, str], min_prob: float) -> str | None:
        """후보 중 1위. 1위 확률이 min_prob보다 낮으면 None."""
        probs = _softmax(self.logits(list(prompts.values())))
        i = int(np.argmax(probs))
        return list(prompts)[i] if probs[i] >= min_prob else None

    def yes_prob(self, pair: list[str]) -> float:
        """[예, 아니오] 문장 쌍에서 '예' 확률."""
        return float(_softmax(self.logits(pair))[0])


def _style_scores(sc: _Scorer, styles_cfg: dict) -> list[dict]:
    ind = styles_cfg["independent_score"]
    neutral = float(sc.logits(ind["neutral_prompts"]).mean())
    return [
        {
            "style_id": s["style_id"],
            "score": _sigmoid(
                ind["temperature"] * (float(sc.logits(s["prompts"]).mean()) - neutral)
                + ind.get("offset", 0.0)
            ),
        }
        for s in styles_cfg["styles"]
    ]


def _category_candidates(
    sc: _Scorer, categories: dict, only: str | None, prompts: dict | None = None
) -> list[dict]:
    """소분류 후보 상위 3. 소분류마다 판정 문장(여러 개면 평균 logit)으로 점수를 낸다."""
    prompts = prompts or {}
    triples = [
        (cat, sub, prompts.get(sub) or [f"a photo of a {sub.replace('_', ' ')}"])
        for cat, subs in categories.items()
        if only is None or cat == only
        for sub in subs
    ]
    probs = _softmax(np.array([sc.logits(t[2]).mean() for t in triples]))
    top3 = np.argsort(probs)[::-1][:3]
    return [
        {"category": triples[i][0], "subcategory": triples[i][1], "score": float(probs[i])}
        for i in top3
    ]


def _attributes(sc: _Scorer, cfg: dict, category: str, subcategory: str | None) -> dict:
    out: dict = {"fit": None, "off_shoulder": None, "sleeve": None, "details": []}

    out["pattern"] = sc.choose(cfg["pattern"]["prompts"], cfg["pattern"]["min_prob"])
    out["material"] = sc.choose(cfg["material"]["prompts"], cfg["material"]["min_prob"])

    fit = cfg["fit"]
    if category in fit["categories"]:
        out["fit"] = sc.choose(fit["prompts"][category], fit["min_prob"])

    off = cfg["off_shoulder"]
    if category in off["categories"]:
        p = sc.yes_prob(off["prompts"])
        out["off_shoulder"] = (
            True if p >= off["true_if_prob_ge"] else False if p <= off["false_if_prob_le"] else None
        )

    sleeve = cfg["sleeve"]
    if category in sleeve["categories"] and subcategory not in sleeve["skip_subcategories"]:
        out["sleeve"] = sc.choose(sleeve["prompts"], sleeve["min_prob"])

    det = cfg["details"]
    out["details"] = [
        d for d in det["verified"] if sc.yes_prob(det["prompts"][d]) >= det["true_if_prob_ge"]
    ]
    return out


def tag_item(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    mask: np.ndarray,
    category: str | None = None,
) -> TagResult:
    """옷 한 벌을 태깅한다. bbox는 픽셀 xyxy.

    category가 주어지면(사용자가 직접 고른 대분류) 그 대분류 안에서만 소분류를 고른다.
    """
    styles_cfg, categories, attr_cfg = _load_configs()
    if category is not None and category not in categories:
        raise ValueError(f"알 수 없는 대분류: {category}")
    model, processor = load_tagger()
    scale = float(model.logit_scale.detach().exp())
    img_feat = _encode_image(model, processor, crop_item(image, bbox, mask))  # (1, D)
    sc = _Scorer(img_feat, scale)

    candidates = _category_candidates(
        sc, categories, category, attr_cfg.get("subcategory", {}).get("prompts")
    )
    cat = category or candidates[0]["category"]
    sub = candidates[0]["subcategory"]
    attrs = _attributes(sc, attr_cfg, cat, sub)

    return TagResult(
        category=cat,
        subcategory=sub,
        category_candidates=candidates,
        style_scores=_style_scores(sc, styles_cfg),
        embedding=img_feat.reshape(-1).tolist(),
        **attrs,
    )

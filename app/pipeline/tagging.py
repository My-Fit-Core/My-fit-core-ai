"""FashionCLIP zero-shot 태깅: 추구미·카테고리 후보·패턴·임베딩 (F-203, T11).

입력은 crop_item()으로 준비한 흰색 배경 박스 크롭 이미지.
카테고리 소분류 목록은 configs/categories.yaml (T2 확정 전 초안).
패턴 값 목록은 안건 7 AI 제안 초안 (T3에서 백엔드 ERD enum과 맞출 것).
"""

import functools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

# ponytail: FashionCLIP(ViT-B/32). 모델 확정 후 configs/로 옮긴다
# 이 저장소는 transformers 형식이다(open_clip hf-hub 설정 파일 없음, #87). 실험 스크립트와 같은 로더를 쓴다
MODEL_NAME = "patrickjohncyh/fashion-clip"
MODEL_VERSION = "fashion-clip-vitb32"

# 패턴 값 목록 초안 (안건 7). T3에서 백엔드 ERD enum과 맞춘다
_PATTERN_PROMPTS: dict[str, str] = {
    "solid": "a solid colored clothing item with no pattern",
    "stripe": "a striped clothing item",
    "check": "a checkered or plaid clothing item",
    "dot": "a polka dot patterned clothing item",
    "floral": "a floral patterned clothing item",
    "animal": "an animal print clothing item like leopard or zebra",
    "graphic": "a clothing item with a graphic print, logo, or lettering",
    "other": "a clothing item with an unusual or unidentifiable pattern",
}


@functools.cache
def _load_configs() -> tuple[list, dict]:
    """(aesthetics list, categories dict) — 파일 변경 시 프로세스 재시작."""
    with open(ROOT / "configs" / "aesthetics.yaml", encoding="utf-8") as f:
        aesthetics = yaml.safe_load(f)["aesthetics"]
    with open(ROOT / "configs" / "categories.yaml", encoding="utf-8") as f:
        categories = yaml.safe_load(f)["categories"]
    return aesthetics, categories


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


@dataclass
class TagResult:
    aesthetic_scores: dict[str, float]  # 합 1, 보정 없음 (plan §4)
    category_candidates: list[dict]     # [{category, subcategory, score}, ...] 상위 3
    pattern: str                         # 패턴 라벨 (안건 7 초안)
    embedding: list[float]               # FashionCLIP 이미지 임베딩
    model_version: str = field(default=MODEL_VERSION)


def tag_item(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    mask: np.ndarray,
) -> TagResult:
    """검출된 옷 한 벌을 태깅한다. bbox는 픽셀 xyxy."""
    crop = crop_item(image, bbox, mask)
    model, processor = load_tagger()
    aesthetics, categories = _load_configs()
    # CLIP 학습 온도. 코사인 유사도에 곱해야 softmax가 의미 있게 갈린다(실험 스크립트와 같은 방식)
    scale = float(model.logit_scale.detach().exp())

    img_feat = _encode_image(model, processor, crop)  # (1, D)

    # ── 추구미 점수 ──────────────────────────────────────────────
    # 각 추구미의 item 프롬프트 N개 유사도 평균 × scale → softmax (보정 없음, plan §4)
    aes_keys = [a["key"] for a in aesthetics]
    aes_means = []
    for aes in aesthetics:
        txt_feats = _encode_texts(model, processor, aes["prompts"]["item"])
        sim = scale * float((img_feat @ txt_feats.T).mean())
        aes_means.append(sim)
    aes_probs = _softmax(np.array(aes_means))
    aesthetic_scores = dict(zip(aes_keys, aes_probs.tolist()))

    # ── 카테고리 후보 (상위 3) ───────────────────────────────────
    # configs/categories.yaml 초안 (T2 확정 전). 소분류별 단순 프롬프트
    triples: list[tuple[str, str, str]] = []
    for cat, subs in categories.items():
        for sub in subs:
            triples.append((cat, sub, f"a photo of a {sub.replace('_', ' ')}"))
    cat_feats = _encode_texts(model, processor, [t[2] for t in triples])
    cat_probs = _softmax(scale * (img_feat @ cat_feats.T).squeeze())
    top3 = np.argsort(cat_probs)[::-1][:3]
    category_candidates = [
        {"category": triples[i][0], "subcategory": triples[i][1], "score": float(cat_probs[i])}
        for i in top3
    ]

    # ── 패턴 ────────────────────────────────────────────────────
    pat_labels = list(_PATTERN_PROMPTS.keys())
    pat_feats = _encode_texts(model, processor, list(_PATTERN_PROMPTS.values()))
    pattern = pat_labels[int(np.argmax((img_feat @ pat_feats.T).squeeze()))]

    # ── 임베딩 ──────────────────────────────────────────────────
    embedding = img_feat.squeeze().tolist()

    return TagResult(
        aesthetic_scores=aesthetic_scores,
        category_candidates=category_candidates,
        pattern=pattern,
        embedding=embedding,
    )

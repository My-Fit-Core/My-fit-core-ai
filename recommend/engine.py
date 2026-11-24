"""추천 엔진 v0 — 규칙 기반.

입력으로 받은 옷장 목록에서 코디 세트를 구성한다.
DB 직접 조회 없음 (MVP 기준, 호출자가 wardrobe를 모두 담아 전달).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ponytail: 아우터 필요 기온 기준 — 실제 체감 기준으로 조정 가능
_OUTER_TEMP_THRESHOLD = 15.0


@dataclass
class WardrobeItem:
    item_id: str
    category: str  # top / bottom / dress / outerwear / shoes / bag
    aesthetic_scores: dict[str, float] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class OutfitSet:
    item_ids: list[str]
    score: float


def check_minimum_condition(wardrobe: list[WardrobeItem]) -> bool:
    """상의+하의 or 원피스 최소 1벌이면 True."""
    categories = {w.category for w in wardrobe}
    return "dress" in categories or ("top" in categories and "bottom" in categories)


def _mean_aesthetic_score(item: WardrobeItem) -> float:
    if not item.aesthetic_scores:
        return 0.0
    return sum(item.aesthetic_scores.values()) / len(item.aesthetic_scores)


def _outfit_score(items: list[WardrobeItem]) -> float:
    # ponytail: 단순 평균, 추구미 가중치·임베딩 유사도 도입 시 교체
    scores = [_mean_aesthetic_score(w) for w in items]
    return sum(scores) / len(scores) if scores else 0.0


def build_outfit_sets(
    wardrobe: list[WardrobeItem],
    temperature: float | None = None,
    *,
    max_sets: int = 3,
) -> list[OutfitSet]:
    """최소 조건(check_minimum_condition) 충족 보장 후 호출. 1~3세트 반환."""
    need_outer = temperature is not None and temperature < _OUTER_TEMP_THRESHOLD

    by_cat: dict[str, list[WardrobeItem]] = {}
    for w in wardrobe:
        by_cat.setdefault(w.category, []).append(w)

    tops = by_cat.get("top", [])
    bottoms = by_cat.get("bottom", [])
    dresses = by_cat.get("dress", [])
    outers = by_cat.get("outerwear", [])
    shoes = by_cat.get("shoes", [])
    bags = by_cat.get("bag", [])

    candidates: list[list[WardrobeItem]] = []

    for top in tops:
        for bottom in bottoms:
            base = [top, bottom]
            if need_outer and outers:
                base.append(outers[0])
            if shoes:
                base.append(shoes[0])
            if bags:
                base.append(bags[0])
            candidates.append(base)

    for dress in dresses:
        base = [dress]
        if need_outer and outers:
            base.append(outers[0])
        if shoes:
            base.append(shoes[0])
        if bags:
            base.append(bags[0])
        candidates.append(base)

    ranked = sorted(candidates, key=_outfit_score, reverse=True)[:max_sets]
    return [OutfitSet(item_ids=[w.item_id for w in items], score=_outfit_score(items)) for items in ranked]


def swap_slot(
    outfit_item_ids: list[str],
    slot_category: str,
    wardrobe: list[WardrobeItem],
    *,
    top_k: int = 3,
) -> list[str]:
    """슬롯 1개 교체 후보 item_id 목록 반환 (점수 높은 순). 후보 없으면 빈 리스트."""
    current_ids = set(outfit_item_ids)
    candidates = [
        w for w in wardrobe
        if w.category == slot_category and w.item_id not in current_ids
    ]
    # ponytail: 임베딩 유사도 도입 전까지 추구미 점수 합으로 정렬
    ranked = sorted(candidates, key=_mean_aesthetic_score, reverse=True)[:top_k]
    return [w.item_id for w in ranked]

"""추천 점수 함수 v0. recommend: 코디 3종(F-302), swap: 슬롯 대체 후보 랭킹(F-304).
날씨 규칙 + 추구미 매칭 + 세트 다양성. 두 함수는 같은 점수(_Scorer)를 쓴다.

순수 함수다: 모델·GPU·DB·날씨 조회 없음. 백엔드가 담아 보낸 옷장·날씨를 그대로 받는다.
카테고리와 모든 규칙 값은 configs/categories.yaml, configs/recommend.yaml에서 읽는다(초안, T2 미확정).
TPO는 1차 제외(10/12 회의 안건 6). 미세먼지는 받기만 하고 v0 규칙에 쓰지 않는다.

recommend reasons 값(내부 코드, API 코드와의 매핑은 T15). swap 사유는 swap() 참고:
  insufficient_base         상의+하의도 원피스도 없어 코디 0개
  fewer_than_requested      만든 코디가 n개보다 적음
  diversity_relaxed         다양성 기준을 완화해 핵심 옷이 겹치는 코디가 있음
  no_outer_for_cold         추운데 아우터가 없어 아우터 없이 반환
  weather_unavailable       기온·체감온도가 모두 없어 따뜻함 규칙을 끔(비 규칙은 적용)
  skipped_unknown_subcategory  설정에 없는 카테고리·소분류 옷을 뺌
  off_shoulder_outer_for_warmth  어울리는 조합이 없어(추워서 아우터 필수) 오프숄더 위에 어울리지 않는 아우터를 넣음

옷 속성(분석 결과의 숨은 속성, 사용자가 고른 따뜻함)은 Item의 선택 필드다(#119). 없으면 쓰지 않는다.
"""

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(configs_dir: Path = ROOT / "configs") -> dict:
    def read(name: str) -> dict:
        return yaml.safe_load((configs_dir / name).read_text(encoding="utf-8"))

    return {**read("recommend.yaml"), **read("categories.yaml")}


CONFIG = load_config()


@dataclass(frozen=True)
class Item:
    id: str
    category: str
    subcategory: str
    aesthetic_scores: dict[str, float] | None = None
    sleeve: str | None = None  # short / long
    warmth: str | None = None  # 사용자 선택 thin / normal / warm
    material: str | None = None
    off_shoulder: bool | None = None
    fit: str | None = None  # oversized / regular / slim


@dataclass(frozen=True)
class Weather:
    temp_c: float | None = None
    # none | rain | snow | rain_snow | shower (백엔드 값 매핑은 T15)
    precip_type: str | None = None
    humidity: float | None = None  # %
    wind_ms: float | None = None
    precip_prob: float | None = None  # %
    feels_like_c: float | None = None  # 선택 값
    pm10: float | None = None  # 선택 값, v0 미사용


@dataclass
class Outfit:
    items: dict[str, str]  # 슬롯(대분류) → item id
    score: float
    aesthetic_score: float
    weather_score: float
    warmth: int


@dataclass
class Recommendation:
    outfits: list[Outfit]
    reasons: list[str] = field(default_factory=list)
    missing_weather: list[str] = field(default_factory=list)
    target_warmth: tuple[float, float] | None = None


def _effective_temp(w: Weather, cfg: dict) -> float | None:
    if w.feels_like_c is not None:
        return w.feels_like_c
    if w.temp_c is None:
        return None
    wind = cfg["wind"]
    if w.wind_ms is not None and w.wind_ms >= wind["threshold_ms"]:
        return w.temp_c - wind["minus_c"]
    return w.temp_c


def _target_warmth(temp: float, w: Weather, cfg: dict) -> tuple[float, float]:
    lo, hi = next(b["warmth"] for b in cfg["temp_bands"] if temp >= b["min_c"])
    hh = cfg["humid_hot"]
    if (
        w.humidity is not None
        and w.humidity >= hh["humidity_ge"]
        and temp >= hh["temp_ge"]
    ):
        lo, hi = max(lo + hh["shift"], 0), max(hi + hh["shift"], 0)
    return lo, hi


def _is_rainy(w: Weather, cfg: dict) -> bool:
    if w.precip_type not in (None, "none"):
        return True
    return w.precip_prob is not None and w.precip_prob >= cfg["rain"]["prob_ge"]


class _Scorer:
    """한 요청(날씨·추구미)에 대해 코디 점수를 매긴다. recommend와 swap이 같이 쓴다."""

    def __init__(self, weather: Weather, aesthetic_key: str, cfg: dict) -> None:
        self.cfg = cfg
        self.key = aesthetic_key
        self.temp = _effective_temp(weather, cfg)
        self.target = (
            None if self.temp is None else _target_warmth(self.temp, weather, cfg)
        )
        self.rainy = _is_rainy(weather, cfg)
        self.outer_required = (
            self.temp is not None and self.temp < cfg["outer_required_below_c"]
        )
        self.core_slots = {s for b in cfg["bases"] for s in b} | {cfg["outer_slot"]}
        self.inner_subs = set(cfg.get("layered", {}).get("inner_subcategories", []))

    def aes(self, i: Item) -> float | None:
        if i.aesthetic_scores is None:
            return None
        return i.aesthetic_scores.get(self.key, 0.0)

    def rain_penalty(self, i: Item) -> float:
        rain = self.cfg["rain"]
        avoid = i.subcategory in rain["avoid"] or i.material in rain.get("avoid_materials", [])
        return rain["penalty"] if self.rainy and avoid else 0.0

    def item_warmth(self, i: Item) -> float:
        adj = self.cfg.get("warmth_adjust", {})
        w = self.cfg["warmth"][i.category][i.subcategory]
        if i.subcategory in adj.get("sleeve_subcategories", []):
            w += adj["sleeve_delta"].get(i.sleeve, 0)
        return w + adj.get("user_warmth_delta", {}).get(i.warmth, 0)

    def off_shoulder_clash(self, chosen: list[Item]) -> bool:
        """오프숄더 상의·원피스 위에 어울리지 않는 아우터가 있으면 True."""
        rule = self.cfg.get("off_shoulder")
        if not rule or not any(i.off_shoulder for i in chosen if i.category in ("top", "dress")):
            return False
        return any(
            i.subcategory in rule["never_with"] or (
                i.subcategory not in rule.get("always_ok", [])
                and i.fit not in rule["ok_if_outer_fit"]
            )
            for i in chosen
            if i.category == self.cfg["outer_slot"]
        )

    def outfit(self, chosen: list[Item]) -> Outfit:
        cfg = self.cfg
        core = [i for i in chosen if i.category in self.core_slots]
        total_warmth = sum(self.item_warmth(i) for i in core)
        # 기온을 모르면 따뜻함은 중립(1.0), 비 감점은 있는 값대로 적용
        w_score = 1.0
        if self.target is not None:
            lo, hi = self.target
            dist = max(lo - total_warmth, total_warmth - hi, 0)
            w_score = max(1 - dist / cfg["warmth_tolerance"], 0.0)
        w_score = max(w_score - sum(self.rain_penalty(i) for i in chosen), 0.0)
        scored = [a for a in map(self.aes, chosen) if a is not None]
        a_score = sum(scored) / len(scored) if scored else 0.0
        weights = cfg["weights"]
        def _slot(i: Item) -> str:
            return "inner_top" if i.subcategory in self.inner_subs else i.category

        return Outfit(
            items={_slot(i): i.id for i in chosen},
            score=weights["aesthetic"] * a_score + weights["weather"] * w_score,
            aesthetic_score=a_score,
            weather_score=w_score,
            warmth=total_warmth,
        )


def _known_items(wardrobe: list[Item], cfg: dict, reasons: list[str]) -> list[Item]:
    """설정에 없는 카테고리·소분류 옷은 뺀다."""
    known = cfg["categories"]
    items = [i for i in wardrobe if i.subcategory in known.get(i.category, [])]
    if len(items) < len(wardrobe):
        reasons.append("skipped_unknown_subcategory")
    return items


def _missing(weather: Weather) -> list[str]:
    return [k for k, v in vars(weather).items() if v is None]


def recommend(
    wardrobe: list[Item],
    weather: Weather,
    aesthetic_key: str,
    n: int = 3,
    config: dict | None = None,
) -> Recommendation:
    cfg = config or CONFIG
    reasons: list[str] = []
    missing = _missing(weather)
    by_slot: dict[str, list[Item]] = {}
    for i in _known_items(wardrobe, cfg, reasons):
        by_slot.setdefault(i.category, []).append(i)

    sc = _Scorer(weather, aesthetic_key, cfg)
    target = sc.target
    if target is None:
        reasons.append("weather_unavailable")

    # 선택 슬롯은 슬롯마다 1개: 비 오면 피할 옷이 아닌 것을 먼저, 그다음 추구미 점수
    extras = [
        max(by_slot[s], key=lambda i: (sc.rain_penalty(i) == 0, sc.aes(i) or 0.0, i.id))
        for s in cfg["optional_slots"]
        if by_slot.get(s)
    ]

    # 이너 슬롯: layered 설정 조건을 모두 만족하면 이너 후보 추가 (없음/있음 모두 후보)
    layered_cfg = cfg.get("layered", {})
    inner_subs = set(layered_cfg.get("inner_subcategories", []))
    inner_options: list[Item | None] = [None]
    if (
        inner_subs
        and aesthetic_key in layered_cfg.get("aesthetic_keys", [])
        and (sc.temp is None or sc.temp <= layered_cfg.get("max_temp_c", 23))
    ):
        inners = [i for i in by_slot.get("top", []) if i.subcategory in inner_subs]
        if inners:
            inner_options = [None, *inners]

    # 아우터: 추우면 필수, 아니면 없음/있음 모두 후보
    outers = by_slot.get(cfg["outer_slot"], [])
    if sc.outer_required and not outers:
        reasons.append("no_outer_for_cold")
    outer_options: list[Item | None] = (
        outers if sc.outer_required and outers else [None, *outers]
    )

    # 이너 소분류는 뼈대 상의에서 제외 — inner_options 경로에서만 사용
    base_slot = dict(by_slot)
    if inner_subs:
        base_slot["top"] = [i for i in by_slot.get("top", []) if i.subcategory not in inner_subs]

    def base_options(base: list[str], slot: str) -> list[Item]:
        # 원피스 + 하의 뼈대에서 하의는 받쳐 입을 수 있는 소분류만(#119)
        if slot == "bottom" and "dress" in base:
            return [i for i in base_slot.get(slot, []) if i.subcategory in cfg.get("dress_layer_bottoms", [])]
        return base_slot.get(slot, [])

    # ponytail: 뼈대 × 아우터 × 이너 전수 조합. 옷장이 수백 벌이면 슬롯별 상위 k개로 자른다
    combos = [
        [*combo, *([outer] if outer else []), *([inner] if inner else []), *extras]
        for base in cfg["bases"]
        for combo in itertools.product(*(base_options(base, s) for s in base))
        for outer in outer_options
        for inner in inner_options
    ]
    # 오프숄더 + 어울리지 않는 아우터는 뺀다. 그것밖에 없으면(추워서 아우터가 꼭 필요) 보온을 우선해 남긴다
    fitting = [c for c in combos if not sc.off_shoulder_clash(c)]
    if combos and not fitting:
        reasons.append("off_shoulder_outer_for_warmth")
    candidates = [sc.outfit(c) for c in (fitting or combos)]

    if not candidates:
        reasons.append("insufficient_base")
        return Recommendation([], reasons, missing, target)

    candidates.sort(key=lambda o: (-o.score, sorted(o.items.values())))

    def core_ids(o: Outfit) -> set[str]:
        return {v for k, v in o.items.items() if k in sc.core_slots}

    # 점수 순 greedy. 이미 고른 코디와 핵심 옷이 limit개 넘게 겹치면 건너뛰고, 못 채우면 limit을 늘린다
    picked: list[Outfit] = []
    limit = cfg["diversity"]["max_shared_core_items"]
    max_core = max(len(b) for b in cfg["bases"]) + 1
    while len(picked) < n and limit <= max_core:
        for o in candidates:
            if len(picked) == n:
                break
            if o in picked:
                continue
            if all(len(core_ids(o) & core_ids(p)) <= limit for p in picked):
                picked.append(o)
                relaxed = limit > cfg["diversity"]["max_shared_core_items"]
                if relaxed and "diversity_relaxed" not in reasons:
                    reasons.append("diversity_relaxed")
        limit += 1

    if len(picked) < n:
        reasons.append("fewer_than_requested")
    return Recommendation(picked, reasons, missing, target)


@dataclass
class SwapCandidate:
    item_id: str  # 새로 넣을 옷
    outfit: Outfit  # 교체한 뒤의 코디와 점수


@dataclass
class SwapResult:
    candidates: list[SwapCandidate]  # 점수 내림차순
    reasons: list[str] = field(default_factory=list)
    missing_weather: list[str] = field(default_factory=list)
    target_warmth: tuple[float, float] | None = None


def swap(
    wardrobe: list[Item],
    weather: Weather,
    aesthetic_key: str,
    outfit_item_ids: list[str],
    slot: str,
    config: dict | None = None,
) -> SwapResult:
    """코디의 한 슬롯(대분류)을 바꿀 후보를 recommend와 같은 점수로 매긴다 (F-304).

    reasons(내부 코드, API 매핑은 T15·T16 엔드포인트):
      unknown_slot           설정에 없는 대분류
      unknown_outfit_item    outfit_item_ids 중 옷장에 없는(또는 설정에 없는) 옷
      slot_not_in_outfit     코디의 뼈대와 다른 뼈대 슬롯(예: 상의+하의 코디에 dress)
      no_candidates          그 슬롯에 바꿀 다른 옷이 없음
      + skipped_unknown_subcategory, weather_unavailable (recommend와 같음)
    """
    cfg = config or CONFIG
    reasons: list[str] = []
    sc = _Scorer(weather, aesthetic_key, cfg)
    result = SwapResult([], reasons, _missing(weather), sc.target)
    if sc.target is None:
        reasons.append("weather_unavailable")
    if slot not in cfg["categories"]:
        reasons.append("unknown_slot")
        return result

    items = _known_items(wardrobe, cfg, reasons)
    by_id = {i.id: i for i in items}
    current = [by_id[i] for i in outfit_item_ids if i in by_id]
    if len(current) < len(outfit_item_ids):
        reasons.append("unknown_outfit_item")

    # 바꿀 수 있는 슬롯: 코디에 있는 슬롯과 같은 뼈대의 슬롯, 아우터, 선택 슬롯
    # 이너(#115)는 대분류가 상의여도 뼈대가 아니다. 원피스 + 이너 코디를 "원피스 + 상의"로 보지 않게 뺀다
    present = {i.category for i in current if i.subcategory not in sc.inner_subs}
    base_slots = {s for b in cfg["bases"] for s in b}
    in_base = present & base_slots
    # 지금 코디의 뼈대를 모두 담는 뼈대만 (상의+하의 코디에 원피스가 끼지 않게, #119)
    allowed = {s for b in cfg["bases"] if in_base and in_base <= set(b) for s in b}
    allowed |= present | {cfg["outer_slot"]} | set(cfg["optional_slots"])
    if any(i.subcategory in sc.inner_subs for i in current):
        allowed.add("top")  # 이너 교체
    if slot not in allowed:
        reasons.append("slot_not_in_outfit")
        return result

    kept = [i for i in current if i.category != slot]
    in_outfit = {i.id for i in current}
    options = [i for i in items if i.category == slot and i.id not in in_outfit]
    has_dress = any(i.category == "dress" for i in kept)
    if slot == "bottom" and has_dress:
        options = [i for i in options if i.subcategory in cfg.get("dress_layer_bottoms", [])]
    if slot == "top" and has_dress:
        # 원피스 코디의 상의 자리는 이너뿐이다(원피스 + 니트 같은 조합을 만들지 않는다)
        options = [i for i in options if i.subcategory in sc.inner_subs]
    if not options:
        reasons.append("no_candidates")
        return result

    # 오프숄더와 어울리지 않는 후보는 다른 후보가 있으면 뺀다(recommend와 같은 규칙)
    fitting = [o for o in options if not sc.off_shoulder_clash([*kept, o])]
    result.candidates = sorted(
        (SwapCandidate(o.id, sc.outfit([*kept, o])) for o in (fitting or options)),
        key=lambda c: (-c.outfit.score, c.item_id),
    )
    return result

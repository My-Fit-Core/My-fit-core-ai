"""추천 점수 함수 v0 (F-302). 날씨 규칙 + 추구미 매칭 + 세트 다양성.

순수 함수다: 모델·GPU·DB·날씨 조회 없음. 백엔드가 담아 보낸 옷장·날씨를 그대로 받는다.
카테고리와 모든 규칙 값은 configs/categories.yaml, configs/recommend.yaml에서 읽는다(초안, T2 미확정).
TPO는 1차 제외(10/12 회의 안건 6). 미세먼지는 받기만 하고 v0 규칙에 쓰지 않는다.

reasons 값(내부 코드, API 코드와의 매핑은 T15):
  insufficient_base         상의+하의도 원피스도 없어 코디 0개
  fewer_than_requested      만든 코디가 n개보다 적음
  diversity_relaxed         다양성 기준을 완화해 핵심 옷이 겹치는 코디가 있음
  no_outer_for_cold         추운데 아우터가 없어 아우터 없이 반환
  weather_unavailable       기온·체감온도가 모두 없어 따뜻함 규칙을 끔(비 규칙은 적용)
  skipped_unknown_subcategory  설정에 없는 카테고리·소분류 옷을 뺌
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


def recommend(
    wardrobe: list[Item],
    weather: Weather,
    aesthetic_key: str,
    n: int = 3,
    config: dict | None = None,
) -> Recommendation:
    cfg = config or CONFIG
    reasons: list[str] = []
    missing = [k for k, v in vars(weather).items() if v is None]

    # 설정에 없는 옷은 뺀다
    known = cfg["categories"]
    items = [i for i in wardrobe if i.subcategory in known.get(i.category, [])]
    if len(items) < len(wardrobe):
        reasons.append("skipped_unknown_subcategory")
    by_slot: dict[str, list[Item]] = {}
    for i in items:
        by_slot.setdefault(i.category, []).append(i)

    def aes(i: Item) -> float | None:
        return (
            None
            if i.aesthetic_scores is None
            else i.aesthetic_scores.get(aesthetic_key, 0.0)
        )

    def warmth(i: Item) -> int:
        return cfg["warmth"][i.category][i.subcategory]

    temp = _effective_temp(weather, cfg)
    target = None if temp is None else _target_warmth(temp, weather, cfg)
    if target is None:
        reasons.append("weather_unavailable")
    rainy = _is_rainy(weather, cfg)
    rain_cfg = cfg["rain"]

    def rain_penalty(i: Item) -> float:
        return (
            rain_cfg["penalty"] if rainy and i.subcategory in rain_cfg["avoid"] else 0.0
        )

    # 선택 슬롯은 슬롯마다 1개: 비 오면 피할 옷이 아닌 것을 먼저, 그다음 추구미 점수
    extras = [
        max(by_slot[s], key=lambda i: (rain_penalty(i) == 0, aes(i) or 0.0, i.id))
        for s in cfg["optional_slots"]
        if by_slot.get(s)
    ]

    # 아우터: 추우면 필수, 아니면 없음/있음 모두 후보
    outers = by_slot.get(cfg["outer_slot"], [])
    outer_required = temp is not None and temp < cfg["outer_required_below_c"]
    if outer_required and not outers:
        reasons.append("no_outer_for_cold")
    outer_options: list[Item | None] = (
        outers if outer_required and outers else [None, *outers]
    )

    # ponytail: 뼈대 × 아우터 전수 조합. 옷장이 수백 벌이면 슬롯별 상위 k개로 자른다
    candidates: list[Outfit] = []
    for base in cfg["bases"]:
        for combo in itertools.product(*(by_slot.get(s, []) for s in base)):
            for outer in outer_options:
                core = [*combo, *([outer] if outer else [])]
                chosen = [*core, *extras]
                total_warmth = sum(warmth(i) for i in core)
                # 기온을 모르면 따뜻함은 중립(1.0), 비 감점은 있는 값대로 적용
                w_score = 1.0
                if target is not None:
                    lo, hi = target
                    dist = max(lo - total_warmth, total_warmth - hi, 0)
                    w_score = max(1 - dist / cfg["warmth_tolerance"], 0.0)
                w_score = max(w_score - sum(rain_penalty(i) for i in chosen), 0.0)
                scored = [a for a in map(aes, chosen) if a is not None]
                a_score = sum(scored) / len(scored) if scored else 0.0
                weights = cfg["weights"]
                candidates.append(
                    Outfit(
                        items={i.category: i.id for i in chosen},
                        score=weights["aesthetic"] * a_score
                        + weights["weather"] * w_score,
                        aesthetic_score=a_score,
                        weather_score=w_score,
                        warmth=total_warmth,
                    )
                )

    if not candidates:
        reasons.append("insufficient_base")
        return Recommendation([], reasons, missing, target)

    candidates.sort(key=lambda o: (-o.score, sorted(o.items.values())))
    core_slots = {s for base in cfg["bases"] for s in base} | {cfg["outer_slot"]}

    def core_ids(o: Outfit) -> set[str]:
        return {v for k, v in o.items.items() if k in core_slots}

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

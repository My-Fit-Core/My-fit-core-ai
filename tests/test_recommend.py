import copy
import dataclasses

import pytest
import yaml

from recommend.scoring import CONFIG, ROOT, Item, Weather, recommend, swap

MILD = Weather(temp_c=20, precip_type="none", humidity=50, wind_ms=2, precip_prob=10)


def _item(id_, cat, sub, **aes):
    base = {
        "feminine": 0.2,
        "street": 0.2,
        "minimal": 0.2,
        "vintage_layered": 0.2,
        "casual": 0.2,
    }
    return Item(id_, cat, sub, {**base, **aes} if aes is not None else None)


# 샘플 옷장: 상의 4, 하의 3, 원피스 1, 아우터 3, 신발 2, 가방 1
WARDROBE = [
    _item("t_tee", "top", "tshirt", casual=0.6),
    _item("t_blouse", "top", "blouse", feminine=0.6),
    _item("t_hood", "top", "hoodie", street=0.6),
    _item("t_knit", "top", "knit", minimal=0.5),
    _item("b_jeans", "bottom", "jeans", casual=0.5),
    _item("b_slacks", "bottom", "slacks", minimal=0.5),
    _item("b_shorts", "bottom", "shorts", street=0.4),
    _item("d_mini", "dress", "mini_dress", feminine=0.7),
    _item("o_card", "outerwear", "cardigan", feminine=0.4),
    _item("o_coat", "outerwear", "coat", minimal=0.5),
    _item("o_pad", "outerwear", "padding", street=0.4),
    _item("s_sneak", "shoes", "sneakers", street=0.4),
    _item("s_sandal", "shoes", "sandals", feminine=0.9),
    Item("g_tote", "bag", "tote_bag"),  # 수동 등록, 추구미 점수 없음
]
CORE = {"top", "bottom", "dress", "outerwear"}


def _core(o):
    return {v for k, v in o.items.items() if k in CORE}


def test_three_diverse_outfits():
    rec = recommend(WARDROBE, MILD, "casual")
    assert len(rec.outfits) == 3
    for a in range(3):
        for b in range(a + 1, 3):
            assert not _core(rec.outfits[a]) & _core(rec.outfits[b])
    assert rec.reasons == []
    assert [o.score for o in rec.outfits] == sorted(
        (o.score for o in rec.outfits), reverse=True
    )


def test_cold_requires_warm_outer():
    rec = recommend(WARDROBE, Weather(temp_c=-5, precip_type="none"), "casual")
    assert rec.outfits
    assert all("outerwear" in o.items for o in rec.outfits)
    assert rec.outfits[0].items["outerwear"] in {"o_coat", "o_pad"}


def test_hot_humid_no_outer_low_warmth():
    rec = recommend(WARDROBE, Weather(temp_c=30, humidity=85), "casual")
    assert rec.target_warmth == (0, 0)
    assert all("outerwear" not in o.items for o in rec.outfits)
    assert rec.outfits[0].warmth <= 1


def test_rain_avoids_sandals():
    sunny = recommend(WARDROBE, MILD, "feminine")
    assert sunny.outfits[0].items["shoes"] == "s_sandal"
    for w in (
        Weather(temp_c=20, precip_type="rain"),
        Weather(temp_c=20, precip_prob=70),
    ):
        rec = recommend(WARDROBE, w, "feminine")
        assert all(o.items["shoes"] == "s_sneak" for o in rec.outfits)


def test_insufficient_base():
    wardrobe = [i for i in WARDROBE if i.category not in {"bottom", "dress"}]
    rec = recommend(wardrobe, MILD, "casual")
    assert rec.outfits == []
    assert "insufficient_base" in rec.reasons


def test_fewer_than_requested():
    rec = recommend([WARDROBE[0], WARDROBE[4]], MILD, "casual")
    assert len(rec.outfits) == 1
    assert rec.outfits[0].items == {"top": "t_tee", "bottom": "b_jeans"}
    assert "fewer_than_requested" in rec.reasons


def test_dress_only():
    rec = recommend([WARDROBE[7]], MILD, "feminine")
    assert rec.outfits[0].items == {"dress": "d_mini"}


def test_cold_without_outer():
    wardrobe = [i for i in WARDROBE if i.category != "outerwear"]
    rec = recommend(wardrobe, Weather(temp_c=0), "casual")
    assert rec.outfits
    assert "no_outer_for_cold" in rec.reasons


def test_missing_temperature_neutral_warmth():
    rec = recommend(WARDROBE, Weather(humidity=50), "feminine")
    assert "weather_unavailable" in rec.reasons
    assert {"temp_c", "feels_like_c", "precip_type"} <= set(rec.missing_weather)
    assert rec.target_warmth is None
    assert all(o.weather_score == 1 for o in rec.outfits)
    assert rec.outfits[0].items.get("dress") == "d_mini"


def test_feels_like_overrides_temp():
    rec = recommend(WARDROBE, Weather(temp_c=20, feels_like_c=-5), "casual")
    assert all("outerwear" in o.items for o in rec.outfits)


def test_wind_lowers_effective_temp():
    calm = recommend(WARDROBE, Weather(temp_c=13, wind_ms=1), "casual")
    windy = recommend(WARDROBE, Weather(temp_c=13, wind_ms=10), "casual")
    assert calm.target_warmth != windy.target_warmth
    assert all("outerwear" in o.items for o in windy.outfits)


def test_aesthetic_key_changes_top_pick():
    street = recommend(WARDROBE, MILD, "street").outfits[0]
    feminine = recommend(WARDROBE, MILD, "feminine").outfits[0]
    assert street.items != feminine.items


def test_unknown_subcategory_skipped():
    rec = recommend([*WARDROBE, Item("x", "top", "corset")], MILD, "casual")
    assert "skipped_unknown_subcategory" in rec.reasons
    assert all("x" not in o.items.values() for o in rec.outfits)


def test_config_swap_without_code_change():
    cfg = copy.deepcopy(CONFIG)
    cfg["bases"] = [["top", "bottom"]]  # 원피스 코디를 끈다
    rec = recommend(WARDROBE, MILD, "feminine", n=10, config=cfg)
    assert all("dress" not in o.items for o in rec.outfits)
    cfg["categories"]["top"].append("corset")
    cfg["warmth"]["top"]["corset"] = 0
    rec = recommend(
        [Item("x", "top", "corset"), WARDROBE[4]], MILD, "casual", config=cfg
    )
    assert rec.outfits[0].items == {"top": "x", "bottom": "b_jeans"}


def test_warmth_matches_categories():
    cats = yaml.safe_load(
        (ROOT / "configs" / "categories.yaml").read_text(encoding="utf-8")
    )
    clothing = {s for base in CONFIG["bases"] for s in base} | {CONFIG["outer_slot"]}
    for slot in clothing:
        assert set(CONFIG["warmth"][slot]) == set(cats["categories"][slot]), slot
    for slot in CONFIG["optional_slots"]:
        assert slot in cats["categories"]


def test_missing_temperature_still_applies_rain():
    wardrobe = [
        _item("t", "top", "tshirt"),
        _item("b_long", "bottom", "long_skirt", casual=0.5),
        _item("b_jeans", "bottom", "jeans"),
    ]
    dry = recommend(wardrobe, Weather(precip_type="none"), "casual")
    wet = recommend(wardrobe, Weather(precip_type="rain"), "casual")
    assert dry.outfits[0].items["bottom"] == "b_long"
    assert wet.outfits[0].items["bottom"] == "b_jeans"


def test_rain_avoid_names_exist():
    known = {s for subs in CONFIG["categories"].values() for s in subs}
    assert set(CONFIG["rain"]["avoid"]) <= known


# --- swap (T16) ---
OUTFIT = ["t_tee", "b_jeans", "s_sneak"]


def test_swap_ranks_same_slot_candidates():
    res = swap(WARDROBE, MILD, "minimal", OUTFIT, "top")
    ids = [c.item_id for c in res.candidates]
    assert set(ids) == {"t_blouse", "t_hood", "t_knit"}  # 현재 상의는 제외
    assert ids[0] == "t_knit"  # minimal 점수가 가장 높은 상의
    scores = [c.outfit.score for c in res.candidates]
    assert scores == sorted(scores, reverse=True)
    for c in res.candidates:  # 다른 슬롯은 그대로
        assert c.outfit.items["bottom"] == "b_jeans"
        assert c.outfit.items["shoes"] == "s_sneak"
        assert c.outfit.items["top"] == c.item_id
    assert res.reasons == []


def test_swap_uses_weather_cold_prefers_warm_outer():
    res = swap(WARDROBE, Weather(temp_c=-5), "casual", OUTFIT, "outerwear")
    assert res.candidates[0].item_id in {"o_coat", "o_pad"}


def test_swap_rain_ranks_sandals_last():
    res = swap(
        WARDROBE,
        Weather(temp_c=20, precip_type="rain"),
        "feminine",
        ["t_blouse", "b_slacks", "s_sneak"],
        "shoes",
    )
    assert [c.item_id for c in res.candidates] == ["s_sandal"]
    assert res.candidates[0].outfit.weather_score < 1


def test_swap_no_candidates_reason():
    res = swap(WARDROBE, MILD, "casual", ["d_mini", "g_tote"], "bag")
    assert res.candidates == []
    assert "no_candidates" in res.reasons


def test_swap_empty_slot_in_wardrobe():
    wardrobe = [i for i in WARDROBE if i.category != "outerwear"]
    res = swap(wardrobe, MILD, "casual", OUTFIT, "outerwear")
    assert res.candidates == []
    assert "no_candidates" in res.reasons


def test_swap_unknown_slot():
    res = swap(WARDROBE, MILD, "casual", OUTFIT, "socks")
    assert res.candidates == [] and "unknown_slot" in res.reasons


def test_swap_slot_not_in_outfit():
    res = swap(WARDROBE, MILD, "casual", OUTFIT, "dress")
    assert res.candidates == [] and "slot_not_in_outfit" in res.reasons


def test_swap_unknown_outfit_item():
    res = swap(WARDROBE, MILD, "casual", ["t_tee", "ghost"], "bottom")
    assert "unknown_outfit_item" in res.reasons
    assert res.candidates  # 남은 옷으로 계속 계산


def test_swap_missing_weather():
    res = swap(WARDROBE, Weather(), "casual", OUTFIT, "top")
    assert "weather_unavailable" in res.reasons
    assert res.target_warmth is None
    assert res.candidates


def test_swap_score_matches_recommend():
    rec = recommend(WARDROBE, MILD, "casual").outfits[0]
    ids = list(rec.items.values())
    slot = "top" if "top" in rec.items else "dress"
    res = swap(WARDROBE, MILD, "casual", ids, slot)
    # 같은 점수 함수: 1위 추천 코디보다 높은 대체는 없다(같은 핵심 조합 안에서)
    assert all(c.outfit.score <= rec.score + 1e-9 for c in res.candidates)


# --- 레이어드 (이너 슬롯) ---

def test_vintage_layered_adds_inner_slot():
    wardrobe = [
        _item("t_hood", "top", "hoodie", vintage_layered=0.7),
        _item("t_inner", "top", "inner_tshirt", vintage_layered=0.5),
        _item("b_jeans", "bottom", "jeans"),
    ]
    rec = recommend(wardrobe, Weather(temp_c=10), "vintage_layered")
    assert rec.outfits
    assert any("inner_top" in o.items for o in rec.outfits)
    assert all(o.items.get("inner_top") != o.items.get("top") for o in rec.outfits if "inner_top" in o.items)


def test_no_inner_in_summer():
    wardrobe = [
        _item("t_hood", "top", "hoodie", vintage_layered=0.7),
        _item("t_inner", "top", "inner_tshirt", vintage_layered=0.5),
        _item("b_jeans", "bottom", "jeans"),
    ]
    rec = recommend(wardrobe, Weather(temp_c=30), "vintage_layered")
    assert all("inner_top" not in o.items for o in rec.outfits)


def test_no_inner_for_non_layered_aesthetic():
    wardrobe = [
        _item("t_hood", "top", "hoodie", casual=0.7),
        _item("t_inner", "top", "inner_tshirt", casual=0.5),
        _item("b_jeans", "bottom", "jeans"),
    ]
    rec = recommend(wardrobe, Weather(temp_c=10), "casual")
    assert all("inner_top" not in o.items for o in rec.outfits)


# ── v1 옷 속성 (#119) ────────────────────────────────────────────


COLD = Weather(temp_c=5, precip_type="none")


def _w(*items):
    return list(items)


def test_off_shoulder_avoids_blazer_when_other_choice():
    """오프숄더 상의 + 블레이저는 다른 조합이 있으면 나오지 않는다."""
    wardrobe = _w(
        Item("t_off", "top", "knit", {"feminine": 0.9}, off_shoulder=True),
        Item("t_tee", "top", "tshirt", {"feminine": 0.2}, sleeve="long"),
        Item("b_skirt", "bottom", "long_skirt", {"feminine": 0.8}),
        Item("o_blazer", "outerwear", "blazer", {"feminine": 0.9}),
        Item("o_coat", "outerwear", "coat", {"feminine": 0.5}, fit="regular"),
    )
    rec = recommend(wardrobe, COLD, "feminine")
    for o in rec.outfits:
        if o.items.get("top") == "t_off":
            pytest.fail(f"오프숄더 + 어깨선 잡힌 아우터: {o.items}")
    assert "off_shoulder_outer_for_warmth" not in rec.reasons


def test_off_shoulder_ok_with_oversized_outer():
    wardrobe = _w(
        Item("t_off", "top", "knit", {"feminine": 0.9}, off_shoulder=True),
        Item("b_skirt", "bottom", "long_skirt", {"feminine": 0.8}),
        Item("o_mustang", "outerwear", "coat", {"feminine": 0.7}, fit="oversized"),
    )
    rec = recommend(wardrobe, COLD, "feminine")
    assert rec.outfits[0].items == {"top": "t_off", "bottom": "b_skirt", "outerwear": "o_mustang"}
    assert "off_shoulder_outer_for_warmth" not in rec.reasons


def test_off_shoulder_warmth_first_when_no_other_choice():
    """오프숄더밖에 없고 추우면 어울리지 않아도 아우터를 넣고 이유를 남긴다."""
    wardrobe = _w(
        Item("t_off", "top", "knit", {"feminine": 0.9}, off_shoulder=True),
        Item("b_skirt", "bottom", "long_skirt", {"feminine": 0.8}),
        Item("o_jacket", "outerwear", "jacket", {"feminine": 0.5}),
    )
    rec = recommend(wardrobe, Weather(temp_c=0, precip_type="none"), "feminine")
    assert rec.outfits[0].items["outerwear"] == "o_jacket"
    assert "off_shoulder_outer_for_warmth" in rec.reasons


def test_long_sleeve_and_user_warmth_raise_warmth():
    from recommend.scoring import _Scorer

    sc = _Scorer(MILD, "casual", CONFIG)
    tee = Item("t", "top", "tshirt")
    assert sc.item_warmth(dataclasses.replace(tee, sleeve="long")) == sc.item_warmth(tee) + 1
    assert sc.item_warmth(dataclasses.replace(tee, warmth="warm")) == sc.item_warmth(tee) + 1
    knit = Item("k", "top", "knit", sleeve="long")  # 니트는 소매로 바꾸지 않는다
    assert sc.item_warmth(knit) == CONFIG["warmth"]["top"]["knit"]


def test_rain_avoids_suede_material():
    wardrobe = _w(
        Item("t", "top", "tshirt", {"casual": 0.5}),
        Item("b", "bottom", "jeans", {"casual": 0.5}),
        Item("s_suede", "shoes", "boots", {"casual": 0.9}, material="suede"),
        Item("s_sneak", "shoes", "sneakers", {"casual": 0.4}),
    )
    assert recommend(wardrobe, MILD, "casual").outfits[0].items["shoes"] == "s_suede"
    rainy = Weather(temp_c=20, precip_type="rain")
    assert recommend(wardrobe, rainy, "casual").outfits[0].items["shoes"] == "s_sneak"


def test_dress_with_pants_is_a_candidate():
    """원피스만 / 원피스 + 바지 둘 다 후보, 스커트는 원피스 밑에 받쳐 입지 않는다."""
    wardrobe = _w(
        Item("d", "dress", "long_dress", {"feminine": 0.9}),
        Item("b_jeans", "bottom", "jeans", {"feminine": 0.9}),
        Item("b_skirt", "bottom", "mini_skirt", {"feminine": 0.9}),
    )
    # 15도: 목표 따뜻함 3~4. 롱원피스(2)만보다 롱원피스 + 청바지(3)가 날씨 점수가 높다
    rec = recommend(wardrobe, Weather(temp_c=15, precip_type="none"), "feminine", n=3)
    combos = [o.items for o in rec.outfits]
    assert combos[0] == {"dress": "d", "bottom": "b_jeans"}
    assert all(not (c.get("dress") and c.get("bottom") == "b_skirt") for c in combos)
    hot = recommend(wardrobe, Weather(temp_c=30, precip_type="none"), "feminine", n=3)
    assert {"dress": "d"} in [o.items for o in hot.outfits]  # 원피스만도 후보


def test_swap_outer_skips_clashing_for_off_shoulder():
    wardrobe = _w(
        Item("t_off", "top", "knit", {"feminine": 0.9}, off_shoulder=True),
        Item("b", "bottom", "long_skirt", {"feminine": 0.8}),
        Item("o_blazer", "outerwear", "blazer", {"feminine": 0.9}),
        Item("o_card", "outerwear", "cardigan", {"feminine": 0.3}, fit="oversized"),
    )
    res = swap(wardrobe, COLD, "feminine", ["t_off", "b"], "outerwear")
    assert [c.item_id for c in res.candidates] == ["o_card"]


LAYERED = [
    Item("d", "dress", "long_dress", {"vintage_layered": 0.9}),
    Item("i_tee", "top", "inner_tshirt", {"vintage_layered": 0.8}),
    Item("t_knit", "top", "knit", {"vintage_layered": 0.9}),
    Item("b_jeans", "bottom", "jeans", {"vintage_layered": 0.7}),
]


def test_swap_dress_with_inner_can_add_pants():
    """원피스 + 이너 코디도 원피스 코디라서 바지를 받쳐 입을 수 있다."""
    res = swap(LAYERED, MILD, "vintage_layered", ["d", "i_tee"], "bottom")
    assert [c.item_id for c in res.candidates] == ["b_jeans"]


def test_swap_dress_top_slot_is_inner_only():
    res = swap(LAYERED, MILD, "vintage_layered", ["d", "i_tee"], "top")
    assert all(c.item_id != "t_knit" for c in res.candidates)
    res = swap(LAYERED, MILD, "vintage_layered", ["d"], "top")
    assert res.reasons == ["slot_not_in_outfit"]


def test_swap_top_bottom_outfit_cannot_add_dress():
    res = swap(WARDROBE, MILD, "casual", ["t_tee", "b_jeans"], "dress")
    assert res.reasons == ["slot_not_in_outfit"]


def test_swap_dress_outfit_can_add_pants_only():
    res = swap(WARDROBE, MILD, "feminine", ["d_mini"], "bottom")
    assert {c.item_id for c in res.candidates} <= {"b_jeans", "b_slacks"}
    assert res.candidates

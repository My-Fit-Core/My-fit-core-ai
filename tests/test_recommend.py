import copy

import yaml

from recommend.scoring import CONFIG, ROOT, Item, Weather, recommend

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

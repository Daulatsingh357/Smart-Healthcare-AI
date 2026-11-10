import pytest
from ml_models import diet_recommender as D

BASE = dict(age=30, sex="male", height_cm=172, weight_kg=82, target_weight_kg=74, activity="light", goal="lose", preference="vegetarian", allergies="peanut", conditions=[])


def foods_in(plan):
    return [i for d in plan["week"] for m in d["meals"].values() for i in m["items"]]


def test_bmi_bmr_formulas():
    assert D.bmi(70, 175) == 22.9
    assert round(D.bmr_mifflin("male", 70, 175, 30)) == 1649          # 10*70+6.25*175-5*30+5
    assert round(D.bmr_mifflin("female", 60, 160, 25)) == 1314


def test_targets_are_safe_and_consistent():
    t = D.compute_targets(D.validate(BASE))
    assert t["calories"] >= 1500 and (t["tdee"] - t["calories"]) <= 500
    assert abs(4 * t["protein_g"] + 4 * t["carbs_g"] + 9 * t["fat_g"] - t["calories"]) < 25
    assert t["carbs_g"] >= 130 and t["water_ml"] and 1500 <= t["water_ml"] <= 3500


def test_calorie_floor_for_small_woman():
    d = D.validate(dict(BASE, sex="female", weight_kg=52, height_cm=160, target_weight_kg=50, age=60, activity="sedentary"))
    t = D.compute_targets(d)
    assert t["calories"] >= 1200 and any("minimum" in w for w in t["warnings"])


@pytest.mark.parametrize("bad", [dict(weight_kg=48, target_weight_kg=45), dict(goal="gain", target_weight_kg=90), dict(goal="lose", target_weight_kg=90),
                                 dict(goal="lose", target_weight_kg=50), dict(conditions=["pregnancy"]), dict(age=15), dict(sex="x"), dict(height_cm=20)])
def test_unsafe_requests_blocked(bad):
    with pytest.raises(D.PlanError):
        D.build_plan(dict(BASE, **bad))


def test_week_plan_shape_and_calories():
    p = D.build_plan(BASE, seed=1)
    assert len(p["week"]) == 7 and all(set(d["meals"]) == {"breakfast", "lunch", "snack", "dinner"} for d in p["week"])
    for d in p["week"]:
        assert abs(d["kcal"] - p["targets"]["calories"]) <= 0.10 * p["targets"]["calories"]
    assert len({i["name"] for i in foods_in(p)}) >= 20            # variety


@pytest.mark.parametrize("pref,banned", [("vegetarian", {"nonveg", "egg"}), ("vegan", {"nonveg", "egg", "veg"}), ("eggetarian", {"nonveg"})])
def test_dietary_preference_respected(pref, banned):
    p = D.build_plan(dict(BASE, preference=pref, allergies=""), seed=2)
    diets = {D.food_row(i["food_id"]).diet for i in foods_in(p)}
    assert not (diets & banned)


def test_allergies_and_conditions_never_violated():
    p = D.build_plan(dict(BASE, goal="healthy", preference="non-vegetarian", allergies="milk, gluten, fish, nuts, egg", conditions=["diabetes", "hypertension", "heart"]), seed=3)
    for i in foods_in(p):
        r = D.food_row(i["food_id"])
        assert not (r.allergen_set & {"dairy", "gluten", "fish", "nuts", "egg"}), i["name"]
        assert not (r.flag_set & {"highgi", "highsodium", "fried", "highsatfat"}), i["name"]


def test_kidney_limits_protein_and_warns():
    t = D.compute_targets(D.validate(dict(BASE, conditions=["kidney"], goal="healthy", target_weight_kg=82)))
    assert t["protein_g"] <= round(82 * 0.8) + 1 and t["water_ml"] is None and any("nephrologist" in w for w in t["warnings"])


def test_insufficient_food_pool_raises_clear_error(monkeypatch):
    allf = D.foods()
    monkeypatch.setattr(D, "allowed_foods", lambda d: allf.iloc[0:0])
    with pytest.raises(D.PlanError, match="dietitian"):
        D.build_plan(BASE, seed=1)
    monkeypatch.setattr(D, "allowed_foods", lambda d: allf[allf.role == "bmain"])      # no staples/vegetables left
    with pytest.raises(D.PlanError, match="Too few foods"):
        D.build_plan(BASE, seed=1)


def test_heavily_restricted_but_feasible_plan_still_safe():
    p = D.build_plan(dict(BASE, preference="vegan", goal="healthy", allergies="gluten, soy, nuts, peanut, sesame, dairy, egg, fish", conditions=["diabetes", "hypertension", "heart"]), seed=1)
    assert len(p["week"]) == 7


def test_alternatives_same_role_and_filtered():
    alts = D.alternatives(21, 4)
    assert alts and all(a["role"] == "staple" and a["id"] != 21 for a in alts)


def test_api_plan_persist_pdf_swap_and_isolation(pc, pc2):
    r = pc.post("/api/diet/plan", json=BASE)
    assert r.status_code == 201, r.get_json()
    plan = r.get_json(); pid = plan["id"]
    assert pc.get(f"/api/diet/plans/{pid}").get_json()["calories"] == plan["calories"]
    assert pc.get(f"/api/diet/plans/{pid}/pdf").data.startswith(b"%PDF")
    assert pc2.get(f"/api/diet/plans/{pid}").status_code == 403 and pc2.get(f"/api/diet/plans/{pid}/pdf").status_code == 403
    meal = plan["week"][0]["meals"]["lunch"]; item = meal["items"][0]
    alt = pc.get(f"/api/diet/foods/{item['food_id']}/alternatives").get_json()[0]
    s = pc.post(f"/api/diet/plans/{pid}/swap", json=dict(day_index=0, meal_type="lunch", food_id=item["food_id"], new_food_id=alt["id"]))
    assert s.status_code == 200 and s.get_json()["items"][0]["food_id"] == alt["id"]
    assert pc.get(f"/api/diet/plans/{pid}").get_json()["week"][0]["meals"]["lunch"]["items"][0]["food_id"] == alt["id"]
    nonveg = next(f for f in pc.get("/api/diet/foods?q=chicken").get_json())
    assert pc.post(f"/api/diet/plans/{pid}/swap", json=dict(day_index=0, meal_type="lunch", food_id=alt["id"], new_food_id=nonveg["id"])).status_code == 422   # violates preference
    assert pc.post("/api/diet/plan", json=dict(BASE, age=12)).status_code == 422


def test_new_plan_deactivates_previous(pc):
    pc.post("/api/diet/plan", json=BASE); pc.post("/api/diet/plan", json=dict(BASE, goal="healthy", target_weight_kg=82))
    plans = pc.get("/api/diet/plans").get_json()
    assert [p["is_active"] for p in plans] == [True, False]


def test_calculator_search_and_metrics_api(pc):
    foods = pc.get("/api/diet/foods?q=roti").get_json()
    roti = next(f for f in foods if f["name"] == "Roti (wheat)")
    r = pc.post("/api/diet/calc", json=dict(items=[dict(food_id=roti["id"], servings=3)])).get_json()
    assert r["totals"]["kcal"] == 300
    assert pc.post("/api/diet/calc", json=dict(items=[dict(food_id=roti["id"], servings=-1)])).status_code == 400
    m = pc.post("/api/diet/metrics", json=dict(age=30, sex="male", height_cm=175, weight_kg=70, activity="light")).get_json()
    assert m["bmi"] == 22.9 and m["bmr"] == 1649


def test_tracking_water_food_weight_and_isolation(pc, pc2):
    pc.post("/api/diet/plan", json=BASE)
    roti = next(f for f in pc.get("/api/diet/foods?q=roti").get_json() if f["name"] == "Roti (wheat)")
    assert pc.post("/api/nutrition/log", json=dict(food_id=roti["id"], servings=2, meal_type="lunch")).status_code == 201
    assert pc.post("/api/nutrition/water", json=dict(ml=250)).status_code == 201
    assert pc.post("/api/nutrition/water", json=dict(ml=99999)).status_code == 400
    assert pc.post("/api/nutrition/log", json=dict(food_id=roti["id"], servings=2, meal_type="brunch")).status_code == 400
    assert pc.post("/api/nutrition/weight", json=dict(weight_kg=80.5)).status_code == 201
    assert pc.post("/api/nutrition/weight", json=dict(weight_kg=5)).status_code == 400
    s = pc.get("/api/nutrition/summary").get_json()
    assert s["totals"]["kcal"] == 200 and s["totals"]["water_ml"] == 250 and s["target"]["kcal"] > 0 and s["weights"][-1]["kg"] == 80.5
    eid = s["entries"][0]["id"]
    assert pc2.delete(f"/api/nutrition/{eid}").status_code == 403
    assert pc.delete(f"/api/nutrition/{eid}").status_code == 200
    assert pc2.get("/api/nutrition/summary").get_json()["totals"]["kcal"] == 0

"""Personalised nutrition engine: BMI / BMR / calorie & macro targets, safe weekly meal planning, food swaps.

AI component (honest description): content-based recommendation. Foods are filtered by diet type, allergens and medical
restrictions, then ranked by cosine similarity between each food's macro-calorie shares and the target macro split
(with a recency penalty for variety). Alternatives come from a k-nearest-neighbour search on standardised nutrient
vectors. There is no generative model. Food values are approximate typical-serving figures (IFCT/NIN-style).
Safety rules: adults only, no pregnancy plans, calorie floors (1500 M / 1200 F), max deficit 500 kcal (~0.5 kg/week),
no weight-loss plan for BMI < 18.5 or target BMI < 18.5, no weight-gain plan when BMI >= 25.
"""
import os, random, re
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FOOD_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "indian_foods.csv")
ACTIVITY = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "very_active": 1.725}
GOALS = ("lose", "gain", "maintain", "healthy")
PREFERENCES = {"vegan": {"vegan"}, "vegetarian": {"vegan", "veg"}, "eggetarian": {"vegan", "veg", "egg"}, "non-vegetarian": {"vegan", "veg", "egg", "nonveg"}}
CONDITIONS = ("diabetes", "hypertension", "kidney", "heart", "pregnancy")
SLOT_SHARE = {"breakfast": 0.25, "lunch": 0.35, "snack": 0.10, "dinner": 0.30}
SCALABLE = {"bmain", "staple", "legume", "protein", "snack"}
ALLERGEN_SYNONYMS = {"milk": "dairy", "lactose": "dairy", "curd": "dairy", "paneer": "dairy", "wheat": "gluten", "maida": "gluten", "atta": "gluten", "nut": "nuts",
                     "almond": "nuts", "cashew": "nuts", "walnut": "nuts", "groundnut": "peanut", "peanuts": "peanut", "eggs": "egg", "seafood": "fish",
                     "prawn": "shellfish", "shrimp": "shellfish", "soya": "soy", "tofu": "soy", "til": "sesame", "nuts": "nuts"}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DISCLAIMER = ("General nutrition guidance for healthy adults, not medical advice. Food values are approximate. People with diabetes, kidney, heart or liver "
              "disease, eating disorders, or who are pregnant/breast-feeding must follow a plan made by their doctor or dietitian.")


class PlanError(ValueError):
    """Raised when a requested plan would be unsafe or impossible; message is shown to the user."""


_foods = None


def foods():
    global _foods
    if _foods is None:
        df = pd.read_csv(FOOD_CSV).fillna({"allergens": "", "flags": ""})
        df["allergen_set"] = df["allergens"].map(lambda s: {a for a in s.split(";") if a})
        df["flag_set"] = df["flags"].map(lambda s: {a for a in s.split(";") if a})
        df["meal_set"] = df["meals"].map(lambda s: set(s.split(";")))
        _foods = df
    return _foods


def food_row(fid):
    df = foods()
    r = df[df.id == int(fid)]
    if r.empty:
        raise KeyError(fid)
    return r.iloc[0]


def search_foods(q="", diet=None, limit=25):
    df = foods()
    if q:
        df = df[df["name"].str.contains(re.escape(q.strip()), case=False)]
    if diet in PREFERENCES:
        df = df[df.diet.isin(PREFERENCES[diet])]
    return [food_dict(r) for _, r in df.head(limit).iterrows()]


def food_dict(r):
    return dict(id=int(r.id), name=r["name"], role=r.role, diet=r.diet, serving=r.serving, kcal=float(r.kcal), protein_g=float(r.protein), carbs_g=float(r.carbs),
                fat_g=float(r.fat), fiber_g=float(r.fiber), allergens=sorted(r.allergen_set), flags=sorted(r.flag_set))


def item_for(r, qty):
    return dict(food_id=int(r.id), name=r["name"], serving=r.serving, qty=float(qty), role=r.role, kcal=round(r.kcal * qty, 1),
                protein_g=round(r.protein * qty, 1), carbs_g=round(r.carbs * qty, 1), fat_g=round(r.fat * qty, 1), fiber_g=round(r.fiber * qty, 1))


def totals(items):
    return {k: round(sum(i.get(k, 0) for i in items), 1) for k in ("kcal", "protein_g", "carbs_g", "fat_g", "fiber_g")}


def calc_items(entries):
    """Food calorie calculator. entries: [{food_id, servings}] -> items + totals."""
    items = []
    for e in entries:
        q = float(e.get("servings", 1))
        if not 0 < q <= 20:
            raise ValueError("Servings must be between 0 and 20.")
        items.append(item_for(food_row(e["food_id"]), q))
    return dict(items=items, totals=totals(items))


# ----------------------------------------------------------------------------- body metrics
def bmi(weight_kg, height_cm):
    return round(weight_kg / ((height_cm / 100) ** 2), 1)


def bmi_category(b):
    return "Underweight" if b < 18.5 else "Normal" if b < 25 else "Overweight" if b < 30 else "Obese"


def bmr_mifflin(sex, weight_kg, height_cm, age):
    return 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if sex == "male" else -161)


def parse_allergies(value):
    if isinstance(value, str):
        value = re.split(r"[,;/\s]+", value)
    out = set()
    for a in value or []:
        a = str(a).strip().lower()
        if a and a not in ("none", "no", "nil"):
            out.add(ALLERGEN_SYNONYMS.get(a, a))
    return out


def validate(inp):
    try:
        d = dict(age=float(inp["age"]), height_cm=float(inp["height_cm"]), weight_kg=float(inp["weight_kg"]))
    except (KeyError, TypeError, ValueError):
        raise PlanError("Age, height and current weight are required numbers.")
    d["sex"] = str(inp.get("sex", "")).lower()
    d["sex"] = {"m": "male", "f": "female"}.get(d["sex"], d["sex"])
    if d["sex"] not in ("male", "female"):
        raise PlanError("Select gender (male or female).")
    if not 18 <= d["age"] <= 90:
        raise PlanError("This planner is for adults aged 18-90. Children and teenagers need a paediatric dietitian.")
    if not 120 <= d["height_cm"] <= 230 or not 30 <= d["weight_kg"] <= 250:
        raise PlanError("Height must be 120-230 cm and weight 30-250 kg.")
    d["activity"] = inp.get("activity", "light")
    if d["activity"] not in ACTIVITY:
        raise PlanError("Choose a valid activity level.")
    d["goal"] = inp.get("goal", "healthy")
    if d["goal"] not in GOALS:
        raise PlanError("Choose a valid goal.")
    d["preference"] = inp.get("preference", "vegetarian")
    if d["preference"] not in PREFERENCES:
        raise PlanError("Choose a valid dietary preference.")
    d["allergies"] = sorted(parse_allergies(inp.get("allergies")))
    d["conditions"] = [c for c in (inp.get("conditions") or []) if c in CONDITIONS]
    tw = inp.get("target_weight_kg")
    d["target_weight_kg"] = float(tw) if tw not in (None, "") else d["weight_kg"]
    if not 30 <= d["target_weight_kg"] <= 250:
        raise PlanError("Target weight must be 30-250 kg.")
    return d


def compute_targets(d):
    """Safe calorie / macro / water targets. Raises PlanError for unsafe requests."""
    warnings = []
    if "pregnancy" in d["conditions"]:
        raise PlanError("Nutrition needs change during pregnancy and breast-feeding. Please follow a plan from your obstetrician or dietitian.")
    b = bmi(d["weight_kg"], d["height_cm"])
    tb = bmi(d["target_weight_kg"], d["height_cm"])
    goal = d["goal"]
    if goal == "lose":
        if b < 18.5:
            raise PlanError("Your BMI is below 18.5, so a weight-loss plan is not safe. Please see a doctor or dietitian.")
        if d["target_weight_kg"] >= d["weight_kg"]:
            raise PlanError("For weight loss, the target weight must be lower than your current weight.")
        if tb < 18.5:
            raise PlanError(f"Target weight {d['target_weight_kg']} kg gives BMI {tb}, below the healthy minimum of 18.5. Choose a higher target.")
    if goal == "gain":
        if b >= 25:
            raise PlanError("Your BMI is 25 or above, so a weight-gain plan is not advised. Choose maintenance/healthy eating or consult a dietitian.")
        if d["target_weight_kg"] <= d["weight_kg"]:
            raise PlanError("For weight gain, the target weight must be higher than your current weight.")
        if tb >= 25:
            raise PlanError(f"Target weight {d['target_weight_kg']} kg gives BMI {tb}, above the healthy range. Choose a lower target.")
    bmr = bmr_mifflin(d["sex"], d["weight_kg"], d["height_cm"], d["age"])
    tdee = bmr * ACTIVITY[d["activity"]]
    adj = {"lose": -min(0.20 * tdee, 500), "gain": min(0.12 * tdee, 400)}.get(goal, 0.0)
    kcal = tdee + adj
    floor = 1500 if d["sex"] == "male" else 1200
    if kcal < floor:
        kcal = floor
        warnings.append(f"Calories were raised to the safe minimum of {floor} kcal/day.")
    kcal = int(round(kcal / 10) * 10)
    weekly = round((kcal - tdee) * 7 / 7700, 2)  # kg/week, approximation
    if goal in ("lose", "gain") and weekly:
        weeks = abs(d["target_weight_kg"] - d["weight_kg"]) / abs(weekly)
        warnings.append(f"Estimated pace {weekly:+.2f} kg/week, so roughly {weeks:.0f} weeks to reach the target (estimate only).")
    h = d["height_cm"] / 100
    ref = d["weight_kg"] if b <= 25 else 25 * h * h + 0.25 * (d["weight_kg"] - 25 * h * h)
    factor = {"lose": 1.4, "gain": 1.6}.get(goal, 1.2 if d["age"] >= 65 else 1.0)
    if "kidney" in d["conditions"]:
        factor = 0.8
        warnings.append("Kidney disease: protein is limited to 0.8 g/kg. Potassium, phosphorus and fluid limits are individual, so this plan must be reviewed by your nephrologist/dietitian.")
    protein = min(round(ref * factor), round(kcal * 0.30 / 4))
    fat_pct = 0.25 if {"heart", "hypertension"} & set(d["conditions"]) else 0.28
    fat = round(kcal * fat_pct / 9)
    carbs = round((kcal - 4 * protein - 9 * fat) / 4)
    if carbs < 130:
        fat = max(round(kcal * 0.20 / 9), round((kcal - 4 * protein - 4 * 130) / 9))
        carbs = round((kcal - 4 * protein - 9 * fat) / 4)
    if "diabetes" in d["conditions"]:
        warnings.append("Diabetes: high-GI foods are excluded and carbohydrates are spread across meals. Do not change your medicines based on this plan.")
    if "hypertension" in d["conditions"]:
        warnings.append("Hypertension: high-sodium foods are excluded. Keep added salt below about 5 g/day.")
    if "heart" in d["conditions"]:
        warnings.append("Heart disease: fried and high-saturated-fat foods are excluded. Follow your cardiologist's advice first.")
    fluids_limited = bool({"kidney", "heart"} & set(d["conditions"]))
    water = None if fluids_limited else int(min(3500, max(1500, round(d["weight_kg"] * 35 / 100) * 100)))
    if fluids_limited:
        warnings.append("Fluid intake: ask your doctor for a personal limit. A default tracker target of 2000 ml is shown but may not suit you.")
    if b >= 23:
        warnings.append("For South-Asian adults, health risk starts at a lower BMI (about 23 and above) than the WHO global cut-off of 25.")
    return dict(bmi=b, bmi_category=bmi_category(b), bmr=round(bmr), tdee=round(tdee), calories=kcal, protein_g=int(protein), carbs_g=int(carbs), fat_g=int(fat),
                fiber_g=int(round(kcal / 1000 * 14)), water_ml=water, weekly_change_kg=weekly, warnings=warnings, disclaimer=DISCLAIMER)


# ----------------------------------------------------------------------------- food filtering & ranking
def allowed_foods(d):
    df = foods()
    df = df[df.diet.isin(PREFERENCES[d["preference"]]) & ~df.role.isin(["treat", "side"])]
    allergens = set(d["allergies"])
    df = df[df.allergen_set.map(lambda s: not (s & allergens))]
    banned = set()
    if "diabetes" in d["conditions"]:
        banned |= {"highgi"}
    if "hypertension" in d["conditions"]:
        banned |= {"highsodium"}
    if "heart" in d["conditions"]:
        banned |= {"fried", "highsatfat", "highsodium"}
    if d["goal"] == "lose":
        banned |= {"fried"}
    return df[df.flag_set.map(lambda s: not (s & banned))]


def _shares(p, c, f):
    v = np.array([4 * p, 4 * c, 9 * f], float)
    return v / v.sum() if v.sum() else v


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _pick(df, role, slot, target_vec, recent, rng):
    c = df[(df.role == role) & df.meal_set.map(lambda s: slot in s)]
    if c.empty:
        return None
    scored = []
    for _, r in c.iterrows():
        s = _cos(_shares(r.protein, r.carbs, r.fat), target_vec) - 0.35 * recent.count(int(r.id)) + rng.random() * 0.08
        scored.append((s, r))
    scored.sort(key=lambda t: -t[0])
    return rng.choice(scored[:4])[1]


def _slot_rows(df, slot, day, target_vec, recent, rng, vegan_like):
    def pick(role):
        return _pick(df, role, slot, target_vec, recent, rng)
    rows = []
    if slot == "breakfast":
        rows = [pick("bmain"), pick("dairy") if day % 2 == 0 else None, pick("fruit")]
    elif slot == "lunch":
        rows = [pick("staple"), pick("legume"), pick("veg"), pick("dairy"), pick("protein") if (day % 3 == 1) else None]
    elif slot == "dinner":
        main = pick("protein") if (day % 2 == 0 or vegan_like) else pick("legume")
        rows = [pick("staple"), main if main is not None else pick("legume"), pick("veg")]
    else:
        rows = [pick("snack"), pick("fruit") if day % 2 else None]
    rows = [r for r in rows if r is not None]
    need = {"breakfast": {"bmain"}, "lunch": {"staple", "veg"}, "dinner": {"staple", "veg"}, "snack": {"snack", "fruit"}}[slot]
    if not (need & {r.role for r in rows}) or (slot in ("lunch", "dinner") and not ({"legume", "protein"} & {r.role for r in rows})):
        raise PlanError("Too few foods remain after applying your preference, allergies and medical restrictions. Please consult a dietitian for a custom plan.")
    return rows


def _scale(rows, target_kcal):
    fixed = sum(r.kcal for r in rows if r.role not in SCALABLE)
    scal = [r for r in rows if r.role in SCALABLE]
    base = sum(r.kcal for r in scal) or 1
    m = float(np.clip((target_kcal - fixed) / base, 0.5, 2.5))
    qty = {int(r.id): (max(0.5, round(m * 2) / 2) if r.role in SCALABLE else 1.0) for r in rows}
    return qty


def _fine_tune(day_slots, target):
    def total():
        return sum(i["kcal"] for s in day_slots.values() for i in s["items"])
    for _ in range(14):
        err = target - total()
        if abs(err) <= 0.04 * target:
            break
        best = None
        for s in day_slots.values():
            for it in s["items"]:
                if it["role"] not in SCALABLE:
                    continue
                step = 0.5 if err > 0 else -0.5
                nq = it["qty"] + step
                if not 0.5 <= nq <= 3:
                    continue
                per = it["kcal"] / it["qty"]
                new_err = abs(err - step * per)
                if new_err < abs(err) and (best is None or new_err < best[0]):
                    best = (new_err, it, nq, per)
        if not best:
            break
        _, it, nq, per = best
        r = food_row(it["food_id"])
        it.update(item_for(r, nq))


def generate_week(d, targets, seed=None):
    df = allowed_foods(d)
    if df.empty:
        raise PlanError("No foods remain after applying your restrictions. Please consult a dietitian.")
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
    target_vec = _shares(targets["protein_g"], targets["carbs_g"], targets["fat_g"])
    vegan_like = d["preference"] in ("vegan", "vegetarian")
    recent, week = [], []
    for day in range(7):
        slots = {}
        for slot, share in SLOT_SHARE.items():
            rows = _slot_rows(df, slot, day, target_vec, recent[-12:], rng, vegan_like)
            qty = _scale(rows, targets["calories"] * share)
            items = [item_for(r, qty[int(r.id)]) for r in rows]
            slots[slot] = dict(items=items)
            recent += [int(r.id) for r in rows]
        _fine_tune(slots, targets["calories"])
        for s in slots.values():
            s.update(totals(s["items"]))
        week.append(dict(day_index=day, day=DAYS[day], meals=slots, **totals([i for s in slots.values() for i in s["items"]])))
    return week


def build_plan(inp, seed=None):
    d = validate(inp)
    t = compute_targets(d)
    week = generate_week(d, t, seed)
    avg = {k: round(sum(day[k] for day in week) / 7, 1) for k in ("kcal", "protein_g", "carbs_g", "fat_g", "fiber_g")}
    adequacy = dict(average=avg, kcal_pct=round(avg["kcal"] / t["calories"] * 100), protein_pct=round(avg["protein_g"] / t["protein_g"] * 100),
                    fiber_pct=round(avg["fiber_g"] / t["fiber_g"] * 100))
    notes = list(t["warnings"])
    if adequacy["protein_pct"] < 85:
        notes.append("Average protein is below target; add a protein food (dal, curd, paneer/tofu, egg, fish or chicken as per your preference).")
    return dict(inputs=d, targets=t, week=week, adequacy=adequacy, notes=notes)


# ----------------------------------------------------------------------------- AI swap suggestions
def alternatives(food_id, n=5, allowed_df=None):
    """k-nearest foods (same role, same meal slots) by standardised nutrient vector."""
    df = foods()
    base = food_row(food_id)
    pool = allowed_df if allowed_df is not None else df[~df.role.isin(["treat", "side"])]
    pool = pool[(pool.role == base.role) & (pool.id != base.id)]
    if pool.empty:
        return []
    cols = ["kcal", "protein", "carbs", "fat", "fiber"]
    sc = StandardScaler().fit(df[cols])
    nn = NearestNeighbors(n_neighbors=min(n, len(pool))).fit(sc.transform(pool[cols]))
    dist, idx = nn.kneighbors(sc.transform(pd.DataFrame([base[cols]], columns=cols)))
    return [dict(food_dict(pool.iloc[i]), distance=round(float(dd), 2), kcal_diff=round(float(pool.iloc[i].kcal - base.kcal), 1)) for dd, i in zip(dist[0], idx[0])]

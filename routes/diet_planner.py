"""Module 4 - AI Diet & Nutrition Planner (plans, calculator, tracking)."""
from datetime import date, datetime, timedelta
from flask import Blueprint, Response, abort, jsonify, render_template, request
from flask_login import current_user
from ml_models import diet_recommender as DR
from models import db, DietPlan, DietMeal, NutritionTracking, HealthMetric
from routes import roles_required, current_patient, api_error, fnum
from utils import context as ctx
from utils.pdf import build_pdf

bp = Blueprint("diet_planner", __name__)
MEALS = ("breakfast", "lunch", "snack", "dinner")


def _own(pid):
    p = db.get_or_404(DietPlan, pid)
    if p.patient_id != current_patient().id:
        abort(403)
    return p


def _plan_json(plan, full=True):
    d = dict(id=plan.id, goal=plan.goal, calories=plan.calories, protein_g=plan.protein_g, carbs_g=plan.carbs_g, fat_g=plan.fat_g, water_ml=plan.water_ml, bmi=plan.bmi,
             bmr=plan.bmr, tdee=plan.tdee, is_active=plan.is_active, created_at=plan.created_at.isoformat(timespec="minutes"))
    if full:
        days = {}
        for m in plan.meals:
            day = days.setdefault(m.day_index, dict(day_index=m.day_index, day=DR.DAYS[m.day_index], meals={}, kcal=0))
            day["meals"][m.meal_type] = dict(id=m.id, items=m.items, kcal=m.kcal, protein_g=m.protein_g, carbs_g=m.carbs_g, fat_g=m.fat_g)
            day["kcal"] += m.kcal or 0
        d.update(inputs=plan.inputs, notes=plan.warnings, week=[days[k] for k in sorted(days)], disclaimer=DR.DISCLAIMER)
    return d


@bp.route("/diet")
@roles_required("patient")
def page():
    p = current_patient()
    return render_template("diet.html", profile=dict(age=p.age, sex=(p.gender or "").lower(), height_cm=p.height_cm, weight_kg=p.weight_kg),
                           activity=list(DR.ACTIVITY), preferences=list(DR.PREFERENCES), conditions=[c for c in DR.CONDITIONS])


@bp.route("/api/diet/metrics", methods=["POST"])
@roles_required("patient")
def metrics():
    """BMI / BMR / calorie-need calculator (no plan is created)."""
    d = request.get_json(silent=True) or {}
    try:
        age, h, w = float(d["age"]), float(d["height_cm"]), float(d["weight_kg"])
        sex = {"m": "male", "f": "female"}.get(str(d.get("sex", "")).lower(), str(d.get("sex", "")).lower())
        act = DR.ACTIVITY[d.get("activity", "light")]
        assert sex in ("male", "female") and 18 <= age <= 90 and 120 <= h <= 230 and 30 <= w <= 250
    except (KeyError, TypeError, ValueError, AssertionError):
        return api_error("Enter valid adult age, gender, height (120-230 cm), weight (30-250 kg) and activity level.")
    b = DR.bmi(w, h); bmr = DR.bmr_mifflin(sex, w, h, age)
    return jsonify(bmi=b, category=DR.bmi_category(b), bmr=round(bmr), maintenance_kcal=round(bmr * act), healthy_weight_range_kg=[round(18.5 * (h / 100) ** 2, 1), round(24.9 * (h / 100) ** 2, 1)])


@bp.route("/api/diet/plan", methods=["POST"])
@roles_required("patient")
def create_plan():
    try:
        res = DR.build_plan(request.get_json(silent=True) or {})
    except DR.PlanError as e:
        return api_error(str(e), 422)
    t, p = res["targets"], current_patient()
    DietPlan.query.filter_by(patient_id=p.id, is_active=True).update({"is_active": False})
    plan = DietPlan(patient_id=p.id, goal=res["inputs"]["goal"], bmi=t["bmi"], bmr=t["bmr"], tdee=t["tdee"], calories=t["calories"], protein_g=t["protein_g"], carbs_g=t["carbs_g"],
                    fat_g=t["fat_g"], water_ml=t["water_ml"], inputs=res["inputs"], warnings=res["notes"], is_active=True)
    for day in res["week"]:
        for slot, m in day["meals"].items():
            plan.meals.append(DietMeal(day_index=day["day_index"], meal_type=slot, items=m["items"], kcal=m["kcal"], protein_g=m["protein_g"], carbs_g=m["carbs_g"], fat_g=m["fat_g"]))
    db.session.add(plan); db.session.commit()
    out = _plan_json(plan); out["targets"] = t; out["adequacy"] = res["adequacy"]
    return jsonify(out), 201


@bp.route("/api/diet/plans")
@roles_required("patient")
def plans():
    return jsonify([_plan_json(x, full=False) for x in DietPlan.query.filter_by(patient_id=current_patient().id).order_by(DietPlan.id.desc()).limit(20)])


@bp.route("/api/diet/plans/<int:pid>")
@roles_required("patient")
def plan(pid):
    return jsonify(_plan_json(_own(pid)))


@bp.route("/api/diet/plans/<int:pid>", methods=["DELETE"])
@roles_required("patient")
def delete_plan(pid):
    db.session.delete(_own(pid)); db.session.commit()
    return jsonify(ok=True)


@bp.route("/api/diet/plans/<int:pid>/swap", methods=["POST"])
@roles_required("patient")
def swap(pid):
    plan, d = _own(pid), request.get_json(silent=True) or {}
    meal = DietMeal.query.filter_by(plan_id=plan.id, day_index=d.get("day_index"), meal_type=d.get("meal_type")).first()
    if not meal:
        return api_error("Meal not found.", 404)
    allowed = DR.allowed_foods(DR.validate(plan.inputs))
    new = allowed[allowed.id == int(d.get("new_food_id", -1))]
    if new.empty:
        return api_error("That food conflicts with your preference, allergies or medical restrictions.", 422)
    items = [dict(i) for i in meal.items]
    idx = next((k for k, i in enumerate(items) if i["food_id"] == int(d.get("food_id", -1))), None)
    if idx is None:
        return api_error("Item not in this meal.", 404)
    items[idx] = DR.item_for(new.iloc[0], items[idx]["qty"])
    t = DR.totals(items)
    meal.items, meal.kcal, meal.protein_g, meal.carbs_g, meal.fat_g = items, t["kcal"], t["protein_g"], t["carbs_g"], t["fat_g"]
    db.session.commit()
    return jsonify(items=items, totals=t)


@bp.route("/api/diet/plans/<int:pid>/pdf")
@roles_required("patient")
def plan_pdf(pid):
    plan = _own(pid)
    secs = [dict(heading="Daily targets", paragraphs=[f"Goal: {plan.goal}   BMI {plan.bmi}   BMR {plan.bmr} kcal   Maintenance {plan.tdee} kcal",
                                                      f"Target: {plan.calories} kcal | Protein {plan.protein_g} g | Carbs {plan.carbs_g} g | Fat {plan.fat_g} g | Water " + (f"{plan.water_ml} ml" if plan.water_ml else "as advised by your doctor")]),
            dict(heading="Safety notes", notes=(plan.warnings or []) + [DR.DISCLAIMER])]
    for day in _plan_json(plan)["week"]:
        rows = [[slot.title(), "; ".join(f"{i['name']} x{i['qty']:g} ({i['serving']})" for i in m["items"]), round(m["kcal"])] for slot, m in day["meals"].items()]
        secs.append(dict(heading=f"{day['day']} - {round(day['kcal'])} kcal", table=dict(header=["Meal", "Foods", "kcal"], rows=rows, widths=[22, 128, 20])))
    return Response(build_pdf("Weekly Diet Plan", f"Patient: {current_user.name}   |   Plan #{plan.id}, created {plan.created_at:%d %b %Y}", secs, DR.DISCLAIMER), mimetype="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=diet_plan_{plan.id}.pdf"})


@bp.route("/api/diet/foods")
@roles_required("patient")
def food_search():
    return jsonify(DR.search_foods(request.args.get("q", ""), request.args.get("diet"), 30))


@bp.route("/api/diet/foods/<int:fid>/alternatives")
@roles_required("patient")
def alternatives(fid):
    try:
        plan = ctx.active_plan(current_patient().id)
        pool = DR.allowed_foods(DR.validate(plan.inputs)) if plan else None
        return jsonify(DR.alternatives(fid, 5, pool))
    except KeyError:
        return api_error("Food not found.", 404)


@bp.route("/api/diet/calc", methods=["POST"])
@roles_required("patient")
def calc():
    try:
        return jsonify(DR.calc_items((request.get_json(silent=True) or {}).get("items", [])))
    except (KeyError, ValueError, TypeError):
        return api_error("Invalid food or servings.")


# ---------------------------------------------------------------- tracking
def _day(s):
    try:
        d = date.fromisoformat(s) if s else date.today()
    except ValueError:
        d = date.today()
    return d


@bp.route("/api/nutrition/summary")
@roles_required("patient")
def summary():
    pid, day = current_patient().id, _day(request.args.get("date"))
    plan = ctx.active_plan(pid)
    tot = ctx.day_totals(pid, day)
    entries = NutritionTracking.query.filter_by(patient_id=pid, log_date=day).order_by(NutritionTracking.id).all()
    hist = []
    for i in range(6, -1, -1):
        dd = date.today() - timedelta(days=i)
        t = ctx.day_totals(pid, dd)
        hist.append(dict(date=dd.isoformat(), kcal=t["kcal"], water_ml=t["water_ml"]))
    wts = HealthMetric.query.filter(HealthMetric.patient_id == pid, HealthMetric.weight_kg.isnot(None)).order_by(HealthMetric.recorded_at).all()
    target = dict(kcal=plan.calories, protein_g=plan.protein_g, carbs_g=plan.carbs_g, fat_g=plan.fat_g, water_ml=plan.water_ml or 2000, water_is_default=plan.water_ml is None) if plan else None
    goal_w = (plan.inputs or {}).get("target_weight_kg") if plan else None
    return jsonify(date=day.isoformat(), totals=tot, target=target, history=hist, target_weight_kg=goal_w,
                   entries=[dict(id=e.id, type=e.entry_type, meal=e.meal_type, name=e.food_name, servings=e.servings, kcal=e.kcal, water_ml=e.water_ml) for e in entries],
                   weights=[dict(date=w.recorded_at.date().isoformat(), kg=w.weight_kg) for w in wts][-60:])


@bp.route("/api/nutrition/log", methods=["POST"])
@roles_required("patient")
def log_food():
    d = request.get_json(silent=True) or {}
    meal = d.get("meal_type", "snack")
    if meal not in MEALS:
        return api_error("Invalid meal type.")
    try:
        r = DR.calc_items([dict(food_id=d["food_id"], servings=d.get("servings", 1))])["items"][0]
    except (KeyError, ValueError, TypeError):
        return api_error("Choose a food and servings between 0 and 20.")
    row = NutritionTracking(patient_id=current_patient().id, log_date=_day(d.get("date")), entry_type="food", meal_type=meal, food_id=r["food_id"], food_name=r["name"], servings=r["qty"],
                            kcal=r["kcal"], protein_g=r["protein_g"], carbs_g=r["carbs_g"], fat_g=r["fat_g"])
    db.session.add(row); db.session.commit()
    return jsonify(id=row.id, totals=ctx.day_totals(current_patient().id, row.log_date)), 201


@bp.route("/api/nutrition/water", methods=["POST"])
@roles_required("patient")
def log_water():
    ml = fnum((request.get_json(silent=True) or {}).get("ml"))
    if ml is None or not 50 <= ml <= 2000:
        return api_error("Water amount must be between 50 and 2000 ml per entry.")
    row = NutritionTracking(patient_id=current_patient().id, log_date=_day((request.get_json() or {}).get("date")), entry_type="water", food_name="Water", water_ml=int(ml))
    db.session.add(row); db.session.commit()
    return jsonify(id=row.id, totals=ctx.day_totals(current_patient().id, row.log_date)), 201


@bp.route("/api/nutrition/<int:eid>", methods=["DELETE"])
@roles_required("patient")
def delete_entry(eid):
    e = db.get_or_404(NutritionTracking, eid)
    if e.patient_id != current_patient().id:
        abort(403)
    db.session.delete(e); db.session.commit()
    return jsonify(ok=True)


@bp.route("/api/nutrition/weight", methods=["POST"])
@roles_required("patient")
def log_weight():
    w = fnum((request.get_json(silent=True) or {}).get("weight_kg"))
    if w is None or not 20 <= w <= 300:
        return api_error("Weight must be between 20 and 300 kg.")
    db.session.add(HealthMetric(patient_id=current_patient().id, weight_kg=w, source="nutrition"))
    current_patient().weight_kg = w
    db.session.commit()
    return jsonify(ok=True, weight_kg=w, bmi=current_patient().bmi), 201

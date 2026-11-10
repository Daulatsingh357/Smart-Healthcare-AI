"""Cross-module data helpers (used by the unified dashboard, chatbot and analytics)."""
from datetime import date, datetime, timedelta
from sqlalchemy import func
from models import (db, DiseasePrediction, MedicalReport, HealthMetric, NutritionTracking, DietPlan, ExtractedReportValue)

DISEASES = ("diabetes", "heart", "kidney", "liver")


def metric_rows(pid, since=None):
    q = HealthMetric.query.filter_by(patient_id=pid)
    if since:
        q = q.filter(HealthMetric.recorded_at >= since)
    return [dict(id=m.id, recorded_at=m.recorded_at, systolic=m.systolic, diastolic=m.diastolic, blood_sugar=m.blood_sugar,
                 heart_rate=m.heart_rate, weight_kg=m.weight_kg, cholesterol=m.cholesterol, source=m.source)
            for m in q.order_by(HealthMetric.recorded_at).all()]


def latest_predictions(pid):
    out = {}
    for d in DISEASES:
        p = DiseasePrediction.query.filter_by(patient_id=pid, disease=d).order_by(DiseasePrediction.created_at.desc(), DiseasePrediction.id.desc()).first()
        if p:
            out[d] = p
    return out


def latest_report(pid):
    return MedicalReport.query.filter_by(patient_id=pid).order_by(MedicalReport.report_date.desc(), MedicalReport.id.desc()).first()


def active_plan(pid):
    return DietPlan.query.filter_by(patient_id=pid, is_active=True).order_by(DietPlan.id.desc()).first()


def day_totals(pid, day):
    rows = NutritionTracking.query.filter_by(patient_id=pid, log_date=day).all()
    food = [r for r in rows if r.entry_type == "food"]
    return dict(kcal=round(sum(r.kcal or 0 for r in food)), protein_g=round(sum(r.protein_g or 0 for r in food), 1), carbs_g=round(sum(r.carbs_g or 0 for r in food), 1),
                fat_g=round(sum(r.fat_g or 0 for r in food), 1), water_ml=int(sum(r.water_ml or 0 for r in rows if r.entry_type == "water")), entries=len(rows))


def nutrition_stats(pid, days=7):
    start = date.today() - timedelta(days=days - 1)
    rows = NutritionTracking.query.filter(NutritionTracking.patient_id == pid, NutritionTracking.log_date >= start).all()
    kc, wt = {}, {}
    for r in rows:
        if r.entry_type == "food":
            kc[r.log_date] = kc.get(r.log_date, 0) + (r.kcal or 0)
        else:
            wt[r.log_date] = wt.get(r.log_date, 0) + (r.water_ml or 0)
    plan = active_plan(pid)
    return dict(days_logged=len(kc), avg_kcal=round(sum(kc.values()) / len(kc)) if kc else 0, target_kcal=plan.calories if plan else None,
                water_days=len(wt), avg_water_ml=round(sum(wt.values()) / len(wt)) if wt else 0,
                water_limit_ok=not (plan and plan.water_ml is None))


def abnormal_values(report_id):
    return [dict(name=v.param_name, key=v.param_key, status=v.status, value=v.value, unit=v.unit)
            for v in ExtractedReportValue.query.filter(ExtractedReportValue.report_id == report_id,
                                                       ExtractedReportValue.status.in_(["Low", "High"])).all()]


def recommendation_ctx(patient, trends=None, anomalies_count=0, n_readings=0):
    preds = latest_predictions(patient.id)
    rep = latest_report(patient.id)
    return dict(bmi=patient.bmi, high_risk=[d for d, p in preds.items() if p.risk_level == "High"],
                moderate_risk=[d for d, p in preds.items() if p.risk_level == "Moderate"], trends=trends or {},
                abnormal_params=abnormal_values(rep.id) if rep else [], has_report=bool(rep), anomalies_count=anomalies_count,
                n_readings=n_readings, nutrition=nutrition_stats(patient.id))

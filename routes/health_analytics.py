"""Module 5 - Predictive Health Analytics."""
import calendar
from datetime import date, datetime, timedelta
from flask import Blueprint, Response, abort, jsonify, render_template, request
from flask_login import current_user
from ml_models import health_analyzer as H
from models import db, HealthMetric, HealthAnalytics, DiseasePrediction, MedicalReport, NutritionTracking
from routes import roles_required, current_patient, api_error, fnum
from utils import context as ctx
from utils.pdf import build_pdf

bp = Blueprint("health_analytics", __name__)
VALID = {"systolic": (60, 260), "diastolic": (30, 160), "blood_sugar": (30, 700), "heart_rate": (25, 230), "weight_kg": (20, 300), "cholesterol": (50, 600)}


def _iso(rows_dt):
    return [d.isoformat(timespec="minutes") for d in rows_dt]


def build_overview(p):
    rows = ctx.metric_rows(p.id)
    series, trends, forecasts = {}, {}, {}
    for k, (label, unit, _) in H.METRICS.items():
        s = H.series_of(rows, k)
        if not s:
            continue
        series[k] = dict(label=label, unit=unit, dates=_iso([d for d, _ in s]), values=[round(v, 1) for _, v in s], rolling=H.rolling_mean(s))
        trends[k] = H.trend(s, k)
        forecasts[k] = H.forecast(s, k)
    anomalies = H.detect_anomalies(rows)
    recent = {}
    cutoff = datetime.utcnow() - timedelta(days=14)
    for r in rows:
        if r["recorded_at"] >= cutoff:
            for k in H.METRICS:
                if r.get(k) is not None:
                    recent[k] = r[k]
    risk = {}
    for dis in ctx.DISEASES:
        pts = [(x.created_at, x.risk_percent) for x in DiseasePrediction.query.filter_by(patient_id=p.id, disease=dis).order_by(DiseasePrediction.created_at, DiseasePrediction.id)]
        if pts:
            risk[dis] = H.risk_trend(pts)
    recs = H.recommendations(ctx.recommendation_ctx(p, trends, len(anomalies["anomalies"]), len(rows)))
    return dict(readings=len(rows), series=series, trends=trends, forecasts=forecasts, anomalies=anomalies, alerts=H.threshold_alerts(recent), risk_trends=risk, recommendations=recs,
                requirements=dict(min_points=H.MIN_POINTS, min_span_days=H.MIN_SPAN_DAYS, min_anomaly_rows=H.MIN_ANOMALY_ROWS),
                disclaimer="Trends and forecasts are statistical estimates from your own logged data, not diagnoses or predictions of illness.")


def build_monthly(p, month):
    y, m = (int(x) for x in month.split("-"))
    start, end = date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])
    pm = start - timedelta(days=1)
    pstart = date(pm.year, pm.month, 1)
    rows = ctx.metric_rows(p.id)
    inr = lambda r, a, b: a <= r["recorded_at"].date() <= b
    cur, prev = [r for r in rows if inr(r, start, end)], [r for r in rows if inr(r, pstart, pm)]
    metrics = {k: H.summarize_metric(cur, prev, k) for k in H.METRICS}
    preds = DiseasePrediction.query.filter(DiseasePrediction.patient_id == p.id, DiseasePrediction.created_at >= datetime.combine(start, datetime.min.time()),
                                           DiseasePrediction.created_at < datetime.combine(end + timedelta(days=1), datetime.min.time())).order_by(DiseasePrediction.created_at).all()
    reps = MedicalReport.query.filter(MedicalReport.patient_id == p.id, MedicalReport.report_date >= start, MedicalReport.report_date <= end).all()
    nt = NutritionTracking.query.filter(NutritionTracking.patient_id == p.id, NutritionTracking.log_date >= start, NutritionTracking.log_date <= end).all()
    kcal, water = {}, {}
    for r in nt:
        (kcal if r.entry_type == "food" else water)[r.log_date] = (kcal if r.entry_type == "food" else water).get(r.log_date, 0) + ((r.kcal if r.entry_type == "food" else r.water_ml) or 0)
    an = H.detect_anomalies(rows)
    an_month = [a for a in an["anomalies"] if start.isoformat() <= a["recorded_at"][:10] <= end.isoformat()]
    return dict(month=month, period=[start.isoformat(), end.isoformat()], readings=len(cur), metrics=metrics,
                predictions=[dict(disease=x.disease, risk=x.risk_percent, level=x.risk_level, date=x.created_at.date().isoformat()) for x in preds],
                reports=dict(count=len(reps), abnormal_values=sum(r.abnormal_count or 0 for r in reps)),
                nutrition=dict(days_logged=len(kcal), avg_kcal=round(sum(kcal.values()) / len(kcal)) if kcal else None, water_days=len(water), avg_water_ml=round(sum(water.values()) / len(water)) if water else None),
                anomalies=an_month, anomaly_method=an["method"], recommendations=H.recommendations(ctx.recommendation_ctx(p, {k: H.trend(H.series_of(rows, k), k) for k in H.METRICS if H.series_of(rows, k)}, len(an_month), len(rows)))[:6])


def _month_arg():
    m = request.args.get("month") or date.today().strftime("%Y-%m")
    try:
        datetime.strptime(m, "%Y-%m")
    except ValueError:
        abort(400, "month must be YYYY-MM")
    return m


@bp.route("/analytics")
@roles_required("patient")
def page():
    return render_template("analytics.html", metrics={k: dict(label=v[0], unit=v[1], lo=VALID[k][0], hi=VALID[k][1]) for k, v in H.METRICS.items()})


@bp.route("/api/metrics", methods=["POST"])
@roles_required("patient")
def add_metric():
    d = request.get_json(silent=True) or {}
    vals, errors = {}, {}
    for k, (lo, hi) in VALID.items():
        v = fnum(d.get(k))
        if v is None:
            if str(d.get(k, "")).strip():
                errors[k] = "Must be a number."
            continue
        if not lo <= v <= hi:
            errors[k] = f"{H.LABEL[k]} must be between {lo} and {hi}."
        else:
            vals[k] = v
    if not vals and not errors:
        return api_error("Enter at least one measurement.")
    if "systolic" in vals and "diastolic" in vals and vals["systolic"] <= vals["diastolic"]:
        errors["systolic"] = "Systolic must be higher than diastolic."
    when = datetime.utcnow()
    if d.get("recorded_at"):
        try:
            when = datetime.fromisoformat(str(d["recorded_at"]))
        except ValueError:
            errors["recorded_at"] = "Invalid date/time."
        else:
            if when > datetime.utcnow() + timedelta(days=1):
                errors["recorded_at"] = "Date cannot be in the future."
    if errors:
        return api_error("Please correct the highlighted fields.", 422, fields=errors)
    row = HealthMetric(patient_id=current_patient().id, recorded_at=when, source="manual", **vals)
    db.session.add(row)
    if "weight_kg" in vals and when.date() >= date.today() - timedelta(days=1):
        current_patient().weight_kg = vals["weight_kg"]
    db.session.commit()
    return jsonify(id=row.id, alerts=H.threshold_alerts(vals)), 201


@bp.route("/api/metrics")
@roles_required("patient")
def list_metrics():
    rows = ctx.metric_rows(current_patient().id)[-int(request.args.get("limit", 100)):]
    return jsonify([dict(r, recorded_at=r["recorded_at"].isoformat(timespec="minutes")) for r in reversed(rows)])


@bp.route("/api/metrics/<int:mid>", methods=["DELETE"])
@roles_required("patient")
def delete_metric(mid):
    m = db.get_or_404(HealthMetric, mid)
    if m.patient_id != current_patient().id:
        abort(403)
    db.session.delete(m); db.session.commit()
    return jsonify(ok=True)


@bp.route("/api/analytics/overview")
@roles_required("patient")
def overview():
    return jsonify(build_overview(current_patient()))


@bp.route("/api/analytics/run", methods=["POST"])
@roles_required("patient")
def run():
    p, ov = current_patient(), build_overview(current_patient())
    snap = HealthAnalytics(patient_id=p.id, analysis_type="snapshot", period_end=date.today(),
                           result=dict(trends=ov["trends"], forecasts={k: dict(reliable=v["reliable"], model=v.get("model"), reason=v.get("reason", "")) for k, v in ov["forecasts"].items()},
                                       anomalies=ov["anomalies"], risk_trends={k: {x: y for x, y in v.items() if x not in ("series", "dates")} for k, v in ov["risk_trends"].items()}, readings=ov["readings"]))
    month = date.today().strftime("%Y-%m")
    ms = build_monthly(p, month)
    db.session.add(snap)
    db.session.add(HealthAnalytics(patient_id=p.id, analysis_type="monthly_summary", period_start=date.fromisoformat(ms["period"][0]), period_end=date.fromisoformat(ms["period"][1]), result=ms))
    db.session.commit()
    return jsonify(saved=True, snapshot_id=snap.id, overview=ov), 201


@bp.route("/api/analytics/history")
@roles_required("patient")
def history():
    rows = HealthAnalytics.query.filter_by(patient_id=current_patient().id).order_by(HealthAnalytics.id.desc()).limit(20).all()
    return jsonify([dict(id=r.id, type=r.analysis_type, created_at=r.created_at.isoformat(timespec="minutes"), period=[str(r.period_start) if r.period_start else None, str(r.period_end) if r.period_end else None]) for r in rows])


@bp.route("/api/analytics/monthly")
@roles_required("patient")
def monthly():
    return jsonify(build_monthly(current_patient(), _month_arg()))


@bp.route("/api/analytics/report.pdf")
@roles_required("patient")
def report_pdf():
    p, month = current_patient(), _month_arg()
    ms, ov = build_monthly(p, month), build_overview(p)
    mrows = [[H.LABEL[k], v["n"], v.get("mean", "-"), v.get("min", "-"), v.get("max", "-"), v.get("change", "-")] for k, v in ms["metrics"].items() if v["n"]]
    trows = [[H.LABEL[k], t["direction"], f"{t['slope_per_week']:+}/week", t["r2"], t["n"]] for k, t in ov["trends"].items() if t.get("available")]
    frows = []
    for k, f in ov["forecasts"].items():
        frows.append([H.LABEL[k], f["model"] + f" - 7 days ahead: {f['points'][-1]['estimate']} (95% range {f['points'][-1]['low']} to {f['points'][-1]['high']})" if f["reliable"] else "No forecast: " + f["reason"]])
    secs = [dict(heading=f"Monthly summary: {month}", paragraphs=[f"{ms['readings']} reading(s) logged. Predictions this month: {len(ms['predictions'])}. Reports: {ms['reports']['count']} ({ms['reports']['abnormal_values']} out-of-range value(s))."],
                 table=dict(header=["Metric", "Readings", "Mean", "Min", "Max", "Change vs last month"], rows=mrows, widths=[38, 22, 22, 22, 22, 44]) if mrows else None),
            dict(heading="Trends (linear, descriptive)", table=dict(header=["Metric", "Direction", "Slope", "R squared", "n"], rows=trows, widths=[40, 30, 36, 30, 20]) if trows else None),
            dict(heading="Forecasts (ESTIMATES - shown only when enough data exist)", table=dict(header=["Metric", "Result"], rows=frows, widths=[34, 140]) if frows else None,
                 paragraphs=[] if frows else ["No metric has enough data yet."]),
            dict(heading="Anomaly detection", paragraphs=[ov["anomalies"]["method"] + ((": " + ov["anomalies"]["reason"]) if ov["anomalies"]["reason"] else "")],
                 notes=[f"{a['label']} {a['value']} on {a['recorded_at']} (usual {a['typical']})" for a in ov["anomalies"]["anomalies"][:8]]),
            dict(heading="Personalised preventive suggestions", notes=[r["text"] for r in ov["recommendations"][:10]])]
    for s in secs:
        if s.get("table") is None:
            s.pop("table", None)
    return Response(build_pdf("Health Analytics Report", f"Patient: {current_user.name}   |   Month {month}", secs, ov["disclaimer"] + " Not a medical diagnosis."), mimetype="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=health_analytics_{month}.pdf"})

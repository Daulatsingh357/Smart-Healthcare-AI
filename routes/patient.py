"""Patient dashboard (unified AI dashboard), profile, medicines and feedback."""
from datetime import date, datetime, timedelta
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import current_user
from models import db, DiseasePrediction, MedicalReport, Medicine, Appointment, Doctor, Feedback, ChatConversation, HealthMetric
from routes import roles_required, fnum
from utils import context as ctx
from ml_models import health_analyzer as H, diet_recommender as DR

bp = Blueprint("patient", __name__)


def me():
    return current_user.patient


@bp.route("/dashboard")
@roles_required("patient", "doctor", "admin")
def dashboard():
    if current_user.role == "admin":
        return redirect(url_for("admin.panel"))
    if current_user.role == "doctor":
        return redirect(url_for("appointments.list_"))
    p = me()
    preds = DiseasePrediction.query.filter_by(patient_id=p.id).order_by(DiseasePrediction.created_at).all()
    latest = ctx.latest_predictions(p.id)
    rows = ctx.metric_rows(p.id)
    trends = {k: H.trend(H.series_of(rows, k), k) for k in H.METRICS if H.series_of(rows, k)}
    an = H.detect_anomalies(rows)
    reports = MedicalReport.query.filter_by(patient_id=p.id).order_by(MedicalReport.report_date.desc(), MedicalReport.id.desc()).all()
    plan = ctx.active_plan(p.id)
    today = ctx.day_totals(p.id, date.today())
    recs = H.recommendations(ctx.recommendation_ctx(p, trends, len(an["anomalies"]), len(rows)))[:5]
    latest_vitals = {}
    for r in rows:
        for k in H.METRICS:
            if r.get(k) is not None:
                latest_vitals[k] = r[k]
    weights = [(r["recorded_at"].strftime("%d %b"), r["weight_kg"]) for r in rows if r.get("weight_kg") is not None][-30:]
    return render_template("patient_dashboard.html", p=p, bmi=p.bmi, bmi_cat=DR.bmi_category(p.bmi) if p.bmi else None, preds=preds, latest=latest, reports=reports[:3], n_reports=len(reports),
                           abnormal=sum(r.abnormal_count or 0 for r in reports[:1]), plan=plan, today=today, recs=recs, vitals=latest_vitals, weights=weights, n_readings=len(rows),
                           n_chats=ChatConversation.query.filter_by(patient_id=p.id).count(), anomalies=len(an["anomalies"]),
                           appt_count=Appointment.query.filter_by(patient_id=p.id).count(),
                           meds=Medicine.query.filter_by(patient_id=p.id).filter(Medicine.end_date >= date.today()).all(),
                           doctors=Doctor.query.order_by(Doctor.experience_years.desc()).limit(3).all())


@bp.route("/profile", methods=["POST"])
@roles_required("patient")
def profile():
    p = me()
    age = fnum(request.form.get("age")); h = fnum(request.form.get("height_cm")); w = fnum(request.form.get("weight_kg"))
    if (age is not None and not 1 <= age <= 120) or (h is not None and not 50 <= h <= 250) or (w is not None and not 2 <= w <= 300):
        flash("Please enter realistic age, height and weight values.", "danger")
        return redirect(url_for("patient.dashboard"))
    p.age = int(age) if age else None
    p.gender = request.form.get("gender") if request.form.get("gender") in ("Male", "Female") else p.gender
    p.height_cm, p.weight_kg = h, w
    db.session.commit()
    flash("Profile updated.", "success")
    return redirect(url_for("patient.dashboard"))


@bp.route("/medicines", methods=["GET", "POST"])
@roles_required("patient")
def medicines():
    p = me()
    if request.method == "POST":
        f = request.form
        try:
            db.session.add(Medicine(patient_id=p.id, name=f["name"], dosage=f["dosage"], reminder_time=f["time"], start_date=date.fromisoformat(f["start"]), end_date=date.fromisoformat(f["end"])))
            db.session.commit(); flash("Reminder added.", "success")
        except (KeyError, ValueError):
            flash("Invalid medicine details.", "danger")
    return render_template("medicines.html", meds=Medicine.query.filter_by(patient_id=p.id).order_by(Medicine.id.desc()).all(), today=date.today())


@bp.route("/medicines/<int:mid>/delete", methods=["POST"])
@roles_required("patient")
def del_med(mid):
    m = db.get_or_404(Medicine, mid)
    if m.patient_id == me().id:
        db.session.delete(m); db.session.commit()
    return redirect(url_for("patient.medicines"))


@bp.route("/feedback", methods=["POST"])
@roles_required("patient", "doctor")
def feedback():
    if request.form.get("message", "").strip():
        db.session.add(Feedback(user_id=current_user.id, message=request.form["message"][:1000])); db.session.commit()
        flash("Thanks for your feedback.", "success")
    return redirect(url_for("patient.dashboard"))

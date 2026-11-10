from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, abort
from flask_login import current_user
from models import db, Doctor, User, Appointment
from routes import roles_required
bp = Blueprint("appointments", __name__)

@bp.route("/doctors")
@roles_required("patient")
def doctors():
    q = request.args.get("q", "").strip()
    query = Doctor.query.join(User)
    if q: query = query.filter(db.or_(User.name.ilike(f"%{q}%"), Doctor.specialization.ilike(f"%{q}%")))
    return render_template("doctors.html", doctors=query.all(), q=q)

@bp.route("/book/<int:did>", methods=["POST"])
@roles_required("patient")
def book(did):
    db.get_or_404(Doctor, did)
    try:
        when = datetime.fromisoformat(request.form["when"])
        if when < datetime.now(): raise ValueError
    except (KeyError, ValueError):
        flash("Choose a valid future date and time.", "danger"); return redirect(url_for("appointments.doctors"))
    if Appointment.query.filter_by(doctor_id=did, scheduled_at=when).filter(Appointment.status != "Cancelled").first():
        flash("That slot is taken.", "danger"); return redirect(url_for("appointments.doctors"))
    db.session.add(Appointment(patient_id=current_user.patient.id, doctor_id=did, scheduled_at=when, reason=request.form.get("reason", "")[:255]))
    db.session.commit(); flash("Appointment requested.", "success")
    return redirect(url_for("appointments.list_"))

@bp.route("/appointments")
@roles_required("patient", "doctor")
def list_():
    q = Appointment.query
    q = q.filter_by(patient_id=current_user.patient.id) if current_user.role == "patient" else q.filter_by(doctor_id=current_user.doctor.id)
    return render_template("appointments.html", appts=q.order_by(Appointment.scheduled_at.desc()).all())

@bp.route("/appointments/<int:aid>/<action>", methods=["POST"])
@roles_required("patient", "doctor", "admin")
def update(aid, action):
    a = db.get_or_404(Appointment, aid)
    owns = (current_user.role == "patient" and a.patient_id == current_user.patient.id) or \
           (current_user.role == "doctor" and a.doctor_id == current_user.doctor.id) or current_user.role == "admin"
    allowed = {"cancel": "Cancelled"} if current_user.role == "patient" else {"cancel": "Cancelled", "confirm": "Confirmed", "complete": "Completed"}
    if not owns or action not in allowed: abort(403)
    a.status = allowed[action]; db.session.commit()
    return redirect(request.referrer or url_for("appointments.list_"))

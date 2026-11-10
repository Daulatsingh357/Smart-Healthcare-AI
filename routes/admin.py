from flask import Blueprint, render_template, redirect, url_for, flash
from sqlalchemy import func
from ml_models import common
from models import db, User, Patient, Doctor, Appointment, Feedback, DiseasePrediction, MedicalReport, ChatConversation, DietPlan, PATIENT_OWNED
from routes import roles_required
bp = Blueprint("admin", __name__, url_prefix="/admin")


@bp.route("/")
@roles_required("admin")
def panel():
    stats = dict(patients=Patient.query.count(), doctors=Doctor.query.count(), appointments=Appointment.query.count(), predictions=DiseasePrediction.query.count(),
                 reports=MedicalReport.query.count(), chats=ChatConversation.query.count(), diet_plans=DietPlan.query.count())
    by_status = dict(db.session.query(Appointment.status, func.count()).group_by(Appointment.status).all())
    return render_template("admin_dashboard.html", stats=stats, by_status=by_status, metrics=common.load_metrics(),
                           users=User.query.filter(User.role != "admin").all(), feedback=Feedback.query.order_by(Feedback.id.desc()).limit(20).all())


@bp.route("/users/<int:uid>/delete", methods=["POST"])
@roles_required("admin")
def delete_user(uid):
    u = db.get_or_404(User, uid)
    if u.role != "admin":
        if u.patient is not None:
            for model in PATIENT_OWNED:  # ORM delete so cascades remove child rows
                for row in model.query.filter_by(patient_id=u.patient.id).all():
                    db.session.delete(row)
            db.session.delete(u.patient)
        if u.doctor is not None:
            Appointment.query.filter_by(doctor_id=u.doctor.id).delete()
            db.session.delete(u.doctor)
        Feedback.query.filter_by(user_id=u.id).delete()
        db.session.delete(u); db.session.commit(); flash("User removed.", "success")
    return redirect(url_for("admin.panel"))

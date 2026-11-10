import re
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Patient, Doctor
bp = Blueprint("auth", __name__)

@bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        f = request.form; email = f.get("email", "").strip().lower(); role = f.get("role")
        if role not in ("patient", "doctor"): role = "patient"
        if not re.match(r"[^@]+@[^@]+\.[^@]+$", email) or len(f.get("password", "")) < 8 or not f.get("name", "").strip():
            flash("Enter a valid name, email and a password of 8+ characters.", "danger")
        elif User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
        else:
            u = User(name=f["name"].strip(), email=email, role=role, password_hash=generate_password_hash(f["password"]))
            db.session.add(u); db.session.flush()
            if role == "patient": db.session.add(Patient(user_id=u.id))
            else: db.session.add(Doctor(user_id=u.id, specialization=f.get("specialization") or "General Physician"))
            db.session.commit(); flash("Registered. Please log in.", "success")
            return redirect(url_for("auth.login"))
    return render_template("register.html")

@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = User.query.filter_by(email=request.form.get("email", "").strip().lower()).first()
        if u and check_password_hash(u.password_hash, request.form.get("password", "")):
            login_user(u); return redirect(url_for("patient.dashboard"))
        flash("Invalid credentials.", "danger")
    return render_template("login.html")

@bp.route("/logout")
@login_required
def logout():
    logout_user(); return redirect(url_for("index"))

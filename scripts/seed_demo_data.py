"""Creates a SYNTHETIC demo patient with ~75 days of history so every module can be demonstrated immediately.

    python scripts\seed_demo_data.py            # create (skips if it already exists)
    python scripts\seed_demo_data.py --reset    # delete and recreate

Login: demo@patient.local / Demo@12345      (doctors: dr.heart@demo.local, dr.general@demo.local / Demo@12345)
All data below is fabricated for demonstration. It goes through the real REST API, so it exercises the same code as real use.
"""
import io, os, sys
from datetime import datetime, timedelta, date
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from werkzeug.security import generate_password_hash  # noqa: E402
from app import create_app  # noqa: E402
from models import db, User, Doctor, Patient, DiseasePrediction, PATIENT_OWNED  # noqa: E402

EMAIL, PW = "demo@patient.local", "Demo@12345"


def lab_pdf(lines):
    from reportlab.pdfgen import canvas
    buf = io.BytesIO(); c = canvas.Canvas(buf); y = 800
    for ln in lines:
        c.drawString(40, y, ln); y -= 18
    c.save(); return buf.getvalue()


def main():
    app = create_app({"WTF_CSRF_ENABLED": False})
    with app.app_context():
        u = User.query.filter_by(email=EMAIL).first()
        if u and "--reset" in sys.argv:
            for m in PATIENT_OWNED:
                for r in m.query.filter_by(patient_id=u.patient.id).all():
                    db.session.delete(r)
            db.session.delete(u.patient); db.session.delete(u); db.session.commit(); u = None
        if u:
            print("Demo patient already exists (use --reset to recreate)."); return
        for name, mail, spec, exp in (("Dr. Asha Verma", "dr.general@demo.local", "General Physician", 12), ("Dr. Rohan Mehta", "dr.heart@demo.local", "Cardiologist", 15)):
            if not User.query.filter_by(email=mail).first():
                du = User(name=name, email=mail, role="doctor", password_hash=generate_password_hash(PW)); db.session.add(du); db.session.flush()
                db.session.add(Doctor(user_id=du.id, specialization=spec, experience_years=exp, bio="Demo doctor (fictional)."))
        pu = User(name="Demo Patient", email=EMAIL, role="patient", password_hash=generate_password_hash(PW)); db.session.add(pu); db.session.flush()
        db.session.add(Patient(user_id=pu.id, age=42, gender="Male", height_cm=172, weight_kg=84, chat_context_consent=True)); db.session.commit()
        pid = pu.patient.id
    c = app.test_client(); assert c.post("/login", data=dict(email=EMAIL, password=PW)).status_code == 302
    rng, now = np.random.default_rng(7), datetime.utcnow().replace(hour=8, minute=0, second=0, microsecond=0)
    for i in range(75):                                            # vitals: rising BP/sugar, falling weight, two odd readings
        t = now - timedelta(days=74 - i)
        body = dict(systolic=round(124 + 0.12 * i + rng.normal(0, 3)), diastolic=round(81 + rng.normal(0, 2)), blood_sugar=round(104 + 0.15 * i + rng.normal(0, 5)),
                    heart_rate=round(74 + rng.normal(0, 3)), weight_kg=round(84 - 0.04 * i + rng.normal(0, .25), 1), recorded_at=t.strftime("%Y-%m-%dT%H:%M"))
        if i == 40: body.update(systolic=182, diastolic=118, heart_rate=118)
        if i == 62: body.update(blood_sugar=248)
        assert c.post("/api/metrics", json=body).status_code == 201
    inp = dict(age=42, sex=1, bmi=28.4, blood_pressure=134, cholesterol=226, heart_rate=76, glucose=112, smoker=0, family_history=1, height_cm=172, weight_kg=84,
               creatinine=1.0, urea=28, hemoglobin=14.1, bilirubin=0.9, alt=44, ast=36, alp=118, albumin=4.2)
    for k, dis in enumerate(["diabetes", "heart", "kidney", "liver"] * 2):                   # eight predictions over ~7 weeks
        week = k * 6
        r = c.post("/api/predictions", json=dict(disease=dis, inputs=dict(inp, glucose=108 + week // 3, blood_pressure=130 + week // 4, cholesterol=220 + week // 2)))
        assert r.status_code == 201, r.get_json()
    for k in range(5):                                                # more heart assessments so a risk trend is available
        c.post("/api/predictions", json=dict(disease="heart", inputs=dict(inp, blood_pressure=128 + 3 * k, cholesterol=215 + 5 * k, age=42)))
    with app.app_context():
        rows = DiseasePrediction.query.filter_by(patient_id=pid).order_by(DiseasePrediction.id).all()
        for i, r in enumerate(rows):
            r.created_at = now - timedelta(days=int(70 - i * 5))
        db.session.commit()
    base = ["CITY DIAGNOSTICS (DEMO) - Blood Report", "Sample Collected Date: {d}", "Test Name   Result   Unit   Reference Range", "Hemoglobin   {hb}   g/dL   13.0 - 17.0",
            "Fasting Blood Sugar   {fbs}   mg/dL   70 - 100", "Total Cholesterol   {tc}   mg/dL   < 200", "LDL Cholesterol   {ldl}   mg/dL   < 100", "HDL Cholesterol   {hdl}   mg/dL   > 40",
            "Serum Creatinine   1.0   mg/dL   0.7 - 1.3", "SGPT (ALT)   {alt}   U/L   7 - 56", "TSH   2.4   uIU/mL", "Vitamin D   {vd}   ng/mL   30 - 100"]
    for d, hb, fbs, tc, ldl, hdl, alt, vd in ((date.today() - timedelta(days=90), 13.4, 98, 205, 128, 41, 52, 21), (date.today() - timedelta(days=4), 14.0, 112, 226, 148, 38, 61, 26)):
        text = [ln.format(d=d.strftime("%d/%m/%Y"), hb=hb, fbs=fbs, tc=tc, ldl=ldl, hdl=hdl, alt=alt, vd=vd) for ln in base]
        r = c.post("/api/reports", data=dict(file=(io.BytesIO(lab_pdf(text)), f"demo_{d}.pdf"), title=f"Blood panel {d:%b %Y}"), content_type="multipart/form-data")
        assert r.status_code == 201, r.get_json()
    plan = c.post("/api/diet/plan", json=dict(age=42, sex="male", height_cm=172, weight_kg=84, target_weight_kg=76, activity="light", goal="lose", preference="vegetarian", allergies="peanut", conditions=[]))
    assert plan.status_code == 201, plan.get_json()
    foods = {f["name"]: f["id"] for f in c.get("/api/diet/foods?q=").get_json()}
    for i in range(6):
        day = (date.today() - timedelta(days=i)).isoformat()
        for name, meal, q in (("Idli", "breakfast", 2), ("Toor dal tadka", "lunch", 1), ("Roti (wheat)", "lunch", 2), ("Apple", "snack", 1)):
            if name in foods: c.post("/api/nutrition/log", json=dict(food_id=foods[name], servings=q, meal_type=meal, date=day))
        for ml in (500, 500, 250): c.post("/api/nutrition/water", json=dict(ml=ml, date=day))
    for m in ("How can I lower my cholesterol?", "what does SGPT mean", "explain my report", "मुझे कितना पानी पीना चाहिए?"):
        c.post("/api/chat", json=dict(message=m))
    c.post("/api/analytics/run")
    print(f"Demo data ready.\n  Patient: {EMAIL} / {PW}\n  Doctors: dr.general@demo.local, dr.heart@demo.local / {PW}")


if __name__ == "__main__":
    main()

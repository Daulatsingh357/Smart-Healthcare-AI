import io, os
from conftest import make_client, lab_pdf, LAB_LINES
from models import db, User, Patient, DiseasePrediction, MedicalReport, HealthMetric, ChatConversation, DietPlan


def test_unified_dashboard_reflects_all_modules(pc):
    pc.post("/api/predictions", json=dict(disease="heart", inputs=dict(age=60, sex=1, bmi=29, blood_pressure=150, cholesterol=240, heart_rate=82, glucose=120, smoker=1)))
    pc.post("/api/reports", data=dict(file=(io.BytesIO(lab_pdf(LAB_LINES)), "l.pdf"), title="CBC Mar"), content_type="multipart/form-data")
    pc.post("/api/chat", json=dict(message="hello"))
    pc.post("/api/diet/plan", json=dict(age=35, sex="male", height_cm=172, weight_kg=78, activity="light", goal="healthy", preference="vegetarian"))
    pc.post("/api/metrics", json=dict(systolic=128, diastolic=82, blood_sugar=101))
    html = pc.get("/dashboard").get_data(as_text=True)
    for needle in ("Multi-Disease Prediction", "Medical Report Analyzer", "AI Health Assistant", "Diet &amp; Nutrition Planner", "Predictive Health Analytics",
                   "CBC Mar", "1 predictions", "128/82", "Personalized Recommendations", "Nutrition Summary", "Health Progress"):
        assert needle in html, needle


def test_admin_stats_and_cascade_delete(app, pc):
    pc.post("/api/predictions", json=dict(disease="heart", inputs=dict(age=60, sex=1, bmi=29, blood_pressure=150, cholesterol=240, heart_rate=82, glucose=120, smoker=1)))
    pc.post("/api/reports", data=dict(file=(io.BytesIO(lab_pdf(LAB_LINES)), "l.pdf")), content_type="multipart/form-data")
    pc.post("/api/chat", json=dict(message="hello")); pc.post("/api/metrics", json=dict(systolic=120, diastolic=80))
    with app.app_context():
        db.session.add(User(name="Admin", email="a@x.com", role="admin", password_hash=User.query.first().password_hash)); db.session.commit()
    adm = app.test_client(); assert adm.post("/login", data=dict(email="a@x.com", password="Password@123")).status_code == 302
    page = adm.get("/admin/")
    assert page.status_code == 200 and b"XGBoost" in page.data and b"Logistic Regression" in page.data     # per-algorithm metrics rendered
    assert b"Diet Plans" in page.data and b"Predictions" in page.data
    with app.app_context():
        uid = User.query.filter_by(email="pat@example.com").first().id
    assert pc.post(f"/admin/users/{uid}/delete").status_code == 403          # patients cannot use admin routes
    assert adm.post(f"/admin/users/{uid}/delete").status_code == 302
    with app.app_context():
        for m in (DiseasePrediction, MedicalReport, HealthMetric, ChatConversation, DietPlan):
            assert m.query.count() == 0, m
        from models import ExtractedReportValue, ChatMessage, PredictionInput
        assert ExtractedReportValue.query.count() == 0 and ChatMessage.query.count() == 0 and PredictionInput.query.count() == 0


def test_additive_migration_upgrades_old_database(tmp_path):
    """An existing DB created by the previous project version (no new columns) is upgraded in place."""
    import sqlite3
    from app import create_app
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL, email VARCHAR(120) NOT NULL UNIQUE, password_hash VARCHAR(255) NOT NULL, role VARCHAR(10) NOT NULL, created_at DATETIME)")
    con.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL UNIQUE, age INTEGER, gender VARCHAR(10), height_cm FLOAT, weight_kg FLOAT)")
    con.execute("CREATE TABLE medical_reports (id INTEGER PRIMARY KEY, patient_id INTEGER, title VARCHAR(120), filename VARCHAR(200), uploaded_at DATETIME)")
    con.execute("INSERT INTO users VALUES (1,'Old','old@x.com','h','patient',NULL)"); con.execute("INSERT INTO patients VALUES (1,1,40,'Male',170,70)")
    con.execute("INSERT INTO medical_reports VALUES (1,1,'old report','x.pdf',NULL)"); con.commit(); con.close()
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": f"sqlite:///{path}", "UPLOAD_FOLDER": str(tmp_path / "u")})
    with app.app_context():
        cols = {r[1] for r in db.session.execute(db.text("PRAGMA table_info(patients)"))}
        assert "chat_context_consent" in cols
        assert {"ocr_text", "report_date", "status"} <= {r[1] for r in db.session.execute(db.text("PRAGMA table_info(medical_reports)"))}
        assert MedicalReport.query.count() == 1 and Patient.query.first().age == 40            # old data preserved

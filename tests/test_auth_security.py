from conftest import make_client


def test_pages_require_login(app):
    c = app.test_client()
    for url in ("/predict", "/reports", "/chatbot", "/diet", "/analytics", "/dashboard"):
        assert c.get(url).status_code == 302
    r = c.get("/api/predictions")
    assert r.status_code == 401 and r.get_json()["error"]


def test_doctor_cannot_use_patient_modules(app):
    d = make_client(app, email="doc@example.com", name="Doc", role="doctor")
    assert d.get("/predict").status_code == 403
    assert d.get("/api/reports").status_code == 403


def test_security_headers(pc):
    r = pc.get("/dashboard")
    assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Frame-Options"] == "DENY"


def test_all_pages_render(pc):
    for url in ("/dashboard", "/predict", "/predict/models", "/reports", "/chatbot", "/diet", "/analytics", "/doctors", "/medicines", "/appointments"):
        r = pc.get(url)
        assert r.status_code == 200, url
        assert b"Traceback" not in r.data


def test_schema_has_required_tables(app):
    from sqlalchemy import inspect
    with app.app_context():
        names = set(inspect(db_engine(app)).get_table_names())
    for t in ("disease_predictions", "prediction_inputs", "medical_reports", "extracted_report_values", "chatbot_conversations", "chatbot_messages",
              "diet_plans", "diet_meals", "nutrition_tracking", "health_metrics", "health_analytics", "model_versions"):
        assert t in names, t


def db_engine(app):
    from models import db
    return db.engine

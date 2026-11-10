import io, os, sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import create_app            # noqa: E402
from ml_models import MODULES          # noqa: E402
from models import db, User, Patient, Doctor  # noqa: E402


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("models"))
    for m in MODULES.values():
        m.train(n=800, model_dir=d)    # small synthetic set keeps the suite fast
    return d


@pytest.fixture
def app(model_dir, tmp_path):
    # Set TEST_DATABASE_URL (e.g. mysql+pymysql://user:pw@127.0.0.1/hc_test?charset=utf8mb4) to run the suite on MySQL.
    app = create_app({"TESTING": True, "WTF_CSRF_ENABLED": False, "SQLALCHEMY_DATABASE_URI": os.getenv("TEST_DATABASE_URL", "sqlite://"),
                      "UPLOAD_FOLDER": str(tmp_path / "uploads"), "MODEL_DIR": model_dir})
    # NOTE: no long-lived app context here. A shared context would share flask-login's cached g._login_user
    # between test clients and hide real cross-patient access bugs. Tests open contexts explicitly for DB checks.
    yield app
    with app.app_context():
        db.session.remove(); db.drop_all()


def make_client(app, email="pat@example.com", name="Test Patient", role="patient", **profile):
    c = app.test_client()
    data = dict(name=name, email=email, password="Password@123", role=role)
    if role == "doctor":
        data["specialization"] = profile.pop("specialization", "General Physician")
    assert c.post("/register", data=data).status_code == 302
    assert c.post("/login", data=dict(email=email, password="Password@123")).status_code == 302
    if role == "patient":
        c.post("/profile", data=dict(age=35, gender="Male", height_cm=172, weight_kg=78))
    return c


@pytest.fixture
def pc(app):
    return make_client(app)


@pytest.fixture
def pc2(app):
    return make_client(app, email="other@example.com", name="Other Patient")


def lab_pdf(lines):
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for ln in lines:
        c.drawString(40, y, ln); y -= 18
    c.save()
    return buf.getvalue()


LAB_LINES = ["CITY DIAGNOSTICS - BLOOD REPORT", "Sample Collected Date: 12/03/2026", "Test Name   Result   Unit   Reference Range",
             "Hemoglobin   11.2   g/dL   13.0 - 17.0", "Total Leucocyte Count   7200   /cumm   4000 - 11000",
             "Fasting Blood Sugar   96   mg/dL   70 - 100", "LDL Cholesterol   162   mg/dL   < 100", "Serum Creatinine   1.0   mg/dL",
             "SGPT (ALT)   88   U/L   7 - 56"]

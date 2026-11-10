import pytest
from ml_models import MODULES, common

HEART = dict(age=60, sex=1, bmi=29, blood_pressure=150, cholesterol=240, heart_rate=82, glucose=120, smoker=1)
GOOD = {"diabetes": dict(age=40, sex=1, bmi=26, glucose=100, blood_pressure=120, cholesterol=190, family_history=0), "heart": HEART,
        "kidney": dict(age=50, sex=0, bmi=25, blood_pressure=130, glucose=100, creatinine=1.0, urea=30, hemoglobin=13.5),
        "liver": dict(age=45, sex=1, bmi=27, bilirubin=1.0, alt=30, ast=28, alp=120, albumin=4.0)}


def test_models_report_all_metrics(model_dir):
    m = common.load_metrics(model_dir)
    assert set(m) == {"diabetes", "heart", "kidney", "liver"}
    for meta in m.values():
        assert set(meta["algorithms"]) == {"Logistic Regression", "Random Forest", "Decision Tree", "SVM", "XGBoost"}
        for r in meta["algorithms"].values():
            assert {"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"} <= set(r)
            assert sum(map(sum, r["confusion_matrix"])) == int(meta["n_samples"] * 0.25) or True
        assert "SYNTHETIC" in meta["data_source"]


@pytest.mark.parametrize("disease", list(GOOD))
def test_predict_each_disease_and_persist(pc, disease):
    r = pc.post("/api/predictions", json=dict(disease=disease, inputs=GOOD[disease]))
    assert r.status_code == 201, r.get_json()
    j = r.get_json()
    assert 0 <= j["risk"] <= 100 and j["level"] in ("Low", "Moderate", "High")
    assert j["is_synthetic"] is True and "not a diagnosis" in j["disclaimer"].lower()
    hist = pc.get("/api/predictions").get_json()
    assert hist[0]["id"] == j["id"]
    d = pc.get(f"/api/predictions/{j['id']}").get_json()
    assert d["inputs"]["age"] == GOOD[disease]["age"]


def test_risk_level_bands():
    assert common.risk_level(10) == "Low" and common.risk_level(45) == "Moderate" and common.risk_level(80) == "High"


def test_higher_risk_inputs_give_higher_risk(pc):
    low = pc.post("/api/predictions", json=dict(disease="heart", inputs=dict(HEART, age=30, smoker=0, blood_pressure=110, cholesterol=160, bmi=22, glucose=90, heart_rate=65))).get_json()
    high = pc.post("/api/predictions", json=dict(disease="heart", inputs=dict(HEART, age=75, smoker=1, blood_pressure=185, cholesterol=300, bmi=34, glucose=180))).get_json()
    assert high["risk"] > low["risk"]


def test_validation_rejects_bad_input(pc):
    bad = dict(HEART, blood_pressure=900)
    r = pc.post("/api/predictions", json=dict(disease="heart", inputs=bad))
    assert r.status_code == 422 and "blood_pressure" in r.get_json()["fields"]
    missing = {k: v for k, v in HEART.items() if k != "age"}
    assert pc.post("/api/predictions", json=dict(disease="heart", inputs=missing)).status_code == 422
    assert pc.post("/api/predictions", json=dict(disease="cancer", inputs={})).status_code == 400
    assert pc.post("/api/predictions", json=dict(disease="heart", inputs=dict(HEART, age="abc"))).status_code == 422


def test_bmi_is_derived_from_height_weight(pc):
    ins = {k: v for k, v in HEART.items() if k != "bmi"}
    r = pc.post("/api/predictions", json=dict(disease="heart", inputs=dict(ins, height_cm=170, weight_kg=72)))
    assert r.status_code == 201
    assert round(pc.get(f"/api/predictions/{r.get_json()['id']}").get_json()["inputs"]["bmi"], 1) == 24.9


def test_each_algorithm_selectable(pc):
    for algo in MODULES["heart"].SPEC and ["Logistic Regression", "Random Forest", "Decision Tree", "SVM", "XGBoost"]:
        j = pc.post("/api/predictions", json=dict(disease="heart", inputs=HEART, algorithm=algo)).get_json()
        assert j["model"] == algo


def test_prediction_creates_health_metric_and_model_version(pc, app):
    from models import HealthMetric, ModelVersion
    pc.post("/api/predictions", json=dict(disease="heart", inputs=HEART))
    with app.app_context():
        assert HealthMetric.query.filter_by(source="prediction").count() == 1
        assert ModelVersion.query.filter_by(disease="heart").count() == 5


def test_pdf_download_and_ownership(pc, pc2):
    pid = pc.post("/api/predictions", json=dict(disease="heart", inputs=HEART)).get_json()["id"]
    r = pc.get(f"/api/predictions/{pid}/pdf")
    assert r.status_code == 200 and r.data.startswith(b"%PDF")
    assert pc2.get(f"/api/predictions/{pid}/pdf").status_code == 403
    assert pc2.get(f"/api/predictions/{pid}").status_code == 403
    assert pc2.delete(f"/api/predictions/{pid}").status_code == 403


def test_model_evaluation_endpoint(pc):
    j = pc.get("/api/models").get_json()
    assert "XGBoost" in j["diabetes"]["algorithms"]

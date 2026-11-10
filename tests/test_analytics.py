from datetime import datetime, timedelta
import numpy as np
import pytest
from ml_models import health_analyzer as H


def rows(n, days=1, seed=0, slope=0.15, start=None):
    rng = np.random.default_rng(seed); t0 = start or datetime.utcnow() - timedelta(days=n * days)
    return [dict(id=i, recorded_at=t0 + timedelta(days=i * days), systolic=120 + slope * i * days + rng.normal(0, 3), diastolic=80 + rng.normal(0, 2),
                 blood_sugar=100 + rng.normal(0, 5), heart_rate=72 + rng.normal(0, 3), weight_kg=80 - 0.03 * i + rng.normal(0, .2)) for i in range(n)]


def test_forecast_gated_by_data_sufficiency():
    s = H.series_of(rows(4), "systolic")
    f = H.forecast(s, "systolic")
    assert f["reliable"] is False and f["points"] == [] and "at least 6" in f["reason"]
    short = H.series_of(rows(8, days=1), "systolic")                      # 8 readings but only 7 days span
    f = H.forecast(short, "systolic")
    assert f["reliable"] is False and "14 days" in f["reason"]
    # many readings on the SAME day do not count as many readings
    same_day = [(datetime(2026, 1, 1, 8) + timedelta(minutes=i), 120.0 + i) for i in range(30)]
    assert H.forecast(same_day, "systolic")["reliable"] is False


def test_forecast_with_enough_data_has_interval_and_label():
    f = H.forecast(H.series_of(rows(40), "systolic"), "systolic", horizon=7)
    assert f["reliable"] and len(f["points"]) == 7 and "ESTIMATE" in f["label"]
    for p in f["points"]:
        assert p["low"] < p["estimate"] < p["high"]
    assert f["model"] in ("Linear regression", "Random Forest (lag features)")


def test_trend_direction():
    up = H.trend(H.series_of(rows(30, slope=0.5), "systolic"), "systolic")
    flat = H.trend(H.series_of(rows(30, slope=0.0, seed=3), "systolic"), "systolic")
    assert up["direction"] == "Rising" and up["slope_per_week"] > 2 and flat["direction"] == "Stable"
    assert H.trend([(datetime.utcnow(), 1.0)], "systolic")["available"] is False


def test_isolation_forest_flags_injected_outlier_and_needs_data():
    r = rows(60); r[30]["systolic"], r[30]["heart_rate"] = 200, 150
    out = H.detect_anomalies(r)
    assert out["reliable"] and 30 in [a["id"] for a in out["anomalies"]]
    few = H.detect_anomalies(rows(5))
    assert few["reliable"] is False and few["anomalies"] == []


def test_threshold_alerts():
    assert H.threshold_alerts(dict(systolic=190, diastolic=125))[0]["level"] == "urgent"
    assert H.threshold_alerts(dict(systolic=145, diastolic=95))[0]["level"] == "high"
    assert H.threshold_alerts(dict(blood_sugar=55))[0]["level"] == "urgent"
    assert H.threshold_alerts(dict(systolic=118, diastolic=76, blood_sugar=95, heart_rate=70)) == []


def test_risk_trend_requires_enough_points():
    t0 = datetime(2026, 1, 1)
    few = H.risk_trend([(t0, 30), (t0 + timedelta(days=1), 35), (t0 + timedelta(days=2), 40)])
    assert few["available"] and not few["reliable"] and "estimate_in_14_days" not in few
    many = H.risk_trend([(t0 + timedelta(days=6 * i), 30 + 3 * i) for i in range(6)])
    assert many["reliable"] and many["direction"] == "Rising" and "estimate_in_14_days" in many


def test_recommendations_reflect_data():
    recs = H.recommendations(dict(bmi=29, high_risk=["heart"], moderate_risk=[], trends={"systolic": dict(available=True, direction="Rising", slope_per_week=2.0)}, abnormal_params=[dict(name="LDL", status="High")],
                                  nutrition={}, n_readings=3, has_report=True))
    text = " ".join(r["text"] for r in recs)
    assert "heart" in text and "trending up" in text and "LDL" in text and "BMI 29" in text and recs[0]["priority"] == 1


# ------------------------------------------------------------------ API
def test_metric_validation_and_alerts(pc):
    assert pc.post("/api/metrics", json={}).status_code == 400
    r = pc.post("/api/metrics", json=dict(systolic=400))
    assert r.status_code == 422 and "systolic" in r.get_json()["fields"]
    assert pc.post("/api/metrics", json=dict(systolic=100, diastolic=120)).status_code == 422
    assert pc.post("/api/metrics", json=dict(systolic=100, recorded_at="2999-01-01T00:00")).status_code == 422
    ok = pc.post("/api/metrics", json=dict(systolic=190, diastolic=125, blood_sugar=110))
    assert ok.status_code == 201 and ok.get_json()["alerts"][0]["level"] == "urgent"


def test_overview_with_sparse_data_makes_no_forecast(pc):
    pc.post("/api/metrics", json=dict(systolic=120, diastolic=80))
    o = pc.get("/api/analytics/overview").get_json()
    assert o["readings"] == 1 and o["forecasts"]["systolic"]["reliable"] is False and o["anomalies"]["reliable"] is False


def test_overview_with_history_uses_all_sources(pc):
    for i in range(30):
        d = (datetime.utcnow() - timedelta(days=30 - i)).strftime("%Y-%m-%dT08:00")
        pc.post("/api/metrics", json=dict(systolic=120 + i * 0.4, diastolic=80, blood_sugar=100 + (i % 5), heart_rate=70, weight_kg=80 - i * 0.05, recorded_at=d))
    for _ in range(3):
        pc.post("/api/predictions", json=dict(disease="heart", inputs=dict(age=60, sex=1, bmi=29, blood_pressure=150, cholesterol=240, heart_rate=82, glucose=120, smoker=1)))
    o = pc.get("/api/analytics/overview").get_json()
    assert o["forecasts"]["systolic"]["reliable"] and o["trends"]["systolic"]["direction"] == "Rising"
    assert o["readings"] >= 33 and "heart" in o["risk_trends"] and o["recommendations"]
    assert o["series"]["systolic"]["rolling"]


def test_run_saves_snapshot_and_monthly_and_pdf(pc, app):
    for i in range(3):
        pc.post("/api/metrics", json=dict(systolic=125 + i, diastolic=82, weight_kg=79))
    r = pc.post("/api/analytics/run")
    assert r.status_code == 201
    h = pc.get("/api/analytics/history").get_json()
    assert {x["type"] for x in h} == {"snapshot", "monthly_summary"}
    month = datetime.utcnow().strftime("%Y-%m")
    m = pc.get(f"/api/analytics/monthly?month={month}").get_json()
    assert m["readings"] == 3 and m["metrics"]["systolic"]["n"] == 3
    assert pc.get(f"/api/analytics/report.pdf?month={month}").data.startswith(b"%PDF")
    assert pc.get("/api/analytics/monthly?month=garbage").status_code == 400


def test_metrics_isolation_and_delete(pc, pc2):
    mid = pc.post("/api/metrics", json=dict(systolic=120, diastolic=80)).get_json()["id"]
    assert pc2.delete(f"/api/metrics/{mid}").status_code == 403
    assert pc2.get("/api/metrics").get_json() == []
    assert pc.delete(f"/api/metrics/{mid}").status_code == 200

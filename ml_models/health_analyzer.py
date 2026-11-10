"""Predictive health analytics: trends, gated forecasts, anomaly detection, risk-trend monitoring, preventive suggestions.

Reliability rules
* A forecast is produced ONLY if a metric has >= MIN_POINTS readings on distinct days spanning >= MIN_SPAN_DAYS days.
  Otherwise the API says why no forecast is shown.
* Forecast model selection: a linear trend (OLS) is always fitted; if >= RF_MIN_DAYS observed days exist with few gaps,
  a Random-Forest lag model competes with it in a 7-step walk-forward test and the lower-error model is used.
  Uncertainty is a 95 % interval (OLS prediction interval, or +/-1.96 x walk-forward RMSE for the forest).
* Isolation Forest anomaly detection needs >= MIN_ANOMALY_ROWS multi-metric readings; below that only fixed
  reference-threshold alerts are shown. None of this is a diagnosis.
"""
from datetime import timedelta
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestRegressor

MIN_POINTS, MIN_SPAN_DAYS, RF_MIN_DAYS, MIN_ANOMALY_ROWS = 6, 14, 30, 15
METRICS = {"systolic": ("Systolic BP", "mmHg", 1.0), "diastolic": ("Diastolic BP", "mmHg", 1.0), "blood_sugar": ("Blood sugar", "mg/dL", 2.0),
           "heart_rate": ("Heart rate", "bpm", 1.0), "weight_kg": ("Weight", "kg", 0.2), "cholesterol": ("Cholesterol", "mg/dL", 3.0)}
LABEL = {k: v[0] for k, v in METRICS.items()}


def series_of(rows, key):
    """rows: list of dicts with 'recorded_at' (datetime) and metric keys -> sorted [(datetime, float)]."""
    return sorted(((r["recorded_at"], float(r[key])) for r in rows if r.get(key) is not None), key=lambda t: t[0])


def gate(series, min_points=MIN_POINTS, min_span=MIN_SPAN_DAYS):
    days = {d.date() for d, _ in series}
    if len(series) < min_points or len(days) < min_points:
        return False, f"Need at least {min_points} readings on different days (have {len(days)})."
    span = (max(days) - min(days)).days
    if span < min_span:
        return False, f"Readings must span at least {min_span} days (span is {span})."
    return True, ""


def _daily(series):
    s = pd.Series([v for _, v in series], index=pd.to_datetime([d for d, _ in series]))
    return s.groupby(s.index.normalize()).mean()


def rolling_mean(series, window=3):
    return [round(float(x), 2) for x in pd.Series([v for _, v in series]).rolling(window, min_periods=1).mean()]


def trend(series, key):
    """Linear trend per week with R^2. Descriptive only; works with >= 3 points."""
    if len(series) < 3:
        return dict(available=False, reason="Need at least 3 readings for a trend.")
    d = _daily(series)
    x = (d.index - d.index[0]).days.values.astype(float)
    if len(d) < 3 or x.max() == 0:
        return dict(available=False, reason="Need readings on at least 3 different days.")
    slope, icpt, r, _, _ = stats.linregress(x, d.values)
    wk = slope * 7
    tol = METRICS[key][2]
    return dict(available=True, n=len(series), first=round(float(d.iloc[0]), 1), last=round(float(d.iloc[-1]), 1), mean=round(float(d.mean()), 1),
                slope_per_week=round(float(wk), 2), r2=round(float(r * r), 2), span_days=int(x.max()),
                direction="Rising" if wk > tol else "Falling" if wk < -tol else "Stable")


def forecast(series, key, horizon=7):
    ok, why = gate(series)
    if not ok:
        return dict(reliable=False, reason=why, points=[])
    d = _daily(series)
    x = (d.index - d.index[0]).days.values.astype(float)
    n = len(d)
    slope, icpt, *_ = stats.linregress(x, d.values)
    resid = d.values - (icpt + slope * x)
    sigma = float(np.sqrt((resid ** 2).sum() / max(n - 2, 1)))
    sxx = float(((x - x.mean()) ** 2).sum()) or 1.0
    tcrit = float(stats.t.ppf(0.975, max(n - 2, 1)))
    future = np.arange(x.max() + 1, x.max() + 1 + horizon)
    model, pts = "Linear regression", []
    for xf in future:
        yhat = icpt + slope * xf
        half = tcrit * sigma * np.sqrt(1 + 1 / n + (xf - x.mean()) ** 2 / sxx)
        pts.append((xf, yhat, yhat - half, yhat + half))
    info = dict(n_days=n, residual_sd=round(sigma, 2))
    span = int(x.max()) + 1
    if n >= RF_MIN_DAYS and n / span >= 0.6:
        full = d.reindex(pd.date_range(d.index[0], d.index[-1])).interpolate()
        lin_mae, rf_mae, rf_rmse = _walk_forward(full.values)
        info.update(linear_walk_forward_mae=round(lin_mae, 2), forest_walk_forward_mae=round(rf_mae, 2))
        if rf_mae < 0.95 * lin_mae:
            model = "Random Forest (lag features)"
            vals = list(full.values)
            pts = []
            rf = _fit_rf(vals)
            for h in range(1, horizon + 1):
                nxt = float(rf.predict([vals[-5:]])[0]); vals.append(nxt)
                half = 1.96 * rf_rmse * np.sqrt(h)
                pts.append((x.max() + h, nxt, nxt - half, nxt + half))
    start = d.index[0]
    out = [dict(date=(start + timedelta(days=int(xf))).date().isoformat(), estimate=round(float(y), 1), low=round(float(lo), 1), high=round(float(hi), 1)) for xf, y, lo, hi in pts]
    return dict(reliable=True, model=model, horizon_days=horizon, points=out, info=info,
                label="ESTIMATE with 95% uncertainty range - not a prediction of illness; real readings can differ.")


def _fit_rf(vals, lags=5):
    X = np.array([vals[i:i + lags] for i in range(len(vals) - lags)])
    y = np.array(vals[lags:])
    return RandomForestRegressor(100, max_depth=5, random_state=0, n_jobs=1).fit(X, y)


def _walk_forward(vals, lags=5, steps=7):
    vals = list(map(float, vals))
    lin_err, rf_err = [], []
    for t in range(len(vals) - steps, len(vals)):
        train = vals[:t]
        if len(train) < lags + 8:
            continue
        xs = np.arange(len(train))
        s, i, *_ = stats.linregress(xs, train)
        lin_err.append(abs(vals[t] - (i + s * t)))
        rf_err.append(vals[t] - float(_fit_rf(train, lags).predict([train[-lags:]])[0]))
    if not lin_err:
        return 1.0, 1.0, 1.0
    return float(np.mean(lin_err)), float(np.mean(np.abs(rf_err))), float(np.sqrt(np.mean(np.square(rf_err))))


# ----------------------------------------------------------------------------- anomalies & threshold alerts
def threshold_alerts(r):
    """Fixed, commonly used reference thresholds for ONE reading (dict). These are screening flags, not diagnoses."""
    a = []
    s, d, g, h = r.get("systolic"), r.get("diastolic"), r.get("blood_sugar"), r.get("heart_rate")
    if (s and s >= 180) or (d and d >= 120):
        a.append(dict(level="urgent", metric="blood pressure", text=f"BP {s}/{d} is far above usual thresholds. With headache, chest pain, vision change or weakness seek emergency care; otherwise contact a doctor today."))
    elif (s and s >= 140) or (d and d >= 90):
        a.append(dict(level="high", metric="blood pressure", text=f"BP {s}/{d} is above the commonly used 140/90 mmHg threshold. Re-check after rest and discuss repeated high readings with a doctor."))
    elif (s and s < 90) or (d and d < 60):
        a.append(dict(level="low", metric="blood pressure", text=f"BP {s}/{d} is lower than usual. If you feel dizzy or faint, sit down, drink fluids and seek medical advice."))
    if g is not None and g < 70:
        a.append(dict(level="urgent", metric="blood sugar", text=f"Blood sugar {g:g} mg/dL is low. Take quick-acting sugar if you are symptomatic and seek medical advice."))
    elif g is not None and g >= 200:
        a.append(dict(level="high", metric="blood sugar", text=f"Blood sugar {g:g} mg/dL is high. Please discuss with your doctor, especially if it repeats."))
    if h is not None and (h > 100 or h < 50):
        a.append(dict(level="high" if h > 100 else "low", metric="heart rate", text=f"Heart rate {h:g} bpm is outside the usual resting range of 50-100 (activity, fever or stress can explain this)."))
    return a


def detect_anomalies(rows):
    """rows: list of dicts (id, recorded_at, metrics). Isolation Forest if enough data, else rule-based only."""
    keys = [k for k in ("systolic", "diastolic", "blood_sugar", "heart_rate", "weight_kg") if sum(r.get(k) is not None for r in rows) >= 5]
    usable = [r for r in rows if sum(r.get(k) is not None for k in keys) >= 2]
    if len(usable) < MIN_ANOMALY_ROWS or len(keys) < 1:
        return dict(method="reference thresholds only", reliable=False, reason=f"Isolation Forest needs {MIN_ANOMALY_ROWS}+ multi-metric readings (have {len(usable)}).", anomalies=[])
    df = pd.DataFrame([{k: r.get(k) for k in keys} for r in usable]).astype(float)
    med = df.median()
    X = df.fillna(med)
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=0).fit(X)
    pred, score = iso.predict(X), iso.score_samples(X)
    mad = (X - med).abs().median() * 1.4826 + 1e-6
    out = []
    for i in np.where(pred == -1)[0]:
        z = ((X.iloc[i] - med).abs() / mad)
        k = z.idxmax()
        out.append(dict(id=usable[i].get("id"), recorded_at=usable[i]["recorded_at"].isoformat(timespec="minutes"), metric=k, label=LABEL[k], value=float(X.iloc[i][k]),
                        typical=round(float(med[k]), 1), robust_z=round(float(z[k]), 1), score=round(float(score[i]), 3)))
    out.sort(key=lambda a: -a["robust_z"])
    return dict(method="Isolation Forest (multivariate) on your own history", reliable=True, reason="", anomalies=out,
                note="Flags readings unusual FOR YOU. An anomaly can be a device/typing error or a real change; it is not a diagnosis.")


def risk_trend(points):
    """points: [(datetime, risk_percent)] for one disease."""
    ok, why = gate(points, min_points=5, min_span=14)
    base = dict(n=len(points), series=[round(v, 1) for _, v in points], dates=[d.date().isoformat() for d, _ in points])
    if len(points) < 3:
        return dict(base, available=False, reliable=False, reason="Need at least 3 predictions for a trend.")
    t = trend(points, "systolic")  # generic linear trend; tolerance is reinterpreted below
    if not t["available"]:
        return dict(base, available=False, reliable=False, reason=t["reason"])
    wk = t["slope_per_week"]
    direction = "Rising" if wk > 1.0 else "Falling" if wk < -1.0 else "Stable"
    res = dict(base, available=True, reliable=ok, direction=direction, slope_pts_per_week=wk, latest=base["series"][-1], reason=why)
    if ok:
        f = forecast(points, "systolic", horizon=14)
        last = f["points"][-1] if f["points"] else None
        if last:
            res["estimate_in_14_days"] = dict(estimate=float(np.clip(last["estimate"], 0, 100)), low=float(max(0, last["low"])), high=float(min(100, last["high"])))
    return res


# ----------------------------------------------------------------------------- suggestions
def recommendations(ctx):
    """Rule-based personalised preventive suggestions from the patient's own data. Each carries its source."""
    r = []
    add = lambda text, area, p=2: r.append(dict(text=text, area=area, priority=p))
    for dis in ctx.get("high_risk", []):
        add(f"Your latest {dis} risk estimate is High. This is only a screening estimate; book a doctor visit to discuss proper testing.", "prediction", 1)
    for dis in ctx.get("moderate_risk", []):
        add(f"Your latest {dis} risk estimate is Moderate. Focus on diet, activity and a routine check-up.", "prediction", 2)
    b = ctx.get("bmi")
    if b and b >= 25:
        add(f"BMI {b} is above 25 (South-Asian risk starts near 23). A 5% weight loss through diet and activity lowers blood pressure and sugar risk; try the Diet Planner.", "weight", 2)
    elif b and b < 18.5:
        add(f"BMI {b} is below 18.5. Discuss unintended low weight with a doctor and consider a nutrition plan for healthy weight gain.", "weight", 2)
    for key, t in ctx.get("trends", {}).items():
        if not t.get("available"):
            continue
        if key in ("systolic", "diastolic") and t["direction"] == "Rising":
            add(f"Your {LABEL[key]} is trending up ({t['slope_per_week']:+} mmHg/week). Cut salt, stay active, and have it checked if it keeps rising.", "blood pressure", 1)
        if key == "blood_sugar" and t["direction"] == "Rising":
            add(f"Your blood sugar readings are trending up ({t['slope_per_week']:+} mg/dL/week). Reduce sugary food, walk after meals, and ask your doctor about HbA1c.", "blood sugar", 1)
        if key == "weight_kg" and abs(t["slope_per_week"]) > 1:
            add(f"Your weight is changing quickly ({t['slope_per_week']:+} kg/week). Unplanned changes deserve a doctor's review.", "weight", 1)
        if key == "heart_rate" and t["direction"] == "Rising":
            add("Your resting heart rate is trending up. Check sleep, caffeine, stress and hydration; see a doctor if it persists.", "heart rate", 2)
    for a in ctx.get("abnormal_params", [])[:4]:
        add(f"Your latest lab report shows {a['name']} {a['status'].lower()} versus the range printed on the report. Review it with your doctor.", "report", 1)
    if ctx.get("anomalies_count"):
        add(f"{ctx['anomalies_count']} unusual reading(s) were flagged in your history. Re-measure and confirm they are not entry errors.", "anomaly", 2)
    n = ctx.get("nutrition", {})
    if n.get("days_logged", 0) >= 3 and n.get("target_kcal"):
        ratio = n["avg_kcal"] / n["target_kcal"]
        if ratio < 0.8:
            add("Your logged intake averages under 80% of your calorie target. Do not skip meals; ask a dietitian if this is intentional.", "nutrition", 2)
        elif ratio > 1.2:
            add("Your logged intake averages over 120% of your calorie target. Review portion sizes and snacks.", "nutrition", 2)
    if n.get("water_days", 0) >= 3 and n.get("avg_water_ml", 0) < 1500 and n.get("water_limit_ok", True):
        add("Your logged water intake is low. Spread about 8 glasses through the day unless your doctor limits fluids.", "hydration", 3)
    if ctx.get("n_readings", 0) < 6:
        add("Log blood pressure, sugar, heart rate and weight regularly (at least weekly) so trends and forecasts become available.", "data", 3)
    if not ctx.get("has_report"):
        add("Upload your latest blood-test report to enable lab-based insights.", "data", 3)
    r.append(dict(text="Annual health check-up, 150 minutes of weekly activity, 7-9 hours of sleep and no tobacco are the basics that protect every organ system.", area="general", priority=3))
    return sorted(r, key=lambda x: x["priority"])


def summarize_metric(rows, prev_rows, key):
    cur = [r[key] for r in rows if r.get(key) is not None]
    prv = [r[key] for r in prev_rows if r.get(key) is not None]
    if not cur:
        return dict(n=0)
    out = dict(n=len(cur), mean=round(float(np.mean(cur)), 1), min=round(float(min(cur)), 1), max=round(float(max(cur)), 1))
    if prv:
        out["prev_mean"] = round(float(np.mean(prv)), 1)
        out["change"] = round(out["mean"] - out["prev_mean"], 1)
    return out

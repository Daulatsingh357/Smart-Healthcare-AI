"""Shared engine for the four disease-risk models: validation, training, evaluation, prediction, explanation.

Honesty notes
* Unless a real CSV is placed in database/datasets/<disease>.csv the models are trained on SYNTHETIC data whose
  labels are generated from a hand-written formula. Metrics on synthetic data only prove that the pipeline works;
  they say nothing about clinical accuracy. Every prediction therefore carries `is_synthetic` and the UI shows it.
* The Low/Moderate/High cut-offs (30 % / 60 %) are illustrative, NOT clinically validated thresholds.
"""
import hashlib, json, os
from datetime import datetime
import joblib, numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "saved_models")
DATASET_DIR = os.path.join(HERE, "..", "database", "datasets")
LOW_MAX, MODERATE_MAX = 30.0, 60.0
DISCLAIMER = ("Experimental risk estimate, not a diagnosis. It cannot confirm or rule out any disease. "
              "Please consult a qualified doctor for interpretation and testing.")
_cache = {}


class ModelNotTrained(RuntimeError):
    pass


# ------------------------------------------------------------------ feature helpers used by the spec modules
def num_feature(name, label, unit, lo, hi, step="any"):
    return dict(name=name, label=label, unit=unit, min=lo, max=hi, step=step, type="number")


def bin_feature(name, label, choices):
    return dict(name=name, label=label, unit="", min=0, max=1, type="select", choices=choices)


SEX = bin_feature("sex", "Gender", [(0, "Female"), (1, "Male")])
CONTEXT_FIELDS = [num_feature("height_cm", "Height", "cm", 80, 250), num_feature("weight_kg", "Weight", "kg", 20, 300)]


def algorithms(seed=1):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(150, max_depth=6, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=seed),
        "SVM": SVC(probability=True, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.08, subsample=0.9,
                                 eval_metric="logloss", random_state=seed, n_jobs=1),
    }


def risk_level(pct):
    return "Low" if pct < LOW_MAX else "Moderate" if pct < MODERATE_MAX else "High"


# ------------------------------------------------------------------ validation / preprocessing
def _num(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def validate_inputs(spec, raw):
    """Returns (clean_values, errors). BMI is derived from height/weight when not supplied.
    Out-of-range values are rejected (never silently clipped)."""
    raw, clean, errors = dict(raw or {}), {}, {}
    if isinstance(raw.get("sex"), str):
        raw["sex"] = {"male": 1, "m": 1, "female": 0, "f": 0}.get(raw["sex"].strip().lower(), raw["sex"])
    h, w = _num(raw.get("height_cm")), _num(raw.get("weight_kg"))
    if _num(raw.get("bmi")) is None and h and w and 80 <= h <= 250 and 20 <= w <= 300:
        raw["bmi"] = round(w / ((h / 100) ** 2), 1)
    for f in spec["features"]:
        v = _num(raw.get(f["name"]))
        if v is None:
            errors[f["name"]] = f"{f['label']} is required and must be a number."
        elif not f["min"] <= v <= f["max"]:
            errors[f["name"]] = f"{f['label']} must be between {f['min']} and {f['max']} {f['unit']}.".replace("  ", " ")
        elif f["type"] == "select" and v not in (0, 1):
            errors[f["name"]] = f"{f['label']} must be one of the listed options."
        else:
            clean[f["name"]] = v
    for f in CONTEXT_FIELDS:  # optional context, stored with the prediction but not fed to the model
        v = _num(raw.get(f["name"]))
        if v is not None:
            if f["min"] <= v <= f["max"]:
                clean[f["name"]] = v
            else:
                errors[f["name"]] = f"{f['label']} must be between {f['min']} and {f['max']} {f['unit']}."
    return clean, errors


# ------------------------------------------------------------------ training
def _load_or_generate(spec, n, seed):
    path = os.path.join(DATASET_DIR, spec["dataset_file"])
    names = [f["name"] for f in spec["features"]]
    if os.path.exists(path):
        df = pd.read_csv(path)
        missing = [c for c in names + ["target"] if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}")
        return df[names + ["target"]].dropna(), f"user-provided dataset: database/datasets/{spec['dataset_file']}"
    return spec["generate"](n, np.random.default_rng(seed)), "SYNTHETIC (generated from a hand-written formula; not real patients)"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def train_disease(spec, n=3000, seed=42, model_dir=None, verbose=False):
    model_dir = model_dir or MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    df, source = _load_or_generate(spec, n, seed)
    names = [f["name"] for f in spec["features"]]
    X, y = df[names], df["target"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    models, results, best = {}, {}, (-1, None)
    for name, algo in algorithms(seed).items():
        clf = CalibratedClassifierCV(make_pipeline(StandardScaler(), algo), method="sigmoid", cv=3).fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        r = dict(accuracy=round(accuracy_score(yte, pred), 4), precision=round(precision_score(yte, pred, zero_division=0), 4),
                 recall=round(recall_score(yte, pred, zero_division=0), 4), f1=round(f1_score(yte, pred, zero_division=0), 4),
                 roc_auc=round(roc_auc_score(yte, p), 4), confusion_matrix=confusion_matrix(yte, pred, labels=[0, 1]).tolist())
        models[name], results[name] = clf, r
        if verbose:
            print(f"  {spec['key']:9s} {name:20s} acc={r['accuracy']:.3f} f1={r['f1']:.3f} auc={r['roc_auc']:.3f}")
        if r["roc_auc"] > best[0]:
            best = (r["roc_auc"], name)
    version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    bundle = dict(disease=spec["key"], models=models, best=best[1], features=names, medians=X.median().to_dict(),
                  version=version, data_source=source, n_samples=int(len(df)), prevalence=round(float(y.mean()), 3))
    path = os.path.join(model_dir, f"{spec['key']}.joblib")
    joblib.dump(bundle, path)
    meta = dict(version=version, data_source=source, n_samples=int(len(df)), prevalence=bundle["prevalence"], best=best[1],
                trained_at=datetime.utcnow().isoformat(timespec="seconds"), artifact_sha256=_sha256(path), algorithms=results)
    mpath = os.path.join(model_dir, "metrics.json")
    allm = json.load(open(mpath)) if os.path.exists(mpath) else {}
    allm[spec["key"]] = meta
    json.dump(allm, open(mpath, "w"), indent=2)
    _cache.pop((model_dir, spec["key"]), None)
    return meta


def load_metrics(model_dir=None):
    p = os.path.join(model_dir or MODEL_DIR, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def load_bundle(disease, model_dir=None):
    model_dir = model_dir or MODEL_DIR
    key = (model_dir, disease)
    path = os.path.join(model_dir, f"{disease}.joblib")
    if not os.path.exists(path):
        raise ModelNotTrained("Models are not trained yet. Run: python ml_models\\train_all.py")
    mtime = os.path.getmtime(path)
    if key not in _cache or _cache[key][0] != mtime:
        _cache[key] = (mtime, joblib.load(path))
    return _cache[key][1]


# ------------------------------------------------------------------ prediction
def _proba(model, features, values):
    X = pd.DataFrame([[values[f] for f in features]], columns=features)
    return float(model.predict_proba(X)[0, 1])


def predict(spec, clean, algorithm=None, model_dir=None):
    b = load_bundle(spec["key"], model_dir)
    algo = algorithm if algorithm in b["models"] else b["best"]
    model, feats = b["models"][algo], b["features"]
    p = _proba(model, feats, clean)
    risk = round(p * 100, 1)
    level = risk_level(risk)
    # Local explanation: replace one feature at a time by the training median and see how the risk moves.
    labels = {f["name"]: f for f in spec["features"]}
    factors = []
    for f in feats:
        alt = dict(clean); alt[f] = b["medians"][f]
        delta = (p - _proba(model, feats, alt)) * 100
        factors.append(dict(feature=f, label=labels[f]["label"], value=clean[f], effect_points=round(delta, 1),
                            direction="raises" if delta > 0 else "lowers"))
    factors = sorted(factors, key=lambda x: -abs(x["effect_points"]))[:3]
    tips = ["Keep a balanced diet, stay physically active and sleep 7-9 hours.", "Have routine check-ups and blood tests as advised by your doctor."]
    tips += spec["tips"].get(level, [])
    if level != "Low":
        tips.append("Please discuss this result with a doctor; only clinical tests can confirm or exclude a condition.")
    return dict(disease=spec["key"], risk=risk, level=level, model=algo, version=b["version"], data_source=b["data_source"],
                is_synthetic=b["data_source"].startswith("SYNTHETIC"), factors=factors, tips=tips, disclaimer=DISCLAIMER,
                available_models=list(b["models"]))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

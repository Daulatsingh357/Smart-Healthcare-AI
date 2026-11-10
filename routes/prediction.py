"""Module 1 - Multi-disease risk prediction (REST API + page)."""
from flask import Blueprint, jsonify, render_template, request, Response
from flask_login import current_user
from ml_models import SPECS, common
from models import db, DiseasePrediction, PredictionInput, ModelVersion, HealthMetric
from routes import roles_required, current_patient, api_error
from utils.model_registry import register_models
from utils.pdf import build_pdf

bp = Blueprint("prediction", __name__)
VITAL_MAP = {"blood_pressure": "systolic", "glucose": "blood_sugar", "heart_rate": "heart_rate", "cholesterol": "cholesterol"}


def _spec_public(spec):
    return dict(key=spec["key"], title=spec["title"], features=spec["features"], context=common.CONTEXT_FIELDS)


def _row(p):
    return dict(id=p.id, disease=p.disease, risk=p.risk_percent, level=p.risk_level, model=p.model_name, created_at=p.created_at.isoformat(timespec="minutes"))


@bp.route("/predict")
@roles_required("patient")
def page():
    p = current_patient()
    metrics = common.load_metrics()
    return render_template("predict.html", specs=[_spec_public(s) for s in SPECS.values()], trained=bool(metrics),
                           profile=dict(age=p.age, sex=(1 if p.gender == "Male" else 0 if p.gender == "Female" else None), height_cm=p.height_cm, weight_kg=p.weight_kg, bmi=p.bmi))


@bp.route("/predict/models")
@roles_required("patient", "admin", "doctor")
def models_page():
    return render_template("model_eval.html", metrics=common.load_metrics(), specs=SPECS)


@bp.route("/api/models")
@roles_required("patient", "admin", "doctor")
def models_api():
    return jsonify(common.load_metrics())


@bp.route("/api/predictions", methods=["POST"])
@roles_required("patient")
def create():
    data = request.get_json(silent=True) or {}
    spec = SPECS.get(data.get("disease"))
    if not spec:
        return api_error("Unknown disease module.", 400)
    clean, errors = common.validate_inputs(spec, data.get("inputs"))
    if errors:
        return api_error("Please correct the highlighted fields.", 422, fields=errors)
    try:
        res = common.predict(spec, clean, data.get("algorithm"))
    except common.ModelNotTrained as e:
        return api_error(str(e), 503)
    mv = ModelVersion.query.filter_by(disease=spec["key"], algorithm=res["model"], version=res["version"]).first()
    if not mv:
        register_models()
        mv = ModelVersion.query.filter_by(disease=spec["key"], algorithm=res["model"], version=res["version"]).first()
    row = DiseasePrediction(patient_id=current_patient().id, disease=spec["key"], model_name=res["model"], model_version_id=mv.id if mv else None,
                            risk_percent=res["risk"], risk_level=res["level"], top_factors=res["factors"])
    row.inputs = [PredictionInput(feature_name=k, feature_value=v) for k, v in clean.items()]
    db.session.add(row)
    vit = {dst: clean[src] for src, dst in VITAL_MAP.items() if src in clean}
    if clean.get("weight_kg"):
        vit["weight_kg"] = clean["weight_kg"]
    if vit:
        db.session.add(HealthMetric(patient_id=current_patient().id, source="prediction", **vit))
    db.session.commit()
    return jsonify(dict(res, id=row.id, created_at=row.created_at.isoformat(timespec="minutes"), pdf_url=f"/api/predictions/{row.id}/pdf")), 201


@bp.route("/api/predictions")
@roles_required("patient")
def history():
    q = DiseasePrediction.query.filter_by(patient_id=current_patient().id)
    if request.args.get("disease") in SPECS:
        q = q.filter_by(disease=request.args["disease"])
    limit = min(int(request.args.get("limit", 50) or 50), 200)
    return jsonify([_row(p) for p in q.order_by(DiseasePrediction.created_at.desc(), DiseasePrediction.id.desc()).limit(limit)])


def _own(pid):
    p = db.get_or_404(DiseasePrediction, pid)
    if p.patient_id != current_patient().id:
        from flask import abort
        abort(403)
    return p


@bp.route("/api/predictions/<int:pid>")
@roles_required("patient")
def detail(pid):
    p = _own(pid)
    return jsonify(dict(_row(p), inputs={i.feature_name: i.feature_value for i in p.inputs}, factors=p.top_factors or [],
                        model_version=p.model_version.version if p.model_version else None,
                        data_source=p.model_version.data_source if p.model_version else None))


@bp.route("/api/predictions/<int:pid>", methods=["DELETE"])
@roles_required("patient")
def delete(pid):
    db.session.delete(_own(pid)); db.session.commit()
    return jsonify(ok=True)


@bp.route("/api/predictions/<int:pid>/pdf")
@roles_required("patient")
def pdf(pid):
    p = _own(pid)
    spec = SPECS[p.disease]
    labels = {f["name"]: f for f in spec["features"] + common.CONTEXT_FIELDS}
    rows = []
    for i in p.inputs:
        f = labels.get(i.feature_name, {})
        shown = dict(f.get("choices", [])).get(int(i.feature_value), i.feature_value) if f.get("type") == "select" else i.feature_value
        rows.append([f.get("label", i.feature_name), shown, f.get("unit", "")])
    mv = p.model_version
    synthetic = bool(mv and mv.data_source.startswith("SYNTHETIC"))
    secs = [dict(heading="Result", paragraphs=[f"Estimated risk: {p.risk_percent}%  ({p.risk_level})", f"Model: {p.model_name}" + (f", version {mv.version}" if mv else ""),
                                              "This is an experimental statistical estimate. It is not a diagnosis and cannot confirm or exclude " + p.disease + " disease."]),
            dict(heading="Inputs provided", table=dict(header=["Parameter", "Value", "Unit"], rows=rows, widths=[80, 50, 40])),
            dict(heading="Main contributing factors (what-if analysis)", notes=[f"{f['label']} = {f['value']}: {f['direction']} the estimate by about {abs(f['effect_points'])} percentage points vs a typical value" for f in (p.top_factors or [])]),
            dict(heading="Limitations", notes=["Risk bands (Low <30%, Moderate 30-60%, High >=60%) are illustrative and not clinically validated.",
                                               ("The model was trained on SYNTHETIC data, so the percentage has no proven clinical meaning." if synthetic else "Performance on your population is unknown; validate before clinical use."),
                                               "Only a doctor using clinical examination and laboratory tests can diagnose a disease."])]
    data = build_pdf(f"{spec['title']} Report", f"Patient: {current_user.name}   |   Generated from prediction #{p.id}, {p.created_at:%d %b %Y %H:%M} UTC", secs)
    return Response(data, mimetype="application/pdf", headers={"Content-Disposition": f"attachment; filename=risk_{p.disease}_{p.id}.pdf"})

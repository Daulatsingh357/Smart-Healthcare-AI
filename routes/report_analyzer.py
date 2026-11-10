"""Module 2 - AI Medical Report Analyzer. Files are validated by magic bytes and stored AES-encrypted (Fernet)."""
import os, uuid
from datetime import date, datetime
from flask import Blueprint, Response, abort, current_app, jsonify, render_template, request
from flask_login import current_user
from sqlalchemy import func
from werkzeug.utils import secure_filename
from ml_models import report_extractor as RX
from models import db, MedicalReport, ExtractedReportValue
from routes import roles_required, current_patient, api_error
from utils.pdf import build_pdf
from utils.security import decrypt_bytes, encrypt_bytes, sha256_hex, sniff_type

bp = Blueprint("report_analyzer", __name__)
EXPLAIN_LABEL = "AI-generated general explanation (educational; not about your personal diagnosis)"


def _own(rid):
    r = db.get_or_404(MedicalReport, rid)
    if r.patient_id != current_patient().id:
        abort(403)
    return r


def _val(v, with_text=True):
    d = dict(id=v.id, key=v.param_key, name=v.param_name, display=RX.PARAMS[v.param_key]["name_en"] if v.param_key in RX.PARAMS else v.param_name, display_hi=RX.PARAMS.get(v.param_key, {}).get("name_hi", v.param_name),
             value=v.value, unit=v.unit, ref_low=v.ref_low, ref_high=v.ref_high, ref_text=v.ref_text, status=v.status, status_source=v.status_source, raw_line=v.raw_line)
    if with_text:
        d["explanation_en"], d["explanation_hi"] = RX.explanation(v.param_key, v.status, "en"), RX.explanation(v.param_key, v.status, "hi")
    return d


def _rep(r, values=False):
    d = dict(id=r.id, title=r.title, report_date=r.report_date.isoformat() if r.report_date else None, uploaded_at=r.uploaded_at.isoformat(timespec="minutes"),
             file_type=r.file_type, size_bytes=r.size_bytes, status=r.status, value_count=r.value_count, abnormal_count=r.abnormal_count, method=r.extraction_method)
    if values:
        d["values"] = [_val(v) for v in r.values]
        d["explanation_label"] = EXPLAIN_LABEL
        d["ocr_text"] = r.ocr_text
    return d


@bp.route("/reports")
@roles_required("patient")
def index():
    return render_template("reports.html")


@bp.route("/reports/<int:rid>")
@roles_required("patient")
def detail(rid):
    _own(rid)
    return render_template("report_detail.html", rid=rid)


@bp.route("/api/reports", methods=["POST"])
@roles_required("patient")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return api_error("Choose a PDF, JPG or PNG file.")
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in current_app.config["ALLOWED_EXT"]:
        return api_error("Only PDF, JPG, JPEG and PNG files are allowed.", 415)
    data = f.read()
    kind = sniff_type(data)
    if not kind or (kind == "jpg" and ext not in ("jpg", "jpeg")) or (kind == "png" and ext != "png") or (kind == "pdf" and ext != "pdf"):
        return api_error("The file content does not match its extension (or is not a valid PDF/JPG/PNG).", 415)
    digest = sha256_hex(data)
    dup = MedicalReport.query.filter_by(patient_id=current_patient().id, sha256=digest).first()
    if dup:
        return api_error("This exact file was already uploaded.", 409, existing_id=dup.id)
    try:
        text, method = RX.extract_text(data, kind)
    except RX.OCRUnavailable as e:
        return api_error(str(e), 503)
    except Exception as e:  # corrupt PDF / unreadable image
        return api_error(f"The file could not be read: {e.__class__.__name__}.", 422)
    values = RX.parse_values(text)
    form_date = request.form.get("report_date", "").strip()
    try:
        rdate = date.fromisoformat(form_date) if form_date else (RX.guess_report_date(text) or date.today())
    except ValueError:
        return api_error("Report date must be YYYY-MM-DD.")
    if rdate > date.today():
        return api_error("Report date cannot be in the future.")
    name = uuid.uuid4().hex + ".enc"
    with open(os.path.join(current_app.config["UPLOAD_FOLDER"], name), "wb") as fh:
        fh.write(encrypt_bytes(data))
    import re
    title = re.sub(r"[\x00-\x1f<>]", "", request.form.get("title") or "").strip() or secure_filename(f.filename) or "report"
    rep = MedicalReport(patient_id=current_patient().id, title=title[:120], filename=name, original_name=secure_filename(f.filename)[:200], file_type=kind, size_bytes=len(data), sha256=digest,
                        report_date=rdate, ocr_text=text, extraction_method=method, status="analyzed" if values else "no_values", value_count=len(values),
                        abnormal_count=sum(v["status"] in ("Low", "High") for v in values))
    rep.values = [ExtractedReportValue(param_key=v["key"], param_name=v["name"][:80], value=v["value"], unit=(v["unit"] or None), ref_low=v["ref_low"], ref_high=v["ref_high"],
                                       ref_text=v["ref_text"], status=v["status"], status_source=v["status_source"], raw_line=v["raw_line"]) for v in values]
    db.session.add(rep); db.session.commit()
    return jsonify(_rep(rep, values=True)), 201


@bp.route("/api/reports")
@roles_required("patient")
def listing():
    rows = MedicalReport.query.filter_by(patient_id=current_patient().id).order_by(MedicalReport.report_date.desc(), MedicalReport.id.desc()).all()
    return jsonify([_rep(r) for r in rows])


@bp.route("/api/reports/summary")
@roles_required("patient")
def summary():
    pid = current_patient().id
    rows = MedicalReport.query.filter_by(patient_id=pid).order_by(MedicalReport.report_date.desc(), MedicalReport.id.desc()).all()
    latest = rows[0] if rows else None
    return jsonify(total=len(rows), latest=_rep(latest) if latest else None, abnormal_in_latest=latest.abnormal_count if latest else 0,
                   abnormal_total=sum(r.abnormal_count or 0 for r in rows),
                   not_assessed_in_latest=sum(v.status == RX.NOT_ASSESSED for v in latest.values) if latest else 0)


@bp.route("/api/reports/<int:rid>")
@roles_required("patient")
def get(rid):
    return jsonify(_rep(_own(rid), values=True))


@bp.route("/api/reports/<int:rid>", methods=["DELETE"])
@roles_required("patient")
def delete(rid):
    r = _own(rid)
    try:
        os.remove(os.path.join(current_app.config["UPLOAD_FOLDER"], r.filename))
    except OSError:
        pass
    db.session.delete(r); db.session.commit()
    return jsonify(ok=True)


@bp.route("/api/reports/<int:rid>/file")
@roles_required("patient")
def file(rid):
    r = _own(rid)
    try:
        with open(os.path.join(current_app.config["UPLOAD_FOLDER"], r.filename), "rb") as fh:
            data = decrypt_bytes(fh.read())
    except (OSError, ValueError):
        return api_error("Stored file is unavailable.", 404)
    mime = {"pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg"}[r.file_type]
    return Response(data, mimetype=mime, headers={"Content-Disposition": f'attachment; filename="{r.original_name or r.title}"', "X-Content-Type-Options": "nosniff"})


@bp.route("/api/reports/parameters")
@roles_required("patient")
def parameters():
    rows = (db.session.query(ExtractedReportValue.param_key, func.count(ExtractedReportValue.id)).join(MedicalReport)
            .filter(MedicalReport.patient_id == current_patient().id).group_by(ExtractedReportValue.param_key).all())
    return jsonify([dict(key=k, name=RX.PARAMS[k]["name_en"] if k in RX.PARAMS else k, name_hi=RX.PARAMS.get(k, {}).get("name_hi"), count=c) for k, c in rows])


def _series(pid, key):
    q = (db.session.query(ExtractedReportValue, MedicalReport).join(MedicalReport).filter(MedicalReport.patient_id == pid, ExtractedReportValue.param_key == key)
         .order_by(MedicalReport.report_date, MedicalReport.id))
    return [dict(report_id=r.id, title=r.title, date=r.report_date.isoformat() if r.report_date else None, value=v.value, unit=v.unit, ref_low=v.ref_low,
                 ref_high=v.ref_high, ref_text=v.ref_text, status=v.status) for v, r in q]


@bp.route("/api/reports/compare")
@roles_required("patient")
def compare():
    key = request.args.get("param", "")
    if key not in RX.PARAMS:
        return api_error("Unknown parameter.", 400)
    pts = _series(current_patient().id, key)
    note = "Reference ranges differ between laboratories; each point is judged against the range printed on its own report."
    return jsonify(param=key, name=RX.PARAMS[key]["name_en"], points=pts, note=note)


@bp.route("/api/reports/compare/latest")
@roles_required("patient")
def compare_latest():
    """Latest vs previous report for every parameter present in both."""
    pid = current_patient().id
    reps = MedicalReport.query.filter_by(patient_id=pid).order_by(MedicalReport.report_date.desc(), MedicalReport.id.desc()).limit(2).all()
    if len(reps) < 2:
        return jsonify(available=False, reason="Upload at least two reports to compare.", rows=[])
    new, old = ({v.param_key: v for v in r.values} for r in reps)
    rows = []
    for k, v in new.items():
        if k in old:
            o = old[k]
            same_unit = (o.unit or "").lower() == (v.unit or "").lower()
            rows.append(dict(key=k, name=RX.PARAMS[k]["name_en"] if k in RX.PARAMS else k, previous=o.value, latest=v.value, unit=v.unit, change=round(v.value - o.value, 2) if same_unit else None,
                             previous_status=o.status, latest_status=v.status, unit_mismatch=not same_unit))
    return jsonify(available=True, latest=_rep(reps[0]), previous=_rep(reps[1]), rows=rows)


@bp.route("/api/reports/<int:rid>/pdf")
@roles_required("patient")
def pdf(rid):
    r = _own(rid)
    vals = list(r.values)
    table = dict(header=["Test (as printed)", "Value", "Unit", "Reference (from report)", "Status"],
                 rows=[[v.param_name, v.value, v.unit or "-", v.ref_text or "not printed", v.status] for v in vals], widths=[48, 22, 24, 44, 34],
                 highlight=[v.status in ("Low", "High") for v in vals])
    abn = [v for v in vals if v.status in ("Low", "High")]
    secs = [dict(heading="A. Extracted from your report (taken literally from the file)", paragraphs=[f"Report: {r.title}   |   Report date: {r.report_date}   |   Extraction: {r.extraction_method}",
                                                                                                         "Values, units and reference ranges below were read from the document. Automatic reading can contain errors; verify against the original."], table=table),
            dict(heading="B. Summary", paragraphs=[f"{len(vals)} value(s) found; {len(abn)} outside the range printed on the report; {sum(v.status == RX.NOT_ASSESSED for v in vals)} could not be assessed because the report printed no range."]),
            dict(heading="C. " + EXPLAIN_LABEL, notes=[f"{v.param_name} ({v.status}): {RX.explanation(v.param_key, v.status, 'en')}" for v in abn] or ["No out-of-range values to explain."])]
    data = build_pdf("Medical Report Summary", f"Patient: {current_user.name}   |   Report #{r.id}", secs)
    return Response(data, mimetype="application/pdf", headers={"Content-Disposition": f"attachment; filename=report_summary_{r.id}.pdf"})

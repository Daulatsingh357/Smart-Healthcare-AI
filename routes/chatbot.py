"""Module 3 - AI Health Assistant. Safety layer + retrieval engine in ml_models/chat_engine.py; this file adds persistence,
consent-gated personal context and doctor recommendations."""
from datetime import date
from flask import Blueprint, abort, jsonify, render_template, request
from models import db, ChatConversation, ChatMessage, Doctor, User, HealthMetric
from ml_models import report_extractor as RX, diet_recommender as DR
from ml_models.chat_engine import get_engine
from ml_models.chat_kb import UI
from routes import roles_required, current_patient, api_error
from utils import context as ctx

bp = Blueprint("chatbot", __name__)
CTX_TEXT = {
    "en": dict(report="Your latest report ({title}, {date}):", abn="Outside the range printed on the report:", none_abn="No values were outside the printed ranges.", na="{n} value(s) had no printed range and were not assessed.",
               bmi="Your BMI is {bmi} ({cat}) based on height {h} cm and weight {w} kg.", vitals="Your latest saved readings ({date}):", pred="Your latest saved risk estimates (experimental, not diagnoses):",
               diet="Your active diet plan targets {kcal} kcal/day. Logged today: {today} kcal.", rep_note="These explanations are general education, not a diagnosis. Please review the report with your doctor."),
    "hi": dict(report="आपकी नवीनतम रिपोर्ट ({title}, {date}):", abn="रिपोर्ट में छपी सीमा से बाहर:", none_abn="कोई मान रिपोर्ट की छपी सीमा से बाहर नहीं था।", na="{n} मान की सीमा रिपोर्ट में नहीं छपी थी, इसलिए उनका आकलन नहीं हुआ।",
               bmi="आपकी ऊँचाई {h} सेमी और वजन {w} किलो के आधार पर आपका बीएमआई {bmi} ({cat}) है।", vitals="आपकी नवीनतम सहेजी गई रीडिंग ({date}):", pred="आपके नवीनतम सहेजे गए जोखिम अनुमान (प्रयोगात्मक, निदान नहीं):",
               diet="आपके सक्रिय डाइट प्लान का लक्ष्य {kcal} किलो कैलोरी/दिन है। आज दर्ज: {today} किलो कैलोरी।", rep_note="ये व्याख्याएँ सामान्य शिक्षा हैं, निदान नहीं। कृपया रिपोर्ट डॉक्टर को दिखाएँ।"),
}


def _context_reply(name, p, lang):
    t, ui = CTX_TEXT[lang], UI[lang]
    if name == "report":
        r = ctx.latest_report(p.id)
        if not r:
            return ui["no_data"]
        lines = [t["report"].format(title=r.title, date=r.report_date)]
        abn = [v for v in r.values if v.status in ("Low", "High")]
        if abn:
            lines.append(t["abn"])
            lines += [f"- {v.param_name}: {v.value:g} {v.unit or ''} ({v.status}). {RX.explanation(v.param_key, v.status, lang)}" for v in abn]
        else:
            lines.append(t["none_abn"])
        na = sum(v.status == RX.NOT_ASSESSED for v in r.values)
        if na:
            lines.append(t["na"].format(n=na))
        return "\n".join(lines + [t["rep_note"]])
    if name == "bmi":
        return t["bmi"].format(bmi=p.bmi, cat=DR.bmi_category(p.bmi), h=p.height_cm, w=p.weight_kg) if p.bmi else ui["no_data"]
    if name == "vitals":
        m = HealthMetric.query.filter_by(patient_id=p.id).order_by(HealthMetric.recorded_at.desc()).first()
        if not m:
            return ui["no_data"]
        parts = [f"{lbl}: {val:g}" for lbl, val in (("BP sys", m.systolic), ("BP dia", m.diastolic), ("Sugar", m.blood_sugar), ("HR", m.heart_rate), ("Weight", m.weight_kg)) if val is not None]
        return t["vitals"].format(date=m.recorded_at.date()) + " " + ", ".join(parts)
    if name == "predictions":
        preds = ctx.latest_predictions(p.id)
        return (t["pred"] + "\n" + "\n".join(f"- {d.title()}: {x.risk_level} ({x.risk_percent}%)" for d, x in preds.items())) if preds else ui["no_data"]
    if name == "diet":
        plan = ctx.active_plan(p.id)
        return t["diet"].format(kcal=plan.calories, today=ctx.day_totals(p.id, date.today())["kcal"]) if plan else ui["no_data"]
    return ui["no_data"]


def _doctors(specialty):
    q = Doctor.query.join(User)
    found = q.filter(Doctor.specialization.ilike(f"%{specialty}%")).limit(3).all() if specialty else []
    if not found:
        found = q.filter(Doctor.specialization.ilike("%General%")).limit(3).all()
    return [dict(id=d.id, name=d.user.name, specialization=d.specialization) for d in found]


def _own_conv(cid):
    c = db.get_or_404(ChatConversation, cid)
    if c.patient_id != current_patient().id:
        abort(403)
    return c


@bp.route("/chatbot")
@roles_required("patient")
def page():
    return render_template("chatbot.html", consent=bool(current_patient().chat_context_consent))


@bp.route("/api/chat", methods=["POST"])
@roles_required("patient")
def chat():
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return api_error("Type a message first.")
    if len(msg) > 600:
        return api_error("Message is too long (max 600 characters).")
    p = current_patient()
    conv = _own_conv(data["conversation_id"]) if data.get("conversation_id") else None
    if conv is None:
        conv = ChatConversation(patient_id=p.id, title=msg[:60], language=data.get("language", "en"))
        db.session.add(conv); db.session.flush()
    last = next((m for m in reversed(conv.messages) if m.role == "assistant"), None)
    state = {"last_topic": (last.meta or {}).get("topic") if last else None}
    res = get_engine().respond(msg, data.get("language", "en"), state, consent=bool(p.chat_context_consent))
    if res["context"] in ("report", "bmi", "vitals", "predictions", "diet"):
        res["reply"] = _context_reply(res["context"], p, res["lang"])
        res["followups"] = get_engine()._followups(["diet", "exercise"], res["lang"])
    docs = _doctors(res["specialty"]) if res["see_doctor"] and res["topic"] != "emergency" else []
    db.session.add(ChatMessage(conversation_id=conv.id, role="user", content=msg, language=res["lang"], is_emergency=res["emergency"]))
    db.session.add(ChatMessage(conversation_id=conv.id, role="assistant", content=res["reply"], language=res["lang"], is_emergency=res["emergency"],
                               meta=dict(topic=res["topic"], level=res["level"], followups=res["followups"], doctors=docs)))
    db.session.commit()
    return jsonify(conversation_id=conv.id, reply=res["reply"], lang=res["lang"], topic=res["topic"], level=res["level"], emergency=res["emergency"],
                   followups=res["followups"], doctors=docs, used_personal_data=bool(res["context"] in ("report", "bmi", "vitals", "predictions", "diet")))


@bp.route("/api/chat/conversations")
@roles_required("patient")
def conversations():
    rows = ChatConversation.query.filter_by(patient_id=current_patient().id).order_by(ChatConversation.updated_at.desc()).limit(50).all()
    return jsonify([dict(id=c.id, title=c.title, updated_at=c.updated_at.isoformat(timespec="minutes")) for c in rows])


@bp.route("/api/chat/conversations/<int:cid>")
@roles_required("patient")
def messages(cid):
    c = _own_conv(cid)
    return jsonify([dict(id=m.id, role=m.role, content=m.content, emergency=m.is_emergency, meta=m.meta or {}, at=m.created_at.isoformat(timespec="minutes")) for m in c.messages])


@bp.route("/api/chat/conversations/<int:cid>", methods=["DELETE"])
@roles_required("patient")
def delete(cid):
    db.session.delete(_own_conv(cid)); db.session.commit()
    return jsonify(ok=True)


@bp.route("/api/chat/consent", methods=["POST"])
@roles_required("patient")
def consent():
    current_patient().chat_context_consent = bool((request.get_json(silent=True) or {}).get("consent"))
    db.session.commit()
    return jsonify(consent=current_patient().chat_context_consent)

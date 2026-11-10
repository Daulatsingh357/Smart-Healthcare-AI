import pytest
from ml_models.chat_engine import ChatEngine, emergency_check, needs_refusal
from ml_models.chat_kb import TOPICS
from conftest import make_client


def chat(c, msg, cid=None, lang="en"):
    r = c.post("/api/chat", json=dict(message=msg, conversation_id=cid, language=lang))
    assert r.status_code == 200, r.get_json()
    return r.get_json()


@pytest.mark.parametrize("msg", ["I have severe chest pain", "I can't breathe properly", "my father has slurred speech and face drooping", "he is unconscious",
                                 "मुझे सीने में दर्द है", "मरीज बेहोश हो गया", "she is vomiting blood"])
def test_emergency_detection(pc, msg):
    j = chat(pc, msg)
    assert j["emergency"] is True and j["level"] == "emergency" and ("112" in j["reply"])


def test_self_harm_gets_helpline_not_advice(pc):
    j = chat(pc, "I want to end my life")
    assert j["emergency"] and "14416" in j["reply"]


def test_negated_chest_pain_is_not_emergency():
    assert emergency_check("I have no chest pain, just want to sleep better") is None


@pytest.mark.parametrize("msg", ["which tablet should I take for fever", "how many mg of paracetamol can I take", "prescribe me an antibiotic", "do i have diabetes",
                                 "बुखार में कौन सी दवा लूँ"])
def test_refuses_diagnosis_prescription_dosage(pc, msg):
    j = chat(pc, msg)
    assert j["topic"] == "refusal" and not j["emergency"]
    assert "mg" not in j["reply"].lower().replace("doses", "") or "can't" in j["reply"] or "नहीं" in j["reply"]


def test_every_canonical_question_and_followup_is_answerable():
    e = ChatEngine()
    for t in TOPICS:
        for q, lang in ((t["q_en"], "en"), (t["q_hi"], "hi")):
            r = e.respond(q, lang)
            assert r["topic"] == t["id"], (q, r["topic"])
            assert r["followups"] and all(f for f in r["followups"])


def test_bilingual_answers_and_language_detection(pc):
    en = chat(pc, "How much water should I drink?")
    hi = chat(pc, "मुझे कितना पानी पीना चाहिए?")
    assert en["lang"] == "en" and hi["lang"] == "hi"
    assert any("\u0900" <= ch <= "\u097f" for ch in hi["reply"]) and not any("\u0900" <= ch <= "\u097f" for ch in en["reply"])


def test_off_topic_falls_back_and_lab_terms_explained(pc):
    assert chat(pc, "who won the cricket match")["topic"] == "fallback"
    assert chat(pc, "what does SGPT mean")["topic"] == "param:alt"


def test_context_followup_uses_previous_topic(pc):
    a = chat(pc, "how can I lower my cholesterol")
    b = chat(pc, "what about exercise for it?", cid=a["conversation_id"])
    assert a["topic"] == "cholesterol" and b["topic"] in ("exercise", "cholesterol", "heart")


def test_duration_triggers_see_doctor_and_suggests_doctors(app):
    c = make_client(app, email="p3@example.com")
    make_client(app, email="d@example.com", name="Dr Gen", role="doctor", specialization="General Physician")
    j = chat(c, "I have had a fever for 4 days")
    assert j["level"] == "see_doctor" and j["doctors"] and "doctor" in j["reply"].lower()


def test_personal_data_requires_consent(pc):
    pc.post("/api/predictions", json=dict(disease="heart", inputs=dict(age=60, sex=1, bmi=29, blood_pressure=150, cholesterol=240, heart_rate=82, glucose=120, smoker=1)))
    j = chat(pc, "what is my bmi")
    assert j["used_personal_data"] is False and "consent" in j["reply"].lower()
    assert pc.post("/api/chat/consent", json={"consent": True}).get_json()["consent"] is True
    j = chat(pc, "what is my bmi")
    assert j["used_personal_data"] is True and "26.4" in j["reply"]
    j = chat(pc, "show my risk predictions") if False else chat(pc, "what are my predictions") 
    pc.post("/api/chat/consent", json={"consent": False})
    assert chat(pc, "what is my bmi")["used_personal_data"] is False


def test_report_explanation_with_consent(pc):
    import io
    from conftest import lab_pdf, LAB_LINES
    pc.post("/api/reports", data=dict(file=(io.BytesIO(lab_pdf(LAB_LINES)), "l.pdf")), content_type="multipart/form-data")
    pc.post("/api/chat/consent", json={"consent": True})
    j = chat(pc, "explain my report")
    assert "Hemoglobin" in j["reply"] and "not a diagnosis" in j["reply"].lower()


def test_history_persisted_listed_and_isolated(pc, pc2):
    a = chat(pc, "hello"); chat(pc, "how to sleep better", cid=a["conversation_id"])
    cid = a["conversation_id"]
    msgs = pc.get(f"/api/chat/conversations/{cid}").get_json()
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert pc.get("/api/chat/conversations").get_json()[0]["id"] == cid
    assert pc2.get(f"/api/chat/conversations/{cid}").status_code == 403
    assert pc2.post("/api/chat", json=dict(message="hi", conversation_id=cid)).status_code == 403
    assert pc.delete(f"/api/chat/conversations/{cid}").status_code == 200
    assert pc.get(f"/api/chat/conversations/{cid}").status_code == 404


def test_input_limits(pc):
    assert pc.post("/api/chat", json=dict(message="  ")).status_code == 400
    assert pc.post("/api/chat", json=dict(message="x" * 601)).status_code == 400

"""Health-assistant NLP engine (offline, no external LLM).

Pipeline per message:
  1. language detection (Devanagari -> Hindi)
  2. EMERGENCY screen (rule-based, always first) -> fixed emergency guidance
  3. diagnosis / prescription / dosage screen -> polite refusal + doctor suggestion
  4. small-talk, personal-context intents (need patient consent; answered by the route from the DB)
  5. TF-IDF retrieval (character n-grams, works for English + Hindi) over the knowledge base + lab-parameter glossary
  6. conversational context: short follow-ups reuse the previous topic
  7. escalation hints (duration / worsening) and suggested follow-up questions
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .chat_kb import TOPICS, TOPIC_BY_ID, UI, STARTERS
from .report_extractor import PARAMS, explanation

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
EMERGENCY_EN = [r"chest (pain|pressure|tightness)", r"(can'?t|cannot|unable to|difficulty|trouble|struggling to) breathe?", r"breathing (problem|difficulty)",
                r"short(ness)? of breath", r"\bstroke\b", r"face (is )?droop", r"slurred speech", r"unconscious", r"passed out", r"\bfaint(ed|ing)\b",
                r"seizure", r"convulsion", r"(severe|heavy|uncontrolled) bleeding", r"vomit(ing)? blood", r"blood in (my )?(vomit|stool)", r"cough(ing)? (up )?blood",
                r"overdose", r"poison", r"anaphyla", r"throat (is )?swelling", r"sudden (severe|worst) headache", r"sudden (weakness|numbness)", r"snake ?bite",
                r"heart attack"]
SELFHARM_EN = [r"suicid", r"kill myself", r"end my life", r"want to die", r"self[- ]?harm", r"hurt myself"]
EMERGENCY_HI = ["सीने में दर्द", "सीने में जकड़न", "सांस नहीं", "साँस नहीं", "सांस लेने में तकलीफ", "साँस लेने में तकलीफ", "सांस फूल", "साँस फूल", "बेहोश", "दौरा पड़", "लकवा",
                "खून की उल्टी", "खून की उलटी", "खून की उल्टियाँ", "बहुत खून", "ज़हर", "जहर", "सांप ने काट", "साँप ने काट", "दिल का दौरा", "हार्ट अटैक"]
SELFHARM_HI = ["आत्महत्या", "जान देना", "मर जाना चाहता", "मरना चाहता", "मरना चाहती", "खुद को नुकसान", "खुद को चोट"]
REFUSE_EN = [r"\bprescrib", r"\bdos(e|age)\b", r"how (many|much) (mg|milligram|tablet|pill|ml|capsule)", r"\b\d+\s?(mg|mcg|ml)\b", r"which (medicine|tablet|drug|antibiotic|pill)",
             r"what (medicine|tablet|drug|antibiotic|pill)s? (should|can|do) i", r"\bdiagnose (me|my)", r"do i have (a |an )?\w+", r"am i (suffering|having|diabetic)",
             r"what (disease|illness|condition) do i have", r"what('s| is) wrong with me", r"\bwhich medicine\b"]
REFUSE_HI = ["कौन सी दवा", "कौन सी गोली", "कौन-सी दवा", "दवा बताओ", "दवा बताइए", "दवाई बताओ", "दवाई बताइए", "खुराक", "कितनी गोली", "कितने मिलीग्राम", "मुझे क्या बीमारी", "क्या मुझे बीमारी", "निदान करो", "निदान कीजिए"]
SOON_RX = re.compile(r"(\d+|few|several|many|two|three)\s*(days|weeks|months)|since (last )?(week|month)|getting worse|worsen|not (improv|getting better)|keeps? (coming|happening)|high fever|\b10[3-9]\b|"
                     r"दिन(ों)? से|हफ्त(े|ों) से|महीन(े|ों) से|बढ़ रहा|बढ़ती जा|ठीक नहीं हो रहा|बहुत तेज़ बुखार", re.I)
GREET_RX = re.compile(r"^\s*(hi|hello|hey|namaste|namaskar|good (morning|afternoon|evening)|नमस्ते|नमस्कार|हेलो|हाय)\b", re.I)
THANKS_RX = re.compile(r"\b(thanks|thank you|thx)\b|धन्यवाद|शुक्रिया", re.I)
BYE_RX = re.compile(r"^\s*(bye|goodbye|see you)\b|अलविदा", re.I)
DOCTOR_RX = re.compile(r"(which|what|find|need|recommend|suggest).{0,25}(doctor|specialist)|book.{0,20}(appointment|doctor)|see a doctor|डॉक्टर (कौन|चाहिए|दिखाऊँ|से मिलना)|अपॉइंटमेंट", re.I)
CTX_PATTERNS = [("report", r"(my|latest|last|recent).{0,15}(report|lab|blood test)|explain.{0,15}report|मेरी.{0,10}रिपोर्ट|रिपोर्ट.{0,10}(समझा|बताओ|बताइए)"),
                ("bmi", r"\bmy bmi\b|मेरा (बीएमआई|bmi)"),
                ("vitals", r"(what('s| is| are)|show|tell me|check|see|view|summari[sz]e)( me)?( about)? my (latest |current |last |recent )?(blood pressure|bp|sugar|weight|heart rate|vitals|health data)|मेरा (बीपी|वजन|शुगर|रक्तचाप) (कितना|क्या)|मेरे (वाइटल|स्वास्थ्य डेटा)"),
                ("predictions", r"my (risk|prediction|predictions|risk score)|मेरा जोखिम|मेरी प्रेडिक्शन"),
                ("diet", r"my (diet|calorie|meal)|मेरा (डाइट|आहार|कैलोरी)")]
THRESHOLD, FOLLOW_THRESHOLD, STRONG = 0.15, 0.10, 0.35
STOP = set("""what how can the and for with about tell from that this have does should when which your you are why who any some please give more
kya hai hain kaise kaun mujhe mera meri ke ki ka ko se me mein hoti hota hoga karna karne""".split()) | set("मुझे क्या है हैं कैसे कौन और के की का को से में पर यह वह कर करें करना चाहिए होता होती".split())


def _tokens(text):
    return {w for w in re.findall(r"[\w\u0900-\u097F+]+", text.lower()) if w not in STOP and len(w) >= 2}


def _overlaps(query_tokens, doc_tokens):
    """True if a query word equals / shares a 4-letter stem with a word in the document (guards against n-gram noise)."""
    stems = {w[:4] for w in doc_tokens if len(w) >= 4}
    return any(w in doc_tokens or (len(w) >= 4 and w[:4] in stems) for w in query_tokens)


def detect_language(text, hint=None):
    return "hi" if DEVANAGARI.search(text or "") else (hint if hint in ("en", "hi") and not re.search(r"[A-Za-z]{3,}", text or "") else "en")


def emergency_check(text):
    t = (text or "").lower()
    if any(re.search(p, t) for p in SELFHARM_EN) or any(p in text for p in SELFHARM_HI):
        return "selfharm"
    if re.search(r"\b(no|not|without|don'?t have|never had)\b[^.?!]{0,20}(chest pain|breath)", t):
        return None
    if any(re.search(p, t) for p in EMERGENCY_EN) or any(p in text for p in EMERGENCY_HI):
        return "emergency"
    return None


def needs_refusal(text):
    t = (text or "").lower()
    return any(re.search(p, t) for p in REFUSE_EN) or any(p in text for p in REFUSE_HI)


def _param_docs():
    return {f"param:{k}": " ".join([p["name_en"], p["name_hi"], *p["aliases"], "test lab report meaning what is high low normal range", "जाँच रिपोर्ट मतलब क्या है"]) for k, p in PARAMS.items()}


class ChatEngine:
    def __init__(self):
        self.ids = [t["id"] for t in TOPICS]
        docs = [" ".join([t["q_en"], t["q_hi"], t["kw"]]).lower() for t in TOPICS]
        pdocs = _param_docs()
        self.ids += list(pdocs)
        docs += [d.lower() for d in pdocs.values()]
        self.tokens = [_tokens(d) for d in docs]
        import numpy as np
        self.weights = np.array([0.85 if i.startswith("param:") else 1.0 for i in self.ids])
        self.alias_rx = [(re.compile(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])"), k) for a, k in
                         sorted(((a, k) for k, p in PARAMS.items() for a in p["aliases"] if len(a) >= 3 or a in ("hb", "tg")), key=lambda t: -len(t[0]))]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(docs)

    def retrieve(self, text, extra=""):
        q = self.vec.transform([(extra + " " + text).lower()])
        sims = cosine_similarity(q, self.matrix)[0] * self.weights   # lab-glossary docs slightly down-weighted vs topics
        qt = _tokens(extra + " " + text)
        order = sims.argsort()[::-1][:3]
        for i in order:   # best-scoring document that also shares at least one real word with the query
            if _overlaps(_tokens(text), self.tokens[i]) or (extra and _overlaps(qt, self.tokens[i])):
                return self.ids[i], float(sims[i])
        return self.ids[int(order[0])], 0.0

    def param_alias(self, text):
        t = text.lower()
        return next((k for rx, k in self.alias_rx if rx.search(t)), None)

    def respond(self, message, lang_hint="en", state=None, consent=False):
        state = state or {}
        message = (message or "").strip()[:600]
        lang = detect_language(message, lang_hint)
        ui = UI[lang]
        base = dict(topic=None, level="info", emergency=False, specialty=None, lang=lang, followups=[], context=None, see_doctor=False)
        em = emergency_check(message)
        if em:
            return {**base, "reply": ui["selfharm" if em == "selfharm" else "emergency"], "level": "emergency", "emergency": True, "see_doctor": True,
                    "topic": "emergency", "followups": []}
        if needs_refusal(message):
            return {**base, "reply": ui["refuse"], "topic": "refusal", "see_doctor": True, "specialty": "General Physician", "followups": self._followups(STARTERS, lang)}
        if GREET_RX.search(message) and len(message.split()) <= 4:
            return {**base, "reply": ui["greet"], "topic": "greeting", "followups": self._followups(STARTERS, lang)}
        if THANKS_RX.search(message) and len(message.split()) <= 5:
            return {**base, "reply": ui["thanks"], "topic": "thanks"}
        if BYE_RX.search(message):
            return {**base, "reply": ui["bye"], "topic": "bye"}
        for name, rx in CTX_PATTERNS:
            if re.search(rx, message, re.I):
                if not consent:
                    return {**base, "reply": ui["need_consent"], "topic": "consent_needed", "context": "consent_needed"}
                return {**base, "reply": "", "topic": f"ctx:{name}", "context": name}
        if DOCTOR_RX.search(message):
            tid, _ = self.retrieve(message, state.get("last_topic", "") if state.get("last_topic") in TOPIC_BY_ID else "")
            spec = TOPIC_BY_ID[tid]["specialty"] if tid in TOPIC_BY_ID else (TOPIC_BY_ID.get(state.get("last_topic"), {}).get("specialty") or "General Physician")
            return {**base, "reply": ui["doctor_hint"], "topic": "doctor", "see_doctor": True, "specialty": spec, "followups": self._followups(STARTERS, lang)}
        extra = ""
        tid, score = self.retrieve(message)
        last = state.get("last_topic")
        if score < THRESHOLD and last in TOPIC_BY_ID and len(message.split()) <= 8:   # context-aware follow-up
            extra = TOPIC_BY_ID[last]["kw"]
            tid2, score2 = self.retrieve(message, extra)
            if score2 >= FOLLOW_THRESHOLD:
                tid, score = tid2, score2
        alias = self.param_alias(message)
        if alias and score < STRONG:   # e.g. "what does SGPT mean"
            tid, score = f"param:{alias}", max(score, THRESHOLD)
        if score < (FOLLOW_THRESHOLD if extra else THRESHOLD):
            return {**base, "reply": ui["fallback"], "topic": "fallback", "followups": self._followups(STARTERS, lang)}
        soon = bool(SOON_RX.search(message))
        if tid.startswith("param:"):
            key = tid.split(":", 1)[1]
            p = PARAMS[key]
            text = f"{ui['param_intro']} {p['name_hi'] if lang == 'hi' else p['name_en']}. {explanation(key, '', lang)} " \
                   f"{p['high_' + lang]} {p['low_' + lang]}"
            return {**base, "reply": f"{text}\n\n{ui['disclaimer']}", "topic": tid, "specialty": "General Physician", "context": None,
                    "followups": self._followups(["diet", "exercise"], lang), "see_doctor": soon}
        t = TOPIC_BY_ID[tid]
        reply = t[lang]
        if soon:
            reply += "\n\n" + ui["soon"]
        return {**base, "reply": f"{reply}\n\n{ui['disclaimer']}", "topic": tid, "specialty": t["specialty"], "see_doctor": soon,
                "level": "see_doctor" if soon else "info", "followups": self._followups(t["fu"], lang)}

    @staticmethod
    def _followups(ids, lang):
        return [TOPIC_BY_ID[i]["q_hi" if lang == "hi" else "q_en"] for i in ids]


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = ChatEngine()
    return _engine

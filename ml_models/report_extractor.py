"""Medical report text extraction (PDF text layer / OCR) and lab-value parsing.

Design rules (per project requirements):
* Values, units and reference ranges are taken ONLY from the report text. If a report prints no range for a test, the
  status is "Not assessed" - the built-in dictionary below is used for name recognition and explanations, never to
  fabricate ranges.
* `explanation` text is general educational content written by the developers (AI-generated style), clearly separate
  from the extracted data.
"""
import io, re
from datetime import date

NOT_ASSESSED = "Not assessed"


class OCRUnavailable(RuntimeError):
    pass


def P(name_en, name_hi, aliases, what_en, what_hi, high_en, high_hi, low_en, low_hi):
    return dict(name_en=name_en, name_hi=name_hi, aliases=aliases, what_en=what_en, what_hi=what_hi,
                high_en=high_en, high_hi=high_hi, low_en=low_en, low_hi=low_hi)


PARAMS = {
 "hemoglobin": P("Hemoglobin", "हीमोग्लोबिन", ["hemoglobin", "haemoglobin", "hb", "hgb"],
   "Protein in red blood cells that carries oxygen.", "लाल रक्त कोशिकाओं का प्रोटीन जो ऑक्सीजन ले जाता है।",
   "Can be seen with dehydration, smoking or living at high altitude; a doctor can tell the cause.", "पानी की कमी, धूम्रपान या ऊँचाई पर रहने से बढ़ सकता है; कारण डॉक्टर बता सकते हैं।",
   "May point to anaemia (low blood count); a doctor can look for the cause.", "यह एनीमिया (खून की कमी) का संकेत हो सकता है; कारण डॉक्टर से पूछें।"),
 "rbc": P("RBC count", "आरबीसी (लाल रक्त कोशिकाएँ)", ["rbc count", "rbc", "red blood cell count", "red blood cells", "total rbc"],
   "Number of red blood cells.", "लाल रक्त कोशिकाओं की संख्या।", "Can occur with dehydration or low oxygen levels.", "पानी की कमी या ऑक्सीजन की कमी में बढ़ सकती है।",
   "Can occur with anaemia or blood loss.", "एनीमिया या खून की कमी में घट सकती है।"),
 "wbc": P("WBC count", "डब्ल्यूबीसी (श्वेत रक्त कोशिकाएँ)", ["total leucocyte count", "total leukocyte count", "tlc", "wbc count", "wbc", "white blood cell count", "white blood cells", "total wbc"],
   "White cells that fight infection.", "श्वेत कोशिकाएँ जो संक्रमण से लड़ती हैं।", "Often rises with infection or inflammation.", "संक्रमण या सूजन में अक्सर बढ़ जाती हैं।",
   "May occur with some infections, medicines or bone-marrow conditions.", "कुछ संक्रमण, दवाओं या अस्थि-मज्जा की स्थितियों में घट सकती हैं।"),
 "platelets": P("Platelet count", "प्लेटलेट काउंट", ["platelet count", "platelets", "plt"],
   "Cell fragments that help blood to clot.", "रक्त का थक्का बनाने में मदद करने वाले कण।", "Can rise with inflammation or iron deficiency.", "सूजन या आयरन की कमी में बढ़ सकते हैं।",
   "Can occur with viral infections such as dengue; low counts need medical follow-up.", "डेंगू जैसे वायरल संक्रमण में घट सकते हैं; कम होने पर डॉक्टर से मिलें।"),
 "hematocrit": P("Hematocrit (PCV)", "हेमाटोक्रिट (पीसीवी)", ["hematocrit", "haematocrit", "pcv", "packed cell volume", "hct"],
   "Share of blood volume made up of red cells.", "रक्त में लाल कोशिकाओं का प्रतिशत।", "Often goes with dehydration or high red cell count.", "पानी की कमी या अधिक लाल कोशिकाओं के साथ बढ़ता है।",
   "Often goes with anaemia.", "अक्सर एनीमिया के साथ घटता है।"),
 "mcv": P("MCV", "एमसीवी", ["mean corpuscular volume", "mcv"], "Average size of red blood cells.", "लाल रक्त कोशिकाओं का औसत आकार।",
   "Large cells can occur with vitamin B12/folate deficiency.", "बड़ी कोशिकाएँ विटामिन B12/फोलेट की कमी में हो सकती हैं।", "Small cells can occur with iron deficiency.", "छोटी कोशिकाएँ आयरन की कमी में हो सकती हैं।"),
 "hba1c": P("HbA1c", "एचबीए1सी", ["glycosylated hemoglobin", "glycated hemoglobin", "glycosylated haemoglobin", "hemoglobin a1c", "haemoglobin a1c", "hba1c", "hb a1c", "a1c"],
   "Average blood sugar over about 3 months.", "लगभग 3 महीनों का औसत ब्लड शुगर।", "Higher values mean sugar has been running high; discuss with a doctor.", "अधिक मान का मतलब शुगर लंबे समय से अधिक है; डॉक्टर से चर्चा करें।",
   "Rarely a concern on its own.", "अकेले यह आमतौर पर चिंता का विषय नहीं होता।"),
 "glucose_fasting": P("Fasting glucose", "फास्टिंग ग्लूकोज़", ["fasting blood sugar", "fasting blood glucose", "fasting plasma glucose", "fasting glucose", "glucose fasting", "glucose (fasting)", "blood sugar fasting", "blood sugar (fasting)", "fbs", "fbg"],
   "Blood sugar after at least 8 hours without food.", "कम से कम 8 घंटे भूखे रहने के बाद का ब्लड शुगर।", "Above the reference range may suggest the body is not handling sugar well; a doctor decides on further tests.", "संदर्भ सीमा से ऊपर होने पर शरीर शुगर को ठीक से नहीं संभाल पा रहा हो सकता है; आगे की जाँच डॉक्टर तय करेंगे।",
   "Low sugar can cause sweating, shakiness and dizziness.", "कम शुगर से पसीना, कंपकंपी और चक्कर आ सकते हैं।"),
 "glucose_pp": P("Post-prandial glucose", "खाने के बाद ग्लूकोज़", ["post prandial blood sugar", "postprandial blood sugar", "post prandial glucose", "postprandial glucose", "post-prandial glucose", "ppbs", "pp blood sugar", "blood sugar pp", "glucose pp"],
   "Blood sugar about 2 hours after a meal.", "भोजन के लगभग 2 घंटे बाद का ब्लड शुगर।", "High values after meals may suggest poor sugar control.", "भोजन के बाद अधिक मान शुगर नियंत्रण में कमी दर्शा सकते हैं।", "Low values can cause weakness.", "कम मान से कमज़ोरी हो सकती है।"),
 "glucose_random": P("Random glucose", "रैंडम ग्लूकोज़", ["random blood sugar", "random blood glucose", "random glucose", "rbs"],
   "Blood sugar at any time of day.", "दिन के किसी भी समय का ब्लड शुगर।", "High values need a doctor's review.", "अधिक मान की डॉक्टर से जाँच करवाएँ।", "Low values can cause weakness and dizziness.", "कम मान से कमज़ोरी और चक्कर आ सकते हैं।"),
 "glucose": P("Blood glucose", "ब्लड ग्लूकोज़", ["blood glucose", "blood sugar", "glucose"],
   "Sugar level in the blood (the report does not say whether fasting).", "रक्त में शुगर का स्तर (रिपोर्ट में फास्टिंग का उल्लेख नहीं)।", "High values need a doctor's review.", "अधिक मान की डॉक्टर से जाँच करवाएँ।", "Low values can cause weakness and dizziness.", "कम मान से कमज़ोरी और चक्कर आ सकते हैं।"),
 "total_cholesterol": P("Total cholesterol", "कुल कोलेस्ट्रॉल", ["total cholesterol", "serum cholesterol", "cholesterol total", "cholesterol"],
   "All cholesterol carried in the blood.", "रक्त में मौजूद कुल कोलेस्ट्रॉल।", "High levels raise long-term heart and blood-vessel risk; diet, activity and sometimes medicines help.", "अधिक स्तर से हृदय और रक्त-नलिकाओं का जोखिम बढ़ता है; आहार, व्यायाम और कभी-कभी दवाएँ मदद करती हैं।",
   "Rarely a concern on its own.", "अकेले यह आमतौर पर चिंता का विषय नहीं होता।"),
 "ldl": P("LDL cholesterol", "एलडीएल कोलेस्ट्रॉल", ["ldl cholesterol", "ldl-c", "ldl c", "ldl"], "'Bad' cholesterol that can build up in arteries.", "'खराब' कोलेस्ट्रॉल जो धमनियों में जम सकता है।",
   "High values raise heart-disease risk.", "अधिक मान से हृदय रोग का जोखिम बढ़ता है।", "Low values are usually not a concern.", "कम मान आमतौर पर चिंताजनक नहीं होते।"),
 "hdl": P("HDL cholesterol", "एचडीएल कोलेस्ट्रॉल", ["hdl cholesterol", "hdl-c", "hdl c", "hdl"], "'Good' cholesterol that helps clear other cholesterol.", "'अच्छा' कोलेस्ट्रॉल जो अन्य कोलेस्ट्रॉल हटाने में मदद करता है।",
   "High values are usually protective.", "अधिक मान आमतौर पर सुरक्षात्मक होते हैं।", "Low values are linked with higher heart risk; exercise and not smoking help.", "कम मान हृदय जोखिम से जुड़ा है; व्यायाम और धूम्रपान छोड़ना मदद करता है।"),
 "triglycerides": P("Triglycerides", "ट्राइग्लिसराइड्स", ["triglycerides", "triglyceride", "tg"], "A type of fat in the blood.", "रक्त में मौजूद एक प्रकार की वसा।",
   "High values are linked with sugary/fatty diets, alcohol, obesity and diabetes.", "अधिक मान मीठे/चिकने भोजन, शराब, मोटापे और मधुमेह से जुड़े हैं।", "Usually not a concern.", "आमतौर पर चिंताजनक नहीं।"),
 "creatinine": P("Creatinine", "क्रिएटिनिन", ["serum creatinine", "creatinine"], "Waste product filtered by the kidneys.", "गुर्दों द्वारा छाना जाने वाला अपशिष्ट पदार्थ।",
   "High values can mean the kidneys are filtering less well; needs a doctor's review.", "अधिक मान का अर्थ गुर्दे कम अच्छी तरह छान रहे हो सकते हैं; डॉक्टर से जाँच करवाएँ।", "Low values can occur with low muscle mass.", "कम मांसपेशी द्रव्यमान में कम हो सकता है।"),
 "urea": P("Blood urea / BUN", "ब्लड यूरिया / बीयूएन", ["blood urea nitrogen", "blood urea", "serum urea", "urea", "bun"], "Waste product from protein breakdown, cleared by the kidneys.", "प्रोटीन टूटने से बना अपशिष्ट जिसे गुर्दे साफ़ करते हैं।",
   "Can rise with dehydration, high-protein diets or reduced kidney function.", "पानी की कमी, अधिक प्रोटीन या गुर्दे की कमज़ोरी में बढ़ सकता है।", "Can occur with low protein intake or liver disease.", "कम प्रोटीन सेवन या लिवर रोग में कम हो सकता है।"),
 "uric_acid": P("Uric acid", "यूरिक एसिड", ["serum uric acid", "uric acid"], "By-product of purine breakdown.", "प्यूरिन टूटने से बना पदार्थ।",
   "High levels are linked with gout and kidney stones.", "अधिक स्तर गाउट और गुर्दे की पथरी से जुड़ा है।", "Usually not a concern.", "आमतौर पर चिंताजनक नहीं।"),
 "bilirubin_total": P("Total bilirubin", "कुल बिलीरुबिन", ["total bilirubin", "bilirubin total", "serum bilirubin", "bilirubin"], "Yellow pigment made when red cells break down; processed by the liver.", "लाल कोशिकाओं के टूटने से बना पीला रंजक; लिवर इसे संसाधित करता है।",
   "High values can cause jaundice and may relate to liver, bile duct or red-cell conditions.", "अधिक मान से पीलिया हो सकता है; लिवर, पित्त नलिका या रक्त संबंधी कारण हो सकते हैं।", "Usually not a concern.", "आमतौर पर चिंताजनक नहीं।"),
 "alt": P("ALT (SGPT)", "एएलटी (एसजीपीटी)", ["alanine aminotransferase", "alanine transaminase", "sgpt", "alt"], "Liver enzyme.", "लिवर का एंज़ाइम।",
   "High values can follow fatty liver, infections, alcohol or some medicines.", "फैटी लिवर, संक्रमण, शराब या कुछ दवाओं के बाद बढ़ सकता है।", "Usually not a concern.", "आमतौर पर चिंताजनक नहीं।"),
 "ast": P("AST (SGOT)", "एएसटी (एसजीओटी)", ["aspartate aminotransferase", "aspartate transaminase", "sgot", "ast"], "Enzyme found in liver and muscle.", "लिवर और मांसपेशियों में पाया जाने वाला एंज़ाइम।",
   "High values can follow liver or muscle injury.", "लिवर या मांसपेशी की चोट के बाद बढ़ सकता है।", "Usually not a concern.", "आमतौर पर चिंताजनक नहीं।"),
 "alp": P("Alkaline phosphatase", "अल्कलाइन फॉस्फेटेज़", ["alkaline phosphatase", "alk phos", "alp"], "Enzyme from liver, bile ducts and bone.", "लिवर, पित्त नलिकाओं और हड्डियों का एंज़ाइम।",
   "High values can relate to bile-duct or bone conditions.", "अधिक मान पित्त नलिका या हड्डी की स्थितियों से जुड़ा हो सकता है।", "Usually not a concern.", "आमतौर पर चिंताजनक नहीं।"),
 "albumin": P("Albumin", "एल्ब्यूमिन", ["serum albumin", "albumin"], "Main protein in blood, made by the liver.", "रक्त का मुख्य प्रोटीन, जो लिवर बनाता है।",
   "High values often reflect dehydration.", "अक्सर पानी की कमी दर्शाता है।", "Low values can occur with liver or kidney disease or poor nutrition.", "लिवर/गुर्दे की बीमारी या कुपोषण में कम हो सकता है।"),
 "tsh": P("TSH", "टीएसएच", ["thyroid stimulating hormone", "tsh"], "Pituitary hormone that controls the thyroid gland.", "पिट्यूटरी हार्मोन जो थायरॉइड ग्रंथि को नियंत्रित करता है।",
   "High TSH can suggest an underactive thyroid.", "अधिक टीएसएच थायरॉइड के कम सक्रिय होने का संकेत हो सकता है।", "Low TSH can suggest an overactive thyroid.", "कम टीएसएच थायरॉइड के अधिक सक्रिय होने का संकेत हो सकता है।"),
 "sodium": P("Sodium", "सोडियम", ["serum sodium", "sodium", "na+"], "Salt balance mineral.", "नमक संतुलन बनाने वाला खनिज।", "Can occur with dehydration.", "पानी की कमी में बढ़ सकता है।", "Can occur with excess water, some medicines or illness; significant lows need urgent review.", "अधिक पानी, कुछ दवाओं या बीमारी में कम हो सकता है; बहुत कम होने पर तुरंत जाँच करवाएँ।"),
 "potassium": P("Potassium", "पोटैशियम", ["serum potassium", "potassium", "k+"], "Mineral vital for heart and muscle rhythm.", "हृदय और मांसपेशियों की लय के लिए ज़रूरी खनिज।",
   "High potassium can affect heart rhythm and needs prompt medical review.", "अधिक पोटैशियम हृदय की लय को प्रभावित कर सकता है; शीघ्र डॉक्टर से मिलें।", "Low potassium can cause weakness and cramps.", "कम पोटैशियम से कमज़ोरी और ऐंठन हो सकती है।"),
 "vitamin_d": P("Vitamin D", "विटामिन डी", ["25-oh vitamin d", "25 hydroxy vitamin d", "vitamin d3", "vitamin d", "25(oh)d"], "Vitamin needed for bones and immunity.", "हड्डियों और प्रतिरक्षा के लिए ज़रूरी विटामिन।",
   "Very high values are usually from excess supplements.", "बहुत अधिक मान आमतौर पर अधिक सप्लीमेंट से होता है।", "Low levels are common; sunlight, diet and supplements prescribed by a doctor help.", "कमी आम है; धूप, आहार और डॉक्टर द्वारा बताए सप्लीमेंट मदद करते हैं।"),
 "vitamin_b12": P("Vitamin B12", "विटामिन बी12", ["vitamin b12", "vit b12", "cyanocobalamin", "b12"], "Vitamin needed for nerves and blood cells.", "नसों और रक्त कोशिकाओं के लिए ज़रूरी विटामिन।",
   "High values are usually from supplements.", "अधिक मान आमतौर पर सप्लीमेंट से होता है।", "Low levels are common in vegetarians and can cause tiredness and tingling.", "शाकाहारियों में कमी आम है; थकान और झनझनाहट हो सकती है।"),
}

# alias -> key, longest alias first so "ldl cholesterol" wins over "cholesterol"
_ALIASES = sorted(((a, k) for k, v in PARAMS.items() for a in v["aliases"]), key=lambda t: -len(t[0]))
_UNITS = sorted(["g/dl", "g/l", "mg/dl", "mg/l", "mmol/l", "u/l", "iu/l", "miu/l", "µiu/ml", "uiu/ml", "ng/ml", "ng/dl", "pg/ml", "µg/dl", "ug/dl",
                 "meq/l", "mm/hr", "mm/h", "fl", "pg", "%", "/cumm", "/mm3", "/µl", "/ul", "lakhs/cumm", "lakh/cumm", "million/cumm", "mill/cumm",
                 "millions/µl", "million/µl", "thou/mm3", "10^3/µl", "10^6/µl", "x10^3/µl", "x10^6/µl", "10^3/ul", "10^6/ul", "x10^3/ul", "x10^6/ul",
                 "10*3/µl", "10*6/µl", "cells/cumm", "cells/µl"], key=lambda s: -len(s))
_UNIT_RX = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(u) for u in _UNITS) + r")(?![a-z])", re.I)
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_VALUE_RX = re.compile(rf"(?<![A-Za-z0-9.^])({_NUM})")
_RANGE_RX = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", re.I)
_LT_RX = re.compile(r"(?:<\s*(=)?|≤|less than|upto|up to)\s*(\d+(?:\.\d+)?)", re.I)
_GT_RX = re.compile(r"(?:>\s*(=)?|≥|greater than|more than)\s*(\d+(?:\.\d+)?)", re.I)
_FLAG_RX = re.compile(r"(?:(?<=\s)|^)(H|L|HIGH|LOW)(?=\s|$|\*)")


def configure_tesseract(cmd):
    if cmd:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = cmd


def _ocr_image(img):
    try:
        import pytesseract
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img).convert("L")
        if min(img.size) < 1200:
            img = img.resize((img.width * 2, img.height * 2))
        return pytesseract.image_to_string(ImageOps.autocontrast(img), config="--psm 6")
    except ImportError as e:
        raise OCRUnavailable("OCR libraries are missing. Run: pip install -r requirements.txt") from e
    except Exception as e:
        if e.__class__.__name__ == "TesseractNotFoundError":
            raise OCRUnavailable("Tesseract OCR is not installed or not on PATH. Install it and set TESSERACT_CMD in .env.") from e
        raise


def extract_text(data: bytes, kind: str):
    """Returns (text, method). kind in {'pdf','png','jpg'}."""
    from PIL import Image
    if kind in ("png", "jpg"):
        return _ocr_image(Image.open(io.BytesIO(data))), "ocr"
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
    if len(text.strip()) >= 40:
        return text, "pdf-text"
    parts = []  # scanned PDF: OCR the page images embedded in the file
    for pg in reader.pages:
        for im in pg.images:
            try:
                parts.append(_ocr_image(Image.open(io.BytesIO(im.data))))
            except OCRUnavailable:
                raise
            except Exception:
                continue
    return "\n".join(parts), "pdf-image-ocr"


def _clean(text):
    return (text.replace("–", "-").replace("—", "-").replace("−", "-").replace("|", " ").replace("μ", "µ")
            .replace("\u00a0", " "))


def _status(value, lo, hi, lo_strict, hi_strict, flag):
    if lo is not None or hi is not None:
        if lo is not None and (value < lo or (lo_strict and value == lo)):
            return "Low", "report-range"
        if hi is not None and (value > hi or (hi_strict and value == hi)):
            return "High", "report-range"
        return "Normal", "report-range"
    if flag:
        return ("High" if flag.upper() in ("H", "HIGH") else "Low"), "report-flag"
    return NOT_ASSESSED, "none"


def parse_values(text):
    """Parse 'Name  value  unit  ref-range' lines. Only values literally present in the text are returned."""
    out, seen = [], set()
    for line in _clean(text).splitlines():
        s = line.strip()
        if len(s) < 4:
            continue
        low = re.sub(r"^[\W\d]{0,4}(?=[A-Za-z])", "", s.lower())  # drop numbering / bullets
        lead = len(s) - len(re.sub(r"^[\W\d]{0,4}(?=[A-Za-z])", "", s))
        hit = next(((a, k) for a, k in _ALIASES if low.startswith(a) and not low[len(a):len(a) + 1].isalnum()), None)
        if not hit or hit[1] in seen:
            continue
        alias, key = hit
        rest = s[lead + len(alias):]
        rest = re.sub(r"^\s*\([^)\d]*\)", "", rest)             # "(Hb)" style abbreviation
        rest = re.sub(r"^[\s:.\-]+", " ", rest)
        m = _VALUE_RX.search(rest)
        if not m:
            continue
        try:
            value = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        tail = rest[m.end():]
        um = _UNIT_RX.search(tail)
        unit = um.group(1) if um else None
        lo = hi = None; lo_strict = hi_strict = False; ref_text = None
        rm = _RANGE_RX.search(tail)
        if rm and float(rm.group(1)) < float(rm.group(2)):
            lo, hi, ref_text = float(rm.group(1)), float(rm.group(2)), rm.group(0).strip()
        else:
            lt, gt = _LT_RX.search(tail), _GT_RX.search(tail)
            if lt:
                hi, hi_strict, ref_text = float(lt.group(2)), not lt.group(1), lt.group(0).strip()
            elif gt:
                lo, lo_strict, ref_text = float(gt.group(2)), not gt.group(1), gt.group(0).strip()
        residue = tail
        for rx in (_UNIT_RX, _RANGE_RX, _LT_RX, _GT_RX):
            residue = rx.sub(" ", residue)
        fm = _FLAG_RX.search(residue.strip())
        status, src = _status(value, lo, hi, lo_strict, hi_strict, fm.group(1) if fm else None)
        out.append(dict(key=key, name=s[lead:lead + len(alias)].strip(), value=value, unit=unit, ref_low=lo, ref_high=hi,
                        ref_text=ref_text, status=status, status_source=src, raw_line=s[:250]))
        seen.add(key)
    return out


_DATE_RX = re.compile(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})|(\d{4})-(\d{2})-(\d{2})")


def guess_report_date(text):
    """Best-effort date from lines mentioning collection/report date. Returns date or None."""
    for line in text.splitlines():
        if re.search(r"collect|report(ed)?\s*(date|on)|date|sample", line, re.I):
            m = _DATE_RX.search(line)
            if m:
                try:
                    if m.group(1):
                        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                    return date(int(m.group(4)), int(m.group(5)), int(m.group(6)))
                except ValueError:
                    continue
    return None


def explanation(key, status, lang="en"):
    """General educational text (not a statement about the patient's diagnosis)."""
    p = PARAMS.get(key)
    if not p:
        return ""
    lang = "hi" if lang == "hi" else "en"
    base = p[f"what_{lang}"]
    if status == "High":
        return f"{base} {p[f'high_{lang}']}"
    if status == "Low":
        return f"{base} {p[f'low_{lang}']}"
    return base

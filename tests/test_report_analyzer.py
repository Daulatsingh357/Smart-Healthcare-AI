import io, os
import pytest
from conftest import lab_pdf, LAB_LINES
from ml_models import report_extractor as RX


def upload(c, data, name="lab.pdf", **form):
    return c.post("/api/reports", data=dict(file=(io.BytesIO(data), name), title="CBC", **form), content_type="multipart/form-data")


def test_parser_never_invents_ranges():
    vals = {v["key"]: v for v in RX.parse_values("\n".join(LAB_LINES))}
    assert vals["hemoglobin"]["status"] == "Low" and vals["hemoglobin"]["ref_low"] == 13.0
    assert vals["wbc"]["status"] == "Normal"
    assert vals["ldl"]["status"] == "High" and vals["ldl"]["ref_high"] == 100
    assert vals["creatinine"]["status"] == RX.NOT_ASSESSED and vals["creatinine"]["ref_low"] is None   # no range printed -> not assessed
    assert vals["alt"]["status"] == "High"


def test_parser_ignores_text_without_numbers_and_non_labs():
    assert RX.parse_values("Patient is asthmatic\nAltitude 5000 m\nHemoglobin") == []


def test_upload_pdf_extracts_stores_encrypted(pc, app):
    r = upload(pc, lab_pdf(LAB_LINES))
    assert r.status_code == 201, r.get_json()
    j = r.get_json()
    assert j["value_count"] >= 5 and j["abnormal_count"] == 3 and j["report_date"] == "2026-03-12"
    v = {x["key"]: x for x in j["values"]}
    assert v["hemoglobin"]["explanation_en"] and v["hemoglobin"]["explanation_hi"]
    assert "AI-generated" in j["explanation_label"]
    files = os.listdir(app.config["UPLOAD_FOLDER"])
    assert len(files) == 1 and files[0].endswith(".enc")
    raw = open(os.path.join(app.config["UPLOAD_FOLDER"], files[0]), "rb").read()
    assert b"%PDF" not in raw[:20] and b"Hemoglobin" not in raw      # encrypted at rest
    dl = pc.get(f"/api/reports/{j['id']}/file")
    assert dl.status_code == 200 and dl.data.startswith(b"%PDF")      # decrypts on download


def test_upload_rejects_fake_and_wrong_types(pc):
    assert upload(pc, b"MZ not a pdf", "evil.pdf").status_code == 415
    assert upload(pc, lab_pdf(LAB_LINES), "report.txt").status_code == 415
    assert upload(pc, lab_pdf(LAB_LINES), "report.png").status_code == 415   # PDF bytes with PNG extension
    assert pc.post("/api/reports", data={}, content_type="multipart/form-data").status_code == 400


def test_duplicate_and_future_date_rejected(pc):
    data = lab_pdf(LAB_LINES)
    assert upload(pc, data).status_code == 201
    assert upload(pc, data).status_code == 409
    assert upload(pc, lab_pdf(LAB_LINES + ["x"]), report_date="2999-01-01").status_code == 400


def test_history_comparison_and_summary(pc):
    a = lab_pdf(LAB_LINES)
    b = lab_pdf([ln.replace("11.2", "12.4") for ln in LAB_LINES] + ["Albumin   4.1   g/dL   3.5 - 5.2"])
    upload(pc, a, report_date="2026-01-10"); upload(pc, b, report_date="2026-03-10")
    s = pc.get("/api/reports/summary").get_json()
    assert s["total"] == 2 and s["latest"]["report_date"] == "2026-03-10"
    c = pc.get("/api/reports/compare?param=hemoglobin").get_json()
    assert [p["value"] for p in c["points"]] == [11.2, 12.4]
    lat = pc.get("/api/reports/compare/latest").get_json()
    row = next(r for r in lat["rows"] if r["key"] == "hemoglobin")
    assert row["change"] == pytest.approx(1.2)
    assert pc.get("/api/reports/compare?param=nonsense").status_code == 400
    assert any(p["key"] == "hemoglobin" for p in pc.get("/api/reports/parameters").get_json())


def test_report_pdf_and_isolation_between_patients(pc, pc2):
    rid = upload(pc, lab_pdf(LAB_LINES)).get_json()["id"]
    assert pc.get(f"/api/reports/{rid}/pdf").data.startswith(b"%PDF")
    for url in (f"/api/reports/{rid}", f"/api/reports/{rid}/file", f"/api/reports/{rid}/pdf", f"/reports/{rid}"):
        assert pc2.get(url).status_code == 403, url
    assert pc2.delete(f"/api/reports/{rid}").status_code == 403
    assert pc2.get("/api/reports").get_json() == []


def test_delete_removes_file_and_rows(pc, app):
    rid = upload(pc, lab_pdf(LAB_LINES)).get_json()["id"]
    assert pc.delete(f"/api/reports/{rid}").status_code == 200
    assert os.listdir(app.config["UPLOAD_FOLDER"]) == []
    from models import ExtractedReportValue
    with app.app_context():
        assert ExtractedReportValue.query.count() == 0


@pytest.mark.skipif(not os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") or not __import__("shutil").which("tesseract"), reason="needs tesseract + font")
def test_image_ocr_end_to_end(pc):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    img = Image.new("RGB", (1100, 420), "white"); d = ImageDraw.Draw(img)
    for i, ln in enumerate(["Hemoglobin   11.2   g/dL   13.0 - 17.0", "Fasting Blood Sugar   96   mg/dL   70 - 100", "SGPT   88   U/L   7 - 56"]):
        d.text((30, 30 + i * 60), ln, fill="black", font=font)
    buf = io.BytesIO(); img.save(buf, "PNG")
    r = upload(pc, buf.getvalue(), "scan.png")
    assert r.status_code == 201
    keys = {v["key"]: v for v in r.get_json()["values"]}
    assert keys["hemoglobin"]["value"] == 11.2 and keys["hemoglobin"]["status"] == "Low"
    assert r.get_json()["method"] == "ocr"

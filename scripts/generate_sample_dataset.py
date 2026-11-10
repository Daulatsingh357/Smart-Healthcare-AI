"""Writes SYNTHETIC patient datasets (500 rows each) in the exact train-ready format used by ml_models.
Copy a file to database/datasets/<disease>.csv to make train_all.py use it instead of generating data on the fly.
These are NOT real patients; labels come from the hand-written formulas in ml_models/*_model.py."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ml_models import MODULES  # noqa: E402

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "sample_data")
os.makedirs(out, exist_ok=True)
for i, (k, m) in enumerate(MODULES.items()):
    df = m.SPEC["generate"](500, np.random.default_rng(100 + i)).round(2)
    df.insert(0, "patient_id", [f"SYN-{k[:3].upper()}-{n:04d}" for n in range(1, len(df) + 1)])
    df.to_csv(os.path.join(out, f"synthetic_{k}.csv"), index=False)
    print(f"wrote synthetic_{k}.csv  rows={len(df)}  positive rate={df.target.mean():.2f}")

# A small fictional lab report so the Report Analyzer can be demonstrated without any real patient document.
from reportlab.pdfgen import canvas  # noqa: E402
lines = ["CITY DIAGNOSTICS (SAMPLE - FICTIONAL) - Blood Report", "Patient: Sample Person        Sample Collected Date: 12/10/2026", "Test Name   Result   Unit   Reference Range",
         "Hemoglobin   11.2   g/dL   13.0 - 17.0", "Total Leucocyte Count   7200   /cumm   4000 - 11000", "Platelet Count   250   10^3/uL   150 - 410",
         "Fasting Blood Sugar   112   mg/dL   70 - 100", "Total Cholesterol   226   mg/dL   < 200", "LDL Cholesterol   148   mg/dL   < 100", "HDL Cholesterol   38   mg/dL   > 40",
         "Serum Creatinine   1.0   mg/dL   0.7 - 1.3", "SGPT (ALT)   64   U/L   7 - 56", "TSH   2.4   uIU/mL", "Vitamin D   22   ng/mL   30 - 100"]
c = canvas.Canvas(os.path.join(out, "sample_lab_report.pdf")); y = 800
for ln in lines:
    c.drawString(40, y, ln); y -= 20
c.save(); print("wrote sample_lab_report.pdf")

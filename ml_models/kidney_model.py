"""Kidney disease risk model definition."""
import pandas as pd
from . import common as C

SPEC = dict(
    key="kidney", title="Kidney Disease Risk", dataset_file="kidney.csv",
    features=[C.num_feature("age", "Age", "years", 18, 100, 1), C.SEX, C.num_feature("bmi", "BMI", "kg/m2", 12, 70, 0.1),
              C.num_feature("blood_pressure", "Systolic blood pressure", "mmHg", 70, 250), C.num_feature("glucose", "Blood sugar (fasting)", "mg/dL", 40, 600),
              C.num_feature("creatinine", "Serum creatinine", "mg/dL", 0.2, 15, 0.01), C.num_feature("urea", "Blood urea", "mg/dL", 2, 300),
              C.num_feature("hemoglobin", "Hemoglobin", "g/dL", 3, 20, 0.1)],
    tips={"Moderate": ["Stay hydrated unless your doctor restricts fluids; avoid over-the-counter painkillers without advice."],
          "High": ["Ask your doctor about eGFR and a urine albumin test.", "Keep blood pressure and blood sugar under control."]},
)


def generate(n, rng):
    d = pd.DataFrame(dict(age=rng.integers(18, 85, n), sex=rng.integers(0, 2, n), bmi=rng.normal(27, 5, n).clip(15, 50),
                          blood_pressure=rng.normal(130, 20, n).clip(80, 210), glucose=rng.normal(110, 30, n).clip(60, 300),
                          creatinine=rng.lognormal(0.1, 0.4, n).clip(0.4, 9), urea=rng.normal(30, 14, n).clip(8, 150),
                          hemoglobin=rng.normal(13.5, 2, n).clip(6, 18)))
    z = (0.02 * (d.age - 50) + 0.01 * (d.bmi - 27) + 0.015 * (d.blood_pressure - 130) + 0.005 * (d.glucose - 110)
         + 2.2 * (d.creatinine - 1.1) + 0.03 * (d.urea - 30) - 0.35 * (d.hemoglobin - 13.5) - 1.2)
    d["target"] = (rng.random(n) < C.sigmoid(z)).astype(int)
    return d


SPEC["generate"] = generate


def train(**kw): return C.train_disease(SPEC, **kw)
def predict(values, algorithm=None, **kw): return C.predict(SPEC, values, algorithm, **kw)

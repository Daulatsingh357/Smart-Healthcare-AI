"""Diabetes risk model definition (features, validation ranges, synthetic generator)."""
import numpy as np, pandas as pd
from . import common as C

SPEC = dict(
    key="diabetes", title="Diabetes Risk", dataset_file="diabetes.csv",
    features=[C.num_feature("age", "Age", "years", 18, 100, 1), C.SEX, C.num_feature("bmi", "BMI", "kg/m2", 12, 70, 0.1),
              C.num_feature("glucose", "Blood sugar (fasting)", "mg/dL", 40, 600), C.num_feature("blood_pressure", "Systolic blood pressure", "mmHg", 70, 250),
              C.num_feature("cholesterol", "Total cholesterol", "mg/dL", 80, 500),
              C.bin_feature("family_history", "Family history of diabetes", [(0, "No"), (1, "Yes")])],
    tips={"Moderate": ["Limit sugary drinks and refined carbohydrates; prefer whole grains and pulses."],
          "High": ["Ask your doctor about HbA1c / fasting glucose testing.", "Monitor blood sugar if your doctor advises it."]},
)


def generate(n, rng):
    d = pd.DataFrame(dict(age=rng.integers(18, 85, n), sex=rng.integers(0, 2, n), bmi=rng.normal(27, 6, n).clip(15, 50),
                          glucose=rng.normal(110, 30, n).clip(60, 300), blood_pressure=rng.normal(125, 18, n).clip(80, 200),
                          cholesterol=rng.normal(200, 40, n).clip(110, 400), family_history=(rng.random(n) < 0.25).astype(int)))
    z = (0.03 * (d.age - 45) + 0.09 * (d.bmi - 27) + 0.045 * (d.glucose - 110) + 0.01 * (d.blood_pressure - 125)
         + 0.004 * (d.cholesterol - 200) + 0.6 * d.family_history - 1.3)
    d["target"] = (rng.random(n) < C.sigmoid(z)).astype(int)
    return d


SPEC["generate"] = generate


def train(**kw): return C.train_disease(SPEC, **kw)
def predict(values, algorithm=None, **kw): return C.predict(SPEC, values, algorithm, **kw)

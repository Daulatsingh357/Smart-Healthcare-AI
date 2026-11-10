"""Liver disease risk model definition."""
import pandas as pd
from . import common as C

SPEC = dict(
    key="liver", title="Liver Disease Risk", dataset_file="liver.csv",
    features=[C.num_feature("age", "Age", "years", 18, 100, 1), C.SEX, C.num_feature("bmi", "BMI", "kg/m2", 12, 70, 0.1),
              C.num_feature("bilirubin", "Total bilirubin", "mg/dL", 0.1, 30, 0.1), C.num_feature("alt", "ALT (SGPT)", "U/L", 1, 2000),
              C.num_feature("ast", "AST (SGOT)", "U/L", 1, 2000), C.num_feature("alp", "Alkaline phosphatase", "U/L", 20, 1500),
              C.num_feature("albumin", "Albumin", "g/dL", 1, 6.5, 0.1)],
    tips={"Moderate": ["Limit alcohol and fried/processed food; maintain a healthy weight."],
          "High": ["Avoid alcohol and unprescribed supplements until you speak to a doctor.", "Ask about a liver function test and ultrasound."]},
)


def generate(n, rng):
    d = pd.DataFrame(dict(age=rng.integers(18, 85, n), sex=rng.integers(0, 2, n), bmi=rng.normal(27, 5, n).clip(15, 50),
                          bilirubin=rng.lognormal(0, 0.6, n).clip(0.1, 15), alt=rng.lognormal(3.4, 0.5, n).clip(5, 600),
                          ast=rng.lognormal(3.3, 0.5, n).clip(5, 600), alp=rng.lognormal(4.8, 0.3, n).clip(30, 800),
                          albumin=rng.normal(4.0, 0.6, n).clip(1.5, 5.5)))
    z = (0.01 * (d.age - 45) + 0.25 * d.sex + 0.02 * (d.bmi - 27) + 0.5 * (d.bilirubin - 1) + 0.012 * (d.alt - 30)
         + 0.012 * (d.ast - 28) + 0.004 * (d.alp - 120) - 1.1 * (d.albumin - 4) - 1.0)
    d["target"] = (rng.random(n) < C.sigmoid(z)).astype(int)
    return d


SPEC["generate"] = generate


def train(**kw): return C.train_disease(SPEC, **kw)
def predict(values, algorithm=None, **kw): return C.predict(SPEC, values, algorithm, **kw)

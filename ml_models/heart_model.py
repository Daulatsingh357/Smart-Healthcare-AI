"""Heart disease risk model definition."""
import pandas as pd
from . import common as C

SPEC = dict(
    key="heart", title="Heart Disease Risk", dataset_file="heart.csv",
    features=[C.num_feature("age", "Age", "years", 18, 100, 1), C.SEX, C.num_feature("bmi", "BMI", "kg/m2", 12, 70, 0.1),
              C.num_feature("blood_pressure", "Systolic blood pressure", "mmHg", 70, 250), C.num_feature("cholesterol", "Total cholesterol", "mg/dL", 80, 500),
              C.num_feature("heart_rate", "Resting heart rate", "bpm", 30, 220), C.num_feature("glucose", "Blood sugar (fasting)", "mg/dL", 40, 600),
              C.bin_feature("smoker", "Current smoker", [(0, "No"), (1, "Yes")])],
    tips={"Moderate": ["Reduce salt and saturated fat; aim for about 150 minutes of brisk activity a week."],
          "High": ["Do not ignore chest pain, breathlessness or fainting; seek urgent care for those symptoms.",
                   "Ask your doctor about a lipid profile and ECG."]},
)


def generate(n, rng):
    d = pd.DataFrame(dict(age=rng.integers(25, 85, n), sex=rng.integers(0, 2, n), bmi=rng.normal(27, 5, n).clip(15, 50),
                          blood_pressure=rng.normal(130, 20, n).clip(80, 210), cholesterol=rng.normal(210, 45, n).clip(110, 400),
                          heart_rate=rng.normal(78, 14, n).clip(45, 190), glucose=rng.normal(105, 25, n).clip(60, 300),
                          smoker=rng.integers(0, 2, n)))
    z = (0.05 * (d.age - 55) + 0.4 * d.sex + 0.03 * (d.bmi - 27) + 0.02 * (d.blood_pressure - 130) + 0.01 * (d.cholesterol - 210)
         + 0.008 * (d.heart_rate - 78) + 0.01 * (d.glucose - 105) + 0.7 * d.smoker - 1.4)
    d["target"] = (rng.random(n) < C.sigmoid(z)).astype(int)
    return d


SPEC["generate"] = generate


def train(**kw): return C.train_disease(SPEC, **kw)
def predict(values, algorithm=None, **kw): return C.predict(SPEC, values, algorithm, **kw)

from . import diabetes_model, heart_model, kidney_model, liver_model

MODULES = {m.SPEC["key"]: m for m in (diabetes_model, heart_model, kidney_model, liver_model)}
SPECS = {k: m.SPEC for k, m in MODULES.items()}

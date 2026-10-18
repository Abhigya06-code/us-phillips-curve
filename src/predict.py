"""Loading artifacts and predicting. The app only talks to the models through this file."""
import json
import joblib
import pandas as pd
from .config import ARTIFACT_DIR, MODELS


def load_artifacts():
    models = joblib.load(ARTIFACT_DIR / "models.joblib")   # {"cpi_yoy:classic": fitted_model, ...}
    with open(ARTIFACT_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return models, metrics


def predict(models, target: str, model_key: str, inputs: dict) -> float:
    feats = MODELS[model_key][1]
    x = pd.DataFrame([[inputs[f] for f in feats]], columns=feats)
    return float(models[f"{target}:{model_key}"].predict(x)[0])


def out_of_range(metrics: dict, inputs: dict, features) -> list:
    """Features whose value lies outside the range seen in training (Random Forest cannot extrapolate)."""
    bad = []
    for f in features:
        lo, hi = metrics["train_ranges"][f]
        if not (lo <= inputs[f] <= hi):
            bad.append(f)
    return bad

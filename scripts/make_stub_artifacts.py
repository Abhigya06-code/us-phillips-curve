"""Creates SYNTHETIC data + models in the real schema so Person B can build the app before Person A finishes.
Run from repo root:  python scripts/make_stub_artifacts.py
Everything it writes is flagged "stub": true and the app shows a warning banner. Replace with the Colab outputs."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd, joblib, sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from src.config import *

rng = np.random.default_rng(0)
idx = pd.date_range("1960-01-01", "2026-08-01", freq="MS")
n = len(idx)

def ar1(phi, sigma, size):
    x = np.zeros(size)
    for i in range(1, size):
        x[i] = phi * x[i - 1] + rng.normal(0, sigma)
    return x

unrate = np.clip(5.8 + ar1(0.985, 0.25, n), 3.0, 14.5)
for a, b, h in [("2009-01", "2010-06", 4), ("2020-03", "2020-08", 9)]:
    m = (idx >= a) & (idx <= b); unrate[m] = np.clip(unrate[m] + h, 3, 14.8)
nrou = pd.Series(5.0 + 0.8 * np.sin(np.linspace(0, 4, n)), idx)
mich = np.clip(3.2 + ar1(0.99, 0.12, n) + 2.5 * np.exp(-((np.arange(n) - 220) / 60) ** 2), 1.0, 11)
fedfunds = np.clip(2.5 + 0.8 * mich - 0.3 * (unrate - 5) + ar1(0.97, 0.3, n), 0, 6)
wti = np.clip(55 + ar1(0.98, 4, n) + 20 * np.sin(np.arange(n) / 40), 15, 135)
m2_yoy = np.clip(6 + ar1(0.97, 0.7, n), -2, 25)
infl = 0.7 * mich - 0.15 * (unrate - nrou.values) + 0.02 * (wti - 60) + 0.1 * m2_yoy + ar1(0.9, 0.25, n)
core = 0.6 * infl + 0.4 * mich + rng.normal(0, 0.1, n)

df = pd.DataFrame({"cpi_yoy": infl, "core_yoy": core, "unrate": unrate, "nrou": nrou.values,
                   "mich": mich, "t5yie": np.where(idx >= "2003-01-01", mich - 0.8 + rng.normal(0, .2, n), np.nan),
                   "fedfunds": fedfunds, "wti": np.where(idx >= "1986-01-01", wti, np.nan), "m2_yoy": m2_yoy}, index=idx)
df.loc[df.index < "1978-01-01", "mich"] = np.nan
df.loc[df.index < "1978-01-01", ["fedfunds", "m2_yoy"]] = df.loc[df.index < "1978-01-01", ["fedfunds", "m2_yoy"]]  # keep
df.index.name = "date"
df[DATA_COLUMNS].round(3).to_csv(DATA_PATH)

d = df.loc["2003-01":].dropna(subset=ALL_FEATURES + ["cpi_yoy", "core_yoy"])
train, test = d.loc[:"2018-12"], d.loc["2019-01":]

def ols_se(X, y):
    X1 = np.column_stack([np.ones(len(X)), X]); b = np.linalg.lstsq(X1, y, rcond=None)[0]
    res = y - X1 @ b; s2 = res @ res / (len(y) - X1.shape[1])
    return b, np.sqrt(np.diag(s2 * np.linalg.inv(X1.T @ X1)))

models, metrics_models = {}, {}
for tgt in TARGETS.values():
    metrics_models[tgt] = {}
    for key, (label, feats) in MODELS.items():
        mk = (lambda: RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_leaf=10, random_state=42)) if key == "rf" else LinearRegression
        m_eval = mk().fit(train[feats], train[tgt]); pred = m_eval.predict(test[feats])
        m_final = mk().fit(d[feats], d[tgt]); models[f"{tgt}:{key}"] = m_final
        metrics_models[tgt][key] = {"r2_in": float(m_final.score(d[feats], d[tgt])),
                                    "r2_oos": float(r2_score(test[tgt], pred)),
                                    "rmse_oos": float(np.sqrt(mean_squared_error(test[tgt], pred)))}
joblib.dump(models, ARTIFACT_DIR / "models.joblib")

sub = []
for label, s, e in ERAS:
    x = df.loc[s:e].dropna(subset=["cpi_yoy", "unrate"])
    b, se = ols_se(x[["unrate"]].values, x["cpi_yoy"].values)
    row = {"era": label, "n": int(len(x)), "slope_classic": float(b[1]), "se_classic": float(se[1])}
    xa = df.loc[s:e].dropna(subset=["cpi_yoy", "unrate", "mich"])
    if len(xa) >= 36:
        b2, se2 = ols_se(xa[["unrate", "mich"]].values, xa["cpi_yoy"].values)
        row.update({"slope_aug": float(b2[1]), "se_aug": float(se2[1]), "coef_mich": float(b2[2])})
    sub.append(row)

defaults = {"unrate": 4.0, "mich": 3.0, "fedfunds": 2.0, "wti": 70.0, "m2_yoy": float(round(d["m2_yoy"].median(), 1))}
metrics = {
    "stub": True, "sklearn_version": sklearn.__version__,
    "sample": {"start": str(d.index.min().date()), "end": str(d.index.max().date()), "n": int(len(d))},
    "split": {"train_end": "2018-12", "test_start": "2019-01", "n_train": int(len(train)), "n_test": int(len(test))},
    "train_ranges": {f: [float(d[f].min()), float(d[f].max())] for f in ALL_FEATURES},
    "slider_ranges": {f: [float(np.floor(d[f].min())), float(np.ceil(d[f].max())),
                          float(min(max(defaults[f], np.floor(d[f].min())), np.ceil(d[f].max())))] for f in ALL_FEATURES},
    "models": metrics_models,
    "coefs": {},  # real notebook fills HAC coefficient tables here: {target: {model: {var: {coef, se, p}}}}
    "subperiods": sub,
    "nrou": {"min": float(df["nrou"].min()), "max": float(df["nrou"].max()), "latest": float(df["nrou"].iloc[-1])},
    "oil": {"r2_oos_gain": 0.02, "coef_wti": 0.02, "t_wti": 3.0},
    "expectations": {"r2_oos_gain": 0.3, "coef_mich": 0.7, "t_mich": 8.0, "t5yie_swap_r2_in": 0.5, "mich_r2_in_same_sample": 0.55},
}
with open(ARTIFACT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("stub artifacts written: n_obs =", len(d))

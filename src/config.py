"""Shared contract between Track A (models) and Track B (app). Change only by agreement."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "phillips_monthly.csv"
ARTIFACT_DIR = ROOT / "artifacts"

# Target (inflation measure) label -> column in the CSV
TARGETS = {"Headline CPI": "cpi_yoy", "Core CPI": "core_yoy"}

ALL_FEATURES = ["unrate", "mich", "fedfunds", "wti", "m2_yoy"]

# model key -> (label, features)
MODELS = {
    "classic":   ("Classic Phillips Curve", ["unrate"]),
    "augmented": ("Expectations-Augmented", ["unrate", "mich"]),
    "full_ols":  ("Augmented + shocks (OLS)", ALL_FEATURES),
    "rf":        ("Random Forest (all variables)", ALL_FEATURES),
}

FEATURE_LABELS = {
    "unrate": "Unemployment (%)",
    "mich": "Expected inflation (%, Michigan survey)",
    "fedfunds": "Federal funds rate (%)",
    "wti": "WTI oil price ($)",
    "m2_yoy": "M2 money supply growth (% YoY)",
}
SLIDER_STEPS = {"unrate": 0.1, "mich": 0.1, "fedfunds": 0.05, "wti": 1.0, "m2_yoy": 0.1}

# Eras for the stability test (H1). (label, start, end)  end=None -> to the latest month
ERAS = [
    ("1960s", "1960-01", "1969-12"),
    ("1970-83 Stagflation & Volcker", "1970-01", "1983-12"),
    ("1984-2007 Great Moderation", "1984-01", "2007-12"),
    ("2008-19 Post-crisis", "2008-01", "2019-12"),
    ("2020-today", "2020-01", None),
]

# Review-1 nine series (slides 8-10)
SERIES = {
    "cpi": "CPIAUCSL", "core": "CPILFESL", "unrate": "UNRATE", "nrou": "NROU",
    "mich": "MICH", "t5yie": "T5YIE", "fedfunds": "FEDFUNDS", "wti": "DCOILWTICO", "m2": "M2SL",
}

# Columns of data/phillips_monthly.csv (index = month-start date, named "date")
DATA_COLUMNS = ["cpi_yoy", "core_yoy", "unrate", "nrou", "mich", "t5yie", "fedfunds", "wti", "m2_yoy"]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.config import (DATA_PATH, TARGETS, MODELS, FEATURE_LABELS, SLIDER_STEPS, ERAS, ALL_FEATURES)
from src.predict import load_artifacts, predict, out_of_range

st.set_page_config(page_title="US Phillips Curve Simulator", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)


@st.cache_resource
def get_artifacts():
    return load_artifacts()


df = load_data()
models, metrics = get_artifacts()
ranges = metrics["slider_ranges"]

# ---------------- Sidebar ----------------
st.sidebar.header("Adjust the scenario")
target_label = st.sidebar.radio("Inflation measure", list(TARGETS))
target = TARGETS[target_label]

inputs = {}
for f in ALL_FEATURES:
    lo, hi, default = ranges[f]
    inputs[f] = st.sidebar.slider(FEATURE_LABELS[f], float(lo), float(hi), float(default), SLIDER_STEPS[f])

# ---------------- Header ----------------
st.title("US Phillips Curve Simulator")
if metrics.get("stub"):
    st.warning("STUB DATA: synthetic placeholder numbers. Replace data/ and artifacts/ with the Colab outputs before presenting.")

tab1, tab2 = st.tabs(["Predictor", "Short-run & long-run curve"])

# ============================================================ TAB 1
with tab1:
    st.subheader("Model")
    keys = list(MODELS)
    model_key = st.radio("Choose a model", keys, format_func=lambda k: MODELS[k][0], horizontal=True)
    label, feats = MODELS[model_key]

    pred = predict(models, target, model_key, inputs)

    inputs_text = ", ".join(FEATURE_LABELS[f] for f in feats)
    if model_key == "classic":
        how = "A straight-line regression of inflation on unemployment alone: the basic tradeoff."
    elif model_key == "augmented":
        how = "A linear regression that adds household inflation expectations to the unemployment tradeoff."
    elif model_key == "full_ols":
        how = "A linear regression that adds the policy rate, oil price and money growth as supply and policy shocks."
    else:
        rp = metrics.get("rf_params", {}).get(target, {})
        rf_txt = (f" Settings for this measure: max depth {rp['max_depth']}, min leaf size {rp['min_samples_leaf']}."
                  if "max_depth" in rp and "min_samples_leaf" in rp else "")
        how = ("A Random Forest that learns non-linear patterns across all variables but cannot extrapolate "
               "beyond the data it has seen." + rf_txt)
    st.caption(f"Parameters used: {inputs_text}. {how}")
    bad = out_of_range(metrics, inputs, feats)
    if bad and model_key == "rf":
        st.warning("Outside the training range for: " + ", ".join(FEATURE_LABELS[b] for b in bad) +
                   ". A Random Forest cannot extrapolate, so this prediction is capped at the edge of what it has seen.")
    elif bad:
        st.info("Outside the training range for: " + ", ".join(FEATURE_LABELS[b] for b in bad) +
                ". Linear models extrapolate, so treat this prediction with caution.")

    st.metric(f"Predicted {target_label} inflation", f"{pred:.2f}%")

    yearly = df.loc[metrics["sample"]["start"]:, target].dropna().resample("YE").mean()
    years = yearly.index.year
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(years, yearly.values, marker="o", color="#4C78A8", lw=1.8, ms=5, label="Historical inflation (annual avg)", zorder=3)
    ax.axhline(pred, color="crimson", ls="--", lw=2.2, label=f"Your prediction ({pred:.2f}%)", zorder=2)
    ax.scatter([years.max() + 1], [pred], color="crimson", s=130, marker="D", zorder=5)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title(f"Model: {label}  |  Predicted {target_label}: {pred:.2f}%", fontsize=12, pad=12)
    ax.set_xlabel("Year"); ax.set_ylabel("Inflation (% YoY)")

    # (year, text, offset in points). Tune offsets by eye.
    episodes = [
        (2009, "2008-09: Global Financial Crisis\nDemand collapse \u2192 inflation turns negative", (-60, -70)),
        (2015, "2014-15: Oil price crash\n(shale oversupply) \u2192 near-zero inflation", (10, -70)),
        (2020, "2020: COVID-19 lockdowns\nDemand shock, muted annual average", (-150, 60)),
        (2022, "2022: Reopening demand surge +\nUkraine war oil shock \u2192 inflation peaks", (-210, 10)),
        (2024, "2023-24: Aggressive Fed rate hikes\npull inflation back down", (-20, 80)),
    ]
    for yr, text, off in episodes:
        if yr in years:
            yv = float(yearly[years == yr].iloc[0])
            ax.annotate(text, xy=(yr, yv), xytext=off, textcoords="offset points", fontsize=8, color="#333333",
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1), annotation_clip=False,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))
    ax.legend(loc="upper left")
    ax.set_ylim(min(yearly.min(), pred) - 3, max(yearly.max(), pred) + 3)
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================ TAB 2
with tab2:
    st.subheader("Short-run curve and the natural rate")
    st.write("Each dot is one month. The short-run curve is the fitted line; the long-run curve is vertical at the "
             "CBO natural rate of unemployment (NROU), the level at which only expected inflation remains.")
    era_labels = [e[0] for e in ERAS]
    chosen = st.multiselect("Eras to include", era_labels, default=era_labels)

    fig2, ax2 = plt.subplots(figsize=(10, 5.5))
    cmap = plt.get_cmap("tab10")
    pts = []
    for i, (lab, s, e) in enumerate(ERAS):
        if lab not in chosen:
            continue
        x = df.loc[s:e, ["unrate", target]].dropna()
        ax2.scatter(x["unrate"], x[target], s=14, alpha=0.65, color=cmap(i), label=lab)
        pts.append(x)
    if pts:
        allx = pd.concat(pts)
        if len(allx) > 2:
            b, a = np.polyfit(allx["unrate"], allx[target], 1)
            xs = np.linspace(allx["unrate"].min(), allx["unrate"].max(), 50)
            ax2.plot(xs, a + b * xs, color="black", lw=2, label=f"Short-run fit (slope {b:.2f})")
    nrou = metrics["nrou"]
    ax2.axvspan(nrou["min"], nrou["max"], color="gray", alpha=0.15, label="NROU range over history")
    ax2.axvline(nrou["latest"], color="crimson", lw=2, label=f"Long-run curve at NROU = {nrou['latest']:.1f}%")
    ax2.set_xlabel("Unemployment rate (%)"); ax2.set_ylabel(f"{target_label} inflation (% YoY)")
    ax2.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig2)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.config import (DATA_PATH, TARGETS, MODELS, FEATURE_LABELS, SLIDER_STEPS, ERAS, SERIES, ALL_FEATURES)
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
st.write(
    "Built on US macro data from FRED. Compares the classic Phillips Curve, an expectations-augmented "
    "version, an augmented model with supply and policy shocks, and a Random Forest, to show why a "
    "central bank cannot rely on a single stable curve to control inflation."
)

tab1, tab2, tab3, tab4 = st.tabs(["Predictor", "Short-run & long-run curve", "Why the Classic Curve Fails", "Data, method & limits"])

# ============================================================ TAB 1
with tab1:
    st.subheader("Model")
    keys = list(MODELS)
    model_key = st.radio("Choose a model", keys, format_func=lambda k: MODELS[k][0], horizontal=True)
    label, feats = MODELS[model_key]
    mm = metrics["models"][target][model_key]

    pred = predict(models, target, model_key, inputs)
    st.caption(
        f"Uses: {', '.join(FEATURE_LABELS[f] for f in feats)}. "
        f"R\u00b2 in-sample = {mm['r2_in']:.2f}; out-of-sample (test {metrics['split']['test_start']} onward) = "
        f"{mm['r2_oos']:.2f}, RMSE = {mm['rmse_oos']:.2f} pp."
    )
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
    st.caption("Annotations mark major US macro episodes and the demand/supply forces behind each swing.")

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

# ============================================================ TAB 3
with tab3:
    st.subheader("Why the Classic Curve Fails")
    st.write("Each block tests one hypothesis from Review 1 using the numbers computed in the Colab notebook.")

    st.markdown("**Model comparison** (train through " + metrics["split"]["train_end"] +
                ", test from " + metrics["split"]["test_start"] + ")")
    rows = [{"Model": MODELS[k][0], "R\u00b2 in-sample": v["r2_in"], "R\u00b2 out-of-sample": v["r2_oos"],
             "RMSE out-of-sample (pp)": v["rmse_oos"]} for k, v in metrics["models"][target].items()]
    st.dataframe(pd.DataFrame(rows).round(3))

    # H1
    st.markdown("**H1: The curve is not one stable relationship.**")
    sp = pd.DataFrame(metrics["subperiods"])
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.bar(sp["era"], sp["slope_classic"], yerr=1.96 * sp["se_classic"], capsize=4, color="#4C78A8")
    ax3.axhline(0, color="gray", lw=0.8)
    ax3.set_ylabel("Slope of inflation on unemployment"); plt.xticks(rotation=20, ha="right")
    plt.tight_layout(); st.pyplot(fig3)
    st.caption(f"Classic slope ranges from {sp['slope_classic'].min():.2f} to {sp['slope_classic'].max():.2f} "
               "across eras (bars show 95% intervals, HAC standard errors). Headline CPI used for the era test.")

    # H2
    st.markdown("**H2: The natural rate is a moving target.**")
    nr = df["nrou"].dropna()
    fig4, ax4 = plt.subplots(figsize=(9, 3.5))
    ax4.plot(nr.index, nr.values, label="NROU (CBO)")
    ax4.plot(df.index, df["unrate"], alpha=0.5, label="Unemployment rate")
    ax4.legend(); ax4.set_ylabel("%"); plt.tight_layout(); st.pyplot(fig4)
    st.caption(f"NROU moves between {metrics['nrou']['min']:.1f}% and {metrics['nrou']['max']:.1f}%, "
               "so the unemployment gap is an uncertain policy signal.")

    # H3, H4
    c1, c2 = st.columns(2)
    o, ex = metrics["oil"], metrics["expectations"]
    with c1:
        st.markdown("**H3: Supply shocks shift the curve on their own.**")
        st.metric("Out-of-sample R\u00b2 gain from adding oil", f"{o['r2_oos_gain']:+.3f}")
        st.caption(f"Oil coefficient {o['coef_wti']:.3f} (t = {o['t_wti']:.1f}).")
    with c2:
        st.markdown("**H4: Expectations may matter more than the tradeoff.**")
        st.metric("Out-of-sample R\u00b2 gain from adding expectations", f"{ex['r2_oos_gain']:+.3f}")
        st.caption(f"Michigan coefficient {ex['coef_mich']:.3f} (t = {ex['t_mich']:.1f}).")

# ============================================================ TAB 4
with tab4:
    st.subheader("Data")
    st.table(pd.DataFrame({"Variable": list(SERIES), "FRED ID": list(SERIES.values())}))
    st.write(f"Model sample: {metrics['sample']['start']} to {metrics['sample']['end']} "
             f"({metrics['sample']['n']} monthly observations). Daily and quarterly series converted to monthly; "
             "inflation is year-over-year; M2 enters as YoY growth.")
    st.subheader("Limits")
    st.markdown(
        "- The Fed's policy rate responds to inflation, so it is context, not an independent cause.\n"
        "- The natural rate is estimated (CBO), not observed.\n"
        "- Policy works with lags, so a simulator cannot show timing.\n"
        "- Monthly year-over-year inflation is highly persistent; we use HAC standard errors and out-of-sample tests.\n"
        "- Random Forest cannot extrapolate beyond the data it has seen.\n"
        "- Aggregate regressions can understate the true slope (Furlanetto & Lepetit, 2024)."
    )

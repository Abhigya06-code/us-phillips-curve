import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="US Phillips Curve", layout="wide")

# ---------- Load data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("us_phillips_data.csv", index_col=0, parse_dates=True)
    return df

df = load_data()

# ---------- Train models (cached, trained once per session) ----------
@st.cache_resource
def train_models(df):
    X_classic = df[["unemployment"]].values
    X_aug = df[["unemployment", "mich"]].values
    X_full = df[["unemployment", "mich", "fedfunds", "oil_wti", "m2sl"]].values
    y = df["inflation_headline"].values

    classic = LinearRegression().fit(X_classic, y)
    augmented = LinearRegression().fit(X_aug, y)
    rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42).fit(X_full, y)

    r2_classic = classic.score(X_classic, y)
    r2_aug = augmented.score(X_aug, y)
    r2_rf = rf.score(X_full, y)

    return classic, augmented, rf, r2_classic, r2_aug, r2_rf

classic, augmented, rf, r2_classic, r2_aug, r2_rf = train_models(df)

# ---------- Sidebar: scenario sliders ----------
st.sidebar.header("Adjust the scenario")

unemployment = st.sidebar.slider("Unemployment (%)", 3.0, 15.0, 4.0, 0.1)
mich = st.sidebar.slider("Expected Inflation (%, Michigan survey)", 1.0, 11.0, 3.0, 0.1)
fedfunds = st.sidebar.slider("Federal Funds Rate (%)", 0.0, 6.0, 2.0, 0.05)
oil = st.sidebar.slider("WTI Oil Price ($)", 15.0, 135.0, 70.0, 1.0)
m2sl = st.sidebar.slider(
    "M2 Money Supply ($ billions)",
    float(df["m2sl"].min()), float(df["m2sl"].max()), float(df["m2sl"].median()), 10.0
)

st.title("US Phillips Curve Simulator")
st.write(
    "Built on US 2003\u20132026 macro data (FRED): compares the classic Phillips Curve, "
    "an expectations-augmented version, and a Random Forest, to show why a central bank "
    "can't rely on a single stable curve to control inflation."
)

tab1, tab2 = st.tabs(["Predictor", "Why the Classic Curve Fails"])

# ============================================================
# TAB 1: Predictor
# ============================================================
with tab1:
    st.subheader("Model")
    model_choice = st.radio(
        "Choose a model",
        ["Classic Phillips Curve", "Expectations-Augmented", "Random Forest (all variables)"],
        horizontal=True,
    )

    if model_choice == "Classic Phillips Curve":
        pred = classic.predict([[unemployment]])[0]
        st.caption(f"Uses only unemployment. R\u00b2 \u2248 {r2_classic:.3f} \u2014 weak predictive power on its own.")
    elif model_choice == "Expectations-Augmented":
        pred = augmented.predict([[unemployment, mich]])[0]
        st.caption(f"Uses unemployment + expected inflation. R\u00b2 \u2248 {r2_aug:.3f} \u2014 expectations do most of the work.")
    else:
        pred = rf.predict([[unemployment, mich, fedfunds, oil, m2sl]])[0]
        st.caption(f"Uses all 5 variables via Random Forest. R\u00b2 \u2248 {r2_rf:.3f} on training data.")

    st.metric("Predicted Inflation", f"{pred:.2f}%")

    # ---------- Historical chart with detailed, accurate annotations ----------
    fig, ax = plt.subplots(figsize=(11, 6.5))
    yearly = df["inflation_headline"].resample("YE").mean()
    years = yearly.index.year

    ax.plot(years, yearly.values, marker="o", color="#4C78A8", linewidth=1.8,
            markersize=5, label="Historical Inflation", zorder=3)
    ax.axhline(pred, color="crimson", linestyle="--", linewidth=2.2,
               label=f"Your Prediction ({pred:.2f}%)", zorder=2)
    ax.scatter([years.max() + 1], [pred], color="crimson", s=130, zorder=5, marker="D")

    ax.set_title(f"Model: {model_choice}  |  Predicted Inflation: {pred:.2f}%", fontsize=12, pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Inflation (%)")
    ax.axhline(0, color="gray", linewidth=0.6, zorder=1)

    # Annotations: (year, label_text, xytext_offset)
    annotations = [
        (2009, "2008-09: Global Financial Crisis\nDemand collapse \u2192 inflation turns negative", (-4.5, -3.2)),
        (2015, "2014-15: Oil price crash\n(shale oversupply) \u2192 near-zero inflation", (0.3, -2.6)),
        (2020, "2020: COVID-19 lockdowns\nDemand shock, muted annual average", (-6, 3.0)),
        (2022, "2022: Reopening demand surge +\nUkraine war oil shock \u2192 inflation peaks near 8%", (-9, 1.8)),
        (2024, "2023-24: Aggressive Fed rate hikes\n(funds rate to ~5.3%) pull inflation back down", (0.4, 2.3)),
    ]

    for year, text, (dx, dy) in annotations:
        if year in years:
            y_val = yearly[yearly.index.year == year].values[0]
            ax.annotate(
                text, xy=(year, y_val), xytext=(year + dx, y_val + dy),
                fontsize=8, color="#333333",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
            )

    ax.legend(loc="upper left")
    ax.set_ylim(yearly.min() - 4, yearly.max() + 4)
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "Annotations mark major US macro episodes and the demand/supply-side forces "
        "behind each inflation swing \u2014 useful talking points for explaining *why* "
        "the curve doesn't hold steady."
    )

# ============================================================
# TAB 2: Why the Classic Curve Fails
# ============================================================
with tab2:
    st.subheader("Why the Classic Curve Fails")
    st.write(
        "The classic Phillips Curve regresses inflation on unemployment alone, assuming "
        "expected inflation stays fixed. Real US data spans eras where expected inflation "
        "itself moved a lot (from ~10%+ in the early 1980s down to ~2% in the 2010s), so "
        "pooling decades together washes out the slope."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Classic (unemployment only)", f"R\u00b2 = {r2_classic:.3f}")
    col2.metric("Expectations-Augmented", f"R\u00b2 = {r2_aug:.3f}")
    col3.metric("Random Forest (all 5 vars)", f"R\u00b2 = {r2_rf:.3f}")

    st.write(
        "The jump from the classic to the augmented model is the actual finding: unemployment "
        "alone barely predicts inflation, but once expected inflation is included, the fit "
        "improves sharply \u2014 which is why a central bank's main lever isn't just the policy "
        "rate itself, but managing what the public *expects* inflation to be."
    )

    st.markdown("**Key episodes that break the simple curve:**")
    st.write(
        "- **2008-09 Financial Crisis** \u2014 unemployment spiked, but so did deflation risk, "
        "not the inflation the simple curve would predict at low unemployment periods nearby.\n"
        "- **2014-15 oil price crash** \u2014 a pure supply-side shock (US shale boom pushed prices "
        "down) that had nothing to do with unemployment, yet moved inflation to near zero.\n"
        "- **2021-22 post-COVID surge** \u2014 unemployment was falling (which the classic curve "
        "reads as inflationary), but the size of the inflation spike (up to 8%) was driven far "
        "more by supply chain breakdowns and the Ukraine war oil shock than by labor market tightness.\n"
        "- **2023-24 disinflation** \u2014 inflation fell back down even as unemployment stayed "
        "historically low, contradicting the basic curve's tradeoff logic \u2014 credited largely "
        "to the Fed's rate hikes cooling expectations rather than unemployment rising."
    )

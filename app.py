import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import requests
import streamlit as st

from src.config import (DATA_PATH, TARGETS, MODELS, FEATURE_LABELS, SLIDER_STEPS, ERAS, ALL_FEATURES)
from src.predict import load_artifacts, predict, out_of_range

st.set_page_config(page_title="US Phillips Curve Simulator", page_icon="\U0001F4C8", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px;}
    h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.4rem;}
    [data-testid="stMetric"] {background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.22);
                              border-radius: 12px; padding: 14px 18px;}
    [data-testid="stMetricValue"] {font-size: 2rem; font-weight: 700;}
    [data-testid="stMetricLabel"] p {font-size: 0.85rem; opacity: 0.8;}
    button[data-baseweb="tab"] p {font-size: 1.02rem; font-weight: 600;}
    [data-testid="stSidebar"] h2 {font-size: 1.3rem;}
    footer {visibility: hidden;}
    .stAppDeployButton {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- chart theme
THEMES = {
    "dark": dict(bg="#0E1117", fg="#FAFAFA", muted="#9AA0AC", grid="#262B36", edge="#3A3F4D",
                 line="#4C9BE8", pred="#FF5A5F", actual="#F2C14E", box="#1A1E29"),
    "light": dict(bg="#FFFFFF", fg="#1F2430", muted="#5F6673", grid="#E6E8EC", edge="#C9CDD4",
                  line="#2F6FB5", pred="#D6333A", actual="#C98A00", box="#F6F7F9"),
}


def theme():
    return THEMES["dark"] if st.get_option("theme.base") == "dark" else THEMES["light"]


def style_axes(fig, ax, T):
    fig.patch.set_facecolor(T["bg"])
    ax.set_facecolor(T["bg"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T["edge"])
    ax.tick_params(colors=T["muted"], labelsize=9)
    ax.xaxis.label.set_color(T["muted"])
    ax.yaxis.label.set_color(T["muted"])
    ax.title.set_color(T["fg"])
    ax.grid(True, color=T["grid"], lw=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- data
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)


@st.cache_resource
def get_artifacts():
    return load_artifacts()


FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
LIVE_IDS = {"unrate": "UNRATE", "mich": "MICH", "fedfunds": "FEDFUNDS", "wti": "DCOILWTICO",
            "m2": "M2SL", "cpi": "CPIAUCSL", "core": "CPILFESL"}
SHORT = {"unrate": ("Unemployment", "%"), "mich": ("Expected inflation", "%"), "fedfunds": ("Fed funds rate", "%"),
         "wti": ("WTI oil", "$"), "m2_yoy": ("M2 growth", "%")}


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def fetch_live():
    """Download the latest observations of each series from FRED's public CSV endpoint (no API key)."""
    start = (pd.Timestamp.today() - pd.DateOffset(months=30)).strftime("%Y-%m-%d")

    def one(item):
        key, sid = item
        try:
            r = requests.get(FRED_URL, params={"id": sid, "cosd": start}, timeout=8,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            raw = pd.read_csv(io.StringIO(r.text))
            s = pd.Series(pd.to_numeric(raw.iloc[:, 1], errors="coerce").values,
                          index=pd.to_datetime(raw.iloc[:, 0], errors="coerce")).dropna()
            return key, (s if len(s) else None)
        except Exception:
            return key, None

    with ThreadPoolExecutor(max_workers=len(LIVE_IDS)) as ex:
        out = dict(ex.map(one, LIVE_IDS.items()))
    if all(v is None for v in out.values()):
        raise RuntimeError("FRED not reachable")   # failures are not cached
    return out


def build_latest(df):
    """Latest value of every scenario input and of both inflation measures.
    Live from FRED when reachable, otherwise the last values of the saved dataset.
    Returns ({name: (value, date, is_live)}, number_of_live_series)."""
    try:
        raw = fetch_live()
    except Exception:
        raw = {}

    def saved(name):
        s = df[name].dropna()
        return float(s.iloc[-1]), s.index[-1], False

    def yoy(s):
        return None if s is None else (s.pct_change(12) * 100).dropna()

    out, n_live = {}, 0
    sources = {"unrate": raw.get("unrate"), "mich": raw.get("mich"), "fedfunds": raw.get("fedfunds"),
               "wti": raw.get("wti"), "m2_yoy": yoy(raw.get("m2")),
               "cpi_yoy": yoy(raw.get("cpi")), "core_yoy": yoy(raw.get("core"))}
    for name, s in sources.items():
        if s is not None and len(s):
            out[name] = (float(s.iloc[-1]), s.index[-1], True)
            n_live += 1
        else:
            out[name] = saved(name)
    return out, n_live


def fmt_date(name, d):
    return d.strftime("%d %b %Y") if name == "wti" else d.strftime("%b %Y")


def snap(v, lo, hi, step):
    v = min(max(v, lo), hi)
    return float(round(round(v / step) * step, 4))


df = load_data()
models, metrics = get_artifacts()
ranges = metrics["slider_ranges"]

# ---------------------------------------------------------------- scenario state
apply_latest = st.session_state.pop("apply_latest", False)
if apply_latest:
    fetch_live.clear()                      # "Use latest data" pulls fresh numbers
with st.spinner("Loading latest data..."):
    latest, n_live = build_latest(df)

if apply_latest or "scenario_init" not in st.session_state:
    for f in ALL_FEATURES:
        lo, hi, _ = ranges[f]
        st.session_state[f"s_{f}"] = snap(latest[f][0], lo, hi, SLIDER_STEPS[f])
    st.session_state["scenario_init"] = True


def _request_latest():
    st.session_state["apply_latest"] = True


def _typical_values():
    for f in ALL_FEATURES:
        st.session_state[f"s_{f}"] = float(ranges[f][2])


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Scenario")
target_label = st.sidebar.radio("Inflation measure", list(TARGETS), horizontal=True)
target = TARGETS[target_label]

st.sidebar.markdown("**Economic conditions**")
inputs = {}
for f in ALL_FEATURES:
    lo, hi, _ = ranges[f]
    inputs[f] = st.sidebar.slider(FEATURE_LABELS[f], float(lo), float(hi), step=SLIDER_STEPS[f], key=f"s_{f}")

b1, b2 = st.sidebar.columns(2)
b1.button("Use latest data", on_click=_request_latest, type="primary", use_container_width=True)
b2.button("Typical values", on_click=_typical_values, use_container_width=True)

if n_live > 0:
    st.sidebar.caption(f"Sliders start at the latest FRED data ({n_live} of {len(latest)} series live).")
else:
    st.sidebar.caption("FRED could not be reached, so sliders start at the latest values in the saved dataset.")
with st.sidebar.expander("Latest values"):
    for f in ALL_FEATURES:
        v, d, live = latest[f]
        name, unit = SHORT[f]
        shown = f"${v:,.2f}" if unit == "$" else f"{v:.2f}%"
        st.write(f"{name}: **{shown}** ({fmt_date(f, d)}{'' if live else ', saved'})")

# ---------------------------------------------------------------- header
st.title("US Phillips Curve Simulator")
if metrics.get("stub"):
    st.warning("STUB DATA: synthetic placeholder numbers. Replace data/ and artifacts/ with the Colab outputs before presenting.")

tab1, tab2 = st.tabs(["Predictor", "Short-run & long-run curve"])

# ============================================================ TAB 1
with tab1:
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

    actual, actual_date, actual_live = latest[target]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Predicted {target_label} inflation", f"{pred:.2f}%")
    c2.metric(f"Latest actual ({fmt_date(target, actual_date)})", f"{actual:.2f}%")
    c3.metric("Prediction minus actual", f"{pred - actual:+.2f} pp")

    T = theme()
    yearly = df.loc[metrics["sample"]["start"]:, target].dropna().resample("YE").mean()
    years = yearly.index.year
    xr = years.max() + 1
    fig, ax = plt.subplots(figsize=(11, 6))
    style_axes(fig, ax, T)
    ax.plot(years, yearly.values, marker="o", color=T["line"], lw=2, ms=5, label="Historical inflation (annual avg)", zorder=3)
    ax.axhline(pred, color=T["pred"], ls="--", lw=1.8, label=f"Your prediction ({pred:.2f}%)", zorder=2)
    ax.scatter([xr], [pred], color=T["pred"], s=120, marker="D", zorder=5)
    ax.scatter([xr], [actual], facecolors="none", edgecolors=T["actual"], s=110, lw=2, zorder=5,
               label=f"Latest actual ({actual:.2f}%)")
    ax.axhline(0, color=T["edge"], lw=0.9)
    ax.set_title(f"{label}  |  Predicted {target_label}: {pred:.2f}%", fontsize=12, pad=12, loc="left", color=T["fg"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Inflation (% YoY)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(years.min() - 0.8, xr + 1.2)

    # (year, text, offset in points). Tune offsets by eye.
    episodes = [
        (2009, "2008-09: Global Financial Crisis\nDemand collapse \u2192 inflation turns negative", (-60, -70)),
        (2015, "2014-15: Oil price crash\n(shale oversupply) \u2192 near-zero inflation", (10, -70)),
        (2020, "2020: COVID-19 lockdowns\nDemand shock, muted annual average", (-150, 105)),
        (2022, "2022: Reopening demand surge +\nUkraine war oil shock \u2192 inflation peaks", (-210, 10)),
        (2024, "2023-24: Aggressive Fed rate hikes\npull inflation back down", (-20, 80)),
    ]
    for yr, text, off in episodes:
        if yr in years:
            yv = float(yearly[years == yr].iloc[0])
            ax.annotate(text, xy=(yr, yv), xytext=off, textcoords="offset points", fontsize=8, color=T["fg"],
                        arrowprops=dict(arrowstyle="->", color=T["muted"], lw=1), annotation_clip=False,
                        bbox=dict(boxstyle="round,pad=0.35", fc=T["box"], ec=T["edge"], alpha=0.95))
    ax.legend(loc="upper left", frameon=False, labelcolor=T["fg"], fontsize=9)
    ax.set_ylim(min(yearly.min(), pred, actual) - 3, max(yearly.max(), pred, actual) + 3)
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================ TAB 2
with tab2:
    st.subheader("Short-run curve and the natural rate")
    st.write("Each dot is one month. The short-run curve is the fitted line; the long-run curve is vertical at the "
             "CBO natural rate of unemployment (NROU), the level at which only expected inflation remains.")
    era_labels = [e[0] for e in ERAS]
    chosen = st.multiselect("Eras to include", era_labels, default=era_labels)

    T = theme()
    fig2, ax2 = plt.subplots(figsize=(10, 5.5))
    style_axes(fig2, ax2, T)
    cmap = plt.get_cmap("tab10")
    pts = []
    for i, (lab, s, e) in enumerate(ERAS):
        if lab not in chosen:
            continue
        x = df.loc[s:e, ["unrate", target]].dropna()
        ax2.scatter(x["unrate"], x[target], s=16, alpha=0.7, color=cmap(i), label=lab, linewidths=0)
        pts.append(x)
    if pts:
        allx = pd.concat(pts)
        if len(allx) > 2:
            b, a = np.polyfit(allx["unrate"], allx[target], 1)
            xs = np.linspace(allx["unrate"].min(), allx["unrate"].max(), 50)
            ax2.plot(xs, a + b * xs, color=T["fg"], lw=2, label=f"Short-run fit (slope {b:.2f})")
    nrou = metrics["nrou"]
    ax2.axvspan(nrou["min"], nrou["max"], color=T["muted"], alpha=0.15, label="NROU range over history")
    ax2.axvline(nrou["latest"], color=T["pred"], lw=2, label=f"Long-run curve at NROU = {nrou['latest']:.1f}%")
    ax2.set_xlabel("Unemployment rate (%)")
    ax2.set_ylabel(f"{target_label} inflation (% YoY)")
    ax2.legend(fontsize=8, loc="upper right", frameon=False, labelcolor=T["fg"])
    plt.tight_layout()
    st.pyplot(fig2)
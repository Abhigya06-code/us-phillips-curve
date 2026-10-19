import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (DATA_PATH, TARGETS, MODELS, FEATURE_LABELS, SLIDER_STEPS, ERAS, ALL_FEATURES)
from src.predict import load_artifacts, predict, out_of_range

st.set_page_config(page_title="US Phillips Curve Simulator", layout="wide")

# ---------------------------------------------------------------- look
# Dollar green carries structure, blue is always "your scenario", ink is always "what actually happened".
GREEN, BLUE, INK, MUTED = "#0F5A46", "#2457F5", "#1C2420", "#6A7670"
PANEL, RULE, AMBER, RED = "#EEF3F0", "#D5DDD8", "#C98A00", "#C8402B"
FONT = "Hanken Grotesk, Helvetica Neue, Arial, sans-serif"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700;12..96,800&family=Hanken+Grotesk:wght@400;500;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"], .stMarkdown, button, input, label {{
        font-family: {FONT}; color: {INK}; }}
    .block-container {{ padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1120px; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    footer, .stAppDeployButton {{ visibility: hidden; display: none; }}

    h1.title {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 2.7rem;
                letter-spacing: -0.03em; line-height: 1.02; margin: 0 0 .5rem; padding: 0; color: {INK}; }}
    p.lede {{ color: {MUTED}; font-size: 1.05rem; margin: 0 0 1.6rem; max-width: 560px; }}
    h2.h {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.35rem;
            letter-spacing: -0.015em; margin: 2rem 0 .6rem; padding: 0; }}

    /* the scale: the one thing on this page that is custom */
    .scale {{ background: {PANEL}; border-radius: 6px; padding: 22px 34px 14px; margin-bottom: .4rem; }}
    .track {{ position: relative; height: 212px; }}
    .axis {{ position: absolute; left: 0; right: 0; top: 108px; height: 2px; background: {INK}; }}
    .gap {{ position: absolute; top: 105px; height: 8px; background: {BLUE}; opacity: .28; border-radius: 4px; }}
    .tick {{ position: absolute; top: 110px; background: {PANEL}; z-index: 1; transform: translateX(-50%); font-size: .74rem; color: {MUTED};
             font-variant-numeric: tabular-nums; padding-top: 5px; }}
    .tick:before {{ content: ""; position: absolute; left: 50%; top: 0; height: 5px; width: 1px; background: {MUTED}; }}
    .target {{ position: absolute; top: 4px; bottom: 6px; width: 0; border-left: 2px dashed {GREEN}; }}
    .target span {{ position: absolute; top: 0; left: 8px; font-size: .78rem; font-weight: 600; color: {GREEN};
                    white-space: nowrap; }}
    .mk {{ position: absolute; transform: translateX(-50%); text-align: center; white-space: nowrap; }}
    .mk .v {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 2.3rem;
              letter-spacing: -0.02em; line-height: 1; font-variant-numeric: tabular-nums; }}
    .mk .t {{ font-size: .8rem; font-weight: 600; margin-top: 3px; }}
    .mk.pred {{ top: 36px; color: {BLUE}; }}
    .mk.pred:after {{ content: ""; position: absolute; left: 50%; top: 66px; width: 2px; height: 26px; background: {BLUE}; }}
    .mk.act {{ top: 120px; padding-top: 30px; color: {INK}; }}
    .mk.act:before {{ content: ""; position: absolute; left: 50%; top: -1px; width: 2px; height: 30px; background: {INK}; }}
    .dotp, .dota {{ position: absolute; top: 101px; width: 16px; height: 16px; transform: translateX(-50%); border-radius: 50%; }}
    .dotp {{ background: {BLUE}; }} .dota {{ background: {INK}; }}
    p.verdict {{ margin: .9rem 0 0; font-size: 1.02rem; }}
    p.verdict b {{ font-weight: 700; }}

    @media (max-width: 640px) {{
      .scale {{ padding: 16px 22px 10px; }}
      .mk .v {{ font-size: 1.7rem; }}
      h1.title {{ font-size: 2.1rem; }}
    }}

    /* tables */
    table.t {{ border-collapse: collapse; width: 100%; font-size: .93rem; }}
    table.t th {{ text-align: left; font-weight: 600; color: {MUTED}; padding: 6px 10px 6px 0; border-bottom: 1px solid {INK}; }}
    table.t td {{ padding: 8px 10px 8px 0; border-bottom: 1px solid {RULE}; font-variant-numeric: tabular-nums; }}
    table.t .r {{ text-align: right; }}
    table.t.side {{ font-size: .82rem; }}
    .muted {{ color: {MUTED}; font-size: .88rem; }}

    /* streamlit controls */
    button[data-baseweb="tab"] p {{ font-size: 1rem; font-weight: 600; }}
    div[data-baseweb="tab-highlight"] {{ background-color: {GREEN} !important; }}
    [data-testid="stSidebar"] {{ border-right: 1px solid {RULE}; }}
    [data-testid="stSidebar"] h3 {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.15rem; margin-top: .4rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def show(ch):
    try:
        st.altair_chart(ch, width="stretch", theme=None)
    except TypeError:                      # older Streamlit
        st.altair_chart(ch, use_container_width=True, theme=None)


def chart_style(ch, height):
    return (ch.properties(height=height)
            .configure(font=FONT, background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, labelFontSize=12, titleFontSize=12, titleFontWeight=500,
                            gridColor="#E6ECE8", domainColor=RULE, tickColor=RULE)
            .configure_legend(labelColor=INK, labelFontSize=12, titleFontSize=0, orient="top", symbolType="circle"))


# ---------------------------------------------------------------- data
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)


@st.cache_resource
def get_artifacts():
    return load_artifacts()


def build_default_scenario(df):
    """Starting values: the latest observation of every input and of both inflation measures in the saved dataset.
    Computed once per session, when the app opens."""
    out = {}
    for name in list(ALL_FEATURES) + list(TARGETS.values()):
        series = df[name].dropna()
        out[name] = (float(series.iloc[-1]), series.index[-1])
    return out


def fmt_date(name, d):
    return d.strftime("%d %b %Y") if name == "wti" else d.strftime("%b %Y")


def snap(v, lo, hi, step):
    v = min(max(v, lo), hi)
    return float(round(round(v / step) * step, 4))


df = load_data()
models, metrics = get_artifacts()
ranges = metrics["slider_ranges"]

# ---------------------------------------------------------------- scenario state
latest = build_default_scenario(df)
if "scenario_init" not in st.session_state:
    for f in ALL_FEATURES:
        lo, hi, _ = ranges[f]
        st.session_state[f"s_{f}"] = snap(latest[f][0], lo, hi, SLIDER_STEPS[f])
    st.session_state["scenario_init"] = True


# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("### Scenario")
target_label = st.sidebar.radio("Inflation measure", list(TARGETS), horizontal=True)
target = TARGETS[target_label]

inputs = {}
for f in ALL_FEATURES:
    lo, hi, _ = ranges[f]
    inputs[f] = st.sidebar.slider(FEATURE_LABELS[f], float(lo), float(hi), step=SLIDER_STEPS[f], key=f"s_{f}")


# ---------------------------------------------------------------- header
st.markdown("<h1 class='title'>US Phillips Curve Simulator</h1>"
            "<p class='lede'>Set the economic conditions and see what each model predicts for inflation.</p>",
            unsafe_allow_html=True)
if metrics.get("stub"):
    st.warning("STUB DATA: synthetic placeholder numbers. Replace data/ and artifacts/ with the Colab outputs before presenting.")

# ---------------------------------------------------------------- model + prediction
keys = [k for k in MODELS if k != "full_ols"]
model_key = st.radio("Model", keys, format_func=lambda k: MODELS[k][0], horizontal=True)
label, feats = MODELS[model_key]
st.markdown(f"<div class='muted' style='margin:-.3rem 0 1rem'>Uses: {', '.join(FEATURE_LABELS[f] for f in feats)}</div>",
            unsafe_allow_html=True)

pred = predict(models, target, model_key, inputs)
actual, actual_date = latest[target]
diff = pred - actual

bad = out_of_range(metrics, inputs, feats)
if bad and model_key == "rf":
    st.warning("Outside the training range for: " + ", ".join(FEATURE_LABELS[b] for b in bad) +
               ". A Random Forest cannot extrapolate, so this prediction is capped at the edge of what it has seen.")
elif bad:
    st.info("Outside the training range for: " + ", ".join(FEATURE_LABELS[b] for b in bad) +
            ". Linear models extrapolate, so treat this prediction with caution.")

# the scale
lo_ax = min(0, int(np.floor(min(pred, actual))) - 1)
hi_ax = max(8, int(np.ceil(max(pred, actual))) + 1)


def pos(v):
    return 100 * (v - lo_ax) / (hi_ax - lo_ax)


ticks = "".join(f"<div class='tick' style='left:{pos(t):.2f}%'>{t}%</div>" for t in range(lo_ax, hi_ax + 1))
gap_l, gap_w = min(pos(pred), pos(actual)), abs(pos(pred) - pos(actual))
if abs(diff) < 0.25:
    verdict = "The model lands close to actual inflation."
elif diff > 0:
    verdict = f"The model is <b>{diff:.2f} points above</b> actual inflation."
else:
    verdict = f"The model is <b>{abs(diff):.2f} points below</b> actual inflation."
st.markdown(
    f"""
    <div class="scale"><div class="track">
      <div class="axis"></div>
      <div class="gap" style="left:{gap_l:.2f}%; width:{gap_w:.2f}%"></div>
      {ticks}
      <div class="target" style="left:{pos(2):.2f}%"><span>Fed target 2%</span></div>
      <div class="dotp" style="left:{pos(pred):.2f}%"></div><div class="dota" style="left:{pos(actual):.2f}%"></div>
      <div class="mk pred" style="left:{pos(pred):.2f}%"><div class="v">{pred:.2f}%</div><div class="t">{label}</div></div>
      <div class="mk act" style="left:{pos(actual):.2f}%"><div class="v">{actual:.2f}%</div>
        <div class="t">Actual, {fmt_date(target, actual_date)}</div></div>
    </div>
    <p class="verdict" style="margin-top:.2rem">{verdict}</p></div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["History", "Short-run and long-run curve"])

# ============================================================ TAB 1
with tab1:
    yearly = df.loc[metrics["sample"]["start"]:, target].dropna().resample("YE").mean()
    hist = pd.DataFrame({"Year": yearly.index.year, "Inflation": yearly.values})
    xr = int(hist["Year"].max()) + 1
    y_lo = min(hist["Inflation"].min(), pred, actual, 0) - 0.8
    y_hi = max(hist["Inflation"].max(), pred, actual) + 1.4

    base = alt.Chart(hist).encode(x=alt.X("Year:Q", axis=alt.Axis(format="d", values=list(range(int(hist["Year"].min()) + 1, xr, 2)), title=None),
                                          scale=alt.Scale(domain=[hist["Year"].min() - 0.5, xr + 3.2], nice=False)),
                                  y=alt.Y("Inflation:Q", axis=alt.Axis(title=f"{target_label} inflation, % year over year"),
                                          scale=alt.Scale(domain=[y_lo, y_hi], nice=False)))
    line = base.mark_line(color=INK, strokeWidth=2.2)
    pts = base.mark_circle(color=INK, size=36, opacity=0).encode(tooltip=[alt.Tooltip("Year:Q", format="d"),
                                                                       alt.Tooltip("Inflation:Q", format=".2f")])
    lines = pd.DataFrame({"y": [2.0, pred], "kind": ["target", "pred"]})
    rules = alt.Chart(lines).mark_rule(strokeDash=[5, 4], strokeWidth=1.6).encode(
        y="y:Q", color=alt.Color("kind:N", scale=alt.Scale(domain=["target", "pred"], range=[GREEN, BLUE]), legend=None))
    ends = pd.DataFrame({"x": [xr + 0.2, xr + 0.2, xr + 0.2],
                         "y": [pred + (0.35 if pred >= actual else -0.35), actual + (-0.35 if pred >= actual else 0.35), 2.0 - 0.4],
                         "text": [f"{pred:.2f}%  scenario", f"{actual:.2f}%  actual", "2% target"],
                         "c": [BLUE, INK, GREEN]})
    labels = alt.Chart(ends).mark_text(fontWeight=600, fontSize=12, baseline="middle", align="left").encode(
        x="x:Q", y="y:Q", text="text:N", color=alt.Color("c:N", scale=None))
    marks = alt.Chart(pd.DataFrame({"x": [xr, xr], "y": [pred, actual], "c": [BLUE, INK]})).mark_point(
        filled=True, size=110).encode(x="x:Q", y="y:Q", color=alt.Color("c:N", scale=None))

    eps = [(2009, "Financial crisis", "below"), (2015, "Oil price crash", "below"), (2020, "COVID-19", "below"),
           (2022, "Reopening and Ukraine war", "above"), (2024, "Fed rate hikes", "hikes")]
    ep_rows = []
    for yr, txt, side in eps:
        m = hist[hist["Year"] == yr]
        if len(m):
            if side == "hikes":
                ep_rows.append({"x": yr + 0.3, "y": float(m["Inflation"].iloc[0]) + 1.5, "text": txt})
            else:
                ep_rows.append({"x": yr, "y": float(m["Inflation"].iloc[0]) + (0.75 if side == "above" else -0.6), "text": txt})
    ep = alt.Chart(pd.DataFrame(ep_rows)).mark_text(fontSize=11.5, color=MUTED).encode(
        x="x:Q", y="y:Q", text="text:N") if ep_rows else None

    layers = [rules, line, pts, marks, labels] + ([ep] if ep is not None else [])
    show(chart_style(alt.layer(*layers), 380))
    st.markdown("<div class='muted'>Models are fitted on 2003 to 2018 and tested from 2019.</div>", unsafe_allow_html=True)

# ============================================================ TAB 2
with tab2:
    era_labels = [e[0] for e in ERAS]
    chosen = st.multiselect("Eras", era_labels, default=era_labels)
    frames, era_rows = [], []
    for lab, s, e in ERAS:
        x = df.loc[s:e, ["unrate", target]].dropna()
        if len(x) > 2:
            era_rows.append((lab, len(x), float(np.polyfit(x["unrate"], x[target], 1)[0])))
        if lab in chosen:
            frames.append(x.assign(Era=lab).rename(columns={"unrate": "Unemployment", target: "Inflation"}))
    nrou = metrics["nrou"]
    if frames:
        allx = pd.concat(frames)
        slope, icpt = np.polyfit(allx["Unemployment"], allx["Inflation"], 1)
        x0, x1 = float(allx["Unemployment"].min()), float(allx["Unemployment"].max())
        palette = [GREEN, BLUE, AMBER, MUTED, RED]
        colors = [palette[era_labels.index(l) % 5] for l in chosen]
        pts2 = alt.Chart(allx.reset_index(drop=True)).mark_circle(size=34, opacity=.72).encode(
            x=alt.X("Unemployment:Q", axis=alt.Axis(title="Unemployment rate, %"), scale=alt.Scale(zero=False)),
            y=alt.Y("Inflation:Q", axis=alt.Axis(title=f"{target_label} inflation, % year over year")),
            color=alt.Color("Era:N", scale=alt.Scale(domain=chosen, range=colors), legend=alt.Legend(columns=3)),
            tooltip=["Era:N", alt.Tooltip("Unemployment:Q", format=".1f"), alt.Tooltip("Inflation:Q", format=".1f")])
        fit = alt.Chart(pd.DataFrame({"x": [x0, x1], "y": [icpt + slope * x0, icpt + slope * x1]})).mark_line(
            color=INK, strokeWidth=2.4).encode(x="x:Q", y="y:Q")
        band = alt.Chart(pd.DataFrame({"a": [nrou["min"]], "b": [nrou["max"]]})).mark_rect(color=GREEN, opacity=.09).encode(
            x="a:Q", x2="b:Q")
        vline = alt.Chart(pd.DataFrame({"x": [nrou["latest"]]})).mark_rule(color=GREEN, strokeWidth=2.2).encode(x="x:Q")
        ymax = float(allx["Inflation"].max())
        t1 = alt.Chart(pd.DataFrame({"x": [nrou["latest"] + 0.12], "y": [ymax],
                                     "t": [f"Long run: natural rate {nrou['latest']:.1f}%"]})).mark_text(
            fontWeight=600, fontSize=12, align="left", color=GREEN).encode(x="x:Q", y="y:Q", text="t:N")
        t2 = alt.Chart(pd.DataFrame({"x": [x1], "y": [icpt + slope * x1],
                                     "t": [f"Short-run fit, slope {slope:.2f}"]})).mark_text(
            fontWeight=600, fontSize=12, align="right", dy=-10, color=INK).encode(x="x:Q", y="y:Q", text="t:N")
        show(chart_style(alt.layer(band, pts2, vline, fit, t1, t2), 420))
    else:
        st.info("Select at least one era.")
    st.markdown("<h2 class='h'>Slope by era</h2>", unsafe_allow_html=True)
    body = "".join(f"<tr><td>{l}</td><td class='r'>{n}</td><td class='r'>{sl:+.2f}</td></tr>" for l, n, sl in era_rows)
    st.markdown(f"<table class='t'><tr><th>Era</th><th class='r'>Months</th><th class='r'>Slope</th></tr>{body}</table>",
                unsafe_allow_html=True)

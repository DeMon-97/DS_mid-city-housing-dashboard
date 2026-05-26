#!/usr/local/bin/python3.13
"""
Interactive dashboard — Case 10.2: Housing Price Structure in Mid City
Tab-based layout: Custom first, then Q1–Q4
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

st.set_page_config(page_title="Mid City Housing", page_icon="🏠", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f7f6f2; color: #1a1a1a; }
    [data-testid="stSidebar"] { background-color: #eeece6; }
    [data-testid="stSidebar"] * { color: #1a1a1a !important; }
    h1,h2,h3,h4,p,label,div { color: #1a1a1a; }
    [data-testid="stMetric"] { background-color: #edeae2; border-radius: 8px; padding: 12px; }
    [data-testid="stMetricValue"] { color: #1a1a1a !important; }
    [data-testid="stMetricLabel"] { color: #444 !important; }
    .stButton > button {
        background-color: #3b5470 !important; color: #fff !important;
        border: none !important; border-radius: 6px !important; font-weight: 600 !important;
    }
    .stButton > button:hover { background-color: #2d4058 !important; }
    .stButton > button p { color: #fff !important; }
    [data-testid="stDataFrame"] { background-color: #f7f6f2; }
    .insight-box {
        background-color: #edeae2; border-left: 4px solid #3b5470;
        border-radius: 6px; padding: 14px 18px; margin-top: 8px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #dedad2 !important; border-radius: 6px 6px 0 0 !important;
        color: #1a1a1a !important; font-weight: 600 !important; padding: 8px 18px !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b5470 !important; color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: #c8c4bb !important; }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH = "C10_02.xlsx"

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)
    df.rename(columns={"Sq Ft": "SqFt"}, inplace=True)
    df["Brick"]       = (df["Brick"] == "Yes").astype(int)
    df["Nbhd2"]       = (df["Nbhd"] == 2).astype(int)
    df["Nbhd3"]       = (df["Nbhd"] == 3).astype(int)
    df["Brick_Nbhd3"] = df["Brick"] * df["Nbhd3"]
    df["Newer"]       = df["Nbhd3"]
    return df

df = load_data()

BASE     = ["SqFt", "Offers", "Bedrooms", "Bathrooms"]
Q12_VARS = BASE + ["Brick", "Nbhd2", "Nbhd3"]
Q3_VARS  = BASE + ["Brick", "Nbhd2", "Nbhd3", "Brick_Nbhd3"]
Q4C_VARS = BASE + ["Brick", "Newer"]
ALL_VARS = {
    "SqFt": "Square Footage", "Offers": "Number of Offers",
    "Bedrooms": "Bedrooms", "Bathrooms": "Bathrooms",
    "Brick": "Brick Construction", "Nbhd2": "Neighborhood 2 dummy",
    "Nbhd3": "Neighborhood 3 dummy", "Brick_Nbhd3": "Brick × Nbhd 3 Interaction",
    "Newer": "Newer (Nbhd 3 collapsed)",
}

# ── Chart style constants ─────────────────────────────────────────────────────
CS = dict(plot_bgcolor="#f7f6f2", paper_bgcolor="#f7f6f2", font=dict(color="#1a1a1a"))
AX = dict(gridcolor="#d0ccc0", linecolor="#999690", linewidth=1,
          tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a"))
LG = dict(font=dict(color="#1a1a1a"), bgcolor="rgba(247,246,242,0.8)",
          bordercolor="#999690", borderwidth=1)
BD = [dict(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1,
           line=dict(color="#999690", width=1))]

# ── Shared helpers ────────────────────────────────────────────────────────────
def latex_var(n): return n.replace("_", r"\_")

def regression_equation(model, selected):
    intercept = model.params["const"]
    terms = [f"{'+'if c>=0 else'-'} {abs(c):,.0f}\\cdot\\text{{{latex_var(v)}}}"
             for v in selected for c in [model.params[v]]]
    return r"\widehat{\text{Price}} = " + f"{intercept:,.0f} " + " ".join(terms)

def coef_plot(plot_df, height=None):
    fig = go.Figure()
    for sig, color, label in [(True, "#2ca02c", "Significant"), (False, "#5b8db8", "Not significant")]:
        sub = plot_df[plot_df["Significant"] == sig]
        if sub.empty: continue
        fig.add_trace(go.Scatter(
            x=sub["Coefficient"], y=sub.index, mode="markers", name=label,
            marker=dict(color=color, size=10),
            error_x=dict(type="data", symmetric=False,
                         array=(sub["CI Upper"]-sub["Coefficient"]).values,
                         arrayminus=(sub["Coefficient"]-sub["CI Lower"]).values,
                         color=color, thickness=2, width=7),
            customdata=sub["p-value"].values,
            hovertemplate="<b>%{y}</b><br>Coef: $%{x:,.0f}<br>p: %{customdata:.3f}<extra></extra>",
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="#555", line_width=1)
    fig.update_layout(
        height=height or max(260, len(plot_df)*52),
        xaxis_title="Coefficient ($)", yaxis_title="Variable",
        xaxis=AX, yaxis={**AX, "gridcolor": "rgba(0,0,0,0)"},
        legend={**LG, "title": dict(text="Significance", font=dict(color="#1a1a1a"))},
        **CS, margin=dict(l=150, r=20, t=15, b=35), shapes=BD,
    )
    return fig

def avf_plot(model):
    fitted = model.fittedvalues
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fitted, y=df["Price"], mode="markers",
                             marker=dict(color="#4C72B0", opacity=0.55, size=6), name="Homes",
                             hovertemplate="Fitted: $%{x:,.0f}<br>Actual: $%{y:,.0f}<extra></extra>"))
    lim = [min(fitted.min(), df["Price"].min())*0.95, max(fitted.max(), df["Price"].max())*1.05]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                             line=dict(color="red", dash="dash", width=1.5), name="Perfect fit"))
    fig.update_layout(
        height=300, xaxis_title="Fitted ($)", yaxis_title="Actual ($)",
        xaxis={**AX, "tickformat":"$,.0f"}, yaxis={**AX, "tickformat":"$,.0f"},
        legend={**LG, "x":0.05, "y":0.95}, **CS,
        margin=dict(t=15, b=35, l=70, r=15), shapes=BD,
    )
    return fig

def model_cdf(model):
    cdf = pd.DataFrame({
        "Coefficient": model.params, "p-value": model.pvalues,
        "CI Lower": model.conf_int()[0], "CI Upper": model.conf_int()[1],
    }).drop("const")
    cdf["Significant"] = cdf["p-value"] < sig_level
    return cdf

def insight_box(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def fit_table(model):
    cdf = pd.DataFrame({
        "Coefficient": model.params, "Std Error": model.bse,
        "t-stat": model.tvalues, "p-value": model.pvalues,
        "CI Lower": model.conf_int()[0], "CI Upper": model.conf_int()[1],
    })
    cdf.loc["const","p-value"] = float("nan")
    def sp(v): return "" if pd.isna(v) else ("color:#1a7a1a;font-weight:bold" if v<sig_level else "color:#cc0000")
    return cdf.style.map(sp, subset=["p-value"]).format(
        {"Coefficient":"${:,.0f}","Std Error":"${:,.0f}","t-stat":"{:.3f}",
         "p-value":"{:.3f}","CI Lower":"${:,.0f}","CI Upper":"${:,.0f}"}, na_rep="—")

# ── Sidebar: controls + price predictor ───────────────────────────────────────
st.sidebar.title("🏠 Controls")
sig_level = st.sidebar.slider("Significance level (α)", 0.01, 1.00, 0.05, 0.01)

# Pre-compute the three canonical models
m1 = sm.OLS(df["Price"], sm.add_constant(df[Q12_VARS])).fit()   # Q1, Q2, Q4
m2 = sm.OLS(df["Price"], sm.add_constant(df[Q3_VARS])).fit()    # Q3 (with interaction)
m3 = sm.OLS(df["Price"], sm.add_constant(df[Q4C_VARS])).fit()   # Q4 collapsed
avg_price = df["Price"].mean()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧮 Price Predictor")
st.sidebar.caption("Adjust inputs and get a live predicted price using the full regression model.")
p_sqft   = st.sidebar.slider("Square Footage",    int(df["SqFt"].min()),     int(df["SqFt"].max()),     int(df["SqFt"].median()),     50)
p_offers = st.sidebar.slider("Number of Offers",  int(df["Offers"].min()),   int(df["Offers"].max()),   int(df["Offers"].median()))
p_beds   = st.sidebar.slider("Bedrooms",          int(df["Bedrooms"].min()), int(df["Bedrooms"].max()), 3)
p_baths  = st.sidebar.slider("Bathrooms",         int(df["Bathrooms"].min()),int(df["Bathrooms"].max()),2)
p_brick  = st.sidebar.radio("Brick Construction", ["No", "Yes"], horizontal=True)
p_nbhd   = st.sidebar.radio("Neighborhood",       [1, 2, 3], horizontal=True)

pb = 1 if p_brick == "Yes" else 0
pn2 = 1 if p_nbhd == 2 else 0
pn3 = 1 if p_nbhd == 3 else 0

def predict_m1(sqft, offers, beds, baths, brick, nbhd2, nbhd3):
    xp = pd.DataFrame([[1, sqft, offers, beds, baths, brick, nbhd2, nbhd3]],
                       columns=["const"] + Q12_VARS)
    return m1.predict(xp)[0]

pred_price     = predict_m1(p_sqft, p_offers, p_beds, p_baths, pb, pn2, pn3)
pred_no_brick  = predict_m1(p_sqft, p_offers, p_beds, p_baths, 0,  pn2, pn3)
pred_yes_brick = predict_m1(p_sqft, p_offers, p_beds, p_baths, 1,  pn2, pn3)

st.sidebar.markdown(f"""
<div style="background:#edeae2;border-left:4px solid #3b5470;border-radius:6px;padding:10px 14px;margin-top:6px">
<b>Predicted Price</b><br>
<span style="font-size:1.5em;font-weight:bold;color:#3b5470">${pred_price:,.0f}</span><br>
<small>🧱 Without brick: <b>${pred_no_brick:,.0f}</b></small><br>
<small>🏠 With brick: <b>${pred_yes_brick:,.0f}</b></small><br>
<small>Brick premium: <b>${pred_yes_brick - pred_no_brick:,.0f}</b></small>
</div>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Case 10.2 — Housing Price Structure in Mid City")
st.markdown("*Albright & Winston, Business Analytics (7e)*")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_custom, tab_q1, tab_q2, tab_q3, tab_q4 = st.tabs([
    "🔧 Custom Explorer",
    "Q1 — Brick Premium",
    "Q2 — Nbhd 3 Premium",
    "Q3 — Brick × Nbhd 3",
    "Q4 — Collapse Nbhds",
])

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM
# ══════════════════════════════════════════════════════════════════════════════
with tab_custom:
    st.subheader("Custom Model Explorer")
    st.markdown("Pick any variables below and build your own regression. Use the sidebar to predict a price.")

    selected = st.multiselect(
        "Variables to include in the model:",
        options=list(ALL_VARS.keys()),
        default=BASE + ["Brick", "Nbhd2", "Nbhd3"],
        format_func=lambda v: ALL_VARS[v],
    )

    if not selected:
        st.warning("Select at least one variable above.")
    else:
        cust_model = sm.OLS(df["Price"], sm.add_constant(df[selected])).fit()

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("R²",           f"{cust_model.rsquared:.4f}")
        cc2.metric("Adj R²",       f"{cust_model.rsquared_adj:.4f}")
        cc3.metric("F p-value",    f"{cust_model.f_pvalue:.3e}")
        cc4.metric("Observations", len(df))

        st.markdown("##### Fitted Equation")
        st.latex(regression_equation(cust_model, selected))

        st.markdown("##### Coefficient Table")
        st.dataframe(fit_table(cust_model), use_container_width=True)
        st.caption(f"Green = significant at α = {sig_level} | Red = not significant | — = not tested for constant")

        cp_col, avf_col = st.columns([1, 1], gap="medium")
        with cp_col:
            st.markdown("##### Coefficient Plot (95% CI)")
            cust_cdf = pd.DataFrame({
                "Coefficient": cust_model.params, "p-value": cust_model.pvalues,
                "CI Lower": cust_model.conf_int()[0], "CI Upper": cust_model.conf_int()[1],
            }).drop("const")
            cust_cdf["Significant"] = cust_cdf["p-value"] < sig_level
            st.plotly_chart(coef_plot(cust_cdf.sort_values("Coefficient")), use_container_width=True, key="cust_coef")
        with avf_col:
            st.markdown("##### Actual vs Fitted")
            st.plotly_chart(avf_plot(cust_model), use_container_width=True, key="cust_avf")

        st.markdown("##### Q1–Q4 Answers from this Model")

        def ab(q, ans, sig, detail=""):
            st.markdown(f"**{q}** {'✅' if sig else '❌'} {ans}")
            if detail: st.caption(detail)

        if "Brick" in selected:
            c, p = cust_model.params.get("Brick"), cust_model.pvalues.get("Brick")
            if c is not None: ab("Q1 — Brick premium?", f"${c:,.0f} (p = {p:.3f})", p < sig_level)
        if "Nbhd3" in selected:
            c, p = cust_model.params.get("Nbhd3"), cust_model.pvalues.get("Nbhd3")
            if c is not None: ab("Q2 — Nbhd 3 premium?", f"${c:,.0f} (p = {p:.3f})", p < sig_level)
        if "Brick_Nbhd3" in selected:
            c, p = cust_model.params.get("Brick_Nbhd3"), cust_model.pvalues.get("Brick_Nbhd3")
            if c is not None: ab("Q3 — Interaction?", f"${c:,.0f} (p = {p:.3f})", p < sig_level)
        else:
            st.markdown("**Q3** ⚠️ Add *Brick × Nbhd 3 Interaction* to answer this question.")
        if "Nbhd2" in selected and "Nbhd3" in selected:
            try:
                Xr  = sm.add_constant(df[[v for v in selected if v != "Nbhd2"]])
                mr  = sm.OLS(df["Price"], Xr).fit()
                fs  = ((mr.ssr - cust_model.ssr) / 1) / (cust_model.ssr / cust_model.df_resid)
                fp  = 1 - stats.f.cdf(fs, 1, cust_model.df_resid)
                ab("Q4 — Collapse Nbhds 1 & 2?",
                   f"Nbhd2 = ${cust_model.params['Nbhd2']:,.0f} (p = {cust_model.pvalues['Nbhd2']:.3f})  |  Partial F p = {fp:.3f}",
                   fp > sig_level)
            except Exception:
                pass

        with st.expander("🧮 Try It — Predict Price for Any House"):
            st.caption("Uses the sidebar inputs (SqFt, Offers, Bedrooms, Bathrooms, Brick, Neighborhood).")
            val_map = {"SqFt": p_sqft, "Offers": p_offers, "Bedrooms": p_beds,
                       "Bathrooms": p_baths, "Brick": pb, "Nbhd2": pn2,
                       "Nbhd3": pn3, "Brick_Nbhd3": pb * pn3, "Newer": pn3}
            xp_cust = pd.DataFrame(
                [[1] + [val_map.get(v, 0) for v in selected]],
                columns=["const"] + selected)
            try:
                p_cust = cust_model.predict(xp_cust)[0]
                conf   = cust_model.get_prediction(xp_cust).summary_frame(alpha=sig_level)
                ex1, ex2, ex3 = st.columns(3)
                ex1.metric("Point Estimate",   f"${p_cust:,.0f}")
                ex2.metric(f"Lower {int((1-sig_level)*100)}% CI", f"${conf['obs_ci_lower'].iloc[0]:,.0f}")
                ex3.metric(f"Upper {int((1-sig_level)*100)}% CI", f"${conf['obs_ci_upper'].iloc[0]:,.0f}")
            except Exception as e:
                st.warning(f"Prediction unavailable for this variable combination: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# Q1 — Brick Premium
# ══════════════════════════════════════════════════════════════════════════════
with tab_q1:
    st.subheader("Q1 — Do buyers pay a premium for a brick house?")
    st.info("Model 1 controls for size, location, and demand. The Brick coefficient isolates the price premium attributable solely to brick construction.")

    brick_coef = m1.params["Brick"];  brick_pval = m1.pvalues["Brick"]
    brick_sig  = brick_pval < sig_level
    brick_pct  = abs(brick_coef) / avg_price * 100
    mean_brick    = df[df["Brick"] == 1]["Price"].mean()
    mean_no_brick = df[df["Brick"] == 0]["Price"].mean()
    raw_diff      = mean_brick - mean_no_brick

    col_l, col_r = st.columns([1, 1], gap="medium")

    with col_l:
        st.markdown("##### Price: Brick vs No Brick")
        fig = go.Figure()
        for val, lbl, c in [(1, "Brick", "#2ca02c"), (0, "No Brick", "#DD8452")]:
            fig.add_trace(go.Box(y=df[df["Brick"] == val]["Price"], name=lbl,
                                 marker_color=c, line_color=c,
                                 hovertemplate="Price: $%{y:,.0f}<extra></extra>"))
        fig.update_layout(yaxis_title="Price ($)", xaxis=AX, yaxis={**AX, "tickformat": "$,.0f"},
                          legend=LG, **CS, height=280,
                          margin=dict(t=10, b=35, l=70, r=10), shapes=BD)
        st.plotly_chart(fig, use_container_width=True, key="q1_brick_box")

        bstats = df.groupby("Brick")["Price"].agg(Count="count", Mean="mean", Median="median", StdDev="std")\
                   .rename(index={0: "No Brick", 1: "Brick"})
        st.dataframe(bstats.style.format(
            {"Count": "{:.0f}", "Mean": "${:,.0f}", "Median": "${:,.0f}", "StdDev": "${:,.0f}"}),
            use_container_width=True)

    with col_r:
        st.markdown("##### Model Fit")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("R²",      f"{m1.rsquared:.4f}")
        mc2.metric("Adj R²",  f"{m1.rsquared_adj:.4f}")
        mc3.metric("F p-val", f"{m1.f_pvalue:.3e}")

        st.markdown("##### Fitted Equation")
        st.latex(regression_equation(m1, Q12_VARS))

        st.markdown("##### Brick Coefficient")
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Coefficient",  f"${brick_coef:,.0f}")
        bc2.metric("p-value",      f"{brick_pval:.3f}")
        bc3.metric("Significant?", "Yes ✅" if brick_sig else "No ❌")

    cp_col, avf_col = st.columns([1, 1], gap="medium")
    with cp_col:
        st.markdown("##### Coefficient Plot (95% CI)")
        st.plotly_chart(coef_plot(model_cdf(m1).sort_values("Coefficient")), use_container_width=True, key="q1_coef")
    with avf_col:
        st.markdown("##### Actual vs Fitted")
        st.plotly_chart(avf_plot(m1), use_container_width=True, key="q1_avf")

    if brick_sig:
        st.success(f"**Q1 Answer ✅** Buyers pay a significant brick premium of **${brick_coef:,.0f}** (p = {brick_pval:.3f}).")
    else:
        st.warning(f"**Q1 Answer ❌** Brick premium of **${brick_coef:,.0f}** is NOT significant (p = {brick_pval:.3f}) at α = {sig_level}.")

    insight_box(f"""
    <b>Key Insights</b><br>
    • Raw mean difference (unadjusted): brick homes average <b>${mean_brick:,.0f}</b> vs <b>${mean_no_brick:,.0f}</b> — a gap of <b>${raw_diff:,.0f}</b>.<br>
    • After controlling for size, location, and demand, the regression estimates the premium at <b>${brick_coef:,.0f}</b> ({brick_pct:.1f}% of the average price of ${avg_price:,.0f}).<br>
    • The difference between the raw gap and the regression coefficient shows how much is explained by other factors (e.g., brick homes tend to be larger).<br>
    • {'This premium is statistically reliable — brick construction adds genuine value independent of house size or neighborhood.' if brick_sig else 'The premium disappears once we control for other factors, suggesting brick homes are more expensive for reasons other than the brick itself.'}
    """)

    with st.expander("🏠 Try It — Brick vs No Brick for Your House"):
        st.caption("Adjust house inputs in the sidebar to personalise the comparison.")
        ti1, ti2, ti3 = st.columns(3)
        ti1.metric("Without Brick",  f"${pred_no_brick:,.0f}")
        ti2.metric("With Brick",     f"${pred_yes_brick:,.0f}")
        ti3.metric("Brick Premium",  f"${pred_yes_brick - pred_no_brick:,.0f}",
                   delta=f"{(pred_yes_brick - pred_no_brick) / pred_no_brick * 100:.1f}%")
        fig_bv = go.Figure(go.Bar(
            x=["No Brick", "Brick"], y=[pred_no_brick, pred_yes_brick],
            marker_color=["#DD8452", "#2ca02c"],
            text=[f"${pred_no_brick:,.0f}", f"${pred_yes_brick:,.0f}"],
            textposition="outside",
            hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
        ))
        fig_bv.update_layout(yaxis_title="Predicted Price ($)", xaxis=AX,
                             yaxis={**AX, "tickformat": "$,.0f", "range": [pred_no_brick * 0.85, pred_yes_brick * 1.1]},
                             **CS, height=240, margin=dict(t=10, b=35, l=70, r=10), shapes=BD)
        st.plotly_chart(fig_bv, use_container_width=True, key="q1_bv")

# ══════════════════════════════════════════════════════════════════════════════
# Q2 — Neighborhood 3 Premium
# ══════════════════════════════════════════════════════════════════════════════
with tab_q2:
    st.subheader("Q2 — Is there a premium for Neighborhood 3?")
    st.info("Neighborhood 1 is the baseline. The Nbhd3 coefficient shows the price premium for Nbhd 3 over Nbhd 1, holding all else constant.")

    nbhd2_coef = m1.params["Nbhd2"]; nbhd2_pval = m1.pvalues["Nbhd2"]
    nbhd3_coef = m1.params["Nbhd3"]; nbhd3_pval = m1.pvalues["Nbhd3"]
    nbhd3_sig  = nbhd3_pval < sig_level
    nbhd3_pct  = abs(nbhd3_coef) / avg_price * 100
    mean_by_nbhd = df.groupby("Nbhd")["Price"].mean()

    col_l, col_r = st.columns([1, 1], gap="medium")

    with col_l:
        st.markdown("##### Price by Neighborhood")
        fig = go.Figure()
        for nbhd, lbl, c in [(1, "Nbhd 1 (Older)", "#4C72B0"),
                              (2, "Nbhd 2 (Older)", "#5ba3c9"),
                              (3, "Nbhd 3 (Newer)", "#DD8452")]:
            fig.add_trace(go.Box(y=df[df["Nbhd"] == nbhd]["Price"], name=lbl,
                                 marker_color=c, line_color=c,
                                 hovertemplate="Price: $%{y:,.0f}<extra></extra>"))
        fig.update_layout(yaxis_title="Price ($)", xaxis=AX, yaxis={**AX, "tickformat": "$,.0f"},
                          legend=LG, **CS, height=280,
                          margin=dict(t=10, b=35, l=70, r=10), shapes=BD)
        st.plotly_chart(fig, use_container_width=True, key="q2_nbhd_box")

        nstats = df.groupby("Nbhd")["Price"].agg(Count="count", Mean="mean", Median="median", StdDev="std")\
                    .rename(index={1: "Nbhd 1", 2: "Nbhd 2", 3: "Nbhd 3"})
        st.dataframe(nstats.style.format(
            {"Count": "{:.0f}", "Mean": "${:,.0f}", "Median": "${:,.0f}", "StdDev": "${:,.0f}"}),
            use_container_width=True)

    with col_r:
        st.markdown("##### Model Fit")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("R²",      f"{m1.rsquared:.4f}")
        mc2.metric("Adj R²",  f"{m1.rsquared_adj:.4f}")
        mc3.metric("F p-val", f"{m1.f_pvalue:.3e}")

        st.markdown("##### Fitted Equation")
        st.latex(regression_equation(m1, Q12_VARS))

        st.markdown("##### Neighborhood Coefficients (vs Nbhd 1 baseline)")
        nr1, nr2 = st.columns(2)
        nr1.metric("Nbhd 2 coef", f"${nbhd2_coef:,.0f}", delta=f"p = {nbhd2_pval:.3f}")
        nr2.metric("Nbhd 3 coef", f"${nbhd3_coef:,.0f}", delta=f"p = {nbhd3_pval:.3f}")
        st.metric("Nbhd 3 Significant?", "Yes ✅" if nbhd3_sig else "No ❌")

    cp_col, avf_col = st.columns([1, 1], gap="medium")
    with cp_col:
        st.markdown("##### Coefficient Plot (95% CI)")
        st.plotly_chart(coef_plot(model_cdf(m1).sort_values("Coefficient")), use_container_width=True, key="q2_coef")
    with avf_col:
        st.markdown("##### Actual vs Fitted")
        st.plotly_chart(avf_plot(m1), use_container_width=True, key="q2_avf")

    if nbhd3_sig:
        st.success(f"**Q2 Answer ✅** Nbhd 3 commands a significant premium of **${nbhd3_coef:,.0f}** over Nbhd 1 (p = {nbhd3_pval:.3f}).")
    else:
        st.warning(f"**Q2 Answer ❌** Nbhd 3 premium of **${nbhd3_coef:,.0f}** is NOT significant (p = {nbhd3_pval:.3f}) at α = {sig_level}.")

    insight_box(f"""
    <b>Key Insights</b><br>
    • Raw mean prices: Nbhd 1 = <b>${mean_by_nbhd[1]:,.0f}</b>, Nbhd 2 = <b>${mean_by_nbhd[2]:,.0f}</b>, Nbhd 3 = <b>${mean_by_nbhd[3]:,.0f}</b>.<br>
    • After controlling for house characteristics, Nbhd 3 carries a regression premium of <b>${nbhd3_coef:,.0f}</b> ({nbhd3_pct:.1f}% of average price) over Nbhd 1.<br>
    • <b>Nbhd 2's coefficient is not significant</b> (p = {nbhd2_pval:.3f}) — its prices are statistically indistinguishable from Nbhd 1 once other factors are controlled. This is a preview of Q4.<br>
    • {'The Nbhd 3 premium is real — buyers are paying for the neighborhood itself, not just larger or better-constructed homes.' if nbhd3_sig else 'Once house characteristics are accounted for, the Nbhd 3 premium disappears.'}
    """)

    with st.expander("🏘️ Try It — Compare Predicted Price Across Neighborhoods"):
        st.caption("Adjust house inputs in the sidebar. The chart shows how moving neighborhoods changes the predicted price.")
        preds_nbhd = {}
        for n in [1, 2, 3]:
            preds_nbhd[n] = predict_m1(p_sqft, p_offers, p_beds, p_baths,
                                        pb, 1 if n == 2 else 0, 1 if n == 3 else 0)
        nb1, nb2, nb3 = st.columns(3)
        nb1.metric("Nbhd 1 (Baseline)", f"${preds_nbhd[1]:,.0f}")
        nb2.metric("Nbhd 2",            f"${preds_nbhd[2]:,.0f}",
                   delta=f"${preds_nbhd[2]-preds_nbhd[1]:+,.0f} vs Nbhd 1")
        nb3.metric("Nbhd 3",            f"${preds_nbhd[3]:,.0f}",
                   delta=f"${preds_nbhd[3]-preds_nbhd[1]:+,.0f} vs Nbhd 1")

        fig_nb = go.Figure(go.Bar(
            x=["Nbhd 1 (Baseline)", "Nbhd 2", "Nbhd 3"],
            y=[preds_nbhd[1], preds_nbhd[2], preds_nbhd[3]],
            marker_color=["#4C72B0", "#5ba3c9", "#DD8452"],
            text=[f"${v:,.0f}" for v in preds_nbhd.values()],
            textposition="outside",
            hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
        ))
        fig_nb.update_layout(
            yaxis_title="Predicted Price ($)", xaxis=AX,
            yaxis={**AX, "tickformat": "$,.0f",
                   "range": [min(preds_nbhd.values()) * 0.9, max(preds_nbhd.values()) * 1.1]},
            **CS, height=260, margin=dict(t=10, b=35, l=70, r=10), shapes=BD)
        st.plotly_chart(fig_nb, use_container_width=True, key="q2_nb")

# ══════════════════════════════════════════════════════════════════════════════
# Q3 — Brick × Nbhd 3 Interaction
# ══════════════════════════════════════════════════════════════════════════════
with tab_q3:
    st.subheader("Q3 — Is there an extra brick premium in Neighborhood 3?")
    st.info("Model 2 adds a Brick × Nbhd 3 interaction. If significant, being brick AND in Nbhd 3 generates an extra premium beyond the two effects combined.")

    brick_coef3 = m2.params["Brick"];       nbhd3_coef3 = m2.params["Nbhd3"]
    int_coef    = m2.params["Brick_Nbhd3"]; int_pval    = m2.pvalues["Brick_Nbhd3"]
    int_sig     = int_pval < sig_level
    total_nb3_brick = brick_coef3 + nbhd3_coef3 + int_coef

    interaction  = df.groupby(["Nbhd", "Brick"])["Price"].mean().reset_index()
    interaction["Group"]     = interaction["Brick"].map({1: "Brick", 0: "No Brick"})
    interaction["NbhdLabel"] = interaction["Nbhd"].map({1: "Nbhd 1", 2: "Nbhd 2", 3: "Nbhd 3"})
    pivot = df.pivot_table(values="Price", index="Nbhd", columns="Brick", aggfunc="mean")
    pivot.index   = ["Nbhd 1", "Nbhd 2", "Nbhd 3"]
    pivot.columns = ["No Brick", "Brick"]
    pivot["Brick Premium ($)"] = pivot["Brick"] - pivot["No Brick"]

    col_l, col_r = st.columns([1, 1], gap="medium")

    with col_l:
        st.markdown("##### Interaction: Mean Price by Brick × Neighborhood")
        fig = go.Figure()
        for bt, c in [("Brick", "#2ca02c"), ("No Brick", "#DD8452")]:
            grp = interaction[interaction["Group"] == bt]
            fig.add_trace(go.Scatter(
                x=grp["NbhdLabel"], y=grp["Price"], mode="lines+markers", name=bt,
                marker=dict(size=10, color=c), line=dict(color=c, width=2.5),
                hovertemplate="%{x}: $%{y:,.0f}<extra>" + bt + "</extra>"))
        fig.update_layout(xaxis_title="Neighborhood", yaxis_title="Mean Price ($)",
                          xaxis=AX, yaxis={**AX, "tickformat": "$,.0f"},
                          legend={**LG, "title": dict(text="House Type", font=dict(color="#1a1a1a"))},
                          **CS, height=280, margin=dict(t=10, b=35, l=70, r=10), shapes=BD)
        st.plotly_chart(fig, use_container_width=True, key="q3_interaction")
        st.caption("Diverging lines toward Nbhd 3 signal an interaction effect.")
        st.dataframe(pivot.style.format("${:,.0f}"), use_container_width=True)

    with col_r:
        st.markdown("##### Model Fit")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("R²",      f"{m2.rsquared:.4f}")
        mc2.metric("Adj R²",  f"{m2.rsquared_adj:.4f}")
        mc3.metric("F p-val", f"{m2.f_pvalue:.3e}")

        st.markdown("##### Fitted Equation")
        st.latex(regression_equation(m2, Q3_VARS))

        st.markdown("##### Main Effects")
        me1, me2 = st.columns(2)
        me1.metric("Brick (main)",   f"${brick_coef3:,.0f}")
        me2.metric("Nbhd 3 (main)",  f"${nbhd3_coef3:,.0f}")

        st.markdown("##### Interaction Term")
        it1, it2, it3 = st.columns(3)
        it1.metric("Interaction coef", f"${int_coef:,.0f}")
        it2.metric("p-value",          f"{int_pval:.3f}")
        it3.metric("Significant?",     "Yes ✅" if int_sig else "No ❌")

        if int_sig:
            st.markdown("##### Total premium — brick house in Nbhd 3")
            st.latex(f"\\${brick_coef3:,.0f} + \\${nbhd3_coef3:,.0f} + \\${int_coef:,.0f} = \\${total_nb3_brick:,.0f}")

    cp_col, avf_col = st.columns([1, 1], gap="medium")
    with cp_col:
        st.markdown("##### Coefficient Plot (95% CI)")
        st.plotly_chart(coef_plot(model_cdf(m2).sort_values("Coefficient")), use_container_width=True, key="q3_coef")
    with avf_col:
        st.markdown("##### Actual vs Fitted")
        st.plotly_chart(avf_plot(m2), use_container_width=True, key="q3_avf")

    if int_sig:
        st.success(f"**Q3 Answer ✅** Interaction significant (p = {int_pval:.3f}). Extra brick premium in Nbhd 3 = **${int_coef:,.0f}**. Total brick-in-Nbhd-3 premium = **${total_nb3_brick:,.0f}**.")
    else:
        st.warning(f"**Q3 Answer ❌** Interaction NOT significant (p = {int_pval:.3f}). Brick and Nbhd 3 effects are simply additive — no extra synergy premium.")

    raw_premiums = pivot["Brick Premium ($)"]
    insight_box(f"""
    <b>Key Insights</b><br>
    • Raw brick premiums: Nbhd 1 = <b>${raw_premiums.iloc[0]:,.0f}</b>, Nbhd 2 = <b>${raw_premiums.iloc[1]:,.0f}</b>, Nbhd 3 = <b>${raw_premiums.iloc[2]:,.0f}</b>.<br>
    • {'The interaction is significant — the brick premium in Nbhd 3 is larger than in other neighborhoods.' if int_sig else 'The interaction is not significant — the brick premium is consistent across all neighborhoods.'}<br>
    • The regression tests whether the Nbhd 3 brick premium survives after controlling for other house characteristics.<br>
    • Model 2 (with interaction) Adj R² = <b>{m2.rsquared_adj:.4f}</b> vs Model 1 Adj R² = <b>{m1.rsquared_adj:.4f}</b>. A marginal improvement suggests limited practical value from the interaction.
    """)

    with st.expander("🧩 Try It — Premium Breakdown by Neighborhood & Brick"):
        st.caption("Select a combination below to see how each effect contributes to the predicted price (using sidebar house inputs).")
        ex_col1, ex_col2 = st.columns(2)
        sel_nbhd  = ex_col1.radio("Neighborhood", [1, 2, 3], horizontal=True, key="q3_nbhd")
        sel_brick = ex_col2.radio("Brick?", ["No", "Yes"], horizontal=True, key="q3_brick")
        sb  = 1 if sel_brick == "Yes" else 0
        sn3 = 1 if sel_nbhd == 3 else 0
        sn2 = 1 if sel_nbhd == 2 else 0

        base   = (m2.params["const"] + m2.params["SqFt"] * p_sqft +
                  m2.params["Offers"] * p_offers + m2.params["Bedrooms"] * p_beds +
                  m2.params["Bathrooms"] * p_baths)
        comps  = {
            "Base (size & demand)":        base,
            "Brick effect":                m2.params["Brick"] * sb,
            "Nbhd 2 effect":               m2.params["Nbhd2"] * sn2,
            "Nbhd 3 effect":               m2.params["Nbhd3"] * sn3,
            "Interaction (Brick×Nbhd3)":   m2.params["Brick_Nbhd3"] * sb * sn3,
        }
        comps = {k: v for k, v in comps.items() if abs(v) > 1}
        total = sum(comps.values())

        fig_dc = go.Figure(go.Bar(
            x=list(comps.keys()), y=list(comps.values()),
            marker_color=["#4C72B0", "#2ca02c", "#5ba3c9", "#DD8452", "#9467bd"][:len(comps)],
            text=[f"${v:,.0f}" for v in comps.values()],
            textposition="outside",
            hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
        ))
        fig_dc.update_layout(
            title=dict(text=f"Predicted: ${total:,.0f}  ({sel_brick} brick, Nbhd {sel_nbhd})",
                       font=dict(color="#1a1a1a", size=13)),
            yaxis_title="Price Component ($)", xaxis=AX,
            yaxis={**AX, "tickformat": "$,.0f"},
            **CS, height=280, margin=dict(t=40, b=35, l=70, r=10), shapes=BD)
        st.plotly_chart(fig_dc, use_container_width=True, key="q3_dc")

# ══════════════════════════════════════════════════════════════════════════════
# Q4 — Collapse Neighborhoods 1 & 2
# ══════════════════════════════════════════════════════════════════════════════
with tab_q4:
    st.subheader("Q4 — Can Neighborhoods 1 and 2 be treated as one group?")
    st.info("ANOVA tests raw (unadjusted) mean prices. OLS regression then controls for house characteristics and tests whether a genuine neighborhood difference remains.")

    f_stat_pf = ((m3.ssr - m1.ssr) / 1) / (m1.ssr / m1.df_resid)
    f_pval_pf = 1 - stats.f.cdf(f_stat_pf, 1, m1.df_resid)
    can_collapse = f_pval_pf > sig_level
    f_anova, p_anova = stats.f_oneway(
        df[df["Nbhd"] == 1]["Price"], df[df["Nbhd"] == 2]["Price"], df[df["Nbhd"] == 3]["Price"])
    nbhd2_coef4  = m1.params["Nbhd2"]; nbhd2_pval4 = m1.pvalues["Nbhd2"]
    mean_by_nbhd = df.groupby("Nbhd")["Price"].mean()
    adj_r2_drop  = abs(m1.rsquared_adj - m3.rsquared_adj)

    nbhd_ci = {}
    for n in [1, 2, 3]:
        grp  = df[df["Nbhd"] == n]["Price"]
        mn   = grp.mean()
        se   = grp.sem()
        t95  = stats.t.ppf(1 - sig_level / 2, df=len(grp) - 1)
        nbhd_ci[n] = (mn, se, mn - t95 * se, mn + t95 * se)

    # ── ANOVA ────────────────────────────────────────────────────────────────
    st.markdown("#### Step 1 — One-Way ANOVA")
    st.markdown("Tests whether all three neighborhoods have equal mean prices. A significant result means at least one group differs.")

    av1, av2, av3 = st.columns(3)
    av1.metric("F-statistic",  f"{f_anova:.3f}")
    av2.metric("p-value",      f"{p_anova:.3e}")
    av3.metric("ANOVA Result", "Significant ✅" if p_anova < sig_level else "Not Significant ❌")

    box_col, means_col = st.columns([1, 1], gap="medium")

    with box_col:
        st.markdown("##### Price Distribution by Neighborhood")
        fig_box = go.Figure()
        for nbhd, lbl, c in [(1, "Nbhd 1 (Older)", "#4C72B0"),
                              (2, "Nbhd 2 (Older)", "#5ba3c9"),
                              (3, "Nbhd 3 (Newer)", "#DD8452")]:
            fig_box.add_trace(go.Box(y=df[df["Nbhd"] == nbhd]["Price"], name=lbl,
                                     marker_color=c, line_color=c,
                                     hovertemplate="Price: $%{y:,.0f}<extra></extra>"))
        fig_box.update_layout(
            yaxis_title="Price ($)", xaxis=AX, yaxis={**AX, "tickformat": "$,.0f"},
            legend=LG, **CS, height=300, margin=dict(t=10, b=35, l=70, r=10), shapes=BD)
        st.plotly_chart(fig_box, use_container_width=True, key="q4_box")

    with means_col:
        st.markdown("##### Means Plot with 95% Confidence Intervals")
        colors_ci = {1: "#4C72B0", 2: "#5ba3c9", 3: "#DD8452"}
        nbhd_lbl  = {1: "Nbhd 1", 2: "Nbhd 2", 3: "Nbhd 3"}
        fig_ci = go.Figure()
        for n in [1, 2, 3]:
            mn, se, ci_lo, ci_hi = nbhd_ci[n]
            c = colors_ci[n]
            fig_ci.add_trace(go.Scatter(
                x=[nbhd_lbl[n]], y=[mn], mode="markers", name=nbhd_lbl[n],
                marker=dict(color=c, size=14),
                error_y=dict(type="data", symmetric=False,
                             array=[ci_hi - mn], arrayminus=[mn - ci_lo],
                             color=c, thickness=2.5, width=10),
                hovertemplate=(f"<b>{nbhd_lbl[n]}</b><br>Mean: ${mn:,.0f}<br>"
                               f"95% CI: [${ci_lo:,.0f}, ${ci_hi:,.0f}]<extra></extra>"),
            ))
        fig_ci.add_hline(y=df["Price"].mean(), line_dash="dot", line_color="#888", line_width=1.5,
                         annotation_text=f"Overall mean: ${df['Price'].mean():,.0f}",
                         annotation_position="top right",
                         annotation_font=dict(color="#555", size=11))
        fig_ci.update_layout(
            yaxis_title="Mean Price ($)", xaxis_title="Neighborhood",
            xaxis={**AX, "gridcolor": "rgba(0,0,0,0)"},
            yaxis={**AX, "tickformat": "$,.0f"},
            legend=LG, **CS, height=300,
            margin=dict(t=10, b=35, l=70, r=10), shapes=BD)
        st.plotly_chart(fig_ci, use_container_width=True, key="q4_ci")
        st.caption(
            "Unadjusted means. Nbhd 1 and 2 CIs don't overlap here — raw prices differ. "
            "The regression below tests whether this gap survives after controlling for house characteristics.")

    # ── OLS ──────────────────────────────────────────────────────────────────
    st.markdown("#### Step 2 — OLS Regression")
    st.markdown("After controlling for house size, brick, and demand, does Nbhd 2 add anything over the Nbhd 1 baseline?")

    ols_l, ols_r = st.columns([1, 1], gap="medium")

    with ols_l:
        st.markdown("##### Fitted Equation — Model 1 (full)")
        st.latex(regression_equation(m1, Q12_VARS))

        st.markdown("##### Model Comparison: Full vs Collapsed")
        mc1, mc2 = st.columns(2)
        mc1.metric("Model 1 Adj R² (separate)",  f"{m1.rsquared_adj:.4f}")
        mc2.metric("Model 3 Adj R² (collapsed)", f"{m3.rsquared_adj:.4f}",
                   delta=f"{m3.rsquared_adj - m1.rsquared_adj:+.4f}")

        st.markdown("##### Partial F-Test (dropping Nbhd2)")
        pf1, pf2, pf3 = st.columns(3)
        pf1.metric("F-statistic",  f"{f_stat_pf:.3f}")
        pf2.metric("p-value",      f"{f_pval_pf:.3f}")
        pf3.metric("Can Collapse?","Yes ✅" if can_collapse else "No ❌")

        st.markdown(f"**Nbhd 2 coefficient:** ${nbhd2_coef4:,.0f} &nbsp; (p = {nbhd2_pval4:.3f})")

    with ols_r:
        st.markdown("##### Coefficient Plot — Model 1 (full)")
        st.plotly_chart(coef_plot(model_cdf(m1).sort_values("Coefficient"), height=310),
                        use_container_width=True, key="q4_coef")

    avf_l, avf_r = st.columns([1, 1], gap="medium")
    with avf_l:
        st.markdown("##### Actual vs Fitted — Model 1 (separate Nbhds)")
        st.plotly_chart(avf_plot(m1), use_container_width=True, key="q4_avf_m1")
    with avf_r:
        st.markdown("##### Actual vs Fitted — Model 3 (Nbhd 1+2 collapsed)")
        st.plotly_chart(avf_plot(m3), use_container_width=True, key="q4_avf_m3")

    if can_collapse:
        st.success(
            f"**Q4 Answer ✅** Nbhd 1 and 2 can be collapsed. "
            f"ANOVA: F = {f_anova:.3f}, p = {p_anova:.3e}. "
            f"Nbhd2 coef = ${nbhd2_coef4:,.0f} (p = {nbhd2_pval4:.3f}); "
            f"Partial F p = {f_pval_pf:.3f}.")
    else:
        st.error(
            f"**Q4 Answer ❌** Nbhd 1 and 2 cannot be collapsed. "
            f"Nbhd2 coef p = {nbhd2_pval4:.3f}; Partial F p = {f_pval_pf:.3f}.")

    insight_box(f"""
    <b>Key Insights</b><br>
    • <b>One-way ANOVA:</b> F = {f_anova:.3f}, p = {p_anova:.3e} — at least one neighborhood has a different mean price. ANOVA tells us <i>that</i> a difference exists, not <i>which</i> group drives it.<br>
    • <b>Means plot (unadjusted):</b> Nbhd 1 = ${nbhd_ci[1][0]:,.0f}, Nbhd 2 = ${nbhd_ci[2][0]:,.0f}, Nbhd 3 = ${nbhd_ci[3][0]:,.0f}. CIs for Nbhd 1 and 2 don't overlap — their raw prices differ. But this is before controlling for house characteristics.<br>
    • <b>Regression (adjusted):</b> Nbhd2 coefficient = ${nbhd2_coef4:,.0f} (p = {nbhd2_pval4:.3f}) — not significant. The raw ${abs(mean_by_nbhd[1]-mean_by_nbhd[2]):,.0f} gap is explained by house characteristics, not the neighborhood.<br>
    • <b>Partial F-test:</b> Dropping Nbhd2 changes Adj R² by only {adj_r2_drop:.4f}. The real distinction is <i>Nbhd 3 vs everyone else</i>.
    """)

    with st.expander("🔬 Try It — Does Collapsing Nbhds Change Your Prediction?"):
        st.caption("Compare Model 1 (separate Nbhd 1 & 2) vs Model 3 (Nbhd 1+2 combined) for your sidebar inputs.")
        xp_full = pd.DataFrame([[1, p_sqft, p_offers, p_beds, p_baths, pb, pn2, pn3]],
                                columns=["const"] + Q12_VARS)
        xp_coll = pd.DataFrame([[1, p_sqft, p_offers, p_beds, p_baths, pb, pn3]],
                                columns=["const"] + Q4C_VARS)
        pred_full4 = m1.predict(xp_full)[0]
        pred_coll4 = m3.predict(xp_coll)[0]
        diff4 = pred_coll4 - pred_full4

        c1, c2, c3 = st.columns(3)
        c1.metric("Model 1 (full)",      f"${pred_full4:,.0f}")
        c2.metric("Model 3 (collapsed)", f"${pred_coll4:,.0f}")
        c3.metric("Difference",          f"${abs(diff4):,.0f}",
                  delta=f"{diff4/pred_full4*100:+.2f}%")
        st.caption("A negligible difference confirms collapsing barely changes predictions — supporting Q4's conclusion.")

st.markdown("---")
st.caption("Case 10.2 — Albright & Winston, Business Analytics (7e)")

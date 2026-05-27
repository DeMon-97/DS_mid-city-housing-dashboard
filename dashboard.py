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

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏠 Controls")
sig_level = st.sidebar.slider("Significance level (α)", 0.01, 1.00, 0.05, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Variable Selection")
st.sidebar.caption("Choose the tab you are editing, then toggle variables below. Switching tabs resets to defaults.")

# Defaults per section
_SDEFS = {
    "Custom": BASE + ["Brick", "Nbhd2", "Nbhd3"],
    "Q1":     BASE + ["Brick", "Nbhd2", "Nbhd3"],
    "Q2":     BASE + ["Brick", "Nbhd2", "Nbhd3"],
    "Q3":     BASE + ["Nbhd2", "Brick_Nbhd3"],
    "Q4":     BASE + ["Brick", "Nbhd2", "Nbhd3"],
}
_SLABELS = {
    "Custom": "🔧 Custom Explorer",
    "Q1":     "Q1 — Brick Premium",
    "Q2":     "Q2 — Nbhd 3 Premium",
    "Q3":     "Q3 — Brick × Nbhd 3",
    "Q4":     "Q4 — Collapse Nbhds",
}

# Initialise session state
if "active_section" not in st.session_state:
    st.session_state.active_section = "Custom"
for _s, _d in _SDEFS.items():
    if f"svars_{_s}" not in st.session_state:
        st.session_state[f"svars_{_s}"] = list(_d)

def _on_section_change():
    _s = st.session_state.active_section
    st.session_state[f"svars_{_s}"] = list(_SDEFS[_s])

section = st.sidebar.selectbox(
    "Editing variables for:",
    list(_SDEFS.keys()),
    format_func=lambda s: _SLABELS[s],
    key="active_section", on_change=_on_section_change,
)

# "Newer" is always auto-derived for Q4; exclude from pill options
_VAR_OPTS = [v for v in ALL_VARS if v != "Newer"]
st.sidebar.pills(
    "Variables:",
    options=_VAR_OPTS,
    selection_mode="multi",
    format_func=lambda v: ALL_VARS[v],
    key=f"svars_{section}",
)

def _gv(s):
    v = st.session_state.get(f"svars_{s}") or []
    return v if v else _SDEFS[s]

cust_sel = _gv("Custom")
q1_sel   = _gv("Q1")
q2_sel   = _gv("Q2")
q3_sel   = _gv("Q3")
q4_sel   = _gv("Q4")

# Derive Q4 collapsed vars: swap Nbhd2+Nbhd3 → Newer
q4c_sel = [v for v in q4_sel if v not in ["Nbhd2", "Nbhd3"]]
if "Nbhd2" in q4_sel or "Nbhd3" in q4_sel:
    q4c_sel = q4c_sel + ["Newer"]

# Fit one model per question
mq1  = sm.OLS(df["Price"], sm.add_constant(df[q1_sel])).fit()  if q1_sel  else None
mq2  = sm.OLS(df["Price"], sm.add_constant(df[q2_sel])).fit()  if q2_sel  else None
mq3  = sm.OLS(df["Price"], sm.add_constant(df[q3_sel])).fit()  if q3_sel  else None
mq4f = sm.OLS(df["Price"], sm.add_constant(df[q4_sel])).fit()  if q4_sel  else None
mq4c = sm.OLS(df["Price"], sm.add_constant(df[q4c_sel])).fit() if q4c_sel else None
avg_price = df["Price"].mean()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Case 10.2 — Housing Price Structure in Mid City")
st.markdown("*Albright & Winston, Business Analytics (7e)*")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_custom, tab_q1, tab_q2, tab_q3, tab_q4, tab_pred = st.tabs([
    "🔧 Custom Explorer",
    "Q1 — Brick Premium",
    "Q2 — Nbhd 3 Premium",
    "Q3 — Brick × Nbhd 3",
    "Q4 — Collapse Nbhds",
    "🏷️ Price Predictor",
])

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM
# ══════════════════════════════════════════════════════════════════════════════
with tab_custom:
    st.subheader("Custom Model Explorer")
    st.markdown("Use the **Variable Selection** panel in the sidebar to choose variables.")

    if not cust_sel:
        st.warning("Select at least one variable in the sidebar.")
    else:
        cust_model = sm.OLS(df["Price"], sm.add_constant(df[cust_sel])).fit()

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("R²",           f"{cust_model.rsquared:.4f}")
        cc2.metric("Adj R²",       f"{cust_model.rsquared_adj:.4f}")
        cc3.metric("F p-value",    f"{cust_model.f_pvalue:.3e}")
        cc4.metric("Observations", len(df))

        st.markdown("##### Fitted Equation")
        st.latex(regression_equation(cust_model, cust_sel))

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

        if "Brick" in cust_sel:
            c, p = cust_model.params.get("Brick"), cust_model.pvalues.get("Brick")
            if c is not None: ab("Q1 — Brick premium?", f"${c:,.0f} (p = {p:.3f})", p < sig_level)
        if "Nbhd3" in cust_sel:
            c, p = cust_model.params.get("Nbhd3"), cust_model.pvalues.get("Nbhd3")
            if c is not None: ab("Q2 — Nbhd 3 premium?", f"${c:,.0f} (p = {p:.3f})", p < sig_level)
        if "Brick_Nbhd3" in cust_sel:
            c, p = cust_model.params.get("Brick_Nbhd3"), cust_model.pvalues.get("Brick_Nbhd3")
            if c is not None: ab("Q3 — Interaction?", f"${c:,.0f} (p = {p:.3f})", p < sig_level)
        else:
            st.markdown("**Q3** ⚠️ Add *Brick × Nbhd 3 Interaction* to answer this question.")
        if "Nbhd2" in cust_sel and "Nbhd3" in cust_sel:
            try:
                Xr  = sm.add_constant(df[[v for v in cust_sel if v != "Nbhd2"]])
                mr  = sm.OLS(df["Price"], Xr).fit()
                fs  = ((mr.ssr - cust_model.ssr) / 1) / (cust_model.ssr / cust_model.df_resid)
                fp  = 1 - stats.f.cdf(fs, 1, cust_model.df_resid)
                ab("Q4 — Collapse Nbhds 1 & 2?",
                   f"Nbhd2 = ${cust_model.params['Nbhd2']:,.0f} (p = {cust_model.pvalues['Nbhd2']:.3f})  |  Partial F p = {fp:.3f}",
                   fp > sig_level)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# Q1 — Brick Premium
# ══════════════════════════════════════════════════════════════════════════════
with tab_q1:
    st.subheader("Q1 — Do buyers pay a premium for a brick house?")
    st.info("The Brick coefficient isolates the price premium attributable solely to brick construction. Use the sidebar to add or remove control variables.")

    if mq1 is None or "Brick" not in q1_sel:
        st.warning("Add **Brick Construction** to Q1 variables in the sidebar to answer this question.")
    else:
        brick_coef = mq1.params["Brick"];  brick_pval = mq1.pvalues["Brick"]
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
            mc1.metric("R²",      f"{mq1.rsquared:.4f}")
            mc2.metric("Adj R²",  f"{mq1.rsquared_adj:.4f}")
            mc3.metric("F p-val", f"{mq1.f_pvalue:.3e}")

            st.markdown("##### Fitted Equation")
            st.latex(regression_equation(mq1, q1_sel))

            st.markdown("##### Brick Coefficient")
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Coefficient",  f"${brick_coef:,.0f}")
            bc2.metric("p-value",      f"{brick_pval:.3f}")
            bc3.metric("Significant?", "Yes ✅" if brick_sig else "No ❌")

        st.markdown("##### Coefficient Table")
        st.dataframe(fit_table(mq1), use_container_width=True)
        st.caption(f"Green = significant at α = {sig_level} | Red = not significant | — = not tested for constant")

        cp_col, avf_col = st.columns([1, 1], gap="medium")
        with cp_col:
            st.markdown("##### Coefficient Plot (95% CI)")
            st.plotly_chart(coef_plot(model_cdf(mq1).sort_values("Coefficient")), use_container_width=True, key="q1_coef")
        with avf_col:
            st.markdown("##### Actual vs Fitted")
            st.plotly_chart(avf_plot(mq1), use_container_width=True, key="q1_avf")

        if brick_sig:
            st.success(f"**Q1 Answer ✅** Buyers pay a significant brick premium of **${brick_coef:,.0f}** (p = {brick_pval:.3f}).")
        else:
            st.warning(f"**Q1 Answer ❌** Brick premium of **${brick_coef:,.0f}** is NOT significant (p = {brick_pval:.3f}) at α = {sig_level}.")

        insight_box(f"""
        <b>Key Insights</b><br>
        • Raw mean difference (unadjusted): brick homes average <b>${mean_brick:,.0f}</b> vs <b>${mean_no_brick:,.0f}</b> — a gap of <b>${raw_diff:,.0f}</b>.<br>
        • After controlling for selected variables, the regression estimates the premium at <b>${brick_coef:,.0f}</b> ({brick_pct:.1f}% of the average price of ${avg_price:,.0f}).<br>
        • The difference between the raw gap and the regression coefficient shows how much is explained by other factors (e.g., brick homes tend to be larger).<br>
        • {'This premium is statistically reliable — brick construction adds genuine value independent of other controls.' if brick_sig else 'The premium is not significant given the current set of controls.'}
        """)


# ══════════════════════════════════════════════════════════════════════════════
# Q2 — Neighborhood 3 Premium
# ══════════════════════════════════════════════════════════════════════════════
with tab_q2:
    st.subheader("Q2 — Is there a premium for Neighborhood 3?")
    st.info("Neighborhood 1 is the baseline. The Nbhd3 coefficient shows the price premium for Nbhd 3 over Nbhd 1. Use the sidebar to add or remove control variables.")

    if mq2 is None or "Nbhd3" not in q2_sel:
        st.warning("Add **Neighborhood 3 dummy** to Q2 variables in the sidebar to answer this question.")
    else:
        nbhd2_coef = mq2.params.get("Nbhd2", 0); nbhd2_pval = mq2.pvalues.get("Nbhd2", 1.0)
        nbhd3_coef = mq2.params["Nbhd3"];         nbhd3_pval = mq2.pvalues["Nbhd3"]
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
            mc1.metric("R²",      f"{mq2.rsquared:.4f}")
            mc2.metric("Adj R²",  f"{mq2.rsquared_adj:.4f}")
            mc3.metric("F p-val", f"{mq2.f_pvalue:.3e}")

            st.markdown("##### Fitted Equation")
            st.latex(regression_equation(mq2, q2_sel))

            st.markdown("##### Neighborhood Coefficients (vs Nbhd 1 baseline)")
            nr1, nr2 = st.columns(2)
            if "Nbhd2" in q2_sel:
                nr1.metric("Nbhd 2 coef", f"${nbhd2_coef:,.0f}", delta=f"p = {nbhd2_pval:.3f}")
            nr2.metric("Nbhd 3 coef", f"${nbhd3_coef:,.0f}", delta=f"p = {nbhd3_pval:.3f}")
            st.metric("Nbhd 3 Significant?", "Yes ✅" if nbhd3_sig else "No ❌")

        st.markdown("##### Coefficient Table")
        st.dataframe(fit_table(mq2), use_container_width=True)
        st.caption(f"Green = significant at α = {sig_level} | Red = not significant | — = not tested for constant")

        cp_col, avf_col = st.columns([1, 1], gap="medium")
        with cp_col:
            st.markdown("##### Coefficient Plot (95% CI)")
            st.plotly_chart(coef_plot(model_cdf(mq2).sort_values("Coefficient")), use_container_width=True, key="q2_coef")
        with avf_col:
            st.markdown("##### Actual vs Fitted")
            st.plotly_chart(avf_plot(mq2), use_container_width=True, key="q2_avf")

        if nbhd3_sig:
            st.success(f"**Q2 Answer ✅** Nbhd 3 commands a significant premium of **${nbhd3_coef:,.0f}** over Nbhd 1 (p = {nbhd3_pval:.3f}).")
        else:
            st.warning(f"**Q2 Answer ❌** Nbhd 3 premium of **${nbhd3_coef:,.0f}** is NOT significant (p = {nbhd3_pval:.3f}) at α = {sig_level}.")

        insight_box(f"""
        <b>Key Insights</b><br>
        • Raw mean prices: Nbhd 1 = <b>${mean_by_nbhd[1]:,.0f}</b>, Nbhd 2 = <b>${mean_by_nbhd[2]:,.0f}</b>, Nbhd 3 = <b>${mean_by_nbhd[3]:,.0f}</b>.<br>
        • After controlling for selected variables, Nbhd 3 carries a regression premium of <b>${nbhd3_coef:,.0f}</b> ({nbhd3_pct:.1f}% of average price) over Nbhd 1.<br>
        {"• <b>Nbhd 2's coefficient is not significant</b> (p = " + f"{nbhd2_pval:.3f}" + ") — its prices are statistically indistinguishable from Nbhd 1 once other factors are controlled. This is a preview of Q4.<br>" if "Nbhd2" in q2_sel else ""}
        • {'The Nbhd 3 premium is real — buyers are paying for the neighborhood itself, not just larger or better-constructed homes.' if nbhd3_sig else 'Once house characteristics are accounted for, the Nbhd 3 premium disappears.'}
        """)


# ══════════════════════════════════════════════════════════════════════════════
# Q3 — Brick × Nbhd 3 Interaction
# ══════════════════════════════════════════════════════════════════════════════
with tab_q3:
    st.subheader("Q3 — Is there an extra brick premium in Neighborhood 3?")
    st.info("The Brick × Nbhd 3 interaction term tests whether being brick AND in Nbhd 3 adds an extra premium. Use the sidebar to adjust variables.")

    if mq3 is None or "Brick_Nbhd3" not in q3_sel:
        st.warning("Add **Brick × Nbhd 3 Interaction** to Q3 variables in the sidebar to answer this question.")
    else:
        int_coef = mq3.params["Brick_Nbhd3"];  int_pval = mq3.pvalues["Brick_Nbhd3"]
        int_sig     = int_pval < sig_level
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
            mc1.metric("R²",      f"{mq3.rsquared:.4f}")
            mc2.metric("Adj R²",  f"{mq3.rsquared_adj:.4f}")
            mc3.metric("F p-val", f"{mq3.f_pvalue:.3e}")

            st.markdown("##### Fitted Equation")
            st.latex(regression_equation(mq3, q3_sel))

            st.markdown("##### Interaction Term")
            it1, it2, it3 = st.columns(3)
            it1.metric("Interaction coef", f"${int_coef:,.0f}")
            it2.metric("p-value",          f"{int_pval:.3f}")
            it3.metric("Significant?",     "Yes ✅" if int_sig else "No ❌")

        st.markdown("##### Coefficient Table")
        st.dataframe(fit_table(mq3), use_container_width=True)
        st.caption(f"Green = significant at α = {sig_level} | Red = not significant | — = not tested for constant")

        cp_col, avf_col = st.columns([1, 1], gap="medium")
        with cp_col:
            st.markdown("##### Coefficient Plot (95% CI)")
            st.plotly_chart(coef_plot(model_cdf(mq3).sort_values("Coefficient")), use_container_width=True, key="q3_coef")
        with avf_col:
            st.markdown("##### Actual vs Fitted")
            st.plotly_chart(avf_plot(mq3), use_container_width=True, key="q3_avf")

        if int_sig:
            st.success(f"**Q3 Answer ✅** Interaction significant (p = {int_pval:.3f}). Extra brick premium in Nbhd 3 = **${int_coef:,.0f}**.")
        else:
            st.warning(f"**Q3 Answer ❌** Interaction NOT significant (p = {int_pval:.3f}). Brick and Nbhd 3 effects are simply additive — no extra synergy premium.")

        raw_premiums = pivot["Brick Premium ($)"]
        insight_box(f"""
        <b>Key Insights</b><br>
        • Raw brick premiums: Nbhd 1 = <b>${raw_premiums.iloc[0]:,.0f}</b>, Nbhd 2 = <b>${raw_premiums.iloc[1]:,.0f}</b>, Nbhd 3 = <b>${raw_premiums.iloc[2]:,.0f}</b>.<br>
        • {'The interaction is significant — the brick premium in Nbhd 3 is larger than in other neighborhoods.' if int_sig else 'The interaction is not significant — the brick premium is consistent across all neighborhoods.'}<br>
        • The regression tests whether the Nbhd 3 brick premium survives after controlling for selected variables.<br>
        • Adj R² = <b>{mq3.rsquared_adj:.4f}</b>. Adding the interaction term improves fit only if it is significant.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# Q4 — Collapse Neighborhoods 1 & 2
# ══════════════════════════════════════════════════════════════════════════════
with tab_q4:
    st.subheader("Q4 — Can Neighborhoods 1 and 2 be treated as one group?")
    st.info("ANOVA tests raw (unadjusted) mean prices. OLS regression then controls for house characteristics and tests whether a genuine neighborhood difference remains.")

    f_anova, p_anova = stats.f_oneway(
        df[df["Nbhd"] == 1]["Price"], df[df["Nbhd"] == 2]["Price"], df[df["Nbhd"] == 3]["Price"])
    mean_by_nbhd = df.groupby("Nbhd")["Price"].mean()

    q4_ready = mq4f is not None and mq4c is not None and "Nbhd2" in q4_sel
    if q4_ready:
        f_stat_pf = ((mq4c.ssr - mq4f.ssr) / 1) / (mq4f.ssr / mq4f.df_resid)
        f_pval_pf = 1 - stats.f.cdf(f_stat_pf, 1, mq4f.df_resid)
        can_collapse = f_pval_pf > sig_level
        nbhd2_coef4  = mq4f.params["Nbhd2"]; nbhd2_pval4 = mq4f.pvalues["Nbhd2"]
        adj_r2_drop  = abs(mq4f.rsquared_adj - mq4c.rsquared_adj)

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
    st.markdown("After controlling for selected variables, does Nbhd 2 add anything over the Nbhd 1 baseline?")

    if not q4_ready:
        st.warning("Add **Neighborhood 2 dummy** and **Neighborhood 3 dummy** to Q4 variables in the sidebar.")
    else:
        ols_l, ols_r = st.columns([1, 1], gap="medium")

        with ols_l:
            st.markdown("##### Fitted Equation — Full model (separate Nbhds)")
            st.latex(regression_equation(mq4f, q4_sel))

            st.markdown("##### Model Comparison: Full vs Collapsed")
            mc1, mc2 = st.columns(2)
            mc1.metric("Full Adj R² (separate)",   f"{mq4f.rsquared_adj:.4f}")
            mc2.metric("Collapsed Adj R²",          f"{mq4c.rsquared_adj:.4f}",
                       delta=f"{mq4c.rsquared_adj - mq4f.rsquared_adj:+.4f}")

            st.markdown("##### Partial F-Test (dropping Nbhd2)")
            pf1, pf2, pf3 = st.columns(3)
            pf1.metric("F-statistic",  f"{f_stat_pf:.3f}")
            pf2.metric("p-value",      f"{f_pval_pf:.3f}")
            pf3.metric("Can Collapse?","Yes ✅" if can_collapse else "No ❌")

            st.markdown(f"**Nbhd 2 coefficient:** ${nbhd2_coef4:,.0f} &nbsp; (p = {nbhd2_pval4:.3f})")

        with ols_r:
            st.markdown("##### Coefficient Plot — Full model")
            st.plotly_chart(coef_plot(model_cdf(mq4f).sort_values("Coefficient"), height=310),
                            use_container_width=True, key="q4_coef")

        st.markdown("##### Coefficient Table — Full model")
        st.dataframe(fit_table(mq4f), use_container_width=True)
        st.caption(f"Green = significant at α = {sig_level} | Red = not significant | — = not tested for constant")

        avf_l, avf_r = st.columns([1, 1], gap="medium")
        with avf_l:
            st.markdown("##### Actual vs Fitted — Full model (separate Nbhds)")
            st.plotly_chart(avf_plot(mq4f), use_container_width=True, key="q4_avf_m1")
        with avf_r:
            st.markdown("##### Actual vs Fitted — Collapsed model (Nbhd 1+2)")
            st.plotly_chart(avf_plot(mq4c), use_container_width=True, key="q4_avf_m3")

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


# ══════════════════════════════════════════════════════════════════════════════
# Price Predictor
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.subheader("🏷️ Price Predictor")
    st.markdown("Adjust the sliders to describe a house. The model uses OLS regression (SqFt, Offers, Bedrooms, Bathrooms, Brick, Neighborhood) to estimate its price.")

    pred_model = sm.OLS(df["Price"], sm.add_constant(df[Q12_VARS])).fit()

    sl_col, res_col = st.columns([1, 1], gap="large")

    with sl_col:
        sqft      = st.slider("Square Footage",    int(df["SqFt"].min()),     int(df["SqFt"].max()),     int(df["SqFt"].median()),     step=50)
        offers    = st.slider("Number of Offers",  int(df["Offers"].min()),   int(df["Offers"].max()),   int(df["Offers"].median()),   step=1)
        bedrooms  = st.slider("Bedrooms",          int(df["Bedrooms"].min()), int(df["Bedrooms"].max()), int(df["Bedrooms"].median()), step=1)
        bathrooms = st.slider("Bathrooms",         int(df["Bathrooms"].min()),int(df["Bathrooms"].max()),int(df["Bathrooms"].median()),step=1)
        brick_in  = st.radio("Brick Construction", ["No", "Yes"], horizontal=True)
        nbhd_in   = st.radio("Neighborhood",       [1, 2, 3],
                             format_func=lambda n: f"Neighborhood {n}", horizontal=True)

    brick_val = 1 if brick_in == "Yes" else 0
    nbhd2_val = 1 if nbhd_in == 2 else 0
    nbhd3_val = 1 if nbhd_in == 3 else 0

    X_new = pd.DataFrame(
        [[sqft, offers, bedrooms, bathrooms, brick_val, nbhd2_val, nbhd3_val]],
        columns=Q12_VARS,
    )
    X_new_c = sm.add_constant(X_new, has_constant="add")

    pred_frame = pred_model.get_prediction(X_new_c).summary_frame(alpha=sig_level)
    predicted  = pred_frame["mean"].iloc[0]
    ci_lo      = pred_frame["obs_ci_lower"].iloc[0]
    ci_hi      = pred_frame["obs_ci_upper"].iloc[0]
    pct        = (df["Price"] < predicted).mean() * 100

    with res_col:
        st.markdown("##### Estimated Price")
        st.metric("Predicted Price", f"${predicted:,.0f}")
        st.metric(f"{int((1 - sig_level) * 100)}% Prediction Interval",
                  f"${ci_lo:,.0f} — ${ci_hi:,.0f}")
        st.metric("Percentile in Dataset", f"{pct:.0f}th")

        st.markdown(f"Model Adj R² = **{pred_model.rsquared_adj:.4f}**")

    st.markdown("##### Where Does This House Fall in the Price Distribution?")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df["Price"], nbinsx=25, name="Actual prices",
        marker_color="#4C72B0", opacity=0.65,
        hovertemplate="Price: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ))
    fig_dist.add_vline(
        x=predicted, line_color="#DD8452", line_width=2.5,
        annotation_text=f"  Predicted: ${predicted:,.0f}",
        annotation_position="top right",
        annotation_font=dict(color="#DD8452", size=12),
    )
    fig_dist.add_vrect(
        x0=ci_lo, x1=ci_hi, fillcolor="#DD8452", opacity=0.12, line_width=0,
        annotation_text=f"  {int((1 - sig_level) * 100)}% PI",
        annotation_position="top left",
        annotation_font=dict(color="#DD8452", size=10),
    )
    fig_dist.update_layout(
        xaxis_title="Price ($)", yaxis_title="Count",
        xaxis={**AX, "tickformat": "$,.0f"}, yaxis=AX,
        legend=LG, **CS, height=320,
        margin=dict(t=15, b=35, l=60, r=15), shapes=BD,
    )
    st.plotly_chart(fig_dist, use_container_width=True, key="pred_dist")

    insight_box(f"""
    <b>How to read this</b><br>
    • The orange line is the model's point estimate: <b>${predicted:,.0f}</b>.<br>
    • The shaded band is the {int((1 - sig_level) * 100)}% <b>prediction interval</b> — the range within which a single new home with these characteristics is expected to sell, with {int((1 - sig_level) * 100)}% confidence.<br>
    • The prediction interval is wider than a confidence interval because it accounts for both model uncertainty and natural house-to-house variation.<br>
    • This house is estimated to be in the <b>{pct:.0f}th percentile</b> of the dataset by price.
    """)


st.markdown("---")
st.caption("Case 10.2 — Albright & Winston, Business Analytics (7e)")

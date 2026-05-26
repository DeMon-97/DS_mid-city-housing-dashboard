#!/usr/local/bin/python3.13
"""
Case 10.2 — Housing Price Structure in Mid City
Albright & Winston, Business Analytics (7e)

Regression analysis addressing 4 key questions:
  Q1. Do buyers pay a premium for a brick house?
  Q2. Is there a premium for Neighborhood 3?
  Q3. Is there an extra brick × Neighborhood 3 interaction premium?
  Q4. Can Neighborhoods 1 and 2 be collapsed into one "older" group?
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves chart without needing a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "/Users/deep/Documents/EPGP/DS/Albright_BusinessAnalytics_7e_Case_Files/C10_02.xlsx"
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── 1. Load & Prepare Data ────────────────────────────────────────────────────
print("=" * 70)
print("CASE 10.2 — HOUSING PRICE STRUCTURE IN MID CITY")
print("=" * 70)

df = pd.read_excel(DATA_PATH)
df.rename(columns={"Sq Ft": "SqFt"}, inplace=True)

# Encode Brick as 1/0
df["Brick"] = (df["Brick"] == "Yes").astype(int)

# Neighborhood dummies (Nbhd 1 = base)
df["Nbhd2"] = (df["Nbhd"] == 2).astype(int)
df["Nbhd3"] = (df["Nbhd"] == 3).astype(int)

# Interaction: Brick in Nbhd 3
df["Brick_Nbhd3"] = df["Brick"] * df["Nbhd3"]

# Collapsed dummy: "Older" = Nbhd 1 or 2 (base), Newer = Nbhd 3
df["Newer"] = df["Nbhd3"]

print(f"\nDataset: {df.shape[0]} homes across {df['Nbhd'].nunique()} neighborhoods")
print(df.groupby("Nbhd").agg(Count=("Price", "count"),
                              AvgPrice=("Price", "mean"),
                              BrickPct=("Brick", "mean")).round(2).to_string())

# ── Helper ────────────────────────────────────────────────────────────────────
def fit_ols(y, X_cols, data, title):
    X = sm.add_constant(data[X_cols])
    model = sm.OLS(data[y], X).fit()
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    print(model.summary2().tables[1].to_string())
    print(f"\n  R² = {model.rsquared:.4f}  |  Adj R² = {model.rsquared_adj:.4f}"
          f"  |  F-stat p = {model.f_pvalue:.4e}")
    return model

# ── Base features used across all models ──────────────────────────────────────
BASE = ["SqFt", "Offers", "Bedrooms", "Bathrooms"]

# ═══════════════════════════════════════════════════════════════════════════════
# Q1 & Q2 — Base model with Brick + Neighborhood dummies
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n══════════════════════════════════════════════════════════════════════")
print("  MODEL 1 — Brick Premium & Neighborhood 3 Premium")
print("══════════════════════════════════════════════════════════════════════")
m1 = fit_ols("Price", BASE + ["Brick", "Nbhd2", "Nbhd3"], df,
             "Q1 & Q2: Price ~ SqFt + Offers + Bedrooms + Bathrooms + Brick + Nbhd2 + Nbhd3")

brick_coef   = m1.params["Brick"]
brick_pval   = m1.pvalues["Brick"]
nbhd3_coef   = m1.params["Nbhd3"]
nbhd3_pval   = m1.pvalues["Nbhd3"]

print(f"\n  ► Q1 Answer: Brick premium = ${brick_coef:,.0f}  (p = {brick_pval:.4f})")
print(f"    {'SIGNIFICANT' if brick_pval < 0.05 else 'NOT significant'} at 5% level")
print(f"\n  ► Q2 Answer: Nbhd 3 premium = ${nbhd3_coef:,.0f}  (p = {nbhd3_pval:.4f})")
print(f"    {'SIGNIFICANT' if nbhd3_pval < 0.05 else 'NOT significant'} at 5% level")

# ═══════════════════════════════════════════════════════════════════════════════
# Q3 — Interaction: extra brick premium in Nbhd 3?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n══════════════════════════════════════════════════════════════════════")
print("  MODEL 2 — Brick × Neighborhood 3 Interaction")
print("══════════════════════════════════════════════════════════════════════")
m2 = fit_ols("Price", BASE + ["Brick", "Nbhd2", "Nbhd3", "Brick_Nbhd3"], df,
             "Q3: Adding Brick × Nbhd3 interaction term")

interaction_coef = m2.params["Brick_Nbhd3"]
interaction_pval = m2.pvalues["Brick_Nbhd3"]

print(f"\n  ► Q3 Answer: Extra brick premium in Nbhd 3 = ${interaction_coef:,.0f}"
      f"  (p = {interaction_pval:.4f})")
print(f"    {'SIGNIFICANT' if interaction_pval < 0.05 else 'NOT significant'} — "
      f"{'there IS' if interaction_pval < 0.05 else 'there is NO'} extra brick premium in Nbhd 3")

# ═══════════════════════════════════════════════════════════════════════════════
# Q4 — Can Nbhd 1 & 2 be collapsed?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n══════════════════════════════════════════════════════════════════════")
print("  MODEL 3 — Collapsed Neighborhoods (Older = 1+2  vs  Newer = 3)")
print("══════════════════════════════════════════════════════════════════════")
m3 = fit_ols("Price", BASE + ["Brick", "Newer"], df,
             "Q4: Price ~ SqFt + Offers + Bedrooms + Bathrooms + Brick + Newer")

# F-test: is the Nbhd2 coefficient in m1 significantly different from 0?
nbhd2_coef  = m1.params["Nbhd2"]
nbhd2_pval  = m1.pvalues["Nbhd2"]

print(f"\n  Nbhd2 coefficient in full model: ${nbhd2_coef:,.0f}  (p = {nbhd2_pval:.4f})")
print(f"\n  Comparing full model (Nbhd1≠Nbhd2) vs collapsed (Nbhd1=Nbhd2):")
print(f"    Full model  Adj R² = {m1.rsquared_adj:.4f}")
print(f"    Collapsed   Adj R² = {m3.rsquared_adj:.4f}")

# Partial F-test
f_stat = ((m3.ssr - m1.ssr) / 1) / (m1.ssr / m1.df_resid)
f_pval = 1 - stats.f.cdf(f_stat, 1, m1.df_resid)
print(f"\n  Partial F-test (dropping Nbhd2): F = {f_stat:.4f},  p = {f_pval:.4f}")
print(f"\n  ► Q4 Answer: {'Nbhd 1 and 2 CAN be collapsed' if f_pval > 0.05 else 'Nbhd 1 and 2 CANNOT be collapsed'}"
      f" — Nbhd2 coefficient is {'NOT significant (p > 0.05)' if f_pval > 0.05 else 'significant (p ≤ 0.05)'}")

# ═══════════════════════════════════════════════════════════════════════════════
# Visualizations
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Case 10.2 — Housing Price Structure in Mid City", fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

nbhd_labels = {1: "Nbhd 1\n(Older)", 2: "Nbhd 2\n(Older)", 3: "Nbhd 3\n(Newer)"}
df["NbhdLabel"] = df["Nbhd"].map(nbhd_labels)
df["BrickLabel"] = df["Brick"].map({1: "Brick", 0: "No Brick"})

# Plot 1 — Price by Neighborhood
ax1 = fig.add_subplot(gs[0, 0])
sns.boxplot(data=df, x="NbhdLabel", y="Price", hue="NbhdLabel", ax=ax1,
            palette=["#4C72B0", "#4C72B0", "#DD8452"], legend=False)
ax1.set_title("Price by Neighborhood", fontweight="bold")
ax1.set_xlabel(""); ax1.set_ylabel("Price ($)")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# Plot 2 — Price by Brick
ax2 = fig.add_subplot(gs[0, 1])
sns.boxplot(data=df, x="BrickLabel", y="Price", hue="BrickLabel", ax=ax2,
            palette=["#DD8452", "#4C72B0"], legend=False)
ax2.set_title("Price: Brick vs. No Brick", fontweight="bold")
ax2.set_xlabel(""); ax2.set_ylabel("Price ($)")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# Plot 3 — Brick × Neighborhood interaction
ax3 = fig.add_subplot(gs[0, 2])
means = df.groupby(["NbhdLabel", "BrickLabel"])["Price"].mean().reset_index()
for brick_type, grp in means.groupby("BrickLabel"):
    ax3.plot(grp["NbhdLabel"], grp["Price"], marker="o", label=brick_type, linewidth=2)
ax3.set_title("Interaction: Brick × Neighborhood", fontweight="bold")
ax3.set_xlabel(""); ax3.set_ylabel("Mean Price ($)")
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax3.legend(title="House Type")

# Plot 4 — Model 1 coefficients (Q1 & Q2)
ax4 = fig.add_subplot(gs[1, 0])
coefs = m1.params.drop("const")
errors = m1.conf_int().drop("const")
colors = ["#2ca02c" if p < 0.05 else "#aec7e8" for p in m1.pvalues.drop("const")]
bars = ax4.barh(coefs.index, coefs.values, color=colors,
                xerr=[coefs - errors[0], errors[1] - coefs], capsize=4)
ax4.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax4.set_title("Model 1 Coefficients\n(green = significant at 5%)", fontweight="bold")
ax4.set_xlabel("Coefficient Value")

# Plot 5 — Actual vs Fitted (best model, m2)
ax5 = fig.add_subplot(gs[1, 1])
fitted = m2.fittedvalues
ax5.scatter(fitted, df["Price"], alpha=0.5, color="#4C72B0", s=30)
lims = [min(fitted.min(), df["Price"].min()), max(fitted.max(), df["Price"].max())]
ax5.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
ax5.set_title("Actual vs. Fitted Prices\n(Model 2 with Interaction)", fontweight="bold")
ax5.set_xlabel("Fitted Price ($)"); ax5.set_ylabel("Actual Price ($)")
ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
ax5.legend()

# Plot 6 — Model R² comparison
ax6 = fig.add_subplot(gs[1, 2])
models = ["Model 1\n(Q1 & Q2)", "Model 2\n(+Interaction)", "Model 3\n(Collapsed)"]
r2_adj = [m1.rsquared_adj, m2.rsquared_adj, m3.rsquared_adj]
bars = ax6.bar(models, r2_adj, color=["#4C72B0", "#2ca02c", "#DD8452"])
for bar, val in zip(bars, r2_adj):
    ax6.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
             f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
ax6.set_ylim(0, 1)
ax6.set_title("Adjusted R² Comparison", fontweight="bold")
ax6.set_ylabel("Adjusted R²")

plt.savefig("/Users/deep/DS Project/housing_analysis.png", bbox_inches="tight")
print("\n\n  Chart saved → /Users/deep/DS Project/housing_analysis.png")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n══════════════════════════════════════════════════════════════════════")
print("  EXECUTIVE SUMMARY")
print("══════════════════════════════════════════════════════════════════════")
print(f"""
  Q1 — Brick Premium:
       Buyers pay ~${brick_coef:,.0f} more for a brick house, all else equal.
       This is {'statistically significant' if brick_pval < 0.05 else 'NOT statistically significant'} (p = {brick_pval:.4f}).

  Q2 — Neighborhood 3 Premium:
       Houses in Nbhd 3 command ~${nbhd3_coef:,.0f} more than Nbhd 1, all else equal.
       This is {'statistically significant' if nbhd3_pval < 0.05 else 'NOT statistically significant'} (p = {nbhd3_pval:.4f}).

  Q3 — Extra Brick × Nbhd 3 Interaction:
       The extra brick premium in Nbhd 3 = ${interaction_coef:,.0f} (p = {interaction_pval:.4f}).
       {'There IS a significant extra premium for brick houses in Nbhd 3.' if interaction_pval < 0.05
        else 'There is NO significant extra premium for brick houses specifically in Nbhd 3.'}

  Q4 — Collapsing Neighborhoods 1 & 2:
       Nbhd2 coefficient = ${nbhd2_coef:,.0f} (p = {nbhd2_pval:.4f}).
       Partial F-test p = {f_pval:.4f}.
       {'Nbhd 1 and 2 CAN be collapsed — the difference between them is not significant.'
        if f_pval > 0.05 else 'Nbhd 1 and 2 CANNOT be collapsed — they differ significantly.'}
""")
print("=" * 70)

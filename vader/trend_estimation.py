"""
RQ3 Trend Estimation — Temporal analysis of sentiment toward ChatGPT-as-therapist on YouTube.

Implements:
  Eq. 1: P(y = c | t) = N_{c,t} / N_t   (proportion per time bin)
  Eq. 2: Logistic regression of non-positive ~ Time + comment_length
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import statsmodels.api as sm
import warnings
import os

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "vader_sentiment_results.csv")
OUTPUT_DIR = os.path.dirname(__file__)


# ─── Load & prep ────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Binary outcome: 1 = non-positive (Neutral or Negative), 0 = Positive
df["non_positive"] = (df["VADER_Sentiment"] != "Positive").astype(int)

# Comment length covariate
df["comment_length"] = df["Comment"].fillna("").str.len()

# Time as numeric: months elapsed since the earliest comment
t0 = df["timestamp"].min()
df["time_months"] = (
    (df["timestamp"] - t0).dt.total_seconds() / (30.44 * 24 * 3600)
)

print(f"Dataset: {len(df):,} comments from {t0.date()} to {df['timestamp'].max().date()}")
print(f"Non-positive rate: {df['non_positive'].mean():.3f}")
print()


# ─── Equation 1: Proportions per quarterly bin ──────────────────────────────

df["quarter"] = df["timestamp"].dt.to_period("Q")
agg = (
    df.groupby(["quarter", "VADER_Sentiment"])
    .size()
    .unstack(fill_value=0)
)
# Ensure all columns present
for col in ["Positive", "Neutral", "Negative"]:
    if col not in agg.columns:
        agg[col] = 0

agg["N_t"] = agg.sum(axis=1)
for col in ["Positive", "Neutral", "Negative"]:
    agg[f"P_{col}"] = agg[col] / agg["N_t"]

# Drop quarters with very few comments (< 10) for the plot
agg_plot = agg[agg["N_t"] >= 10].copy()

print("=== Equation 1: Sentiment proportions per quarter ===")
display_cols = ["N_t", "P_Positive", "P_Neutral", "P_Negative"]
print(agg_plot[display_cols].to_string())
print()


# ─── Equation 2: Logistic regression ────────────────────────────────────────

# Standardize time to aid interpretation (1 unit = 1 SD)
time_mean = df["time_months"].mean()
time_std  = df["time_months"].std()
df["time_std"] = (df["time_months"] - time_mean) / time_std
df["comment_length_std"] = (
    (df["comment_length"] - df["comment_length"].mean()) / df["comment_length"].std()
)

y = df["non_positive"]
X = sm.add_constant(df[["time_months", "comment_length"]])

model = sm.Logit(y, X)
result = model.fit(disp=False)

print("=== Equation 2: Logistic Regression Results ===")
print(result.summary2())
print()

b1       = result.params["time_months"]
se1      = result.bse["time_months"]
z1       = result.tvalues["time_months"]
p1       = result.pvalues["time_months"]
or1      = np.exp(b1)
ci_lo    = np.exp(result.conf_int().loc["time_months", 0])
ci_hi    = np.exp(result.conf_int().loc["time_months", 1])

print(f"β₁ (Time, per month)  = {b1:+.5f}  (SE = {se1:.5f})")
print(f"z = {z1:.3f},  p = {p1:.4g}")
print(f"Odds Ratio = {or1:.4f}  [95% CI: {ci_lo:.4f} – {ci_hi:.4f}]")

if p1 < 0.05:
    direction = "increases" if b1 > 0 else "decreases"
    print(f"\nConclusion: Non-positive sentiment {direction} significantly over time (p={p1:.4g}).")
else:
    print(f"\nConclusion: No significant temporal trend in non-positive sentiment (p={p1:.4g}).")

# Also fit with standardized time for per-SD effect
X_std = sm.add_constant(df[["time_std", "comment_length_std"]])
result_std = sm.Logit(y, X_std).fit(disp=False)
b1_std = result_std.params["time_std"]
or_std  = np.exp(b1_std)
ci_lo_std = np.exp(result_std.conf_int().loc["time_std", 0])
ci_hi_std = np.exp(result_std.conf_int().loc["time_std", 1])
print(f"\nStandardized β₁ (per 1-SD in time) = {b1_std:+.4f}, OR = {or_std:.4f} [{ci_lo_std:.4f}–{ci_hi_std:.4f}]")
print()


# ─── Figures ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle("RQ3: Temporal Sentiment Trend — ChatGPT as Therapist (YouTube)", fontsize=14, fontweight="bold")

# Panel A: Stacked bar of proportions per quarter
ax = axes[0]
quarters = [str(q) for q in agg_plot.index]
x = np.arange(len(quarters))
width = 0.6

bars_pos = ax.bar(x, agg_plot["P_Positive"], width, label="Positive", color="#2196F3", alpha=0.85)
bars_neu = ax.bar(x, agg_plot["P_Neutral"],  width, bottom=agg_plot["P_Positive"],
                  label="Neutral",  color="#9E9E9E", alpha=0.85)
bars_neg = ax.bar(x, agg_plot["P_Negative"], width,
                  bottom=agg_plot["P_Positive"] + agg_plot["P_Neutral"],
                  label="Negative", color="#F44336", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(quarters, rotation=45, ha="right", fontsize=9)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_ylabel("Proportion of Comments")
ax.set_title("Panel A — Eq. (1): Sentiment Proportions by Quarter")
ax.legend(loc="upper left")

# Annotate N on each bar
for xi, (_, row) in zip(x, agg_plot.iterrows()):
    ax.text(xi, 1.01, f"n={int(row['N_t'])}", ha="center", va="bottom", fontsize=7, color="#333")

ax.set_ylim(0, 1.10)

# Panel B: Smoothed non-positive rate over time (monthly) + regression line
ax2 = axes[1]

monthly = (
    df.groupby("month")
      .agg(N=("non_positive", "count"), nonpos=("non_positive", "sum"))
      .reset_index()
)
monthly = monthly[monthly["N"] >= 10].copy()
monthly["rate"] = monthly["nonpos"] / monthly["N"]
monthly["month_dt"] = pd.to_datetime(monthly["month"])

ax2.bar(monthly["month_dt"], monthly["rate"], width=20, color="#607D8B", alpha=0.5, label="Monthly non-positive rate")

# Overlay logistic regression trend line
t_range = np.linspace(df["time_months"].min(), df["time_months"].max(), 300)
log_odds = result.params["const"] + result.params["time_months"] * t_range + result.params["comment_length"] * df["comment_length"].mean()
prob = 1 / (1 + np.exp(-log_odds))
t_dates = pd.to_datetime(t0) + pd.to_timedelta(t_range * 30.44, unit="D")
ax2.plot(t_dates, prob, color="#E91E63", linewidth=2.5,
         label=f"Logistic trend (β₁={b1:+.5f}, OR={or1:.3f}, p={p1:.3g})")

ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax2.set_ylabel("P(Non-positive)")
ax2.set_xlabel("Date")
ax2.set_title("Panel B — Eq. (2): Logistic Regression of Non-Positive Sentiment over Time")
ax2.legend()
ax2.set_ylim(0, 1)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "trend_estimation.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Figure saved → {out_path}")
plt.show()

# ─── Export quarterly table ─────────────────────────────────────────────────
csv_out = os.path.join(OUTPUT_DIR, "trend_quarterly_proportions.csv")
agg_plot[["N_t", "P_Positive", "P_Neutral", "P_Negative"]].to_csv(csv_out)
print(f"Quarterly proportions saved → {csv_out}")

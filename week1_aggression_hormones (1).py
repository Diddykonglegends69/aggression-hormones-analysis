"""
Hormones and self-reported aggression in a single participant.

Dataset: OpenNeuro ds005115 ("28andHe") - 40 sessions (20 morning, 20 evening)
from one adult male, with salivary hormones and questionnaire measures at
each session.

This script tests whether salivary testosterone, salivary cortisol, and
perceived stress predict aggression, and whether any such relationship
survives controls for session order and time of day.

Usage:
    python week1_aggression_hormones.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats

URL = ("https://raw.githubusercontent.com/OpenNeuroDatasets/"
       "ds005115/main/participants.tsv")

# Missing values are coded "n/a"; step counts contain thousands separators.
df = pd.read_csv(URL, sep="\t", na_values="n/a", thousands=",")

# Session labels ("ses-07") become integers so session order can be modeled.
df["session_num"] = df["session_id"].str.replace("ses-", "").astype(int)

# Saliva measures were collected at every session; serum measures were not,
# so the saliva versions are used to retain all 40 sessions.
cols = ["session_num", "time_of_day", "aggression",
        "total_testosterone_saliva", "cortisol_saliva",
        "perceived_stress", "poms_anger"]
df = df[cols].dropna().sort_values("session_num").reset_index(drop=True)

print(f"Sessions: {len(df)}")
print("\nMeans by time of day:")
print(df.groupby("time_of_day")[["aggression", "total_testosterone_saliva",
                                 "cortisol_saliva"]].mean().round(2))

PREDICTORS = ["total_testosterone_saliva", "cortisol_saliva",
              "perceived_stress"]

# ---------------------------------------------------------------
# Zero-order correlations
# ---------------------------------------------------------------
print("\nCorrelations with aggression:")
for var in PREDICTORS:
    r, p = stats.pearsonr(df[var], df["aggression"])
    print(f"  {var:28s} r = {r:5.2f}, p = {p:.4f}")

# Session order is examined as a potential confound: hormone levels follow a
# diurnal pattern, and morning sessions were concentrated early in the study.
r, p = stats.pearsonr(df["session_num"], df["aggression"])
print(f"\n  {'session_num':28s} r = {r:5.2f}, p = {p:.4f}")

# ---------------------------------------------------------------
# Regression models controlling for session order and time of day
# ---------------------------------------------------------------
for var in PREDICTORS:
    model = smf.ols(
        f"aggression ~ {var} + session_num + C(time_of_day)", data=df
    ).fit()
    print(f"\nModel: aggression ~ {var} + session_num + time_of_day")
    print(model.summary().tables[1])

# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sns.lineplot(data=df, x="session_num", y="aggression", marker="o", ax=axes[0])
axes[0].set_title("Aggression across sessions")
axes[0].set_xlabel("Session")
axes[0].set_ylabel("Aggression (Buss-Perry)")

sns.scatterplot(data=df, x="total_testosterone_saliva", y="aggression",
                hue="time_of_day", s=70, ax=axes[1])
axes[1].set_title("Testosterone and aggression by time of day")
axes[1].set_xlabel("Salivary testosterone")
axes[1].set_ylabel("Aggression (Buss-Perry)")

plt.tight_layout()
plt.savefig("aggression_plots.png", dpi=150)
print("\nSaved aggression_plots.png")

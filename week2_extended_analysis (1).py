"""
Extended analysis: mood, sleep, difference scores, and robustness checks.

Follows week1_aggression_hormones.py using the same dataset
(OpenNeuro ds005115, 40 sessions from one adult male).

Adds:
    1. Same-day anger (POMS) and sleep quality (PSQI) as predictors
    2. Difference scores, which remove the linear trend in aggression
    3. HAC (Newey-West) standard errors to account for temporal
       autocorrelation between nearby sessions
    4. The final figure reported in the README

Usage:
    python week2_extended_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

URL = ("https://raw.githubusercontent.com/OpenNeuroDatasets/"
       "ds005115/main/participants.tsv")

df = pd.read_csv(URL, sep="\t", na_values="n/a", thousands=",")
df["session_num"] = df["session_id"].str.replace("ses-", "").astype(int)
df = df.sort_values("session_num").reset_index(drop=True)


def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------
# 1. Same-day mood and sleep as predictors
# ---------------------------------------------------------------
# Anger and sleep quality are measured at the same session as aggression,
# so they are tested alongside session order and time of day.
header("1. Anger and sleep quality as predictors")
m1 = smf.ols(
    "aggression ~ poms_anger + psqi + session_num + C(time_of_day)",
    data=df
).fit()
print(m1.summary().tables[1])


# ---------------------------------------------------------------
# 2. Difference scores
# ---------------------------------------------------------------
# Aggression declined monotonically across the study. Analyzing the change
# between consecutive sessions removes that trend, so any remaining
# association reflects session-to-session covariation rather than shared
# drift over time.
header("2. Difference scores (change between consecutive sessions)")
cols = ["aggression", "poms_anger", "perceived_stress", "psqi",
        "total_testosterone_saliva", "cortisol_saliva"]
diffs = df[cols].diff().dropna()
print(f"Sessions analyzed: {len(diffs)}\n")

print("Correlation of each change score with change in aggression:")
print(diffs.corr()["aggression"].drop("aggression").round(2))

print("\nChange in aggression predicted by change in hormones:")
m2 = smf.ols(
    "aggression ~ total_testosterone_saliva + cortisol_saliva", data=diffs
).fit()
print(m2.summary().tables[1])

print("\nChange in aggression predicted by change in mood and stress:")
m3 = smf.ols("aggression ~ poms_anger + perceived_stress", data=diffs).fit()
print(m3.summary().tables[1])


# ---------------------------------------------------------------
# 3. Robustness of the session effect
# ---------------------------------------------------------------
# Repeated measures taken close together in time are often correlated, which
# can understate standard errors. HAC standard errors correct for this.
header("3. Session effect with HAC standard errors")
m4 = smf.ols("aggression ~ session_num + C(time_of_day)", data=df).fit(
    cov_type="HAC", cov_kwds={"maxlags": 1}
)
print(m4.summary().tables[1])


# ---------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(9, 5.5))

sns.regplot(data=df, x="session_num", y="aggression", scatter=False,
            color="0.4", line_kws={"linewidth": 2}, ax=ax)
sns.scatterplot(data=df, x="session_num", y="aggression", hue="time_of_day",
                palette={"Morning": "#E69F00", "Evening": "#0072B2"},
                s=110, edgecolor="white", linewidth=1.2, ax=ax)

ax.set_xlabel("Session number")
ax.set_ylabel("Aggression (Buss-Perry)")
ax.set_title("Aggression declined across 40 sessions in a single participant",
             fontsize=15, pad=14)
ax.legend(title="Time of day", frameon=True)
fig.tight_layout()
fig.savefig("figure1_aggression_decline.png", dpi=300)
print("\nSaved figure1_aggression_decline.png")

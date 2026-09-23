"""
Week 2: Extended analysis of hormones, mood, sleep, and aggression
Dataset: OpenNeuro ds005115 ("28andHe"), 40 sessions from one adult male

What this adds to week 1:
    1. Two new predictors: same-day anger (POMS) and sleep quality (PSQI)
    2. Difference scores, which remove the downward trend in aggression
    3. Robust standard errors that account for measurements being
       close together in time
    4. One clean figure suitable for a report or poster

How to run:
    python week2_extended_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

URL = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds005115/main/participants.tsv"
df = pd.read_csv(URL, sep="\t", na_values="n/a", thousands=",")
df["session_num"] = df["session_id"].str.replace("ses-", "").astype(int)
df = df.sort_values("session_num").reset_index(drop=True)


def header(text):
    """Print a section title so the output is easy to read."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------
# ANALYSIS 1: Do same-day mood and sleep predict aggression?
# ---------------------------------------------------------------
# Hormones failed in week 1. Anger and sleep are psychological
# measures taken the same day, so they might do better.
header("1. Anger and sleep quality as predictors")
m1 = smf.ols(
    "aggression ~ poms_anger + psqi + session_num + C(time_of_day)",
    data=df
).fit()
print(m1.summary().tables[1])


# ---------------------------------------------------------------
# ANALYSIS 2: Difference scores
# ---------------------------------------------------------------
# Aggression drifted downward all study long, which can create fake
# correlations. Instead of raw values, we analyze the CHANGE from one
# session to the next (session 2 minus session 1, and so on). If a
# predictor truly moves with aggression, the changes should track too.
header("2. Difference scores (change from one session to the next)")
cols = ["aggression", "poms_anger", "perceived_stress", "psqi",
        "total_testosterone_saliva", "cortisol_saliva"]
diffs = df[cols].diff().dropna()
print(f"Sessions analyzed: {len(diffs)} (one fewer, since the first has "
      "nothing to compare to)\n")

print("Correlation of each change with the change in aggression:")
print(diffs.corr()["aggression"].drop("aggression").round(2))

print("\nModel: change in aggression predicted by changes in hormones")
m2 = smf.ols(
    "aggression ~ total_testosterone_saliva + cortisol_saliva", data=diffs
).fit()
print(m2.summary().tables[1])

print("\nModel: change in aggression predicted by changes in mood and stress")
m3 = smf.ols("aggression ~ poms_anger + perceived_stress", data=diffs).fit()
print(m3.summary().tables[1])


# ---------------------------------------------------------------
# ANALYSIS 3: Is the session effect trustworthy?
# ---------------------------------------------------------------
# Measurements taken close together in time tend to resemble each
# other (autocorrelation), which can make p-values look too good.
# HAC standard errors correct for that.
header("3. Session effect with robust (HAC) standard errors")
m4 = smf.ols("aggression ~ session_num + C(time_of_day)", data=df).fit(
    cov_type="HAC", cov_kwds={"maxlags": 1}
)
print(m4.summary().tables[1])


# ---------------------------------------------------------------
# FIGURE: the main result, in one clean panel
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

# ---------------------------------------------------------------
# YOUR TURN
# ---------------------------------------------------------------
# 1. Add poms_fatigue or steps to one of the models. Anything significant?
# 2. In analysis 2, which change score has the LARGEST correlation with
#    change in aggression? Is it statistically significant?
# 3. Write one sentence explaining why difference scores are a fairer
#    test than the raw correlations from week 1.

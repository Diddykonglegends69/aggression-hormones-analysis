"""
Week 1 starter project: Hormones and aggression in one person over time
Dataset: OpenNeuro ds005115 ("28andHe")

Research question:
    On days when this person's testosterone or cortisol is higher,
    do they also report more aggression?

How to run:
    1. Install packages (once):  pip install pandas matplotlib seaborn statsmodels
    2. Run:                      python week1_aggression_hormones.py
The script downloads the data itself, so you don't need to find the file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats

# ---------------------------------------------------------------
# STEP 1: Load the data
# ---------------------------------------------------------------
# The file is tab-separated (sep="\t"), missing values are written
# as "n/a", and some numbers have commas like "3,541".
URL = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds005115/main/participants.tsv"
df = pd.read_csv(URL, sep="\t", na_values="n/a", thousands=",")

print("Rows and columns:", df.shape)
print(df.head())

# ---------------------------------------------------------------
# STEP 2: Clean it up
# ---------------------------------------------------------------
# "ses-07" -> 7, so we can track change over time
df["session_num"] = df["session_id"].str.replace("ses-", "").astype(int)

# Saliva hormones were collected at every session (serum was not),
# so we use the saliva versions to keep all 40 sessions.
cols = ["session_num", "time_of_day", "aggression",
        "total_testosterone_saliva", "cortisol_saliva",
        "perceived_stress", "poms_anger"]
df = df[cols].dropna()

print("\nAverages by time of day:")
print(df.groupby("time_of_day")[["aggression", "total_testosterone_saliva",
                                 "cortisol_saliva"]].mean().round(2))

# ---------------------------------------------------------------
# STEP 3: The naive analysis (simple correlation)
# ---------------------------------------------------------------
r, p = stats.pearsonr(df["total_testosterone_saliva"], df["aggression"])
print(f"\nTestosterone vs aggression: r = {r:.2f}, p = {p:.4f}")

r, p = stats.pearsonr(df["cortisol_saliva"], df["aggression"])
print(f"Cortisol vs aggression:     r = {r:.2f}, p = {p:.4f}")

# ---------------------------------------------------------------
# STEP 4: Look for confounds (this is the important part)
# ---------------------------------------------------------------
# Hormones are naturally higher in the morning, and most early
# sessions were mornings. If aggression scores also drift down over
# the study, the correlation above could be fake.
r, p = stats.pearsonr(df["session_num"], df["aggression"])
print(f"\nSession number vs aggression: r = {r:.2f}, p = {p:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sns.lineplot(data=df, x="session_num", y="aggression", marker="o", ax=axes[0])
axes[0].set_title("Aggression score across sessions")
axes[0].set_xlabel("Session")

sns.scatterplot(data=df, x="total_testosterone_saliva", y="aggression",
                hue="time_of_day", s=70, ax=axes[1])
axes[1].set_title("Testosterone vs aggression, colored by time of day")
axes[1].set_xlabel("Salivary testosterone")

plt.tight_layout()
plt.savefig("aggression_plots.png", dpi=150)
print("\nSaved plots to aggression_plots.png")

# ---------------------------------------------------------------
# STEP 5: Control for the confounds with regression
# ---------------------------------------------------------------
# Does testosterone still predict aggression after accounting for
# session order and time of day?
model = smf.ols(
    "aggression ~ perceived_stress + session_num + C(time_of_day)",
    data=df
).fit()
print(model.summary())

# ---------------------------------------------------------------
# YOUR TURN (try these after you understand the code above)
# ---------------------------------------------------------------
# 1. Swap testosterone for cortisol_saliva in the model. What changes?
# 2. Does perceived_stress predict aggression once you control for session?
# 3. Only 40 data points from ONE person: write 2-3 sentences on
#    what you can and can't conclude from that.

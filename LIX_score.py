# %% [markdown]
# # Readability Score Analysis Notebook (LIX Version)

# %% [markdown]
# ### 1. Importing Data

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import scipy.stats as stats
from scipy.stats import levene, ttest_ind, mannwhitneyu
from nltk.tokenize import word_tokenize, sent_tokenize

# Create output directory
os.makedirs("plot_outputs", exist_ok=True)

# Ensure necessary models are available
nltk.download('punkt')

# Load dataset
df_all = pd.read_csv("df_desc_filt.csv", encoding="utf-8", low_memory=False)

# %% [markdown]
# ### 2. Compute LIX Readability Scores

# %%
def calculate_lix(text):
    if not isinstance(text, str) or not text.strip():
        return np.nan
    words = word_tokenize(text)
    words = [w for w in words if w.isalpha()]
    long_words = [w for w in words if len(w) > 6]
    sentences = sent_tokenize(text, language="danish")
    num_sentences = max(len(sentences), 1)
    if len(words) == 0:
        return np.nan
    return (len(words) / num_sentences) + (100 * len(long_words) / len(words))

df_rs = df_all.copy()
df_rs["summary_PostingCreated"] = pd.to_datetime(df_rs["summary_PostingCreated"], errors="coerce")
df_rs["Posting_Year"] = df_rs["summary_PostingCreated"].dt.year
df_rs["Period_Group"] = df_rs["Posting_Year"].apply(lambda x: "2022" if x == 2022 else ("2024/2025" if x in [2024, 2025] else None))
df_rs_grouped = df_rs[df_rs["Period_Group"].notnull()].copy()
df_rs_grouped["lix"] = df_rs_grouped["details_JobPositionPosting_JobPositionInformation_Purpose"].apply(calculate_lix)
df_rs_grouped = df_rs_grouped[df_rs_grouped["lix"].between(0, 100)]

# Save summary stats
summary_stats = df_rs_grouped.groupby("Period_Group")["lix"].describe()
summary_stats.to_csv("plot_outputs/lix_summary_stats.csv")
print(summary_stats)

# %% [markdown]
# ### 3. Distribution Plots and Q–Q Plots

# %%
# Faceted histogram
g = sns.FacetGrid(df_rs_grouped, col="Period_Group", col_order=["2022", "2024/2025"],
                  height=5, aspect=1.2, sharex=True, sharey=True)
g.map(sns.histplot, "lix", bins=30, kde=True, color="skyblue")
g.set_axis_labels("LIX Score", "Frequency")
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Distribution of LIX Scores by Period Group", fontsize=14)

path_hist = os.path.join("plot_outputs", "lix_score_distribution_by_period_group.png")
plt.savefig(path_hist, dpi=300)
plt.show()
print(f"Plot saved to '{path_hist}'")

# Q-Q plots
group_2022 = df_rs_grouped[df_rs_grouped["Period_Group"] == "2022"]["lix"].dropna()
group_2024_2025 = df_rs_grouped[df_rs_grouped["Period_Group"] == "2024/2025"]["lix"].dropna()

def save_qqplot(data, title, filename, log=False):
    plt.figure()
    stats.probplot(np.log(data) if log else data, dist="norm", plot=plt)
    plt.title(title)
    path = os.path.join("plot_outputs", filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved Q-Q plot: {path}")

save_qqplot(group_2022, "Q-Q Plot for 2022 LIX Scores", "qqplot_2022_raw.png")
save_qqplot(group_2024_2025, "Q-Q Plot for 2024/2025 LIX Scores", "qqplot_2024_2025_raw.png")
save_qqplot(group_2022, "Q-Q Plot (Log) - 2022", "qqplot_2022_log.png", log=True)
save_qqplot(group_2024_2025, "Q-Q Plot (Log) - 2024/2025", "qqplot_2024_2025_log.png", log=True)

# %% [markdown]
# ### 4. Statistical Testing

# %%
results = []

# Levene’s Test
results.append("Levene’s Test:")
stat, p_value = levene(group_2022, group_2024_2025)
results.append(f"Raw: stat = {stat:.3f}, p = {p_value:.4f}")
stat, p_value = levene(np.log(group_2022), np.log(group_2024_2025))
results.append(f"Log: stat = {stat:.3f}, p = {p_value:.4f}")

# Welch’s T-test
results.append("\nWelch’s T-test:")
stat, p_value = ttest_ind(group_2022, group_2024_2025, equal_var=False)
results.append(f"Raw: stat = {stat:.3f}, p = {p_value:.4f}")
stat, p_value = ttest_ind(np.log(group_2022), np.log(group_2024_2025), equal_var=False)
results.append(f"Log: stat = {stat:.3f}, p = {p_value:.4f}")

# Mann–Whitney U Test
results.append("\nMann–Whitney U Test:")
stat, p_value = mannwhitneyu(group_2022, group_2024_2025, alternative="two-sided")
results.append(f"Raw: U = {stat}, p = {p_value:.4f}")
stat, p_value = mannwhitneyu(np.log(group_2022), np.log(group_2024_2025), alternative="two-sided")
results.append(f"Log: U = {stat}, p = {p_value:.4f}")

# Cohen’s d
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*x.std(ddof=1)**2 + (ny-1)*y.std(ddof=1)**2) / (nx+ny-2))
    return (x.mean() - y.mean()) / pooled_std

results.append("\nCohen's d:")
d_raw = cohens_d(group_2024_2025, group_2022)
d_log = cohens_d(np.log(group_2024_2025), np.log(group_2022))
results.append(f"Raw: d = {d_raw:.3f}")
results.append(f"Log: d = {d_log:.3f}")

# Print and save results
for line in results:
    print(line)

with open("plot_outputs/lix_stats_results.txt", "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

# %% [markdown]
# ### 5. Boxplot for Group Comparison

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df_rs_grouped,
    x='Period_Group',
    y='lix',
    order=["2022", "2024/2025"],
    palette='pastel'
)
plt.grid(axis='y', linestyle='--', alpha=0.6)

means = df_rs_grouped.groupby("Period_Group")["lix"].mean()
for i, period in enumerate(["2022", "2024/2025"]):
    plt.scatter(i, means[period], color='red', zorder=10, marker='D', s=50)

plt.title("LIX Score by Period Group", fontsize=14)
plt.xlabel("Period")
plt.ylabel("LIX Score")
plt.ylim(0, 100)

boxplot_path = os.path.join("plot_outputs", "lix_score_boxplot_by_period_group.png")
plt.savefig(boxplot_path, dpi=300)
plt.show()
print(f"Boxplot saved to: {boxplot_path}")

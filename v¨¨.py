# %%
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load CSV
gender_distribution = pd.read_csv("gendered_title_counts_only.csv", low_memory=False)
gender_distribution_percent = pd.read_csv("gendered_title_percents_only.csv", low_memory=False)


# %%

# --- NEW: Filter to only gendered titles (drop 'none' and 'both') ---
df_gendered_only = df_gtot_grouped[df_gtot_grouped['gender_category'].isin(['feminine', 'masculine'])].copy()

# --- Count and proportion for filtered data ---
filtered_counts = df_gendered_only.groupby(['Period_Group', 'gender_category']).size().unstack(fill_value=0)
filtered_percents = filtered_counts.div(filtered_counts.sum(axis=1), axis=0) * 100

# --- Save filtered outputs ---
filtered_counts.to_csv(os.path.join(output_dir, "gendered_title_counts_only.csv"))
filtered_percents.to_csv(os.path.join(output_dir, "gendered_title_percents_only.csv"))


# %% [markdown]
# 

# %%
# --- Plot full stacked bar charts (all categories) ---
gender_distribution.plot(kind='bar', stacked=True)
plt.title("Gendered Job Title Counts by Period")
plt.ylabel("Number of Job Titles")
plt.xlabel("Period Group")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "gender_counts_by_period.png"))
plt.close()

gender_distribution_percent.plot(kind='bar', stacked=True)
plt.title("Gendered Job Title Percentages by Period")
plt.ylabel("Percentage (%)")
plt.xlabel("Period Group")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "gender_percent_by_period.png"))
plt.close()

# %%
from statsmodels.stats.proportion import proportions_ztest

# Example: comparing feminine proportions
count_fem_2022 = filtered_counts.loc['2022', 'feminine']
count_fem_2024 = filtered_counts.loc['2024_2025', 'feminine']
total_2022 = filtered_counts.loc['2022'].sum()
total_2024 = filtered_counts.loc['2024_2025'].sum()

z_stat, p_value = proportions_ztest([count_fem_2022, count_fem_2024], [total_2022, total_2024])

# %%
# Loads CSV file 
import pandas as pd 

# Load the CSV file
file_path = 'filtered_counts.csv'




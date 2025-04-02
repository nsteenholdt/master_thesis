import pandas as pd
import re
import os
import matplotlib.pyplot as plt

# --- Load CSV ---
df = pd.read_csv("df_desc.csv")

# --- Ensure expected columns exist ---
if 'summary_Occupation' not in df.columns or 'summary_PostingCreated' not in df.columns:
    raise ValueError("Expected columns 'summary_Occupation' and 'summary_PostingCreated' not found in the CSV.")

# --- Convert posting date to datetime ---
df['summary_PostingCreated'] = pd.to_datetime(df['summary_PostingCreated'], errors='coerce')

# --- Define suffixes ---
feminine_suffixes = ["dame", "frue", "inde", "trice", "øse", "ske", "kone", "ert", "mor", "jomfru"]
masculine_suffixes = ["mand"]

# --- Gender classification function ---
def classify_gender(title):
    title = str(title).strip().lower()
    words = re.findall(r'\b\w+\b', title)

    for word in words:
        if any(word.endswith(suffix) for suffix in feminine_suffixes):
            return 'feminine'
        if any(word.endswith(suffix) for suffix in masculine_suffixes):
            return 'masculine'

    return 'none'

# --- Apply classification ---
df['gender_category'] = df['summary_Occupation'].apply(classify_gender)

# --- Extract year and group into periods ---
df['Posting_Year'] = df['summary_PostingCreated'].dt.year
df['Period_Group'] = df['Posting_Year'].apply(
    lambda y: '2022' if y == 2022 else ('2024_2025' if y in [2024, 2025] else None)
)

# --- Drop rows not in comparison periods ---
df_grouped = df[df['Period_Group'].notnull()]

# --- Count gender categories per group ---
gender_distribution = df_grouped.groupby(['Period_Group', 'gender_category']).size().unstack(fill_value=0)

# --- Also calculate proportions ---
gender_distribution_percent = gender_distribution.div(gender_distribution.sum(axis=1), axis=0) * 100

# --- Save outputs ---
output_dir = "stats_outputs"
os.makedirs(output_dir, exist_ok=True)

gender_distribution.to_csv(os.path.join(output_dir, "gender_counts_by_period.csv"))
gender_distribution_percent.to_csv(os.path.join(output_dir, "gender_percent_by_period.csv"))

# --- Visualization output ---
plot_dir = "plot_outputs"
os.makedirs(plot_dir, exist_ok=True)

# Plot counts
gender_distribution.plot(kind='bar', stacked=True)
plt.title("Gendered Job Title Counts by Period")
plt.ylabel("Number of Job Titles")
plt.xlabel("Period Group")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "gender_counts_by_period.png"))
plt.close()

# Plot percentages
gender_distribution_percent.plot(kind='bar', stacked=True)
plt.title("Gendered Job Title Percentages by Period")
plt.ylabel("Percentage (%)")
plt.xlabel("Period Group")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "gender_percent_by_period.png"))
plt.close()

import ace_tools as tools; tools.display_dataframe_to_user(name="Gendered Job Titles by Period", dataframe=gender_distribution)

print(f"Done. Results saved in '{output_dir}' and plots in '{plot_dir}'.")

# --- Visualization output ---
plot_dir = "plot_outputs"
os.makedirs(plot_dir, exist_ok=True)

# Plot counts
gender_distribution.plot(kind='bar', stacked=True)
plt.title("Gendered Job Title Counts by Period")
plt.ylabel("Number of Job Titles")
plt.xlabel("Period Group")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "gender_counts_by_period.png"))
plt.close()

# Plot percentages
gender_distribution_percent.plot(kind='bar', stacked=True)
plt.title("Gendered Job Title Percentages by Period")
plt.ylabel("Percentage (%)")
plt.xlabel("Period Group")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "gender_percent_by_period.png"))
plt.close()

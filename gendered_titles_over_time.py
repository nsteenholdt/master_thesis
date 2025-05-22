import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load CSV
df_gtot = pd.read_csv("df_desc_filt.csv", low_memory=False)

# --- Ensure expected columns exist ---
if 'summary_Occupation' not in df_gtot.columns or 'summary_PostingCreated' not in df_gtot.columns:
    raise ValueError("Expected columns 'summary_Occupation' and 'summary_PostingCreated' not found.")

# --- Convert posting date to datetime ---
df_gtot['summary_PostingCreated'] = pd.to_datetime(df_gtot['summary_PostingCreated'], errors='coerce')

# --- Define suffixes ---
feminine_suffixes = ["dame", "frue", "inde", "trice", "øse", "ske", "kone", "ert", "mor", "jomfru"]
masculine_suffixes = ["mand", "svend"]

# --- Gender classification function ---
def classify_gender(title):
    title = str(title).strip().lower()
    words = re.findall(r'\b\w+\b', title)
    has_feminine = any(word.endswith(suffix) for word in words for suffix in feminine_suffixes)
    has_masculine = any(word.endswith(suffix) for word in words for suffix in masculine_suffixes)

    if has_feminine and has_masculine:
        return 'both'
    elif has_feminine:
        return 'feminine'
    elif has_masculine:
        return 'masculine'
    else:
        return 'none'

# --- Apply classification ---
df_gtot['gender_category'] = df_gtot['summary_Occupation'].apply(classify_gender)

# --- Logging counts of gender categories ---
print("\nGender category counts (all periods):")
print(df_gtot['gender_category'].value_counts())

# --- Extract year and group into periods ---
df_gtot['Posting_Year'] = df_gtot['summary_PostingCreated'].dt.year
df_gtot['Period_Group'] = df_gtot['Posting_Year'].apply(
    lambda y: '2022' if y == 2022 else ('2024_2025' if y in [2024, 2025] else None)
)

# --- Drop rows not in comparison periods ---
df_gtot_grouped = df_gtot[df_gtot['Period_Group'].notnull()].copy()

# --- Count gender categories per group (including 'none') ---
gender_distribution = df_gtot_grouped.groupby(['Period_Group', 'gender_category']).size().unstack(fill_value=0)
gender_distribution_percent = gender_distribution.div(gender_distribution.sum(axis=1), axis=0) * 100

# --- Save outputs ---
output_dir = "stats_outputs"
plot_dir = "plot_outputs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

gender_distribution.to_csv(os.path.join(output_dir, "gender_counts_by_period.csv"))
gender_distribution_percent.to_csv(os.path.join(output_dir, "gender_percent_by_period.csv"))

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

# --- NEW: Filter to only gendered titles (drop 'none' and 'both') ---
df_gendered_only = df_gtot_grouped[df_gtot_grouped['gender_category'].isin(['feminine', 'masculine'])].copy()

# --- Count and proportion for filtered data ---
filtered_counts = df_gendered_only.groupby(['Period_Group', 'gender_category']).size().unstack(fill_value=0)
filtered_percents = filtered_counts.div(filtered_counts.sum(axis=1), axis=0) * 100

# --- Save filtered outputs ---
filtered_counts.to_csv(os.path.join(output_dir, "gendered_title_counts_only.csv"))
filtered_percents.to_csv(os.path.join(output_dir, "gendered_title_percents_only.csv"))

# --- Plot: Counts of only gendered job titles ---
filtered_counts.plot(kind='bar', stacked=True)
plt.title("Gendered Job Title Counts (Only Gendered Titles)")
plt.ylabel("Number of Job Titles")
plt.xlabel("Period Group")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "gendered_counts_only.png"))
plt.close()

# --- Plot: Percentages of only gendered job titles ---
filtered_percents.plot(kind='bar', stacked=True)
plt.title("Gendered Job Title Percentages (Only Gendered Titles)")
plt.ylabel("Percentage (%)")
plt.xlabel("Period Group")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "gendered_percents_only.png"))
plt.close()

# --- Chi-squared test on filtered counts ---
chi2, p, dof, expected = chi2_contingency(filtered_counts)
print("\nChi-squared test on gendered title distribution (feminine vs masculine only):")
print(f"  Chi² = {chi2:.3f}, p = {p:.4f}, dof = {dof}")
print(f"  Expected counts:\n{pd.DataFrame(expected, columns=filtered_counts.columns, index=filtered_counts.index)}")

print(f"\nDone. Results saved in '{output_dir}' and plots in '{plot_dir}'.")

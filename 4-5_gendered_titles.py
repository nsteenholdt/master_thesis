import pandas as pd
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

# Load CSV
df_gtot = pd.read_csv("df_desc_filt.csv", low_memory=False)

# --- DATA QUALITY FIXES ---
def clean_title(title):
    title = str(title).strip()
    # Fix common Danish character encoding issues
    title = title.replace('Ã¦', 'æ').replace('Ã¸', 'ø').replace('Ã¥', 'å')
    # Remove extra whitespace
    title = re.sub(r'\s+', ' ', title)
    return title

def is_valid_title(title):
    if pd.isna(title) or str(title).strip() == '':
        return False
    if len(str(title).strip()) < 3:  # Too short to be meaningful
        return False
    if str(title).strip().lower() in ['nan', 'none', 'null']:
        return False
    return True

# Apply cleaning
df_gtot['summary_Occupation'] = df_gtot['summary_Occupation'].apply(clean_title)
# Filter out invalid titles
df_gtot = df_gtot[df_gtot['summary_Occupation'].apply(is_valid_title)]

# --- Ensure expected columns exist ---
if 'summary_Occupation' not in df_gtot.columns or 'summary_PostingCreated' not in df_gtot.columns:
    raise ValueError("Expected columns 'summary_Occupation' and 'summary_PostingCreated' not found.")

# --- Convert posting date to datetime ---
df_gtot['summary_PostingCreated'] = pd.to_datetime(df_gtot['summary_PostingCreated'], errors='coerce')

# --- Define suffixes ---
feminine_suffixes = ["dame", "frue", "inde", "ine", "ette", "trice", "øse", "kone", "ert", "mor", "jomfru", "esse", "isse"]
masculine_suffixes = ["mand", "svend", "ør", "mester", "ist", "nom"]

# --- Gender classification function ---
def classify_gender_enhanced(title):
    title = str(title).strip().lower()
    
    # Original word-boundary approach
    words = re.findall(r'\b\w+\b', title)
    word_based_fem = any(word.endswith(suffix) for word in words for suffix in feminine_suffixes)
    word_based_masc = any(word.endswith(suffix) for word in words for suffix in masculine_suffixes)
    
    # Compound word approach (suffix anywhere in title)
    compound_fem = any(suffix in title for suffix in feminine_suffixes)
    compound_masc = any(suffix in title for suffix in masculine_suffixes)
    
    # Combine both approaches
    has_feminine = word_based_fem or compound_fem
    has_masculine = word_based_masc or compound_masc

    if has_feminine and has_masculine:
        return 'both'
    elif has_feminine:
        return 'feminine'
    elif has_masculine:
        return 'masculine'
    else:
        return 'none'

# --- Apply classification ---
df_gtot['gender_category'] = df_gtot['summary_Occupation'].apply(classify_gender_enhanced)

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

# --- Filter to only gendered titles (drop 'none' and 'both') ---
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

# =============================================================================
# ENHANCED STATISTICAL ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("STATISTICAL ANALYSIS: COMPARING GENDER PROPORTIONS BETWEEN PERIODS")
print("="*80)

# Extract counts for analysis
count_fem_2022 = filtered_counts.loc['2022', 'feminine']
count_fem_2024 = filtered_counts.loc['2024_2025', 'feminine']
count_masc_2022 = filtered_counts.loc['2022', 'masculine']
count_masc_2024 = filtered_counts.loc['2024_2025', 'masculine']

total_2022 = filtered_counts.loc['2022'].sum()
total_2024 = filtered_counts.loc['2024_2025'].sum()

# Calculate proportions
prop_fem_2022 = count_fem_2022 / total_2022
prop_fem_2024 = count_fem_2024 / total_2024
prop_masc_2022 = count_masc_2022 / total_2022
prop_masc_2024 = count_masc_2024 / total_2024

# --- DESCRIPTIVE STATISTICS ---
print("\nDESCRIPTIVE STATISTICS:")
print("-" * 40)
print(f"2022 Total Gendered Titles: {total_2022:,}")
print(f"  - Feminine: {count_fem_2022:,} ({prop_fem_2022:.1%})")
print(f"  - Masculine: {count_masc_2022:,} ({prop_masc_2022:.1%})")
print(f"\n2024-2025 Total Gendered Titles: {total_2024:,}")
print(f"  - Feminine: {count_fem_2024:,} ({prop_fem_2024:.1%})")
print(f"  - Masculine: {count_masc_2024:,} ({prop_masc_2024:.1%})")

# --- CHECK ASSUMPTIONS FOR NORMAL APPROXIMATION ---
print("\nASSUMPTION CHECKS (n*p ≥ 5 and n*(1-p) ≥ 5 for normal approximation):")
print("-" * 70)
print("Feminine proportions:")
print(f"  2022: n*p = {total_2022 * prop_fem_2022:.1f}, n*(1-p) = {total_2022 * (1-prop_fem_2022):.1f}")
print(f"  2024-25: n*p = {total_2024 * prop_fem_2024:.1f}, n*(1-p) = {total_2024 * (1-prop_fem_2024):.1f}")
print("Masculine proportions:")
print(f"  2022: n*p = {total_2022 * prop_masc_2022:.1f}, n*(1-p) = {total_2022 * (1-prop_masc_2022):.1f}")
print(f"  2024-25: n*p = {total_2024 * prop_masc_2024:.1f}, n*(1-p) = {total_2024 * (1-prop_masc_2024):.1f}")

# Function to calculate Cohen's h effect size
def cohens_h(p1, p2):
    """Calculate Cohen's h effect size for proportions"""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

def interpret_cohens_h(h):
    """Interpret Cohen's h effect size"""
    abs_h = abs(h)
    if abs_h < 0.2:
        return "negligible"
    elif abs_h < 0.5:
        return "small"
    elif abs_h < 0.8:
        return "medium"
    else:
        return "large"

# --- FEMININE PROPORTION ANALYSIS ---
print("\n" + "="*50)
print("FEMININE PROPORTION ANALYSIS")
print("="*50)

# Z-test for feminine proportions
z_stat_fem, p_value_fem = proportions_ztest([count_fem_2022, count_fem_2024], 
                                           [total_2022, total_2024])

# Effect size
effect_size_fem = cohens_h(prop_fem_2022, prop_fem_2024)
percentage_point_diff_fem = (prop_fem_2024 - prop_fem_2022) * 100

# Confidence interval for difference
ci_low_fem, ci_high_fem = confint_proportions_2indep(count_fem_2022, total_2022, 
                                                    count_fem_2024, total_2024, 
                                                    method='wald')

print(f"Feminine proportion 2022: {prop_fem_2022:.3f} ({prop_fem_2022:.1%})")
print(f"Feminine proportion 2024-25: {prop_fem_2024:.3f} ({prop_fem_2024:.1%})")
print(f"Difference: {percentage_point_diff_fem:+.2f} percentage points")
print(f"Z-statistic: {z_stat_fem:.3f}")
print(f"P-value: {p_value_fem:.4f}")
print(f"95% CI for difference: [{ci_low_fem:.4f}, {ci_high_fem:.4f}]")
print(f"Cohen's h: {effect_size_fem:.3f} ({interpret_cohens_h(effect_size_fem)} effect)")

# --- MASCULINE PROPORTION ANALYSIS ---
print("\n" + "="*50)
print("MASCULINE PROPORTION ANALYSIS")
print("="*50)

# Z-test for masculine proportions
z_stat_masc, p_value_masc = proportions_ztest([count_masc_2022, count_masc_2024], 
                                             [total_2022, total_2024])

# Effect size
effect_size_masc = cohens_h(prop_masc_2022, prop_masc_2024)
percentage_point_diff_masc = (prop_masc_2024 - prop_masc_2022) * 100

# Confidence interval for difference
ci_low_masc, ci_high_masc = confint_proportions_2indep(count_masc_2022, total_2022, 
                                                      count_masc_2024, total_2024, 
                                                      method='wald')

print(f"Masculine proportion 2022: {prop_masc_2022:.3f} ({prop_masc_2022:.1%})")
print(f"Masculine proportion 2024-25: {prop_masc_2024:.3f} ({prop_masc_2024:.1%})")
print(f"Difference: {percentage_point_diff_masc:+.2f} percentage points")
print(f"Z-statistic: {z_stat_masc:.3f}")
print(f"P-value: {p_value_masc:.4f}")
print(f"95% CI for difference: [{ci_low_masc:.4f}, {ci_high_masc:.4f}]")
print(f"Cohen's h: {effect_size_masc:.3f} ({interpret_cohens_h(effect_size_masc)} effect)")

# --- MULTIPLE COMPARISONS ADJUSTMENT ---
print("\n" + "="*50)
print("MULTIPLE COMPARISONS ADJUSTMENT")
print("="*50)
bonferroni_alpha = 0.05 / 2  # Two tests (feminine and masculine)
print(f"Bonferroni-adjusted significance level: α = {bonferroni_alpha:.3f}")
print(f"Feminine test: {'Significant' if p_value_fem < bonferroni_alpha else 'Not significant'} after Bonferroni correction")
print(f"Masculine test: {'Significant' if p_value_masc < bonferroni_alpha else 'Not significant'} after Bonferroni correction")

# --- SUMMARY TABLE ---
summary_stats = pd.DataFrame({
    'Period': ['2022', '2024-2025'],
    'Total_Gendered': [total_2022, total_2024],
    'Feminine_Count': [count_fem_2022, count_fem_2024],
    'Feminine_Prop': [prop_fem_2022, prop_fem_2024],
    'Masculine_Count': [count_masc_2022, count_masc_2024],
    'Masculine_Prop': [prop_masc_2022, prop_masc_2024]
})

test_results = pd.DataFrame({
    'Test': ['Feminine Proportion', 'Masculine Proportion'],
    'Z_Statistic': [z_stat_fem, z_stat_masc],
    'P_Value': [p_value_fem, p_value_masc],
    'Effect_Size_Cohens_h': [effect_size_fem, effect_size_masc],
    'Effect_Interpretation': [interpret_cohens_h(effect_size_fem), interpret_cohens_h(effect_size_masc)],
    'Percentage_Point_Diff': [percentage_point_diff_fem, percentage_point_diff_masc],
    'Significant_Bonferroni': [p_value_fem < bonferroni_alpha, p_value_masc < bonferroni_alpha]
})

# Save summary tables
summary_stats.to_csv(os.path.join(output_dir, "summary_statistics.csv"), index=False)
test_results.to_csv(os.path.join(output_dir, "statistical_test_results.csv"), index=False)

print(f"\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Results saved in '{output_dir}' directory:")
print("- summary_statistics.csv")
print("- statistical_test_results.csv")
print("- gender_counts_by_period.csv")
print("- gender_percent_by_period.csv")
print("- gendered_title_counts_only.csv")
print("- gendered_title_percents_only.csv")
print(f"\nPlots saved in '{plot_dir}' directory:")
print("- gender_counts_by_period.png")
print("- gender_percent_by_period.png")
print("- gendered_counts_only.png")
print("- gendered_percents_only.png")
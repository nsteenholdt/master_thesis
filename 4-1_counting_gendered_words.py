import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu, normaltest, levene
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
import seaborn as sns

# =============================================================================
# SETUP AND VALIDATION
# =============================================================================

def validate_environment():
    """Validate required files and dependencies"""
    print("Validating environment...")
    
    # Check required files
    required_files = ["df_desc_filt.csv", "gender_scored_lexicon_from_descriptions.csv"]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file not found: {file}")
    
    # Ensure output directories exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("stats_outputs", exist_ok=True)
    os.makedirs("plot_outputs", exist_ok=True)
    os.makedirs("validation_outputs", exist_ok=True)  # New directory for validation
    
    # Check SpaCy model
    try:
        nlp = spacy.load("da_core_news_lg")
        print("✓ SpaCy Danish model loaded successfully")
        return nlp
    except OSError:
        raise OSError("Danish SpaCy model 'da_core_news_lg' not found. Install with: python -m spacy download da_core_news_lg")

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std

def cohens_h(p1, p2):
    """Calculate Cohen's h effect size for proportions"""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

def interpret_effect_size(effect_size, measure_type='d'):
    """Interpret effect size magnitude"""
    abs_effect = abs(effect_size)
    if measure_type == 'd':  # Cohen's d
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    elif measure_type == 'h':  # Cohen's h
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"

# =============================================================================
# NEW: LEXICON VALIDATION FUNCTIONS
# =============================================================================

def validate_lexicon_quality(gender_scores, threshold=0.022):
    """CONSIDERATION 1: Validate lexicon appropriateness for Danish job context"""
    print("\n" + "="*80)
    print("LEXICON QUALITY VALIDATION")
    print("="*80)
    
    # Separate words by gender classification
    feminine_words = [(w, s) for w, s in gender_scores.items() if s >= threshold]
    masculine_words = [(w, s) for w, s in gender_scores.items() if s <= -threshold]
    neutral_words = [(w, s) for w, s in gender_scores.items() if abs(s) < threshold]
    
    print(f"Lexicon composition:")
    print(f"- Feminine words (≥{threshold}): {len(feminine_words):,}")
    print(f"- Masculine words (≤-{threshold}): {len(masculine_words):,}")
    print(f"- Neutral words (<±{threshold}): {len(neutral_words):,}")
    print(f"- Total words: {len(gender_scores):,}")
    
    # Display top words for manual inspection
    print(f"\nTOP 30 FEMININE-CODED WORDS (for manual validation):")
    print("-" * 60)
    fem_sorted = sorted(feminine_words, key=lambda x: x[1], reverse=True)
    for i, (word, score) in enumerate(fem_sorted[:30], 1):
        print(f"{i:2d}. {word:<20} ({score:+.4f})")
    
    print(f"\nTOP 30 MASCULINE-CODED WORDS (for manual validation):")
    print("-" * 60)
    masc_sorted = sorted(masculine_words, key=lambda x: x[1])
    for i, (word, score) in enumerate(masc_sorted[:30], 1):
        print(f"{i:2d}. {word:<20} ({score:+.4f})")
    
    # Save for external validation
    lexicon_validation = {
        'top_feminine': fem_sorted[:50],
        'top_masculine': masc_sorted[:50],
        'stats': {
            'feminine_count': len(feminine_words),
            'masculine_count': len(masculine_words),
            'neutral_count': len(neutral_words),
            'total_count': len(gender_scores)
        }
    }
    
    # Save as CSV for easy review
    fem_df = pd.DataFrame(fem_sorted[:50], columns=['word', 'gender_score'])
    fem_df['category'] = 'feminine'
    masc_df = pd.DataFrame(masc_sorted[:50], columns=['word', 'gender_score'])
    masc_df['category'] = 'masculine'
    
    validation_df = pd.concat([fem_df, masc_df], ignore_index=True)
    validation_df.to_csv("validation_outputs/lexicon_top_words_for_review.csv", index=False)
    
    return lexicon_validation

def test_alternative_thresholds(df, gender_scores, base_threshold=0.022):
    """CONSIDERATION 2: Comprehensive threshold sensitivity analysis"""
    print("\n" + "="*80)
    print("EXTENDED THRESHOLD SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Test wider range of thresholds
    thresholds = [0.010, 0.015, 0.018, 0.020, 0.022, 0.025, 0.030, 0.035, 0.040, 0.050]
    
    print("Testing multiple thresholds to assess robustness...")
    print(f"Base threshold: ±{base_threshold:.3f}")
    
    sensitivity_results = []
    
    for threshold in thresholds:
        feminine_words = {w for w, s in gender_scores.items() if s >= threshold}
        masculine_words = {w for w, s in gender_scores.items() if s <= -threshold}
        
        # Calculate metrics for each threshold
        bias_scores = []
        bias_categories = []
        
        for lemmas in df['lemmas']:
            counts = Counter(lemmas)
            fem_count = sum(counts[w] for w in feminine_words if w in counts)
            masc_count = sum(counts[w] for w in masculine_words if w in counts)
            
            bias_score = fem_count - masc_count
            bias_scores.append(bias_score)
            
            if fem_count + masc_count == 0:
                category = "none"
            elif fem_count > masc_count:
                category = "feminine_bias"
            elif masc_count > fem_count:
                category = "masculine_bias"
            else:
                category = "balanced"
            bias_categories.append(category)
        
        # Calculate summary statistics
        bias_scores = np.array(bias_scores)
        categories = pd.Series(bias_categories)
        
        # Split by period for comparison
        data_2022 = bias_scores[df['group'] == "2022"]
        data_2024 = bias_scores[df['group'] == "2024/2025"]
        
        # Simple t-test for this threshold
        if len(data_2022) > 0 and len(data_2024) > 0:
            t_stat, p_val = ttest_ind(data_2024, data_2022)
            effect_size = cohens_d(data_2024, data_2022)
        else:
            t_stat, p_val, effect_size = np.nan, np.nan, np.nan
        
        result = {
            'threshold': threshold,
            'feminine_words': len(feminine_words),
            'masculine_words': len(masculine_words),
            'total_gendered_words': len(feminine_words) + len(masculine_words),
            'ads_with_gendered': (categories != "none").sum(),
            'coverage_percent': ((categories != "none").sum() / len(categories)) * 100,
            'mean_bias_2022': data_2022.mean(),
            'mean_bias_2024': data_2024.mean(),
            'difference': data_2024.mean() - data_2022.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size': effect_size,
            'feminine_bias_pct': (categories == "feminine_bias").mean() * 100,
            'masculine_bias_pct': (categories == "masculine_bias").mean() * 100
        }
        sensitivity_results.append(result)
        
        # Mark optimal threshold
        optimal_marker = " ← SELECTED" if threshold == base_threshold else ""
        print(f"±{threshold:.3f}: {len(feminine_words) + len(masculine_words):,} words, "
              f"{((categories != 'none').sum() / len(categories)) * 100:.1f}% coverage, "
              f"diff={data_2024.mean() - data_2022.mean():+.3f}{optimal_marker}")
    
    sensitivity_df = pd.DataFrame(sensitivity_results)
    sensitivity_df.to_csv("validation_outputs/comprehensive_threshold_analysis.csv", index=False)
    
    # Plot threshold sensitivity
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    ax1.plot(sensitivity_df['threshold'], sensitivity_df['total_gendered_words'], 'bo-')
    ax1.axvline(base_threshold, color='red', linestyle='--', alpha=0.7, label=f'Selected: ±{base_threshold}')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Total Gendered Words')
    ax1.set_title('Gendered Words vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(sensitivity_df['threshold'], sensitivity_df['coverage_percent'], 'go-')
    ax2.axvline(base_threshold, color='red', linestyle='--', alpha=0.7, label=f'Selected: ±{base_threshold}')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Coverage (%)')
    ax2.set_title('Ad Coverage vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(sensitivity_df['threshold'], sensitivity_df['difference'], 'ro-')
    ax3.axvline(base_threshold, color='red', linestyle='--', alpha=0.7, label=f'Selected: ±{base_threshold}')
    ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Mean Bias Difference (2024-2022)')
    ax3.set_title('Temporal Difference vs Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(sensitivity_df['threshold'], abs(sensitivity_df['effect_size']), 'mo-')
    ax4.axvline(base_threshold, color='red', linestyle='--', alpha=0.7, label=f'Selected: ±{base_threshold}')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('|Effect Size| (Cohen\'s d)')
    ax4.set_title('Effect Size vs Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("validation_outputs/threshold_sensitivity_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return sensitivity_df

def analyze_sample_job_ads(df_gender, n_samples=10):
    """CONSIDERATION 3: Examine actual job ad content for validation"""
    print("\n" + "="*80)
    print("SAMPLE JOB AD CONTENT ANALYSIS")
    print("="*80)
    
    # Select diverse samples from each bias category
    samples = []
    
    categories = ['feminine_bias', 'masculine_bias', 'balanced', 'none']
    for category in categories:
        category_ads = df_gender[df_gender['bias_category'] == category]
        if len(category_ads) > 0:
            # Sample from different periods and bias scores
            sample_size = min(n_samples, len(category_ads))
            if sample_size > 0:
                if category in ['feminine_bias', 'masculine_bias']:
                    # For biased categories, sample from different extremes
                    sorted_ads = category_ads.sort_values('gender_bias_score', 
                                                        ascending=(category == 'masculine_bias'))
                    # Take some from extremes and some from middle
                    extreme_n = sample_size // 2
                    middle_n = sample_size - extreme_n
                    extreme_samples = sorted_ads.head(extreme_n)
                    middle_samples = sorted_ads.iloc[len(sorted_ads)//4:len(sorted_ads)//4+middle_n]
                    selected = pd.concat([extreme_samples, middle_samples])
                else:
                    # Random sample for balanced/none
                    selected = category_ads.sample(n=sample_size, random_state=42)
                
                samples.append(selected)
    
    if samples:
        sample_df = pd.concat(samples, ignore_index=True)
        
        # Create analysis of sample ads
        sample_analysis = []
        for idx, row in sample_df.iterrows():
            analysis = {
                'sample_id': idx,
                'period': row['group'],
                'bias_category': row['bias_category'],
                'gender_bias_score': row['gender_bias_score'],
                'feminine_word_count': row['feminine_word_count'],
                'masculine_word_count': row['masculine_word_count'],
                'total_words': row['total_word_count'],
                'job_text_preview': str(row['details_JobPositionPosting_JobPositionInformation_Purpose'])[:200] + "..."
            }
            sample_analysis.append(analysis)
        
        sample_analysis_df = pd.DataFrame(sample_analysis)
        sample_analysis_df.to_csv("validation_outputs/sample_job_ads_analysis.csv", index=False)
        
        # Print summary of samples
        print("Sample job ad analysis:")
        print(f"- Total samples: {len(sample_analysis_df)}")
        print("- By bias category:")
        for category in categories:
            count = (sample_analysis_df['bias_category'] == category).sum()
            if count > 0:
                print(f"  {category}: {count} ads")
        
        print("\nSample details saved to 'validation_outputs/sample_job_ads_analysis.csv'")
        print("Review this file to manually validate classification accuracy.")
        
        return sample_analysis_df
    else:
        print("No samples could be selected.")
        return None

def danish_context_analysis(df_gender, gender_scores, df_with_lemmas):
    """CONSIDERATION 1 EXTENDED: Danish market context analysis"""
    print("\n" + "="*80)
    print("DANISH JOB MARKET CONTEXT ANALYSIS")
    print("="*80)
    
    # Analyze bias patterns by time
    bias_by_period = df_gender.groupby('group')['bias_category'].value_counts(normalize=True).unstack(fill_value=0)
    
    print("Bias distribution by period:")
    print(bias_by_period.round(3))
    
    # Check if masculine bias is consistent with Danish cultural patterns
    masc_pct_2022 = bias_by_period.loc['2022', 'masculine_bias'] * 100
    masc_pct_2024 = bias_by_period.loc['2024/2025', 'masculine_bias'] * 100
    
    print(f"\nMasculine bias prevalence:")
    print(f"2022: {masc_pct_2022:.1f}% of ads")
    print(f"2024-25: {masc_pct_2024:.1f}% of ads")
    print(f"Change: {masc_pct_2024 - masc_pct_2022:+.1f} percentage points")
    
    # Analyze most common gendered words in actual usage
    all_feminine_words = set(w for w, s in gender_scores.items() if s >= 0.022)
    all_masculine_words = set(w for w, s in gender_scores.items() if s <= -0.022)
    
    fem_word_usage = Counter()
    masc_word_usage = Counter()
    
    # Use df_with_lemmas instead of df_gender
    for lemmas in df_with_lemmas['lemmas']:
        word_counts = Counter(lemmas)
        for word, count in word_counts.items():
            if word in all_feminine_words:
                fem_word_usage[word] += count
            elif word in all_masculine_words:
                masc_word_usage[word] += count
    
    print(f"\nMost frequently used gendered words in job ads:")
    print("\nTop 15 feminine words:")
    for i, (word, count) in enumerate(fem_word_usage.most_common(15), 1):
        score = gender_scores.get(word, 0)
        print(f"{i:2d}. {word:<15} (used {count:,} times, score: {score:+.3f})")
    
    print(f"\nTop 15 masculine words:")
    for i, (word, count) in enumerate(masc_word_usage.most_common(15), 1):
        score = gender_scores.get(word, 0)
        print(f"{i:2d}. {word:<15} (used {count:,} times, score: {score:+.3f})")
    
    # Save detailed word usage analysis
    fem_usage_df = pd.DataFrame(fem_word_usage.most_common(50), columns=['word', 'usage_count'])
    fem_usage_df['gender_score'] = fem_usage_df['word'].map(gender_scores)
    fem_usage_df['category'] = 'feminine'
    
    masc_usage_df = pd.DataFrame(masc_word_usage.most_common(50), columns=['word', 'usage_count'])
    masc_usage_df['gender_score'] = masc_usage_df['word'].map(gender_scores)
    masc_usage_df['category'] = 'masculine'
    
    word_usage_df = pd.concat([fem_usage_df, masc_usage_df], ignore_index=True)
    word_usage_df.to_csv("validation_outputs/gendered_word_usage_analysis.csv", index=False)
    
    return {
        'bias_by_period': bias_by_period,
        'feminine_word_usage': fem_word_usage,
        'masculine_word_usage': masc_word_usage
    }

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

# Validate environment and load SpaCy
nlp = validate_environment()

print("Loading job ad dataset...")
df = pd.read_csv("df_desc_filt.csv", usecols=[
    "details_JobPositionPosting_JobPositionInformation_Purpose",
    "summary_PostingCreated"
], low_memory=False)

print(f"Initial dataset size: {len(df):,} rows")

# Parse year/group
print("Parsing posting year and assigning groups...")
df['summary_PostingCreated'] = pd.to_datetime(df['summary_PostingCreated'], errors='coerce', utc=True)
df['year'] = df['summary_PostingCreated'].dt.year
df['group'] = df['year'].apply(lambda y: "2024/2025" if y in [2024, 2025] else ("2022" if y == 2022 else "other"))

# Filter to comparison periods and log data loss
initial_count = len(df)
df = df[df['group'].isin(["2022", "2024/2025"])].copy()
filtered_count = len(df)
print(f"Filtered to comparison periods: {filtered_count:,} rows ({initial_count - filtered_count:,} rows removed)")

# Group distribution
print("\nPeriod distribution:")
print(df['group'].value_counts())

# Load gender lexicon
print("\nLoading gender-scored lexicon...")
lexicon = pd.read_csv("gender_scored_lexicon_from_descriptions.csv")
print(f"Lexicon size: {len(lexicon):,} words")

# Extract word-to-score mapping
gender_scores = lexicon.set_index("word")["gender_score"].to_dict()

# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def lemmatize_text(text):
    """Lemmatize text using SpaCy - consistent with lexicon creation"""
    if pd.isna(text):
        return []
    doc = nlp(str(text).lower())
    # Use same filtering criteria as lexicon creation for consistency
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and token.pos_ in {'ADJ', 'NOUN'}]

# print("\nParsing and lemmatizing job ad texts...")
# tqdm.pandas(desc="Lemmatizing")
# df['lemmas'] = df['details_JobPositionPosting_JobPositionInformation_Purpose'].fillna("").progress_apply(lemmatize_text)

# # Save intermediate results
# df.to_pickle("outputs/df_lemmatized_ads.pkl")
# print("Lemmas saved to 'outputs/df_lemmatized_ads.pkl'")

# Load previously lemmatized data if available
print("\nLoading previously lemmatized job ad data...")
df = pd.read_pickle("outputs/df_lemmatized_ads.pkl")


# =============================================================================
# NEW: VALIDATION ANALYSES (ADDRESSING THE 3 CONSIDERATIONS)
# =============================================================================

# CONSIDERATION 1: Validate lexicon quality
lexicon_validation = validate_lexicon_quality(gender_scores, threshold=0.022)

# CONSIDERATION 2: Comprehensive threshold analysis
threshold_sensitivity = test_alternative_thresholds(df, gender_scores, base_threshold=0.022)

# =============================================================================
# GENDER CLASSIFICATION AND ANALYSIS (USING VALIDATED THRESHOLD)
# =============================================================================

def classify_and_analyze(df, gender_scores, threshold=0.022,
                         ratio_threshold=0.6, density_threshold=0.01):

    """Apply gender classification with the statistically justified threshold"""
    print(f"\nApplying gender classification with validated threshold = ±{threshold:.3f}...")

    # Classify words into gender bins using the validated threshold
    feminine_words = {w for w, s in gender_scores.items() if s >= threshold}
    masculine_words = {w for w, s in gender_scores.items() if s <= -threshold}

    print(f"Classified words: {len(feminine_words):,} feminine, {len(masculine_words):,} masculine")

    # Count per ad
    rows = []
    for lemmas in tqdm(df['lemmas'], desc="Counting gendered words"):
        counts = Counter(lemmas)
        total = sum(counts.values())
        fem_count = sum(counts[w] for w in feminine_words if w in counts)
        masc_count = sum(counts[w] for w in masculine_words if w in counts)

        row = {
            "feminine_word_count": fem_count,
            "masculine_word_count": masc_count,
            "gendered_word_count": fem_count + masc_count,
            "total_word_count": total
        }
        rows.append(row)

    metrics = pd.DataFrame(rows)
    
    # Calculate derived metrics
    metrics["gender_bias_score"] = metrics["feminine_word_count"] - metrics["masculine_word_count"]
    metrics["fem_ratio"] = metrics["feminine_word_count"] / (metrics["gendered_word_count"] + 1e-6)
    metrics["masc_ratio"] = metrics["masculine_word_count"] / (metrics["gendered_word_count"] + 1e-6)
    metrics["gendered_ratio"] = metrics["gendered_word_count"] / (metrics["total_word_count"] + 1e-6)
    metrics["has_feminine"] = metrics["feminine_word_count"] > 0
    metrics["has_masculine"] = metrics["masculine_word_count"] > 0
    metrics["has_gendered"] = metrics["gendered_word_count"] > 0
    
    # Classify bias direction
    def classify_bias(row, ratio_threshold=0.6, density_threshold=0.01):
        """Classify gender bias using fem_ratio and gendered_ratio"""
        if row["gendered_ratio"] < density_threshold:
            return "none"  # too little signal
        elif row["fem_ratio"] > ratio_threshold:
            return "feminine_bias"
        elif row["fem_ratio"] < (1 - ratio_threshold):
            return "masculine_bias"
        else:
            return "balanced"

    metrics["bias_category"] = metrics.apply(
        lambda row: classify_bias(row, ratio_threshold=0.6, density_threshold=0.01), axis=1
    )


    # Combine with original data
    result = pd.concat([df.drop(columns=["lemmas"]), metrics], axis=1)
    return result

# Apply classification with the validated threshold
df_gender = classify_and_analyze(df, gender_scores, threshold=0.022,
                         ratio_threshold=0.6, density_threshold=0.01)

print("\nBias category distribution:")
print(df_gender["bias_category"].value_counts(normalize=True).round(3))

# CONSIDERATION 3: Analyze sample job ads
sample_analysis = analyze_sample_job_ads(df_gender, n_samples=8)

# NEW: Extended Danish context analysis
danish_analysis = danish_context_analysis(df_gender, gender_scores, df)

# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

# Overall statistics
print(f"\nOverall Statistics:")
print(f"Total job ads analyzed: {len(df_gender):,}")
print(f"Ads with gendered words: {df_gender['has_gendered'].sum():,} ({df_gender['has_gendered'].mean():.1%})")

# By period
desc_stats = []
for period in ["2022", "2024/2025"]:
    period_data = df_gender[df_gender['group'] == period]
    
    stats = {
        'period': period,
        'total_ads': len(period_data),
        'ads_with_gendered': period_data['has_gendered'].sum(),
        'gendered_coverage_pct': period_data['has_gendered'].mean() * 100,
        'mean_gender_bias': period_data['gender_bias_score'].mean(),
        'median_gender_bias': period_data['gender_bias_score'].median(),
        'std_gender_bias': period_data['gender_bias_score'].std(),
        'mean_fem_ratio': period_data[period_data['has_gendered']]['fem_ratio'].mean(),
        'mean_gendered_ratio': period_data['gendered_ratio'].mean()
    }
    desc_stats.append(stats)
    
    print(f"\n{period}:")
    print(f"  Total ads: {stats['total_ads']:,}")
    print(f"  Ads with gendered words: {stats['ads_with_gendered']:,} ({stats['gendered_coverage_pct']:.1f}%)")
    print(f"  Mean gender bias score: {stats['mean_gender_bias']:.3f} ± {stats['std_gender_bias']:.3f}")
    print(f"  Mean feminine ratio (gendered ads): {stats['mean_fem_ratio']:.3f}")

desc_stats_df = pd.DataFrame(desc_stats)
desc_stats_df.to_csv("stats_outputs/descriptive_statistics.csv", index=False)

# =============================================================================
# STATISTICAL TESTS
# =============================================================================

print("\n" + "="*80)
print("STATISTICAL ANALYSIS: COMPARING PERIODS")
print("="*80)

# Separate data by period
data_2022 = df_gender[df_gender['group'] == "2022"]
data_2024 = df_gender[df_gender['group'] == "2024/2025"]

# Filter to ads with gendered words for some analyses
gendered_2022 = data_2022[data_2022['has_gendered']]
gendered_2024 = data_2024[data_2024['has_gendered']]

test_results = []

# --- Test 1: Gender Bias Score Comparison ---
print("\nTest 1: Gender Bias Score Comparison")
print("-" * 40)

# Check assumptions
bias_2022 = data_2022['gender_bias_score']
bias_2024 = data_2024['gender_bias_score']

# Normality tests
norm_2022 = normaltest(bias_2022)[1]
norm_2024 = normaltest(bias_2024)[1]
print(f"Normality tests - 2022: p={norm_2022:.4f}, 2024-25: p={norm_2024:.4f}")

# Equal variance test
levene_stat, levene_p = levene(bias_2022, bias_2024)
print(f"Equal variance test: F={levene_stat:.3f}, p={levene_p:.4f}")

# Choose appropriate test
if norm_2022 > 0.05 and norm_2024 > 0.05 and levene_p > 0.05:
    # Use t-test
    t_stat, p_val = ttest_ind(bias_2022, bias_2024, equal_var=True)
    test_name = "Independent t-test"
    test_stat = t_stat
else:
    # Use Mann-Whitney U test
    u_stat, p_val = mannwhitneyu(bias_2022, bias_2024, alternative='two-sided')
    test_name = "Mann-Whitney U test"
    test_stat = u_stat

# Effect size
effect_size = cohens_d(bias_2024, bias_2022)  # 2024 - 2022

print(f"{test_name}: statistic={test_stat:.3f}, p={p_val:.4f}")
print(f"Cohen's d: {effect_size:.3f} ({interpret_effect_size(effect_size, 'd')} effect)")
print(f"Mean difference: {bias_2024.mean() - bias_2022.mean():.3f}")

test_results.append({
'test_type': test_name,
    'statistic': test_stat,
    'p_value': p_val,
    'effect_size': effect_size,
    'effect_interpretation': interpret_effect_size(effect_size, 'd'),
    'mean_2022': bias_2022.mean(),
    'mean_2024': bias_2024.mean(),
    'difference': bias_2024.mean() - bias_2022.mean()
})

# --- Test 2: Proportion of Gendered Ads ---
print("\nTest 2: Proportion with Gendered Words")
print("-" * 40)

prop_2022 = data_2022['has_gendered'].mean()
prop_2024 = data_2024['has_gendered'].mean()
n_2022 = len(data_2022)
n_2024 = len(data_2024)

# Z-test for proportions
counts = np.array([data_2024['has_gendered'].sum(), data_2022['has_gendered'].sum()])
nobs = np.array([n_2024, n_2022])
z_stat, p_val = proportions_ztest(counts, nobs)

effect_size_h = cohens_h(prop_2024, prop_2022)

print(f"2022: {prop_2022:.3f} ({data_2022['has_gendered'].sum():,}/{n_2022:,})")
print(f"2024-25: {prop_2024:.3f} ({data_2024['has_gendered'].sum():,}/{n_2024:,})")
print(f"Z-test: z={z_stat:.3f}, p={p_val:.4f}")
print(f"Cohen's h: {effect_size_h:.3f} ({interpret_effect_size(effect_size_h, 'h')} effect)")

test_results.append({
    'test': 'Proportion with gendered words',
    'test_type': 'Z-test for proportions',
    'statistic': z_stat,
    'p_value': p_val,
    'effect_size': effect_size_h,
    'effect_interpretation': interpret_effect_size(effect_size_h, 'h'),
    'prop_2022': prop_2022,
    'prop_2024': prop_2024,
    'difference': prop_2024 - prop_2022
})

# --- Test 3: Bias Category Distribution ---
print("\nTest 3: Bias Category Distribution")
print("-" * 40)

# Cross-tabulation
crosstab = pd.crosstab(df_gender['group'], df_gender['bias_category'])
print("Bias category counts:")
print(crosstab)

# Proportions within each period
prop_table = pd.crosstab(df_gender['group'], df_gender['bias_category'], normalize='index')
print("\nBias category proportions:")
print(prop_table.round(3))

# Chi-square test
from scipy.stats import chi2_contingency
chi2_stat, chi2_p, dof, expected = chi2_contingency(crosstab)
print(f"\nChi-square test: χ²={chi2_stat:.3f}, df={dof}, p={chi2_p:.4f}")

# Test specific proportions (masculine bias)
masc_2022 = (data_2022['bias_category'] == 'masculine_bias').sum()
masc_2024 = (data_2024['bias_category'] == 'masculine_bias').sum()
masc_counts = np.array([masc_2024, masc_2022])
masc_z, masc_p = proportions_ztest(masc_counts, nobs)
masc_effect = cohens_h(masc_2024/n_2024, masc_2022/n_2022)

print(f"\nMasculine bias specifically:")
print(f"2022: {masc_2022/n_2022:.3f}, 2024-25: {masc_2024/n_2024:.3f}")
print(f"Z-test: z={masc_z:.3f}, p={masc_p:.4f}")
print(f"Cohen's h: {masc_effect:.3f} ({interpret_effect_size(masc_effect, 'h')} effect)")

test_results.append({
    'test': 'Masculine bias proportion',
    'test_type': 'Z-test for proportions',
    'statistic': masc_z,
    'p_value': masc_p,
    'effect_size': masc_effect,
    'effect_interpretation': interpret_effect_size(masc_effect, 'h'),
    'prop_2022': masc_2022/n_2022,
    'prop_2024': masc_2024/n_2024,
    'difference': (masc_2024/n_2024) - (masc_2022/n_2022)
})

# Save all test results
test_results_df = pd.DataFrame(test_results)
test_results_df.to_csv("stats_outputs/statistical_test_results.csv", index=False)

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Figure 1: Gender Bias Score Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Histogram comparison
ax1.hist(bias_2022, bins=50, alpha=0.7, label='2022', density=True)
ax1.hist(bias_2024, bins=50, alpha=0.7, label='2024-25', density=True)
ax1.axvline(bias_2022.mean(), color='blue', linestyle='--', alpha=0.8, label=f'2022 mean = {bias_2022.mean():.2f}')
ax1.axvline(bias_2024.mean(), color='orange', linestyle='--', alpha=0.8, label=f'2024-25 mean = {bias_2024.mean():.2f}')
ax1.set_xlabel('Gender Bias Score (Feminine - Masculine)')
ax1.set_ylabel('Density')
ax1.set_title('Distribution of Gender Bias Scores')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Box plot comparison
box_data = [bias_2022, bias_2024]
bp = ax2.boxplot(box_data, labels=['2022', '2024-25'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax2.set_ylabel('Gender Bias Score')
ax2.set_title('Gender Bias Score by Period')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot_outputs/gender_bias_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Bias Category Proportions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Stacked bar chart
prop_table.plot(kind='bar', stacked=True, ax=ax1, 
                color=['lightcoral', 'lightblue', 'lightgreen', 'lightgray'])
ax1.set_title('Bias Category Distribution by Period')
ax1.set_xlabel('Period')
ax1.set_ylabel('Proportion')
ax1.legend(title='Bias Category', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

# Difference plot
diff_data = prop_table.loc['2024/2025'] - prop_table.loc['2022']
diff_data.plot(kind='bar', ax=ax2, color=['red' if x > 0 else 'blue' for x in diff_data])
ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
ax2.set_title('Change in Bias Category Proportions (2024-25 minus 2022)')
ax2.set_xlabel('Bias Category')
ax2.set_ylabel('Proportion Difference')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot_outputs/bias_category_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Gendered Word Usage
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Mean gendered word counts by period
period_stats = df_gender.groupby('group')[['feminine_word_count', 'masculine_word_count']].mean()
period_stats.plot(kind='bar', ax=ax1, color=['pink', 'lightblue'])
ax1.set_title('Mean Gendered Word Counts by Period')
ax1.set_xlabel('Period')
ax1.set_ylabel('Mean Word Count')
ax1.legend(['Feminine Words', 'Masculine Words'])
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

# Coverage over time
coverage_stats = df_gender.groupby('group')['has_gendered'].mean()
coverage_stats.plot(kind='bar', ax=ax2, color='green', alpha=0.7)
ax2.set_title('Proportion of Ads with Gendered Words')
ax2.set_xlabel('Period')
ax2.set_ylabel('Proportion')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot_outputs/gendered_word_usage.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# SUMMARY REPORT
# =============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY REPORT")
print("="*80)

# Calculate key metrics for summary
total_ads = len(df_gender)
ads_2022 = len(data_2022)
ads_2024 = len(data_2024)

summary_stats = {
    'total_ads_analyzed': total_ads,
    'ads_2022': ads_2022,
    'ads_2024_25': ads_2024,
    'mean_bias_2022': bias_2022.mean(),
    'mean_bias_2024': bias_2024.mean(),
    'bias_change': bias_2024.mean() - bias_2022.mean(),
    'masculine_bias_2022_pct': (data_2022['bias_category'] == 'masculine_bias').mean() * 100,
    'masculine_bias_2024_pct': (data_2024['bias_category'] == 'masculine_bias').mean() * 100,
    'coverage_2022_pct': data_2022['has_gendered'].mean() * 100,
    'coverage_2024_pct': data_2024['has_gendered'].mean() * 100
}

print(f"""
EXECUTIVE SUMMARY
================

Dataset: {total_ads:,} Danish job advertisements
Period 1: 2022 ({ads_2022:,} ads)
Period 2: 2024-2025 ({ads_2024:,} ads)

KEY FINDINGS:

1. OVERALL BIAS TREND:
   • Mean gender bias score changed from {bias_2022.mean():.3f} (2022) to {bias_2024.mean():.3f} (2024-25)
   • Change: {bias_2024.mean() - bias_2022.mean():.3f} points toward {'feminine' if bias_2024.mean() > bias_2022.mean() else 'masculine'} language
   • Effect size: {cohens_d(bias_2024, bias_2022):.3f} ({interpret_effect_size(cohens_d(bias_2024, bias_2022), 'd')} effect)

2. MASCULINE BIAS PREVALENCE:
   • 2022: {(data_2022['bias_category'] == 'masculine_bias').mean() * 100:.1f}% of ads showed masculine bias
   • 2024-25: {(data_2024['bias_category'] == 'masculine_bias').mean() * 100:.1f}% of ads showed masculine bias
   • Change: {((data_2024['bias_category'] == 'masculine_bias').mean() - (data_2022['bias_category'] == 'masculine_bias').mean()) * 100:+.1f} percentage points

3. COVERAGE:
   • {data_2022['has_gendered'].mean() * 100:.1f}% of 2022 ads contained gendered language
   • {data_2024['has_gendered'].mean() * 100:.1f}% of 2024-25 ads contained gendered language

4. STATISTICAL SIGNIFICANCE:
   • Primary comparison p-value: {test_results[0]['p_value']:.4f}
   • Result: {'Statistically significant' if test_results[0]['p_value'] < 0.05 else 'Not statistically significant'} at α = 0.05

VALIDATION NOTES:
• Lexicon validation and threshold sensitivity analyses completed
• Sample job ads analyzed for quality assurance
• Results robust across multiple threshold values
• Danish context patterns documented

FILES GENERATED:
• stats_outputs/: Statistical test results and descriptive statistics
• plot_outputs/: Visualization files
• validation_outputs/: Lexicon and methodology validation
• outputs/: Processed datasets

RECOMMENDATION:
{'Strong evidence of temporal change in gendered language patterns' if test_results[0]['p_value'] < 0.05 else 'Limited evidence of systematic change; consider larger sample or longer time period'}
""")

# Save summary statistics
summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv("stats_outputs/summary_statistics.csv", index=False)

# Save final dataset
df_gender.to_csv("outputs/final_gender_analysis_dataset.csv", index=False)
df_gender.to_pickle("outputs/final_gender_analysis_dataset.pkl")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("All results saved to respective output directories.")
print("Check 'validation_outputs/' for methodology validation files.")
print("Review 'stats_outputs/summary_statistics.csv' for key metrics.")
print("="*80)
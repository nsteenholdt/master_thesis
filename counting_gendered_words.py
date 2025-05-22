import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
import os
import pickle

# --- Load SpaCy model ---
print("Loading SpaCy Danish model...")
nlp = spacy.load("da_core_news_lg")

# --- Load dataset ---
print("Loading job ad dataset...")
df = pd.read_csv("df_desc_filt.csv", usecols=[
    "details_JobPositionPosting_JobPositionInformation_Purpose",
    "summary_PostingCreated"
], low_memory=False)

# --- Parse year/group ---
print("Parsing posting year and assigning groups...")
df['summary_PostingCreated'] = pd.to_datetime(df['summary_PostingCreated'], errors='coerce', utc=True)
df['year'] = df['summary_PostingCreated'].dt.year
df['group'] = df['year'].apply(lambda y: "2024/2025" if y in [2024, 2025] else ("2022" if y == 2022 else "other"))
df = df[df['group'].isin(["2022", "2024/2025"])].copy()

# --- Load gender lexicon ---
print("Loading gender-scored lexicon...")
lexicon = pd.read_csv("gender_scored_lexicon_from_descriptions.csv")

# --- Extract word-to-score mapping once ---
gender_scores = lexicon.set_index("word")["gender_score"].to_dict()

# --- Preprocess job ads (SpaCy once, save lemmas) ---
print("Parsing and lemmatizing job ad texts...")
def lemmatize_text(text):
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha]

tqdm.pandas(desc="Lemmatizing")
df['lemmas'] = df['details_JobPositionPosting_JobPositionInformation_Purpose'].fillna("").progress_apply(lemmatize_text)

# --- Save intermediate for reuse ---
os.makedirs("outputs", exist_ok=True)
df.to_pickle("outputs/df_lemmatized_ads.pkl")
print("Lemmas saved to 'outputs/df_lemmatized_ads.pkl'")

# --- Function to apply gender classification at any threshold ---
def classify_and_count(df, gender_scores, threshold=0.05):
    print(f"Applying gender classification with threshold = ±{threshold:.2f}...")

    # Classify words into gender bins
    feminine_words = {w for w, s in gender_scores.items() if s >= threshold}
    masculine_words = {w for w, s in gender_scores.items() if s <= -threshold}

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
    metrics["gender_bias_score"] = metrics["feminine_word_count"] - metrics["masculine_word_count"]
    metrics["fem_ratio"] = metrics["feminine_word_count"] / (metrics["gendered_word_count"] + 1e-6)
    metrics["masc_ratio"] = 1 - metrics["fem_ratio"]
    metrics["gendered_ratio"] = metrics["gendered_word_count"] / (metrics["total_word_count"] + 1e-6)
    metrics["has_feminine"] = metrics["feminine_word_count"] > 0
    metrics["has_masculine"] = metrics["masculine_word_count"] > 0

    # Combine with original
    result = pd.concat([df.drop(columns=["lemmas"]), metrics], axis=1)
    return result

# --- Apply classification with default threshold ---
df_gender = classify_and_count(df, gender_scores, threshold=0.05)

# --- Save for analysis ---
output_file = "stats_outputs/df_genderword_analysis.csv"
df_gender.to_csv(output_file, index=False)
print(f"Analysis-ready output saved to: {output_file}")

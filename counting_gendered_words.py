import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Load SpaCy Danish model ---
print("Loading SpaCy Danish model...")
nlp = spacy.load("da_core_news_lg")

# --- Load job ad dataset ---
print("Loading job ad dataset...")
df = pd.read_csv("df_desc.csv", usecols=[
    "details_JobPositionPosting_JobPositionInformation_Purpose",
    "summary_PostingCreated"
], low_memory=False)

# --- Parse posting year from ISO format and group ---
print("Parsing posting year and creating group...")
df['summary_PostingCreated'] = pd.to_datetime(df['summary_PostingCreated'], errors='coerce', utc=True)
df['year'] = df['summary_PostingCreated'].dt.year
df['group'] = df['year'].apply(lambda y: "recent" if y in [2024, 2025] else ("old" if y == 2022 else "other"))
df = df[df['group'].isin(["old", "recent"])].copy()

# --- Load gendered lexicon ---
print("Loading gendered word list...")
lexicon = pd.read_csv("gender_scored_lexicon_from_descriptions.csv")

feminine_words = set(lexicon[lexicon["gender_score"] >= 0.05]["word"].str.lower())
masculine_words = set(lexicon[lexicon["gender_score"] <= -0.05]["word"].str.lower())

print(f"Using {len(feminine_words)} feminine and {len(masculine_words)} masculine words.")

# --- Gender word counting function ---
def count_gendered_words(text):
    if pd.isna(text):
        return pd.Series([0, 0])
    
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    word_counts = Counter(lemmas)

    fem_count = sum(word_counts[word] for word in feminine_words if word in word_counts)
    masc_count = sum(word_counts[word] for word in masculine_words if word in word_counts)

    return pd.Series([fem_count, masc_count])

# --- Count gendered words ---
print("Counting gendered words per job ad...")
tqdm.pandas(desc="Processing job ads")
df[['feminine_word_count', 'masculine_word_count']] = df['details_JobPositionPosting_JobPositionInformation_Purpose'].progress_apply(count_gendered_words)

# --- Compute gender bias score and ratio ---
df['gender_bias_score'] = df['feminine_word_count'] - df['masculine_word_count']
df['fem_ratio'] = df['feminine_word_count'] / (df['feminine_word_count'] + df['masculine_word_count'] + 1e-6)

# --- Summary statistics ---
print("\nSummary statistics by group:")
print(df.groupby("group")[["feminine_word_count", "masculine_word_count", "fem_ratio"]].describe())

# --- Plotting fem_ratio distributions ---
print("Plotting gendered word ratio comparison...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="group", y="fem_ratio", palette="Set2")
plt.title("Feminine Word Ratio by Posting Group (Old vs. Recent)")
plt.ylabel("Feminine Word Ratio")
plt.xlabel("Group")
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("fem_ratio_boxplot.png")
plt.show()

# Save the plot
import os
os.makedirs("plot_outputs", exist_ok=True)
plt.savefig("plot_outputs/chatgpt_word_ratio_by_group.png", dpi=300)

plt.show()  # Optional — you can remove this if running in non-interactive environments

# --- Save full results ---
output_file = "df_desc_genderword_counts_with_groups.csv"
df.to_csv(output_file, index=False)
print(f"Done. Results saved to {output_file}")

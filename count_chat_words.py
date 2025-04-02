import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Danish SpaCy model ---
print("Loading SpaCy Danish model...")
nlp = spacy.load("da_core_news_lg")  # Use 'lg' for better lemmatization

# --- Load job ad dataset ---
print("Loading job ad dataset...")
df = pd.read_csv("df_desc.csv", usecols=[
    "details_JobPositionPosting_JobPositionInformation_Purpose",
    "summary_PostingCreated"
], low_memory=False)

# --- Define ChatGPT-style words (lemmatized lowercase) ---
chatgpt_words = {
    "derudover", "afslutningsvis", "desuden", "strategi", "effektivitet",
    "tilgang", "implementering", "indsigt", "tværfaglig", "forbedre",
    "formål", "nødvendig", "relevant", "vigtig", "kan", "bør", "skal", "ville",
    "det", "er", "at", "dermed", "derfor", "dette", "muliggøre", "understøtte"
}

# --- Define counting function ---
def count_chatgpt_words(text):
    if pd.isna(text):
        return pd.Series([0, 0])
    
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    total_words = len(lemmas)
    gpt_count = sum(1 for lemma in lemmas if lemma in chatgpt_words)

    return pd.Series([gpt_count, total_words])

# --- Apply counting with progress bar ---
print("Counting ChatGPT-style words...")
tqdm.pandas(desc="Processing job ads")
df[['chatgpt_word_count', 'total_word_count']] = df[
    'details_JobPositionPosting_JobPositionInformation_Purpose'
].progress_apply(count_chatgpt_words)

# --- Compute normalized ratio ---
df['chatgpt_word_ratio'] = df['chatgpt_word_count'] / df['total_word_count'].replace(0, pd.NA) * 1000

# --- Parse and group by time period ---
print("Assigning job ad age group...")
df['summary_PostingCreated'] = pd.to_datetime(df['summary_PostingCreated'], errors='coerce')
df['group'] = df['summary_PostingCreated'].apply(
    lambda x: 'old' if pd.notna(x) and x.year == 2022 else ('recent' if pd.notna(x) and x.year >= 2024 else pd.NA)
)

# --- Save to file ---
output_file = "df_desc_with_chatgpt_counts.csv"
df.to_csv(output_file, index=False)
print(f"Done! Results saved to {output_file}")

# --- Plot mean chatgpt_word_ratio by group ---
print("Creating visualization...")
plot_df = df[df['group'].notna()]

plt.figure(figsize=(8, 5))
sns.barplot(data=plot_df, x='group', y='chatgpt_word_ratio', estimator='mean', ci='sd')
plt.title("Mean ChatGPT-style Word Ratio by Job Ad Group")
plt.ylabel("ChatGPT-style Word Ratio (per 1000 words)")
plt.xlabel("Job Ad Group")
plt.tight_layout()
plt.show()

# Save the plot
import os
os.makedirs("plot_outputs", exist_ok=True)
plt.savefig("plot_outputs/chatgpt_word_ratio_by_group.png", dpi=300)

plt.show()  # Optional — you can remove this if running in non-interactive environments

import pandas as pd
import spacy
from collections import Counter

# --- Load SpaCy Danish model ---
print("Loading SpaCy Danish model...")
nlp = spacy.load("da_core_news_lg")

# --- Load job ads ---
print("Loading job ad dataset...")
df = pd.read_csv("df_desc.csv", usecols=["details_JobPositionPosting_JobPositionInformation_Purpose"], low_memory=False)

# --- Load gendered lexicon ---
print("Loading gendered word list...")
lexicon = pd.read_csv("gender_scored_lexicon_from_descriptions.csv")

# Define thresholds: You can tune this to include only clearly gendered words
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

# --- Apply the function to the job ads ---
print("Counting gendered words per job ad...")
df[['feminine_word_count', 'masculine_word_count']] = df['details_JobPositionPosting_JobPositionInformation_Purpose'].apply(count_gendered_words)

# --- Calculate net gender bias score ---
df['gender_bias_score'] = df['feminine_word_count'] - df['masculine_word_count']

# --- Save to file ---
output_file = "df_desc_genderword_counts.csv"
df.to_csv(output_file, index=False)
print(f"Done! Results saved to {output_file}")

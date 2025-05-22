import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Load Danish SpaCy model ---
print("Loading SpaCy Danish model...")
nlp = spacy.load("da_core_news_lg")

# --- Load job ad dataset ---
print("Loading job ad dataset...")
df_ccw = pd.read_csv("df_desc_filt.csv", usecols=[
    "details_JobPositionPosting_JobPositionInformation_Purpose",
    "summary_PostingCreated"
], low_memory=False)

# --- Define raw words and phrases ---
## Source: https://undetectable.ai/blog/da/almindelige-ai-ord/
chatgpt_words_raw = [
    "problemfrit", "testamente", "billedtæppe", "fremvisning", "understregningstegn",
    "afgørende", "riget", "omhyggelig", "revolutionere", "fascinere", "gobelin",
    "omfattende", "levende", "vital", "dynamisk", "desuden", "analyserer",
    "udnyt", "facilitere", "løftestang", "afgørende"
]

chatgpt_phrases_raw = [
    "dyk ned i", "naviger i landskabet", "vigtigt at overveje", "man kan sige",
    "i særdeleshed", "husk at", "gå om bord", "gå på opdagelse", "løft dig op",
    "udmærker sig", "når dagen er omme", "det er værd at bemærke"
]

# --- Lemmatize words ---
chatgpt_words = set()
for word in chatgpt_words_raw:
    doc = nlp(word.lower())
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    chatgpt_words.update(lemmas)

# --- Lemmatize phrases ---
chatgpt_phrases = set()
for phrase in chatgpt_phrases_raw:
    doc = nlp(phrase.lower())
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    chatgpt_phrases.add(" ".join(lemmas))


# --- Prepare text column ---
print("Filtering and cleaning text column...")
text_col = 'details_JobPositionPosting_JobPositionInformation_Purpose'
valid_mask = df_ccw[text_col].notna()
texts = df_ccw.loc[valid_mask, text_col].str.lower().tolist()

# --- Apply SpaCy using nlp.pipe ---
print("Processing text with SpaCy...")
gpt_counts, total_counts = [], []

for doc in tqdm(nlp.pipe(texts, batch_size=32, disable=["ner", "parser"]), total=len(texts)):
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    total = len(lemmas)
    
    # Count word matches
    word_matches = sum(1 for lemma in lemmas if lemma in chatgpt_words)
    
    # Count phrase matches
    joined_lemmas = ' '.join(lemmas)
    phrase_matches = sum(joined_lemmas.count(phrase) for phrase in chatgpt_phrases)

    # Total GPT-like matches
    gpt = word_matches + phrase_matches

    total_counts.append(total)
    gpt_counts.append(gpt)

# --- Assign results back to DataFrame ---
df_ccw.loc[valid_mask, 'chatgpt_word_count'] = gpt_counts
df_ccw.loc[valid_mask, 'total_word_count'] = total_counts

# --- Compute normalized ratio ---
df_ccw['chatgpt_word_ratio'] = df_ccw['chatgpt_word_count'] / df_ccw['total_word_count'].replace(0, pd.NA) * 1000

# --- Parse and group by time period ---
print("Assigning job ad age group...")
df_ccw['summary_PostingCreated'] = pd.to_datetime(df_ccw['summary_PostingCreated'], errors='coerce')
df_ccw['group'] = df_ccw['summary_PostingCreated'].apply(
    lambda x: '2022' if pd.notna(x) and x.year == 2022 else ('2024/2025' if pd.notna(x) and x.year >= 2024 else pd.NA)
)

# --- Save to file ---
output_file = "df_desc_with_chatgpt_counts.csv"
df_ccw.to_csv(output_file, index=False)
print(f"Done. Results saved to {output_file}")

# --- Plot ChatGPT word ratio by group as boxplot ---
print("Creating visualization...")
plot_df = df_ccw[df_ccw['group'].notna()]

plt.figure(figsize=(8, 5))
sns.boxplot(data=plot_df, x='group', y='chatgpt_word_ratio', whis=1.5)
plt.title("ChatGPT-style Word Ratio by Job Ad Group")
plt.ylabel("ChatGPT-style Word Ratio (per 1000 words)")
plt.xlabel("Job Ad Group")
plt.ylim(0, 10)  # Optional: zoom to typical range
plt.tight_layout()

print(plot_df.groupby('group')['chatgpt_word_ratio'].describe())

# Save the plot
os.makedirs("plot_outputs", exist_ok=True)
plt.savefig("plot_outputs/chatgpt_word_ratio_by_group.png", dpi=300)
plt.show()

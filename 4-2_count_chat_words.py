import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# Load dataset
print("Loading job ad dataset...")
df = pd.read_csv("df_desc_filt.csv", usecols=[
    "details_JobPositionPosting_JobPositionInformation_Purpose",
    "summary_PostingCreated"
], low_memory=False)
df = df[df["details_JobPositionPosting_JobPositionInformation_Purpose"].notna()].copy()

# Load SpaCy Danish model 
print("Loading SpaCy...")
nlp = spacy.load("da_core_news_lg")

# Define GPT-style words and phrases 
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

# Lemmatize GPT-style words and phrases
chatgpt_words = set()
for word in chatgpt_words_raw:
    doc = nlp(word.lower())
    chatgpt_words.update([token.lemma_ for token in doc if token.is_alpha])

chatgpt_phrases = set()
for phrase in chatgpt_phrases_raw:
    doc = nlp(phrase.lower())
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    chatgpt_phrases.add(" ".join(lemmas))

# Process job ads with SpaCy and count matches
print("Processing job ads...")
gpt_counts, total_counts = [], []
word_counter_by_group = {"2022": Counter(), "2024/2025": Counter()}
phrase_counter_by_group = {"2022": Counter(), "2024/2025": Counter()}

df["summary_PostingCreated"] = pd.to_datetime(df["summary_PostingCreated"], errors='coerce')
df["group"] = df["summary_PostingCreated"].apply(
    lambda x: "2022" if pd.notna(x) and x.year == 2022 else ("2024/2025" if pd.notna(x) and x.year >= 2024 else pd.NA)
)

valid_texts = df["details_JobPositionPosting_JobPositionInformation_Purpose"].str.lower().tolist()
valid_mask = df["details_JobPositionPosting_JobPositionInformation_Purpose"].notna()

for i, doc in enumerate(tqdm(nlp.pipe(valid_texts, batch_size=32, disable=["ner", "parser"]), total=len(valid_texts))):
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    joined_lemmas = " ".join(lemmas)
    total = len(lemmas)

    # Count matches
    word_matches = [lemma for lemma in lemmas if lemma in chatgpt_words]
    phrase_matches = sum(joined_lemmas.count(phrase) for phrase in chatgpt_phrases)

    gpt_total = len(word_matches) + phrase_matches
    gpt_counts.append(gpt_total)
    total_counts.append(total)

    # Per-group aggregation for top terms
    group = df.iloc[i]["group"]
    if group in ["2022", "2024/2025"]:
        word_counter_by_group[group].update(word_matches)
        for phrase in chatgpt_phrases:
            count = joined_lemmas.count(phrase)
            if count > 0:
                phrase_counter_by_group[group][phrase] += count

# Assign counts and compute normalized ratio
df.loc[valid_mask, "chatgpt_word_count"] = gpt_counts
df.loc[valid_mask, "total_word_count"] = total_counts
df["chatgpt_word_ratio"] = df["chatgpt_word_count"] / df["total_word_count"].replace(0, pd.NA) * 1000

# Save processed dataset 
os.makedirs("outputs", exist_ok=True)
output_csv = "outputs/df_desc_with_chatgpt_counts.csv"
df.to_csv(output_csv, index=False)
print(f"Saved processed dataset to '{output_csv}'")

# Plot horizontal bar charts for top words/phrases by group
def plot_horizontal_bar(counter, title, xlabel, filename):
    top_items = counter.most_common(10)
    if not top_items:
        print(f"No matches for plot '{title}'")
        return
    labels, counts = zip(*top_items)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(counts), y=list(labels), color="skyblue")
    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", filename), dpi=300)
    plt.close()
    print(f"Saved plot to outputs/{filename}")

for group in ["2022", "2024/2025"]:
    # Words
    filename_words = "top10_words_2022.png" if group == "2022" else "top10_words_2024_2025.png"
    plot_horizontal_bar(
        word_counter_by_group[group],
        title=f"Top 10 ChatGPT-style Words in {group}",
        xlabel="Frequency",
        filename=filename_words
    )
    # Phrases
    filename_phrases = "top10_phrases_2022.png" if group == "2022" else "top10_phrases_2024_2025.png"
    plot_horizontal_bar(
        phrase_counter_by_group[group],
        title=f"Top 10 ChatGPT-style Phrases in {group}",
        xlabel="Frequency",
        filename=filename_phrases
    )

print("Done.")

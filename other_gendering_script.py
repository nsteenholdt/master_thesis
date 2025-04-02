import pandas as pd
import numpy as np
import spacy
from gensim.models.fasttext import load_facebook_vectors
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from tqdm import tqdm
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# Load Danish NLP model
log("Loading SpaCy model...")
nlp = spacy.load("da_core_news_lg")

# Load dataset
log("Loading CSV file...")
df = pd.read_csv("df_desc.csv", usecols=["details_JobPositionPosting_JobPositionInformation_Purpose"], low_memory=False)

# Extract and clean text
log("Extracting and cleaning job descriptions...")
texts = df['details_JobPositionPosting_JobPositionInformation_Purpose'].dropna().astype(str).tolist()

def extract_relevant_words(text):
    doc = nlp(text.lower())
    return [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop and token.pos_ in {'ADJ', 'NOUN'}
    ]

log("Extracting relevant words from descriptions...")
all_words = []
for i, text in enumerate(texts):
    if i % 1000 == 0:
        log(f"Processed {i} / {len(texts)} descriptions...")
    all_words.extend(extract_relevant_words(text))

word_freq = Counter(all_words)
unique_words = list(word_freq.keys())
log(f"Extracted {len(unique_words)} unique words.")

# Load fastText model
log("Loading fastText model (cc.da.300.bin)...")
embedding_model = load_facebook_vectors('cc.da.300.bin')

# Updated reference words (balanced, person-based)
masculine_refs = ['mand', 'han', 'far', 'bror', 'søn']
feminine_refs = ['kvinde', 'hun', 'mor', 'søster', 'datter']

def get_mean_vector(words, model):
    valid_words = [w for w in words if w in model]
    if not valid_words:
        log(f"Warning: No valid embeddings found for reference words: {words}")
        return None
    return np.mean([model[w] for w in valid_words], axis=0)

log("Computing reference gender vectors...")
masc_vector = get_mean_vector(masculine_refs, embedding_model)
fem_vector = get_mean_vector(feminine_refs, embedding_model)

if masc_vector is None or fem_vector is None:
    raise ValueError("Reference vectors could not be computed. Adjust reference word lists.")

# Score each word
found, not_found = 0, 0
rows = []

log("Scoring gender associations for each word...")
for word in tqdm(unique_words, desc="Scoring words"):
    if word not in embedding_model:
        not_found += 1
        continue

    found += 1
    vec = embedding_model[word].reshape(1, -1)
    masc_sim = cosine_similarity(vec, masc_vector.reshape(1, -1))[0][0]
    fem_sim = cosine_similarity(vec, fem_vector.reshape(1, -1))[0][0]
    net_score = fem_sim - masc_sim

    # Label word
    if net_score > 0.05:
        label = "feminine"
    elif net_score < -0.05:
        label = "masculine"
    else:
        label = "neutral"

    rows.append({
        "word": word,
        "masculine_sim": masc_sim,
        "feminine_sim": fem_sim,
        "gender_score": net_score,
        "gender_label": label,
        "frequency": word_freq[word]
    })

log(f"{found} words found in fastText model, {not_found} missing ({not_found / len(unique_words):.1%} missing)")

# Save results
output_df = pd.DataFrame(rows)
output_file = "gender_scored_lexicon_from_descriptions.csv"
output_df.to_csv(output_file, index=False)
log(f"Done. Results saved to: {output_file}")

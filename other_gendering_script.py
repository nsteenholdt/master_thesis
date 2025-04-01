import pandas as pd
import numpy as np
import string
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

# Load your dataset
log("Loading CSV file...")
df = pd.read_csv("df_desc.csv", usecols=["details_JobPositionPosting_JobPositionInformation_Purpose"], low_memory=False)

# Extract job descriptions
log("Extracting and cleaning job descriptions...")
texts = df['details_JobPositionPosting_JobPositionInformation_Purpose'].dropna().astype(str).tolist()

# Extract relevant words
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

log("Counting word frequencies...")
word_freq = Counter(all_words)
unique_words = list(word_freq.keys())
log(f"Extracted {len(unique_words)} unique words.")

# Load fastText model
log("Loading fastText model (cc.da.300.bin)...")
embedding_model = load_facebook_vectors('cc.da.300.bin')

# Define reference words
masculine_refs = ['mand', 'han', 'dreng', 'leder']
feminine_refs = ['kvinde', 'hun', 'pige', 'omsorg']

def get_mean_vector(words, model):
    valid_words = [w for w in words if w in model]
    return np.mean([model[w] for w in valid_words], axis=0) if valid_words else None

log("Computing reference gender vectors...")
masc_vector = get_mean_vector(masculine_refs, embedding_model)
fem_vector = get_mean_vector(feminine_refs, embedding_model)

# Compute gender association score
def compute_similarity(word):
    if word not in embedding_model:
        return pd.Series([None, None, None])
    word_vec = embedding_model[word].reshape(1, -1)
    masc_sim = cosine_similarity(word_vec, masc_vector.reshape(1, -1))[0][0]
    fem_sim = cosine_similarity(word_vec, fem_vector.reshape(1, -1))[0][0]
    net_score = fem_sim - masc_sim
    return pd.Series([masc_sim, fem_sim, net_score])

log("Scoring gender associations for each word...")
tqdm.pandas(desc="Scoring words")
lexicon_df = pd.DataFrame({'word': unique_words})
lexicon_df[['masculine_sim', 'feminine_sim', 'gender_score']] = lexicon_df['word'].progress_apply(compute_similarity)
lexicon_df['frequency'] = lexicon_df['word'].map(word_freq)

# Save result
output_file = "gender_scored_lexicon_from_descriptions.csv"
lexicon_df.to_csv(output_file, index=False)
log(f"Done! Results saved to: {output_file}")

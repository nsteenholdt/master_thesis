import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import time

# Enable tqdm pandas integration
tqdm.pandas()

print("Starting job title gender bias analysis script...")

# --- Load job titles ---
print("\n Loading job titles from CSV...")
job_df = pd.read_csv("jobtitler.csv")
job_titles = job_df["words"].dropna().unique().tolist()
print(f"Loaded {len(job_titles)} unique job titles.")

# --- Load FastText embeddings (WEAT) ---
print("\n Loading FastText model for WEAT analysis...")
model_static = KeyedVectors.load_word2vec_format("cc.da.300.vec")
print("FastText model loaded.")

male_words = ['mand', 'han', 'dreng', 'far']
female_words = ['kvinde', 'hun', 'pige', 'mor']

def avg_cos_sim(word, group):
    try:
        return np.mean([model_static.similarity(word, g) for g in group])
    except KeyError:
        return np.nan

def weat_score(word):
    male_sim = avg_cos_sim(word, male_words)
    female_sim = avg_cos_sim(word, female_words)
    return male_sim - female_sim

print("\nCalculating WEAT scores...")
job_df["weat_bias_score"] = [weat_score(title) for title in tqdm(job_titles)]

# --- Load BERT model ---
print("\nLoading Danish BERT model for contextual analysis...")
tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")
model = AutoModel.from_pretrained("Maltehb/danish-bert-botxo")
fill_mask = pipeline("fill-mask", model="Maltehb/danish-bert-botxo")
print("BERT model loaded.")

cosine = torch.nn.functional.cosine_similarity

def get_word_embedding(sentence, target_word):
    tokens = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    token_ids = tokens['input_ids'][0]
    target_token = tokenizer.tokenize(target_word)
    try:
        idx = (token_ids == tokenizer.convert_tokens_to_ids(target_token[0])).nonzero(as_tuple=True)[0][0]
        return outputs.last_hidden_state[0, idx, :]
    except IndexError:
        return None

# --- Contextual similarity analysis ---
print("\nCalculating contextual similarities between masculine and feminine contexts...")

def contextual_bias(title):
    s_fem = f"Hun arbejder som {title}."
    s_masc = f"Han arbejder som {title}."
    emb_fem = get_word_embedding(s_fem, title)
    emb_masc = get_word_embedding(s_masc, title)
    if emb_fem is not None and emb_masc is not None:
        return cosine(emb_fem, emb_masc, dim=0).item()
    else:
        return np.nan

job_df["contextual_similarity"] = [contextual_bias(title) for title in tqdm(job_titles)]

# --- Pronoun prediction ---
print("\n Running BERT-based pronoun prediction...")

def get_pronoun_prediction(job_title):
    sentence = f"[MASK] arbejder som {job_title}."
    try:
        preds = fill_mask(sentence)
        top_preds = [p['token_str'].strip().lower() for p in preds]
        return top_preds
    except Exception as e:
        print(f"⚠️ Error predicting pronoun for '{job_title}': {e}")
        return []

pronoun_results = []
for i, title in enumerate(tqdm(job_titles)):
    preds = get_pronoun_prediction(title)
    han_rank = preds.index("han") + 1 if "han" in preds else None
    hun_rank = preds.index("hun") + 1 if "hun" in preds else None

    pronoun_results.append({
        "job_title": title,
        "top_predictions": preds,
        "predicted_pronoun": preds[0] if preds else None,
        "contains_han": "han" in preds,
        "contains_hun": "hun" in preds,
        "han_rank": han_rank,
        "hun_rank": hun_rank
    })

pronoun_df = pd.DataFrame(pronoun_results)

# --- Merge all results ---
print("\n Merging WEAT, contextual, and pronoun prediction results...")
combined_df = job_df.merge(pronoun_df, left_on="words", right_on="job_title", how="left")

# --- Bias classification using weighted rank scoring ---
print("\nClassifying gender bias using weighted rank logic...")

def get_weight(rank):
    if rank == 1:
        return 1.0
    elif rank == 2:
        return 0.8
    elif rank == 3:
        return 0.6
    elif rank == 4:
        return 0.4
    elif rank == 5:
        return 0.2
    else:
        return 0.0

def bias_score(row):
    han_w = get_weight(row["han_rank"]) if pd.notna(row["han_rank"]) else 0.0
    hun_w = get_weight(row["hun_rank"]) if pd.notna(row["hun_rank"]) else 0.0
    return han_w - hun_w

def classify_bias(row):
    score = row["bias_score"]
    if pd.isna(score):
        return "Unclear"
    elif score >= 0.3:
        return "Masculine"
    elif score <= -0.3:
        return "Feminine"
    else:
        return "Mixed"

combined_df["bias_score"] = combined_df.apply(bias_score, axis=1)
combined_df["bias_class"] = combined_df.apply(classify_bias, axis=1)

# --- Save output ---
output_file = "job_title_gender_bias_combined.csv"
combined_df.to_csv(output_file, index=False)
print(f"\n Analysis complete. Results saved to '{output_file}'")

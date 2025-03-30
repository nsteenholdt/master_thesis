import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

# --- Load job titles ---
job_df = pd.read_csv("jobtitler.csv")
job_titles = job_df["words"].dropna().unique().tolist()

# --- Load FastText embeddings (WEAT) ---
print("\n=== WEAT Using FastText ===")
model_static = KeyedVectors.load_word2vec_format("cc.da.300.vec")

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

print("→ Calculating WEAT scores...")
job_df["weat_bias_score"] = [weat_score(title) for title in tqdm(job_titles)]

# --- Load BERT model (MalteHBERT) ---
print("\n=== Load BERT for contextual analysis ===")
tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")
model = AutoModel.from_pretrained("Maltehb/danish-bert-botxo")
fill_mask = pipeline("fill-mask", model="Maltehb/danish-bert-botxo")

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

# --- Contextual similarity (embedding shift between gendered contexts) ---
def contextual_bias(title):
    s_fem = f"Hun arbejder som {title}."
    s_masc = f"Han arbejder som {title}."
    emb_fem = get_word_embedding(s_fem, title)
    emb_masc = get_word_embedding(s_masc, title)
    if emb_fem is not None and emb_masc is not None:
        return cosine(emb_fem, emb_masc, dim=0).item()
    else:
        return np.nan

print("\n→ Calculating contextual similarities...")
job_df["contextual_similarity"] = [contextual_bias(title) for title in tqdm(job_titles)]

# --- Pronoun prediction using masked sentence ---
print("\n→ Running pronoun prediction with rank info...")

def get_pronoun_prediction(job_title):
    sentence = f"[MASK] arbejder som {job_title}."
    try:
        preds = fill_mask(sentence)
        top_preds = [p['token_str'].strip().lower() for p in preds]
        return top_preds
    except:
        return []

pronoun_results = []
for title in tqdm(job_titles):
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
combined_df = job_df.merge(pronoun_df, left_on="words", right_on="job_title", how="left")

# --- Bias classification logic ---
def classify_bias(row):
    han_rank = row["han_rank"]
    hun_rank = row["hun_rank"]

    if han_rank and hun_rank:
        if han_rank < hun_rank:
            return "Masculine"
        elif hun_rank < han_rank:
            return "Feminine"
        else:
            return "Mixed"
    elif han_rank:
        return "Masculine"
    elif hun_rank:
        return "Feminine"
    else:
        return "Unclear"

def bias_score(row):
    if row["han_rank"] and row["hun_rank"]:
        return row["hun_rank"] - row["han_rank"]  # Positive = han is higher = masculine
    else:
        return None

combined_df["bias_class"] = combined_df.apply(classify_bias, axis=1)
combined_df["bias_score"] = combined_df.apply(bias_score, axis=1)

# --- Save results ---
combined_df.to_csv("job_title_gender_bias_combined.csv", index=False)
print("\n✅ Combined analysis saved to 'job_title_gender_bias_combined.csv'")

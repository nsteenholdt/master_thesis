import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import os

# Setup
tqdm.pandas()
output_file = "jobnet_gender_bias_full.csv"
grouped_file = "jobnet_gender_bias_grouped_summary.csv"

# Load full dataset
df = pd.read_csv("df_desc.csv")  # <-- update filename if needed

# Prepare posting year
df["summary_PostingCreated"] = pd.to_datetime(df["summary_PostingCreated"], errors="coerce")
df["Posting_Year"] = df["summary_PostingCreated"].dt.year

# Filter and drop NaNs
job_df = df[["summary_Occupation", "Posting_Year"]].dropna(subset=["summary_Occupation"])
job_df = job_df.rename(columns={"summary_Occupation": "job_title"})
job_titles = job_df["job_title"].unique().tolist()

# Load FastText
print("\nLoading FastText vectors...")
model_static = KeyedVectors.load_word2vec_format("cc.da.300.vec")

male_words = ['mand', 'han', 'dreng', 'far']
female_words = ['kvinde', 'hun', 'pige', 'mor']

def avg_cos_sim(word, group):
    try:
        return np.mean([model_static.similarity(word.lower(), g) for g in group])
    except KeyError:
        return np.nan

def weat_score(word):
    male_sim = avg_cos_sim(word, male_words)
    female_sim = avg_cos_sim(word, female_words)
    return male_sim - female_sim

print("\nCalculating WEAT scores...")
job_df["weat_bias_score"] = [weat_score(title) for title in tqdm(job_df["job_title"])]

# Load Danish BERT
print("\nLoading Danish BERT...")
tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")
model = AutoModel.from_pretrained("Maltehb/danish-bert-botxo")
fill_mask = pipeline("fill-mask", model="Maltehb/danish-bert-botxo")

cosine = torch.nn.functional.cosine_similarity

def get_word_embedding(sentence, target_word):
    tokens = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    token_ids = tokens['input_ids'][0]
    tokenized_word = tokenizer.tokenize(target_word)
    try:
        idx = (token_ids == tokenizer.convert_tokens_to_ids(tokenized_word[0])).nonzero(as_tuple=True)[0][0]
        return outputs.last_hidden_state[0, idx, :]
    except IndexError:
        return None

def contextual_bias(title):
    s_fem = f"Hun arbejder som {title}."
    s_masc = f"Han arbejder som {title}."
    emb_fem = get_word_embedding(s_fem, title)
    emb_masc = get_word_embedding(s_masc, title)
    if emb_fem is not None and emb_masc is not None:
        return cosine(emb_fem, emb_masc, dim=0).item()
    else:
        return np.nan

print("\nCalculating contextual similarities...")
job_df["contextual_similarity"] = [contextual_bias(title) for title in tqdm(job_df["job_title"])]

# Pronoun prediction
print("\nRunning pronoun prediction...")
def get_pronoun_prediction(job_title):
    sentence = f"[MASK] arbejder som {job_title}."
    try:
        preds = fill_mask(sentence)
        top_preds = [p['token_str'].strip().lower() for p in preds]
        return top_preds
    except Exception as e:
        print(f"⚠️ Error predicting pronoun for '{job_title}': {e}")
        return []

results = []
for title in tqdm(job_df["job_title"]):
    preds = get_pronoun_prediction(title)
    han_rank = preds.index("han") + 1 if "han" in preds else None
    hun_rank = preds.index("hun") + 1 if "hun" in preds else None
    results.append({
        "job_title": title,
        "predicted_pronoun": preds[0] if preds else None,
        "contains_han": "han" in preds,
        "contains_hun": "hun" in preds,
        "han_rank": han_rank,
        "hun_rank": hun_rank,
        "predicted_list": preds
    })

pronoun_df = pd.DataFrame(results)

# Merge all
combined_df = job_df.merge(pronoun_df, on="job_title", how="left")

# Bias score & classification
def get_weight(rank):
    return {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}.get(rank, 0.0)

def bias_score(row):
    han = get_weight(row["han_rank"]) if pd.notna(row["han_rank"]) else 0.0
    hun = get_weight(row["hun_rank"]) if pd.notna(row["hun_rank"]) else 0.0
    return han - hun

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

# Group into periods
def map_period(year):
    if pd.isna(year):
        return "Unknown"
    if year == 2022:
        return "2022"
    elif year in [2024, 2025]:
        return "2024/2025"
    else:
        return "Other"

combined_df["Posting_Period"] = combined_df["Posting_Year"].apply(map_period)

# Save full result
combined_df.to_csv(output_file, index=False)
print(f"\nFull gender bias analysis saved to '{output_file}'")

# Optional: Summary grouping
summary = combined_df.groupby(["Posting_Period", "bias_class"]).size().unstack(fill_value=0)
summary.to_csv(grouped_file)
print(f" Grouped summary saved to '{grouped_file}'")

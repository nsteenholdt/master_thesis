import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from functools import lru_cache  

# Setup
tqdm.pandas()
output_file = "jobnet_gender_bias_full.csv"
grouped_file = "jobnet_gender_bias_grouped_summary.csv"
plot_file = "plot_outputs/bias_class_by_period.png"
os.makedirs("plot_outputs", exist_ok=True)

# Load full dataset
df_jtgba = pd.read_csv("df_desc_filt.csv")
df_jtgba["summary_PostingCreated"] = pd.to_datetime(df_jtgba["summary_PostingCreated"], errors="coerce")
df_jtgba["Posting_Year"] = df_jtgba["summary_PostingCreated"].dt.year

# Prepare job titles
job_df = df_jtgba[["summary_Occupation", "Posting_Year"]].dropna(subset=["summary_Occupation"])
job_df = job_df.rename(columns={"summary_Occupation": "job_title"})

# Load FastText vectors
print("\nLoading FastText vectors...")
model_static = KeyedVectors.load_word2vec_format("cc.da.300.vec")

male_words = ['mand', 'han']
female_words = ['kvinde', 'hun']

def avg_cos_sim(word, group):
    try:
        return np.mean([model_static.similarity(word.lower(), g) for g in group])
    except KeyError:
        return np.nan

def weat_score(title):
    words = title.lower().split()
    scores = [avg_cos_sim(word, male_words) - avg_cos_sim(word, female_words) for word in words if word in model_static]
    return np.mean(scores) if scores else np.nan

print("\nCalculating WEAT scores...")
job_df["weat_bias_score"] = [weat_score(title) for title in tqdm(job_df["job_title"])]

# Load Danish BERT
print("\nLoading Danish BERT...")
tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")
model = AutoModel.from_pretrained("Maltehb/danish-bert-botxo")
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

cosine = torch.nn.functional.cosine_similarity

# 🆕 More robust token matching
def get_word_embedding(sentence, target_word):
    tokens = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    input_ids = tokens['input_ids'][0]
    tokenized_word = tokenizer.tokenize(target_word)
    
    for i, token_id in enumerate(input_ids):
        decoded_token = tokenizer.convert_ids_to_tokens([token_id])[0]
        if decoded_token in tokenized_word:
            return outputs.last_hidden_state[0, i, :]
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

# 🆕 Cached pronoun prediction
print("\nRunning pronoun prediction...")

@lru_cache(maxsize=None)
def get_pronoun_prediction(job_title):
    sentence = f"[MASK] arbejder som {job_title}."
    try:
        preds = fill_mask(sentence)
        top_preds = [p['token_str'].strip().lower() for p in preds]
        return top_preds
    except Exception as e:
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

# Summary grouping
summary_df = combined_df.groupby(["Posting_Period", "bias_class"]).size().unstack(fill_value=0)
summary_df.to_csv(grouped_file)
print(f" Grouped summary saved to '{grouped_file}'")

summary_df = pd.read_csv("jobnet_gender_bias_grouped_summary.csv")

# import matplotlib.pyplot as plt
import seaborn as sns

# --- Load summary (optional if running separately) ---
#summary_df = pd.read_csv("jobnet_gender_bias_grouped_summary.csv", index_col=0)

# --- Ensure correct order (optional but recommended) ---
desired_order = ["2022", "2024/2025"]
summary_df = summary_df.reindex([p for p in desired_order if p in summary_df.index])

# ✅ Option 1: FIXED version of Pandas default bar plot
summary_df.plot(kind="bar", stacked=True, colormap="Set2", figsize=(10, 6))
plt.title("Bias Class Distribution by Posting Period")
plt.xlabel("Posting Period")
plt.ylabel("Number of Job Titles")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plot_outputs/bias_class_by_period_fixed.png", dpi=300)
plt.show()

# ✅ Option 2 (Recommended): Seaborn version with cleaner look
# Convert to long format for seaborn
summary_long = summary_df.reset_index().melt(id_vars="Posting_Period", 
                                              var_name="Bias Class", 
                                              value_name="Count")

plt.figure(figsize=(10, 6))
sns.barplot(data=summary_long, x="Posting_Period", y="Count", hue="Bias Class", palette="Set2")
plt.title("Bias Class Distribution by Posting Period")
plt.xlabel("Posting Period")
plt.ylabel("Number of Job Titles")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plot_outputs/bias_class_by_period_seaborn.png", dpi=300)
plt.show()

import pandas as pd
import re
from collections import Counter

# --- Load your job ads dataset ---
df = pd.read_csv("df_desc.csv")

# --- Load gendered word list ---
gendered_words_df = pd.read_csv("gendered_words_da.csv")
feminine_words = set(gendered_words_df[gendered_words_df['gender'] == 'feminine']['word'].str.lower())
masculine_words = set(gendered_words_df[gendered_words_df['gender'] == 'masculine']['word'].str.lower())

# --- Load ChatGPT-style word list (from file or define directly) ---
# Option A: Load from file
chatgpt_words_df = pd.read_csv("chatgpt_words_da.csv")
chatgpt_words = set(chatgpt_words_df['word'].str.lower())

# Option B: Define directly
# chatgpt_words = {"effektiv", "fleksibel", "proaktiv", "motiveret", "dynamisk", "nytænkende", "selvstændig"}

# --- Preprocessing and word counting function ---
def count_all_words(text):
    if pd.isna(text):
        return 0, 0, 0
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    word_counts = Counter(words)
    fem_count = sum(word_counts[word] for word in feminine_words if word in word_counts)
    masc_count = sum(word_counts[word] for word in masculine_words if word in word_counts)
    gpt_count = sum(word_counts[word] for word in chatgpt_words if word in word_counts)
    return fem_count, masc_count, gpt_count

# --- Apply the function to your job descriptions ---
df[['feminine_word_count', 'masculine_word_count', 'chatgpt_word_count']] = df[
    'details_JobPositionPosting_JobPositionInformation_Purpose'
].apply(lambda x: pd.Series(count_all_words(x)))

# Optional: Net gender bias score
df['gender_bias_score'] = df['feminine_word_count'] - df['masculine_word_count']

# --- Save result to file ---
df.to_csv("df_desc_with_gender_chatgpt_counts.csv", index=False)
print("Done! Counts added and saved.")

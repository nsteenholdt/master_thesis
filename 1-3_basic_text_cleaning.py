import os
import pandas as pd
import re
import unicodedata
import logging
import spacy
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")

# Load SpaCy and Danish stopwords
nlp = spacy.load("da_core_news_lg")
stop_words = set(stopwords.words("danish"))


# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "processed_data_danish")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data_preprocessed")
TEXT_COLUMN = "details_JobPositionPosting_JobPositionInformation_Purpose"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging setup 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Preprocessing function 
def basic_cleaning(text):
    try:
        # Use SpaCy to tokenize, lemmatize, remove punctuation and stopwords
        doc = nlp(str(text).lower())
        cleaned_tokens = [
            token.lemma_ for token in doc
            if token.is_alpha and token.lemma_ not in stop_words and not token.is_stop
        ]
        return " ".join(cleaned_tokens)
    except Exception as e:
        logging.error(f"Error cleaning text: {e}")
        return ""


# Processing loop 
for file in os.listdir(INPUT_DIR):
    if not file.endswith(".csv"):
        continue

    input_path = os.path.join(INPUT_DIR, file)
    output_path = os.path.join(OUTPUT_DIR, file)

    try:
        df = pd.read_csv(input_path, encoding="utf-8")

        if TEXT_COLUMN not in df.columns:
            logging.warning(f"'{TEXT_COLUMN}' column missing in {file}. Skipping.")
            continue

        df["text_preprocessed"] = df[TEXT_COLUMN].fillna("").apply(basic_cleaning)
        df = df[df["text_preprocessed"].str.len() > 5]  # remove near-empty rows

        df.to_csv(output_path, index=False, encoding="utf-8")
        logging.info(f"Cleaned and saved: {file} ({len(df)} rows)")

    except Exception as e:
        logging.error(f"Error processing {file}: {e}")

logging.info(f"\nAll preprocessed files saved to: {OUTPUT_DIR}")

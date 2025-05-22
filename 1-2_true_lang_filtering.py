import os
import pandas as pd
import fasttext
import logging

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "processed_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data_danish")
MODEL_PATH = os.path.join(BASE_DIR, "lid.176.bin")

# Column to use for language detection
TEXT_COLUMNS = [
    "details_JobPositionPosting_JobPositionInformation_Purpose"
]

# Confidence threshold for FastText language detection
MIN_CONFIDENCE = 0.90

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load FastText model ---
model = fasttext.load_model(MODEL_PATH)

# --- Language detection helper ---
def is_danish(text):
    try:
        clean_text = text.replace("\n", " ").strip()
        if len(clean_text) < 10:
            return False
        predictions = model.predict(clean_text)
        lang_label, confidence = predictions[0][0], predictions[1][0]
        return lang_label == "__label__da" and confidence >= MIN_CONFIDENCE
    except Exception as e:
        logging.error(f"Error in detection: {e}")
        return False

# --- Process each CSV file ---
for file in os.listdir(INPUT_DIR):
    if not file.endswith(".csv"):
        continue

    input_path = os.path.join(INPUT_DIR, file)
    output_path = os.path.join(OUTPUT_DIR, file)

    try:
        df = pd.read_csv(input_path, encoding="utf-8")

        # Use only the columns that exist in this file
        available_cols = [col for col in TEXT_COLUMNS if col in df.columns]
        if not available_cols:
            logging.warning(f"No usable text columns in {file}. Skipping.")
            continue

        # Filter out very short entries before detection
        df[available_cols[0]] = df[available_cols[0]].fillna("").astype(str)
        df = df[df[available_cols[0]].str.len() > 100]

        # Detect language
        mask = df[available_cols[0]].apply(is_danish)
        filtered_df = df[mask]

        if not filtered_df.empty:
            filtered_df.to_csv(output_path, index=False, encoding="utf-8")
            logging.info(f"Saved {len(filtered_df)} Danish rows from {file}.")
        else:
            logging.info(f"No confident Danish rows in {file}. Skipping.")

    except Exception as e:
        logging.error(f"Error processing {file}: {e}")

logging.info(f"\nFinished filtering. Output saved to: {OUTPUT_DIR}")

import os
import json
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define base directory dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to clean HTML content
def clean_html(text):
    if text:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ").strip()
    return None

# Function to recursively flatten nested dictionaries
def flatten_json(nested_json, parent_key='', sep='_'):
    flattened_dict = {}
    for key, value in nested_json.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened_dict.update(flatten_json(value, new_key, sep))
        elif isinstance(value, list):
            flattened_dict[new_key] = ', '.join(str(v) for v in value)
        else:
            flattened_dict[new_key] = value
    return flattened_dict

# Loop through all subdirectories
for subdir, _, files in os.walk(BASE_DIR):
    if ".git" in subdir:
        continue

    for file in tqdm(files, desc=f"Processing {os.path.basename(subdir)}"):
        if not file.lower().endswith(".json") or file.startswith("E"):
            continue

        file_path = os.path.join(subdir, file)
        relative_dir = os.path.relpath(subdir, BASE_DIR).replace(os.sep, "_")
        csv_filename = os.path.join(OUTPUT_DIR, f"{relative_dir}_{file.replace('.json', '.csv')}")

        # Skip if already converted
        if os.path.exists(csv_filename):
            continue

        try:
            # Check if the file is empty
            if os.path.getsize(file_path) == 0:
                logging.warning(f"Skipping empty file: {file_path}")
                continue

            # Read the JSON file safely
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()

                if not content:
                    logging.warning(f"Skipping file with only whitespace: {file_path}")
                    continue

                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON file: {file_path}")
                    continue

            # Flatten the JSON structure dynamically
            flattened_data = flatten_json(data)

            # Clean HTML fields
            for key in flattened_data.keys():
                if "description" in key.lower() or "purpose" in key.lower():
                    flattened_data[key] = clean_html(str(flattened_data[key]))

            # Convert to DataFrame and save
            df = pd.DataFrame([flattened_data])
            df.to_csv(csv_filename, index=False, encoding="utf-8")

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")

logging.info(f"\nData extraction complete. Individual CSV files are stored in: {OUTPUT_DIR}")

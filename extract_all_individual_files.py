import os
import json
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

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
        new_key = f"{parent_key}{sep}{key}" if parent_key else key  # Append parent key for nested keys
        if isinstance(value, dict):
            flattened_dict.update(flatten_json(value, new_key, sep))
        elif isinstance(value, list):
            # Convert lists to a readable string format
            flattened_dict[new_key] = ', '.join(str(v) for v in value)
        else:
            flattened_dict[new_key] = value
    return flattened_dict

# Loop through all subdirectories
for subdir, _, files in os.walk(BASE_DIR):
    for file in tqdm(files, desc=f"Processing {os.path.basename(subdir)}"):
        # Skip files that start with 'E'
        if file.startswith("E"):
            continue

        file_path = os.path.join(subdir, file)

        try:
            # Read the JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Flatten the JSON structure dynamically
            flattened_data = flatten_json(data)

            # Clean any HTML-containing fields (detect potential fields)
            for key in flattened_data.keys():
                if "description" in key.lower() or "purpose" in key.lower():
                    flattened_data[key] = clean_html(str(flattened_data[key]))

            # Convert to DataFrame
            df = pd.DataFrame([flattened_data])

            # Define output CSV path (same name as JSON but with .csv extension)
            csv_filename = os.path.join(OUTPUT_DIR, f"{file.replace('.json', '.csv')}")
            
            # Save each job entry as an individual CSV file
            df.to_csv(csv_filename, index=False, encoding="utf-8")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

print("\nData extraction complete! Individual CSV files are stored in:", OUTPUT_DIR)

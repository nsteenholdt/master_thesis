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
    if ".git" in subdir:  # Skip .git and system files
        continue

    for file in tqdm(files, desc=f"Processing {os.path.basename(subdir)}"):
        # Skip files that start with 'E'
        if file.startswith("E"):
            continue

        file_path = os.path.join(subdir, file)

        try:
            # Check if the file is empty
            if os.path.getsize(file_path) == 0:
                print(f"Skipping empty file: {file_path}")
                continue

            # Read the JSON file safely
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()

                if not content:
                    print(f"Skipping file with only whitespace: {file_path}")
                    continue

                # Attempt to parse JSON
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON file: {file_path}")
                    continue

            # Flatten the JSON structure dynamically
            flattened_data = flatten_json(data)

            # Clean any HTML-containing fields
            for key in flattened_data.keys():
                if "description" in key.lower() or "purpose" in key.lower():
                    flattened_data[key] = clean_html(str(flattened_data[key]))

            # Convert to DataFrame
            df = pd.DataFrame([flattened_data])

            # Define output CSV path
            csv_filename = os.path.join(OUTPUT_DIR, f"{file.replace('.json', '.csv')}")

            # Save each job entry as an individual CSV file
            df.to_csv(csv_filename, index=False, encoding="utf-8")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

print("\n✅ Data extraction complete! Individual CSV files are stored in:", OUTPUT_DIR)

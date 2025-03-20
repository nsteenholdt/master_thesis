import os
import json
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# Define base directory dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# List to store extracted job data
job_data = []

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

# Counter to limit number of files processed
file_count = 0
max_files = 10  # Limit processing to 10 files

# Loop through all subdirectories
for subdir, _, files in os.walk(BASE_DIR):
    for file in tqdm(files, desc=f"Processing {os.path.basename(subdir)}"):
        # Stop if we've processed 10 files
        if file_count >= max_files:
            break

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

            # Append to list
            job_data.append(flattened_data)
            file_count += 1  # Increment file count

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if file_count >= max_files:
        break  # Stop processing once we reach the limit

# Convert to DataFrame
df = pd.DataFrame(job_data)

# Save extracted data
output_file = "jobnet_sample_extracted_data.csv"
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"\nSample extraction complete! Saved to {output_file}")

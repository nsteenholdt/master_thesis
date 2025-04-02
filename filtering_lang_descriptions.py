import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define input and output directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "processed_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data_filtered")

# Ensure the filtered output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COLUMN = "details_JobPositionPosting_JobPositionInformation_Purpose"

# Loop through all CSVs in the processed_data folder
for file in os.listdir(INPUT_DIR):
    if not file.lower().endswith(".csv"):
        continue

    input_path = os.path.join(INPUT_DIR, file)
    output_path = os.path.join(OUTPUT_DIR, file)

    try:
        df = pd.read_csv(input_path, encoding="utf-8")

        if TARGET_COLUMN not in df.columns:
            logging.info(f"Column not found in {file}. Skipping.")
            continue

        # Filter out rows where the target column contains " you "
        mask = df[TARGET_COLUMN].astype(str).str.lower().str.contains(" you ")
        filtered_df = df[~mask]

        if not filtered_df.empty:
            filtered_df.to_csv(output_path, index=False, encoding="utf-8")
        else:
            logging.info(f"All rows in {file} were filtered out.")

    except Exception as e:
        logging.error(f"Failed to process {file}: {e}")

logging.info(f"\nFiltered job ads saved to: {OUTPUT_DIR}")

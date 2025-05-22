import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "processed_data")
FILTERED_DIR = os.path.join(BASE_DIR, "processed_data_filtered")

summary = []

for file in os.listdir(INPUT_DIR):
    if not file.endswith(".csv"):
        continue

    input_path = os.path.join(INPUT_DIR, file)
    filtered_path = os.path.join(FILTERED_DIR, file)

    try:
        original_df = pd.read_csv(input_path)
        filtered_df = pd.read_csv(filtered_path) if os.path.exists(filtered_path) else pd.DataFrame()

        summary.append({
            "File": file,
            "Original Ads": len(original_df),
            "Filtered Ads": len(filtered_df),
            "Removed": len(original_df) - len(filtered_df)
        })
    except Exception as e:
        print(f"Error processing {file}: {e}")

summary_df = pd.DataFrame(summary)
summary_df.loc["Total"] = summary_df[["Original Ads", "Filtered Ads", "Removed"]].sum()
print(summary_df)

# Save as table
summary_df.to_csv(os.path.join(BASE_DIR, "language_filtering_summary.csv"), index=False)

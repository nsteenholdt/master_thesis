import pandas as pd
import re

# --- Load CSV ---
# Replace with your actual file path
df = pd.read_csv("your_file.csv")

# Make sure date column is in datetime format
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# --- Define suffixes ---
feminine_suffixes = ["dame", "frue", "inde", "trice", "øse", "ske", "kone", "ert", "mor", "jomfru"]
masculine_suffixes = ["mand"]

# --- Function to classify gendered suffixes ---
def classify_gender(title):
    # Normalize title
    title = str(title).lower()
    
    # Match individual words in title against suffixes
    words = re.findall(r'\b\w+\b', title)
    
    for word in words:
        if any(word.endswith(suffix) for suffix in feminine_suffixes):
            return 'feminine'
        if any(word.endswith(suffix) for suffix in masculine_suffixes):
            return 'masculine'
    
    return 'none'  # No match

# --- Apply classification ---
df['gender_category'] = df['job_title'].apply(classify_gender)

# --- Summary counts ---
total_counts = df['gender_category'].value_counts()

# --- Distribution over time ---
df['year_month'] = df['date'].dt.to_period('M')  # or use 'Y' for year
time_distribution = df.groupby(['year_month', 'gender_category']).size().unstack(fill_value=0)

# --- Output results ---
print("\nTotal counts by gender category:")
print(total_counts)

print("\nTime distribution (monthly):")
print(time_distribution)

# Optional: Save to CSV
total_counts.to_csv("gendered_counts_total.csv")
time_distribution.to_csv("gendered_counts_over_time.csv")

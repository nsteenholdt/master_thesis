# Master's Thesis
### Nanna Marie Steenholdt 
This repository is authored by Nanna Marie Steenholdt for her Master's Thesis at Aarhus University, June 2025.

Start by running the startup.sh file by executing the following in a bash terminal:
./startup.sh

This will activate the virtual environment, download the necessary requirements. Ensure that you have a working kernel, so you can use the Jupiter Notebook(s).

The Repository consists of several parts.

NOTE: You will have to download the following resources and put into your repository by yourself. These resources have been excluded from the GitHub repository and put in a gitignore as they are quite big and uploading them is computationally extensive.

## Resources

This project uses several external models and datasets that are either large or require separate downloading. Please make sure you download or cache the following:

- **SpaCy Danish Model (`da_core_news_lg` or alternatively `da_core_news_sm`)**  
  Used for lemmatization and tokenization of Danish job descriptions.  
  Download using:  
  `python -m spacy download da_core_news_lg`
  or  `python3 -m spacy download da_core_news_lg`

- **FastText Word Vectors (Danish)**  
  Used to calculate WEAT gender bias scores for individual words and job titles.  
  Download `cc.da.300.vec` from:  
  [https://fasttext.cc/docs/en/crawl-vectors.html](https://fasttext.cc/docs/en/crawl-vectors.html)  
  Place the file in your project root.

- **Danish BERT (`Maltehb/danish-bert-botxo`)**  
  Used for contextual embeddings and pronoun prediction in job titles.  
  Automatically downloaded via Hugging Face Transformers when first run.  
  You can also pre-download it using:
  
    `from transformers import AutoTokenizer, AutoModel`
    `AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")`
    `AutoModel.from_pretrained("Maltehb/danish-bert-botxo")`

- **NLTK Resources (punkt, stopwords)**
  Used in various text processing and lexical diversity calculations.
  Automatically downloaded by the startup.sh script, or manually using:
    `import nltk`
    `nltk.download('punkt')`
    `nltk.download('stopwords')`

## Part 0 - Unused Scripts
These scripts were created in the process of making this analysis pipeline, but were not used in this thesis' final version of its analysis - however, some may find them relevant for other purposes, hence they have been kept in the repository.

### 0_extract_ALL_jobnet_data.py 
This script is NOT necessary to run, and is an alternative to the extract_json_to_csv.py script. Instead of processing and storing the files individually, it stores the data as one big .csv file. I dismissed this approach as the large file was difficult to work with going forward, but the script remains in the repository, in case it can be helpful for others, who may have different goals.

### 0_10_sample_extraction.py
A script that will extracts 10 samples. Available to check if extraction works as expected.

### gendered_titles_over_time.py 
This script identifies and counts job titles that contain explicitly gendered suffixes (e.g., *-mand*, *-inde*) and compares their frequency and proportions across two time periods: 2022 vs. 2024–2025. It outputs both summary tables and visualizations. NB: This specific analysis was dropped in the effort of saving time and to focus on the remaining analyses. It can be used for future studies though.

## Part 1 - Loading and preprocessing

After running the startup file, downloading the necessary resources you can proceed to the following:

### 1-1_extract_json_to_csv.py
This script processes the .json files that were acquired through the jobnet scraper. 
It takes the relevant files and transforms them to .csv files and stores them in a folder called 'processed_data'.
The script attempts to handle and avoid potential errors gracefully.

### 1-2_true_lang_filtering.py
Since only Danish job ads are desired, this script will filter out English ads using a FastText model. This script was seperated from the extraction script, in case one does not need to filter English job ads out.

### 1-3_basic_text_cleaning.py
This script performs basic text normalization on the Danish job ads. It processes each .csv file in the processed_data_danish folder, cleans the relevant text column (lowercasing, whitespace cleanup, Unicode normalization), removes nearly empty rows, and saves the result to the processed_data_preprocessed folder. The script includes logging and is designed to be robust to missing columns or file issues.

## Part 2 - Exploring Data & Descriptive Statistics

### 2-1_descriptive_stats.ipynb
This Jupiter Notebook includes several chunks, that are meant to explore the data: 

- Imports and parses the csv's as a dataframe
- Checks for and removes potential duplciates
- Drops empty columns and lists dropped columns
- Checks for missing values
- Provides quick infor and a summary statistic
- Plots the distributions of Job Titles
- Plots the Top 10 most used jobtitles for each Year Group.
- Plots bar charts showing the count of the amount of postings for the 2022 group and the 2024/2025 group
- Plots a word cloud of most used words for job titles (the one-liner in a jobadvertisement)
- Saves a now filtered version of the dataset
- Ensures that NLTK resources are available
- Removes data where job advertisements begin with a link 

### 2-2_word_count.ipynb
Compares word count between the 2022 group and the 2024/2025 group and tests for statistical significance
- Preprocessing
- NA Removal
- Computing Word Counts
- Summary Statistics
- Histogram: Word Count Distribution by Period Group 
- Box Plot: Word Count Boxplot By Period Group
- Q-Q Plots, regular and with log-transformation
- Levene's Test, regular and with log-transformation
- Welsch's T-test, regular and with log-transformation
- Cohen's D (Effect Size), regular and with log-transformation
- Robustness Check: 1) Mann-Whitney, 2) Welsch's T-test on a random subsample

### 2-3_lexical_diversity.ipynb
Compares Lexical Diversity between the 2022 group and the 2024/2025 group and tests for statistical significance.
- Preprocessing
- NA Removal
- Computing Lexical Diversity
- Summary Statistics
- Histogram: Distribution of Lexical Diversity by Period Group 
- Box Plot: Lexical Diversity Boxplot By Period Group
- Q-Q Plots, regular and with log-transformation
- Levene's Test, regular and with log-transformation
- Welsch's T-test, regular and with log-transformation
- Cohen's D (Effect Size), regular and with log-transformation
- Robustness Check: 1) Mann-Whitney, 2) Welsch's T-test on a random subsample

### 2-4_readability_score.ipynb
Compares the Readability Scores (Flesch-Kinkaid) between the 2022 group and the 2024/2025 group and tests for statistical significance.
- Imports nltk corpora
- Preprocessing
- NA Removal
- Computing Readability Scores
- Summary Statistics
- Histogram: Distribution of Readability Scores by Period Group 
- Box Plot: Readability Score Boxplot By Period Group
- Q-Q Plots, regular and with log-transformation
- Levene's Test, regular and with log-transformation
- Welsch's T-test, regular and with log-transformation
- Cohen's D (Effect Size), regular and with log-transformation
- Robustness Check: 1) Mann-Whitney, 2) Welsch's T-test on a random subsample

## Part 3 - Making a lexicon of gendered words

### 3-1_other_gendering_script.py 
Extracts frequently used adjectives and nouns from Danish job descriptions and calculates a gender association score for each word using fastText embeddings and balanced gender reference words.

## Part 4 - Word frequencies and other tallies

### 4-1_counting_gendered_words.py
Analyzes Danish job ads to count and compare gendered language over time, grouping them into "old" (2022) and "recent" (2024–2025) based on the frequency and ratio of feminine and masculine-coded words.

### 4-2_count_chat_words.py
Analyzes Danish job ads to count and compare the amount of "ChatGPT words" over time, grouping them into "old" (2022) and "recent" (2024–2025).

## Part 4 - Subtle Bias in Titles

### job_titles_gender_bias_analysis.py
Script that analyzes gender bias in Danish job titles using word embeddings, contextual similarity, and pronoun prediction, and classifies each title as masculine, feminine, mixed, or unclear.

## Part 5 - Statistical Testing

### Analysis.ipynb (WIP)
Notebook, in which we take the results so far and test for significant differences for the outputs from part 3 and 4. The analyses will incude:
- Distribution check: Histogram
- Q-Q plot 
- Box Plot 
- Levene's test (Homogeneity of Variance)
- Welsch's T-test
- Cohen's D
- Robustness check: Whitney-Mann U
- ...



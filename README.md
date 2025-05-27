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

  Download `lid.176.bin` through the scripts. 

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
This script identifies and counts job titles that contain explicitly gendered suffixes (e.g., *-mand*, *-inde*) and compares their frequency and proportions across two time periods: 2022 vs. 2024–2025. It outputs both summary tables and visualisations. NB: This specific analysis was dropped in the effort of saving time and to focus on the remaining analyses. It can be used for future studies though.

## Part 1 - Loading and preprocessing

After running the startup file, downloading the necessary resources you can proceed to the following:

### 1-1_extract_json_to_csv.py
This script processes the .json files that were acquired through the jobnet scraper. 
It takes the relevant files and transforms them to .csv files and stores them in a folder called 'processed_data'.
The script attempts to handle and avoid potential errors gracefully.

### 1-2_true_lang_filtering.py
Since only Danish job ads are desired, this script will filter out English ads using a FastText model. This script was seperated from the extraction script, in case one does not need to filter English job ads out.

### 1-3_basic_text_cleaning.py
This script performs basic text normalisation on the Danish job ads. It processes each .csv file in the processed_data_danish folder, cleans the relevant text column (lowercasing, whitespace cleanup, Unicode normalisation), removes nearly empty rows, and saves the result to the processed_data_preprocessed folder. The script includes logging and is designed to be robust to missing columns or file issues.

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

Text extraction and lemmatization using SpaCy

Frequency counting of relevant words (nouns/adjectives)

Vector-based gender scoring with fastText embeddings

Explicit labeling as masculine, feminine, or neutral

Logging progress and saving results

### 3-2_threshold_justification.py
Provides a statistical framework for selecting and justifying the optimal gender association threshold used to classify Danish words as masculine, feminine, or neutral. It analyses the distribution of gender scores from a previously generated lexicon and explores multiple threshold selection methods, including:

Percentile-based thresholding (e.g., 25th/75th percentiles)

K-means clustering of score distributions

Variance-based separation ratio for within vs. between-group spread

Custom scoring of candidate thresholds based on balance and separation quality

The script outputs a detailed justification report summarising the statistical properties of the distribution, the classification impact at the chosen threshold, and the agreement between different methods.

## Part 4 - Word frequencies and other tallies

### 4-1_counting_gendered_words.py
This script conducts a full analysis of gendered language in Danish job advertisements using a validated gender-scored lexicon. It performs preprocessing and lemmatization of job ad text, applies gender bias classification based on a selected threshold, and validates results through lexicon quality checks, threshold sensitivity analysis, and sampling of classified ads. The script compares gendered language use across time periods (2022 vs. 2024/2025) using statistical tests, and generates summary statistics, visualisations, and output files for further review. All results are saved in structured directories for validation, plots, and statistical summaries.
This is the main analysis script that processes Danish job ads, classifies gendered language, and validates results. It generates datasets, statistical test results, and visualizations used in subsequent analysis.

Main tasks and methods:

- Text preprocessing and lemmatization (using SpaCy)

- Gender classification using a validated threshold

- Lexicon validation:

- Top word inspection

- Gender label distribution

Threshold sensitivity analysis:

- Classification coverage

- Bias score differences

- Job ad sampling for manual inspection

- Danish context analysis (temporal and lexical trends)

Statistical tests performed:

- Welch’s t-test and Mann-Whitney U (bias score comparison)

- Levene’s test (equality of variance)

- Proportion Z-tests:

- Proportion of ads with gendered language

- Proportion of masculine-biased ads

- Chi-squared test (bias category distribution)

### 4-2_gender_wordcount.ipynb
This notebook analyzes the distribution and temporal change in gendered language usage in Danish job advertisements. It uses preprocessed output from the validated classification script and performs statistical comparisons between time periods (2022 vs. 2024/2025). The notebook includes analysis of gendered word ratios and gender bias scores, using:

- Descriptive statistics

- Histograms, boxplots, and Q-Q plots

- Welch’s t-tests, Mann-Whitney U, Levene’s test

- Cohen’s d for effect size estimation

### 4-3_count_chat_words.py

This script identifies and quantifies the use of ChatGPT-style language in Danish job advertisements from 2022 and 2024/2025. It scans job descriptions for a predefined list of GPT-like words and phrases, lemmatized using SpaCy, and calculates their frequency and normalized occurrence rate per ad. The output includes:

- Counts and ratios of GPT-style expressions per ad

- Aggregated frequency data by year group

- Horizontal bar plots showing the top 10 GPT-style words and phrases in each period

- A processed dataset saved as outputs/df_desc_with_chatgpt_counts.csv for downstream analysis

### 4-4_chatgpt_words.ipynb
This notebook analyzes the intensity and distribution of ChatGPT-style language in Danish job advertisements, using the output generated by the detect_chatgpt_language.py script. It focuses on comparing the normalized frequency of GPT-like words and phrases across two time periods: 2022 and 2024/2025.

The notebook includes:

- Summary statistics and distributions of GPT-style word/phrase usage

"Histograms, Q-Q plots, and boxplots (including log-transformed values)"

Statistical tests:

Levene’s test, Welch’s t-test, Mann-Whitney U (log-transformed ratios)

Proportion Z-test (presence vs. absence of GPT-style words)

A grouped bar chart showing the proportion of ads containing GPT-like language

All visualizations are saved to the plot_outputs/ directory for reporting.
### 4-5_gendered_titles.py
This script analyzes gender bias in Danish job titles based on linguistic suffixes traditionally associated with masculine or feminine forms. It processes job ads from 2022 and 2024/2025, classifies titles using an enhanced rule-based method, and compares distributions across time.

The analysis includes:

- Cleaning and validating job title data

- Rule-based gender classification using both word-final and embedded suffix patterns

- Aggregation of gendered title counts and percentages

- Stacked bar plots for overall and gendered-only distributions

- Statistical testing:

- Z-tests comparing feminine and masculine title proportions

- Confidence intervals and Cohen’s h effect sizes

- Bonferroni-adjusted significance thresholds

Outputs are saved to the stats_outputs/ and plot_outputs/ directories, including descriptive summaries, statistical results, and comparison visuals.

### 4-6_title_bias.ipynb
This notebook analyzes the distribution of gender-coded job titles across time periods (2022 vs. 2024/2025). It uses pre-labeled job titles classified as feminine, masculine, both, or none, and focuses on titles with a clear gender association.

The notebook includes:

- Filtering of job titles to only feminine and masculine categories

- Aggregation of gendered title counts and percentages by period

- Stacked bar plots visualizing gendered title distributions over time

- Proportion Z-tests comparing changes in feminine title usage between periods

Outputs include CSV summaries and plots saved to the appropriate output directories for downstream reporting.


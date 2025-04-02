# master_thesis
Repository for my Master's Thesis

Start by running the startup.sh file by executing the following in a bash terminal:
./startup.sh

This will activate the virtual environment, download the necessary requirements. Ensure that you have a working kernel, so you can use the Jupiter Notebook(s).

The Repository consists of several parts.

NOTE: You will have to download the following resources and put into your repository by yourself. These resources have been excluded from the GitHub repository and put in a gitignore as they are quite big and uploading them is computationally extensive.

### Resources


### Part 1 - Loading and preprocessing

After running the startup file, downloading the necessary resources you can proceed to the following:

# extract_json_to_csv.py
This script processes the .json files that were acquired through the jobnet scraper. 
It takes the relevant files and transforms them to .csv files and stores them in a folder called 'processed_data'.
The script attempts to handle and avoid potential errors gracefully.

# extract_ALL_jobnet_data.py
This script is NOT necessary to run, and is an alternative to the extract_json_to_csv.py script. Instead of processing and storing the files individually, it stores the data as one big .csv file. I dismissed this approach as the large file was difficult to work with going forward, but the script remains in the repository, in case it can be helpful for others, who may have different goals.

# filtering_lang_descriptions.py
Since only Danish job ads are desired, this script will filter out English ads by detecting the word "you" in the job descriptions. This script was seperated from the extraction script, in case one may need the English job ads for something later. 

### Part 2 - Exploring Data & Descriptive Statistics

# descriptive_stats.ipynb
This Jupiter Notebook includes several chunks, that are meant to explore the data: 

- Imports the csv's as a dataframe
- Provides basic info on the df
- Provides summary statistics
- Checks for missing values
- Provides a list of how many times each unique job title appears
- Plots the Top 10 jobtitles by count
- Plots the count of the amount of postings for each time point
- Plots a word cloud for job titles
- Ensures there are no duplicate job postings
- Compares word count distribution between time groups
- Plots the Lexical Diversity for both time groups
- Plots the Readability Scores (Flesch-Kinkaid) for both time groups
- Makes word clouds for most common words used in the job descriptions

### Part 3 - Making a lexicon of gendered words

# other_gendering_script.py 
Extracts frequently used adjectives and nouns from Danish job descriptions and calculates a gender association score for each word using fastText embeddings and balanced gender reference words.

### Part 4 - Word frequencies and other tallies

# counting_gendered_words.py
Analyzes Danish job ads to count and compare gendered language over time, grouping them into "old" (2022) and "recent" (2024–2025) based on the frequency and ratio of feminine and masculine-coded words.

# count_chat_words.py
Analyzes Danish job ads to count and compare the amount of "ChatGPT words" over time, grouping them into "old" (2022) and "recent" (2024–2025).

# gendered_titles_over_time.py
This script identifies and counts job titles that contain explicitly gendered suffixes (e.g., *-mand*, *-inde*) and compares their frequency and proportions across two time periods: 2022 vs. 2024–2025. It outputs both summary tables and visualizations.

### Part 4 - Subtle Bias in Titles

# job_titles_gender_bias_analysis.py
Script that analyzes gender bias in Danish job titles using word embeddings, contextual similarity, and pronoun prediction, and classifies each title as masculine, feminine, mixed, or unclear.

### Part 5 - Statistical Testing

# Analysis.ipynb (WIP)
Notebook, in which we take the results so far and test for significant differences.
- Distribution checks
- Normalisation checks
- (T-test)
- ...



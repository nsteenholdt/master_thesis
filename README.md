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

### Part 2 - Exploring Data & Descriptive Statistics

# descriptive_stats.ipynb
This Jupiter Notebook includes several chunks, that are meant to explore the data such as: 


### Part 3 - 

# job_titles_gender_bias_analysis.py

# gendered_titles_over_time.py

# other_gendering_script.py *

# count_chat_words.py

# counting_gendered_words.py



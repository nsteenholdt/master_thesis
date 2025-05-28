#!/bin/bash

# Navigate to the jobnet folder
cd "$(dirname "$0")"


# Check for virtual environment
if [ ! -d "jobnet_venv" ]; then
    echo "Error: Virtual environment 'jobnet_venv' not found!"
    exit 1
fi

# Activate virtual environment
source jobnet_venv/bin/activate
echo "Success: Virtual environment activated: $(which python)"

# Check for requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found!"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# macOS SSL fix
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🔧 Running macOS SSL certificate fix..."
    /Applications/Python\ 3.12/Install\ Certificates.command
fi

# Create output folder for plots if needed
mkdir -p plot_outputs

# Download required NLTK resources
echo "📚 Downloading NLTK resources..."
python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True)"

# Download SpaCy Danish model (large) if missing
echo "Checking SpaCy model..."
if ! python -c "import spacy; spacy.load('da_core_news_lg')" &> /dev/null; then
    echo "Downloading SpaCy model da_core_news_lg..."
    python -m spacy download da_core_news_lg
fi

# Warm-up Hugging Face model
echo "Initializing Hugging Face Danish BERT model (Maltehb/danish-bert-botxo)..."
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('Maltehb/danish-bert-botxo'); AutoModel.from_pretrained('Maltehb/danish-bert-botxo')"

# Reminder for FastText
echo "   Reminder: Please manually download FastText Danish vectors (cc.da.300.vec) from:"
echo "   https://fasttext.cc/docs/en/crawl-vectors.html"
echo "   And place them in the project root folder."

# Done
echo "You're ready to run the scripts. Make sure your Jupyter kernel uses the virtual environment."

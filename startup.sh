#!/bin/bash

# Navigate to the jobnet folder
cd "$(dirname "$0")"

# Check for virtual environment
if [ ! -d "jobnet_venv" ]; then
    echo "Virtual environment 'jobnet_venv' not found!"
    exit 1
fi

# Activate virtual environment
source jobnet_venv/bin/activate
echo "Virtual environment activated: $(which python)"

# Check for requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "requirements.txt not found!"
    exit 1
fi

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# macOS SSL fix
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Running macOS SSL certificate fix..."
    /Applications/Python\ 3.12/Install\ Certificates.command
fi

# Create output folder for plots if needed
mkdir -p plot_outputs

# Download required NLTK resources
echo "Downloading NLTK resources..."
python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True)"

# Done
echo "All set. Launching Jupyter Notebook..."
jupyter notebook

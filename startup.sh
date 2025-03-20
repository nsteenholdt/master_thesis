#!/bin/bash

# Navigate to the jobnet folder 
cd "$(dirname "$0")"

# Activate virtual environment
source jobnet_venv/bin/activate

# Print status
echo "Virtual environment activated: $(which python)"

# Launch Jupyter Notebook
jupyter notebook

#!/bin/bash
# Script to run the Streamlit application

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate t2f

# Verify Python and streamlit are available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found in conda environment 't2f'"
    echo "Please run: conda activate t2f && conda install python pip -y"
    exit 1
fi

if ! command -v streamlit &> /dev/null; then
    echo "Error: Streamlit not found in conda environment 't2f'"
    echo "Please run: conda activate t2f && pip install streamlit"
    exit 1
fi

# Run Streamlit app
echo "Starting Streamlit app..."
streamlit run Generation/app.py
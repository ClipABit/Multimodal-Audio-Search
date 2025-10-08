#!/bin/bash

echo "ClipABit Audio Search - macOS/Linux Setup Script"
echo "================================================"
echo

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found! Please install Python 3.8+ first"
    echo "macOS: Install from python.org or use: brew install python"
    echo "Linux: sudo apt-get install python3 python3-pip python3-venv"
    exit 1
fi

echo "Python found! Creating virtual environment..."
python3 -m venv audio_search_venv

echo "Activating virtual environment..."
source audio_search_venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "================================================"
echo "Setup complete!"
echo
echo "To run the app:"
echo "1. Open Terminal"
echo "2. Navigate to this folder"  
echo "3. Run: source audio_search_venv/bin/activate"
echo "4. Run: streamlit run streamlit_app.py"
echo
echo "The app will open at http://localhost:8501"
echo "================================================"
#!/bin/bash

# Setup script for the document extraction system
echo "Setting up Document Text Extraction System..."

# Check Python version
python3 --version

# Install dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Download spaCy model
echo "Downloading spaCy English model..."
python3 -m spacy download en_core_web_sm

# Test basic functionality
echo "Testing basic functionality..."
python3 test_basic.py

echo "Setup complete! Run 'python3 main.py --help' to get started."
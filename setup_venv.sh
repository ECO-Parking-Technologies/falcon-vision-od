#!/bin/bash

# Name of the virtual environment
VENV_NAME="falcon-vision-od-venv"

# Create Python virtual environment
echo "Creating virtual environment..."
python3 -m venv $VENV_NAME

# Activate the environment
source $VENV_NAME/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Environment setup complete."
echo "To activate manually later, run: source $VENV_NAME/bin/activate"
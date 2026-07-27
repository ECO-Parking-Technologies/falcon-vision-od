#!/usr/bin/env bash
set -e

VENV_NAME="falcon-vision-od-venv"

echo "🗑  Cleaning up old virtual environment..."
rm -rf $VENV_NAME

echo "🛠  Creating virtual environment..."
python3 -m venv $VENV_NAME
source $VENV_NAME/bin/activate

echo "🆙  Upgrading pip & wheel..."
python3 -m pip install --upgrade pip wheel

echo "📦  Installing requirements..."
python3 -m pip install -r requirements.txt

echo "✅  Setup complete! Activate with:"
echo "    source $VENV_NAME/bin/activate"

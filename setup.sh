#!/bin/bash

echo "Creating virtual environment..."
python -m venv venv

echo "Virtual environment created."

echo "Activating virtual environment..."
source venv/Scripts/activate

echo "Virtual environment activated."

echo "Installing required libraries..."
pip install torch diffusers transformers accelerate scipy pillow matplotlib

echo "Libraries installed."

echo "Freezing requirements into requirements.txt..."
pip freeze > requirements.txt

echo "Setup complete!"
echo "You can now open the notebook and start working!"

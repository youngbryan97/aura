#!/bin/bash
# Aura Zenith: Apple Silicon setup script
set -e

echo "🍏 Initializing Aura Zenith Environment (Apple Silicon)..."

# Check for Brew
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew is required."
    exit 1
fi

# Install dependencies
echo "📦 Installing system dependencies..."
brew install python@3.12 prometheus redis llama.cpp

# Setup Virtualenv
python3.12 -m venv venv
source venv/bin/activate

# Install Python requirements
pip install --upgrade pip
pip install numpy psutil rich pydantic pydantic-settings pytest pytest-asyncio RestrictedPython prometheus-client motor redis

echo "📚 Installing Python requirements..."
pip install -r requirements_hardened.txt

echo "🧠 Fetching Aura runtime models..."
AURA_LOCAL_BACKEND=llama_cpp python scripts/fetch_models.py

echo "✅ Setup complete. Use 'source venv/bin/activate' to enter environment."

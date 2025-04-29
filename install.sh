#!/bin/bash

# ByBit AI Bot Installer

echo "🛠 Updating system..."
sudo apt update && sudo apt upgrade -y

echo "🛠 Installing dependencies..."
sudo apt install -y python3 python3-pip python3-venv git

echo "🛠 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

echo "🛠 Installing Python requirements..."
pip install --upgrade pip
pip install fastapi uvicorn python-binance pandas scikit-learn tensorflow transformers pyyaml requests aiohttp websockets

echo "✅ Installation complete."
echo "➡️ Edit your API keys in config/settings.yaml before starting."

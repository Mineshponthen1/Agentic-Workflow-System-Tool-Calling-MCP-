#!/usr/bin/env bash
# One-time setup when the codespace is created: Python packages, the panel, Ollama and llama3.2.
set -e

echo "==> 1/4 Installing Python packages"
pip install -r requirements.txt

echo "==> 2/4 Building the control panel"
(cd panel && npm ci && npm run build)

echo "==> 3/4 Installing Ollama"
sudo apt-get update -qq && sudo apt-get install -y -qq zstd > /dev/null
curl -fsSL https://ollama.com/install.sh | sh

echo "==> 4/4 Downloading llama3.2 (about 2 GB, takes a few minutes)"
bash .devcontainer/start-ollama.sh
ollama pull llama3.2

echo ""
echo "Setup complete. Try:  python eval/run_eval.py"

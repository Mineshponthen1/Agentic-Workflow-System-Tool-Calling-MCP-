#!/usr/bin/env bash
# Starts the Ollama model server in the background (runs every time the codespace starts).
if ! curl -s http://127.0.0.1:11434/api/version > /dev/null 2>&1; then
  setsid nohup ollama serve > /tmp/ollama.log 2>&1 < /dev/null &
  for _ in $(seq 1 30); do
    curl -s http://127.0.0.1:11434/api/version > /dev/null 2>&1 && break
    sleep 1
  done
fi
echo "Ollama is running."

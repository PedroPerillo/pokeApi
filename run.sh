#!/usr/bin/env bash
# Start the FastAPI backend and the Streamlit frontend together. Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
[ -f .env ] || { echo "Missing .env - copy .env.example and add your GROQ_API_KEY"; exit 1; }

.venv/bin/uvicorn backend.main:app --port 8000 &
API_PID=$!
trap 'kill $API_PID' EXIT
.venv/bin/streamlit run frontend/app.py --server.port 8501

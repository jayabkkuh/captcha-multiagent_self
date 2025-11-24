#!/bin/bash
export OPENAI_API_KEY="sk-MaIP9em9tPt8Dqh5nYYkfgbRJIM5E8EyN8HRnlvFSDCknViY"
export OPENAI_BASE_URL="https://api.qingyuntop.top/v1"
export OPENAI_MODEL="gemini-2.5-flash"

echo "Starting CAPTCHA server..."
echo "Model: $OPENAI_MODEL"
echo "URL: $OPENAI_BASE_URL"

python3 main.py serve --port 8001 > server.log 2>&1 &
echo "Server started on port 8001 (PID: $!)"

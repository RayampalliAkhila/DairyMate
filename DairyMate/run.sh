#!/usr/bin/env bash
# Dairy Mate launcher (macOS / Linux)
set -euo pipefail
cd "$(dirname "$0")"
[ -f venv/bin/activate ] && source venv/bin/activate
exec streamlit run app.py

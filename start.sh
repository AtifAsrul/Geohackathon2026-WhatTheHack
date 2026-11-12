#!/bin/bash
# Start Exploration Copilot.     bash start.sh
# Leave this terminal window open while you use the app. Press Ctrl+C here to stop it.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "The app is not installed yet. Type:   bash setup.sh"
  exit 1
fi
export COPILOT_OPEN_BROWSER=1
exec .venv/bin/python -m copilot.server

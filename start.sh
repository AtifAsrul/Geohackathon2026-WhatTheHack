#!/bin/bash
# Start Exploration Copilot.     bash start.sh
# Leave this terminal window open while you use the app. Press Ctrl+C here to stop it.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "The app is not installed yet. Type:   bash setup.sh"
  exit 1
fi
export COPILOT_OPEN_BROWSER=1
export PYTHONUNBUFFERED=1
# These exports affect this app and its children, not the login shell/system.
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export OMP_NUM_THREADS=1 OMP_MAX_ACTIVE_LEVELS=1
LOGDIR="${COPILOT_WORKSPACE:-$PWD/workspace}/logs"
mkdir -p "$LOGDIR" || exit 1
exec > >(tee -a "$LOGDIR/server.log") 2>&1
echo "Terminal progress and local pipeline/GPU/conversion logs will appear below."
echo "Server log: $LOGDIR/server.log"
echo "Bars show stage position, not estimated time remaining."
exec .venv/bin/python -u scripts/terminal.py run --label "App server" --workspace "${COPILOT_WORKSPACE:-$PWD/workspace}" -- .venv/bin/python -u -m copilot.server

#!/bin/bash
# App-local installation; existing environments/models are reused.
cd "$(dirname "$0")" || exit 1
set -o pipefail
PYBIN="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=0
export HF_HOME="${HF_HOME:-$PWD/models/.cache/huggingface}"
LOGDIR="${COPILOT_WORKSPACE:-$PWD/workspace}/logs"
mkdir -p "$LOGDIR" || exit 1
exec > >(tee -a "$LOGDIR/setup.log") 2>&1
SECONDS=0
WARNINGS=0
say() {
  local done=$(( $1 - 1 )) bar="" i
  for ((i=0; i<6; i++)); do
    if ((i < done)); then bar+="#"; elif ((i == done)); then bar+=">"; else bar+="-"; fi
  done
  echo
  echo "[$bar] Setup step $1/6 - $2 (elapsed ${SECONDS}s)"
}
run() {
  local label="$1"; shift
  "$PYBIN" -u scripts/terminal.py run --label "$label" --sizes -- "$@"
  local code=$?
  if ((code == 130 || code == 143)); then exit "$code"; fi
  return "$code"
}
optional() {
  if ! run "$@"; then
    WARNINGS=$((WARNINGS + 1))
    echo "[WARNING] $1 failed; see output above. Its fallback will be used if available."
  fi
}
trap 'code=$?; if ((code != 0)); then echo "[FAILED] Setup stopped (exit $code). Log: $LOGDIR/setup.log"; fi' EXIT
echo "Setup started: $(date -Is)"
echo "Detailed output is also saved to $LOGDIR/setup.log"
echo "Bars count setup steps, not elapsed-time percentage. Quiet commands report every 15 seconds."

say 1 "Python environment"
"$PYBIN" -V || { echo "Python was not found. Stop and ask a mentor."; exit 1; }
if [ ! -x .venv/bin/python ]; then
  run "Create app environment" "$PYBIN" -m venv --system-site-packages .venv \
    || run "Create isolated app environment" "$PYBIN" -m venv .venv || exit 1
else
  echo "[REUSE] Existing .venv"
fi
optional "Upgrade app pip" .venv/bin/python -m pip install --upgrade pip --progress-bar on

say 2 "App packages"
# Required packages must succeed before setup can report completion.
run "Install required app packages" .venv/bin/python -m pip install --progress-bar on -r requirements.txt || exit 1
if ! .venv/bin/python -c "import torch" 2>/dev/null; then
  optional "Install PyTorch (large download)" .venv/bin/python -m pip install --progress-bar on torch torchvision
else
  echo "[REUSE] PyTorch is available"
fi

say 3 "NCS foundation model code"
optional "Install transformers" .venv/bin/python -m pip install --progress-bar on "transformers[torch]~=4.50.2" "huggingface-hub>=0.24,<1.0"
if ! run "Install NCS from GitHub" .venv/bin/python -m pip install --progress-bar on --no-deps "git+https://github.com/NorskRegnesentral/NCS_models"; then
  optional "Install NCS from ZIP fallback" .venv/bin/python -m pip install --progress-bar on --no-deps "https://github.com/NorskRegnesentral/NCS_models/archive/refs/heads/main.zip"
fi

say 4 "OpenVDS converter and reader (app environment only)"
optional "Install OpenVDS wheel" .venv/bin/python -m pip install --progress-bar on --only-binary=:all: "openvds==3.4.9"

say 5 "Pretrained RGT, NCS and FaultSSL models"
optional "Download RGT-Est and NCS models" .venv/bin/python -u scripts/download_models.py
optional "Install FaultSSL download helper" .venv/bin/python -m pip install gdown
optional "Download FaultSSL Precision First" .venv/bin/python -u scripts/download_faultssl.py

say 6 "Self-check"
run "Check installed components and HPC access" .venv/bin/python -u scripts/check.py || exit 1

echo
echo "[######] Setup steps finished in ${SECONDS}s; $WARNINGS optional command warning(s)."
echo "Review [--] in the self-check too: these identify missing components/fallbacks."
echo "Log: $LOGDIR/setup.log"
echo "To start: bash start.sh"

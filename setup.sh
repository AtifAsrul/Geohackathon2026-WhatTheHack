#!/bin/bash
# One-time installation of Exploration Copilot on the HPC.
#     bash setup.sh
# Safe to run again: finished parts are skipped. Nothing here needs administrator rights.

cd "$(dirname "$0")" || exit 1
PYBIN="${PYTHON:-python3}"
say() { echo; echo "=================================================================="; echo "  $1"; echo "=================================================================="; }

say "Step 1 of 5 - Python environment"
"$PYBIN" -V || { echo "Python was not found. Stop and ask a mentor."; exit 1; }
if [ ! -x .venv/bin/python ]; then
  # reuse the HPC's own PyTorch if the default Python has it
  "$PYBIN" -m venv --system-site-packages .venv || "$PYBIN" -m venv .venv || { echo "Could not create the environment."; exit 1; }
fi
.venv/bin/python -m pip install --upgrade pip -q

say "Step 2 of 5 - App packages"
.venv/bin/pip install -q -r requirements.txt || echo "!! Some app packages failed to install (see messages above)."
if ! .venv/bin/python -c "import torch" 2>/dev/null; then
  echo "PyTorch is not in the default Python - installing it (large download, please wait)..."
  .venv/bin/pip install -q torch torchvision || echo "!! PyTorch could not be installed. The app will use its simpler fallback methods."
fi

say "Step 3 of 5 - NCS seismic foundation model code"
.venv/bin/pip install -q "transformers[torch]~=4.50.2" "huggingface-hub>=0.24,<1.0" || echo "!! transformers could not be installed."
.venv/bin/pip install -q --no-deps "git+https://github.com/NorskRegnesentral/NCS_models" 2>/dev/null \
  || .venv/bin/pip install -q --no-deps "https://github.com/NorskRegnesentral/NCS_models/archive/refs/heads/main.zip" \
  || echo "!! NCS could not be installed. Facies will use the simpler fallback."

say "Step 4 of 5 - SDIP converter (SEG-Y to AI-ready format), in its own environment"
if [ ! -x .venv-sdip/bin/sdip ]; then
  "$PYBIN" -m venv .venv-sdip && .venv-sdip/bin/python -m pip install --upgrade pip -q
  .venv-sdip/bin/pip install -q "git+https://github.com/zahidaramai/sdip@v1.7.0" 2>/dev/null \
    || .venv-sdip/bin/pip install -q "https://github.com/zahidaramai/sdip/archive/refs/tags/v1.7.0.zip" \
    || echo "!! SDIP could not be installed. The app will read SEG-Y with its built-in reader instead."
fi

say "Step 5 of 5 - Pretrained models (RGT-Est and NCS), about 1 GB in total"
.venv/bin/python scripts/download_models.py

say "Self-check"
.venv/bin/python scripts/check.py

echo
echo "Setup finished. To start the app, type:   bash start.sh"

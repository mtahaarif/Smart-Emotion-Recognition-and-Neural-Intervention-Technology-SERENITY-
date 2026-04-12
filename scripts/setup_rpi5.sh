#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
PIWHEELS_URL="https://www.piwheels.org/simple"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN not found. Install Python 3.11 first." >&2
  exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" ]]; then
  echo "WARNING: Detected architecture '$ARCH'. Raspberry Pi 5 production path expects 64-bit aarch64." >&2
fi

"$PYTHON_BIN" - <<'PY'
import sys
v = sys.version_info[:2]
if not ((3, 10) <= v < (3, 13)):
    raise SystemExit(
        f"ERROR: Unsupported Python {sys.version.split()[0]}. Use Python 3.10-3.12 (3.11 recommended)."
    )
print(f"Using Python {sys.version.split()[0]}")
PY

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "Installing edge dependencies from requirements-edge.txt ..."
if ! pip install --extra-index-url "$PIWHEELS_URL" -r requirements-edge.txt; then
  echo "Primary install failed. Retrying without tflite-runtime and using TensorFlow fallback ..."
  TMP_REQ="$(mktemp)"
  grep -v '^tflite-runtime' requirements-edge.txt > "$TMP_REQ"
  pip install --extra-index-url "$PIWHEELS_URL" -r "$TMP_REQ"
  pip install tensorflow==2.18.0 numpy==1.26.4 opencv-python-headless==4.8.1.78
  rm -f "$TMP_REQ"
fi

echo "Setup complete. Activate environment with: source .venv/bin/activate"

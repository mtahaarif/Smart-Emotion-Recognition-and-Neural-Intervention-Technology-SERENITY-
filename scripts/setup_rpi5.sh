#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
PIWHEELS_URL="https://www.piwheels.org/simple"
FORCE_RECREATE_VENV="${FORCE_RECREATE_VENV:-false}"

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

TARGET_MM="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"

if [[ -d .venv ]]; then
  if [[ "$FORCE_RECREATE_VENV" == "true" ]]; then
    echo "FORCE_RECREATE_VENV=true -> recreating .venv"
    rm -rf .venv
  elif [[ ! -x .venv/bin/python ]]; then
    echo "Existing .venv is missing python executable. Recreating .venv"
    rm -rf .venv
  else
    CURRENT_MM="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
    if [[ -z "$CURRENT_MM" ]]; then
      echo "Could not detect existing .venv Python version. Recreating .venv"
      rm -rf .venv
    elif ! .venv/bin/python - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)
PY
    then
      echo "Existing .venv uses unsupported Python $CURRENT_MM. Recreating with $TARGET_MM"
      rm -rf .venv
    elif [[ "$CURRENT_MM" != "$TARGET_MM" ]]; then
      echo "Existing .venv Python ($CURRENT_MM) differs from PYTHON_BIN target ($TARGET_MM). Recreating .venv"
      rm -rf .venv
    fi
  fi
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python - <<'PY'
import sys
v = sys.version_info[:2]
if not ((3, 10) <= v < (3, 13)):
    raise SystemExit(f"ERROR: .venv uses unsupported Python {sys.version.split()[0]}")
print(f"Active venv Python: {sys.version.split()[0]}")
PY

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

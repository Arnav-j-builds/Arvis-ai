#!/usr/bin/env bash
# ============================================================
#   Jarvis - One-click launcher (macOS / Linux)
#   Run with:  ./run.sh
# ============================================================
set -e

cd "$(dirname "$0")"

echo
echo "============================================"
echo "  Starting Jarvis..."
echo "============================================"
echo

# ---- 1. Locate Python ------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=python
else
    echo "[ERROR] Python 3 was not found. Please install Python 3.10+."
    exit 1
fi

echo "[OK] Using Python: $PYTHON_CMD"
$PYTHON_CMD --version

# ---- 2. Create virtualenv on first run ------------------------------
if [ ! -x ".venv/bin/python" ]; then
    echo
    echo "[INFO] First run - creating virtual environment in .venv ..."
    $PYTHON_CMD -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# ---- 3. Install / update requirements --------------------------------
echo
echo "[INFO] Checking dependencies..."
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

# ---- 4. Make sure Ollama is running ----------------------------------
if command -v ollama >/dev/null 2>&1; then
    echo
    echo "[INFO] Checking Ollama..."
    if ! ollama list >/dev/null 2>&1; then
        echo "[INFO] Starting Ollama server..."
        ollama serve >/dev/null 2>&1 &
        sleep 3
    else
        echo "[OK] Ollama is already running."
    fi
else
    echo
    echo "[WARN] Ollama was not found. Install it from https://ollama.com"
    echo "       and run: ollama pull qwen3:1.7b"
fi

# ---- 5. Launch Jarvis ------------------------------------------------
echo
echo "============================================"
echo "  Jarvis is starting. Say \"arvis\" to wake it."
echo "  Press Ctrl+C to stop it."
echo "============================================"
echo

python app.py

echo
echo "============================================"
echo "  Jarvis has stopped."
echo "============================================"

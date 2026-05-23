#!/usr/bin/env bash
set -euo pipefail

OPENCODE_CONFIG="$HOME/.config/opencode/opencode.jsonc"
DCP_CONFIG="$HOME/.config/opencode/dcp.jsonc"
PROVIDER_NAME="mlx-turboquant"

echo ""
echo "======================================================"
echo "  TurboQuant MLX → opencode Launcher"
echo "======================================================"
echo ""

# ── Auto-detect port ──────────────────────────────────────
PORT=""

# 1. Check saved defaults
PORT=$(python3 -c "
import json
from pathlib import Path
cfg = Path.home() / '.tq_defaults.json'
if cfg.exists():
    d = json.loads(cfg.read_text())
    print(d.get('port', ''))
" 2>/dev/null || true)

# 2. Scan common ports for running TurboQuant server
if [[ -z "$PORT" ]]; then
    for p in $(seq 8080 8095); do
        if curl -sf "http://127.0.0.1:${p}/health" > /dev/null 2>&1; then
            PORT=$p
            break
        fi
    done
fi

# 3. Scan all listening ports for python processes serving /health
if [[ -z "$PORT" ]]; then
    for p in $(lsof -iTCP -sTCP:LISTEN -P -n 2>/dev/null | grep python | awk '{print $9}' | cut -d: -f2 | sort -un); do
        if curl -sf "http://127.0.0.1:${p}/health" > /dev/null 2>&1; then
            PORT=$p
            break
        fi
    done
fi

if [[ -z "$PORT" ]]; then
    echo "✗ No running TurboQuant server found"
    echo "  Start one with: mlx-turboquant.py"
    exit 1
fi

# ── Check server health and get model info ────────────────
HEALTH_URL="http://127.0.0.1:${PORT}/health"
MODELS_URL="http://127.0.0.1:${PORT}/v1/models"

echo -n "  Checking TurboQuant server on port ${PORT}... "
HEALTH=$(curl -sf "$HEALTH_URL" 2>/dev/null) || {
    echo "NOT RUNNING"
    echo "  Start: mlx-turboquant.py"
    exit 1
}
echo "OK"

MODEL_NAME=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model','unknown'))")
STRATEGY=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('strategy','unknown'))")
MODEL_ID=$(curl -sf "$MODELS_URL" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',[{}])[0].get('id','default'))" 2>/dev/null || echo "$MODEL_NAME")

echo "  Model:    $MODEL_NAME"
echo "  Strategy: $STRATEGY"
echo "  Model ID: $MODEL_ID"
echo ""

# ── Update opencode config ────────────────────────────────
echo -n "  Updating opencode config... "

python3 -c "
import json, os, re

cfg_path = '$OPENCODE_CONFIG'

with open(cfg_path) as f:
    raw = f.read()

# Try plain JSON first
try:
    cfg = json.loads(raw)
except json.JSONDecodeError:
    # Strip JSONC comments — but not :// inside strings
    # Remove single-line comments (// not preceded by :)
    raw = re.sub(r'(?<!:)//.*$', '', raw, flags=re.MULTILINE)
    # Remove multi-line comments
    raw = re.sub(r'/\*.*?\*/', '', raw, flags=re.DOTALL)
    cfg = json.loads(raw)

# Update or create provider
cfg['provider']['$PROVIDER_NAME'] = {
    'name': 'TurboQuant MLX',
    'api': 'openai',
    'options': {
        'baseURL': 'http://127.0.0.1:${PORT}/v1',
        'apiKey': 'mlx-turboquant'
    },
    'models': {
        'default': {
            'id': '$MODEL_ID'
        }
    }
}

# Set default model
cfg['model'] = '$PROVIDER_NAME/default'

# Write back as clean JSON (no comments)
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2)
    f.write('\n')
" 2>&1 || { echo "FAILED"; exit 1; }

echo "OK"
echo "  Provider: $PROVIDER_NAME"
echo "  Model:    $MODEL_ID"
echo ""

# ── Fix dcp.jsonc if it has unknown keys ──────────────────
if [[ -f "$DCP_CONFIG" ]]; then
    python3 -c "
import json, re

dcp_path = '$DCP_CONFIG'
with open(dcp_path) as f:
    raw = f.read()

# Strip comments
raw = re.sub(r'//.*$', '', raw, flags=re.MULTILINE)
raw = re.sub(r'/\*.*?\*/', '', raw, flags=re.DOTALL)

try:
    cfg = json.loads(raw)
    # Remove unknown keys that cause warnings
    known_compress_keys = {'minContextLimit', 'maxContextPercent', 'protectUserMessages', 'enabled'}
    if 'compress' in cfg:
        cfg['compress'] = {k: v for k, v in cfg['compress'].items() if k in known_compress_keys}
    # Remove top-level unknown keys
    known_top_keys = {'compress', 'rules'}
    cfg = {k: v for k, v in cfg.items() if k in known_top_keys}

    with open(dcp_path, 'w') as f:
        json.dump(cfg, f, indent=2)
        f.write('\n')
except Exception as e:
    pass
" 2>/dev/null || true
fi

# ── Launch opencode ───────────────────────────────────────
echo "  Launching opencode..."
echo ""
opencode

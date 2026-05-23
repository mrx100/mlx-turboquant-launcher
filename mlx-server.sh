#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin"

VMLX_DIR="$HOME/workspace/mlx-studio"
VMLX_VENV="$VMLX_DIR/.venv"
PYTHON="$VMLX_VENV/bin/python3"
VMLX_CMD="$PYTHON -m vmlx_engine.cli"
MODELS_DIR="$HOME/.lmstudio/models"
CONFIG_FILE="$HOME/.mlx_defaults"

DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT=8000
DEFAULT_CTX=8192
DEFAULT_MAX_TOKENS=2048
DEFAULT_TEMPERATURE=0.7
DEFAULT_KV_QUANT="q4"
DEFAULT_JIT="off"
DEFAULT_HOST="0.0.0.0"

CONTEXT_OPTIONS=(2048 4096 8192 16384 32768 65536 131072)
KV_QUANT_OPTIONS=("none" "q4" "q8")

# ── helpers ──────────────────────────────────────────────
fmt_size() {
  local bytes=$1
  if (( bytes >= 1073741824 )); then
    echo "$($PYTHON -c "print(f'{ $bytes/1073741824:.1f}')") GB"
  elif (( bytes >= 1048576 )); then
    echo "$($PYTHON -c "print(f'{ $bytes/1048576:.0f}')") MB"
  else
    echo "${bytes} B"
  fi
}

read_config() {
  HOST="$DEFAULT_HOST"
  PORT="$DEFAULT_PORT"
  CONTEXT="$DEFAULT_CTX"
  MAX_TOKENS="$DEFAULT_MAX_TOKENS"
  TEMPERATURE="$DEFAULT_TEMPERATURE"
  KV_QUANT="$DEFAULT_KV_QUANT"
  JIT="$DEFAULT_JIT"
  MODEL_PATH=""
  MODEL_LABEL=""
  if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
  fi
}

save_config() {
  cat > "$CONFIG_FILE" <<EOF
# MLX server defaults – edit this file or re-run to change
HOST="$HOST"
PORT="$PORT"
CONTEXT="$CONTEXT"
MAX_TOKENS="$MAX_TOKENS"
TEMPERATURE="$TEMPERATURE"
KV_QUANT="$KV_QUANT"
JIT="$JIT"
MODEL_PATH="$MODEL_PATH"
MODEL_LABEL="$MODEL_LABEL"
EOF
  echo "✓ defaults saved to $CONFIG_FILE"
}

# ── scan models ──────────────────────────────────────────
declare -a MODEL_OPTIONS=()
declare -a MODEL_PATHS=()
declare -a MODEL_LABELS=()

scan_models() {
  MODEL_OPTIONS=()
  MODEL_PATHS=()
  MODEL_LABELS=()

  if ! [[ -d "$MODELS_DIR" ]]; then
    echo "✗ Model directory not found: $MODELS_DIR" >&2
    exit 1
  fi

  # Only MLX models (directories with .safetensors files) — deduplicate via temp file
  local tmp_seen
  tmp_seen=$(mktemp)
  while IFS= read -r -d '' safetensors_file; do
    local dir="$(dirname "$safetensors_file")"
    if grep -qxF "$dir" "$tmp_seen" 2>/dev/null; then
      continue
    fi
    echo "$dir" >> "$tmp_seen"

    local org="$(basename "$(dirname "$dir")")"
    local name="$(basename "$dir")"
    local label="${org}/${name}"

    local total_size=0
    while IFS= read -r -d '' f; do
      local sz
      sz=$(stat -f%z "$f" 2>/dev/null || echo 0)
      total_size=$((total_size + sz))
    done < <(find "$dir" -type f -print0 2>/dev/null)

    MODEL_OPTIONS+=("${label} ($(fmt_size $total_size))")
    MODEL_PATHS+=("$dir")
    MODEL_LABELS+=("$label")
  done < <(find "$MODELS_DIR" -mindepth 3 -maxdepth 4 -name '*.safetensors' -print0 2>/dev/null)
  rm -f "$tmp_seen"

  if [[ ${#MODEL_OPTIONS[@]} -eq 0 ]]; then
    echo "✗ No MLX models found in $MODELS_DIR" >&2
    exit 1
  fi
}

# ── menus ────────────────────────────────────────────────
pick_model() {
  echo
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║  Available MLX Models in ~/.lmstudio/models         ║"
  echo "╚══════════════════════════════════════════════════════╝"
  echo
  for i in "${!MODEL_OPTIONS[@]}"; do
    printf "  %2d) %s\n" "$((i+1))" "${MODEL_OPTIONS[$i]}"
  done
  echo
  read -r -p "Select model [1-${#MODEL_OPTIONS[@]}]: " sel
  sel=$((sel-1))
  if [[ $sel -lt 0 || $sel -ge ${#MODEL_OPTIONS[@]} ]]; then
    echo "Invalid selection" >&2; exit 1
  fi
  MODEL_PATH="${MODEL_PATHS[$sel]}"
  MODEL_LABEL="${MODEL_LABELS[$sel]}"
}

pick_port() {
  read -r -p "Server port [$PORT]: " sel
  if [[ -n "$sel" ]]; then
    PORT="$sel"
  fi
}

pick_context() {
  echo
  echo "Context size (default: $CONTEXT):"
  for i in "${!CONTEXT_OPTIONS[@]}"; do
    local m=""
    [[ ${CONTEXT_OPTIONS[$i]} -eq $CONTEXT ]] && m=" (current default)"
    echo "  $((i+1))) ${CONTEXT_OPTIONS[$i]}$m"
  done
  echo "  c) Custom value"
  echo
  read -r -p "Context size [$CONTEXT]: " sel
  if [[ -z "$sel" ]]; then
    :
  elif [[ "$sel" == "c" || "$sel" == "C" ]]; then
    read -r -p "Enter custom context size: " custom
    CONTEXT="$custom"
  elif [[ "$sel" =~ ^[0-9]+$ ]]; then
    local idx=$((sel-1))
    if [[ $idx -ge 0 && $idx -lt ${#CONTEXT_OPTIONS[@]} ]]; then
      CONTEXT="${CONTEXT_OPTIONS[$idx]}"
    else
      CONTEXT="$sel"
    fi
  else
    CONTEXT="$sel"
  fi
}

pick_max_tokens() {
  read -r -p "Max tokens per request [$MAX_TOKENS]: " sel
  if [[ -n "$sel" ]]; then
    MAX_TOKENS="$sel"
  fi
}

pick_temperature() {
  read -r -p "Default temperature [$TEMPERATURE]: " sel
  if [[ -n "$sel" ]]; then
    TEMPERATURE="$sel"
  fi
}

pick_kv_quant() {
  echo
  echo "KV-Cache quantization:"
  for i in "${!KV_QUANT_OPTIONS[@]}"; do
    local m=""
    [[ "${KV_QUANT_OPTIONS[$i]}" == "$KV_QUANT" ]] && m=" (current default)"
    echo "  $((i+1))) ${KV_QUANT_OPTIONS[$i]}$m"
  done
  echo
  read -r -p "KV-Cache quantization [$KV_QUANT]: " sel
  if [[ -z "$sel" ]]; then
    :
  else
    local idx=$((sel-1))
    KV_QUANT="${KV_QUANT_OPTIONS[$idx]}"
  fi
}

pick_jit() {
  read -r -p "Enable JIT compilation? (y/n) [${JIT}]: " sel
  if [[ "$sel" =~ ^[yY] ]]; then
    JIT="on"
  elif [[ "$sel" =~ ^[nN] ]]; then
    JIT="off"
  fi
}

# ── kill existing vmlx instances ────────────────────────
kill_existing() {
  local pids
  pids=$(pgrep -f "vmlx_engine.cli serve" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "Killing existing vmlx-engine instances: $(echo $pids | tr '\n' ' ')"
    kill $pids 2>/dev/null || true
    sleep 2
    pids=$(pgrep -f "vmlx_engine.cli serve" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

# ── copy AGENTS.md if in new/empty dir ──────────────────
setup_agents() {
  local agents_file="$HOME/workspace/system.txt"
  local target="$(pwd)/AGENTS.md"
  if [[ -f "$agents_file" && ! -f "$target" ]]; then
    cp "$agents_file" "$target"
    echo "✓ AGENTS.md copied from system.txt"
  fi
}

# ── main ─────────────────────────────────────────────────
main() {
  kill_existing
  setup_agents
  scan_models
  read_config

  echo
  echo "╔══════════════════════════════════════════╗"
  echo "║  MLX Server Launcher (vmlx-engine)       ║"
  echo "╚══════════════════════════════════════════╝"

  if [[ -n "$MODEL_PATH" && -d "$MODEL_PATH" ]]; then
    echo "Saved default: $MODEL_LABEL"
    read -r -p "Use this model? [Y/n]: " ok
    if [[ "$ok" =~ ^[nN] ]]; then
      MODEL_PATH=""
    fi
  fi

  if [[ -z "$MODEL_PATH" ]]; then
    pick_model
  fi

  pick_port
  pick_context
  pick_max_tokens
  pick_temperature
  pick_kv_quant
  pick_jit

  echo
  echo "╔══════════════════════════════════════════╗"
  echo "║  Launch configuration                    ║"
  echo "╠══════════════════════════════════════════╣"
  printf "║ Model:    %-30s ║\n" "$MODEL_LABEL"
  printf "║ Host:     %-30s ║\n" "$HOST"
  printf "║ Port:     %-30s ║\n" "$PORT"
  printf "║ Context:  %-30s ║\n" "$CONTEXT"
  printf "║ Max Tok:  %-30s ║\n" "$MAX_TOKENS"
  printf "║ Temp:     %-30s ║\n" "$TEMPERATURE"
  printf "║ KV Quant: %-30s ║\n" "$KV_QUANT"
  printf "║ JIT:      %-30s ║\n" "$JIT"
  echo "╚══════════════════════════════════════════╝"

  if [[ -f "$CONFIG_FILE" ]]; then
    read -r -p "Save as defaults? [Y/n]: " save
    if [[ ! "$save" =~ ^[nN] ]]; then
      save_config
    fi
  else
    save_config
  fi

  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "✗ Model not found: $MODEL_PATH" >&2
    exit 1
  fi

  echo
  echo "Starting vmlx-engine server on $HOST:$PORT ..."
  echo "Model: $MODEL_LABEL"
  echo

  cd "$VMLX_DIR"

  SERVE_ARGS=(
    "$PYTHON" -m vmlx_engine.cli serve "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --max-tokens "$MAX_TOKENS"
    --default-temperature "$TEMPERATURE"
  )
  if [[ "$KV_QUANT" != "none" ]]; then
    SERVE_ARGS+=(--kv-cache-quantization "$KV_QUANT")
  fi
  if [[ "$JIT" == "on" ]]; then
    SERVE_ARGS+=(--enable-jit)
  fi

  set +u
  "${SERVE_ARGS[@]}" "$@" &
  set -u
  PID=$!
  echo "vmlx-engine running (PID $PID)"
  echo "OpenAI endpoint: http://${HOST}:${PORT}/v1"
  wait "$PID"
}

main "$@"

#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-markquant}"
HOST="${MLFLOW_HOST:-0.0.0.0}"
PORT="${MLFLOW_PORT:-5058}"
BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow.db}"
DEFAULT_ARTIFACT_ROOT="${MLFLOW_DEFAULT_ARTIFACT_ROOT:-./mlartifacts}"

usage() {
  cat <<EOF
Usage:
  bash scripts/start_mlflow_service.sh [options]

Options:
  --env NAME                  Conda environment. Default: ${CONDA_ENV}
  --host HOST                 MLflow host. Default: ${HOST}
  --port PORT                 MLflow port. Default: ${PORT}
  --backend-store-uri URI     Backend store URI. Default: ${BACKEND_STORE_URI}
  --default-artifact-root URI Default artifact root. Default: ${DEFAULT_ARTIFACT_ROOT}
  -h, --help                  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --backend-store-uri)
      BACKEND_STORE_URI="$2"
      shift 2
      ;;
    --default-artifact-root)
      DEFAULT_ARTIFACT_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

log() {
  printf '==> %s\n' "$*"
}

port_listener_pids() {
  lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | sort -u
}

process_command() {
  ps -p "$1" -o command= 2>/dev/null || true
}

process_parent_pid() {
  ps -p "$1" -o ppid= 2>/dev/null | awk '{print $1}' || true
}

is_mlflow_pid() {
  local pid="$1"
  local command
  local depth=0

  while [[ -n "$pid" && "$pid" != "0" && "$depth" -lt 6 ]]; do
    command="$(process_command "$pid")"
    if [[ "$command" == *mlflow* ]]; then
      return 0
    fi
    pid="$(process_parent_pid "$pid")"
    depth=$((depth + 1))
  done

  return 1
}

stop_existing_mlflow_on_port() {
  local pids=()
  local pid
  local command
  local non_mlflow=()

  while IFS= read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done < <(port_listener_pids)

  if [[ ${#pids[@]} -eq 0 ]]; then
    return
  fi

  for pid in "${pids[@]}"; do
    command="$(process_command "$pid")"
    if is_mlflow_pid "$pid"; then
      log "Restart existing MLflow service on port ${PORT}: pid ${pid}"
      kill "$pid" 2>/dev/null || true
    else
      non_mlflow+=("${pid}: ${command}")
    fi
  done

  if [[ ${#non_mlflow[@]} -gt 0 ]]; then
    echo "Port ${PORT} is already in use by a non-MLflow process:" >&2
    printf '  %s\n' "${non_mlflow[@]}" >&2
    echo "Use --port to choose another port, or stop that process manually." >&2
    exit 1
  fi

  for _ in {1..20}; do
    if [[ -z "$(port_listener_pids)" ]]; then
      return
    fi
    sleep 0.5
  done

  while IFS= read -r pid; do
    [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null || true
  done < <(port_listener_pids)
}

find_conda_base() {
  if command -v conda >/dev/null 2>&1; then
    conda info --base
    return
  fi

  for candidate in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "/opt/miniforge3" "/opt/miniconda3"; do
    if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  return 1
}

CONDA_BASE="$(find_conda_base)" || {
  echo "Could not find conda. Please install conda or add it to PATH." >&2
  exit 1
}

# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

log "Activate conda environment: ${CONDA_ENV}"
conda activate "$CONDA_ENV"

command -v mlflow >/dev/null 2>&1 || {
  echo "mlflow is not installed in conda environment '${CONDA_ENV}'." >&2
  exit 1
}

if [[ "$BACKEND_STORE_URI" == sqlite:///* ]]; then
  DB_PATH="${BACKEND_STORE_URI#sqlite:///}"
  DB_DIR="$(dirname "$DB_PATH")"
  if [[ "$DB_DIR" != "." ]]; then
    mkdir -p "$DB_DIR"
  fi
fi

mkdir -p "$DEFAULT_ARTIFACT_ROOT"

stop_existing_mlflow_on_port

log "Start MLflow service"
mlflow server \
  --host "$HOST" \
  --port "$PORT" \
  --backend-store-uri "$BACKEND_STORE_URI" \
  --default-artifact-root "$DEFAULT_ARTIFACT_ROOT" &

MLFLOW_PID="$!"
CLEANED_UP=0

cleanup() {
  trap - INT TERM EXIT
  if [[ "$CLEANED_UP" -eq 1 ]]; then
    return
  fi
  CLEANED_UP=1

  if kill -0 "$MLFLOW_PID" >/dev/null 2>&1; then
    log "Stop MLflow service: pid ${MLFLOW_PID}"
    kill "$MLFLOW_PID"
  fi
}
trap cleanup INT TERM EXIT

HEALTH_URL="http://127.0.0.1:${PORT}"
for _ in {1..30}; do
  if ! kill -0 "$MLFLOW_PID" >/dev/null 2>&1; then
    echo "MLflow service failed to start." >&2
    wait "$MLFLOW_PID" || true
    exit 1
  fi

  if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    break
  fi

  sleep 1
done

if ! curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
  echo "MLflow service did not become reachable at ${HEALTH_URL} within 30 seconds." >&2
  exit 1
fi

log "MLflow service is running: pid ${MLFLOW_PID}"
printf 'Local URL:   http://127.0.0.1:%s\n' "$PORT"

LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
if [[ -z "$LAN_IP" ]]; then
  LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
fi
if [[ -n "$LAN_IP" ]]; then
  printf 'Network URL: http://%s:%s\n' "$LAN_IP" "$PORT"
else
  printf 'Network URL: http://<server-ip>:%s\n' "$PORT"
fi

wait "$MLFLOW_PID"

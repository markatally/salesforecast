#!/usr/bin/env bash
set -euo pipefail

STREAMFILE="${1:-code/30d-jenny/streamlit/month_end_cumsum_break_dashboard.py}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ ! -f "${STREAMFILE}" ]]; then
  echo "Streamlit file not found: ${STREAMFILE}" >&2
  exit 1
fi

streamlit run "${STREAMFILE}"

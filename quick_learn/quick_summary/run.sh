#!/usr/bin/env bash
set -euo pipefail
SUMMARY_ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -f "$SUMMARY_ROOT/config.local.json" ]]; then
  echo "请先运行：python3 $SUMMARY_ROOT/scripts/setup_omnivoice.py" >&2
  exit 1
fi
SUMMARY_PYTHON="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$SUMMARY_ROOT/config.local.json")"
exec "$SUMMARY_PYTHON" "$SUMMARY_ROOT/scripts/build.py" "$@"

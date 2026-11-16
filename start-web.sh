#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/tools/linux/common.sh"
exec "$FLOWHMR_PYTHON" app.py --host 127.0.0.1 --port "${FLOWHMR_PORT:-8080}" --device cuda:0 --sam-chunk-default 4 "$@"

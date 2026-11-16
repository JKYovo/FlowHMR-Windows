#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/tools/linux/common.sh"
exec "$FLOWHMR_PYTHON" tools/windows/check_install.py "$@"

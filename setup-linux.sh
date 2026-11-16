#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/tools/linux/common.sh"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
    echo 'This setup supports Linux x86_64 with an NVIDIA CUDA GPU.'; exit 1;
}
command -v uv >/dev/null || { echo 'Please install uv first: https://docs.astral.sh/uv/'; exit 1; }
git submodule update --init --depth 1 third_party/YOLOX third_party/sam-3d-body third_party/vggt-omega
if [[ ! -x "$FLOWHMR_PYTHON" ]]; then
    uv python install 3.10
    uv venv --managed-python --python 3.10 --seed .venv
fi
# Both package indices below are official; upstream requirements select cu118 explicitly.
uv pip install --python "$FLOWHMR_PYTHON" --index-strategy unsafe-best-match -r requirements-linux.lock.txt
uv pip install --python "$FLOWHMR_PYTHON" --no-build-isolation --no-deps -e third_party/YOLOX
"$FLOWHMR_PYTHON" tools/windows/adapt_sam.py
uv pip install --python "$FLOWHMR_PYTHON" --no-build-isolation --no-deps ./third_party/KimodoMotionCorrection ./third_party/RGBMocapCorrection
# The lock file includes the upstream Linux CPython 3.10 FBX SDK wheel and SHA-256.
"$FLOWHMR_PYTHON" tools/windows/check_install.py --environment-only
echo 'Environment ready. Download models with ./download-models.sh, then launch ./open-flowhmr.sh.'

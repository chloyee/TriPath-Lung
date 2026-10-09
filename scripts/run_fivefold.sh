#!/usr/bin/env bash
set -euo pipefail
VLR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VLR_ROOT"
exec "${VLR_PYTHON:-$VLR_ROOT/.venv/bin/python}" -u -m vlr_mil.train --config configs/main.json "$@"

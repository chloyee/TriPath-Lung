#!/usr/bin/env bash
set -euo pipefail
TRIPATH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TRIPATH_ROOT"
exec "${TRIPATH_PYTHON:-$TRIPATH_ROOT/.venv/bin/python}" -u -m vlr_mil.evaluate "$@"

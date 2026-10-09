#!/bin/bash
# Build the eldercare-monitor app package with the recamera_pro packager.
#   RECAMERA_PRO_REPO  recamera_pro checkout (kit + market/packaging)
#   PAYLOAD_ROOTS      colon-separated dirs holding the declared models/ and wheels/ (see MODELS.md)
#   --test             run the app tests first
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
REPO=${RECAMERA_PRO_REPO:-$HOME/project/recamera/recamera_pro}
CACHE=$HOME/.cache/recamera-build/asr-eval/optimization-20261009/t172/package-inputs
ROOTS=${PAYLOAD_ROOTS:-$CACHE/baseline-assets:$CACHE/new-model-canonical}
OUT=${OUT:-$HERE/dist}
[ -f "$REPO/market/packaging/build.py" ] || { echo "recamera_pro not found at $REPO (set RECAMERA_PRO_REPO)" >&2; exit 1; }
if [ "${1:-}" = "--test" ]; then
  (cd "$REPO" && RECAMERA_PRO_REPO="$REPO" uv run --frozen python -m pytest -q -p no:cacheprovider "$HERE/app")
fi
ARGS=()
IFS=: read -ra DIRS <<< "$ROOTS"
for d in "${DIRS[@]}"; do [ -d "$d" ] || { echo "payload root missing: $d" >&2; exit 1; }; ARGS+=(--payload-root "$d"); done
mkdir -p "$OUT"
python3 "$REPO/market/packaging/build.py" "$HERE/app" --out "$OUT" "${ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${LIBCLANG_PATH:-}" ]]; then
  if command -v llvm-config >/dev/null 2>&1; then
    export LIBCLANG_PATH="$(llvm-config --libdir)"
  fi
fi

exec "$@"

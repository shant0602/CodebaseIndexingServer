#!/usr/bin/env bash
set -euo pipefail

# Change to app directory to ensure Python can find modules
cd /app || exit 1

# Set up libclang path if not already set
if [[ -z "${LIBCLANG_PATH:-}" ]]; then
  # Try to find libclang in common locations
  LIBCLANG_FILE=""
  
  # First, try to find LLVM 18 specifically (for compatibility with Python clang bindings)
  if command -v llvm-config-18 >/dev/null 2>&1; then
    LIBDIR="$(llvm-config-18 --libdir)"
    # Look for libclang.so (C API) - exclude libclang-cpp.so (C++ API)
    found=$(find "${LIBDIR}" -maxdepth 1 \( -name "libclang.so*" -a ! -name "libclang-cpp.so*" \) \( -type f -o -type l \) 2>/dev/null | head -1)
    if [[ -n "${found}" ]]; then
      # Resolve symlink to actual file
      if [[ -L "${found}" ]]; then
        LIBCLANG_FILE="$(readlink -f "${found}")"
      else
        LIBCLANG_FILE="${found}"
      fi
    fi
  # Fallback to default llvm-config
  elif command -v llvm-config >/dev/null 2>&1; then
    LIBDIR="$(llvm-config --libdir)"
    # Look for libclang.so (C API) - exclude libclang-cpp.so (C++ API)
    found=$(find "${LIBDIR}" -maxdepth 1 \( -name "libclang.so*" -a ! -name "libclang-cpp.so*" \) \( -type f -o -type l \) 2>/dev/null | head -1)
    if [[ -n "${found}" ]]; then
      # Resolve symlink to actual file
      if [[ -L "${found}" ]]; then
        LIBCLANG_FILE="$(readlink -f "${found}")"
      else
        LIBCLANG_FILE="${found}"
      fi
    fi
  fi
  
  # If not found, search common system locations (prefer version 18)
  if [[ -z "${LIBCLANG_FILE}" ]]; then
    # Search in versioned llvm directories, prefer 18
    # Look for libclang.so (C API) - exclude libclang-cpp.so (C++ API)
    for llvm_dir in /usr/lib/llvm-18/lib /usr/lib/llvm-*/lib; do
      found=$(find ${llvm_dir} \( -name "libclang.so*" -a ! -name "libclang-cpp.so*" \) \( -type f -o -type l \) 2>/dev/null | head -1)
      if [[ -n "${found}" ]]; then
        if [[ -L "${found}" ]]; then
          LIBCLANG_FILE="$(readlink -f "${found}")"
        else
          LIBCLANG_FILE="${found}"
        fi
        break
      fi
    done
  fi
  
  # Try standard library paths
  if [[ -z "${LIBCLANG_FILE}" ]]; then
    for search_path in "/usr/lib/x86_64-linux-gnu" "/usr/lib"; do
      # Look for libclang.so (C API) - exclude libclang-cpp.so (C++ API)
      found=$(find "${search_path}" -maxdepth 1 \( -name "libclang.so*" -a ! -name "libclang-cpp.so*" \) \( -type f -o -type l \) 2>/dev/null | head -1)
      if [[ -n "${found}" ]]; then
        if [[ -L "${found}" ]]; then
          LIBCLANG_FILE="$(readlink -f "${found}")"
        else
          LIBCLANG_FILE="${found}"
        fi
        break
      fi
    done
  fi
  
  # Set LIBCLANG_PATH to the actual library file path (preferred) or directory
  if [[ -n "${LIBCLANG_FILE}" ]]; then
    export LIBCLANG_PATH="${LIBCLANG_FILE}"
  elif command -v llvm-config-18 >/dev/null 2>&1; then
    # Fallback: use llvm-config-18 libdir
    export LIBCLANG_PATH="$(llvm-config-18 --libdir)"
  elif command -v llvm-config-17 >/dev/null 2>&1; then
    # Fallback: use llvm-config-17 libdir
    export LIBCLANG_PATH="$(llvm-config-17 --libdir)"
  elif command -v llvm-config-16 >/dev/null 2>&1; then
    # Fallback: use llvm-config-16 libdir
    export LIBCLANG_PATH="$(llvm-config-16 --libdir)"
  elif command -v llvm-config >/dev/null 2>&1; then
    # Fallback: use llvm-config libdir
    export LIBCLANG_PATH="$(llvm-config --libdir)"
  fi
fi

exec "$@"

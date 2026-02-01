#!/usr/bin/env bash
set -euo pipefail

# Download python-build-standalone for a target platform
# Usage: ./download-python.sh [--platform mac-arm64|mac-x64|win-x64|linux-x64]
# Without --platform, auto-detects the current machine.

PYTHON_VERSION="3.12.7"
PBS_TAG="20241008"

PLATFORM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) PLATFORM="$2"; shift 2 ;;
    --all) PLATFORM="all"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Auto-detect if not specified
if [[ -z "$PLATFORM" ]]; then
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)  PLATFORM="mac-arm64" ;;
    Darwin-x86_64) PLATFORM="mac-x64" ;;
    Linux-x86_64)  PLATFORM="linux-x64" ;;
    *)             echo "Unsupported: $(uname -s)-$(uname -m)"; exit 1 ;;
  esac
fi

download_python() {
  local plat="$1"
  local triple dest_dir

  case "$plat" in
    mac-arm64)  triple="aarch64-apple-darwin" ;;
    mac-x64)    triple="x86_64-apple-darwin" ;;
    win-x64)    triple="x86_64-pc-windows-msvc-shared" ;;
    linux-x64)  triple="x86_64-unknown-linux-gnu" ;;
    *)          echo "Unknown platform: $plat"; return 1 ;;
  esac

  dest_dir="vendor/python-${plat}"

  local filename="cpython-${PYTHON_VERSION}+${PBS_TAG}-${triple}-install_only_stripped.tar.gz"
  local url="https://github.com/indygreg/python-build-standalone/releases/download/${PBS_TAG}/${filename}"

  echo "=== Downloading Python ${PYTHON_VERSION} for ${plat} (${triple}) ==="
  echo "URL: ${url}"

  mkdir -p "${dest_dir}"
  local tmpfile
  tmpfile=$(mktemp)

  curl -L --progress-bar -o "${tmpfile}" "${url}"

  echo "Extracting to ${dest_dir}..."
  tar xzf "${tmpfile}" -C "${dest_dir}" --strip-components=1

  rm -f "${tmpfile}"
  echo "Done: ${dest_dir}"
}

if [[ "$PLATFORM" == "all" ]]; then
  for p in mac-arm64 mac-x64 win-x64 linux-x64; do
    download_python "$p"
  done
else
  download_python "$PLATFORM"
fi

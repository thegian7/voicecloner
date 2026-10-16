#!/usr/bin/env bash
# Convert recordings to 16 kHz mono FLAC and drop them into a voice's training dir.
#
# usage: training/ingest.sh DEST SOURCE FILE_OR_DIR...
#   DEST    the voice's training dir, local or host:path (e.g. gpubox:voicecloner-train/alice)
#   SOURCE  a label for where the audio came from (streams, calls, screencasts, ...).
#           Reports and held-out scores are split by it, so keep one label per kind.
#   FILE_OR_DIR  audio/video files, or dirs searched recursively for them.
#
# Takes anything ffmpeg reads (mp4, mov, mkv, webm, m4a, mp3, wav, flac). A `mic.pcm` is a
# sidebar call's mic-only track (raw s16le, 16 kHz, mono) and is named after its call dir.
# Recordings with other people talking are fine: prep_segments.py scores every clip against
# the reference and build_manifest.py drops the ones that aren't the speaker. Isolated mic
# tracks still give far more usable minutes than a mixed recording.
#
# Runs on the machine that has the recordings and ffmpeg; the training box doesn't need ffmpeg.
# Existing files at DEST are never overwritten, so re-running only adds new recordings.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
dest=$1 source=$2
shift 2
command -v ffmpeg >/dev/null || { echo "ffmpeg not found" >&2; exit 1; }

stage=$(mktemp -d "${TMPDIR:-/tmp}/vc-ingest.XXXXXX")
cleanup() { rm -f "$stage"/*.flac; rmdir "$stage" 2>/dev/null || true; }
trap cleanup EXIT

convert() {
  local in=$1 name out
  if [[ $(basename "$in") == mic.pcm ]]; then
    name=$(basename "$(dirname "$in")")
    out="$stage/$name.flac"
    ffmpeg -nostdin -v error -y -f s16le -ar 16000 -ac 1 -i "$in" -c:a flac "$out"
  else
    name=$(basename "${in%.*}" | tr ' ' '_')
    out="$stage/$name.flac"
    ffmpeg -nostdin -v error -y -i "$in" -vn -ac 1 -ar 16000 -c:a flac "$out"
  fi
  echo "  $in -> $name.flac"
}

n=0
for arg in "$@"; do
  if [[ -d $arg ]]; then
    while IFS= read -r -d '' f; do convert "$f"; n=$((n + 1)); done < <(
      find "$arg" -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.webm' \
        -o -iname '*.m4a' -o -iname '*.mp3' -o -iname '*.wav' -o -iname '*.flac' -o -name 'mic.pcm' \) -print0 | sort -z)
  else
    convert "$arg"; n=$((n + 1))
  fi
done
[[ $n -gt 0 ]] || { echo "no recordings found" >&2; exit 1; }

if [[ $dest == *:* ]]; then
  host=${dest%%:*} path=${dest#*:}
  path=${path#\~/}
  ssh "$host" "mkdir -p '$path/data/raw/$source'"
  rsync -a --ignore-existing "$stage"/ "$host:$path/data/raw/$source/"
else
  mkdir -p "$dest/data/raw/$source"
  rsync -a --ignore-existing "$stage"/ "$dest/data/raw/$source/"
fi
echo "$n recording(s) -> $dest/data/raw/$source/"

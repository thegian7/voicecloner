#!/usr/bin/env bash
# Train (or continue) a speaker LoRA with VoxCPM's trainer, inside hard memory limits.
#
# usage: training/train.sh ROOT RUN [STEPS]
#   ROOT   the voice's training dir (data/train.jsonl + val.jsonl from build_manifest.py)
#   RUN    run name; checkpoints land in ROOT/runs/RUN/ckpt/step_NNNNNNN
#   STEPS  total steps, default 1000. Rerunning an existing RUN with a higher STEPS
#          continues it: the trainer resumes from ckpt/latest.
#
# env:
#   VC_PYTHON      python with voxcpm installed  (~/voicecloner/.venv/bin/python)
#   VOXCPM_DIR     VoxCPM checkout holding scripts/train_voxcpm_finetune.py  (~/VoxCPM)
#   VC_BASE_MODEL  local VoxCPM2 snapshot dir; default: resolved via the HF cache (downloads if missing)
#   VC_GPU_GB      CUDA allocator cap, GB  (12)
#   VC_MEM_MAX     cgroup memory ceiling for the whole job  (24G)
#   HF_HOME        HF cache  (~/voicecloner-train/hf)
#
# Runs in the foreground and logs to ROOT/runs/RUN/train.out; start it under nohup or tmux.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$1" && pwd)
run=$2
steps=${3:-}
run_dir="$root/runs/$run"
cfg="$run_dir/run.yaml"

py=${VC_PYTHON:-$HOME/voicecloner/.venv/bin/python}
export VOXCPM_DIR=${VOXCPM_DIR:-$HOME/VoxCPM}
export HF_HOME=${HF_HOME:-$HOME/voicecloner-train/hf}
gpu_gb=${VC_GPU_GB:-12}
mem_max=${VC_MEM_MAX:-24G}

[[ -x $py ]] || { echo "no python at $py (set VC_PYTHON)" >&2; exit 1; }
[[ -f $VOXCPM_DIR/scripts/train_voxcpm_finetune.py ]] || {
  echo "no VoxCPM trainer under $VOXCPM_DIR (git clone https://github.com/OpenBMB/VoxCPM, or set VOXCPM_DIR)" >&2; exit 1; }
[[ -f $root/data/train.jsonl ]] || { echo "no $root/data/train.jsonl; run build_manifest.py build first" >&2; exit 1; }

mkdir -p "$run_dir"
if [[ ! -f $cfg ]]; then
  base=${VC_BASE_MODEL:-$("$py" -c "from huggingface_hub import snapshot_download; print(snapshot_download('openbmb/VoxCPM2'))")}
  sed -e "s|@BASE_MODEL@|${base%/}/|" -e "s|@ROOT@|$root|g" -e "s|@RUN_DIR@|$run_dir|g" \
      -e "s|@STEPS@|${steps:-1000}|g" "$here/run.template.yaml" > "$cfg"
  echo "wrote $cfg"
elif [[ -n $steps ]]; then
  done_step=$(cat "$run_dir/ckpt/latest/training_state.json" 2>/dev/null | "$py" -c 'import json,sys; print(json.load(sys.stdin).get("step",""))' 2>/dev/null || true)
  sed -i.bak -e "s|^num_iters: .*|num_iters: $steps|" -e "s|^max_steps: .*|max_steps: $steps|" "$cfg"
  echo "continuing $run to $steps steps${done_step:+ (latest checkpoint: step $done_step)}"
fi

cmd=(nice -n 10 env "HF_HOME=$HF_HOME" "VOXCPM_DIR=$VOXCPM_DIR" "VC_GPU_GB=$gpu_gb"
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     "$py" "$here/train_capped.py" --config_path "$cfg")
if command -v systemd-run >/dev/null && systemctl --user show-environment >/dev/null 2>&1; then
  # A cgroup ceiling on top of the CUDA cap: if anything balloons, the kernel stops this
  # job instead of squeezing the host's other services. No swap, so it fails fast.
  cmd=(systemd-run --user --scope --quiet -p "MemoryMax=$mem_max" -p MemorySwapMax=0 "${cmd[@]}")
fi

echo "training $run: GPU cap ${gpu_gb} GB, memory ceiling $mem_max, log $run_dir/train.out"
set +e
"${cmd[@]}" >> "$run_dir/train.out" 2>&1
status=$?
echo "EXIT=$status" >> "$run_dir/train.out"
tail -3 "$run_dir/train.out"
exit $status

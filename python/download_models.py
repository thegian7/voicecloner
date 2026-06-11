"""Download VoxCPM2 model with JSON progress output for Electron UI."""

import json
import sys
from huggingface_hub import snapshot_download

# VoxCPM2 collapses Qwen3's clone + design into a single model.
MODELS = {
    "voxcpm2": "openbmb/VoxCPM2",
}


def progress(event: dict):
    print(json.dumps(event), flush=True)


def download_model(name: str, repo_id: str):
    progress({"type": "start", "model": name, "repo": repo_id})
    try:
        path = snapshot_download(repo_id)
        progress({"type": "done", "model": name, "path": path})
    except Exception as e:
        progress({"type": "error", "model": name, "error": str(e)})
        return False
    return True


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    ok = True
    for name in models:
        if name not in MODELS:
            progress({"type": "error", "model": name, "error": f"Unknown model: {name}"})
            ok = False
            continue
        if not download_model(name, MODELS[name]):
            ok = False
    progress({"type": "complete", "success": ok})
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

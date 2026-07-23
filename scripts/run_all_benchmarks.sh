#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

UV_PROJECT_ENVIRONMENT="$repo_dir/.venv-tts" scripts/bootstrap.sh --tts
UV_PROJECT_ENVIRONMENT="$repo_dir/.venv-vlm" scripts/bootstrap.sh --vlm
"$repo_dir/.venv-tts/bin/python" -m xrbench all --config configs/default.yaml "$@"

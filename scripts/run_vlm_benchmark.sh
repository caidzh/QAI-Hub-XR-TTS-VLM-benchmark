#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
uv run python -m xrbench vlm all --config configs/vlm_smolvlm_256m.yaml "$@"

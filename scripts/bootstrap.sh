#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
project_environment="${UV_PROJECT_ENVIRONMENT:-$repo_dir/.venv}"
if [[ "$project_environment" != /* ]]; then
  project_environment="$repo_dir/$project_environment"
fi

extras=(dev)
install_piper=0
for arg in "$@"; do
  case "$arg" in
    --tts)
      extras+=(tts)
      install_piper=1
      ;;
    --vlm)
      extras+=(vlm)
      ;;
    --all)
      echo "The --all environment is unsupported: official PiperTTS and SmolVLM require" >&2
      echo "conflicting Transformers versions. Create separate --tts and --vlm environments." >&2
      exit 2
      ;;
    *)
      echo "Usage: $0 [--tts|--vlm|--all]" >&2
      exit 2
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

sync_args=(uv sync)
for extra in "${extras[@]}"; do
  sync_args+=(--extra "$extra")
done
"${sync_args[@]}"

if [[ "$install_piper" == "1" ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to fetch the pinned Piper source archive." >&2
    exit 1
  fi
  piper_commit="73c04d81d5590ecc46e522de3601ce7fb29fc2be"
  piper_temp="$(mktemp -d)"
  trap 'rm -rf "$piper_temp"' EXIT
  curl -LsSf \
    "https://github.com/rhasspy/piper/archive/$piper_commit.tar.gz" \
    --output "$piper_temp/piper.tar.gz"
  tar -xzf "$piper_temp/piper.tar.gz" -C "$piper_temp"
  uv pip install \
    --python "$project_environment/bin/python" \
    "$piper_temp/piper-$piper_commit/src/python" \
    --no-deps \
    --no-build-isolation
fi

echo "Environment ready at $project_environment"

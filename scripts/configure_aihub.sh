#!/usr/bin/env bash
set -euo pipefail

if ! command -v qai-hub >/dev/null 2>&1; then
  echo "qai-hub is not installed. Run scripts/bootstrap.sh first." >&2
  exit 1
fi

read -r -s -p "Qualcomm AI Hub API token (input hidden): " qaihub_token
echo
if [[ -z "$qaihub_token" ]]; then
  echo "No token supplied." >&2
  exit 2
fi
qai-hub configure --api_token "$qaihub_token"
unset qaihub_token
qai-hub list-devices

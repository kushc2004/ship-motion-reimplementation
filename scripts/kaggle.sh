#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KAGGLE_CONFIG_DIR="${SHIPMOTION_KAGGLE_CONFIG_DIR:-$project_root/.kaggle}"

exec kaggle "$@"

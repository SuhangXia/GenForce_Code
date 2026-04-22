#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/run_uniforce_same_scale_rect_safe_499k.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Missing base script: ${BASE_SCRIPT}" >&2
  exit 1
fi

# 154 episodes / indenter gives 997,920 images, which is the closest value to
# 1,000,000 under the fixed 18 indenters x 18 frames x 20 markers setup.
export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/suhang/datasets/uniforce_same_scale_rect_safe_1m_short16_all18}"
export EPISODES_PER_INDENTER="${EPISODES_PER_INDENTER:-154}"

exec bash "${BASE_SCRIPT}"

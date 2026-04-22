#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GENERATOR="${REPO_ROOT}/generate_uniforce_same_scale_rect_safe_depth.py"
MARKER_SOURCE_DIR="${REPO_ROOT}/sim/marker/marker_pattern/4_3"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/suhang/datasets/uniforce_same_scale_rect_safe_499k_short16_all18}"
SHORT_EDGE_MM="${SHORT_EDGE_MM:-16.0}"
EPISODES_PER_INDENTER="${EPISODES_PER_INDENTER:-70}"
DEPTH_MIN_MM="${DEPTH_MIN_MM:-0.5}"
REQUESTED_DEPTH_MAX_MM="${REQUESTED_DEPTH_MAX_MM:-2.0}"
SEED="${SEED:-42}"
RENDER_DEVICE="${RENDER_DEVICE:-cpu}"
RENDER_SAMPLES=16
MAX_PHYSICS_WORKERS="${MAX_PHYSICS_WORKERS:-3}"
MAX_MESHING_WORKERS="${MAX_MESHING_WORKERS:-4}"
MAX_RENDER_WORKERS="${MAX_RENDER_WORKERS:-12}"
AUTO_BALANCE_PIPELINE="${AUTO_BALANCE_PIPELINE:-1}"
RENDER_BACKLOG_HIGH_WATERMARK="${RENDER_BACKLOG_HIGH_WATERMARK:-0}"
RENDER_BACKLOG_LOW_WATERMARK="${RENDER_BACKLOG_LOW_WATERMARK:-0}"
PHYSICS_NPZ_CLEANUP="${PHYSICS_NPZ_CLEANUP:-delete_after_scale_complete}"
EST_IMAGES_PER_HOUR="${EST_IMAGES_PER_HOUR:-6000}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-60}"

OBJECTS=(
  cone
  cylinder
  cylinder_sh
  cylinder_si
  dotin
  dots
  hemisphere
  hexagon
  line
  moon
  pacman
  prism
  random
  sphere
  sphere_s
  torus
  triangle
  wave
)

INDENTER_COUNT="${#OBJECTS[@]}"
FRAMES_PER_EPISODE=18
if [[ ! -d "${MARKER_SOURCE_DIR}" ]]; then
  echo "Missing marker source dir: ${MARKER_SOURCE_DIR}" >&2
  exit 1
fi
MARKERS_PER_FRAME="$(python3 - <<'PY' "${MARKER_SOURCE_DIR}"
from pathlib import Path
import sys
root = Path(sys.argv[1])
exts = {".jpg", ".jpeg", ".png"}
count = sum(1 for path in root.iterdir() if path.is_file() and path.suffix.lower() in exts)
print(count)
PY
)"
if [[ "${MARKERS_PER_FRAME}" -le 0 ]]; then
  echo "No marker textures found in ${MARKER_SOURCE_DIR}" >&2
  exit 1
fi
TASKS_PER_EPISODE=$((FRAMES_PER_EPISODE + 2))
TASKS_PER_INDENTER=$((EPISODES_PER_INDENTER * TASKS_PER_EPISODE))
RENDER_TASKS_PER_INDENTER=$((EPISODES_PER_INDENTER * FRAMES_PER_EPISODE))
IMAGES_PER_INDENTER=$((RENDER_TASKS_PER_INDENTER * MARKERS_PER_FRAME))
TOTAL_EVENTS=$((INDENTER_COUNT * RENDER_TASKS_PER_INDENTER))
TOTAL_IMAGES=$((TOTAL_EVENTS * MARKERS_PER_FRAME))
TOTAL_TASKS=$((INDENTER_COUNT * TASKS_PER_INDENTER))

if [[ -n "${EST_IMAGES_PER_HOUR}" ]] && [[ "${EST_IMAGES_PER_HOUR}" != "0" ]]; then
  ETA_HOURS="$(python3 - <<'PY' "${TOTAL_IMAGES}" "${EST_IMAGES_PER_HOUR}"
import sys
total = float(sys.argv[1])
rate = float(sys.argv[2])
hours = total / rate
print(f"{hours:.2f}")
PY
)"
else
  ETA_HOURS="unknown"
fi

echo "Repo root             : ${REPO_ROOT}"
echo "Generator             : ${GENERATOR}"
echo "Output root           : ${OUTPUT_ROOT}"
echo "Marker source dir     : ${MARKER_SOURCE_DIR}"
echo "Short edge mm         : ${SHORT_EDGE_MM}"
echo "Indenters             : ${INDENTER_COUNT}"
echo "Episodes / indenter   : ${EPISODES_PER_INDENTER}"
echo "Frames / episode      : ${FRAMES_PER_EPISODE}"
echo "Markers / frame       : ${MARKERS_PER_FRAME}"
echo "Tasks / indenter      : ${TASKS_PER_INDENTER}"
echo "Total tasks           : ${TOTAL_TASKS}"
echo "Render tasks / indent : ${RENDER_TASKS_PER_INDENTER}"
echo "Total render tasks    : ${TOTAL_EVENTS}"
echo "Images / indenter     : ${IMAGES_PER_INDENTER}"
echo "Total images          : ${TOTAL_IMAGES}"
echo "Depth range mm        : ${DEPTH_MIN_MM} .. ${REQUESTED_DEPTH_MAX_MM}"
echo "Render device         : ${RENDER_DEVICE}"
echo "Render samples        : ${RENDER_SAMPLES} (fixed)"
echo "Physics workers       : ${MAX_PHYSICS_WORKERS}"
echo "Meshing workers       : ${MAX_MESHING_WORKERS}"
echo "Render workers        : ${MAX_RENDER_WORKERS}"
echo "Auto-balance pipeline : ${AUTO_BALANCE_PIPELINE}"
echo "Backlog high/low      : ${RENDER_BACKLOG_HIGH_WATERMARK}/${RENDER_BACKLOG_LOW_WATERMARK} (0=auto)"
echo "Physics NPZ cleanup   : ${PHYSICS_NPZ_CLEANUP}"
echo "Rough total ETA (h)   : ${ETA_HOURS}  [EST_IMAGES_PER_HOUR=${EST_IMAGES_PER_HOUR}]"
echo "Monitor interval      : ${MONITOR_INTERVAL_SEC}s"
echo

if [[ ! -f "/home/suhang/anaconda3/etc/profile.d/conda.sh" ]]; then
  echo "Missing conda activation script: /home/suhang/anaconda3/etc/profile.d/conda.sh" >&2
  exit 1
fi

source /home/suhang/anaconda3/etc/profile.d/conda.sh
conda activate genforce

START_EPOCH="$(date +%s)"
START_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"
RUN_LOG="${RUN_LOG:-${OUTPUT_ROOT%/}.run.log}"
echo "Start time            : ${START_HUMAN}"
echo "Run log               : ${RUN_LOG}"
echo

monitor_progress() {
  local log_file="$1"
  local interval_sec="$2"
  local last_render_line=""
  local last_global_line=""
  local last_balance_line=""

  while true; do
    sleep "${interval_sec}" || break
    [[ -f "${log_file}" ]] || continue

    local render_line global_line balance_line
    render_line="$(grep -F "Render progress |" "${log_file}" | tail -n 1 || true)"
    global_line="$(grep -F "UniForce overall progress |" "${log_file}" | tail -n 1 || true)"
    balance_line="$(grep -F "Pipeline auto-balance |" "${log_file}" | tail -n 1 || true)"

    if [[ -n "${balance_line}" && "${balance_line}" != "${last_balance_line}" ]]; then
      last_balance_line="${balance_line}"
      echo
      echo "[dynamic] ${balance_line}"
    fi
    if [[ -n "${render_line}" && "${render_line}" != "${last_render_line}" ]]; then
      last_render_line="${render_line}"
      echo "[dynamic] ${render_line}"
    fi
    if [[ -n "${global_line}" && "${global_line}" != "${last_global_line}" ]]; then
      last_global_line="${global_line}"
      echo "[dynamic] ${global_line}"
      echo
    fi
  done
}

cleanup_monitor() {
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "${MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
}

trap cleanup_monitor EXIT INT TERM

GENFORCE_ARGS=(
  --render-device "${RENDER_DEVICE}"
  --render-samples "${RENDER_SAMPLES}"
  --max-physics-workers "${MAX_PHYSICS_WORKERS}"
  --max-meshing-workers "${MAX_MESHING_WORKERS}"
  --max-render-workers "${MAX_RENDER_WORKERS}"
  --physics-npz-cleanup "${PHYSICS_NPZ_CLEANUP}"
)

case "${AUTO_BALANCE_PIPELINE}" in
  1|true|TRUE|yes|YES|on|ON)
    GENFORCE_ARGS+=(--auto-balance-pipeline)
    ;;
esac

if [[ "${RENDER_BACKLOG_HIGH_WATERMARK}" != "0" ]]; then
  GENFORCE_ARGS+=(--render-backlog-high-watermark "${RENDER_BACKLOG_HIGH_WATERMARK}")
fi
if [[ "${RENDER_BACKLOG_LOW_WATERMARK}" != "0" ]]; then
  GENFORCE_ARGS+=(--render-backlog-low-watermark "${RENDER_BACKLOG_LOW_WATERMARK}")
fi

monitor_progress "${RUN_LOG}" "${MONITOR_INTERVAL_SEC}" &
MONITOR_PID=$!

PYTHONUNBUFFERED=1 python "${GENERATOR}" \
  --output-root "${OUTPUT_ROOT}" \
  --short-edge-mm "${SHORT_EDGE_MM}" \
  --objects "${OBJECTS[@]}" \
  --episodes-per-indenter "${EPISODES_PER_INDENTER}" \
  --depth-min "${DEPTH_MIN_MM}" \
  --requested-depth-max-mm "${REQUESTED_DEPTH_MAX_MM}" \
  --seed "${SEED}" \
  --no-keep-work \
  --genforce-args \
  "${GENFORCE_ARGS[@]}" \
  2>&1 | tee "${RUN_LOG}"

cleanup_monitor
trap - EXIT INT TERM

END_EPOCH="$(date +%s)"
END_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"
ELAPSED_SEC="$((END_EPOCH - START_EPOCH))"

ELAPSED_HMS="$(python3 - <<'PY' "${ELAPSED_SEC}"
import sys
sec = int(sys.argv[1])
h = sec // 3600
m = (sec % 3600) // 60
s = sec % 60
print(f"{h:02d}:{m:02d}:{s:02d}")
PY
)"

ACTUAL_IMAGES_PER_HOUR="$(python3 - <<'PY' "${TOTAL_IMAGES}" "${ELAPSED_SEC}"
import sys
total = float(sys.argv[1])
elapsed = max(float(sys.argv[2]), 1.0)
rate = total / (elapsed / 3600.0)
print(f"{rate:.2f}")
PY
)"

echo
echo "End time              : ${END_HUMAN}"
echo "Elapsed               : ${ELAPSED_HMS}"
echo "Actual img/hour       : ${ACTUAL_IMAGES_PER_HOUR}"

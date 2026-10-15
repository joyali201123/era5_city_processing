#!/bin/bash
#SBATCH --job-name=era5land_t2m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=./slurm_logs/slurm_%x_%j.out
#SBATCH --error=./slurm_logs/slurm_%x_%j.err

# Submit from the Linux directory containing this SH and the Python downloader:
#   mkdir -p slurm_logs
#   sbatch run_era5_land_t2m_download.sh
#
# Slurm opens --output/--error before this script starts, so slurm_logs must
# already exist when sbatch is executed.
#
# IMPORTANT: F:\ is a Windows path and cannot be used directly on a Linux
# cluster. Fill the Linux absolute paths below with the cluster equivalents.

set -Eeuo pipefail

# --------------------------- USER SETTINGS ---------------------------
PYTHON_SCRIPT=""       # REQUIRED, e.g. /path/download_era5_land_t2m_global_1050_2026.py
VENV_ACTIVATE=""       # REQUIRED, e.g. /path/precip_env/bin/activate
OUTPUT_PATH=""         # REQUIRED Linux path for annual files and .parts
TRACKER_PATH=""        # REQUIRED Linux path to the xlsx tracker

START_YEAR=1950
END_YEAR=2026
WORKERS=2
RETRIES=3
FLUSH_SECONDS=60

# Number of year-grid tasks handled by one Slurm submission.
# Monthly files are resumable, so resubmit the same SH to continue.
# Set 0 to remove the limit and keep running until Slurm stops the job.
TASK_LIMIT=50

# Optional single-grid restriction for testing, e.g. G0001. Leave empty globally.
GRID_ID=""

# Error-log monitoring behavior.
ERROR_SCAN_INTERVAL_SECONDS=60
# --------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-$PWD}"
JOB_TAG="${SLURM_JOB_NAME:-era5land_t2m}_${SLURM_JOB_ID:-manual_$$}"
LOG_DIR="$PWD/slurm_logs"
RUN_LOG="$LOG_DIR/run_${JOB_TAG}.log"
ALERT_LOG="$LOG_DIR/alerts_${JOB_TAG}.log"
mkdir -p "$LOG_DIR"
: > "$RUN_LOG"
: > "$ALERT_LOG"

fail_setting() {
    echo "CONFIG ERROR: $1" | tee -a "$RUN_LOG" >&2
    exit 2
}

for key in PYTHON_SCRIPT VENV_ACTIVATE OUTPUT_PATH TRACKER_PATH; do
    [[ -n "${!key}" ]] || fail_setting "Fill $key at the top of this SH file."
done
[[ "$PYTHON_SCRIPT" == /* ]] || fail_setting "PYTHON_SCRIPT must be a Linux absolute path."
[[ "$VENV_ACTIVATE" == /* ]] || fail_setting "VENV_ACTIVATE must be a Linux absolute path."
[[ "$OUTPUT_PATH" == /* ]] || fail_setting "OUTPUT_PATH must be a Linux absolute path."
[[ "$TRACKER_PATH" == /* ]] || fail_setting "TRACKER_PATH must be a Linux absolute path."
[[ -r "$PYTHON_SCRIPT" ]] || fail_setting "Python script is unreadable: $PYTHON_SCRIPT"
[[ -r "$VENV_ACTIVATE" ]] || fail_setting "Environment activation file is unreadable: $VENV_ACTIVATE"

# Keep the module choice aligned with the existing city-mask Slurm wrapper.
module purge
module load Python/3.11.3-GCCcore-12.3.0
source "$VENV_ACTIVATE"
PYTHON_BIN="$(command -v python)"

WATCHER_PID=""
cleanup() {
    local rc=$?
    trap - EXIT
    if [[ -n "$WATCHER_PID" ]]; then
        kill "$WATCHER_PID" 2>/dev/null || true
        wait "$WATCHER_PID" 2>/dev/null || true
    fi
    if (( rc != 0 )); then
        echo "Job exited with status $rc at $(date -u --iso-8601=seconds)" | tee -a "$RUN_LOG" "$ALERT_LOG" >&2
    fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Avoid corrupting the shared Excel tracker if the same wrapper is submitted twice.
command -v flock >/dev/null 2>&1 || fail_setting "flock is required to protect the Excel tracker."
mkdir -p "$OUTPUT_PATH" "$(dirname "$TRACKER_PATH")"
LOCK_PATH="${TRACKER_PATH}.lock"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    echo "ERROR: Another ERA5-Land job already holds $LOCK_PATH" | tee -a "$RUN_LOG" "$ALERT_LOG" >&2
    exit 3
fi

{
    echo "Job tag: $JOB_TAG"
    echo "Start UTC: $(date -u --iso-8601=seconds)"
    echo "Python: $PYTHON_BIN"
    echo "Python script: $PYTHON_SCRIPT"
    echo "Output path: $OUTPUT_PATH"
    echo "Tracker: $TRACKER_PATH"
    echo "Years: $START_YEAR-$END_YEAR"
    echo "Workers: $WORKERS"
    echo "Task limit: $TASK_LIMIT"
    echo "Grid filter: ${GRID_ID:-all}"
    df -h "$OUTPUT_PATH" || true
} >> "$RUN_LOG" 2>&1

# Dependency check only; never install packages inside a compute job.
"$PYTHON_BIN" -c \
    'import cdsapi,openpyxl,xarray,netCDF4,pandas; print("Dependency check: OK")' \
    >> "$RUN_LOG" 2>&1

# The downloader writes INFO and ERROR messages to stderr. Therefore stderr
# cannot be treated as an error-only stream. Scan the combined log for actual
# error patterns instead. Each newly appended section is scanned only once.
(
    last_offset=0
    new_chunk="$LOG_DIR/.new_${JOB_TAG}.log"
    while sleep "$ERROR_SCAN_INTERVAL_SECONDS"; do
        current_size=$(stat -c %s "$RUN_LOG" 2>/dev/null || echo 0)
        if (( current_size < last_offset )); then
            last_offset=0
        fi
        if (( current_size > last_offset )); then
            tail -c "+$((last_offset + 1))" "$RUN_LOG" > "$new_chunk" || true
            if grep -Eiq \
                '(^|[[:space:]])(ERROR|CRITICAL)([[:space:]]|:)|Traceback \(most recent call last\)|[A-Za-z]+Error:' \
                "$new_chunk"; then
                {
                    echo "===== Detected at $(date -u --iso-8601=seconds) ====="
                    grep -Ein -C 4 \
                        '(^|[[:space:]])(ERROR|CRITICAL)([[:space:]]|:)|Traceback \(most recent call last\)|[A-Za-z]+Error:' \
                        "$new_chunk" || true
                } >> "$ALERT_LOG"
            fi
            last_offset=$current_size
        fi
    done
) &
WATCHER_PID=$!

ARGS=(
    --start-year "$START_YEAR"
    --end-year "$END_YEAR"
    --workers "$WORKERS"
    --retries "$RETRIES"
    --flush-seconds "$FLUSH_SECONDS"
    --output-dir "$OUTPUT_PATH"
    --tracker "$TRACKER_PATH"
    --log-level INFO
)
if (( TASK_LIMIT > 0 )); then
    ARGS+=(--limit "$TASK_LIMIT")
fi
if [[ -n "$GRID_ID" ]]; then
    ARGS+=(--grid-id "$GRID_ID")
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

set +e
"$PYTHON_BIN" -u "$PYTHON_SCRIPT" "${ARGS[@]}" 2>&1 | tee -a "$RUN_LOG"
PYTHON_RC=${PIPESTATUS[0]}
set -e

if (( PYTHON_RC != 0 )); then
    echo "ERROR: downloader exited with status $PYTHON_RC" | tee -a "$RUN_LOG" "$ALERT_LOG" >&2
    exit "$PYTHON_RC"
fi

echo "Completed UTC: $(date -u --iso-8601=seconds)" | tee -a "$RUN_LOG"

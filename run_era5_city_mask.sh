#!/bin/bash
#SBATCH --job-name=era5_city_mask
#SBATCH --nodes=1
#SBATCH --time=120:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --error=./slurm_logs/error_%x_%j.txt
#SBATCH --output=./slurm_logs/output_%x_%j.txt

# Submit from the directory containing BOTH scripts:
#   mkdir -p slurm_logs
#   sbatch run_era5_city_mask.sh
# IMPORTANT: Slurm opens its logs before this script runs; mkdir above is required.
# Fill Linux paths below. Windows drive letters cannot be used on Linux.
ARCHIVE_PATH=""   # REQUIRED fresh run: server absolute path to temp.tar.gz
GPKG_PATH=""      # REQUIRED: server absolute path to regional UCDB .gpkg
OUTPUT_PATH=""    # REQUIRED: Linux mapping of P:\oscar_team\users\biqing\era5_analysis_city_mask
EMAIL_TO="lizhuoran6@connect.hku.hk"       
# Process every NetCDF member in the archive (all available years).
FIRST_MEMBER_ONLY=0
RESUME_DAILY=0    # Set 1 to reuse OUTPUT_PATH/daily; ARCHIVE_PATH then optional.

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_PATH="$PWD/era5_city_stats.py"
JOB_TAG="${SLURM_JOB_NAME:-era5_city_mask}_${SLURM_JOB_ID:-manual_$$}"
mkdir -p slurm_logs
RUN_LOG="$PWD/slurm_logs/run_${JOB_TAG}.log"
ERR_LOG="$PWD/slurm_logs/python_error_${JOB_TAG}.log"
NOTICE_LOG="$PWD/slurm_logs/mail_${JOB_TAG}.log"
: > "$RUN_LOG"
: > "$ERR_LOG"
: > "$NOTICE_LOG"

for key in GPKG_PATH OUTPUT_PATH EMAIL_TO; do
    if [[ -z "${!key}" ]]; then
        echo "ERROR: Fill $key at the top of this SH file." >&2
        exit 2
    fi
done
if [[ "$RESUME_DAILY" != 1 && -z "$ARCHIVE_PATH" ]]; then
    echo 'ERROR: Fill ARCHIVE_PATH for a fresh run.' >&2
    exit 2
fi
if [[ "$OUTPUT_PATH" != /* || "$GPKG_PATH" != /* ]]; then
    echo 'ERROR: OUTPUT_PATH and GPKG_PATH must be Linux absolute paths.' >&2
    exit 2
fi

module purge
module load Python/3.11.3-GCCcore-12.3.0
source /pdrive/projects/oscar_team/shared/venvs/cmip6_py3_11_3/bin/activate
PYTHON_BIN="$(command -v python)"
SENDMAIL_BIN="$(command -v sendmail || true)"
if [[ -z "$SENDMAIL_BIN" && -x /usr/sbin/sendmail ]]; then
    SENDMAIL_BIN=/usr/sbin/sendmail
fi
if [[ -z "$SENDMAIL_BIN" ]]; then
    echo 'ERROR: sendmail is required to email log attachments. Ask the cluster administrator for a supported mail transport.' >&2
    exit 2
fi

send_notice() {
    local reason="$1"
    # Email is queued through the cluster mail relay; no password is stored.
    # Full current run/error logs are attachments. Delivery depends on relay policy.
    "$PYTHON_BIN" - "$EMAIL_TO" "$SENDMAIL_BIN" "$reason" "$JOB_TAG" "$RUN_LOG" "$ERR_LOG" >> "$NOTICE_LOG" 2>&1 <<'PYMAIL'
import pathlib, subprocess, sys
from email.message import EmailMessage
from email.utils import formatdate
recipient, transport, reason, job, *logs = sys.argv[1:]
m = EmailMessage()
m['To'] = recipient
m['Subject'] = f'ERA5 {job}: {reason}'
m['Date'] = formatdate(localtime=False)
m.set_content(f'Job: {job}\nStatus: {reason}\nCurrent logs are attached.\n')
for filename in logs:
    p = pathlib.Path(filename)
    if p.exists():
        m.add_attachment(p.read_bytes(), maintype='text', subtype='plain', filename=p.name)
subprocess.run([transport, '-t', '-oi'], input=m.as_bytes(), check=True, timeout=60)
print(f'Queued notice: {reason}', flush=True)
PYMAIL
}

WATCHER_PID=""
cleanup() {
    local rc=$?
    trap - EXIT
    if [[ -n "$WATCHER_PID" ]]; then
        kill "$WATCHER_PID" 2>/dev/null || true
        wait "$WATCHER_PID" 2>/dev/null || true
    fi
    if (( rc != 0 )); then
        echo "Job exited with status $rc at $(date -u --iso-8601=seconds)" >> "$ERR_LOG"
        send_notice "FAILED (exit $rc)" || echo 'ERROR: failure email could not be queued; inspect mail log.' >&2
    fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# An initial notification verifies queuing before expensive computation.
send_notice 'STARTED: notification transport check'
(
    last_periodic=$SECONDS
    last_error_size=0
    while sleep 60; do
        error_size=$(wc -c < "$ERR_LOG")
        if (( error_size > last_error_size )); then
            send_notice 'New stderr output (error or warning)' || true
            last_error_size=$error_size
        fi
        if (( SECONDS - last_periodic >= 43200 )); then
            send_notice '12-hour progress update' || true
            last_periodic=$SECONDS
        fi
    done
) &
WATCHER_PID=$!

# Python dependency check, no automatic installation on the server.
"$PYTHON_BIN" -c 'import numpy,pandas,xarray,netCDF4,geopandas,pyproj; from shapely import make_valid' 2>> "$ERR_LOG"
[[ -r "$SCRIPT_PATH" && -r "$GPKG_PATH" ]] || { echo 'Script or GPKG unreadable' >> "$ERR_LOG"; exit 2; }
ARGS=(--gpkg "$GPKG_PATH" --output "$OUTPUT_PATH")
if [[ "$RESUME_DAILY" == 1 ]]; then
    ARGS+=(--resume-daily)
else
    [[ -r "$ARCHIVE_PATH" ]] || { echo 'Archive unreadable' >> "$ERR_LOG"; exit 2; }
    ARGS+=(--archive "$ARCHIVE_PATH")
    [[ "$FIRST_MEMBER_ONLY" == 1 ]] && ARGS+=(--first-member-only)
fi
# Node-local scratch avoids extracting hourly members onto the shared output drive.
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    ARGS+=(--scratch "$SLURM_TMPDIR")
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# stderr stays separate so it can trigger notification; all stdout is streamed.
"$PYTHON_BIN" -u "$SCRIPT_PATH" "${ARGS[@]}" 2>> "$ERR_LOG" | tee -a "$RUN_LOG"
send_notice 'COMPLETED' || echo 'WARNING: completion email could not be queued.' >&2
# Logs are retained for audit, including empty stderr logs.

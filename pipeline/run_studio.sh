#!/bin/bash
# Full inference run on the Mac Studio. caffeinate holds the machine awake;
# every stage is chunk-restartable, so rerunning this script resumes.
set -uo pipefail
cd "$(dirname "$0")/.."
export FIGHTLAB_VIDEO="$PWD/work/fight_cfr30.mp4"
export FIGHTLAB_MODELS="$HOME/kinesis/models"
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY="$HOME/kinesis/venv/bin/python"
mkdir -p work
echo "== 02 track  $(date)"
caffeinate -is "$PY" pipeline/02_track.py 2>&1 | tee work/02.log
echo "== 03 masks  $(date)"
caffeinate -is "$PY" pipeline/03_masks.py 2>&1 | tee work/03.log
echo "== 04 pose3d $(date)"
caffeinate -is "$PY" pipeline/04_pose3d.py 2>&1 | tee work/04.log
echo "== all stages done $(date)"

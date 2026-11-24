#!/usr/bin/env bash
# Stage-1 clean-data grid, new 6:2:2 split (split_20261013.json): window length x bands x screening, 3 seeds each.
# Finished runs (summary.json present) are skipped, so rerunning resumes.
set -u
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
CACHE=/d/dronerfa_cache_20261013/files
SCREEN=outputs/window_screening_20261013/screening_windows.csv
OUT=outputs/stage1_clean_newsplit_20261013
mkdir -p "$OUT"
run() {  # name, extra args...
  local name=$1; shift
  for seed in 2026 2027 2028; do
    mkdir -p "$OUT/$name"
    local dir="$OUT/$name/seed_$seed"
    if [ -f "$dir/summary.json" ]; then echo "skip $name $seed"; continue; fi
    echo "=== $name seed $seed $(date +%H:%M:%S)"
    PYTHONIOENCODING=utf-8 $PY run_stage1_cached.py --cache $CACHE --screening $SCREEN --out "$dir" --seed $seed "$@" \
      > "$dir.log" 2>&1 || echo "FAILED $name $seed"
    tail -4 "$dir.log"
  done
}
for L in 10 15 20; do run "both_L${L}_screened" --window-ms $L --bands both; done
for L in 10 15 20; do run "rf0_L${L}_screened" --window-ms $L --bands rf0; done
for L in 10 15 20; do run "both_L${L}_unscreened" --window-ms $L --bands both --no-screening; done
echo "grid finished $(date +%H:%M:%S)"

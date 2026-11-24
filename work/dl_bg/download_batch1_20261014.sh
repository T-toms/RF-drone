#!/usr/bin/env bash
# Batch 1 (2026-10-14): outdoor D00 post-switch files of the 9 outdoor drones
# (S1000-S1111) and every missing file of the other 12 indoor classes.
# One class per RAR: download -> extract -> CRC check of every .mat against
# its RAR header -> copy into original/ -> delete RAR, so at most ~16 GB of
# temporary space is used. Downloading and extracting happen on the internal
# D: drive: the USB drive G: logged controller errors (event id 11) during the
# 2026-10-14 runs and the archives written there contained 1 MiB zero runs.
# Classes whose files are all present in original/ are skipped: rerun resumes.
set -u
ROOT=C:/Users/Administrator/Desktop/数据集处理
PY=/d/anaconda/python
ORIG=/g/DroneRFa数据集/original
WORK=/d/dronerfa_dl_staging
LOGS=/g/DroneRFa数据集/download_records_batch1_20261014
VPY=C:/Users/Administrator/Desktop/数据集处理/.venv/Scripts/python.exe
UNRAR="/c/Program Files/WinRAR/UnRAR.exe"
mkdir -p "$WORK" "$LOGS"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1

OUTDOOR="T0001 T0010 T0011 T0100 T0101 T0110 T0111 T1000 T1001"
INDOOR="T10000 T10001 T10010 T10011 T10100 T10101 T10110 T10111 T11000 T1101 T1110 T1111"
POST="S1000,S1001,S1010,S1011,S1100,S1101,S1110,S1111"

# missing segments of an indoor class, from the archive index
missing_indoor() {
  "$PY" -I -c "
import json, os, sys
t = sys.argv[1]
m = json.load(open(r'$ROOT/dronerfa_manifest.json', encoding='utf-8'))
have = set(os.listdir(r'G:/DroneRFa数据集/original'))
segs = sorted(n.rsplit('_', 1)[1][:-4] for n in (e['name'].split('/')[-1] for e in m['entries'])
              if n.endswith('.mat') and n.split('_')[0] == t and n not in have)
print(','.join(segs))
" "$1"
}

fetch() {  # target, distance-token, segments
  local t=$1 dist=$2 segs=$3
  [ -z "$segs" ] && { echo "[$(date +%m-%d\ %H:%M)] $t: 已齐全，跳过"; return 0; }
  echo "[$(date +%m-%d\ %H:%M)] === $t 开始：$segs"
  "$PY" "$ROOT/work/dl_bg/download_curl_wrapper.py" --manifest "$ROOT/dronerfa_manifest.json" \
    --targets "$t" --distances "$dist" --segments "$segs" \
    --output "$WORK/$t.rar" --extract-dir "$WORK/${t}_extract" --unrar "$UNRAR" \
    > "$LOGS/$t.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] && [ -f "$WORK/$t.rar" ]; then
    # Archive complete but extraction failed (CRC): re-fetch zero-filled pages, extract again.
    echo "[$(date +%m-%d\ %H:%M)] $t 解压校验失败，修复全零数据页后重试"
    "$VPY" "$ROOT/work/dl_bg/repair_zero_holes.py" "$WORK/$t.rar" >> "$LOGS/$t.log" 2>&1
    "$UNRAR" x -o+ -idq "$WORK/$t.rar" "$(cygpath -w "$WORK/${t}_extract")\\" >> "$LOGS/$t.log" 2>&1
    rc=$?
  fi
  if [ $rc -ne 0 ]; then echo "[$(date +%m-%d\ %H:%M)] $t 失败 rc=$rc，见 $LOGS/$t.log"; tail -3 "$LOGS/$t.log"; return 1; fi
  local n=0 bad=0
  for f in $(find "$WORK/${t}_extract" -name "*.mat"); do
    local b; b=$(basename "$f")
    [ -e "$ORIG/$b" ] && continue
    cp "$f" "$ORIG/$b.tmp"
    if "$VPY" "$ROOT/work/dl_bg/verify_mat_crc.py" "$ORIG/$b.tmp" --name "$b" >> "$LOGS/$t.log" 2>&1; then
      mv "$ORIG/$b.tmp" "$ORIG/$b" && n=$((n+1))
    else
      echo "[$(date +%m-%d\ %H:%M)] $b 复制到 G 盘后 CRC 不符，保留暂存文件"; rm -f "$ORIG/$b.tmp"; bad=$((bad+1))
    fi
  done
  echo "[$(date +%m-%d\ %H:%M)] $t 完成：CRC 校验后移入 $n 个文件，失败 $bad 个"
  [ $bad -eq 0 ] && rm -f "${WORK:?}/${t:?}.rar" && rm -rf "${WORK:?}/${t:?}_extract"
}

for t in $OUTDOOR; do
  segs=$(ls "$ORIG"/${t}_D00_S1*.mat 2>/dev/null | wc -l)
  if [ "$segs" -ge 8 ]; then echo "$t 切换后已齐全，跳过"; continue; fi
  fetch "$t" D00 "$POST"
done
for t in $INDOOR; do
  # indoor names have no distance token; pass the target itself as the distance filter
  fetch "$t" "$t" "$(missing_indoor "$t")"
done
echo "[$(date +%m-%d\ %H:%M)] 第一批结束；original 中共 $(ls "$ORIG"/*.mat | wc -l) 个文件"

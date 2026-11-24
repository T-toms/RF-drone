# -*- coding: utf-8 -*-
"""Step 12: which drone models exist in BOTH data/ and v3, with state / bandwidth / band per recording.
v3 state + video bandwidth come from the raw/ JSON sidecars (declared.state / declared.bw), matched by session id."""
import csv, json, glob, os, re, collections
from pathlib import Path
O = Path(__file__).resolve().parent
T = Path(r"C:/Users/Administrator/Desktop/数据集处理/work/crop_slice_20261107")
recs = list(csv.DictReader(open(O / "floor_per_recording.csv", encoding="utf-8")))
blocks = list(csv.DictReader(open(T / "blocks.csv", encoding="utf-8-sig")))
bw_of = {}
for b in blocks:
    bw_of.setdefault(b["group_id"], b["bw_label"])
print("data/ models:", sorted({r["model"] for r in recs if r["dataset"] == "data"}))
print("v3 models:", sorted({r["model"] for r in recs if r["dataset"] == "v3" and r["label"] == "drone"}))
# raw JSON metadata per session
meta = {}
for j in glob.glob(r"G:/drone/raw/**/*.json", recursive=True):
    d = json.load(open(j, encoding="utf-8"))
    sid = d["session"]["id"]
    if d["declared"].get("class") == "drone":
        meta.setdefault(sid, dict(state=d["declared"].get("state"), bw=d["declared"].get("bw"), links=d["declared"].get("links"),
                                  notes=d.get("notes"), fc=d["capture"]["center_freq_hz"] / 1e6, gain=d["receiver"]["gain_db"]))
for want in ["mini5", "evo"]:
    print("\n=====", want)
    for r in sorted(recs, key=lambda r: (r["band"], r["dataset"], r["group_id"])):
        if r["label"] != "drone" or want not in r["model"].replace(" ", "").lower():
            continue
        extra = ""
        if r["dataset"] == "v3":
            sid = re.search(r"(\d{8}-\d{6}_\w+)$", r["group_id"]).group(1)
            extra = str(meta.get(sid))
        print(r["band"], r["dataset"], r["model"], r["state"], bw_of.get(r["group_id"]), r["n_cells"], "ms", r["group_id"], extra)

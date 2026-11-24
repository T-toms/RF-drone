"""Independent record-level audit of the locked cascade development evaluation."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from evaluate_cascade12 import CONDITIONS, OUT, ROLES, summarize, verify_lock
from stage2_12class_data import CODE_TO_LABEL, load_contract_rows, sha256


def verify(out: Path = OUT) -> dict:
    lock_path = out / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    verify_lock(lock)
    summary = json.loads((out/"summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "completed": raise RuntimeError("Evaluation not completed")
    if datetime.fromisoformat(lock["locked_at"]) >= datetime.fromisoformat(summary["completed_at"]):
        raise RuntimeError("Model lock was not created before evaluation")
    with (out/"window_predictions.csv").open(encoding="utf-8",newline="") as f:
        raw = list(csv.DictReader(f))
    if len(raw) != 15360: raise RuntimeError(f"Expected 15360 records, got {len(raw)}")
    full,_ = load_contract_rows()
    expected = {(role,condition,int(r["row_index"])):r
                for role in ROLES for condition in CONDITIONS for r in full if r["split"]==role}
    actual_keys = [(r["role"],r["condition"],int(r["row_index"])) for r in raw]
    if len(expected)!=15360 or set(actual_keys)!=set(expected) or len(set(actual_keys))!=15360:
        raise RuntimeError("Row coverage, role, or uniqueness mismatch")
    records=[]
    for row in raw:
        source=expected[(row["role"],row["condition"],int(row["row_index"]))]
        if row["target_code"]!=source["target_code"] or row["filename"]!=source["filename"]:
            raise RuntimeError("Source identity mismatch")
        if int(row["full_split_position"])!=int(source["full_split_position"]):
            raise RuntimeError("Noise coordinate mismatch")
        truth_stage2=CODE_TO_LABEL.get(row["target_code"])
        truth_cascade=0 if truth_stage2 is None else truth_stage2+1
        if int(row["truth_cascade"])!=truth_cascade or (row["truth_stage2"]=="")!=(truth_stage2 is None) or (truth_stage2 is not None and int(row["truth_stage2"])!=truth_stage2):
            raise RuntimeError("Truth mapping mismatch")
        l0=np.float32(float(row["stage1_logit_background"]))
        l1=np.float32(float(row["stage1_logit_drone"]))
        margin=np.float32(l1-l0)
        gate=int(margin>0)
        if gate!=int(row["stage1_prediction"]) or abs(float(margin)-float(row["stage1_margin"]))>1e-6:
            raise RuntimeError("Stage1 gate mismatch")
        if truth_stage2 is not None or gate:
            logits=np.asarray([float(row[f"stage2_logit_{i:02d}"]) for i in range(12)],dtype=np.float32)
            pred2=int(logits.argmax())
            if pred2!=int(row["stage2_prediction"]): raise RuntimeError("Stage2 argmax mismatch")
        else:
            pred2=None
            if row["stage2_prediction"]!="": raise RuntimeError("Background rejected but Stage2 prediction present")
        final=0 if not gate else pred2+1
        if final!=int(row["cascade_prediction"]): raise RuntimeError("Cascade decision mismatch")
        if len(row["input_sha256"])!=64: raise RuntimeError("Missing input digest")
        records.append({"role":row["role"],"condition":row["condition"],"row_index":int(row["row_index"]),
                        "target_code":row["target_code"],"filename":row["filename"],
                        "truth_cascade":truth_cascade,"truth_stage2":truth_stage2,
                        "stage1_prediction":gate,"stage2_prediction":pred2,"cascade_prediction":final})
    if sum(r["truth_stage2"] is not None for r in records)!=14240: raise RuntimeError("Oracle coverage")
    computed, per_class, confusion=summarize(records)
    with (out/"metrics_by_condition.csv").open(encoding="utf-8",newline="") as f:
        reported={(r["role"],r["condition"]):r for r in csv.DictReader(f)}
    if len(reported)!=8: raise RuntimeError("Condition metric coverage")
    for row in computed:
        rep=reported[(row["role"],row["condition"])]
        for key,value in row.items():
            if key in ("role","condition"): continue
            if value is None:
                if rep[key]!="": raise RuntimeError(f"{key} must be N/A")
            elif abs(float(rep[key])-float(value))>1e-10:
                raise RuntimeError(f"Metric mismatch {row['role']} {row['condition']} {key}")
    saved_confusion=json.loads((out/"confusion_matrices.json").read_text(encoding="utf-8"))
    if saved_confusion!=confusion: raise RuntimeError("Confusion matrix mismatch")
    audit=json.loads((out/"read_audit.json").read_text(encoding="utf-8"))
    counts=Counter((r["role"],r["condition"]) for r in records)
    if {(a["role"],a["condition"]):a["windows_read"] for a in audit}!=counts:
        raise RuntimeError("Read audit count mismatch")
    result={"passed":True,"records":len(records),"oracle_drone_records":14240,
            "conditions":len(computed),"per_class_records":len(per_class),
            "lock_sha256":sha256(lock_path),"window_predictions_sha256":sha256(out/"window_predictions.csv"),
            "test_iq_read_by_verifier":False}
    (out/"independent_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=OUT)
    a=parser.parse_args();print(json.dumps(verify(a.output),ensure_ascii=False,indent=2))

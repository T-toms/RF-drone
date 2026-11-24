"""Independent CSV and checkpoint audit of the completed Stage2 run."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from run_stage2_12drones import CODE_TO_LABEL, CONDITIONS, EPOCHS, OUT, metrics
from stage2_12class_data import ROLE_COUNTS, sha256

FIRST_SEGMENT_AUDIT = Path(__file__).resolve().parent.parent / "work/stage2-epoch1-read-audit-20260929.json"


def verify(out: Path = OUT) -> dict:
    summary = json.loads((out/"summary.json").read_text(encoding="utf-8"))
    selection = json.loads((out/"selection.json").read_text(encoding="utf-8"))
    config = json.loads((out/"run_config.json").read_text(encoding="utf-8"))
    audit = json.loads((out/"read_audit.json").read_text(encoding="utf-8"))
    issues = []
    if summary.get("status") != "completed": issues.append("summary not completed")
    if audit.get("train_windows_read") != 8990*5: issues.append("train read count")
    if audit.get("validation_windows_read") != 2080*4: issues.append("validation read count")
    if audit.get("test_files_touched"): issues.append("test IQ touched")
    if audit.get("train_files") != 58 or audit.get("validation_files") != 14: issues.append("file counts")
    if config.get("class_counts") != {str(i): n for i,n in enumerate([760,590,770,760,620,780,750,730,870,670,840,850])}:
        issues.append("train class counts")
    if len(config.get("class_weights",[])) != 12: issues.append("weight vector")
    scores = {}
    for epoch in range(1,EPOCHS+1):
        path = out/f"validation_scores_epoch_{epoch:02d}.csv"
        if not path.exists(): issues.append(f"missing epoch {epoch}"); continue
        with path.open(encoding="utf-8",newline="") as f: raw=list(csv.DictReader(f))
        keys={(r["condition"],int(r["row_index"])) for r in raw}
        if len(raw)!=8320 or len(keys)!=8320: issues.append(f"epoch {epoch} coverage")
        by_cond={}
        for cond in CONDITIONS:
            group=[]
            for r in raw:
                if r["condition"]!=cond: continue
                label=int(r["label"]); pred=int(r["prediction"])
                logits=np.asarray([float(r[f"logit_{j:02d}"]) for j in range(12)],dtype=np.float32)
                if pred!=int(logits.argmax()): issues.append(f"epoch {epoch} argmax")
                if label!=CODE_TO_LABEL[r["target_code"]]: issues.append(f"epoch {epoch} mapping")
                group.append({"target_code":r["target_code"],"filename":r["filename"],"label":label,"prediction":pred})
            if len(group)!=ROLE_COUNTS["val"]: issues.append(f"epoch {epoch} {cond} count")
            else: by_cond[cond]=metrics(group)["class_file_equal_recall"]
        if len(by_cond)==4: scores[epoch]=float(np.mean(list(by_cond.values())))
    if len(scores)==EPOCHS:
        best=max(scores,key=lambda epoch:(scores[epoch],-epoch))
        if best!=selection.get("selected_epoch"): issues.append("selected epoch")
        if abs(scores[best]-float(selection.get("selected_score",-1)))>1e-12: issues.append("selected score")
    checkpoint=out/"best_model.pt"
    if sha256(checkpoint)!=summary.get("best_model_sha256"): issues.append("checkpoint hash")
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if saved.get("epoch")!=selection.get("selected_epoch"): issues.append("checkpoint epoch")
    if saved.get("label_mapping")!=CODE_TO_LABEL: issues.append("checkpoint mapping")
    if saved["model_state"]["classifier.weight"].shape[0]!=12: issues.append("checkpoint output width")
    prior = json.loads(FIRST_SEGMENT_AUDIT.read_text(encoding="utf-8")) if FIRST_SEGMENT_AUDIT.exists() else None
    physical_validation_reads = audit.get("validation_windows_read", 0) + (prior.get("validation_windows_read", 0) if prior else 0)
    if prior and prior.get("train_windows_read") != 8990: issues.append("first-segment train read count")
    if prior and physical_validation_reads != 16640: issues.append("resumed physical validation read count")
    result={"passed":not issues,"issues":sorted(set(issues)),"score_by_epoch":scores,
            "selected_epoch":selection.get("selected_epoch"),"best_model_sha256":summary.get("best_model_sha256")}
    result["physical_read_accounting"] = {"train_windows_total":audit.get("train_windows_read"),
                                          "validation_first_segment":prior.get("validation_windows_read") if prior else None,
                                          "validation_resume_segment":audit.get("validation_windows_read"),
                                          "validation_windows_total":physical_validation_reads,
                                          "test_files_touched":audit.get("test_files_touched")}
    (out/"independent_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    if issues: raise RuntimeError(f"Stage2 verification failed: {sorted(set(issues))}")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--run-root",type=Path,default=OUT)
    a=parser.parse_args();print(json.dumps(verify(a.run_root),ensure_ascii=False,indent=2))

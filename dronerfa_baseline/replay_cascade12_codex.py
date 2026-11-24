"""Small raw-IQ replay of the locked cascade outputs on both test roles."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_cascade12 import CONDITIONS, OUT, ROLES, load_models, verify_lock
from stage2_12class_data import FULL_COUNTS, NOISE_SEEDS, load_contract_rows, read_iq
from train_resnet18 import extract_features


def main():
    lock=json.loads((OUT/"selection_lock.json").read_text(encoding="utf-8"))
    verify_lock(lock)
    full,_=load_contract_rows()
    sources={int(r["row_index"]):r for r in full}
    with (OUT/"window_predictions.csv").open(encoding="utf-8",newline="") as f:
        predictions=list(csv.DictReader(f))
    chosen=[]
    for role in ROLES:
        for condition in CONDITIONS:
            group=[r for r in predictions if r["role"]==role and r["condition"]==condition]
            chosen.append(group[0])
            if role=="development_test_exposed":
                chosen.append(next(r for r in group if r["target_code"]!="T0000"))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend,model1,model2=load_models(device)
    records=[]
    with torch.inference_mode():
        for row in chosen:
            source=sources[int(row["row_index"])]
            iq=read_iq(source,seed=NOISE_SEEDS[row["role"]],epoch=0,
                       full_role_length=FULL_COUNTS[row["role"]],fixed_snr=CONDITIONS[row["condition"]])
            digest=hashlib.sha256(iq.numpy().tobytes()).hexdigest()
            if digest!=row["input_sha256"]: raise RuntimeError("Raw-IQ replay digest mismatch")
            features=extract_features(frontend,iq.unsqueeze(0).to(device))
            with torch.amp.autocast(device_type=device.type,enabled=device.type=="cuda",dtype=torch.float16):
                l1=model1(features).float().cpu().numpy()[0]
                l2=model2(features).float().cpu().numpy()[0]
            old1=np.array([float(row["stage1_logit_background"]),float(row["stage1_logit_drone"])],dtype=np.float32)
            d1=float(np.max(np.abs(l1-old1)))
            d2=None
            if row["stage2_prediction"]!="":
                old2=np.array([float(row[f"stage2_logit_{j:02d}"]) for j in range(12)],dtype=np.float32)
                d2=float(np.max(np.abs(l2-old2)))
            gate=int(np.float32(l1[1])-np.float32(l1[0])>0)
            if gate!=int(row["stage1_prediction"]): raise RuntimeError("Stage1 replay gate mismatch")
            if row["stage2_prediction"]!="" and int(l2.argmax())!=int(row["stage2_prediction"]):
                raise RuntimeError("Stage2 replay argmax mismatch")
            if d1>0.05 or (d2 is not None and d2>0.05):
                raise RuntimeError(f"Replay logit difference exceeds tolerance: {d1}, {d2}")
            records.append({"role":row["role"],"condition":row["condition"],"row_index":int(row["row_index"]),
                            "target_code":row["target_code"],"input_sha256_match":True,
                            "stage1_max_abs_logit_difference":d1,"stage2_max_abs_logit_difference":d2,
                            "stage1_gate_match":True,"stage2_argmax_match":row["stage2_prediction"]!=""})
    result={"passed":True,"windows_replayed":len(records),"max_stage1_logit_difference":max(r["stage1_max_abs_logit_difference"] for r in records),
            "max_stage2_logit_difference":max(r["stage2_max_abs_logit_difference"] or 0 for r in records),
            "records":records,"note":"Spot replay only; independent full raw-IQ replay was not performed."}
    (OUT/"raw_iq_spot_replay.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="records"},ensure_ascii=False,indent=2))


if __name__=="__main__": main()

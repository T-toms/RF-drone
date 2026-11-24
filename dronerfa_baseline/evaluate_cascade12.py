"""Locked development evaluation for Stage1 detector + Stage2 twelve-way classifier."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import ResNet18, SpectrogramFrontEnd
from stage2_12class_data import (CODE_TO_LABEL, FULL_COUNTS, INDEX, INDEX_SHA256,
                                 NOISE_SEEDS, OUT as CONTRACT_DIR, SPLIT_SHA256,
                                 load_contract_rows, read_iq, sha256)
from train_resnet18 import extract_features

HERE = Path(__file__).resolve().parent
STAGE1 = HERE / "outputs/stage1_12drones_20260926/seed_2026/best_model.pt"
STAGE1_HASH = "37dbb815450b8ec0f798a113a7f922858fcb2be604a863eb7594eb5c52ff814b"
STAGE2 = HERE / "outputs/stage2_12drones_20260927/seed_2027/best_model.pt"
STAGE2_SUMMARY = STAGE2.parent / "summary.json"
OUT = HERE / "outputs/cascade12_development_20260927"
CONDITIONS = {"clean": None, "-10dB": -10.0, "-12dB": -12.0, "-15dB": -15.0}
ROLES = ("development_test", "development_test_exposed")
BATCH = 4


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def lock_models(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / "selection_lock.json"
    if lock_path.exists(): raise FileExistsError("Selection lock already exists")
    if sha256(STAGE1) != STAGE1_HASH: raise RuntimeError("Stage1 model changed")
    verification_path = STAGE2.parent / "independent_verification.json"
    if not verification_path.exists(): raise RuntimeError("Stage2 independent verification is required before lock")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if not verification.get("passed"): raise RuntimeError("Stage2 independent verification failed")
    summary = json.loads(STAGE2_SUMMARY.read_text(encoding="utf-8"))
    selection = json.loads((STAGE2.parent/"selection.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed" or summary["selected_epoch"] != selection["selected_epoch"]:
        raise RuntimeError("Stage2 run is not complete and selected")
    if summary["train_windows_read"] != 44950 or summary["validation_windows_read"] != 8320 or summary["test_files_touched"]:
        raise RuntimeError("Stage2 read budget incorrect")
    if sha256(STAGE2) != summary["best_model_sha256"]: raise RuntimeError("Stage2 model changed")
    checkpoint = torch.load(STAGE2, map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != selection["selected_epoch"] or checkpoint["label_mapping"] != CODE_TO_LABEL:
        raise RuntimeError("Stage2 checkpoint epoch or mapping mismatch")
    contract = json.loads((CONTRACT_DIR/"protocol.json").read_text(encoding="utf-8"))
    if contract["index_sha256"] != INDEX_SHA256 or contract["split_sha256"] != SPLIT_SHA256:
        raise RuntimeError("Contract disagrees with frozen input")
    lock = {"schema":"dronerfa-cascade12-development-lock/v1", "locked_at":datetime.now(timezone.utc).isoformat(),
            "selection_source":"work/stage1-12drones-freeze-20260929.json and Stage2 frozen file-class-equal validation rule",
            "stage1_checkpoint":str(STAGE1), "stage1_sha256":STAGE1_HASH, "stage1_epoch":3,
            "stage1_threshold":"float32(logit_drone) - float32(logit_background) > 0",
            "stage2_checkpoint":str(STAGE2), "stage2_sha256":summary["best_model_sha256"],
            "stage2_epoch":selection["selected_epoch"], "stage2_val_score":selection["selected_score"],
            "stage2_decision":"argmax of float32 12 logits", "mapping":CODE_TO_LABEL,
            "index_sha256":INDEX_SHA256, "split_sha256":SPLIT_SHA256,
            "mapping_sha256":sha256(CONTRACT_DIR/"mapping.json"),
            "stage2_view_sha256":sha256(CONTRACT_DIR/"stage2_index.csv"),
            "stage2_verification_sha256":sha256(verification_path),
            "conditions":list(CONDITIONS), "noise_seeds":{role:NOISE_SEEDS[role] for role in ROLES},
            "frontend":{"n_fft":512,"hop_length":256,"image_size":256},
            "evaluation_source_sha256":sha256(Path(__file__)), "test_iq_opened_before_lock":False}
    dump(lock_path,lock)
    return lock


def verify_lock(lock):
    checks = {"stage1_sha256":sha256(STAGE1), "stage2_sha256":sha256(STAGE2),
              "index_sha256":sha256(INDEX), "split_sha256":sha256(HERE.parent/"work/meta_split2/split_candidate_104_20260926.csv"),
              "mapping_sha256":sha256(CONTRACT_DIR/"mapping.json"),
              "stage2_view_sha256":sha256(CONTRACT_DIR/"stage2_index.csv"),
              "stage2_verification_sha256":sha256(STAGE2.parent/"independent_verification.json"),
              "evaluation_source_sha256":sha256(Path(__file__))}
    for key, actual in checks.items():
        if lock[key] != actual: raise RuntimeError(f"Selection lock mismatch: {key}")
    if lock["mapping"] != CODE_TO_LABEL or lock["stage1_threshold"] != "float32(logit_drone) - float32(logit_background) > 0":
        raise RuntimeError("Selection lock decision mapping changed")


class EvalDataset(Dataset):
    def __init__(self, full_rows, role, snr):
        self.rows = [r for r in full_rows if r["split"] == role]
        self.role, self.snr = role, snr
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row = self.rows[i]
        return read_iq(row, seed=NOISE_SEEDS[self.role], epoch=0,
                       full_role_length=FULL_COUNTS[self.role], fixed_snr=self.snr), i


def load_models(device):
    one = torch.load(STAGE1,map_location="cpu",weights_only=False)
    two = torch.load(STAGE2,map_location="cpu",weights_only=False)
    model1 = ResNet18(1,2).to(device); model1.load_state_dict(one["model_state"]); model1.eval()
    model2 = ResNet18(1,12).to(device); model2.load_state_dict(two["model_state"]); model2.eval()
    front = SpectrogramFrontEnd(512,256,256).to(device)
    front.load_state_dict(one["frontend_state"]); front.eval()
    # Frontend is parameter-free under this protocol; both checkpoints must agree.
    if one["frontend_state"].keys() != two["frontend_state"].keys():
        raise RuntimeError("Frontends differ")
    for key in one["frontend_state"]:
        if not torch.equal(one["frontend_state"][key],two["frontend_state"][key]):
            raise RuntimeError("Frontends differ")
    return front,model1,model2


def summarize(records):
    role_condition = defaultdict(list)
    for r in records: role_condition[(r["role"],r["condition"])].append(r)
    summaries, per_class = [], []
    confusion = {}
    for (role,condition), group in role_condition.items():
        bg = [r for r in group if r["truth_cascade"] == 0]
        drones = [r for r in group if r["truth_cascade"] != 0]
        bg_fp = sum(r["stage1_prediction"] == 1 for r in bg)
        missed = sum(r["stage1_prediction"] == 0 for r in drones)
        passed_wrong = sum(r["stage1_prediction"] == 1 and r["stage2_prediction"] != r["truth_stage2"] for r in drones)
        total_error = sum(r["cascade_prediction"] != r["truth_cascade"] for r in group)
        if bg_fp+missed+passed_wrong != total_error: raise RuntimeError("Error decomposition does not close")
        file_groups = defaultdict(list)
        for r in drones: file_groups[(r["target_code"],r["filename"])].append(r)
        class_scores = {}
        for code in sorted({r["target_code"] for r in drones}):
            files = [g for (c,_),g in file_groups.items() if c==code]
            class_scores[code] = float(np.mean([sum(x["cascade_prediction"]==x["truth_cascade"] for x in f)/len(f) for f in files]))
            class_group = [r for r in drones if r["target_code"]==code]
            per_class.append({"role":role,"condition":condition,"target_code":code,"windows":len(class_group),
                              "files":len(files),"cascade_recall":sum(r["cascade_prediction"]==r["truth_cascade"] for r in class_group)/len(class_group),
                              "file_equal_recall":class_scores[code]})
        summaries.append({"role":role,"condition":condition,"windows":len(group),"background_windows":len(bg),
                          "drone_windows":len(drones),"cascade_accuracy":(len(group)-total_error)/len(group),
                          "class_file_equal_recall":float(np.mean(list(class_scores.values()))),
                          "background_fpr":bg_fp/len(bg) if bg else None,
                          "stage1_drone_fnr":missed/len(drones) if drones else None,
                          "oracle_stage2_accuracy":sum(r["stage2_prediction"]==r["truth_stage2"] for r in drones)/len(drones) if drones else None,
                          "detected_drone_stage2_accuracy":sum(r["stage2_prediction"]==r["truth_stage2"] for r in drones if r["stage1_prediction"]==1)/max(len(drones)-missed,1),
                          "background_false_positive":bg_fp,"stage1_drone_missed":missed,
                          "stage2_wrong_after_gate":passed_wrong,"total_errors":total_error})
        matrix = np.zeros((13,13),dtype=np.int64)
        for r in group: matrix[r["truth_cascade"],r["cascade_prediction"]] += 1
        confusion[f"{role}/{condition}"] = matrix.tolist()
    return summaries,per_class,confusion


@torch.inference_mode()
def evaluate(out:Path):
    lock_path=out/"selection_lock.json"
    if not lock_path.exists(): raise RuntimeError("Model selection must be locked before test IQ opens")
    lock=json.loads(lock_path.read_text(encoding="utf-8"))
    verify_lock(lock)
    if (out/"window_predictions.csv").exists(): raise FileExistsError("Evaluation already exists")
    full,_=load_contract_rows()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(10)
    front,model1,model2=load_models(device)
    records=[]; audit=[]
    amp=device.type=="cuda"
    for role in ROLES:
        for condition,snr in CONDITIONS.items():
            ds=EvalDataset(full,role,snr)
            loader=DataLoader(ds,batch_size=BATCH,shuffle=False,num_workers=0,
                              generator=torch.Generator().manual_seed(101027),pin_memory=amp)
            for batch,(iq,indices) in enumerate(loader,1):
                features=extract_features(front,iq.to(device))
                with torch.amp.autocast(device_type=device.type,enabled=amp,dtype=torch.float16):
                    logits1=model1(features); logits2=model2(features)
                l1=logits1.float().cpu().numpy(); l2=logits2.float().cpu().numpy()
                margins=l1[:,1].astype(np.float32)-l1[:,0].astype(np.float32)
                gates=(margins>0).astype(np.int64); pred2=l2.argmax(1)
                for i,idx in enumerate(indices.tolist()):
                    row=ds.rows[idx]; code=row["target_code"]
                    truth_stage2=CODE_TO_LABEL.get(code)
                    truth_cascade=0 if truth_stage2 is None else truth_stage2+1
                    final=0 if gates[i]==0 else int(pred2[i])+1
                    iq_hash=hashlib.sha256(iq[i].numpy().tobytes()).hexdigest()
                    records.append({"role":role,"condition":condition,"row_index":int(row["row_index"]),
                                    "full_split_position":int(row["full_split_position"]),
                                    "filename":row["filename"],"target_code":code,"window_index":int(row["window_index"]),
                                    "truth_stage2":truth_stage2,"truth_cascade":truth_cascade,
                                    "stage1_logit_background":float(l1[i,0]),"stage1_logit_drone":float(l1[i,1]),
                                    "stage1_margin":float(margins[i]),"stage1_prediction":int(gates[i]),
                                    "stage2_prediction":int(pred2[i]) if truth_stage2 is not None or gates[i] else None,
                                    "cascade_prediction":final,"input_sha256":iq_hash,
                                    **{f"stage2_logit_{j:02d}":float(l2[i,j]) if truth_stage2 is not None or gates[i] else None for j in range(12)}})
                if batch % 100 == 0 or batch == len(loader):
                    print(f"{role} {condition}: {batch}/{len(loader)}",flush=True)
            audit.append({"role":role,"condition":condition,"windows_read":len(ds),
                          "files_opened":len({r["file_path"] for r in ds.rows})})
    if len(records)!=15360 or len({(r["role"],r["condition"],r["row_index"]) for r in records})!=15360:
        raise RuntimeError("Missing or duplicate evaluation windows")
    if sum(r["truth_stage2"] is not None for r in records)!=14240: raise RuntimeError("Oracle count mismatch")
    summaries, per_class, confusion=summarize(records)
    write_csv(out/"window_predictions.csv",records)
    write_csv(out/"metrics_by_condition.csv",summaries)
    write_csv(out/"per_class_metrics.csv",per_class)
    dump(out/"confusion_matrices.json",confusion)
    dump(out/"read_audit.json",audit)
    result={"status":"completed","lock_sha256":sha256(lock_path),"rows":len(records),
            "oracle_drone_rows":14240,"conditions":summaries,"test_read_audit":audit,
            "completed_at":datetime.now(timezone.utc).isoformat()}
    dump(out/"summary.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--lock-only",action="store_true")
    args=parser.parse_args()
    print(json.dumps(lock_models(args.output) if args.lock_only else evaluate(args.output),ensure_ascii=False,indent=2),flush=True)

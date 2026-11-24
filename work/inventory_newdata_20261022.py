"""Inventory of the lab-collected dataset under <root>/data/{train,test}.

Parses the directory/file naming convention
    <code>_<band>_<bw>[_combination]-signal/<same>-20260807-<YYYY-MM-DD-HH-MM-SS>-epoch-<k>_seg<NN>[_<dup>].mat
and reports, per (state, model, band, bandwidth): file counts, capture sessions,
epochs, missing segment indices, duplicate files, and train/test segment overlap.

Usage: python -I inventory_newdata_20261022.py <data_dir> [--csv out.csv]
"""
import collections
import csv
import os
import re
import sys

MODELS = {
    "001": "mini4 pro",
    "002": "Flip",
    "003": "mini 4k",
    "004": "neo",
    "005": "neo2",
    "006": "air 3s",
    "007": "mini5 pro",
    "008": "mavic 4",
    "009": "avata2",
    "010": "evo lite 6k",
    "011": "evo ii pro v3",
    "012": "avata",
    "013": "atom se",
    "014": "atom",
}

DIR_RE = re.compile(r"^(F?)(\d{3})_(2\.4GH[Zz]|5\.8GH[Zz])_(\d+M?)(_combination)?-signal$")
FILE_RE = re.compile(r"-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})-epoch-(\d+)_seg(\d+)(_\d{8}_\d{6})?\.mat$")


def scan(data_dir):
    recs = []
    for split in ("train", "test"):
        sp = os.path.join(data_dir, split)
        if not os.path.isdir(sp):
            continue
        for batch in sorted(os.listdir(sp)):
            bp = os.path.join(sp, batch)
            if not os.path.isdir(bp):
                continue
            for rec in sorted(os.listdir(bp)):
                rp = os.path.join(bp, rec)
                if not os.path.isdir(rp):
                    continue
                m = DIR_RE.match(rec)
                if not m:
                    print(f"  !! unparsed dir name: {split}/{batch}/{rec}")
                    continue
                flying, code, band, bw, comb = m.groups()
                for fn in sorted(os.listdir(rp)):
                    if not fn.endswith(".mat"):
                        print(f"  !! non-mat file: {split}/{batch}/{rec}/{fn}")
                        continue
                    fm = FILE_RE.search(fn)
                    if not fm:
                        print(f"  !! unparsed file name: {split}/{batch}/{rec}/{fn}")
                        continue
                    sess, epoch, seg, dup = fm.groups()
                    recs.append(
                        {
                            "split": split,
                            "batch": batch,
                            "state": "flight" if flying else "standby",
                            "code": code,
                            "model": MODELS.get(code, "?"),
                            "band": band.replace("GHZ", "GHz"),
                            "bw": bw + ("+comb" if comb else ""),
                            "session": sess,
                            "epoch": int(epoch),
                            "seg": int(seg),
                            "dup": bool(dup),
                            "size": os.path.getsize(os.path.join(rp, fn)),
                        }
                    )
    return recs


def main(argv):
    data_dir = argv[0]
    recs = scan(data_dir)
    key = lambda r: (r["state"], r["code"], r["band"], r["bw"])  # noqa: E731

    groups = collections.defaultdict(list)
    for r in recs:
        groups[key(r)].append(r)

    print()
    hdr = f"{'state':<8} {'code':<4} {'model':<14} {'band':<7} {'bw':<9} {'train':>6} {'test':>5} {'GB':>6} {'sessions':>8} {'epochs':>6} {'dup':>4} {'missing segs':<28} {'overlap'}"
    print(hdr)
    print("-" * len(hdr))
    tot_tr = tot_te = 0
    for k in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3])):
        g = groups[k]
        tr = [r for r in g if r["split"] == "train"]
        te = [r for r in g if r["split"] == "test"]
        tot_tr += len(tr)
        tot_te += len(te)
        sess = sorted({r["session"] for r in g})
        epochs = sorted({(r["session"], r["epoch"]) for r in g})
        dups = sum(r["dup"] for r in g)
        # missing seg indices per (session, epoch), assuming 0..max
        miss = []
        for s, e in epochs:
            segs = {r["seg"] for r in g if r["session"] == s and r["epoch"] == e}
            full = set(range(max(segs) + 1))
            gap = sorted(full - segs)
            if gap:
                miss.append(f"e{e}:{','.join(map(str, gap))}")
        # does any (session, epoch, seg) appear in both train and test?
        tr_ids = {(r["session"], r["epoch"], r["seg"]) for r in tr}
        te_ids = {(r["session"], r["epoch"], r["seg"]) for r in te}
        both = tr_ids & te_ids
        same_sess = {r["session"] for r in tr} & {r["session"] for r in te}
        ov = "SAME-FILE" if both else ("same-session" if same_sess else "-")
        print(
            f"{k[0]:<8} {k[1]:<4} {MODELS.get(k[1], '?'):<14} {k[2]:<7} {k[3]:<9} "
            f"{len(tr):>6} {len(te):>5} {sum(r['size'] for r in g) / 2**30:>6.2f} "
            f"{len(sess):>8} {len(epochs):>6} {dups:>4} {';'.join(miss)[:28]:<28} {ov}"
        )

    print()
    print(f"total files: train={tot_tr} test={tot_te} ({tot_tr + tot_te})   "
          f"size={sum(r['size'] for r in recs) / 2**30:.1f} GB")
    print("models present (standby):", sorted({r["code"] for r in recs if r["state"] == "standby"}))
    print("models present (flight) :", sorted({r["code"] for r in recs if r["state"] == "flight"}))
    print("missing entirely        :", sorted(set(MODELS) - {r["code"] for r in recs}))
    print("duplicate-suffixed files:", sum(r["dup"] for r in recs))
    print("batch dirs per split    :",
          {s: sorted({r["batch"] for r in recs if r["split"] == s}, key=int) for s in ("train", "test")})

    if "--csv" in argv:
        out = argv[argv.index("--csv") + 1]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
        print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))

from core import BANDS, CACHE_ROOT, OUT, SEEDS, disk_guard, window_dirs


class ProcessCounters(ctypes.Structure):
    _fields_ = [("cb",ctypes.c_ulong),("PageFaultCount",ctypes.c_ulong)] + [
        (name,ctypes.c_size_t) for name in ("PeakWorkingSetSize","WorkingSetSize",
            "QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")
    ]


def working_set(handle):
    metrics = ProcessCounters()
    metrics.cb = ctypes.sizeof(metrics)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.c_void_p(handle),ctypes.byref(metrics),metrics.cb,
    )
    return metrics.PeakWorkingSetSize if ok else 0


def run_child(script,args,env,watch_memory=False):
    command = [sys.executable,"-I","-B","-X","utf8","-u",str(OUT / script),*args]
    print("START",script,*args,flush=True)
    token = str(time.monotonic_ns())
    open_process = ctypes.windll.kernel32.OpenProcess
    open_process.restype = ctypes.c_void_p
    handle = None
    peak = 0
    retry = False
    try:
        with subprocess.Popen(command,env=dict(env,WINDOW_W_PROCESS_TOKEN=token)) as proc:
            while proc.poll() is None:
                try:
                    disk_guard()
                except OSError:
                    subprocess.run(["taskkill","/PID",str(proc.pid),"/T","/F"],check=False)
                    raise
                if watch_memory and not handle:
                    info = OUT / f"L{env['WINDOW_W_L']}" / "training_process.json"
                    if info.exists():
                        try:
                            child = json.loads(info.read_text(encoding="utf-8"))
                        except json.JSONDecodeError:
                            child = {}
                        if child.get("token") == token:
                            handle = open_process(0x1000,False,child["pid"])
                if handle:
                    peak = max(peak,working_set(handle))
                if peak > 8_000_000_000:
                    subprocess.run(["taskkill","/PID",str(proc.pid),"/T","/F"],check=False)
                    if env.get("WINDOW_W_MMAP") == "1":
                        raise MemoryError(f"{script} working set above 8 GB with mmap: {peak}")
                    print("WORKING SET",peak,"retry mmap",flush=True)
                    retry = True
                    break
                time.sleep(3)
            if not retry and proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode,command)
    finally:
        if handle:
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle))
    if retry:
        env["WINDOW_W_MMAP"] = "1"
        return run_child(script,args,env,watch_memory)
    print("DONE",script,*args,"peak_host_bytes",peak,flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step",choices=("cache","eval"))
    parser.add_argument("--window",choices=("10","30","40","50t"))
    parser.add_argument("--band",choices=BANDS)
    args = parser.parse_args()
    if not (OUT / "preflight_pass.json").exists():
        raise RuntimeError("Preflight checks have not passed")
    if args.step:
        if args.step == "cache":
            from cache_build import build
            build(args.window,args.band)
        else:
            from eval_window import evaluate
            evaluate(args.window,args.band)
        return
    env = dict(os.environ,PYTHONIOENCODING="utf-8",CUDA_CACHE_DISABLE="1",
               MPLCONFIGDIR=str(OUT / "matplotlib_cache"))
    for window in ("10","30","40","50t"):
        out,cache = window_dirs(window)
        env["WINDOW_W_L"] = window
        env.pop("WINDOW_W_MMAP",None)
        for band in BANDS:
            step = ["--window",window,"--band",band]
            run_child("run.py",["--step","cache",*step],env)
            low = band == BANDS[0]
            script = "train_M.py" if low else "train_H.py"
            train_args = ["--band",band] + (["--variant","Lb"] if low else [])
            smoke_tag = f"{band}_Lb_seed2026" if low else f"{band}_seed2026"
            smoke = (cache if low else out) / f"{smoke_tag}_smoke_done.json"
            if not smoke.exists():
                run_child(script,[*train_args,"--seed","2026","--epochs","3"],env,True)
            for seed in SEEDS:
                run_child(script,[*train_args,"--seed",str(seed),"--epochs","40"],env,True)
            run_child("run.py",["--step","eval",*step],env)
    run_child("preflight.py",["hash"],env)


if __name__ == "__main__":
    main()

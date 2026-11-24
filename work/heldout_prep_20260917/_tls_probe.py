"""Diagnose Python TLS handshake failure against ScienceDB (read-only, tiny)."""
from __future__ import annotations

import json
import socket
import ssl
import sys

HOST = "china.scidb.cn"
PORT = 443
URL = "https://china.scidb.cn/download?fileId=c403fc76444e4b9989e4f3ff570f3b3d"

print("python:", sys.version.replace("\n", " "))
print("openssl:", ssl.OPENSSL_VERSION)
print("default verify paths:", ssl.get_default_verify_paths())

paths = ssl.get_default_verify_paths()
print("cafile exists:", paths.cafile, paths.cafile and __import__("os").path.exists(paths.cafile))
print("capath exists:", paths.capath, paths.capath and __import__("os").path.exists(paths.capath))

# 1) raw TCP + explicit handshake, no verification
for label, make_context in (
    ("unverified", lambda: ssl._create_unverified_context()),
    ("verified-default", lambda: ssl.create_default_context()),
    (
        "verified-tls12",
        lambda: ssl.create_default_context(),
    ),
):
    try:
        context = make_context()
        if label == "verified-tls12":
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.maximum_version = ssl.TLSVersion.TLSv1_2
        with socket.create_connection((HOST, PORT), timeout=20) as raw:
            with context.wrap_socket(raw, server_hostname=HOST) as tls:
                print(f"{label}: OK  version={tls.version()}  cipher={tls.cipher()}")
    except Exception as exc:  # noqa: BLE001 - diagnostic
        print(f"{label}: FAIL {type(exc).__name__}: {exc}")

# 2) curl-style: which curl does subprocess find, and does it succeed from Python?
import subprocess  # noqa: E402

for args in (
    ["curl.exe", "--version"],
    ["curl.exe", "-sS", "--max-time", "30", "--range", "0-7", "-o", "-", URL],
):
    try:
        result = subprocess.run(args, capture_output=True, timeout=60, check=False)
        print(
            f"subprocess {args[0]} {' '.join(args[1:4])}: rc={result.returncode} "
            f"stdout_len={len(result.stdout)} stderr={result.stderr[:200]!r}"
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic
        print(f"subprocess {args}: FAIL {type(exc).__name__}: {exc}")

print(json.dumps({"done": True}))

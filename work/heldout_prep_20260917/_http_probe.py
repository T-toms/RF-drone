"""Minimal urllib vs http.client vs curl comparison for the ScienceDB range endpoint."""
from __future__ import annotations

import http.client
import ssl
import subprocess
import urllib.request

URL = "https://china.scidb.cn/download?fileId=c403fc76444e4b9989e4f3ff570f3b3d"
HOST = "china.scidb.cn"
PATH = "/download?fileId=c403fc76444e4b9989e4f3ff570f3b3d"

print("--- A: urllib one-shot, default headers ---")
try:
    with urllib.request.urlopen(URL, timeout=30) as r:
        print("ok", r.status, len(r.read()))
except Exception as exc:
    print("FAIL", type(exc).__name__, exc)

print("--- B: urllib one-shot, Range + UA ---")
req = urllib.request.Request(
    URL,
    headers={"Range": "bytes=0-7", "User-Agent": "curl/8.0", "Accept": "*/*"},
)
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
        print("ok", r.status, r.headers.get("Content-Range"), body.hex())
except Exception as exc:
    print("FAIL", type(exc).__name__, exc)

print("--- C: http.client explicit ---")
try:
    ctx = ssl.create_default_context()
    conn = http.client.HTTPSConnection(HOST, 443, timeout=30, context=ctx)
    conn.request("GET", PATH, headers={"Range": "bytes=0-7", "Host": HOST, "Accept": "*/*"})
    resp = conn.getresponse()
    body = resp.read()
    print("ok", resp.status, resp.getheader("Content-Range"), body.hex())
    conn.close()
except Exception as exc:
    print("FAIL", type(exc).__name__, exc)

print("--- D: curl subprocess ---")
result = subprocess.run(
    ["curl.exe", "-sS", "--max-time", "30", "--range", "0-7", "-o", "-", URL],
    capture_output=True,
    timeout=60,
    check=False,
)
print("rc", result.returncode, "len", len(result.stdout), result.stdout.hex(), result.stderr[:200])

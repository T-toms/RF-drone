"""LAUNCHER1 一键启动器（dronerfa_baseline/open_signal_analyzer.ps1）轻量测试。

用法：
    .venv\\Scripts\\python.exe dronerfa_baseline\\test_open_signal_analyzer.py
    .venv\\Scripts\\python.exe dronerfa_baseline\\test_open_signal_analyzer.py --device cpu

覆盖场景：
    S0 静态检查：UTF-8 BOM、Windows PowerShell 5.1 解析零错误、不含 PS7 专有写法、
       源码中没有任何结束其它进程的调用（不 Stop-Process / taskkill）。
    S1 模拟服务已就绪：退出码 0，不写 PID 文件（不启动第二个服务）。
    S2 真实服务已就绪（若 127.0.0.1:8765 正在运行）：退出码 0，不写 PID 文件，uptime 连续。
    S3 服务在跑但模型加载中，随后就绪：退出码 0，只等待同一服务，不写 PID 文件。
    S4 等待超时：退出码 2，中文提示包含日志位置。
    S5 模型加载失败：退出码 4，中文提示包含服务报错。
    S6 端口被其它程序占用：退出码 6。
    S7 后台启动失败（stub 子进程立刻退出）：退出码 3，中文提示包含 stderr 日志位置。
    S8 真实后台启动（临时端口 + 真实 launch_signal_analyzer.ps1）：
       退出码 0、写出 PID 文件、/api/health 就绪，然后只结束本测试自己创建的那棵进程树。
    S9 端口已在监听但健康检查无响应（服务在忙）：只等待同一个服务，退出码 2，不写 PID 文件。

边界：只用 127.0.0.1 回环；不联网下载、不改防火墙、不安装依赖；
不结束任何不是本测试创建的进程。子进程输出一律重定向到文件，不使用管道。

退出码：0 = 所有可运行场景通过；1 = 有失败；2 = 环境不具备（例如无法创建子进程）。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_DIR.parent
LAUNCHER = BASELINE_DIR / "open_signal_analyzer.ps1"
SERVICE_LAUNCHER = BASELINE_DIR / "launch_signal_analyzer.ps1"
OUTPUT_DIR = BASELINE_DIR / "outputs" / "signal_analyzer_launcher_validation_20260920"
SCRATCH_DIR = BASELINE_DIR / "scratch" / "_launcher_tests"
LIVE_PORT = 8765
LIVE_URL = f"http://127.0.0.1:{LIVE_PORT}"
WINDOWS_POWERSHELL = (
    shutil.which("powershell")
    or r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
)

SCENARIOS: list[dict] = []


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def read_text(path: Path) -> str:
    """尽量读出文本：先 UTF-8（含 BOM），再退回本机 ANSI 代码页。"""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "mbcs"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def port_is_free(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(1.0)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def health_of(url: str, timeout: float = 3.0) -> dict:
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as error:  # noqa: BLE001 - 测试需要把失败当作数据
        return {"_error": f"{type(error).__name__}: {error}"}


def run_hidden(args: list[str], stdout_path: Path, stderr_path: Path, timeout: float) -> dict:
    """启动子进程并把 stdout/stderr 重定向到文件（不使用管道）。"""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        process = subprocess.Popen(
            args, stdout=out, stderr=err, stdin=subprocess.DEVNULL, cwd=str(PROJECT_ROOT)
        )
        try:
            code: int | None = process.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
            code = None
            timed_out = True
    return {"exit_code": code, "timed_out": timed_out}


class FakeHealthServer:
    """只在本机回环上提供 /api/health 的假服务，用于确定性地驱动启动器分支。"""

    def __init__(self, state: dict | None = None, raw_body: bytes | None = None) -> None:
        self.state = state if state is not None else {"ok": True, "state": "loading", "error": ""}
        self.raw_body = raw_body
        self.requests = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "FakeHealthServer":
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_GET(self) -> None:  # noqa: N802 - http.server 接口
                outer.requests += 1
                if outer.raw_body is not None:
                    body = outer.raw_body
                else:
                    body = json.dumps(
                        {
                            "ok": outer.state.get("ok", True),
                            "service": outer.state.get("service", "DroneRFa 任意IQ信号分析界面"),
                            "uptime_seconds": outer.state.get("uptime_seconds", 1.0),
                            "model": {
                                "state": outer.state.get("state", "loading"),
                                "error": outer.state.get("error", ""),
                            },
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:  # 静默
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


class SilentTcpListener:
    """只监听并接受 TCP 连接、从不回应 HTTP 的假服务：模拟“服务在忙，健康检查没响应”。"""

    def __enter__(self) -> "SilentTcpListener":
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self._sock.settimeout(0.5)
        self.port = int(self._sock.getsockname()[1])
        self._stop = False
        self._connections: list[socket.socket] = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop:
            try:
                connection, _ = self._sock.accept()
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                break
            self._connections.append(connection)  # 故意不回任何数据

    def __exit__(self, *exc_info) -> None:
        self._stop = True
        for connection in self._connections:
            try:
                connection.close()
            except OSError:
                pass
        try:
            self._sock.close()
        except OSError:
            pass


def launcher_command(
    port: int,
    log_dir: Path,
    ready_timeout: int,
    device: str = "auto",
    quiet: bool = True,
    extra: list[str] | None = None,
) -> tuple[list[str], Path]:
    pid_file = log_dir / "service.pid"
    args = [
        WINDOWS_POWERSHELL,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(LAUNCHER),
        "-Port",
        str(port),
        "-NoOpenBrowser",
        "-NoMessageBox",
        "-ReadyTimeoutSeconds",
        str(ready_timeout),
        "-Device",
        device,
        "-LogDirectory",
        str(log_dir),
        "-PidFile",
        str(pid_file),
    ]
    if quiet:
        args.append("-Quiet")
    if extra:
        args.extend(extra)
    return args, pid_file


def run_launcher(
    scenario_id: str,
    port: int,
    ready_timeout: int,
    device: str = "auto",
    quiet: bool = True,
    timeout: float = 240.0,
    extra: list[str] | None = None,
) -> dict:
    log_dir = SCRATCH_DIR / scenario_id
    if log_dir.exists():
        shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    args, pid_file = launcher_command(port, log_dir, ready_timeout, device, quiet, extra)
    stdout_path = log_dir / "runner_stdout.log"
    stderr_path = log_dir / "runner_stderr.log"
    started = time.time()
    result = run_hidden(args, stdout_path, stderr_path, timeout)
    result["seconds"] = round(time.time() - started, 2)
    result["log_dir"] = str(log_dir)
    result["pid_file_exists"] = pid_file.is_file()
    result["pid"] = int(pid_file.read_text(encoding="ascii").strip()) if pid_file.is_file() else None
    launcher_log = log_dir / "open_signal_analyzer.log"
    result["launcher_log"] = read_text(launcher_log) if launcher_log.is_file() else ""
    result["runner_stdout"] = read_text(stdout_path) if stdout_path.is_file() else ""
    result["runner_stderr"] = read_text(stderr_path) if stderr_path.is_file() else ""
    result["command"] = args
    return result


def record(scenario_id: str, name: str, status: str, detail: str, **extra) -> dict:
    entry = {"id": scenario_id, "name": name, "status": status, "detail": detail}
    entry.update(extra)
    SCENARIOS.append(entry)
    marker = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}.get(status, status)
    print(f"[{marker}] {scenario_id} {name} - {detail}")
    return entry


def copy_evidence(source_dir: Path, scenario_id: str) -> str:
    """把该场景的日志复制到验证产物目录。"""
    target = OUTPUT_DIR / "scenario_logs" / scenario_id
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(source_dir.glob("*")):
        if item.is_file():
            shutil.copy2(item, target / item.name)
    return str(target)


# ---------------------------------------------------------------------------
# 静态检查（S0 以及子进程能力探测）
# ---------------------------------------------------------------------------
def powershell_parse_check() -> dict:
    quoted = str(LAUNCHER).replace("'", "''")
    command = (
        "$t=$null;$e=$null;"
        f"[void][System.Management.Automation.Language.Parser]::ParseFile('{quoted}',[ref]$t,[ref]$e);"
        "Write-Output ('version=' + $PSVersionTable.PSVersion.ToString());"
        "Write-Output ('errors=' + $e.Count);"
        "foreach($x in $e){ Write-Output ('ERR ' + $x.Extent.StartLineNumber + ': ' + $x.Message) }"
    )
    stdout_path = SCRATCH_DIR / "parse_check" / "stdout.log"
    stderr_path = SCRATCH_DIR / "parse_check" / "stderr.log"
    result = run_hidden(
        [WINDOWS_POWERSHELL, "-NoProfile", "-Command", command], stdout_path, stderr_path, 120
    )
    text = read_text(stdout_path) if stdout_path.is_file() else ""
    errors = [line for line in text.splitlines() if line.startswith("ERR ")]
    version = ""
    for line in text.splitlines():
        if line.startswith("version="):
            version = line.split("=", 1)[1].strip()
    result.update({"text": text, "errors": errors, "version": version})
    return result


def scenario_static() -> None:
    raw = LAUNCHER.read_bytes()
    source = read_text(LAUNCHER)
    problems: list[str] = []
    if not raw.startswith(b"\xef\xbb\xbf"):
        problems.append("缺少 UTF-8 BOM：Windows PowerShell 5.1 会按 ANSI 读取中文")
    if not source.startswith("#"):
        problems.append("脚本首行不是注释行")
    for forbidden in ("Stop-Process", "taskkill", "/IM ", "Get-Process -Name python"):
        if forbidden in source:
            problems.append(f"脚本包含结束进程的写法：{forbidden}")
    if "$Host_:" in source:
        problems.append("包含 PowerShell 7 会拒绝的 \"$Host_:\" 写法，应写成 ${Host_}")
    for required in ("launch_signal_analyzer.ps1", "api/health", "127.0.0.1", "NoOpenBrowser"):
        if required not in source:
            problems.append(f"缺少必需内容：{required}")
    if "${Host_}" not in source:
        problems.append("缺少 ${Host_} 花括号写法")

    parse = powershell_parse_check()
    if parse["errors"]:
        problems.append("PowerShell 5.1 解析报错：" + " | ".join(parse["errors"][:5]))
    if not parse["version"].startswith("5.1"):
        problems.append(f"解析用的不是 Windows PowerShell 5.1：{parse['version']}")

    evidence = copy_evidence(SCRATCH_DIR / "parse_check", "S0") if (SCRATCH_DIR / "parse_check").is_dir() else ""
    if problems:
        record(
            "S0",
            "静态检查（BOM/5.1 解析/无结束进程调用）",
            "failed",
            "；".join(problems),
            powershell_version=parse["version"],
            evidence=evidence,
        )
    else:
        record(
            "S0",
            "静态检查（BOM/5.1 解析/无结束进程调用）",
            "passed",
            f"BOM 正确，PowerShell {parse['version']} 解析 0 错误，未发现结束其它进程的调用",
            powershell_version=parse["version"],
            evidence=evidence,
        )


def probe_start_process() -> dict:
    """探测当前环境能否创建**可用**的分离子进程。

    沙箱/策略可能禁止 Start-Process：要么直接抛 Access is denied，要么进程能创建但
    立刻以 0xC0000142 (STATUS_DLL_INIT_FAILED) 退出。两种情况都算不可用。
    """
    probe_dir = SCRATCH_DIR / "start_process_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_script = probe_dir / "probe.ps1"
    probe_script.write_text(
        "try {\n"
        "    $p = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', 'exit 0') "
        "-WindowStyle Hidden -PassThru -Wait\n"
        "    if ($p.ExitCode -eq 0) {\n"
        "        $text = 'ok 0'\n"
        "    } else {\n"
        "        $text = 'blocked 子进程退出码 ' + $p.ExitCode\n"
        "    }\n"
        "} catch {\n"
        "    $text = 'blocked ' + $_.Exception.Message\n"
        "}\n"
        "Set-Content -LiteralPath (Join-Path $PSScriptRoot 'result.txt') -Value $text -Encoding UTF8\n",
        encoding="utf-8",
    )
    stdout_path = probe_dir / "stdout.log"
    stderr_path = probe_dir / "stderr.log"
    run_hidden(
        [WINDOWS_POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(probe_script)],
        stdout_path,
        stderr_path,
        90,
    )
    result_file = probe_dir / "result.txt"
    text = read_text(result_file).strip() if result_file.is_file() else ""
    return {"available": text == "ok 0", "detail": text or "探测脚本没有写出结果"}


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
def scenario_fake_ready(server: FakeHealthServer) -> None:
    before = server.requests
    result = run_launcher("S1", server.port, ready_timeout=10)
    problems = []
    if result["exit_code"] != 0:
        problems.append(f"退出码 {result['exit_code']}，期望 0")
    if result["pid_file_exists"]:
        problems.append("写入了 PID 文件：说明又启动了一个服务")
    if "不启动第二个服务" not in result["launcher_log"]:
        problems.append("启动器日志没有记录“不启动第二个服务”")
    evidence = copy_evidence(Path(result["log_dir"]), "S1")
    if problems:
        record("S1", "模拟服务已就绪：直接打开、不拉第二个服务", "failed", "；".join(problems), evidence=evidence)
    else:
        record(
            "S1",
            "模拟服务已就绪：直接打开、不拉第二个服务",
            "passed",
            f"退出码 0，用时 {result['seconds']}s，未写 PID 文件，健康检查请求 {server.requests - before} 次",
            evidence=evidence,
        )


def wait_for_live_ready(timeout: float = 60.0, interval: float = 3.0) -> dict | None:
    """等待真实服务可响应；响应不了就返回 None（可能正忙，不作为失败）。"""
    deadline = time.time() + timeout
    while True:
        payload = health_of(LIVE_URL)
        if (
            not payload.get("_error")
            and payload.get("ok")
            and payload.get("model", {}).get("state") == "ready"
        ):
            return payload
        if time.time() >= deadline:
            return None
        time.sleep(interval)


def scenario_live_ready() -> None:
    health = wait_for_live_ready(timeout=60.0)
    if health is None:
        record(
            "S2",
            "真实服务已就绪（127.0.0.1:8765）",
            "skipped",
            "127.0.0.1:8765 在 60 秒内没有响应 /api/health（没有服务，或服务正忙）；"
            "“端口在监听但不响应”的安全行为由 S9 场景确定性覆盖",
        )
        return
    uptime_before = float(health.get("uptime_seconds", 0.0))
    result = run_launcher("S2", LIVE_PORT, ready_timeout=45)
    after_payload = health_of(LIVE_URL)
    uptime_after = (
        None if after_payload.get("_error") else float(after_payload.get("uptime_seconds", 0.0))
    )
    problems = []
    if result["pid_file_exists"]:
        problems.append("写入了 PID 文件：说明又启动了一个服务")
    busy = "已有服务在监听" in result["launcher_log"]
    evidence = copy_evidence(Path(result["log_dir"]), "S2")

    if result["exit_code"] == 0:
        if uptime_after is not None and uptime_after < uptime_before:
            problems.append("真实服务 uptime 倒退：可能被重启")
        detail = (
            f"退出码 0，用时 {result['seconds']}s，未写 PID 文件，"
            f"真实服务 uptime {uptime_before:.1f}s → {uptime_after if uptime_after is None else round(uptime_after, 1)}s（连续未重启）"
        )
        status = "passed" if not problems else "failed"
        if problems:
            detail = "；".join(problems)
        record("S2", "真实服务已就绪（127.0.0.1:8765）", status, detail, evidence=evidence)
        return

    if busy and not result["pid_file_exists"]:
        # 服务在等待期间一直没响应；启动器没有重复启动（安全性质成立），
        # 但“已就绪 → 直接打开”这条路径这次无法验证。
        record(
            "S2",
            "真实服务已就绪（127.0.0.1:8765）",
            "skipped",
            f"真实服务在等待 {result['seconds']}s 内始终没响应健康检查（exit={result['exit_code']}）；"
            "启动器没有重复启动服务，但本场景未能验证“已就绪直接打开”",
            evidence=evidence,
        )
        return

    record(
        "S2",
        "真实服务已就绪（127.0.0.1:8765）",
        "failed",
        "；".join(problems) or f"退出码 {result['exit_code']}，期望 0",
        evidence=evidence,
    )


def scenario_loading_then_ready(server: FakeHealthServer) -> None:
    server.state.update({"ok": True, "state": "loading", "error": ""})

    def flip() -> None:
        time.sleep(4.0)
        server.state.update({"state": "ready"})

    thread = threading.Thread(target=flip, daemon=True)
    thread.start()
    result = run_launcher("S3", server.port, ready_timeout=40)
    thread.join(timeout=5)
    problems = []
    if result["exit_code"] != 0:
        problems.append(f"退出码 {result['exit_code']}，期望 0")
    if result["pid_file_exists"]:
        problems.append("写入了 PID 文件：说明在服务已运行时又启动了第二个服务")
    if "只等待同一个服务" not in result["launcher_log"]:
        problems.append("启动器日志没有记录等待同一个服务")
    if result["seconds"] > 35:
        problems.append(f"耗时 {result['seconds']}s 过长")
    evidence = copy_evidence(Path(result["log_dir"]), "S3")
    if problems:
        record("S3", "服务加载中→就绪：只等待同一个服务", "failed", "；".join(problems), evidence=evidence)
    else:
        record(
            "S3",
            "服务加载中→就绪：只等待同一个服务",
            "passed",
            f"退出码 0，用时 {result['seconds']}s，未写 PID 文件",
            evidence=evidence,
        )


def scenario_timeout(server: FakeHealthServer) -> None:
    server.state.update({"ok": True, "state": "loading", "error": ""})
    result = run_launcher("S4", server.port, ready_timeout=6, quiet=False, timeout=120)
    problems = []
    if result["exit_code"] != 2:
        problems.append(f"退出码 {result['exit_code']}，期望 2（等待超时）")
    log = result["launcher_log"]
    if "exit=2" not in log:
        problems.append("启动器日志没有记录 exit=2")
    if "服务仍未就绪" not in log:
        problems.append("中文提示缺少“服务仍未就绪”")
    if "open_signal_analyzer.log" not in log:
        problems.append("中文提示没有包含日志位置")
    banner_ok = "一键启动失败" in result["runner_stdout"]
    evidence = copy_evidence(Path(result["log_dir"]), "S4")
    detail = f"退出码 {result['exit_code']}，用时 {result['seconds']}s，中文超时提示包含日志位置"
    if problems:
        record("S4", "等待就绪超时：中文提示 + 退出码 2", "failed", "；".join(problems), evidence=evidence)
    else:
        record(
            "S4",
            "等待就绪超时：中文提示 + 退出码 2",
            "passed",
            detail + (f"，控制台横幅中文正常" if banner_ok else "，控制台横幅未按 UTF-8 解出（仅日志核对）"),
            console_banner_ok=banner_ok,
            evidence=evidence,
        )


def scenario_model_failed(server: FakeHealthServer) -> None:
    server.state.update({"ok": True, "state": "failed", "error": "RuntimeError: 测试用模型加载失败"})
    result = run_launcher("S5", server.port, ready_timeout=10)
    problems = []
    if result["exit_code"] != 4:
        problems.append(f"退出码 {result['exit_code']}，期望 4（模型加载失败）")
    log = result["launcher_log"]
    if "exit=4" not in log:
        problems.append("启动器日志没有记录 exit=4")
    if "测试用模型加载失败" not in log:
        problems.append("中文提示没有带回服务报告的模型错误")
    evidence = copy_evidence(Path(result["log_dir"]), "S5")
    if problems:
        record("S5", "模型加载失败：中文提示 + 退出码 4", "failed", "；".join(problems), evidence=evidence)
    else:
        record(
            "S5",
            "模型加载失败：中文提示 + 退出码 4",
            "passed",
            f"退出码 4，用时 {result['seconds']}s，提示包含服务报错与日志位置",
            evidence=evidence,
        )


def scenario_foreign_port() -> None:
    with FakeHealthServer(raw_body=b"<html>another service</html>") as foreign:
        result = run_launcher("S6", foreign.port, ready_timeout=10)
    problems = []
    if result["exit_code"] != 6:
        problems.append(f"退出码 {result['exit_code']}，期望 6（端口被其它程序占用）")
    log = result["launcher_log"]
    if "exit=6" not in log:
        problems.append("启动器日志没有记录 exit=6")
    if "已被其他程序占用" not in log:
        problems.append("中文提示缺少“已被其他程序占用”")
    if result["pid_file_exists"]:
        problems.append("写入了 PID 文件：不应尝试启动第二个服务")
    evidence = copy_evidence(Path(result["log_dir"]), "S6")
    if problems:
        record("S6", "端口被其它程序占用：中文提示 + 退出码 6", "failed", "；".join(problems), evidence=evidence)
    else:
        record(
            "S6",
            "端口被其它程序占用：中文提示 + 退出码 6",
            "passed",
            f"退出码 6，用时 {result['seconds']}s，未启动任何子服务",
            evidence=evidence,
        )


def write_stub_launcher(stub_dir: Path, exit_code: int, delay_seconds: float) -> Path:
    """写一个假“现有启动脚本”，用于测试子进程提前退出/无法启动的分支。"""
    if stub_dir.exists():
        shutil.rmtree(stub_dir, ignore_errors=True)
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub = stub_dir / "stub_launch_signal_analyzer.ps1"
    body = (
        "# 测试用假启动脚本：只记录收到的参数并退出，不启动任何服务。\n"
        "[CmdletBinding()]\n"
        "param(\n"
        "    [int]$Port = 0,\n"
        "    [string]$Host_ = '',\n"
        "    [string]$OutputRoot = '',\n"
        "    [string[]]$AllowedRoot = @(),\n"
        "    [string]$Device = '',\n"
        "    [switch]$NoBrowser,\n"
        "    [switch]$Quiet\n"
        ")\n"
        f"Start-Sleep -Seconds {delay_seconds}\n"
        "$payload = [pscustomobject]@{\n"
        "    port = $Port\n"
        "    host = $Host_\n"
        "    output_root = $OutputRoot\n"
        "    allowed_root = $AllowedRoot\n"
        "    device = $Device\n"
        "    no_browser = [bool]$NoBrowser\n"
        "    quiet = [bool]$Quiet\n"
        "}\n"
        "$payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath "
        "(Join-Path $PSScriptRoot 'stub_params.json') -Encoding UTF8\n"
        # 直接写 stderr，避免 Write-Error 影响 powershell.exe -Command 的退出码传递
        "[Console]::Error.WriteLine('stub: 故意让后台启动失败')\n"
        f"exit {exit_code}\n"
    )
    stub.write_text(body, encoding="utf-8-sig")  # Windows PowerShell 5.1 需要 BOM 才能读中文
    return stub


def scenario_start_failure(start_available: bool) -> None:
    stub_dir = SCRATCH_DIR / "S7_stub"
    stub = write_stub_launcher(stub_dir, exit_code=5, delay_seconds=1)
    stub_port = free_port()
    result = run_launcher(
        "S7",
        stub_port,
        ready_timeout=20,
        extra=["-LauncherScript", str(stub)],
    )
    problems = []
    if result["exit_code"] != 3:
        problems.append(f"退出码 {result['exit_code']}，期望 3（后台启动失败）")
    log = result["launcher_log"]
    if "exit=3" not in log:
        problems.append("启动器日志没有记录 exit=3")
    if "open_signal_analyzer.log" not in log:
        problems.append("中文提示没有包含日志位置")
    branch = "child_exited" if "提前退出" in log else ("start_denied" if "无法在后台启动" in log else "unknown")
    if branch == "unknown":
        problems.append("日志里没有可识别的后台启动失败原因")
    child_exit_code = None
    match = re.search(r"提前退出（退出码 ([^）]+)）", log)
    if match:
        child_exit_code = match.group(1)

    stub_params = stub_dir / "stub_params.json"
    params_source: str | None = None
    direct_exit_code: int | None = None
    command_match = None
    if stub_params.is_file():
        params_source = "launcher_child"
    else:
        # 子进程被环境阻止时，用启动器日志里记录的同一条命令行直接执行一次，
        # 仍然可以核对参数转发（数组、中文路径、开关）是否正确。
        command_match = re.search(r"子进程命令行：(.*)", log)
        if command_match:
            direct_dir = SCRATCH_DIR / "S7_direct"
            direct_dir.mkdir(parents=True, exist_ok=True)
            direct = run_hidden(
                [
                    WINDOWS_POWERSHELL,
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    command_match.group(1).strip(),
                ],
                direct_dir / "stdout.log",
                direct_dir / "stderr.log",
                90,
            )
            direct_exit_code = direct["exit_code"]
            if stub_params.is_file():
                params_source = "reexecuted_command"
        else:
            problems.append("启动器日志没有记录子进程命令行，无法核对参数转发")

    params_ok: bool | None = None
    if stub_params.is_file():
        payload = json.loads(stub_params.read_text(encoding="utf-8-sig"))
        expected_roots = ["F:/DroneRFa数据集", "C:/Users/Administrator/Desktop/数据集处理"]
        params_ok = (
            payload["port"] == stub_port
            and payload["allowed_root"] == expected_roots
            and payload["output_root"] == "F:/DroneRFa数据集/analysis_outputs"
            and payload["no_browser"] is True
            and payload["quiet"] is True
        )
        if not params_ok:
            problems.append(f"子进程收到的参数不符合预期：{payload}")
    elif command_match:
        problems.append("子进程没有写出参数文件：命令行可能无法正确绑定参数")

    # 0xC0000142 (STATUS_DLL_INIT_FAILED) 表示当前环境拒绝真正创建子进程，
    # 此时子进程不会执行 stub 代码，参数核对改由“复跑同一条命令行”完成。
    environment_limited = child_exit_code in (None, "-1073741502", "未知")
    if branch == "child_exited" and not environment_limited and child_exit_code != "5":
        problems.append(f"stub 子进程退出码 {child_exit_code}，期望 5")
    if branch == "child_exited" and not environment_limited and "stub: 故意让后台启动失败" not in log:
        problems.append("启动器没有把子进程 stderr 末尾写进中文提示")
    if direct_exit_code is not None and direct_exit_code != 5:
        problems.append(f"复跑子进程命令行退出码 {direct_exit_code}，期望 5")

    if branch == "start_denied":
        status = "passed" if not problems else "failed"
        detail = (
            "退出码 3，中文提示包含日志位置；当前环境禁止 Start-Process，"
            "走到“无法在后台启动分析服务”分支"
        )
    else:
        status = "passed" if not problems else "failed"
        detail = (
            f"退出码 3，中文提示包含日志位置；分支={branch}，子进程退出码={child_exit_code}，"
            f"参数核对={params_ok}（来源={params_source}），直接复跑命令行退出码={direct_exit_code}"
        )

    evidence = copy_evidence(Path(result["log_dir"]), "S7")
    if stub_params.is_file():
        shutil.copy2(stub_params, Path(evidence) / "stub_params.json")
    record(
        "S7",
        "后台启动失败：中文提示 + 退出码 3",
        status,
        detail if not problems else "；".join(problems),
        branch=branch,
        child_exit_code=child_exit_code,
        child_process_environment_limited=environment_limited,
        child_parameters_ok=params_ok,
        child_parameters_source=params_source,
        direct_rerun_exit_code=direct_exit_code,
        evidence=evidence,
    )


def scenario_silent_listener() -> None:
    """端口在监听但健康检查无响应：绝不能拉起第二个服务。"""
    with SilentTcpListener() as silent:
        result = run_launcher("S9", silent.port, ready_timeout=8, timeout=180)
    problems = []
    if result["exit_code"] != 2:
        problems.append(f"退出码 {result['exit_code']}，期望 2（等待超时）")
    if result["pid_file_exists"]:
        problems.append("写入了 PID 文件：说明在端口已有服务时又启动了第二个服务")
    log = result["launcher_log"]
    if "已有服务在监听" not in log:
        problems.append("启动器日志没有记录“已有服务在监听，只等待同一个服务”")
    if "exit=2" not in log:
        problems.append("启动器日志没有记录 exit=2")
    evidence = copy_evidence(Path(result["log_dir"]), "S9")
    if problems:
        record("S9", "端口在监听但健康检查无响应：只等待、不重复启动", "failed", "；".join(problems), evidence=evidence)
    else:
        record(
            "S9",
            "端口在监听但健康检查无响应：只等待、不重复启动",
            "passed",
            f"退出码 2，用时 {result['seconds']}s，未写 PID 文件，中文提示说明已有服务在监听",
            evidence=evidence,
        )


def stop_process_tree(pid: int, log_dir: Path) -> dict:
    """只结束本测试自己创建的那棵进程树（先子后父）。"""
    stdout_path = log_dir / "taskkill_stdout.log"
    stderr_path = log_dir / "taskkill_stderr.log"
    result = run_hidden(
        ["taskkill", "/PID", str(pid), "/T", "/F"], stdout_path, stderr_path, 60
    )
    result["stdout"] = read_text(stdout_path) if stdout_path.is_file() else ""
    result["stderr"] = read_text(stderr_path) if stderr_path.is_file() else ""
    return result


def scenario_real_start(start_available: bool, device: str) -> None:
    if not start_available:
        record(
            "S8",
            "真实后台启动（临时端口）→ 就绪 → 只停自己创建的进程",
            "skipped",
            "当前执行环境禁止 Start-Process（沙箱），无法验证真实启动路径；"
            "需在普通桌面会话中重跑本测试",
        )
        return

    port = free_port()
    result = run_launcher("S8", port, ready_timeout=240, device=device, timeout=420)
    problems = []
    if result["exit_code"] != 0:
        problems.append(f"退出码 {result['exit_code']}，期望 0")
    if not result["pid_file_exists"]:
        problems.append("没有写出 PID 文件：无法确认后台服务进程")
    health = health_of(f"http://127.0.0.1:{port}")
    if health.get("_error") or health.get("model", {}).get("state") != "ready":
        problems.append(f"临时端口上的服务未就绪：{health}")
    devices = health.get("model", {}).get("environment", {}).get("device") if not health.get("_error") else None
    if result["launcher_log"] and "服务已就绪" not in result["launcher_log"]:
        problems.append("启动器日志没有记录服务已就绪")

    kill_info = None
    if result["pid"]:
        kill_info = stop_process_tree(result["pid"], Path(result["log_dir"]))
        deadline = time.time() + 30
        while time.time() < deadline and not port_is_free(port):
            time.sleep(1.0)
        if not port_is_free(port):
            problems.append(f"结束进程树后端口 {port} 仍被占用")

    evidence = copy_evidence(Path(result["log_dir"]), "S8")
    detail = (
        f"退出码 0，用时 {result['seconds']}s，临时端口 {port} 上服务就绪（device={devices}），"
        f"随后只结束本测试创建的进程树 pid={result['pid']}"
    )
    if problems:
        record("S8", "真实后台启动（临时端口）→ 就绪 → 只停自己创建的进程", "failed", "；".join(problems), evidence=evidence)
    else:
        record(
            "S8",
            "真实后台启动（临时端口）→ 就绪 → 只停自己创建的进程",
            "passed",
            detail,
            port=port,
            device=devices,
            killed=kill_info,
            evidence=evidence,
        )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="LAUNCHER1 一键启动器测试")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-real-start", action="store_true", help="跳过真实后台启动场景")
    options = parser.parse_args()

    if not LAUNCHER.is_file():
        print(f"找不到启动器：{LAUNCHER}")
        return 2
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    scenario_static()
    probe = probe_start_process()
    start_available = bool(probe["available"])

    with FakeHealthServer({"ok": True, "state": "ready"}) as server:
        scenario_fake_ready(server)
    scenario_live_ready()
    with FakeHealthServer({"ok": True, "state": "loading"}) as server:
        scenario_loading_then_ready(server)
    with FakeHealthServer({"ok": True, "state": "loading"}) as server:
        scenario_timeout(server)
    with FakeHealthServer({"ok": True, "state": "loading"}) as server:
        scenario_model_failed(server)
    scenario_foreign_port()
    scenario_start_failure(start_available)
    scenario_silent_listener()
    if options.skip_real_start:
        record("S8", "真实后台启动（临时端口）", "skipped", "命令行要求跳过")
    else:
        scenario_real_start(start_available, options.device)

    failures = [entry for entry in SCENARIOS if entry["status"] == "failed"]
    skipped = [entry for entry in SCENARIOS if entry["status"] == "skipped"]
    report = {
        "task": "LAUNCHER1 一键启动器验证",
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "launcher": str(LAUNCHER),
        "launch_service_script": str(SERVICE_LAUNCHER),
        "powershell": WINDOWS_POWERSHELL,
        "environment": {
            "start_process_available": start_available,
            "start_process_detail": probe["detail"],
            "device_parameter": options.device,
        },
        "scenarios": SCENARIOS,
        "passed": len([entry for entry in SCENARIOS if entry["status"] == "passed"]),
        "failed": len(failures),
        "skipped": len(skipped),
    }
    report_path = OUTPUT_DIR / "launcher_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("")
    print(f"通过 {report['passed']} / 失败 {report['failed']} / 跳过 {report['skipped']}")
    print(f"报告：{report_path}")
    if failures:
        return 1
    if skipped and not start_available:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

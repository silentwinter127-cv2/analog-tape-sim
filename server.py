"""
Start / stop / check the WebUI server.

Written after two incidents caused by the absence of it: servers were restarted by
hand several times, and an attempt to stop one used `Get-Process python | Stop-Process`,
which killed every Python process on the machine including ones that had nothing to
do with this project.  A start/stop tool that records the PID it created is the
minimum needed to stop *its own* process and nothing else.

Usage (from the project root):

    python server.py start            # port 8947, opens nothing
    python server.py start --open     # also open the browser
    python server.py start --port 9000 --data-dir runs
    python server.py stop
    python server.py restart
    python server.py status
    python server.py logs             # tail the log of the running server

The PID and log live in `.server/`.  `stop` only ever signals the recorded PID, and
it verifies that the PID is still this server before doing so -- a PID can be reused
by an unrelated process after a crash, and signalling the wrong process is exactly
the failure this tool exists to prevent.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(HERE, ".server")
PID_FILE = os.path.join(STATE_DIR, "pid.json")
LOG_FILE = os.path.join(STATE_DIR, "server.log")
DEFAULT_PORT = 8947


def _ensure_state_dir() -> None:
    os.makedirs(STATE_DIR, exist_ok=True)


def _read_state() -> dict | None:
    try:
        with open(PID_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_state(state: dict) -> None:
    _ensure_state_dir()
    with open(PID_FILE, "w", encoding="utf-8") as fh:
        json.dump(state, fh)


def _clear_state() -> None:
    try:
        os.remove(PID_FILE)
    except OSError:
        pass


def _pid_exists(pid: int) -> bool:
    """Is a process with this PID running at all?"""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", "PID eq %d" % pid, "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=20)
            return str(pid) in (out.stdout or "")
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _alive(pid: int) -> bool:
    """
    Is the recorded PID still our server?

    `wmic` was the obvious way to read a command line on Windows and is now removed
    from current builds, so it cannot be the only method -- when it failed the check
    reported a healthy server as "gone" while it was answering on its port.

    Falls back to `tasklist` alone, and then to the PID existing *and* its port
    answering.  That last combination is what actually matters: the worst outcome to
    avoid is signalling a recycled PID that belongs to an unrelated process, and a
    live PID whose recorded port is serving HTTP is this server for practical
    purposes.
    """
    if not _pid_exists(pid):
        return False
    if os.name == "nt":
        try:
            wmi = subprocess.run(
                ["wmic", "process", "where", "ProcessId=%d" % pid,
                 "get", "CommandLine", "/value"],
                capture_output=True, text=True, timeout=20)
            cmd = wmi.stdout or ""
            if cmd.strip():
                return ("atsim.web.server" in cmd) or ("atsim\\web\\server" in cmd)
        except Exception:
            pass
    state = _read_state()
    if state and state.get("pid") == pid:
        return _probe(int(state.get("port", DEFAULT_PORT)))
    return True


def _probe(port: int, timeout: float = 4.0) -> bool:
    """Is something answering HTTP on this port?"""
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/" % port, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def cmd_start(args: argparse.Namespace) -> int:
    state = _read_state()
    if state and _alive(state.get("pid", -1)):
        print("already running on port %s (pid %s)" % (state["port"], state["pid"]))
        print("   http://127.0.0.1:%s/" % state["port"])
        return 0
    if _probe(args.port):
        print("port %d already answers HTTP, but no PID file of ours matches."
              % args.port)
        print("   refusing to start a second server; use `status` to see it")
        return 1

    _ensure_state_dir()
    log = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
    log.write("\n=== start %s  port %d ===\n"
              % (time.strftime("%Y-%m-%d %H:%M:%S"), args.port))
    cmd = [sys.executable, "-u", "-m", "atsim.web.server",
           "--port", str(args.port), "--host", args.host]
    if args.data_dir:
        cmd += ["--data-dir", args.data_dir]
    if args.open:
        cmd.append("--open")

    creation = 0
    if os.name == "nt":
        # own process group, so a later stop can address it precisely
        creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=creation)
    _write_state({"pid": proc.pid, "port": args.port, "host": args.host,
                  "data_dir": args.data_dir or "",
                  "started": time.time()})

    for _ in range(40):
        time.sleep(0.5)
        if _probe(args.port):
            print("started: pid %d on port %d" % (proc.pid, args.port))
            print("   http://127.0.0.1:%d/" % args.port)
            print("   log: %s" % LOG_FILE)
            return 0
        if proc.poll() is not None:
            print("server exited immediately (code %s) -- last lines of %s:"
                  % (proc.returncode, LOG_FILE))
            _tail(12)
            _clear_state()
            return 1
    print("server started (pid %d) but is not answering on port %d after 20s"
          % (proc.pid, args.port))
    print("   check the log: %s" % LOG_FILE)
    return 1


def cmd_stop(args: argparse.Namespace) -> int:
    state = _read_state()
    if not state:
        print("no PID file -- nothing recorded as running")
        return 0
    pid, port = state.get("pid", -1), state.get("port", DEFAULT_PORT)
    if not _alive(pid):
        print("recorded pid %s is not a running server; clearing stale PID file" % pid)
        _clear_state()
        return 0

    if os.name == "nt":
        # /T also ends the child python process; sending it to the PID only would
        # leave the interpreter running
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, text=True, timeout=30)
    else:
        os.kill(pid, signal.SIGTERM)

    for _ in range(20):
        time.sleep(0.5)
        if not _alive(pid):
            break
    _clear_state()
    still = _probe(port)
    print("stopped pid %s (port %s)" % (pid, port))
    if still:
        print("   note: port %s still answers -- something else is serving it" % port)
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    cmd_stop(args)
    time.sleep(1.0)
    return cmd_start(args)


def cmd_status(args: argparse.Namespace) -> int:
    state = _read_state()
    if not state:
        print("not running (no PID file)")
    else:
        pid, port = state.get("pid", -1), state.get("port", DEFAULT_PORT)
        running = _alive(pid)
        print("recorded : pid %s on port %s" % (pid, port))
        print("process  : %s" % ("alive" if running else "gone (stale PID file)"))
        print("http     : %s" % ("answering" if _probe(port) else "not answering"))
        if running and _probe(port):
            print("url      : http://127.0.0.1:%s/" % port)
    # anything else on the usual ports, that this tool did not start
    for port in (DEFAULT_PORT, 8941):
        if _probe(port) and (not state or state.get("port") != port):
            print("note: port %d answers HTTP but was not started by this tool" % port)
    return 0


def _tail(lines: int) -> None:
    try:
        with open(LOG_FILE, encoding="utf-8", errors="replace") as fh:
            rows = fh.readlines()
        for row in rows[-lines:]:
            print("   " + row.rstrip())
    except OSError as exc:
        print("   (no log: %s)" % exc)


def cmd_logs(args: argparse.Namespace) -> int:
    if not os.path.exists(LOG_FILE):
        print("no log yet at %s" % LOG_FILE)
        return 0
    _tail(args.lines)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="start/stop the WebUI server without touching other processes")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--port", type=int, default=DEFAULT_PORT)
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--data-dir", default="")
        p.add_argument("--open", action="store_true", help="open a browser")

    common(sub.add_parser("start"))
    common(sub.add_parser("restart"))
    sub.add_parser("stop")
    sub.add_parser("status")
    pl = sub.add_parser("logs")
    pl.add_argument("--lines", type=int, default=30)

    args = ap.parse_args(argv)
    if args.cmd == "start":
        return cmd_start(args)
    if args.cmd == "stop":
        return cmd_stop(args)
    if args.cmd == "restart":
        return cmd_restart(args)
    if args.cmd == "status":
        return cmd_status(args)
    return cmd_logs(args)


if __name__ == "__main__":
    raise SystemExit(main())

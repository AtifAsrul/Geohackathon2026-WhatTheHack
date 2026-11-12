"""Local terminal monitoring. Uses only the standard library; never submits jobs."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def elapsed(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def stage_line(status):
    """A stage-position bar, deliberately not an estimate of time remaining."""
    raw_stages = status.get("stages", [])
    stages = [s for s in raw_stages if isinstance(s, dict) and s.get("id")] if isinstance(raw_stages, list) else []
    stage = status.get("stage", "waiting")
    ids = [s["id"] for s in stages]
    n = len(ids)
    if status.get("state") == "done":
        return f"[{'#' * n}] {n}/{n} stages finished | {status.get('message', 'Finished')}"
    if stage not in ids:
        return f"{stage}: {status.get('message', 'Waiting for interpretation')}"
    index = ids.index(stage)
    marker = "!" if status.get("state") == "error" else ">"
    bar = "#" * index + marker + "-" * (n - index - 1)
    label = stages[index].get("label", stage)
    return f"[{bar}] Stage {index + 1}/{n}: {label} | {status.get('message', '')}"


class Monitor:
    """Read existing status/log files. Does not import or initialize the app."""
    def __init__(self, workspace, emit=print):
        self.workspace = Path(workspace)
        self.emit = emit
        self.signature = None
        self.run_id = None
        self.offsets = {}
        self.heads = {}
        self.status = {}
        # Do not replay historical logs when attaching to a running app.
        for path in self.log_files():
            try:
                self.offsets[path] = path.stat().st_size
                with path.open("rb") as stream:
                    self.heads[path] = stream.read(64)
            except OSError:
                pass

    def log_files(self):
        folder = self.workspace / "logs"
        return sorted(p for pattern in ("pipeline.log", "gpu_job_*.log", "sdip_*.log")
                      for p in folder.glob(pattern))

    def poll(self):
        try:
            status = json.loads((self.workspace / "status.json").read_text(encoding="utf-8"))
            if isinstance(status, dict):
                self.status = status
        except (OSError, ValueError):
            pass  # Atomic status-file replacement may race a read; try next poll.
        status = self.status
        run_id = status.get("started")
        if run_id is not None and run_id != self.run_id:
            if self.run_id is not None:
                self.offsets[self.workspace / "logs" / "pipeline.log"] = 0
            self.run_id = run_id
        signature = (status.get("stage"), status.get("state"), status.get("message"), run_id)
        if status and signature != self.signature:
            self.emit(f"[progress {time.strftime('%H:%M:%S')}] {stage_line(status)}")
            self.signature = signature
        for path in self.log_files():
            self.follow(path)

    def follow(self, path):
        try:
            size = path.stat().st_size
            with path.open("rb") as stream:
                head = stream.read(64)
                offset = self.offsets.get(path, 0)
                old_head = self.heads.get(path, b"")
                if size < offset or (old_head and not head.startswith(old_head)):
                    offset = 0
                start = max(offset, size - 32768)  # Bound reads of large logs.
                stream.seek(start)
                data = stream.read(32768)
                self.offsets[path] = stream.tell()
                self.heads[path] = head
            if start > offset:
                self.emit(f"[{path.name}] ... older output omitted; full output remains in the log")
            for line in data.decode("utf-8", errors="replace").replace("\r", "\n").splitlines():
                if line.strip():
                    self.emit(f"[{path.name}] {line}")
        except OSError:
            pass

    def heartbeat(self):
        status = self.status
        if status.get("state") == "running":
            try:
                age = elapsed(time.time() - float(status.get("updated", time.time())))
                runtime = elapsed(time.time() - float(status.get("started", time.time())))
            except (ValueError, TypeError):
                age, runtime = "unknown", "unknown"
            return f"Interpretation reports running: {status.get('stage', '?')} | elapsed {runtime} | status updated {age} ago"
        return "Server/monitor active; waiting for interpretation" if not status else f"Interpretation state: {status.get('state', 'unknown')}"


def run_command(command, label, heartbeat=15, monitor=None):
    started = last = time.monotonic()
    print(f"[START {time.strftime('%H:%M:%S')}] {label}", flush=True)
    try:
        child = subprocess.Popen(command)
    except OSError as exc:
        print(f"[FAILED] {label}: {exc}", flush=True)
        return 1
    try:
        while True:
            if monitor:
                try:
                    monitor.poll()
                except Exception as exc:
                    print(f"[MONITOR WARNING] {exc}; command continues. Use the log files directly.", flush=True)
                    monitor = None
            try:
                code = child.wait(timeout=min(1, heartbeat))
                break
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                if now - last >= heartbeat:
                    detail = f" | {monitor.heartbeat()}" if monitor else " | command still running; see output above"
                    print(f"[ACTIVE {time.strftime('%H:%M:%S')}] {label} | elapsed {elapsed(now - started)}{detail}", flush=True)
                    last = now
    except KeyboardInterrupt:
        # Only this wrapper's child; never signal other processes or Slurm jobs.
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        print(f"[STOPPED] {label} | elapsed {elapsed(time.monotonic() - started)}", flush=True)
        return 130
    if monitor:
        try:
            monitor.poll()
        except Exception as exc:
            print(f"[MONITOR WARNING] {exc}", flush=True)
    print(f"[{'OK' if code == 0 else 'FAILED'}] {label} | elapsed {elapsed(time.monotonic() - started)} | exit {code}", flush=True)
    return code if code >= 0 else 128 - code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "watch"))
    parser.add_argument("--label", default="Command")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--heartbeat", type=float, default=15)
    argv = sys.argv[1:]
    boundary = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:boundary])
    command = argv[boundary + 1:]
    if args.heartbeat <= 0:
        parser.error("--heartbeat must be positive")
    workspace = args.workspace or Path(os.environ.get("COPILOT_WORKSPACE", Path(__file__).resolve().parent.parent / "workspace"))
    monitor = Monitor(workspace, emit=lambda msg: print(msg, flush=True)) if args.mode == "watch" or args.workspace else None
    if args.mode == "run":
        if not command:
            parser.error("run needs a command after --")
        return run_command(command, args.label, args.heartbeat, monitor)
    print(f"Read-only monitor: {workspace}\nBars show stage position, not time remaining. Ctrl+C stops only this monitor.", flush=True)
    last = time.monotonic()
    try:
        while True:
            monitor.poll()
            if time.monotonic() - last >= args.heartbeat:
                print(f"[ACTIVE {time.strftime('%H:%M:%S')}] {monitor.heartbeat()}", flush=True)
                last = time.monotonic()
            time.sleep(min(1, args.heartbeat))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())

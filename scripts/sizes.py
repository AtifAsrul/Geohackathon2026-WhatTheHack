"""Bounded, read-only app-directory size sampling; no file contents are read."""
import os
import stat
import time
from pathlib import Path


def size_text(value):
    return f"{value / 1e9:.3f} GB" if abs(value) >= 1e9 else f"{value / 1e6:.2f} MB"


def scan_tree(root, max_entries=4000, budget_seconds=0.15):
    """Return a lower bound if a directory is too large/slow or unreadable.

    Never follow symlinks/junctions into shared input data or other locations.
    st_blocks (when supplied by the filesystem) distinguishes sparse allocation.
    """
    result = {"logical": 0, "allocated": 0 if hasattr(os.stat_result, "st_blocks") else None,
              "files": 0, "complete": True}
    root = Path(root)
    if root.is_symlink() or (hasattr(root, "is_junction") and root.is_junction()):
        result["complete"] = False
        return result
    pending = [root]
    visited, started = 0, time.monotonic()
    while pending:
        try:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    visited += 1
                    if visited > max_entries or time.monotonic() - started > budget_seconds:
                        result["complete"] = False
                        return result
                    try:
                        if entry.is_symlink() or (hasattr(entry, "is_junction") and entry.is_junction()):
                            continue
                        info = entry.stat(follow_symlinks=False)
                        if stat.S_ISDIR(info.st_mode):
                            pending.append(Path(entry.path))
                        elif stat.S_ISREG(info.st_mode):
                            result["logical"] += info.st_size
                            blocks = getattr(info, "st_blocks", None)
                            if blocks is None:
                                result["allocated"] = None
                            elif result["allocated"] is not None:
                                result["allocated"] += blocks * 512
                            result["files"] += 1
                    except OSError:
                        result["complete"] = False
        except FileNotFoundError:
            # Missing top-level output folders are normal before the first run.
            if root.exists():
                result["complete"] = False
        except OSError:
            result["complete"] = False
    return result


class SizeMonitor:
    def __init__(self, workspace, models, emit=print, interval=30):
        self.paths = {"models": Path(models), "stores": Path(workspace) / "stores",
                      "interpretation": Path(workspace) / "interp", "logs": Path(workspace) / "logs"}
        self.emit = emit
        self.interval = interval
        self.last = None
        self.previous = {}

    def poll(self):
        now = time.monotonic()
        if self.last is not None and now - self.last < self.interval:
            return
        self.last = now
        for label, path in self.paths.items():
            current = scan_tree(path)
            sampled = time.monotonic()
            previous = self.previous.get(label)
            prefix = "" if current["complete"] else ">="
            detail = f"{prefix}{size_text(current['logical'])} logical; {current['files']} files"
            if current["allocated"] is not None:
                detail += f"; {prefix}{size_text(current['allocated'])} allocated"
            if previous and current["complete"] and previous[1]["complete"]:
                seconds = max(sampled - previous[0], 0.001)
                rate = (current["logical"] - previous[1]["logical"]) / 1e6 / seconds
                detail += f"; {rate:+.2f} MB/s net file-size growth"
                if current["allocated"] is not None and previous[1]["allocated"] is not None:
                    rate = (current["allocated"] - previous[1]["allocated"]) / 1e6 / seconds
                    detail += f"; {rate:+.2f} MB/s net allocation growth"
            elif not current["complete"]:
                detail += "; partial scan, rate unavailable"
            else:
                detail += "; rate available at next sample"
            self.emit(f"[SIZE {time.strftime('%H:%M:%S')}] {label}: {detail}")
            self.previous[label] = (sampled, current)


def safe_sample(monitor):
    if monitor:
        try:
            monitor.poll()
        except Exception as exc:
            # Diagnostics must not interrupt an install, server, or pipeline.
            print(f"[SIZE WARNING] {exc}", flush=True)

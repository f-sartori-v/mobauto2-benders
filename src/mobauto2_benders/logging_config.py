from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import TextIO


class _StdoutTee:
    def __init__(self, a: TextIO, b: TextIO):
        self.a = a
        self.b = b

    def write(self, s: str) -> int:
        try:
            self.a.write(s)
        except Exception:
            pass
        try:
            self.b.write(s)
        except Exception:
            pass
        try:
            self.a.flush()
        except Exception:
            pass
        try:
            self.b.flush()
        except Exception:
            pass
        return len(s)

    def flush(self) -> None:
        try:
            self.a.flush()
        except Exception:
            pass
        try:
            self.b.flush()
        except Exception:
            pass


def setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s %(levelname)s | %(name)s | %(message)s"
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format=fmt)

    if str(level).upper() == "DEBUG":
        try:
            out_dir = Path("Report")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = out_dir / f"benders_debug_{ts}.txt"
            # Open one shared file for both logging and print tee
            f = open(log_path, mode="w", encoding="utf-8", buffering=1)
            # Attach a logging handler that writes to this file
            fh = logging.StreamHandler(f)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(fmt))
            root = logging.getLogger()
            root.addHandler(fh)
            # Tee stdout (print) to the same file, while keeping console output
            sys.stdout = _StdoutTee(sys.stdout, f)
            # Notify destination (goes to both console and file)
            print(f"[LOG] Writing DEBUG logs and prints to {log_path}")
        except Exception:
            # Fail-safe: continue without file logging if any error occurs
            pass


__all__ = ["setup_logging"]

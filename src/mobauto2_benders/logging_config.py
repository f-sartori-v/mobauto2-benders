from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)


__all__ = ["setup_logging"]


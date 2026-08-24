# # utils/logger.py

import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np

from config.settings import LOG_PATH


# ── Run-level paths (fixed once per process) ─────────────────────

_RUN_TAG: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _sibling(name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(LOG_PATH)), name)


RUN_LOG_PATH: str = _sibling(f"distraction_log_{_RUN_TAG}.txt")



# LOGGER SETUP


def setup_logger(name: str = "gadget_monitor") -> logging.Logger:
    os.makedirs(os.path.dirname(RUN_LOG_PATH), exist_ok=True)

    logger = logging.getLogger(name)
    # FIX (file descriptor leak): logger.handlers.clear() only drops the
    # references — it never closes the underlying FileHandler/StreamHandler,
    # so every call to setup_logger() (once per video) leaked an open file
    # descriptor for the life of the worker process. Explicitly close()
    # each handler before removing it.
    for h in list(logger.handlers):
        try:
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(RUN_LOG_PATH, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(message)s"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False

    print(f"[Logger] Text log : {RUN_LOG_PATH}")

    return logger



# HELPERS


def video_timestamp(seconds: float) -> str:
    t  = int(seconds)
    hh = t // 3600
    mm = (t % 3600) // 60
    ss = t % 60
    return f"[{hh:02d}:{mm:02d}:{ss:02d}]"


# PUBLIC API


def log_distraction(
    logger,
    video_time: float,
    event:      str,
    severity:   str = "CRITICAL",
    frame:      Optional[np.ndarray] = None,   # kept for call-site compatibility
) -> None:
    """Write a timestamped line to the plain-text .txt log."""
    ts  = video_timestamp(video_time)
    msg = f"{ts} {event}  [{severity}]"
    logger.info(msg)
    for h in logger.handlers:
        h.flush()


def finalize_report() -> None:
    pass
"""
Structured JSON logger for the agentic loop.
Each step (perceive/reason/act/reflect) emits one log line to stdout and a file.
"""

import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path("agent_loop.log")


def _write(entry: dict) -> None:
    line = json.dumps(entry)
    print(line, flush=True)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


def log_step(
    iteration: int,
    step: str,
    input_summary: str,
    output_summary: str,
    latency_ms: float,
    error=None,
) -> None:
    """
    Emit one structured log entry.

    Fields: timestamp, iteration, step, input_summary,
            output_summary, latency_ms, error.
    """
    _write({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration,
        "step": step,
        "input_summary": input_summary[:200],
        "output_summary": output_summary[:200],
        "latency_ms": round(latency_ms, 2),
        "error": error,
    })

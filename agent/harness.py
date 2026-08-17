"""
Harness: retry logic, fallback strategies, and loop guardrails.
Wraps the core loop functions so failures are handled gracefully.
"""

import json
import time
import random
import logging
import warnings

logger = logging.getLogger(__name__)


# ── Retry with exponential backoff + jitter ───────────────────────────────────

def call_with_retry(fn, *args, max_retries: int = 3, base_delay: float = 1.0, **kwargs):
    """
    Call fn(*args, **kwargs) with exponential backoff on failure.

    Handles: rate-limit errors, timeouts, and JSON parse errors.
    On final failure raises the last exception.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)

        except json.JSONDecodeError as e:
            # LLM returned unparseable output → retry with same args
            last_error = e
            logger.warning(f"[Retry {attempt+1}] JSON parse error: {e}")

        except Exception as e:
            error_str = str(e).lower()
            # Rate-limit or timeout → always retry
            if any(k in error_str for k in ("rate limit", "timeout", "429", "503")):
                last_error = e
                logger.warning(f"[Retry {attempt+1}] Transient error: {e}")
            else:
                raise   # non-retryable, bubble up immediately

        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
        time.sleep(delay)

    raise last_error


# ── Fallback wrappers ─────────────────────────────────────────────────────────

def safe_reason(reason_fn, observation, memory, client, model, max_retries, base_delay):
    """
    Wraps reason() with retry.
    Fallback: returns a default 'break_goal' plan if all retries fail.
    """
    try:
        return call_with_retry(
            reason_fn, observation, memory, client, model,
            max_retries=max_retries, base_delay=base_delay
        )
    except Exception as e:
        logger.error(f"[Fallback] reason() failed: {e}. Using default plan.")
        return {
            "action": "break_goal",
            "parameters": {"goal": observation.get("goal", "")},
            "reasoning": "Fallback: LLM unavailable.",
        }


def safe_act(act_fn, plan, tools, client, model, max_retries, base_delay, last_steps=None):
    try:
        return call_with_retry(
            act_fn, plan, tools, client, model, last_steps,
            max_retries=max_retries, base_delay=base_delay
        )
    except Exception as e:
        logger.error(f"[Fallback] act() failed: {e}. Returning error observation.")
        return {
            "steps": [],
            "error": str(e),
            "status": "TOOL_ERROR",
        }


# ── Token budget tracker ──────────────────────────────────────────────────────

class TokenBudget:
    """Tracks cumulative token usage and warns when threshold is exceeded."""

    def __init__(self, limit: int = 10_000):
        self.limit = limit
        self.used = 0

    def add(self, tokens: int) -> None:
        self.used += tokens
        if self.used > self.limit:
            warnings.warn(
                f"[TokenBudget] Cumulative tokens {self.used} exceeded limit {self.limit}.",
                stacklevel=2,
            )

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

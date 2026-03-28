"""
retry.py - Retry utility module for LinkedIn AI Poster.

Provides exponential backoff with optional jitter for both async and sync
functions. Implemented from scratch using asyncio.sleep / time.sleep;
no third-party retry library (e.g. tenacity) is used.

Usage examples
--------------
Async (decorator):
    @retry_decorator(RetryConfig(max_attempts=5))
    async def call_api() -> dict: ...

Async (explicit):
    result = await with_retry(call_api, RetryConfig(), arg1, kwarg=val)

Sync (decorator):
    @retry_sync_decorator(RetryConfig(base_delay=0.5))
    def fetch_data() -> bytes: ...

Sync (explicit):
    result = with_retry_sync(fetch_data, RetryConfig(), arg1, kwarg=val)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
F = TypeVar("F", bound=Callable[..., Any])
AsyncFunc = Callable[..., Coroutine[Any, Any, Any]]
SyncFunc = Callable[..., Any]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RetryConfig:
    """Configuration for retry behaviour.

    Attributes
    ----------
    max_attempts:
        Total number of attempts (including the first one). Must be >= 1.
    base_delay:
        Initial delay in seconds between attempts.
    max_delay:
        Upper bound on the computed delay (before jitter) in seconds.
    exponential_base:
        Base of the exponent used when computing the next delay.
        delay_n = base_delay * (exponential_base ** (attempt - 1))
    jitter:
        When *True* a uniform random fraction in [0, 1) is multiplied onto
        the clamped delay, which helps avoid thundering-herd problems.
    retry_on:
        Tuple of exception types that should trigger a retry.  Defaults to
        (Exception,) which retries on any exception.  Set to a narrower set
        (e.g. ``(IOError, TimeoutError)``) to avoid retrying on programming
        errors.
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: Tuple[Type[BaseException], ...] = field(
        default_factory=lambda: (Exception,)
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.exponential_base <= 1:
            raise ValueError("exponential_base must be > 1")

    def compute_delay(self, attempt: int) -> float:
        """Return the sleep duration (in seconds) before *attempt*.

        Parameters
        ----------
        attempt:
            Zero-based index of the *upcoming* retry (0 = first retry,
            i.e. after the first failure).

        Returns
        -------
        float
            Sleep duration in seconds, clamped to *max_delay* and optionally
            jittered.
        """
        # Exponential backoff: delay doubles (or grows by exponential_base) each retry
        delay = self.base_delay * (self.exponential_base**attempt)
        # Clamp to ceiling
        delay = min(delay, self.max_delay)
        # Optional full jitter: uniform sample in [0, delay)
        if self.jitter:
            delay = random.uniform(0.0, delay)
        return delay


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class RetryError(Exception):
    """Raised when all retry attempts have been exhausted.

    Attributes
    ----------
    attempts_made:
        The total number of call attempts that were made.
    last_exception:
        The exception raised on the final attempt.
    """

    def __init__(
        self,
        message: str,
        attempts_made: int,
        last_exception: BaseException,
    ) -> None:
        super().__init__(message)
        self.attempts_made: int = attempts_made
        self.last_exception: BaseException = last_exception

    def __repr__(self) -> str:
        return (
            f"RetryError(attempts_made={self.attempts_made}, "
            f"last_exception={self.last_exception!r})"
        )


# ---------------------------------------------------------------------------
# Core async implementation
# ---------------------------------------------------------------------------


async def with_retry(
    func: AsyncFunc,
    config: Optional[RetryConfig] = None,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute an async callable with exponential back-off retry logic.

    Parameters
    ----------
    func:
        The async callable to invoke.
    config:
        Retry configuration.  Defaults to ``RetryConfig()`` (3 attempts,
        1 s base delay, jitter enabled).
    *args, **kwargs:
        Forwarded verbatim to *func*.

    Returns
    -------
    Any
        The return value of *func* on success.

    Raises
    ------
    RetryError
        When all attempts are exhausted.  Wraps the last exception.
    """
    cfg = config if config is not None else RetryConfig()
    func_name = getattr(func, "__qualname__", repr(func))
    last_exc: Optional[BaseException] = None

    for attempt in range(1, cfg.max_attempts + 1):
        try:
            logger.debug(
                "Async attempt %d/%d for '%s'",
                attempt,
                cfg.max_attempts,
                func_name,
            )
            result = await func(*args, **kwargs)
            if attempt > 1:
                logger.info(
                    "Async call '%s' succeeded on attempt %d/%d",
                    func_name,
                    attempt,
                    cfg.max_attempts,
                )
            return result

        except tuple(cfg.retry_on) as exc:  # type: ignore[misc]
            last_exc = exc
            is_last_attempt = attempt == cfg.max_attempts

            if is_last_attempt:
                logger.error(
                    "Async call '%s' failed after %d attempt(s). "
                    "No more retries. Last exception: %s: %s",
                    func_name,
                    attempt,
                    type(exc).__name__,
                    exc,
                )
                break

            delay = cfg.compute_delay(attempt - 1)  # attempt-1 = 0-based retry index
            logger.warning(
                "Async call '%s' failed on attempt %d/%d "
                "(exception: %s: %s). Retrying in %.3f s...",
                func_name,
                attempt,
                cfg.max_attempts,
                type(exc).__name__,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    raise RetryError(
        f"All {cfg.max_attempts} attempt(s) for '{func_name}' failed. "
        f"Last error: {last_exc!r}",
        attempts_made=cfg.max_attempts,
        last_exception=last_exc,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Core sync implementation
# ---------------------------------------------------------------------------


def with_retry_sync(
    func: SyncFunc,
    config: Optional[RetryConfig] = None,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute a synchronous callable with exponential back-off retry logic.

    Parameters
    ----------
    func:
        The callable to invoke.
    config:
        Retry configuration.  Defaults to ``RetryConfig()``.
    *args, **kwargs:
        Forwarded verbatim to *func*.

    Returns
    -------
    Any
        The return value of *func* on success.

    Raises
    ------
    RetryError
        When all attempts are exhausted.  Wraps the last exception.
    """
    cfg = config if config is not None else RetryConfig()
    func_name = getattr(func, "__qualname__", repr(func))
    last_exc: Optional[BaseException] = None

    for attempt in range(1, cfg.max_attempts + 1):
        try:
            logger.debug(
                "Sync attempt %d/%d for '%s'",
                attempt,
                cfg.max_attempts,
                func_name,
            )
            result = func(*args, **kwargs)
            if attempt > 1:
                logger.info(
                    "Sync call '%s' succeeded on attempt %d/%d",
                    func_name,
                    attempt,
                    cfg.max_attempts,
                )
            return result

        except tuple(cfg.retry_on) as exc:  # type: ignore[misc]
            last_exc = exc
            is_last_attempt = attempt == cfg.max_attempts

            if is_last_attempt:
                logger.error(
                    "Sync call '%s' failed after %d attempt(s). "
                    "No more retries. Last exception: %s: %s",
                    func_name,
                    attempt,
                    type(exc).__name__,
                    exc,
                )
                break

            delay = cfg.compute_delay(attempt - 1)
            logger.warning(
                "Sync call '%s' failed on attempt %d/%d "
                "(exception: %s: %s). Retrying in %.3f s...",
                func_name,
                attempt,
                cfg.max_attempts,
                type(exc).__name__,
                exc,
                delay,
            )
            time.sleep(delay)

    raise RetryError(
        f"All {cfg.max_attempts} attempt(s) for '{func_name}' failed. "
        f"Last error: {last_exc!r}",
        attempts_made=cfg.max_attempts,
        last_exception=last_exc,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Decorator factories
# ---------------------------------------------------------------------------


def retry_decorator(
    config: Optional[RetryConfig] = None,
) -> Callable[[AsyncFunc], AsyncFunc]:
    """Decorator factory that wraps an **async** function with retry logic.

    Parameters
    ----------
    config:
        Optional ``RetryConfig``.  Falls back to defaults when *None*.

    Returns
    -------
    Callable
        A decorator that, when applied to an async function, returns a new
        async function with retry behaviour.

    Example
    -------
    ::

        @retry_decorator(RetryConfig(max_attempts=5, base_delay=0.5))
        async def unstable_api_call() -> dict:
            ...
    """
    cfg = config if config is not None else RetryConfig()

    def decorator(func: AsyncFunc) -> AsyncFunc:
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(
                f"retry_decorator expects an async function, got {func!r}. "
                "Use retry_sync_decorator for synchronous functions."
            )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await with_retry(func, cfg, *args, **kwargs)

        # Expose config on the wrapper for introspection / testing
        wrapper.retry_config = cfg  # type: ignore[attr-defined]
        return wrapper

    return decorator


def retry_sync_decorator(
    config: Optional[RetryConfig] = None,
) -> Callable[[SyncFunc], SyncFunc]:
    """Decorator factory that wraps a **sync** function with retry logic.

    Parameters
    ----------
    config:
        Optional ``RetryConfig``.  Falls back to defaults when *None*.

    Returns
    -------
    Callable
        A decorator that, when applied to a sync function, returns a new
        sync function with retry behaviour.

    Example
    -------
    ::

        @retry_sync_decorator(RetryConfig(max_attempts=4, jitter=False))
        def read_remote_file(path: str) -> bytes:
            ...
    """
    cfg = config if config is not None else RetryConfig()

    def decorator(func: SyncFunc) -> SyncFunc:
        if asyncio.iscoroutinefunction(func):
            raise TypeError(
                f"retry_sync_decorator expects a sync function, got {func!r}. "
                "Use retry_decorator for async functions."
            )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return with_retry_sync(func, cfg, *args, **kwargs)

        wrapper.retry_config = cfg  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Module-level default config (convenience)
# ---------------------------------------------------------------------------

DEFAULT_RETRY_CONFIG = RetryConfig()
"""Shared default config (3 attempts, 1 s base, 60 s max, jitter on)."""

AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=2.0,
    max_delay=120.0,
    exponential_base=2.0,
    jitter=True,
)
"""Preset for long-running or expensive remote calls."""

FAST_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=0.1,
    max_delay=2.0,
    exponential_base=2.0,
    jitter=False,
)
"""Preset for fast in-process retries (e.g. DB lock contention)."""

"""
Task Runner - Runs Celery tasks synchronously when Celery is not available.

Provides the same dispatch interface as Celery (.delay() and .apply_async())
but executes tasks inline when Redis / the Celery broker is unreachable.

This allows every call-site in the application to be written once:

    from scheduler.tasks.task_runner import get_task_runner, make_task_proxy
    from scheduler.tasks.content_tasks import generate_post_task

    # Dispatch – works whether Celery is up or not
    runner = get_task_runner()
    result = runner.delay(generate_post_task, topic, goal)

Or wrap individual task functions into proxies at import time:

    safe_generate = make_task_proxy(generate_post_task)
    safe_generate.delay("AI Tools", "educational")
    safe_generate.apply_async(args=["AI Tools"], kwargs={"goal": "educational"})

Public API
----------
- SyncTaskRunner        – executes tasks synchronously; mirrors Celery's result shape.
- TaskProxy             – wraps any callable; exposes .delay() / .apply_async().
- make_task_proxy()     – convenience factory for TaskProxy.
- get_task_runner()     – returns the live Celery app when Redis is available,
                          otherwise returns a SyncTaskRunner instance.
"""

from __future__ import annotations

import logging
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _build_result(
    *,
    task_id: str,
    task_name: str,
    success: bool,
    result: Any = None,
    error: Optional[str] = None,
    traceback_str: Optional[str] = None,
    started_at: str,
    completed_at: str,
    duration_ms: float,
) -> Dict[str, Any]:
    """Build the structured result dict returned by all sync executions."""
    return {
        "task_id": task_id,
        "task_name": task_name,
        "success": success,
        "result": result,
        "error": error,
        "traceback": traceback_str,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": round(duration_ms, 3),
    }


# ---------------------------------------------------------------------------
# SyncEagerResult
# ---------------------------------------------------------------------------


class SyncEagerResult:
    """
    Minimal stand-in for a Celery ``AsyncResult`` / ``EagerResult``.

    Returned by :class:`SyncTaskRunner` and :class:`TaskProxy` so call-sites
    that inspect ``.result``, ``.successful()``, or ``.failed()`` keep working
    without modification.
    """

    def __init__(self, task_id: str, result_dict: Dict[str, Any]) -> None:
        self.id: str = task_id
        self.task_id: str = task_id
        self._result_dict: Dict[str, Any] = result_dict

        # Surface the inner task return value as .result so callers can do
        #   eager = task.apply(...)
        #   data  = eager.result           # the dict returned by the task itself
        self.result: Any = result_dict.get("result")

        # Mimic Celery status strings
        self.status: str = "SUCCESS" if result_dict.get("success") else "FAILURE"
        self.state: str = self.status

    # ------------------------------------------------------------------
    # Celery-compatible query helpers
    # ------------------------------------------------------------------

    def successful(self) -> bool:
        """Return True if the task completed without raising."""
        return bool(self._result_dict.get("success"))

    def failed(self) -> bool:
        """Return True if the task raised an unhandled exception."""
        return not self.successful()

    def get(self, timeout: Optional[float] = None, propagate: bool = True) -> Any:
        """
        Retrieve the task result.

        Mirrors ``AsyncResult.get()``.  When *propagate* is ``True`` and the
        task failed, re-raises the stored error as a ``RuntimeError``.
        """
        if propagate and self.failed():
            error_msg = self._result_dict.get("error", "Task failed")
            raise RuntimeError(error_msg)
        return self.result

    def as_dict(self) -> Dict[str, Any]:
        """Return the full structured result dictionary."""
        return self._result_dict

    def __repr__(self) -> str:
        return (
            f"SyncEagerResult("
            f"task_id={self.task_id!r}, "
            f"status={self.status!r}, "
            f"success={self.successful()!r}"
            f")"
        )


# ---------------------------------------------------------------------------
# SyncTaskRunner
# ---------------------------------------------------------------------------


class SyncTaskRunner:
    """
    Synchronous task executor that mirrors the Celery dispatch interface.

    Use this when Redis / the Celery broker is unavailable.  Every task is
    executed in the current process and thread, blocking until it completes.

    Thread-safety
    -------------
    :class:`SyncTaskRunner` is stateless after construction and is therefore
    safe to share across threads.

    Example
    -------
    ::

        runner = SyncTaskRunner()

        # Direct execution
        result = runner.run_task(generate_post_task, "AI Tools", goal="educational")
        print(result["success"], result["result"])

        # Via Celery-style dispatch helpers
        eager = runner.delay(generate_post_task, "AI Tools")
        print(eager.result)
    """

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def run_task(
        self,
        task_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute *task_func* synchronously and return a structured result dict.

        The returned dict always contains:

        ``success``      bool – True when the function returned without raising.
        ``result``       Any  – the function's return value (None on failure).
        ``error``        Optional[str] – exception message on failure.
        ``traceback``    Optional[str] – full traceback string on failure.
        ``task_id``      str  – a new UUID4 assigned to this execution.
        ``task_name``    str  – ``__name__`` of the callable.
        ``started_at``   str  – UTC ISO-8601 timestamp.
        ``completed_at`` str  – UTC ISO-8601 timestamp.
        ``duration_ms``  float – wall-clock execution time in milliseconds.

        Parameters
        ----------
        task_func:
            Any callable – typically a Celery task function or a plain function.
        *args:
            Positional arguments forwarded to *task_func*.
        **kwargs:
            Keyword arguments forwarded to *task_func*.

        Returns
        -------
        dict
        """
        import time

        task_id: str = str(uuid.uuid4())
        task_name: str = getattr(task_func, "name", None) or getattr(
            task_func, "__name__", repr(task_func)
        )
        started_at: str = _utcnow_iso()
        t0: float = time.perf_counter()

        logger.info(
            "sync_runner|task_id=%s|task=%s|status=started",
            task_id,
            task_name,
        )

        try:
            # If *task_func* is a bound Celery task, calling it directly
            # bypasses the broker and runs the underlying Python function.
            return_value: Any = task_func(*args, **kwargs)
            duration_ms: float = (time.perf_counter() - t0) * 1_000
            completed_at: str = _utcnow_iso()

            logger.info(
                "sync_runner|task_id=%s|task=%s|duration_ms=%.1f|status=success",
                task_id,
                task_name,
                duration_ms,
            )

            return _build_result(
                task_id=task_id,
                task_name=task_name,
                success=True,
                result=return_value,
                error=None,
                traceback_str=None,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )

        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - t0) * 1_000
            completed_at = _utcnow_iso()
            tb: str = traceback.format_exc()

            logger.error(
                "sync_runner|task_id=%s|task=%s|duration_ms=%.1f"
                "|status=failed|error=%s",
                task_id,
                task_name,
                duration_ms,
                exc,
            )
            logger.debug("sync_runner|task_id=%s|traceback:\n%s", task_id, tb)

            return _build_result(
                task_id=task_id,
                task_name=task_name,
                success=False,
                result=None,
                error=str(exc),
                traceback_str=tb,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )

    # ------------------------------------------------------------------
    # Celery-compatible dispatch helpers
    # ------------------------------------------------------------------

    def delay(
        self,
        task_func: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> SyncEagerResult:
        """
        Synchronous equivalent of ``task_func.delay(*args, **kwargs)``.

        Parameters
        ----------
        task_func:
            Callable to execute.
        *args / **kwargs:
            Forwarded to *task_func*.

        Returns
        -------
        :class:`SyncEagerResult`
        """
        result_dict = self.run_task(task_func, *args, **kwargs)
        return SyncEagerResult(task_id=result_dict["task_id"], result_dict=result_dict)

    def apply_async(
        self,
        task_func: Callable[..., Any],
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        **options: Any,
    ) -> SyncEagerResult:
        """
        Synchronous equivalent of ``task_func.apply_async(args, kwargs)``.

        All Celery-specific *options* (``countdown``, ``eta``, ``queue``, etc.)
        are silently ignored – the task runs immediately.

        Parameters
        ----------
        task_func:
            Callable to execute.
        args:
            Positional arguments list.
        kwargs:
            Keyword arguments dict.
        **options:
            Ignored Celery routing/scheduling options.

        Returns
        -------
        :class:`SyncEagerResult`
        """
        _args: List[Any] = args or []
        _kwargs: Dict[str, Any] = kwargs or {}
        result_dict = self.run_task(task_func, *_args, **_kwargs)
        return SyncEagerResult(task_id=result_dict["task_id"], result_dict=result_dict)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return "SyncTaskRunner(mode=synchronous)"


# ---------------------------------------------------------------------------
# TaskProxy
# ---------------------------------------------------------------------------


class TaskProxy:
    """
    Wraps any callable and exposes a Celery-like dispatch interface.

    This lets you write dispatch code once and swap between Celery and sync
    execution transparently:

    ::

        safe_publish = make_task_proxy(publish_post_task)

        # Works with or without Celery
        safe_publish.delay(post_id=42)
        safe_publish.apply_async(args=[42], kwargs={"image_url": "https://..."})
        safe_publish(42)  # also callable directly

    When the wrapped function is a Celery task and Celery is available, its
    own ``.delay()`` / ``.apply_async()`` are called directly.  Otherwise the
    :class:`SyncTaskRunner` is used.

    Attributes
    ----------
    __name__:
        Mirrors the wrapped function's ``__name__``.
    __doc__:
        Mirrors the wrapped function's docstring.
    name:
        Celery task name (from ``task_func.name``) or ``__name__``.
    """

    def __init__(self, func: Callable[..., Any]) -> None:
        self._func: Callable[..., Any] = func
        self.__name__: str = getattr(func, "__name__", repr(func))
        self.__doc__: Optional[str] = getattr(func, "__doc__", None)
        self.__module__: str = getattr(func, "__module__", __name__)
        # Expose Celery task name if present
        self.name: str = getattr(func, "name", self.__name__)
        self._runner: SyncTaskRunner = SyncTaskRunner()

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    def delay(self, *args: Any, **kwargs: Any) -> SyncEagerResult:
        """
        Dispatch the task.

        Behaviour
        ---------
        - **Celery available** and the wrapped function has its own ``.delay()``:
          delegates to ``func.delay(*args, **kwargs)`` and wraps the
          ``AsyncResult`` in a lightweight shim so the returned object always
          has a consistent ``.result`` attribute.
        - **Otherwise**: runs synchronously via :class:`SyncTaskRunner`.

        Returns
        -------
        :class:`SyncEagerResult`
        """
        from scheduler.tasks.celery_app import is_celery_available

        if is_celery_available() and hasattr(self._func, "delay"):
            try:
                async_result = self._func.delay(*args, **kwargs)
                logger.debug(
                    "task_proxy|task=%s|mode=celery_async|task_id=%s|status=queued",
                    self.name,
                    getattr(async_result, "id", "?"),
                )
                # Wrap AsyncResult so callers always get the same interface
                return _wrap_async_result(async_result, task_name=self.name)
            except Exception as dispatch_exc:  # noqa: BLE001
                logger.warning(
                    "task_proxy|task=%s|celery_dispatch_failed=%s|falling_back_to_sync",
                    self.name,
                    dispatch_exc,
                )
                # Fall through to sync execution

        logger.debug("task_proxy|task=%s|mode=sync_fallback|status=running", self.name)
        return self._runner.delay(self._func, *args, **kwargs)

    def apply_async(
        self,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        **options: Any,
    ) -> SyncEagerResult:
        """
        Dispatch the task with explicit args / kwargs lists.

        Mirrors the Celery ``apply_async(args=[], kwargs={}, **options)``
        signature.  Routing options (``queue``, ``countdown``, ``eta``, …)
        are honoured when Celery is live and silently ignored in sync mode.

        Returns
        -------
        :class:`SyncEagerResult`
        """
        _args: List[Any] = args or []
        _kwargs: Dict[str, Any] = kwargs or {}

        from scheduler.tasks.celery_app import is_celery_available

        if is_celery_available() and hasattr(self._func, "apply_async"):
            try:
                async_result = self._func.apply_async(
                    args=_args, kwargs=_kwargs, **options
                )
                logger.debug(
                    "task_proxy|task=%s|mode=celery_apply_async|task_id=%s|status=queued",
                    self.name,
                    getattr(async_result, "id", "?"),
                )
                return _wrap_async_result(async_result, task_name=self.name)
            except Exception as dispatch_exc:  # noqa: BLE001
                logger.warning(
                    "task_proxy|task=%s|celery_apply_async_failed=%s"
                    "|falling_back_to_sync",
                    self.name,
                    dispatch_exc,
                )

        logger.debug(
            "task_proxy|task=%s|mode=sync_apply_async|status=running", self.name
        )
        return self._runner.apply_async(self._func, args=_args, kwargs=_kwargs)

    def apply(
        self,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> SyncEagerResult:
        """
        Execute the task synchronously regardless of broker availability.

        Useful inside other tasks when you need the result before continuing
        (mirrors Celery's ``task.apply()`` / ``CELERY_ALWAYS_EAGER`` behaviour).

        Returns
        -------
        :class:`SyncEagerResult`
        """
        _args: List[Any] = args or []
        _kwargs: Dict[str, Any] = kwargs or {}
        return self._runner.apply_async(self._func, args=_args, kwargs=_kwargs)

    # ------------------------------------------------------------------
    # Direct call – lets TaskProxy be used as a plain function
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function directly, bypassing all runner logic."""
        return self._func(*args, **kwargs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"TaskProxy(func={self.__name__!r}, celery_name={self.name!r})"


# ---------------------------------------------------------------------------
# Internal helper – wrap a live Celery AsyncResult in our shim
# ---------------------------------------------------------------------------


def _wrap_async_result(
    async_result: Any,
    task_name: str,
) -> SyncEagerResult:
    """
    Wrap a live Celery ``AsyncResult`` in a :class:`SyncEagerResult` shim.

    This gives callers a consistent ``.result`` attribute even when the task
    is running remotely.  The ``result`` is ``None`` for genuinely async tasks
    (the caller must ``.get()`` if they need it) but the wrapper is still
    useful for its ``.successful()`` / ``.failed()`` convenience methods once
    the task settles.
    """
    task_id: str = getattr(async_result, "id", str(uuid.uuid4()))
    result_dict: Dict[str, Any] = {
        "task_id": task_id,
        "task_name": task_name,
        "success": True,  # optimistic – task was accepted by the broker
        "result": None,  # not yet available; caller must .get() if needed
        "error": None,
        "traceback": None,
        "started_at": _utcnow_iso(),
        "completed_at": _utcnow_iso(),
        "duration_ms": 0.0,
        "_async_result": async_result,  # allow advanced callers to unwrap
    }
    shim = SyncEagerResult(task_id=task_id, result_dict=result_dict)
    # Override .get() to delegate to the real AsyncResult
    _original_async = async_result

    def _delegating_get(
        timeout: Optional[float] = None,
        propagate: bool = True,
    ) -> Any:
        return _original_async.get(timeout=timeout, propagate=propagate)

    shim.get = _delegating_get  # type: ignore[method-assign]
    return shim


# ---------------------------------------------------------------------------
# make_task_proxy
# ---------------------------------------------------------------------------


def make_task_proxy(func: Callable[..., Any]) -> TaskProxy:
    """
    Wrap *func* in a :class:`TaskProxy`.

    Parameters
    ----------
    func:
        Any callable – typically a Celery task (decorated with
        ``@celery_app.task``).

    Returns
    -------
    :class:`TaskProxy`

    Example
    -------
    ::

        from scheduler.tasks.content_tasks import generate_post_task
        from scheduler.tasks.task_runner import make_task_proxy

        safe_generate = make_task_proxy(generate_post_task)

        # Queue via Celery when available, else run inline
        result = safe_generate.delay("FastAPI Best Practices", goal="educational")
        print(result.successful())
    """
    if not callable(func):
        raise TypeError(
            f"make_task_proxy() requires a callable, got {type(func).__name__!r}"
        )
    return TaskProxy(func)


# ---------------------------------------------------------------------------
# get_task_runner
# ---------------------------------------------------------------------------


def get_task_runner() -> Union[Any, SyncTaskRunner]:
    """
    Return the appropriate task runner for the current environment.

    Returns
    -------
    ``celery_app``
        The live Celery application instance when Redis is reachable.
        Callers can then use ``celery_app.send_task(...)`` or import tasks
        directly and call ``.delay()`` / ``.apply_async()`` on them.

    :class:`SyncTaskRunner`
        A synchronous executor when Redis is unavailable (e.g. local dev,
        CI, environments without Redis).

    Example
    -------
    ::

        runner = get_task_runner()

        if isinstance(runner, SyncTaskRunner):
            # Sync path
            result = runner.run_task(generate_post_task, "AI Tools")
        else:
            # Celery path – use task functions directly
            from scheduler.tasks.content_tasks import generate_post_task
            async_result = generate_post_task.delay("AI Tools")

    Notes
    -----
    For most application code :func:`make_task_proxy` is more ergonomic
    because it hides the branching entirely:

    ::

        safe_task = make_task_proxy(generate_post_task)
        safe_task.delay("AI Tools")   # works in both modes
    """
    try:
        from scheduler.tasks.celery_app import celery_app, is_celery_available

        if is_celery_available():
            logger.debug("get_task_runner|mode=celery|broker=available")
            return celery_app

        logger.info("get_task_runner|mode=sync|reason=broker_unavailable")
        return SyncTaskRunner()

    except ImportError as import_exc:
        logger.warning(
            "get_task_runner|mode=sync|reason=celery_not_installed|error=%s",
            import_exc,
        )
        return SyncTaskRunner()

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "get_task_runner|mode=sync|reason=unexpected_error|error=%s",
            exc,
        )
        return SyncTaskRunner()

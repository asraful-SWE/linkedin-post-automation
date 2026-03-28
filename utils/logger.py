"""
logger.py - Enhanced logging for LinkedIn AI Poster.

Provides both plain-text (legacy) and structured JSON logging modes, selected
at startup via the ``ENABLE_JSON_LOGS`` environment variable.

Backward-compatible public API
-------------------------------
- ``setup_logging()``        – initialise the root logger (call once at startup)
- ``get_logger(name)``       – return a standard :class:`logging.Logger`
- ``ContextualLogger``       – logger wrapper that injects key=value context

New additions
-------------
- ``JSONFormatter``          – :class:`logging.Formatter` that emits JSON lines
- ``StructuredLogger``       – wrapper with a ``log_event()`` method for
                               typed, schema-consistent event records
- ``log_event()``            – module-level convenience that calls
                               ``StructuredLogger.log_event`` by logger name
- ``EVENT_*`` constants      – canonical event-name strings

Usage
-----
Plain-text (default)::

    from linkedin_ai_poster.utils.logger import setup_logging, get_logger

    setup_logging()
    log = get_logger(__name__)
    log.info("Service started")

JSON mode (set ``ENABLE_JSON_LOGS=true`` in the environment or .env)::

    # same code – formatter is switched transparently by setup_logging()

Structured event::

    from linkedin_ai_poster.utils.logger import log_event, EVENT_POST_PUBLISHED

    log_event(
        __name__,
        EVENT_POST_PUBLISHED,
        post_id=42,
        linkedin_urn="urn:li:share:123",
    )
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Event name constants
# ---------------------------------------------------------------------------

EVENT_POST_GENERATED: str = "post.generated"
EVENT_POST_APPROVED: str = "post.approved"
EVENT_POST_PUBLISHED: str = "post.published"
EVENT_POST_FAILED: str = "post.failed"
EVENT_IMAGE_UPLOADED: str = "image.uploaded"
EVENT_IMAGE_FAILED: str = "image.failed"
EVENT_EMAIL_SENT: str = "email.sent"
EVENT_EMAIL_FAILED: str = "email.failed"
EVENT_PUBLISH_RETRY: str = "publish.retry"
EVENT_QUEUE_TASK_STARTED: str = "queue.task_started"
EVENT_QUEUE_TASK_COMPLETED: str = "queue.task_completed"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

#: Plain-text format used by the legacy (non-JSON) handlers.
_DETAILED_FORMAT: str = (
    "%(asctime)s | %(name)-20s | %(levelname)-8s | %(funcName)-15s | %(message)s"
)
_SIMPLE_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(message)s"

#: Noisy third-party loggers that we always silence to WARNING.
_QUIET_LOGGERS: tuple[str, ...] = (
    "urllib3",
    "requests",
    "httpcore",
    "httpx",
    "apscheduler",
    "multipart",
    "passlib",
)


def _is_json_mode() -> bool:
    """Return True when structured JSON logging is requested."""
    return os.getenv("ENABLE_JSON_LOGS", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _resolve_level(default: str = "INFO") -> int:
    """Read LOG_LEVEL from env and return the corresponding int constant."""
    level_name: str = os.getenv("LOG_LEVEL", default).strip().upper()
    level: int = getattr(logging, level_name, logging.INFO)
    return level


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Formatter that serialises each :class:`logging.LogRecord` as a JSON line.

    Output fields
    -------------
    - ``timestamp``  – ISO-8601 UTC timestamp (e.g. ``"2024-06-01T12:00:00.123456Z"``)
    - ``level``      – uppercase level name (``"INFO"``, ``"ERROR"`` …)
    - ``logger``     – logger name (``record.name``)
    - ``function``   – function/method that called the logger
    - ``line``       – source line number
    - ``message``    – formatted log message
    - ``event``      – (optional) structured event name, when present in ``extra``
    - any extra keys passed via ``logging.Logger.info(..., extra={...})``
    - ``exc_info``   – formatted traceback string, only when an exception is attached

    Example output
    --------------
    ::

        {
          "timestamp": "2024-06-01T12:00:00.000123Z",
          "level": "INFO",
          "logger": "linkedin_ai_poster.services.publisher",
          "function": "publish_post",
          "line": 42,
          "message": "Post published successfully.",
          "event": "post.published",
          "post_id": 7
        }
    """

    #: Keys that are part of the standard LogRecord and should NOT be
    #: re-emitted as free-form extra fields.
    _STDLIB_ATTRS: frozenset[str] = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",  # Python 3.12+
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        """Serialise *record* to a JSON-encoded string."""
        # Build the base payload in a deterministic key order.
        payload: dict[str, Any] = {
            "timestamp": self._utc_iso(record.created),
            "level": record.levelname,
            "logger": record.name,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Promote any extra fields the caller injected (e.g. post_id, event …)
        for key, value in record.__dict__.items():
            if key not in self._STDLIB_ATTRS and not key.startswith("_"):
                payload[key] = value

        # Attach formatted traceback when an exception is present.
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_info"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exc_info"] = record.exc_text

        # Stack info (Python 3.5+)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        try:
            return json.dumps(payload, default=str, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            # Fallback: never let the formatter itself crash the application.
            return json.dumps(
                {
                    "timestamp": self._utc_iso(record.created),
                    "level": "ERROR",
                    "logger": record.name,
                    "function": record.funcName,
                    "line": record.lineno,
                    "message": f"[JSONFormatter] serialisation error: {exc}",
                },
                ensure_ascii=False,
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _utc_iso(epoch: float) -> str:
        """Convert a Unix timestamp (float) to an ISO-8601 UTC string."""
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


# ---------------------------------------------------------------------------
# Structured logger
# ---------------------------------------------------------------------------


class StructuredLogger:
    """Wraps a standard :class:`logging.Logger` with a structured event API.

    All regular logger methods (``debug``, ``info``, ``warning``, ``error``,
    ``critical``, ``exception``) are delegated transparently, so existing
    call-sites do not need to change.

    The additional :meth:`log_event` method emits a log record that carries a
    mandatory ``event`` field along with any caller-supplied keyword arguments
    as extra fields.  When JSON mode is active these appear as top-level keys in
    the JSON output.  In plain-text mode they are appended to the message.

    Example
    -------
    ::

        from linkedin_ai_poster.utils.logger import StructuredLogger, EVENT_POST_PUBLISHED

        log = StructuredLogger(__name__)
        log.log_event(EVENT_POST_PUBLISHED, post_id=42, urn="urn:li:share:123")
        # JSON output:
        # {"timestamp": "...", "level": "INFO", "event": "post.published",
        #  "post_id": 42, "urn": "urn:li:share:123", ...}
    """

    def __init__(self, name: str) -> None:
        self._logger: logging.Logger = logging.getLogger(name)

    # ------------------------------------------------------------------
    # Delegate standard methods
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._logger.name

    def debug(self, msg: object, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: object, *args: Any, **kwargs: Any) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: object, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(msg, *args, **kwargs)

    # alias
    warn = warning

    def error(self, msg: object, *args: Any, **kwargs: Any) -> None:
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: object, *args: Any, **kwargs: Any) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: object, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("exc_info", True)
        self._logger.error(msg, *args, **kwargs)

    def isEnabledFor(self, level: int) -> bool:  # noqa: N802
        return self._logger.isEnabledFor(level)

    # ------------------------------------------------------------------
    # Structured event API
    # ------------------------------------------------------------------

    def log_event(
        self,
        event_name: str,
        level: str = "INFO",
        **kwargs: Any,
    ) -> None:
        """Emit a structured log record for *event_name*.

        Parameters
        ----------
        event_name:
            A dot-separated event identifier, ideally one of the
            ``EVENT_*`` constants defined in this module
            (e.g. ``EVENT_POST_PUBLISHED``).
        level:
            Log level string: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``, or ``"CRITICAL"``.  Defaults to ``"INFO"``.
        **kwargs:
            Arbitrary key/value pairs to include in the structured record.
            In JSON mode they appear as top-level fields; in plain-text mode
            they are appended to the message as ``key=value`` pairs.

        Example
        -------
        ::

            log.log_event(
                EVENT_POST_FAILED,
                level="ERROR",
                post_id=99,
                reason="LinkedIn API timeout",
            )
        """
        level_int: int = getattr(logging, level.upper(), logging.INFO)
        if not self._logger.isEnabledFor(level_int):
            return

        # Build the human-readable message (useful in plain-text mode and as a
        # fallback when the JSON payload cannot be rendered).
        kv_str = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
        message = f"[{event_name}] {kv_str}" if kv_str else f"[{event_name}]"

        # Inject ``event`` and all kwargs as extra fields so that
        # JSONFormatter can surface them as top-level JSON keys.
        extra: dict[str, Any] = {"event": event_name, **kwargs}

        self._logger.log(level_int, message, extra=extra, stacklevel=2)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup_logging() -> None:
    """Initialise the root logger.

    Call this **once** at application startup (e.g. in ``main.py`` or the
    FastAPI ``lifespan`` function).  Calling it multiple times is safe – each
    call clears existing handlers before adding new ones.

    Behaviour
    ---------
    * Reads ``LOG_LEVEL`` (default ``"INFO"``) and ``DEBUG`` (default
      ``"false"``) from the environment.
    * Reads ``ENABLE_JSON_LOGS`` (default ``"false"``).  When truthy, all
      handlers use :class:`JSONFormatter`; otherwise plain-text formatters are
      used (preserving the legacy format exactly).
    * Creates a ``logs/`` directory and attaches rotating file handlers
      (general log + error-only log + optional debug log).
    * Silences noisy third-party loggers.
    """
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    log_level: int = _resolve_level()
    debug_mode: bool = os.getenv("DEBUG", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    json_mode: bool = _is_json_mode()

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    # ------------------------------------------------------------------
    # Choose formatters
    # ------------------------------------------------------------------
    if json_mode:
        console_formatter: logging.Formatter = JSONFormatter()
        file_formatter: logging.Formatter = JSONFormatter()
        debug_formatter: logging.Formatter = JSONFormatter()
    else:
        _detailed = logging.Formatter(_DETAILED_FORMAT)
        _simple = logging.Formatter(_SIMPLE_FORMAT)
        console_formatter = _simple if not debug_mode else _detailed
        file_formatter = _detailed
        debug_formatter = _detailed

    # ------------------------------------------------------------------
    # Console handler
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # ------------------------------------------------------------------
    # Rotating file handler – all levels >= INFO
    # ------------------------------------------------------------------
    main_file_handler = RotatingFileHandler(
        os.path.join(logs_dir, "linkedin_poster.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    main_file_handler.setLevel(logging.INFO)
    main_file_handler.setFormatter(file_formatter)
    root_logger.addHandler(main_file_handler)

    # ------------------------------------------------------------------
    # Rotating file handler – errors only
    # ------------------------------------------------------------------
    error_file_handler = RotatingFileHandler(
        os.path.join(logs_dir, "errors.log"),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_file_handler)

    # ------------------------------------------------------------------
    # Debug file handler (only when DEBUG=true)
    # ------------------------------------------------------------------
    if debug_mode:
        debug_file_handler = RotatingFileHandler(
            os.path.join(logs_dir, "debug.log"),
            maxBytes=20 * 1024 * 1024,  # 20 MB
            backupCount=2,
            encoding="utf-8",
        )
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(debug_formatter)
        root_logger.addHandler(debug_file_handler)

    # ------------------------------------------------------------------
    # Silence noisy third-party loggers
    # ------------------------------------------------------------------
    for _noisy in _QUIET_LOGGERS:
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    # ------------------------------------------------------------------
    # Startup confirmation
    # ------------------------------------------------------------------
    startup_log = logging.getLogger("startup")
    startup_log.info("Logging system initialised")
    startup_log.info("Log level    : %s", logging.getLevelName(log_level))
    startup_log.info("Debug mode   : %s", debug_mode)
    startup_log.info("JSON mode    : %s", json_mode)
    startup_log.info("Log directory: %s", os.path.abspath(logs_dir))


# ---------------------------------------------------------------------------
# Public convenience accessors
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """Return a standard :class:`logging.Logger` for *name*.

    This is the primary way to obtain a logger throughout the codebase.
    It is a thin wrapper around :func:`logging.getLogger` and does **not**
    add any handlers – that is the responsibility of :func:`setup_logging`.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# ContextualLogger (backward-compatible, enhanced)
# ---------------------------------------------------------------------------


class ContextualLogger:
    """Logger wrapper that prepends key=value context to every message.

    Backward-compatible with the original implementation; adds ``exception``
    and ``log_event`` methods, and properly forwards ``*args``/``**kwargs``
    so that %-style format strings work as expected.

    Example
    -------
    ::

        log = ContextualLogger(__name__, {"post_id": 42, "user": "alice"})
        log.info("Publishing post")
        # → [post_id=42 user=alice] Publishing post
    """

    def __init__(
        self,
        name: str,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        self.logger: logging.Logger = logging.getLogger(name)
        self.context: dict[str, Any] = context or {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_message(self, message: str) -> str:
        """Prepend the current context to *message*."""
        if self.context:
            context_str = " ".join(f"{k}={v}" for k, v in self.context.items())
            return f"[{context_str}] {message}"
        return message

    def _extra(self) -> dict[str, Any]:
        """Return the context dict as a ``logging.Logger`` extra payload."""
        return dict(self.context)

    # ------------------------------------------------------------------
    # Standard log methods
    # ------------------------------------------------------------------

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(
            self._format_message(message), *args, extra=self._extra(), **kwargs
        )

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(
            self._format_message(message), *args, extra=self._extra(), **kwargs
        )

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(
            self._format_message(message), *args, extra=self._extra(), **kwargs
        )

    # legacy alias preserved from original
    def warn(self, message: str, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        self.warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(
            self._format_message(message), *args, extra=self._extra(), **kwargs
        )

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.critical(
            self._format_message(message), *args, extra=self._extra(), **kwargs
        )

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log *message* at ERROR level and attach the current exception info."""
        kwargs.setdefault("exc_info", True)
        self.logger.error(
            self._format_message(message), *args, extra=self._extra(), **kwargs
        )

    # ------------------------------------------------------------------
    # Structured event (new addition)
    # ------------------------------------------------------------------

    def log_event(
        self,
        event_name: str,
        level: str = "INFO",
        **kwargs: Any,
    ) -> None:
        """Emit a structured event record, merging context into *kwargs*.

        Delegates to :class:`StructuredLogger` semantics but injects the
        current ``self.context`` into the extra fields automatically.

        Parameters
        ----------
        event_name:
            Dot-separated event identifier (preferably one of ``EVENT_*``).
        level:
            Log level string.  Defaults to ``"INFO"``.
        **kwargs:
            Additional fields merged with the current context.
        """
        level_int: int = getattr(logging, level.upper(), logging.INFO)
        if not self.logger.isEnabledFor(level_int):
            return

        merged = {**self.context, **kwargs}
        kv_str = " ".join(f"{k}={v!r}" for k, v in merged.items())
        message = f"[{event_name}] {kv_str}" if kv_str else f"[{event_name}]"
        extra: dict[str, Any] = {"event": event_name, **merged}
        self.logger.log(level_int, message, extra=extra, stacklevel=2)

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def add_context(self, **kwargs: Any) -> None:
        """Merge *kwargs* into the current context."""
        self.context.update(kwargs)

    def clear_context(self) -> None:
        """Remove all context fields."""
        self.context.clear()

    def with_context(self, **kwargs: Any) -> "ContextualLogger":
        """Return a **new** :class:`ContextualLogger` with extended context.

        The original logger is not mutated.

        Example
        -------
        ::

            base_log = ContextualLogger(__name__)
            post_log = base_log.with_context(post_id=42)
            post_log.info("Processing")   # → [post_id=42] Processing
        """
        return ContextualLogger(self.logger.name, {**self.context, **kwargs})


# ---------------------------------------------------------------------------
# Module-level log_event convenience function
# ---------------------------------------------------------------------------


def log_event(
    logger_name: str,
    event_name: str,
    level: str = "INFO",
    **kwargs: Any,
) -> None:
    """Emit a structured event record via the named logger.

    This is a module-level shortcut for::

        StructuredLogger(logger_name).log_event(event_name, level, **kwargs)

    It is useful in functional (non-class) code where constructing a
    :class:`StructuredLogger` instance would be verbose.

    Parameters
    ----------
    logger_name:
        Name of the logger to use (typically ``__name__`` of the caller).
    event_name:
        Dot-separated event identifier (preferably one of ``EVENT_*``).
    level:
        Log level string.  Defaults to ``"INFO"``.
    **kwargs:
        Arbitrary key/value pairs to include in the structured record.

    Example
    -------
    ::

        from linkedin_ai_poster.utils.logger import log_event, EVENT_EMAIL_SENT

        log_event(
            __name__,
            EVENT_EMAIL_SENT,
            recipient="alice@example.com",
            subject="Your post was approved",
        )
    """
    level_int: int = getattr(logging, level.upper(), logging.INFO)
    _log = logging.getLogger(logger_name)

    if not _log.isEnabledFor(level_int):
        return

    kv_str = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
    message = f"[{event_name}] {kv_str}" if kv_str else f"[{event_name}]"
    extra: dict[str, Any] = {"event": event_name, **kwargs}
    _log.log(level_int, message, extra=extra, stacklevel=2)

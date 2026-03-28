"""
security.py - Security utilities for LinkedIn AI Poster.

Provides:
- Admin API key validation using timing-safe comparison (hmac.compare_digest)
- FastAPI dependency for protecting admin endpoints via the X-Admin-Key header
- Log-data sanitiser that redacts sensitive fields before they reach log sinks
- URL validator that enforces HTTPS and a structurally valid domain

All helpers are deliberately free of side-effects so they can be unit-tested
without a running FastAPI application or database.

Usage
-----
Protecting an endpoint::

    from linkedin_ai_poster.utils.security import require_admin

    @router.post("/admin/trigger")
    async def trigger(payload: dict, _: None = Depends(require_admin)):
        ...

Sanitising before logging::

    from linkedin_ai_poster.utils.security import sanitize_log_data

    logger.info("Calling LinkedIn API", extra=sanitize_log_data(payload))
"""

from __future__ import annotations

import hmac
import logging
import re
from typing import Any, Optional
from urllib.parse import urlparse

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: HTTP header name expected to carry the admin API key.
ADMIN_KEY_HEADER_NAME: str = "X-Admin-Key"

#: Sentinel value used in place of redacted content in sanitised dicts.
REDACTED: str = "***REDACTED***"

#: Case-insensitive substrings that flag a dict key as sensitive.
_SENSITIVE_KEY_FRAGMENTS: frozenset[str] = frozenset(
    {
        "token",
        "password",
        "passwd",
        "api_key",
        "apikey",
        "access_token",
        "secret",
        "authorization",
        "auth",
        "credential",
        "private_key",
        "privatekey",
        "client_secret",
    }
)

#: Compiled regex for basic domain validation:
#:   - at least one label (letters, digits, hyphens)
#:   - at least one dot separating labels
#:   - TLD of 2–63 chars, letters only
_DOMAIN_RE: re.Pattern[str] = re.compile(
    r"""
    ^
    (?:
        (?:[a-zA-Z0-9]          # label must start with alphanumeric
           (?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?  # optional middle + end
        )
        \.                      # dot separator
    )+
    [a-zA-Z]{2,63}              # TLD: letters only, 2-63 chars
    $
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Admin key validation
# ---------------------------------------------------------------------------


def validate_admin_key(api_key: str, settings: Any) -> bool:
    """Return *True* when *api_key* matches the configured admin API key.

    The comparison is performed with :func:`hmac.compare_digest` to prevent
    timing-based side-channel attacks.

    Parameters
    ----------
    api_key:
        The key supplied by the caller (e.g. from a request header).
    settings:
        An object that exposes an ``admin_api_key`` attribute (typically the
        application :class:`~linkedin_ai_poster.app.config.Settings` instance).

    Returns
    -------
    bool
        ``True`` if the key is non-empty, the configured key is non-empty, and
        both values are identical.  ``False`` in every other case.

    Notes
    -----
    Both operands are encoded to UTF-8 bytes before comparison because
    :func:`hmac.compare_digest` requires that both arguments are of the same
    type (either both ``str`` or both ``bytes``).  Using ``bytes`` avoids any
    ambiguity around string interning optimisations.
    """
    configured_key: Optional[str] = getattr(settings, "admin_api_key", None)

    # Reject immediately if either key is absent / empty – no need for timing
    # safety here because the branch is taken before any comparison.
    if not api_key or not configured_key:
        logger.debug(
            "validate_admin_key: rejected – %s",
            "supplied key is empty"
            if not api_key
            else "admin_api_key is not configured",
        )
        return False

    # Timing-safe comparison
    match = hmac.compare_digest(
        api_key.encode("utf-8"),
        configured_key.encode("utf-8"),
    )

    if not match:
        logger.warning(
            "validate_admin_key: key mismatch – supplied key did not match "
            "the configured admin_api_key."
        )

    return match


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

#: FastAPI's built-in APIKeyHeader scheme.
#: ``auto_error=False`` lets us produce a richer 401 response ourselves.
_admin_key_scheme = APIKeyHeader(name=ADMIN_KEY_HEADER_NAME, auto_error=False)


class AdminKeyHeader:
    """FastAPI dependency class that enforces admin API key authentication.

    Reads the ``X-Admin-Key`` header from the incoming request and validates
    it against ``settings.admin_api_key``.  Raises :class:`HTTPException`
    with status 401 if the header is missing or the key is invalid.

    This class is modelled after FastAPI's ``HTTPBearer`` security scheme so
    it can be used both as a class-based dependency and as a Swagger UI
    security scheme.

    Example
    -------
    ::

        admin_auth = AdminKeyHeader()

        @router.delete("/admin/posts/{post_id}")
        async def delete_post(post_id: int, _=Depends(admin_auth)):
            ...
    """

    def __init__(self, *, auto_error: bool = True) -> None:
        """
        Parameters
        ----------
        auto_error:
            When *True* (default) an :class:`HTTPException` is raised on
            failure.  Set to *False* to receive ``None`` instead, allowing
            callers to implement custom fallback logic.
        """
        self.auto_error = auto_error

    async def __call__(
        self,
        request: Request,
        api_key: Optional[str] = Depends(_admin_key_scheme),
    ) -> Optional[str]:
        """Validate the ``X-Admin-Key`` header.

        Parameters
        ----------
        request:
            The current FastAPI / Starlette request (injected by the DI
            framework; used to access ``app.state.settings`` if available).
        api_key:
            Value extracted from the ``X-Admin-Key`` header by
            :data:`_admin_key_scheme`.  ``None`` when the header is absent.

        Returns
        -------
        str or None
            The validated API key string on success, or ``None`` when
            *auto_error* is ``False`` and validation fails.

        Raises
        ------
        HTTPException
            Status 401 when *auto_error* is ``True`` and validation fails.
        """
        # Retrieve settings from app state (set during application startup)
        # or fall back to the module-level singleton.
        settings = getattr(getattr(request, "app", None), "state", None)
        settings = getattr(settings, "settings", None)

        if settings is None:
            # Lazy import avoids a circular dependency at module load time.
            from linkedin_ai_poster.app.config import get_settings  # noqa: PLC0415

            settings = get_settings()

        if not api_key:
            logger.warning(
                "AdminKeyHeader: request to '%s' rejected – "
                "X-Admin-Key header is missing.",
                request.url.path,
            )
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="X-Admin-Key header is required.",
                    headers={"WWW-Authenticate": "ApiKey"},
                )
            return None

        if not validate_admin_key(api_key, settings):
            logger.warning(
                "AdminKeyHeader: request to '%s' rejected – invalid admin key. "
                "Client IP: %s",
                request.url.path,
                _get_client_ip(request),
            )
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid admin API key.",
                    headers={"WWW-Authenticate": "ApiKey"},
                )
            return None

        logger.debug(
            "AdminKeyHeader: admin key accepted for path '%s'.",
            request.url.path,
        )
        return api_key


# ---------------------------------------------------------------------------
# Module-level dependency (convenience singleton)
# ---------------------------------------------------------------------------

#: Pre-built :class:`AdminKeyHeader` instance, ready to use with
#: ``Depends(require_admin)``.
_admin_key_dep = AdminKeyHeader()


async def require_admin(
    request: Request,
    api_key: Optional[str] = Depends(_admin_key_scheme),
) -> Optional[str]:
    """FastAPI ``Depends``-compatible dependency that requires a valid admin key.

    This is a thin, named wrapper around :class:`AdminKeyHeader` so that
    endpoints can be protected with the idiomatic ``Depends(require_admin)``
    pattern without explicitly constructing a class instance.

    Parameters
    ----------
    request:
        Injected by FastAPI.
    api_key:
        Injected by :data:`_admin_key_scheme`.

    Returns
    -------
    str
        The validated admin key string.

    Raises
    ------
    HTTPException
        Status 401 when the key is missing or invalid.

    Example
    -------
    ::

        from linkedin_ai_poster.utils.security import require_admin

        @router.post("/admin/generate")
        async def generate(_: str = Depends(require_admin)):
            ...
    """
    return await _admin_key_dep(request, api_key)


# ---------------------------------------------------------------------------
# Log data sanitiser
# ---------------------------------------------------------------------------


def sanitize_log_data(data: dict[str, Any]) -> dict[str, Any]:
    """Return a *shallow copy* of *data* with sensitive values redacted.

    A key is considered sensitive when its lowercase string representation
    contains any of the substrings defined in
    :data:`_SENSITIVE_KEY_FRAGMENTS`.  Nested dicts are recursively sanitised.
    Non-dict values at nested positions are left unchanged.

    Parameters
    ----------
    data:
        The dictionary to sanitise (e.g. a request payload, a log ``extra``
        dict).  The original is **not** mutated.

    Returns
    -------
    dict
        A new dictionary with sensitive values replaced by
        :data:`REDACTED`.

    Example
    -------
    ::

        payload = {
            "username": "alice",
            "password": "s3cr3t",
            "metadata": {"access_token": "tok_abc", "retries": 3},
        }
        safe = sanitize_log_data(payload)
        # safe == {
        #     "username": "alice",
        #     "password": "***REDACTED***",
        #     "metadata": {"access_token": "***REDACTED***", "retries": 3},
        # }
    """
    if not isinstance(data, dict):
        # Defensive: accept only dicts; return as-is for other types
        return data  # type: ignore[return-value]

    sanitised: dict[str, Any] = {}

    for key, value in data.items():
        key_lower = str(key).lower()
        is_sensitive = any(
            fragment in key_lower for fragment in _SENSITIVE_KEY_FRAGMENTS
        )

        if is_sensitive:
            sanitised[key] = REDACTED
        elif isinstance(value, dict):
            # Recurse into nested dicts
            sanitised[key] = sanitize_log_data(value)
        elif isinstance(value, (list, tuple)):
            # Recurse into sequences that may contain dicts
            sanitised[key] = type(value)(
                sanitize_log_data(item) if isinstance(item, dict) else item
                for item in value
            )
        else:
            sanitised[key] = value

    return sanitised


# ---------------------------------------------------------------------------
# URL validator
# ---------------------------------------------------------------------------


def validate_url(url: str) -> bool:
    """Return *True* when *url* is a well-formed HTTPS URL with a valid domain.

    Checks performed (in order):
    1. ``url`` is a non-empty string.
    2. Scheme is exactly ``https``.
    3. A ``netloc`` (host) is present.
    4. The host (stripped of any port) matches :data:`_DOMAIN_RE`, ensuring it
       is either a valid fully-qualified domain name *or* a bare hostname
       (single label – useful for internal service URLs in Docker networks).
    5. The path component does not contain ``..`` path traversal sequences.

    Parameters
    ----------
    url:
        The URL string to validate.

    Returns
    -------
    bool
        ``True`` when all checks pass, ``False`` otherwise.

    Notes
    -----
    This is an intentionally **strict** validator: it rejects ``http://``,
    bare IP addresses, and malformed hostnames.  If you need to allow HTTP
    in development environments, gate the call on a ``debug`` flag rather than
    weakening this function.

    Example
    -------
    ::

        validate_url("https://api.linkedin.com/v2/posts")   # True
        validate_url("http://api.linkedin.com/v2/posts")    # False – not HTTPS
        validate_url("https://")                            # False – no host
        validate_url("https://bad..domain.com/path")        # False – bad domain
    """
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url.strip())
    except Exception:  # pragma: no cover – urlparse rarely raises
        return False

    # 1. Scheme must be https
    if parsed.scheme != "https":
        logger.debug(
            "validate_url: rejected '%s' – scheme is '%s', expected 'https'.",
            url,
            parsed.scheme,
        )
        return False

    # 2. A host must be present
    if not parsed.netloc:
        logger.debug("validate_url: rejected '%s' – no netloc/host.", url)
        return False

    # 3. Strip optional port and userinfo (user:pass@host:port → host)
    hostname: str = parsed.hostname or ""
    if not hostname:
        logger.debug("validate_url: rejected '%s' – could not extract hostname.", url)
        return False

    # 4. Validate domain structure
    #    Allow single-label hostnames (e.g. "redis") for internal networks,
    #    but still require the label itself to be alphanumeric/hyphen only.
    _single_label_re = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?$")
    is_valid_domain = bool(_DOMAIN_RE.match(hostname))
    is_valid_single_label = bool(_single_label_re.match(hostname))

    if not (is_valid_domain or is_valid_single_label):
        logger.debug(
            "validate_url: rejected '%s' – hostname '%s' failed domain validation.",
            url,
            hostname,
        )
        return False

    # 5. Reject path traversal
    if ".." in (parsed.path or ""):
        logger.debug("validate_url: rejected '%s' – path contains '..'.", url)
        return False

    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_client_ip(request: Request) -> str:
    """Extract the best-effort client IP from *request*.

    Checks ``X-Forwarded-For`` first (set by reverse proxies), then falls
    back to the direct connection's remote address.

    Parameters
    ----------
    request:
        The current Starlette / FastAPI request object.

    Returns
    -------
    str
        IP address string, or ``"unknown"`` if it cannot be determined.
    """
    forwarded_for: Optional[str] = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Header may contain a comma-separated list; first is the originating IP
        return forwarded_for.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"

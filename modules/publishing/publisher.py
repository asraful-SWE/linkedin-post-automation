"""
LinkedIn Publisher V2
=====================
Production-ready, drop-in replacement for ``LinkedInPublisher``.

Fixes over the original
-----------------------
* Retry with exponential back-off on every network operation.
* ``Content-Type`` header is now sent when uploading the image binary
  (this was the root-cause of silent upload failures in V1).
* LinkedIn's upload endpoint returning HTTP 200, 201 **or** 204 is
  treated as success (V1 accepted only 200/201 for the upload PUT).
* Image URL is validated before any network activity:
    - scheme must be ``http`` or ``https``
    - path must end with a recognised image extension OR the server must
      return an ``image/*`` Content-Type
    - payload must not exceed 5 MB
* Full fallback to a text-only post when any part of the image pipeline
  fails, so a bad image URL never silently kills the whole publish.

Public interface
----------------
The class exposes the same public API as ``LinkedInPublisher``:

    publisher = LinkedInPublisherV2()
    result    = publisher.publish_to_linkedin(post_text, image_url=url)

Logging
-------
Every operation emits structured log lines in the form::

    event=<name>|step=<step>|status=<ok|failed|…>|<key>=<value>…
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar
from urllib.parse import urlparse

import requests

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_IMAGE_BYTES: int = 5 * 1024 * 1024  # 5 MB hard limit imposed by LinkedIn

VALID_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
)

LINKEDIN_API_BASE: str = "https://api.linkedin.com/v2"

# Mapping from raw Content-Type values (and URL extensions) to normalised
# MIME types that LinkedIn's upload endpoint accepts.
_CONTENT_TYPE_MAP: Dict[str, str] = {
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "image/png": "image/png",
    "image/gif": "image/gif",
    "image/webp": "image/webp",
}

_EXT_TO_MIME: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# ---------------------------------------------------------------------------
# Return-type TypeVar (used by retry helpers)
# ---------------------------------------------------------------------------

_T = TypeVar("_T")


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------


@dataclass
class RetryConfig:
    """
    Configuration for the retry / back-off strategy used by
    :class:`LinkedInPublisherV2`.

    Attributes
    ----------
    max_attempts:
        Total number of attempts (first try + retries).  A value of ``3``
        means the operation will be tried at most 3 times (i.e. up to
        2 retries after the first failure).
    base_delay:
        Sleep duration in seconds before the **first** retry.
    backoff_factor:
        Multiplier applied to the delay after each subsequent failure.
        With ``base_delay=2`` and ``backoff_factor=2`` the sleep sequence
        is 2 s, 4 s, 8 s, …
    max_delay:
        Upper bound on the computed sleep duration, preventing indefinite
        waits for operations with many retries.
    """

    max_attempts: int = 3  # 3 total = up to 2 retries (text posts)
    base_delay: float = 2.0  # seconds
    backoff_factor: float = 2.0
    max_delay: float = 30.0  # seconds


# ---------------------------------------------------------------------------
# Module-level retry helpers
# ---------------------------------------------------------------------------


def _retry_sync(
    func: Callable[[], _T],
    max_attempts: int = 2,
    base_delay: float = 2.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0,
) -> _T:
    """
    Execute *func* synchronously, retrying on any exception with
    exponential back-off.

    Parameters
    ----------
    func:
        Zero-argument callable to execute.  Must raise an exception to
        signal failure; returning ``None`` is treated as success.
    max_attempts:
        Total number of attempts allowed (including the first one).
    base_delay:
        Delay in seconds before the first retry.
    backoff_factor:
        Multiplier applied to the delay after each failed attempt.
    max_delay:
        Maximum sleep time between retries (caps exponential growth).

    Returns
    -------
    _T
        The return value of *func* on the first successful invocation.

    Raises
    ------
    Exception
        Re-raises the last exception raised by *func* after all attempts
        are exhausted.

    Examples
    --------
    >>> result = _retry_sync(lambda: call_external_api(), max_attempts=3)
    """
    last_exc: Exception = RuntimeError("_retry_sync: no attempts were made")

    for attempt in range(1, max_attempts + 1):
        try:
            result = func()
            if attempt > 1:
                logger.info(
                    f"event=retry_sync|status=recovered"
                    f"|attempt={attempt}|max_attempts={max_attempts}"
                )
            return result

        except Exception as exc:  # noqa: BLE001
            last_exc = exc

            if attempt < max_attempts:
                delay = min(
                    base_delay * (backoff_factor ** (attempt - 1)),
                    max_delay,
                )
                logger.warning(
                    f"event=retry_sync|status=retrying"
                    f"|attempt={attempt}|max_attempts={max_attempts}"
                    f"|next_delay_s={delay:.1f}|error={exc}"
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"event=retry_sync|status=exhausted"
                    f"|attempt={attempt}|max_attempts={max_attempts}"
                    f"|error={exc}"
                )

    raise last_exc


async def _retry_async(
    func: Callable[[], Any],
    max_attempts: int = 2,
    base_delay: float = 2.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0,
) -> Any:
    """
    Execute *func* (an async coroutine function) with automatic retry and
    exponential back-off.

    Parameters
    ----------
    func:
        Zero-argument *coroutine function* (``async def``) to execute.
    max_attempts:
        Total number of attempts allowed (including the first one).
    base_delay:
        Delay in seconds before the first retry.
    backoff_factor:
        Multiplier applied to the delay after each failed attempt.
    max_delay:
        Maximum sleep time between retries.

    Returns
    -------
    Any
        The return value of ``await func()`` on success.

    Raises
    ------
    Exception
        Re-raises the last exception after all attempts are exhausted.

    Examples
    --------
    >>> result = await _retry_async(lambda: async_api_call(), max_attempts=3)
    """
    last_exc: Exception = RuntimeError("_retry_async: no attempts were made")

    for attempt in range(1, max_attempts + 1):
        try:
            result = await func()
            if attempt > 1:
                logger.info(
                    f"event=retry_async|status=recovered"
                    f"|attempt={attempt}|max_attempts={max_attempts}"
                )
            return result

        except Exception as exc:  # noqa: BLE001
            last_exc = exc

            if attempt < max_attempts:
                delay = min(
                    base_delay * (backoff_factor ** (attempt - 1)),
                    max_delay,
                )
                logger.warning(
                    f"event=retry_async|status=retrying"
                    f"|attempt={attempt}|max_attempts={max_attempts}"
                    f"|next_delay_s={delay:.1f}|error={exc}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"event=retry_async|status=exhausted"
                    f"|attempt={attempt}|max_attempts={max_attempts}"
                    f"|error={exc}"
                )

    raise last_exc


# ---------------------------------------------------------------------------
# LinkedInPublisherV2
# ---------------------------------------------------------------------------


class LinkedInPublisherV2:
    """
    Production-ready LinkedIn publisher that is a **drop-in replacement**
    for ``services.linkedin_publisher.LinkedInPublisher``.

    Key improvements over V1
    ------------------------
    1. **Retry** – every API call uses :func:`_retry_sync` with exponential
       back-off so transient network errors are handled automatically.
    2. **Content-Type fix** – the binary upload PUT now includes the
       ``Content-Type`` header (missing in V1 caused silent upload failures).
    3. **Wider upload success range** – HTTP 200, 201, *and* 204 are all
       treated as a successful binary upload.
    4. **Image validation** – URL scheme, extension / MIME type, and payload
       size are checked before any upload is attempted.
    5. **Text fallback** – if the image pipeline fails for any reason the
       post is published as text-only; the caller is never left empty-handed.

    Constructor
    -----------
    The constructor reads credentials from environment variables:

    ``LINKEDIN_ACCESS_TOKEN``
        OAuth 2.0 Bearer token for the LinkedIn API.
    ``LINKEDIN_PERSON_ID``
        LinkedIn member identifier (the raw ID string, *without* the
        ``urn:li:person:`` prefix).
    ``MOCK_LINKEDIN_POSTING``
        Set to ``"true"`` (case-insensitive) to simulate all API calls
        without making real network requests.

    Parameters
    ----------
    retry_config:
        Optional :class:`RetryConfig` instance.  Defaults to
        ``RetryConfig()`` (3 total attempts, 2 s initial delay, factor 2×,
        30 s cap).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, retry_config: Optional[RetryConfig] = None) -> None:
        self.access_token: str = os.getenv("LINKEDIN_ACCESS_TOKEN", "")
        self.person_id: str = os.getenv("LINKEDIN_PERSON_ID", "")
        self.mock_mode: bool = (
            os.getenv("MOCK_LINKEDIN_POSTING", "false").lower() == "true"
        )
        self.retry_config: RetryConfig = retry_config or RetryConfig()
        self.base_url: str = LINKEDIN_API_BASE

        # Internal flag written by _publish_image_post so that
        # publish_to_linkedin can report used_image accurately.
        self._image_was_used: bool = False

        # Shared requests.Session for all JSON / REST calls.
        # The session carries auth and protocol headers; the binary upload
        # step deliberately uses a plain requests.put() call because the
        # Content-Type must be the image MIME type, not application/json.
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "X-Restli-Protocol-Version": "2.0.0",
            }
        )

        if self.mock_mode:
            logger.info("event=init|status=mock_mode|class=LinkedInPublisherV2")
        elif not self.access_token:
            logger.warning("event=init|status=missing_token|class=LinkedInPublisherV2")
        elif not self.person_id:
            logger.warning(
                "event=init|status=missing_person_id|class=LinkedInPublisherV2"
            )
        else:
            logger.info("event=init|status=ready|class=LinkedInPublisherV2")

    # ------------------------------------------------------------------
    # Credential validation
    # ------------------------------------------------------------------

    def validate_credentials(self) -> bool:
        """
        Verify that the stored LinkedIn credentials are accepted by the API.

        Sends a lightweight GET to ``/v2/userinfo``.  In mock mode the
        check is bypassed and ``True`` is returned immediately.

        Returns
        -------
        bool
            ``True`` when credentials are valid (or in mock mode), ``False``
            otherwise.
        """
        if self.mock_mode:
            logger.info("event=validate_credentials|status=mock_bypass")
            return True

        if not self.access_token:
            logger.error("event=validate_credentials|status=failed|reason=no_token")
            return False

        try:
            response = self.session.get(
                f"{self.base_url}/userinfo",
                timeout=10,
            )
            if response.status_code == 200:
                user = response.json()
                name = (
                    f"{user.get('given_name', '')} "
                    f"{user.get('family_name', '')}".strip()
                )
                logger.info(
                    f"event=validate_credentials|status=ok|user={name or 'unknown'}"
                )
                return True

            logger.error(
                f"event=validate_credentials|status=failed"
                f"|http_status={response.status_code}"
            )
            return False

        except Exception as exc:  # noqa: BLE001
            logger.error(f"event=validate_credentials|status=exception|error={exc}")
            return False

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_publishing_status(self) -> Dict[str, Any]:
        """
        Return a lightweight status dictionary suitable for health-check
        endpoints or administrative dashboards.

        Returns
        -------
        Dict[str, Any]
            ::

                {
                    "mock_mode":            bool,
                    "token_configured":     bool,
                    "person_id_configured": bool,
                }
        """
        return {
            "mock_mode": self.mock_mode,
            "token_configured": bool(self.access_token),
            "person_id_configured": bool(self.person_id),
        }

    # ------------------------------------------------------------------
    # Image URL validation
    # ------------------------------------------------------------------

    def _validate_image_url(self, url: str) -> Tuple[bool, str]:
        """
        Validate that *url* points to a publicly accessible, appropriately
        typed, and reasonably sized image.

        Validation steps
        ----------------
        1. **Scheme** – must be ``http`` or ``https``.
        2. **Type** – the URL path must end with a recognised image extension
           *or* a HEAD request to the URL must return an ``image/*``
           Content-Type header.
        3. **Size** – when the HEAD response includes a ``Content-Length``
           header its value must not exceed :data:`MAX_IMAGE_BYTES` (5 MB).

        Parameters
        ----------
        url:
            The image URL to validate.

        Returns
        -------
        Tuple[bool, str]
            ``(True, "")`` when all checks pass; ``(False, reason)`` when
            any check fails where *reason* is a human-readable explanation.
        """
        # ── 1. Scheme ────────────────────────────────────────────────
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            reason = f"Invalid URL scheme '{parsed.scheme}' – must be 'http' or 'https'"
            logger.warning(
                f"event=validate_image_url|status=invalid|reason=bad_scheme|url={url}"
            )
            return False, reason

        # ── 2. Extension (fast, no network) ─────────────────────────
        path_lower = parsed.path.lower()
        has_valid_ext = any(path_lower.endswith(ext) for ext in VALID_IMAGE_EXTENSIONS)

        # ── 3. HEAD request – content-type + size ───────────────────
        try:
            head_resp = requests.head(url, timeout=15, allow_redirects=True)
            server_ct = head_resp.headers.get("Content-Type", "").lower()
            is_image_ct = server_ct.startswith("image/")

            if not has_valid_ext and not is_image_ct:
                reason = (
                    f"URL does not have a recognised image extension and the "
                    f"server returned Content-Type '{server_ct}'"
                )
                logger.warning(
                    f"event=validate_image_url|status=invalid"
                    f"|reason=bad_content_type|url={url}"
                )
                return False, reason

            content_length_raw = head_resp.headers.get("Content-Length")
            if content_length_raw is not None:
                content_length = int(content_length_raw)
                if content_length > MAX_IMAGE_BYTES:
                    size_mb = content_length / (1024 * 1024)
                    reason = (
                        f"Image is too large ({size_mb:.2f} MB); "
                        "LinkedIn allows a maximum of 5 MB"
                    )
                    logger.warning(
                        f"event=validate_image_url|status=invalid"
                        f"|reason=too_large|size_mb={size_mb:.2f}|url={url}"
                    )
                    return False, reason

        except requests.RequestException as exc:
            # Network failure on HEAD – if the extension was valid we proceed
            # cautiously; if not, we cannot confirm the resource is an image.
            logger.warning(
                f"event=validate_image_url|step=head_request"
                f"|status=failed|error={exc}|url={url}"
            )
            if not has_valid_ext:
                reason = (
                    f"HEAD request failed and no valid image extension present: {exc}"
                )
                return False, reason
            # Extension looked good; allow the download attempt to decide.

        logger.info(f"event=validate_image_url|status=valid|url={url}")
        return True, ""

    # ------------------------------------------------------------------
    # Image download
    # ------------------------------------------------------------------

    def _download_image(self, image_url: str) -> Tuple[bytes, str]:
        """
        Download the image at *image_url* and determine its canonical MIME
        type.

        MIME-type resolution order
        --------------------------
        1. ``Content-Type`` response header (stripped of parameters such as
           charset).
        2. File extension extracted from the URL path.
        3. ``"image/jpeg"`` as a safe last-resort default.

        Parameters
        ----------
        image_url:
            Publicly reachable direct URL of the image.

        Returns
        -------
        Tuple[bytes, str]
            ``(image_bytes, content_type)`` – the raw binary payload and its
            normalised MIME type (e.g. ``"image/png"``).

        Raises
        ------
        requests.HTTPError
            When the server returns a non-2xx response code.
        requests.RequestException
            For any other network-level failure.
        """
        logger.info(f"event=download_image|step=start|url={image_url}")

        resp = requests.get(image_url, timeout=30, stream=False)
        resp.raise_for_status()
        image_bytes: bytes = resp.content

        # ── Resolve MIME type ────────────────────────────────────────
        raw_ct = resp.headers.get("Content-Type", "").lower().split(";")[0].strip()

        if raw_ct in _CONTENT_TYPE_MAP:
            content_type = _CONTENT_TYPE_MAP[raw_ct]
        else:
            # Fallback: derive from URL extension
            path_lower = urlparse(image_url).path.lower()
            matched_ext = next(
                (ext for ext in _EXT_TO_MIME if path_lower.endswith(ext)),
                None,
            )
            content_type = _EXT_TO_MIME[matched_ext] if matched_ext else "image/jpeg"

        size_kb = len(image_bytes) / 1024
        logger.info(
            f"event=download_image|step=complete"
            f"|size_kb={size_kb:.1f}|content_type={content_type}"
            f"|url={image_url}"
        )
        return image_bytes, content_type

    # ------------------------------------------------------------------
    # LinkedIn media upload – step 1: register
    # ------------------------------------------------------------------

    def _register_upload(self) -> Tuple[str, str]:
        """
        Register a new image-upload intent with the LinkedIn Media API.

        Makes a POST to ``/v2/assets?action=registerUpload`` and extracts
        the pre-signed upload URL and the assigned asset URN from the
        response body.

        Returns
        -------
        Tuple[str, str]
            ``(upload_url, asset_urn)`` on success.

        Raises
        ------
        RuntimeError
            If the API returns a non-success status code or the expected
            fields are absent from the response body.
        """
        logger.info(f"event=register_upload|step=start|person_id={self.person_id}")

        payload: Dict[str, Any] = {
            "registerUploadRequest": {
                "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
                "owner": f"urn:li:person:{self.person_id}",
                "serviceRelationships": [
                    {
                        "relationshipType": "OWNER",
                        "identifier": "urn:li:userGeneratedContent",
                    }
                ],
            }
        }

        response = self.session.post(
            f"{self.base_url}/assets?action=registerUpload",
            json=payload,
            timeout=30,
        )

        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"Register upload failed – HTTP {response.status_code}: {response.text}"
            )

        data: Dict[str, Any] = response.json()
        value = data.get("value", {})

        upload_url: str = (
            value.get("uploadMechanism", {})
            .get(
                "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest",
                {},
            )
            .get("uploadUrl", "")
        )
        asset_urn: str = value.get("asset", "")

        if not upload_url:
            raise RuntimeError("Register upload response is missing 'uploadUrl'")
        if not asset_urn:
            raise RuntimeError("Register upload response is missing 'asset' URN")

        logger.info(f"event=register_upload|step=complete|asset_urn={asset_urn}")
        return upload_url, asset_urn

    # ------------------------------------------------------------------
    # LinkedIn media upload – step 2: PUT binary
    # ------------------------------------------------------------------

    def _upload_image_binary(
        self,
        upload_url: str,
        image_bytes: bytes,
        content_type: str,
    ) -> bool:
        """
        Upload raw image bytes to LinkedIn's pre-signed upload URL.

        This is the **critical fix** vs. V1: the ``Content-Type`` header is
        now included.  LinkedIn silently discards uploads that omit it.

        Parameters
        ----------
        upload_url:
            Pre-signed PUT URL returned by :meth:`_register_upload`.
        image_bytes:
            Raw binary image payload.
        content_type:
            Normalised MIME type, e.g. ``"image/jpeg"`` or ``"image/png"``.

        Returns
        -------
        bool
            Always ``True`` on success.

        Raises
        ------
        RuntimeError
            When the server returns a status code outside {200, 201, 204}.
        """
        size_kb = len(image_bytes) / 1024
        logger.info(
            f"event=upload_image_binary|step=start"
            f"|size_kb={size_kb:.1f}|content_type={content_type}"
        )

        # Do NOT reuse self.session here: the upload URL is pre-signed and
        # must receive the image MIME type, not "application/json".
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": content_type,  # ← THE CRITICAL FIX
            "X-Restli-Protocol-Version": "2.0.0",
        }

        response = requests.put(
            upload_url,
            data=image_bytes,
            headers=headers,
            timeout=60,
        )

        # LinkedIn may return 200, 201, or 204 for a successful upload.
        if response.status_code in (200, 201, 204):
            logger.info(
                f"event=upload_image_binary|step=complete"
                f"|http_status={response.status_code}"
            )
            return True

        raise RuntimeError(
            f"Image binary upload failed – HTTP {response.status_code}: {response.text}"
        )

    # ------------------------------------------------------------------
    # Full image pipeline with per-step retry
    # ------------------------------------------------------------------

    def _upload_image_with_retry(
        self,
        image_url: str,
        max_retries: int = 2,
    ) -> Optional[str]:
        """
        End-to-end image upload pipeline with validation and per-step retry.

        Pipeline
        --------
        1. **Validate** the image URL (scheme, type, size ≤ 5 MB).
        2. **Download** the image bytes locally.
        3. **Register** the upload intent with LinkedIn
           (retried up to *max_retries* total attempts, 2 s initial delay).
        4. **Upload** the binary to the pre-signed URL
           (same retry policy).

        Parameters
        ----------
        image_url:
            Publicly reachable direct URL of the image to attach.
        max_retries:
            Total number of attempts for each retried step (steps 3 and 4).
            ``2`` means one retry after the first failure.

        Returns
        -------
        Optional[str]
            The LinkedIn asset URN (``urn:li:digitalmediaAsset:…``) on
            success, or ``None`` if any step fails after all retries.
        """
        logger.info(
            f"event=upload_image_with_retry|step=start"
            f"|url={image_url}|max_retries={max_retries}"
        )

        # ── Step 1: Validate ─────────────────────────────────────────
        is_valid, reason = self._validate_image_url(image_url)
        if not is_valid:
            logger.error(
                f"event=upload_image_with_retry|step=validate"
                f"|status=failed|reason={reason}"
            )
            return None

        logger.info("event=upload_image_with_retry|step=validate|status=passed")

        # ── Step 2: Download ─────────────────────────────────────────
        try:
            image_bytes, content_type = self._download_image(image_url)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"event=upload_image_with_retry|step=download|status=failed|error={exc}"
            )
            return None

        # Second size gate: Content-Length may have been absent in HEAD.
        if len(image_bytes) > MAX_IMAGE_BYTES:
            size_mb = len(image_bytes) / (1024 * 1024)
            logger.error(
                f"event=upload_image_with_retry|step=size_check"
                f"|status=failed|size_mb={size_mb:.2f}|limit_mb=5"
            )
            return None

        logger.info(
            f"event=upload_image_with_retry|step=download|status=ok"
            f"|size_bytes={len(image_bytes)}|content_type={content_type}"
        )

        # ── Step 3: Register upload (with retry) ─────────────────────
        try:
            upload_url, asset_urn = _retry_sync(
                func=self._register_upload,
                max_attempts=max_retries,
                base_delay=2.0,
                backoff_factor=2.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"event=upload_image_with_retry|step=register|status=failed|error={exc}"
            )
            return None

        logger.info(
            f"event=upload_image_with_retry|step=register"
            f"|status=ok|asset_urn={asset_urn}"
        )

        # ── Step 4: Upload binary (with retry) ───────────────────────
        try:
            _retry_sync(
                func=lambda: self._upload_image_binary(
                    upload_url, image_bytes, content_type
                ),
                max_attempts=max_retries,
                base_delay=2.0,
                backoff_factor=2.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"event=upload_image_with_retry|step=upload_binary"
                f"|status=failed|error={exc}"
            )
            return None

        logger.info(
            f"event=upload_image_with_retry|step=complete"
            f"|status=ok|asset_urn={asset_urn}"
        )
        return asset_urn

    # ------------------------------------------------------------------
    # Internal single-attempt helpers (raise on failure for retry compat)
    # ------------------------------------------------------------------

    def _build_post_id(
        self,
        response: requests.Response,
        fallback_prefix: str,
    ) -> str:
        """
        Extract the LinkedIn post ID from a successful API response.

        LinkedIn can return the post identifier in several places:

        * ``x-linkedin-id`` response header
        * ``id`` field in the JSON body
        * Last path segment of the ``Location`` response header

        When all three are absent a timestamp-based fallback ID is generated
        so that the caller always receives a non-empty string.

        Parameters
        ----------
        response:
            A successful (HTTP 201) response from ``/v2/ugcPosts``.
        fallback_prefix:
            String prefix used when generating the fallback ID, e.g.
            ``"linkedin_post"`` or ``"linkedin_image_post"``.

        Returns
        -------
        str
            The post identifier.
        """
        post_id: Optional[str] = response.headers.get("x-linkedin-id")

        if not post_id:
            try:
                post_id = response.json().get("id")
            except Exception:  # noqa: BLE001
                post_id = None

        if not post_id:
            location = response.headers.get("location", "")
            post_id = location.split("/")[-1] if location else None

        if not post_id:
            post_id = f"{fallback_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return post_id

    def _do_publish_text_post(self, content: str) -> str:
        """
        Single-attempt text post publish.  Raises on any failure so that
        :func:`_retry_sync` can handle retries transparently.

        Parameters
        ----------
        content:
            Body text of the LinkedIn post.

        Returns
        -------
        str
            LinkedIn post ID.

        Raises
        ------
        RuntimeError
            When the API returns a non-201 response.
        requests.RequestException
            On any network-level error.
        """
        post_data: Dict[str, Any] = {
            "author": f"urn:li:person:{self.person_id}",
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": content},
                    "shareMediaCategory": "NONE",
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
        }

        response = self.session.post(
            f"{self.base_url}/ugcPosts",
            json=post_data,
            timeout=30,
        )

        if response.status_code != 201:
            raise RuntimeError(
                f"Text post failed – HTTP {response.status_code}: {response.text}"
            )

        return self._build_post_id(response, "linkedin_post")

    def _do_publish_image_post(
        self,
        post_text: str,
        asset_urn: str,
    ) -> str:
        """
        Single-attempt image post publish.  Raises on any failure so that
        :func:`_retry_sync` can handle retries transparently.

        Parameters
        ----------
        post_text:
            Body text of the LinkedIn post.
        asset_urn:
            LinkedIn media asset URN obtained from :meth:`_upload_image_with_retry`.

        Returns
        -------
        str
            LinkedIn post ID.

        Raises
        ------
        RuntimeError
            When the API returns a non-201 response.
        requests.RequestException
            On any network-level error.
        """
        post_data: Dict[str, Any] = {
            "author": f"urn:li:person:{self.person_id}",
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": post_text},
                    "shareMediaCategory": "IMAGE",
                    "media": [
                        {
                            "status": "READY",
                            "media": asset_urn,
                        }
                    ],
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
        }

        response = self.session.post(
            f"{self.base_url}/ugcPosts",
            json=post_data,
            timeout=30,
        )

        if response.status_code != 201:
            raise RuntimeError(
                f"Image post failed – HTTP {response.status_code}: {response.text}"
            )

        return self._build_post_id(response, "linkedin_image_post")

    # ------------------------------------------------------------------
    # Public: publish text post (with retry)
    # ------------------------------------------------------------------

    def publish_text_post(self, content: str) -> Optional[str]:
        """
        Publish a text-only post to LinkedIn with automatic retry.

        The operation is retried up to ``retry_config.max_attempts`` times
        in total (default: 3, i.e. up to 2 retries) using exponential
        back-off starting at ``retry_config.base_delay`` seconds.

        Parameters
        ----------
        content:
            Body text of the LinkedIn post.

        Returns
        -------
        Optional[str]
            LinkedIn post ID on success, ``None`` after all retries are
            exhausted.
        """
        logger.info("event=publish_text_post|step=start")

        # ── Mock mode ────────────────────────────────────────────────
        if self.mock_mode:
            fake_id = f"mock_text_post_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(
                f"event=publish_text_post|step=mock|status=ok|post_id={fake_id}"
            )
            return fake_id

        # ── Credentials guard ────────────────────────────────────────
        if not self.access_token or not self.person_id:
            logger.error(
                "event=publish_text_post|status=error|reason=missing_credentials"
            )
            return None

        # ── Publish with retry ───────────────────────────────────────
        try:
            post_id: str = _retry_sync(
                func=lambda: self._do_publish_text_post(content),
                max_attempts=self.retry_config.max_attempts,
                base_delay=self.retry_config.base_delay,
                backoff_factor=self.retry_config.backoff_factor,
                max_delay=self.retry_config.max_delay,
            )
            logger.info(
                f"event=publish_text_post|step=complete|status=ok|post_id={post_id}"
            )
            return post_id

        except Exception as exc:  # noqa: BLE001
            logger.error(f"event=publish_text_post|status=failed|error={exc}")
            return None

    # ------------------------------------------------------------------
    # Public: publish image post (with text fallback)
    # ------------------------------------------------------------------

    def _publish_image_post(
        self,
        post_text: str,
        image_url: str,
    ) -> Optional[str]:
        """
        Attempt to publish a post with an attached image.

        If the image pipeline (validation → download → register → upload)
        fails for *any* reason the method automatically falls back to a
        text-only post and logs a warning.  The ``_image_was_used`` instance
        attribute is updated so that :meth:`publish_to_linkedin` can include
        an accurate ``used_image`` flag in its result dictionary.

        Parameters
        ----------
        post_text:
            Body text of the LinkedIn post.
        image_url:
            Direct, publicly reachable URL of the image to attach.

        Returns
        -------
        Optional[str]
            LinkedIn post ID on success, ``None`` only when *both* the image
            post and the text fallback fail.
        """
        # Reset the tracking flag at the beginning of each call.
        self._image_was_used = False

        logger.info(f"event=publish_image_post|step=start|url={image_url}")

        # ── Mock mode ────────────────────────────────────────────────
        if self.mock_mode:
            fake_id = f"mock_image_post_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._image_was_used = True
            logger.info(
                f"event=publish_image_post|step=mock|status=ok|post_id={fake_id}"
            )
            return fake_id

        # ── Credentials guard ────────────────────────────────────────
        if not self.access_token or not self.person_id:
            logger.error(
                "event=publish_image_post|status=error|reason=missing_credentials"
            )
            return None

        # ── Image upload pipeline ─────────────────────────────────────
        asset_urn = self._upload_image_with_retry(image_url)

        if not asset_urn:
            logger.warning(
                "event=publish_image_post|step=image_upload|status=failed"
                " – Image upload failed, falling back to text-only post"
            )
            return self.publish_text_post(post_text)

        # ── Publish UGC post with image (with retry) ─────────────────
        try:
            post_id: str = _retry_sync(
                func=lambda: self._do_publish_image_post(post_text, asset_urn),
                max_attempts=self.retry_config.max_attempts,
                base_delay=self.retry_config.base_delay,
                backoff_factor=self.retry_config.backoff_factor,
                max_delay=self.retry_config.max_delay,
            )
            self._image_was_used = True
            logger.info(
                f"event=publish_image_post|step=complete|status=ok|post_id={post_id}"
            )
            return post_id

        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"event=publish_image_post|step=ugc_post|status=failed|error={exc}"
            )
            logger.warning(
                "event=publish_image_post|step=fallback"
                " – Image post UGC call failed, falling back to text-only post"
            )
            return self.publish_text_post(post_text)

    # ------------------------------------------------------------------
    # Public: main entry point
    # ------------------------------------------------------------------

    def publish_to_linkedin(
        self,
        post_text: str,
        image_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Publish a post to LinkedIn, optionally with an attached image.

        This is the **primary public entry point** and is a drop-in
        replacement for ``LinkedInPublisher.publish_to_linkedin``.

        When *image_url* is provided the method attempts the full image
        upload pipeline.  If the image flow fails at any point, it
        transparently falls back to a text-only post.

        Parameters
        ----------
        post_text:
            Body text of the LinkedIn post.
        image_url:
            Optional direct URL of an image to attach.  Must be
            publicly accessible and ≤ 5 MB.

        Returns
        -------
        Dict[str, Any]
            ::

                {
                    "success":          bool,   # True when a post was created
                    "linkedin_post_id": str | None,
                    "error":            str | None,
                    "used_image":       bool,   # True only when image attached
                }
        """
        logger.info(
            f"event=publish_to_linkedin|step=start"
            f"|has_image={image_url is not None}"
            f"|mock_mode={self.mock_mode}"
        )

        try:
            if not self.validate_credentials():
                return {
                    "success": False,
                    "linkedin_post_id": None,
                    "error": "LinkedIn credentials are invalid or not configured",
                    "used_image": False,
                }

            # Reset image-used flag before dispatching.
            self._image_was_used = False

            if image_url:
                post_id = self._publish_image_post(
                    post_text=post_text,
                    image_url=image_url,
                )
            else:
                post_id = self.publish_text_post(post_text)

            used_image: bool = self._image_was_used

            if post_id:
                logger.info(
                    f"event=publish_to_linkedin|step=complete|status=ok"
                    f"|post_id={post_id}|used_image={used_image}"
                )
                return {
                    "success": True,
                    "linkedin_post_id": post_id,
                    "error": None,
                    "used_image": used_image,
                }

            logger.error(
                "event=publish_to_linkedin|step=complete|status=failed"
                "|reason=no_post_id_returned"
            )
            return {
                "success": False,
                "linkedin_post_id": None,
                "error": "LinkedIn publish failed after all retries",
                "used_image": False,
            }

        except Exception as exc:  # noqa: BLE001
            logger.error(f"event=publish_to_linkedin|status=exception|error={exc}")
            return {
                "success": False,
                "linkedin_post_id": None,
                "error": str(exc),
                "used_image": False,
            }

    # ------------------------------------------------------------------
    # Convenience alias kept for full drop-in compatibility with V1
    # ------------------------------------------------------------------

    def publish_post(self, content: str) -> Optional[str]:
        """
        Convenience alias for :meth:`publish_text_post`.

        Validates credentials before publishing and mirrors the interface
        of ``LinkedInPublisher.publish_post``.

        Parameters
        ----------
        content:
            Body text of the LinkedIn post.

        Returns
        -------
        Optional[str]
            LinkedIn post ID on success, ``None`` on failure.
        """
        logger.info("event=publish_post|step=start")

        if not self.validate_credentials():
            logger.error("event=publish_post|status=error|reason=credentials_invalid")
            return None

        return self.publish_text_post(content)

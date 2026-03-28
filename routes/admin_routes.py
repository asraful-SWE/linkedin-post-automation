"""
Admin Routes - Protected endpoints for system management.

All routes require the ``X-Admin-Key`` HTTP header to match the
``ADMIN_API_KEY`` environment variable.  The comparison is performed with
``hmac.compare_digest`` to prevent timing-oracle attacks.

Mount in main.py::

    from routes.admin_routes import router as admin_router
    app.include_router(admin_router)
"""

from __future__ import annotations

import hmac
import logging
import os
import platform
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from database.models import DatabaseManager, Post
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel
from services.post_generator import PostGenerator
from services.topic_engine import TopicEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class ForceGenerateRequest(BaseModel):
    """Request body for POST /admin/force-generate."""

    topic: Optional[str] = None
    goal: str = "educational"
    use_image: bool = False


class ForceGenerateResponse(BaseModel):
    success: bool
    post_id: Optional[int] = None
    topic: Optional[str] = None
    content_preview: Optional[str] = None
    score: Optional[float] = None
    error: Optional[str] = None


class RetryFailedResponse(BaseModel):
    retried_count: int
    post_ids: List[int]


class DeletePostResponse(BaseModel):
    success: bool
    post_id: int


class TopicWeightsResponse(BaseModel):
    success: bool
    message: str


# ---------------------------------------------------------------------------
# Security dependency
# ---------------------------------------------------------------------------


async def verify_admin_key(x_admin_key: str = Header(None)) -> None:
    """FastAPI dependency — validates the ``X-Admin-Key`` header.

    Raises:
        HTTPException 401: Header is missing or ``ADMIN_API_KEY`` env-var is
                           not configured on the server.
        HTTPException 403: Header value does not match the configured key
                           (constant-time comparison).
    """
    admin_key = os.getenv("ADMIN_API_KEY", "")

    # Fail loudly when the server has no key configured — don't silently
    # allow unauthenticated access.
    if not admin_key:
        raise HTTPException(
            status_code=401,
            detail="Admin API key not configured on this server.",
        )

    if not x_admin_key:
        raise HTTPException(
            status_code=401,
            detail="Admin API key required (send via X-Admin-Key header).",
        )

    # Constant-time comparison prevents timing-oracle attacks.
    if not hmac.compare_digest(
        x_admin_key.encode("utf-8"),
        admin_key.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid admin API key.",
        )


# ---------------------------------------------------------------------------
# Router — every route inherits the security dependency
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(verify_admin_key)],
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_db_manager(request: Request) -> DatabaseManager:
    """Extract the initialised DatabaseManager from application state."""
    db_manager: Optional[DatabaseManager] = getattr(
        request.app.state, "db_manager", None
    )
    if db_manager is None:
        raise HTTPException(
            status_code=500,
            detail="Database manager not initialised.",
        )
    return db_manager


def _env_status(var: str) -> str:
    """Return ``'configured'`` or ``'not configured'`` for an env-var — never
    its value."""
    return "configured" if os.getenv(var) else "not configured"


def _compute_content_score(content: str) -> float:
    """Heuristic content-quality score in the range 0.0 – 10.0.

    Scoring factors
    ~~~~~~~~~~~~~~~
    * Base score: **4.0**
    * Word count in the optimal 100-400 word range: **+2.0**
    * Word count in the acceptable 50-100 or 400-600 range: **+1.0**
    * Contains a question mark (audience engagement): **+0.5**
    * Contains at least one hashtag: **+0.5**
    * Contains at least one emoji: **+0.5**
    * Has ≥ 3 distinct paragraphs (structured content): **+0.5**
    * Contains common AI-generated closing phrases: **−0.5 each**
    """
    score = 4.0
    words = content.split()
    word_count = len(words)

    # Word-count bonus
    if 100 <= word_count <= 400:
        score += 2.0
    elif (50 <= word_count < 100) or (400 < word_count <= 600):
        score += 1.0

    # Engagement signal
    if "?" in content:
        score += 0.5

    # Hashtag presence
    if re.search(r"#\w+", content):
        score += 0.5

    # Emoji presence
    emoji_re = re.compile(
        r"[\U0001F600-\U0001F64F"
        r"\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF"
        r"\U00002600-\U000027BF"
        r"\U0001F900-\U0001F9FF]+",
        flags=re.UNICODE,
    )
    if emoji_re.search(content):
        score += 0.5

    # Structural variety
    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    if len(paragraphs) >= 3:
        score += 0.5

    # Penalise generic AI closing phrases
    ai_phrases = [
        "আশা করি",
        "ধন্যবাদ সবাইকে",
        "পোস্টটি ভালো লাগলে",
        "সংক্ষেপে বলতে গেলে",
        "উপসংহারে",
        "পরিশেষে",
        "I hope this",
        "Thank you for reading",
    ]
    for phrase in ai_phrases:
        if phrase.lower() in content.lower():
            score -= 0.5

    return round(max(0.0, min(10.0, score)), 2)


def _get_db_tables(db_path: str) -> List[str]:
    """Return sorted list of user-visible table names in the SQLite database."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name
                FROM   sqlite_master
                WHERE  type = 'table'
                  AND  name NOT LIKE 'sqlite_%'
                ORDER  BY name
                """
            )
            return [row[0] for row in cursor.fetchall()]
    except Exception as exc:
        logger.warning("Could not list DB tables: %s", exc)
        return []


# ---------------------------------------------------------------------------
# GET /admin/system-status
# ---------------------------------------------------------------------------


@router.get("/system-status", response_model=Dict[str, Any])
async def get_system_status(request: Request) -> Dict[str, Any]:
    """Return a comprehensive system health and configuration snapshot.

    Includes Python runtime info, environment-variable status (values are
    **never** exposed), database metadata, and active feature flags.

    Returns:
        200 — dict with keys ``runtime``, ``environment``, ``database``,
        ``feature_flags``, ``generated_at``.
    """
    db_manager = _get_db_manager(request)

    # ---- Runtime info ----
    runtime = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
    }

    # ---- Environment-variable status (NO values exposed) ----
    environment = {
        "LINKEDIN_ACCESS_TOKEN": _env_status("LINKEDIN_ACCESS_TOKEN"),
        "OPENAI_API_KEY": _env_status("OPENAI_API_KEY"),
        "SMTP_HOST": _env_status("SMTP_HOST"),
        "UNSPLASH_ACCESS_KEY": _env_status("UNSPLASH_ACCESS_KEY"),
        "PEXELS_API_KEY": _env_status("PEXELS_API_KEY"),
        "ADMIN_API_KEY": _env_status("ADMIN_API_KEY"),
        "CELERY_BROKER_URL": _env_status("CELERY_BROKER_URL"),
    }

    # ---- Database metadata ----
    db_path = db_manager.db_path
    try:
        db_size_kb = round(os.path.getsize(db_path) / 1024, 2)
    except OSError:
        db_size_kb = -1.0

    database = {
        "path": db_path,
        "size_kb": db_size_kb,
        "tables": _get_db_tables(db_path),
    }

    # ---- Feature flags ----
    feature_flags = {
        "use_celery": os.getenv("USE_CELERY", "false").lower() == "true",
        "enable_images": os.getenv("ENABLE_IMAGES", "false").lower() == "true",
        "mock_linkedin": os.getenv("MOCK_LINKEDIN_POSTING", "false").lower() == "true",
        "auto_schedule_enabled": os.getenv("AUTO_SCHEDULE_ENABLED", "true").lower()
        == "true",
    }

    return {
        "runtime": runtime,
        "environment": environment,
        "database": database,
        "feature_flags": feature_flags,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# POST /admin/force-generate
# ---------------------------------------------------------------------------


@router.post("/force-generate", response_model=ForceGenerateResponse)
async def force_generate(
    body: ForceGenerateRequest,
    request: Request,
) -> ForceGenerateResponse:
    """Force immediate post generation, bypassing the posting scheduler.

    A topic is auto-selected if none is provided.  The generated post is
    stored with ``status = 'pending'`` so it can still go through the normal
    approval workflow.  Optional image fetching is attempted when
    ``use_image=true`` and at least one image-provider API key is configured.

    Request body::

        {
            "topic":     "optional topic string",
            "goal":      "educational",    // default
            "use_image": false             // default
        }

    Returns:
        ForceGenerateResponse with ``post_id``, ``topic``,
        ``content_preview`` (first 100 chars), and ``score`` (0-10).
    """
    db_manager = _get_db_manager(request)

    try:
        # ---- 1. Topic selection ----
        if body.topic and body.topic.strip():
            selected_topic = body.topic.strip()
            logger.info("force_generate | using provided topic: %s", selected_topic)
        else:
            topic_engine = TopicEngine(db_manager)
            selected_topic = topic_engine.select_topic()
            logger.info("force_generate | auto-selected topic: %s", selected_topic)

        # ---- 2. Content generation ----
        generator = PostGenerator()
        content = generator.generate_post(selected_topic)
        if not content or not content.strip():
            raise ValueError("Post generator returned empty content.")

        # ---- 3. Quality score ----
        quality_score = _compute_content_score(content)

        # ---- 4. Optional image fetch ----
        image_url: Optional[str] = None
        if body.use_image:
            unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY")
            pexels_key = os.getenv("PEXELS_API_KEY")
            if unsplash_key or pexels_key:
                try:
                    from modules.image.fetcher import ImageFetcher

                    fetcher = ImageFetcher()
                    images = fetcher.fetch_images(selected_topic, count=1)
                    if images:
                        image_url = images[0].get("url")
                        logger.info(
                            "force_generate | image fetched: %s",
                            image_url,
                        )
                except Exception as img_exc:
                    logger.warning(
                        "force_generate | image fetch failed (non-fatal): %s",
                        img_exc,
                    )
            else:
                logger.info(
                    "force_generate | use_image=True but no image API keys configured"
                )

        # ---- 5. Persist post ----
        post = Post(
            topic=selected_topic,
            content=content,
            status="pending",
            image_url=image_url,
        )
        post_id = db_manager.save_post(post)

        # ---- 6. Update new columns (post_goal, content_score) via direct SQL ----
        try:
            with sqlite3.connect(db_manager.db_path) as conn:
                conn.execute(
                    "UPDATE posts SET post_goal = ?, content_score = ? WHERE id = ?",
                    (body.goal, quality_score, post_id),
                )
                conn.commit()
        except Exception as col_exc:
            # Non-fatal: new columns might not exist yet if migrations haven't run.
            logger.warning(
                "force_generate | could not update post_goal/content_score "
                "(run migrations): %s",
                col_exc,
            )

        logger.info(
            "force_generate | post_id=%d | topic=%s | goal=%s | score=%.2f | image=%s",
            post_id,
            selected_topic,
            body.goal,
            quality_score,
            image_url or "none",
        )

        return ForceGenerateResponse(
            success=True,
            post_id=post_id,
            topic=selected_topic,
            content_preview=content[:100],
            score=quality_score,
        )

    except Exception as exc:
        logger.error("force_generate failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Post generation failed: {exc}",
        )


# ---------------------------------------------------------------------------
# POST /admin/retry-failed-posts
# ---------------------------------------------------------------------------


@router.post("/retry-failed-posts", response_model=RetryFailedResponse)
async def retry_failed_posts(request: Request) -> RetryFailedResponse:
    """Re-queue posts stuck in ``publish_failed`` or ``retry_pending`` status.

    Sets each matching post back to ``approved`` so the publisher can attempt
    delivery again.  Also increments ``retry_count`` (requires migration 4).

    Returns:
        RetryFailedResponse with ``retried_count`` and list of ``post_ids``.
    """
    db_manager = _get_db_manager(request)
    retryable_statuses = ("publish_failed", "retry_pending")

    try:
        with sqlite3.connect(db_manager.db_path) as conn:
            cursor = conn.cursor()

            # Collect candidates.
            placeholders = ",".join("?" * len(retryable_statuses))
            cursor.execute(
                f"SELECT id FROM posts WHERE status IN ({placeholders})",
                retryable_statuses,
            )
            post_ids: List[int] = [row[0] for row in cursor.fetchall()]

            if not post_ids:
                logger.info("retry_failed_posts | no posts to retry")
                return RetryFailedResponse(retried_count=0, post_ids=[])

            # Batch update: set status → approved, bump retry_count.
            # The retry_count column exists after migration 4; use a
            # safe fallback UPDATE that won't error if the column is absent.
            try:
                conn.executemany(
                    """
                    UPDATE posts
                    SET    status      = 'approved',
                           retry_count = COALESCE(retry_count, 0) + 1
                    WHERE  id          = ?
                    """,
                    [(pid,) for pid in post_ids],
                )
            except sqlite3.OperationalError:
                # retry_count column absent — update status only.
                conn.executemany(
                    "UPDATE posts SET status = 'approved' WHERE id = ?",
                    [(pid,) for pid in post_ids],
                )

            conn.commit()

        logger.info(
            "retry_failed_posts | retried=%d | post_ids=%s",
            len(post_ids),
            post_ids,
        )
        return RetryFailedResponse(retried_count=len(post_ids), post_ids=post_ids)

    except Exception as exc:
        logger.error("retry_failed_posts failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retry posts: {exc}",
        )


# ---------------------------------------------------------------------------
# DELETE /admin/posts/{post_id}
# ---------------------------------------------------------------------------


@router.delete("/posts/{post_id}", response_model=DeletePostResponse)
async def delete_post(post_id: int, request: Request) -> DeletePostResponse:
    """Soft-delete a post by setting its status to ``'deleted'``.

    Only posts with status ``'pending'`` or ``'rejected'`` may be deleted.
    Published posts are immutable through this endpoint to preserve audit
    trails and LinkedIn integrity.

    Args:
        post_id: Database ID of the post to delete.

    Returns:
        DeletePostResponse with ``success=True`` and echoed ``post_id``.

    Raises:
        404: Post not found.
        409: Post status prevents deletion (e.g. ``'published'``).
    """
    db_manager = _get_db_manager(request)
    deletable_statuses = {"pending", "rejected"}

    post = db_manager.get_post_by_id(post_id)
    if post is None:
        raise HTTPException(
            status_code=404,
            detail=f"Post {post_id} not found.",
        )

    current_status: str = post.get("status", "")
    if current_status not in deletable_statuses:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot delete post {post_id} with status '{current_status}'. "
                f"Only posts with status {sorted(deletable_statuses)} may be deleted."
            ),
        )

    db_manager.update_post_status(post_id, "deleted")
    logger.info(
        "admin_delete_post | post_id=%d | previous_status=%s", post_id, current_status
    )

    return DeletePostResponse(success=True, post_id=post_id)


# ---------------------------------------------------------------------------
# GET /admin/logs
# ---------------------------------------------------------------------------


@router.get("/logs", response_model=Dict[str, Any])
async def get_logs(
    lines: int = 100,
    level: str = "INFO",
) -> Dict[str, Any]:
    """Read the last *N* lines from the application log file.

    Query parameters:
        lines: Number of tail lines to return (default: 100, capped at 5000).
        level: Filter lines to those containing this log level string.
               Pass ``"ALL"`` or ``""`` to skip filtering (default: ``"INFO"``).

    The log file is expected at ``logs/linkedin_poster.log`` relative to the
    current working directory, matching the path configured in
    ``utils/logger.py``.

    Returns:
        Dict with ``lines`` (list of matching strings), ``total_lines``
        (total lines in file before filtering), and ``file_size_kb``.

    Raises:
        404: Log file does not exist.
        400: ``lines`` parameter out of range.
    """
    # ---- Validate parameters ----
    lines = max(1, min(lines, 5_000))  # cap between 1 and 5000
    log_path = Path("logs") / "linkedin_poster.log"

    if not log_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Log file not found at '{log_path}'. "
            "Ensure the application has written at least one log entry.",
        )

    try:
        raw_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read log file: {exc}",
        )

    all_lines = raw_text.splitlines()
    total_lines = len(all_lines)
    tail = all_lines[-lines:]

    # ---- Level filter ----
    level_upper = level.strip().upper()
    if level_upper and level_upper not in ("ALL", ""):
        tail = [ln for ln in tail if level_upper in ln]

    try:
        file_size_kb = round(log_path.stat().st_size / 1024, 2)
    except OSError:
        file_size_kb = -1.0

    return {
        "lines": tail,
        "total_lines": total_lines,
        "file_size_kb": file_size_kb,
    }


# ---------------------------------------------------------------------------
# POST /admin/update-topic-weights
# ---------------------------------------------------------------------------


@router.post("/update-topic-weights", response_model=TopicWeightsResponse)
async def update_topic_weights(request: Request) -> TopicWeightsResponse:
    """Trigger a full topic-weight refresh based on the latest engagement data.

    Instantiates a fresh :class:`TopicEngine` and calls
    ``force_topic_refresh()``, which resets all weights and re-derives them
    from the current state of the database.

    Returns:
        TopicWeightsResponse with ``success=True`` and a status message.
    """
    db_manager = _get_db_manager(request)

    try:
        topic_engine = TopicEngine(db_manager)
        topic_engine.force_topic_refresh()

        logger.info("admin_update_topic_weights | refresh complete")
        return TopicWeightsResponse(
            success=True,
            message="Topic weights refreshed successfully from latest engagement data.",
        )

    except Exception as exc:
        logger.error("admin_update_topic_weights failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Topic weight refresh failed: {exc}",
        )


# ---------------------------------------------------------------------------
# GET /admin/config
# ---------------------------------------------------------------------------


@router.get("/config", response_model=Dict[str, Any])
async def get_config(request: Request) -> Dict[str, Any]:
    """Return non-sensitive application configuration values.

    Secret keys, tokens, and credentials are **never** included.
    Uses :func:`app.config.get_settings` where available; falls back to
    reading environment variables directly so the endpoint is robust even
    when the Settings object cannot be instantiated (e.g. validation error
    on an optional field).

    Returns:
        Dict with scheduler, content, image, and infrastructure settings plus
        the active ``posting_windows``.
    """
    # ---- Attempt to load Settings ----
    try:
        from app.config import get_settings  # type: ignore

        s = get_settings()
        timezone = s.timezone
        max_posts_per_day = s.max_posts_per_day
        min_hours_between_posts = s.min_hours_between_posts
        auto_schedule_enabled = s.auto_schedule_enabled
        test_mode = s.test_mode
        mock_linkedin_posting = s.mock_linkedin_posting
        enable_images = s.enable_images
        use_celery = s.use_celery
    except Exception as settings_exc:
        logger.warning(
            "admin_get_config | settings load failed, falling back to env vars: %s",
            settings_exc,
        )
        timezone = os.getenv("TIMEZONE", "Asia/Dhaka")
        max_posts_per_day = int(os.getenv("MAX_POSTS_PER_DAY", "2"))
        min_hours_between_posts = float(os.getenv("MIN_HOURS_BETWEEN_POSTS", "4"))
        auto_schedule_enabled = (
            os.getenv("AUTO_SCHEDULE_ENABLED", "true").lower() == "true"
        )
        test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
        mock_linkedin_posting = (
            os.getenv("MOCK_LINKEDIN_POSTING", "false").lower() == "true"
        )
        enable_images = os.getenv("ENABLE_IMAGES", "false").lower() == "true"
        use_celery = os.getenv("USE_CELERY", "false").lower() == "true"

    # ---- Posting windows ----
    # Prefer live values from the scheduler instance; fall back to defaults.
    posting_windows: List[Dict[str, str]] = []
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "posting_windows"):
        for window in scheduler.posting_windows:
            try:
                posting_windows.append(
                    {
                        "start": window["start"].strftime("%H:%M"),
                        "end": window["end"].strftime("%H:%M"),
                    }
                )
            except Exception:
                pass

    if not posting_windows:
        # Default windows as defined in PostingScheduler.
        posting_windows = [
            {"start": "09:30", "end": "10:30"},
            {"start": "19:30", "end": "20:30"},
        ]

    return {
        "timezone": timezone,
        "max_posts_per_day": max_posts_per_day,
        "min_hours_between_posts": min_hours_between_posts,
        "auto_schedule_enabled": auto_schedule_enabled,
        "test_mode": test_mode,
        "mock_linkedin_posting": mock_linkedin_posting,
        "enable_images": enable_images,
        "use_celery": use_celery,
        "posting_windows": posting_windows,
    }

"""
Publishing Tasks - Background post publishing with retry and fallback.

Tasks
-----
- publish_post_task        : Publish an approved post with image-then-text-only fallback.
- retry_failed_posts_task  : Periodic sweeper that re-queues posts whose publish failed.
- send_email_task          : Deliver an approval email with exponential back-off retry.
"""

import logging
from typing import Any, Dict, List, Optional

import requests
from celery.utils.log import get_task_logger
from scheduler.tasks.celery_app import celery_app

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = get_task_logger(__name__)

# ---------------------------------------------------------------------------
# Status constants (kept here so they never diverge from each other)
# ---------------------------------------------------------------------------
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_PUBLISHED = "published"
STATUS_REJECTED = "rejected"
STATUS_PUBLISH_FAILED = "publish_failed"
STATUS_RETRY_PENDING = "retry_pending"

# Maximum posts the sweeper task processes per run (prevents thundering herd)
_SWEEPER_BATCH_LIMIT = 5


# ===========================================================================
# Task 1 – publish_post_task
# ===========================================================================


@celery_app.task(
    bind=True,
    name="scheduler.tasks.publishing_tasks.publish_post_task",
    max_retries=2,
    default_retry_delay=60,
    serializer="json",
    acks_late=True,
    track_started=True,
)
def publish_post_task(
    self,
    post_id: int,
    image_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Publish a single LinkedIn post identified by *post_id*.

    Publishing Strategy
    -------------------
    Attempt 1  – publish **with image** (only when ``image_url`` is supplied).
    Attempt 2  – publish **text-only** as a fallback when the image attempt
                 fails or no image was provided.

    Retry behaviour
    ---------------
    On :class:`requests.exceptions.RequestException` or any ``ConnectionError``
    the task is re-queued with exponential back-off
    (``60 * 2^retry_number`` seconds) up to ``max_retries`` times.

    All other unrecoverable failures mark the post ``publish_failed`` and
    return without retrying.

    Returns
    -------
    dict with keys: success, post_id, linkedin_post_id, used_image, attempts, error
    """
    task_attempt: int = self.request.retries + 1

    logger.info(
        "task=publish_post|post_id=%d|image_url=%s|status=started|attempt=%d",
        post_id,
        image_url or "none",
        task_attempt,
    )

    # ------------------------------------------------------------------
    # Resolve dependencies
    # ------------------------------------------------------------------
    try:
        from database.models import DatabaseManager

        try:
            from modules.publishing.publisher import (
                LinkedInPublisherV2 as LinkedInPublisher,
            )
        except ImportError:
            from services.linkedin_publisher import LinkedInPublisher

        db = DatabaseManager()
        publisher = LinkedInPublisher()
    except Exception as init_exc:  # noqa: BLE001
        logger.error(
            "task=publish_post|post_id=%d|status=init_error|error=%s",
            post_id,
            init_exc,
        )
        return {
            "success": False,
            "post_id": post_id,
            "linkedin_post_id": None,
            "used_image": False,
            "attempts": task_attempt,
            "error": f"Dependency initialisation failed: {init_exc}",
        }

    # ------------------------------------------------------------------
    # Fetch the post record
    # ------------------------------------------------------------------
    post: Optional[Dict[str, Any]] = db.get_post_by_id(post_id)

    if not post:
        logger.error("task=publish_post|post_id=%d|status=not_found", post_id)
        return {
            "success": False,
            "post_id": post_id,
            "linkedin_post_id": None,
            "used_image": False,
            "attempts": task_attempt,
            "error": "Post not found",
        }

    if post["status"] == STATUS_PUBLISHED:
        logger.warning(
            "task=publish_post|post_id=%d|status=already_published – skipping",
            post_id,
        )
        return {
            "success": True,
            "post_id": post_id,
            "linkedin_post_id": post.get("linkedin_post_id"),
            "used_image": bool(post.get("image_url")),
            "attempts": task_attempt,
            "error": None,
        }

    if post["status"] == STATUS_REJECTED:
        logger.warning(
            "task=publish_post|post_id=%d|status=rejected – cannot publish rejected post",
            post_id,
        )
        return {
            "success": False,
            "post_id": post_id,
            "linkedin_post_id": None,
            "used_image": False,
            "attempts": task_attempt,
            "error": "Post has been rejected",
        }

    content: str = post["content"]
    # Prefer the explicitly passed image_url; fall back to whatever is stored
    effective_image_url: Optional[str] = image_url or post.get("image_url")

    # Mark the post as approved so downstream systems know it passed the gate
    db.update_post_status(post_id, STATUS_APPROVED)
    if effective_image_url:
        db.set_post_image_url(post_id, effective_image_url)

    publish_attempts: int = 0
    used_image: bool = False
    linkedin_post_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Attempt 1 – publish WITH image
    # ------------------------------------------------------------------
    if effective_image_url:
        publish_attempts += 1
        logger.info(
            "task=publish_post|post_id=%d|publish_attempt=%d|mode=with_image"
            "|status=attempting",
            post_id,
            publish_attempts,
        )
        try:
            result: Dict[str, Any] = publisher.publish_to_linkedin(
                post_text=content,
                image_url=effective_image_url,
            )

            if result.get("success"):
                linkedin_post_id = result.get("linkedin_post_id")
                used_image = True
                logger.info(
                    "task=publish_post|post_id=%d|publish_attempt=%d|mode=with_image"
                    "|linkedin_id=%s|status=success",
                    post_id,
                    publish_attempts,
                    linkedin_post_id,
                )
            else:
                logger.warning(
                    "task=publish_post|post_id=%d|publish_attempt=%d|mode=with_image"
                    "|status=api_failure|error=%s – falling back to text-only",
                    post_id,
                    publish_attempts,
                    result.get("error", "unknown"),
                )

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as conn_exc:
            countdown: int = 60 * (2**self.request.retries)
            logger.error(
                "task=publish_post|post_id=%d|publish_attempt=%d|mode=with_image"
                "|status=connection_error|error=%s|retrying_in=%ds",
                post_id,
                publish_attempts,
                conn_exc,
                countdown,
            )
            db.update_post_status(post_id, STATUS_RETRY_PENDING)
            raise self.retry(exc=conn_exc, countdown=countdown)

        except Exception as img_exc:  # noqa: BLE001
            logger.warning(
                "task=publish_post|post_id=%d|publish_attempt=%d|mode=with_image"
                "|status=unexpected_error|error=%s – falling back to text-only",
                post_id,
                publish_attempts,
                img_exc,
            )
            # Non-connection errors: don't retry the whole task, just fall through
            # to the text-only attempt.

    # ------------------------------------------------------------------
    # Attempt 2 – publish TEXT-ONLY (fallback / default path)
    # ------------------------------------------------------------------
    if not linkedin_post_id:
        publish_attempts += 1
        logger.info(
            "task=publish_post|post_id=%d|publish_attempt=%d|mode=text_only"
            "|status=attempting",
            post_id,
            publish_attempts,
        )
        try:
            result = publisher.publish_to_linkedin(
                post_text=content,
                image_url=None,
            )

            if result.get("success"):
                linkedin_post_id = result.get("linkedin_post_id")
                used_image = False
                logger.info(
                    "task=publish_post|post_id=%d|publish_attempt=%d|mode=text_only"
                    "|linkedin_id=%s|status=success",
                    post_id,
                    publish_attempts,
                    linkedin_post_id,
                )
            else:
                logger.error(
                    "task=publish_post|post_id=%d|publish_attempt=%d|mode=text_only"
                    "|status=api_failure|error=%s",
                    post_id,
                    publish_attempts,
                    result.get("error", "unknown"),
                )

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as conn_exc:
            countdown = 60 * (2**self.request.retries)
            logger.error(
                "task=publish_post|post_id=%d|publish_attempt=%d|mode=text_only"
                "|status=connection_error|error=%s|retrying_in=%ds",
                post_id,
                publish_attempts,
                conn_exc,
                countdown,
            )
            db.update_post_status(post_id, STATUS_RETRY_PENDING)
            raise self.retry(exc=conn_exc, countdown=countdown)

        except Exception as txt_exc:  # noqa: BLE001
            logger.error(
                "task=publish_post|post_id=%d|publish_attempt=%d|mode=text_only"
                "|status=unexpected_error|error=%s",
                post_id,
                publish_attempts,
                txt_exc,
            )

    # ------------------------------------------------------------------
    # Persist final outcome
    # ------------------------------------------------------------------
    if linkedin_post_id:
        db.set_linkedin_post_id(post_id, linkedin_post_id)
        db.update_post_status(post_id, STATUS_PUBLISHED)

        logger.info(
            "task=publish_post|post_id=%d|linkedin_id=%s|used_image=%s"
            "|publish_attempts=%d|task_attempt=%d|status=completed",
            post_id,
            linkedin_post_id,
            used_image,
            publish_attempts,
            task_attempt,
        )

        return {
            "success": True,
            "post_id": post_id,
            "linkedin_post_id": linkedin_post_id,
            "used_image": used_image,
            "attempts": publish_attempts,
            "error": None,
        }

    # Both attempts exhausted without a LinkedIn post ID
    db.update_post_status(post_id, STATUS_PUBLISH_FAILED)

    logger.error(
        "task=publish_post|post_id=%d|status=failed|publish_attempts=%d"
        "|task_attempt=%d",
        post_id,
        publish_attempts,
        task_attempt,
    )

    return {
        "success": False,
        "post_id": post_id,
        "linkedin_post_id": None,
        "used_image": False,
        "attempts": publish_attempts,
        "error": "All publish attempts exhausted – post marked publish_failed",
    }


# ===========================================================================
# Task 2 – retry_failed_posts_task  (periodic sweeper)
# ===========================================================================


@celery_app.task(
    bind=True,
    name="scheduler.tasks.publishing_tasks.retry_failed_posts_task",
    max_retries=1,
    serializer="json",
    acks_late=True,
    track_started=True,
)
def retry_failed_posts_task(self) -> Dict[str, Any]:
    """
    Periodic sweeper – re-queues posts whose last publish attempt failed.

    Queries the database for posts with status ``publish_failed`` or
    ``retry_pending`` and dispatches a fresh :func:`publish_post_task` for
    each one (up to :data:`_SWEEPER_BATCH_LIMIT` per run to avoid saturating
    the queue).

    This task is intended to be called by Celery Beat every 30 minutes; see
    :data:`celery_app.conf.beat_schedule`.

    Returns
    -------
    dict with keys: posts_retried, post_ids
    """
    logger.info("task=retry_failed_posts|status=started")

    retried_ids: List[int] = []

    try:
        from database.models import DatabaseManager

        db = DatabaseManager()

        # Collect candidates from both failure statuses
        failed_posts: List[Dict[str, Any]] = db.list_posts(status=STATUS_PUBLISH_FAILED)
        retry_pending_posts: List[Dict[str, Any]] = db.list_posts(
            status=STATUS_RETRY_PENDING
        )

        candidates: List[Dict[str, Any]] = failed_posts + retry_pending_posts

        # Deduplicate by post ID (a post may theoretically appear in both lists
        # if a race condition updated its status between the two queries)
        seen_ids: set = set()
        unique_candidates: List[Dict[str, Any]] = []
        for p in candidates:
            if p["id"] not in seen_ids:
                seen_ids.add(p["id"])
                unique_candidates.append(p)

        # Apply batch limit
        batch: List[Dict[str, Any]] = unique_candidates[:_SWEEPER_BATCH_LIMIT]

        if not batch:
            logger.info("task=retry_failed_posts|status=no_candidates|posts_retried=0")
            return {"posts_retried": 0, "post_ids": []}

        logger.info(
            "task=retry_failed_posts|candidates_total=%d|batch_size=%d"
            "|status=dispatching",
            len(unique_candidates),
            len(batch),
        )

        for post in batch:
            pid: int = int(post["id"])

            # Reset status to retry_pending so we know it was picked up
            db.update_post_status(pid, STATUS_RETRY_PENDING)

            # Re-use the stored image URL if available
            stored_image: Optional[str] = post.get("image_url")

            publish_post_task.delay(
                post_id=pid,
                image_url=stored_image,
            )
            retried_ids.append(pid)

            logger.info(
                "task=retry_failed_posts|post_id=%d|image_url=%s"
                "|status=queued_for_retry",
                pid,
                stored_image or "none",
            )

    except Exception as exc:  # noqa: BLE001
        logger.error("task=retry_failed_posts|status=sweep_error|error=%s", exc)
        # Non-fatal for the sweeper – return whatever we managed before failing
        return {"posts_retried": len(retried_ids), "post_ids": retried_ids}

    logger.info(
        "task=retry_failed_posts|posts_retried=%d|post_ids=%s|status=completed",
        len(retried_ids),
        retried_ids,
    )

    return {"posts_retried": len(retried_ids), "post_ids": retried_ids}


# ===========================================================================
# Task 3 – send_email_task
# ===========================================================================


@celery_app.task(
    bind=True,
    name="scheduler.tasks.publishing_tasks.send_email_task",
    max_retries=3,
    default_retry_delay=300,  # 5 minutes base delay
    serializer="json",
    acks_late=True,
    track_started=True,
)
def send_email_task(
    self,
    post_id: int,
    topic: str,
    content: str,
    token: str,
) -> Dict[str, Any]:
    """
    Deliver a post-approval email for *post_id* in the background.

    The task delegates to :meth:`EmailService.send_post_approval_email` and
    retries up to 3 times on failure using exponential back-off
    (300 s → 600 s → 1200 s).

    Parameters
    ----------
    post_id:
        Database ID of the post awaiting approval.
    topic:
        Human-readable topic label shown in the email subject/body.
    content:
        Full post text to embed in the email.
    token:
        One-time approval token used to generate approve/reject URLs.

    Returns
    -------
    dict with keys: success, post_id, attempt, error
    """
    attempt_number: int = self.request.retries + 1

    logger.info(
        "task=send_email|post_id=%d|topic=%s|status=started|attempt=%d",
        post_id,
        topic,
        attempt_number,
    )

    try:
        from services.email_service import EmailService

        email_service = EmailService()

        if not email_service.is_configured():
            logger.warning(
                "task=send_email|post_id=%d|status=not_configured"
                "|reason=EmailService.is_configured() returned False",
                post_id,
            )
            return {
                "success": False,
                "post_id": post_id,
                "attempt": attempt_number,
                "error": "Email service is not configured",
            }

        sent: bool = email_service.send_post_approval_email(
            post_id=post_id,
            topic=topic,
            content=content,
            token=token,
        )

        if sent:
            logger.info(
                "task=send_email|post_id=%d|topic=%s|attempt=%d|status=completed",
                post_id,
                topic,
                attempt_number,
            )
            return {
                "success": True,
                "post_id": post_id,
                "attempt": attempt_number,
                "error": None,
            }

        # Service returned False without raising – treat as a soft failure and retry
        raise RuntimeError(
            f"send_post_approval_email returned False for post_id={post_id}"
        )

    except Exception as exc:  # noqa: BLE001
        countdown: int = 300 * (2**self.request.retries)  # 300, 600, 1200 s

        logger.error(
            "task=send_email|post_id=%d|topic=%s|attempt=%d"
            "|status=failed|error=%s|retrying_in=%ds",
            post_id,
            topic,
            attempt_number,
            exc,
            countdown,
        )

        # After the final retry exhaustion Celery will mark the task as FAILURE;
        # we still want a structured log entry for observability.
        if self.request.retries >= self.max_retries:
            logger.error(
                "task=send_email|post_id=%d|topic=%s|status=permanently_failed"
                "|total_attempts=%d",
                post_id,
                topic,
                attempt_number,
            )
            return {
                "success": False,
                "post_id": post_id,
                "attempt": attempt_number,
                "error": str(exc),
            }

        raise self.retry(exc=exc, countdown=countdown)

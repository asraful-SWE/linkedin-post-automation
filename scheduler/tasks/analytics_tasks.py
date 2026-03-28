"""
Analytics Tasks - Background analytics processing.

Tasks
-----
- update_post_analytics_task        : Pull LinkedIn analytics for a single post and persist them.
- bulk_analytics_update_task        : Periodic (daily) – fan-out analytics refresh for all
                                      published posts from the last 7 days.
- generate_topic_recommendations_task : Periodic (weekly) – produce AI-powered topic
                                        recommendations and persist them to a JSON file.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from celery.utils.log import get_task_logger
from scheduler.tasks.celery_app import celery_app

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = get_task_logger(__name__)

# ---------------------------------------------------------------------------
# Path where weekly topic recommendations are stored
# ---------------------------------------------------------------------------
_DATA_DIR: str = os.getenv("DATA_DIR", "data")
_RECOMMENDATIONS_FILE: str = os.path.join(_DATA_DIR, "topic_recommendations.json")

# ---------------------------------------------------------------------------
# Optional / NEW module imports – degrade gracefully if not yet implemented
# ---------------------------------------------------------------------------
try:
    from modules.analytics.advanced_engine import AdvancedAnalyticsEngine

    _ADVANCED_ANALYTICS_AVAILABLE = True
    logger.debug("AdvancedAnalyticsEngine loaded successfully.")
except ImportError:
    AdvancedAnalyticsEngine = None  # type: ignore[assignment,misc]
    _ADVANCED_ANALYTICS_AVAILABLE = False
    logger.warning(
        "modules.analytics.advanced_engine not found – "
        "generate_topic_recommendations_task will use a basic fallback."
    )


# ===========================================================================
# Helpers
# ===========================================================================


def _ensure_data_dir() -> None:
    """Create the data directory if it does not exist."""
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
    except OSError as exc:
        logger.error("Failed to create data directory %s: %s", _DATA_DIR, exc)
        raise


def _utcnow() -> datetime:
    """Return the current UTC-aware datetime."""
    return datetime.now(tz=timezone.utc)


def _posts_from_last_n_days(
    db,  # DatabaseManager instance
    days: int = 7,
) -> List[Dict[str, Any]]:
    """
    Return all published posts whose ``created_at`` falls within the last
    *days* calendar days AND that have a ``linkedin_post_id`` set.

    We iterate over posts returned by ``list_posts(status='published')``
    and filter by ``created_at``.  SQLite stores timestamps as plain strings
    (``YYYY-MM-DD HH:MM:SS``), so we parse them here.
    """
    cutoff: datetime = _utcnow() - timedelta(days=days)
    published: List[Dict[str, Any]] = db.list_posts(status="published")

    eligible: List[Dict[str, Any]] = []
    for post in published:
        linkedin_post_id: Optional[str] = post.get("linkedin_post_id")
        if not linkedin_post_id:
            continue

        created_raw: Optional[str] = post.get("created_at")
        if not created_raw:
            # No timestamp – include it conservatively
            eligible.append(post)
            continue

        try:
            # SQLite default format: "YYYY-MM-DD HH:MM:SS"
            created_at = datetime.strptime(
                str(created_raw)[:19], "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=timezone.utc)
            if created_at >= cutoff:
                eligible.append(post)
        except ValueError:
            # Unparseable timestamp – include conservatively
            eligible.append(post)

    return eligible


# ===========================================================================
# Task 1 – update_post_analytics_task
# ===========================================================================


@celery_app.task(
    bind=True,
    name="scheduler.tasks.analytics_tasks.update_post_analytics_task",
    max_retries=3,
    default_retry_delay=120,
    serializer="json",
    acks_late=True,
    track_started=True,
)
def update_post_analytics_task(
    self,
    post_id: int,
    linkedin_post_id: str,
) -> Dict[str, Any]:
    """
    Fetch engagement analytics for a single LinkedIn post and persist them.

    Steps
    -----
    1. Call :meth:`LinkedInPublisher.get_post_analytics(linkedin_post_id)`.
    2. Persist the returned metrics via :meth:`DatabaseManager.update_analytics`.
    3. Return a structured result dict.

    On transient failure the task retries up to ``max_retries`` times with a
    120-second base delay (exponential back-off applied on retry).

    Parameters
    ----------
    post_id:
        Internal database ID of the post.
    linkedin_post_id:
        The LinkedIn-assigned post URN / ID string used to query the API.

    Returns
    -------
    dict with keys: success, post_id, linkedin_post_id, analytics, attempts, error
    """
    attempt_number: int = self.request.retries + 1

    logger.info(
        "task=update_post_analytics|post_id=%d|linkedin_post_id=%s"
        "|status=started|attempt=%d",
        post_id,
        linkedin_post_id,
        attempt_number,
    )

    # ------------------------------------------------------------------
    # Initialise dependencies
    # ------------------------------------------------------------------
    try:
        from database.models import DatabaseManager
        from services.linkedin_publisher import LinkedInPublisher

        db = DatabaseManager()
        publisher = LinkedInPublisher()
    except Exception as init_exc:  # noqa: BLE001
        logger.error(
            "task=update_post_analytics|post_id=%d|status=init_error|error=%s",
            post_id,
            init_exc,
        )
        return {
            "success": False,
            "post_id": post_id,
            "linkedin_post_id": linkedin_post_id,
            "analytics": None,
            "attempts": attempt_number,
            "error": f"Dependency initialisation failed: {init_exc}",
        }

    # ------------------------------------------------------------------
    # Fetch analytics from LinkedIn
    # ------------------------------------------------------------------
    analytics: Optional[Dict[str, Any]] = None
    try:
        analytics = publisher.get_post_analytics(linkedin_post_id)

        if analytics is None:
            raise ValueError(
                f"get_post_analytics returned None for linkedin_post_id={linkedin_post_id!r}"
            )

        logger.info(
            "task=update_post_analytics|post_id=%d|linkedin_post_id=%s"
            "|likes=%s|comments=%s|impressions=%s|status=fetched",
            post_id,
            linkedin_post_id,
            analytics.get("likes", "?"),
            analytics.get("comments", "?"),
            analytics.get("impressions", "?"),
        )

    except Exception as fetch_exc:  # noqa: BLE001
        countdown: int = 120 * (2**self.request.retries)  # 120, 240, 480 s

        logger.error(
            "task=update_post_analytics|post_id=%d|linkedin_post_id=%s"
            "|status=fetch_failed|error=%s|attempt=%d|retrying_in=%ds",
            post_id,
            linkedin_post_id,
            fetch_exc,
            attempt_number,
            countdown,
        )

        if self.request.retries >= self.max_retries:
            logger.error(
                "task=update_post_analytics|post_id=%d|status=permanently_failed"
                "|total_attempts=%d",
                post_id,
                attempt_number,
            )
            return {
                "success": False,
                "post_id": post_id,
                "linkedin_post_id": linkedin_post_id,
                "analytics": None,
                "attempts": attempt_number,
                "error": str(fetch_exc),
            }

        raise self.retry(exc=fetch_exc, countdown=countdown)

    # ------------------------------------------------------------------
    # Persist to database
    # ------------------------------------------------------------------
    try:
        db.update_analytics(
            post_id=post_id,
            likes=int(analytics.get("likes", 0)),
            comments=int(analytics.get("comments", 0)),
            impressions=int(analytics.get("impressions", 0)),
        )

        logger.info(
            "task=update_post_analytics|post_id=%d|linkedin_post_id=%s"
            "|status=completed|attempts=%d",
            post_id,
            linkedin_post_id,
            attempt_number,
        )

    except Exception as db_exc:  # noqa: BLE001
        # Database errors are usually non-transient; log and return without retry.
        logger.error(
            "task=update_post_analytics|post_id=%d|status=db_update_failed|error=%s",
            post_id,
            db_exc,
        )
        return {
            "success": False,
            "post_id": post_id,
            "linkedin_post_id": linkedin_post_id,
            "analytics": analytics,
            "attempts": attempt_number,
            "error": f"DB update failed: {db_exc}",
        }

    return {
        "success": True,
        "post_id": post_id,
        "linkedin_post_id": linkedin_post_id,
        "analytics": analytics,
        "attempts": attempt_number,
        "error": None,
    }


# ===========================================================================
# Task 2 – bulk_analytics_update_task  (periodic – daily)
# ===========================================================================


@celery_app.task(
    bind=True,
    name="scheduler.tasks.analytics_tasks.bulk_analytics_update_task",
    max_retries=1,
    default_retry_delay=300,
    serializer="json",
    acks_late=True,
    track_started=True,
)
def bulk_analytics_update_task(self) -> Dict[str, Any]:
    """
    Daily fan-out – schedule an analytics refresh for every published post
    from the last 7 days that has a ``linkedin_post_id``.

    Each eligible post gets its own :func:`update_post_analytics_task` queued
    via ``.delay()`` so the work is distributed across workers and retried
    independently.

    This task is registered in Celery Beat's schedule (once per day).

    Returns
    -------
    dict with keys: posts_updated_count, post_ids, window_days, status
    """
    window_days: int = int(os.getenv("ANALYTICS_WINDOW_DAYS", "7"))

    logger.info(
        "task=bulk_analytics_update|window_days=%d|status=started",
        window_days,
    )

    queued_ids: List[int] = []

    try:
        from database.models import DatabaseManager

        db = DatabaseManager()
        eligible: List[Dict[str, Any]] = _posts_from_last_n_days(db, days=window_days)

        if not eligible:
            logger.info(
                "task=bulk_analytics_update|window_days=%d|status=no_eligible_posts",
                window_days,
            )
            return {
                "posts_updated_count": 0,
                "post_ids": [],
                "window_days": window_days,
                "status": "completed_no_posts",
            }

        logger.info(
            "task=bulk_analytics_update|eligible_posts=%d|status=dispatching",
            len(eligible),
        )

        for post in eligible:
            pid: int = int(post["id"])
            li_id: str = str(post["linkedin_post_id"])

            update_post_analytics_task.delay(
                post_id=pid,
                linkedin_post_id=li_id,
            )
            queued_ids.append(pid)

            logger.debug(
                "task=bulk_analytics_update|post_id=%d|linkedin_post_id=%s"
                "|status=queued",
                pid,
                li_id,
            )

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "task=bulk_analytics_update|status=sweep_error|error=%s"
            "|posts_queued_before_error=%d",
            exc,
            len(queued_ids),
        )
        # Return partial result rather than propagating
        return {
            "posts_updated_count": len(queued_ids),
            "post_ids": queued_ids,
            "window_days": window_days,
            "status": "partial_error",
        }

    logger.info(
        "task=bulk_analytics_update|posts_queued=%d|window_days=%d|status=completed",
        len(queued_ids),
        window_days,
    )

    return {
        "posts_updated_count": len(queued_ids),
        "post_ids": queued_ids,
        "window_days": window_days,
        "status": "completed",
    }


# ===========================================================================
# Task 3 – generate_topic_recommendations_task  (periodic – weekly)
# ===========================================================================


@celery_app.task(
    bind=True,
    name="scheduler.tasks.analytics_tasks.generate_topic_recommendations_task",
    max_retries=2,
    default_retry_delay=600,
    serializer="json",
    acks_late=True,
    track_started=True,
)
def generate_topic_recommendations_task(self) -> Dict[str, Any]:
    """
    Weekly task – generate data-driven topic recommendations and persist them.

    Generation strategy
    -------------------
    1. **AdvancedAnalyticsEngine** (preferred) – produces AI-enriched
       recommendations based on historical post performance and trending signals.
    2. **TopicEngine fallback** – if the advanced engine is unavailable, derive
       recommendations from :meth:`TopicEngine.get_next_recommended_topics`.

    Persistence
    -----------
    Results are written to :data:`_RECOMMENDATIONS_FILE`
    (``data/topic_recommendations.json``).  The file is written atomically via
    a temporary name to prevent partial reads.

    Returns
    -------
    dict with keys: success, recommendations_count, output_file, generated_at, error
    """
    attempt_number: int = self.request.retries + 1

    logger.info(
        "task=generate_topic_recommendations|status=started|attempt=%d",
        attempt_number,
    )

    recommendations: List[Dict[str, Any]] = []
    engine_used: str = "none"

    # ------------------------------------------------------------------
    # Step 1 – Generate recommendations
    # ------------------------------------------------------------------

    # 1a. AdvancedAnalyticsEngine (primary)
    if _ADVANCED_ANALYTICS_AVAILABLE:
        try:
            analytics_engine = AdvancedAnalyticsEngine()
            raw = analytics_engine.generate_topic_recommendations()

            if raw:
                # Normalise: engine may return strings or dicts
                for idx, item in enumerate(raw):
                    if isinstance(item, str):
                        recommendations.append(
                            {
                                "rank": idx + 1,
                                "topic": item,
                                "score": None,
                                "reasoning": None,
                            }
                        )
                    elif isinstance(item, dict):
                        item.setdefault("rank", idx + 1)
                        recommendations.append(item)

                engine_used = "AdvancedAnalyticsEngine"

                logger.info(
                    "task=generate_topic_recommendations|engine=%s"
                    "|recommendations=%d|status=engine_success",
                    engine_used,
                    len(recommendations),
                )

        except Exception as eng_exc:  # noqa: BLE001
            logger.warning(
                "task=generate_topic_recommendations|engine=AdvancedAnalyticsEngine"
                "|status=engine_error|error=%s – falling back to TopicEngine",
                eng_exc,
            )
            recommendations = []

    # 1b. TopicEngine fallback
    if not recommendations:
        try:
            from database.models import DatabaseManager
            from services.topic_engine import TopicEngine

            db = DatabaseManager()
            topic_engine = TopicEngine(db)

            # Fetch the top recommended topics and enrich with basic metadata
            raw_topics: List[str] = topic_engine.get_next_recommended_topics(count=20)
            topic_weights: Dict[str, float] = topic_engine.topic_weights

            recommendations = [
                {
                    "rank": idx + 1,
                    "topic": topic,
                    "score": round(topic_weights.get(topic, 1.0), 4),
                    "reasoning": "Derived from historical engagement weighting via TopicEngine",
                }
                for idx, topic in enumerate(raw_topics)
            ]
            engine_used = "TopicEngine"

            logger.info(
                "task=generate_topic_recommendations|engine=%s"
                "|recommendations=%d|status=fallback_success",
                engine_used,
                len(recommendations),
            )

        except Exception as fallback_exc:  # noqa: BLE001
            countdown: int = 600 * (2**self.request.retries)

            logger.error(
                "task=generate_topic_recommendations|status=all_engines_failed"
                "|error=%s|attempt=%d|retrying_in=%ds",
                fallback_exc,
                attempt_number,
                countdown,
            )

            if self.request.retries >= self.max_retries:
                logger.error(
                    "task=generate_topic_recommendations|status=permanently_failed"
                    "|total_attempts=%d",
                    attempt_number,
                )
                return {
                    "success": False,
                    "recommendations_count": 0,
                    "output_file": _RECOMMENDATIONS_FILE,
                    "generated_at": _utcnow().isoformat(),
                    "error": str(fallback_exc),
                }

            raise self.retry(exc=fallback_exc, countdown=countdown)

    # Guard
    if not recommendations:
        logger.warning(
            "task=generate_topic_recommendations|engine=%s"
            "|status=empty_result – nothing to persist",
            engine_used,
        )
        return {
            "success": False,
            "recommendations_count": 0,
            "output_file": _RECOMMENDATIONS_FILE,
            "generated_at": _utcnow().isoformat(),
            "error": "No recommendations produced by any engine",
        }

    # ------------------------------------------------------------------
    # Step 2 – Persist to JSON file (atomic write)
    # ------------------------------------------------------------------
    generated_at: str = _utcnow().isoformat()

    output_payload: Dict[str, Any] = {
        "generated_at": generated_at,
        "engine_used": engine_used,
        "recommendations_count": len(recommendations),
        "recommendations": recommendations,
    }

    tmp_path: str = _RECOMMENDATIONS_FILE + ".tmp"

    try:
        _ensure_data_dir()

        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(output_payload, fh, ensure_ascii=False, indent=2)

        # Atomic rename – prevents partial reads by other processes
        os.replace(tmp_path, _RECOMMENDATIONS_FILE)

        logger.info(
            "task=generate_topic_recommendations|engine=%s"
            "|recommendations=%d|output_file=%s|status=completed",
            engine_used,
            len(recommendations),
            _RECOMMENDATIONS_FILE,
        )

    except OSError as io_exc:
        logger.error(
            "task=generate_topic_recommendations|status=write_failed"
            "|output_file=%s|error=%s",
            _RECOMMENDATIONS_FILE,
            io_exc,
        )
        # Clean up orphaned tmp file if it exists
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass

        return {
            "success": False,
            "recommendations_count": len(recommendations),
            "output_file": _RECOMMENDATIONS_FILE,
            "generated_at": generated_at,
            "error": f"File write failed: {io_exc}",
        }

    return {
        "success": True,
        "recommendations_count": len(recommendations),
        "output_file": _RECOMMENDATIONS_FILE,
        "generated_at": generated_at,
        "error": None,
    }

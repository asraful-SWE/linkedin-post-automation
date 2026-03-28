"""
Celery Application - Background task queue
Falls back gracefully if Redis is not available.
"""

import logging
import os

from celery import Celery
from celery.utils.log import get_task_logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
task_logger = get_task_logger(__name__)

# ---------------------------------------------------------------------------
# Broker / Backend configuration (read from environment)
# ---------------------------------------------------------------------------
BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

# ---------------------------------------------------------------------------
# Celery application instance
# ---------------------------------------------------------------------------
celery_app = Celery("linkedin_poster")

celery_app.conf.update(
    # Transport
    broker_url=BROKER_URL,
    result_backend=RESULT_BACKEND,
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Locale / time
    timezone="Asia/Dhaka",
    enable_utc=True,
    # Reliability
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Retry defaults (tasks may override per-task)
    task_max_retries=3,
    task_default_retry_delay=60,  # seconds
    # Result expiry – keep results for 1 hour
    result_expires=3600,
    # Prevent tasks from silently swallowing tracebacks
    task_eager_propagates=True,
)

# ---------------------------------------------------------------------------
# Auto-discover tasks in the scheduler.tasks package
# ---------------------------------------------------------------------------
celery_app.autodiscover_tasks(
    packages=[
        "scheduler.tasks",
    ],
    related_name="content_tasks",
)
celery_app.autodiscover_tasks(
    packages=[
        "scheduler.tasks",
    ],
    related_name="publishing_tasks",
)
celery_app.autodiscover_tasks(
    packages=[
        "scheduler.tasks",
    ],
    related_name="analytics_tasks",
)


# ---------------------------------------------------------------------------
# Beat schedule – periodic tasks
# (Only active when `celery beat` is running alongside `celery worker`)
# ---------------------------------------------------------------------------
celery_app.conf.beat_schedule = {
    # Retry failed posts every 30 minutes
    "retry-failed-posts-every-30-min": {
        "task": "scheduler.tasks.publishing_tasks.retry_failed_posts_task",
        "schedule": 1800.0,  # seconds
        "options": {"expires": 1800},
    },
    # Pull analytics for recent published posts once a day at midnight (BD time)
    "bulk-analytics-update-daily": {
        "task": "scheduler.tasks.analytics_tasks.bulk_analytics_update_task",
        "schedule": 86400.0,
        "options": {"expires": 86400},
    },
    # Generate topic recommendations every Sunday (weekly)
    "topic-recommendations-weekly": {
        "task": "scheduler.tasks.analytics_tasks.generate_topic_recommendations_task",
        "schedule": 604800.0,  # 7 days
        "options": {"expires": 604800},
    },
}


# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------


def is_celery_available() -> bool:
    """
    Probe the Redis broker with a low-timeout PING.

    Returns
    -------
    bool
        ``True``  – Redis is reachable and Celery tasks can be dispatched.
        ``False`` – Redis is unreachable; caller should fall back to
                    synchronous execution via :class:`task_runner.SyncTaskRunner`.
    """
    try:
        import redis  # type: ignore[import]

        client = redis.from_url(
            BROKER_URL,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        response = client.ping()
        if response:
            logger.debug("Redis broker is reachable at %s", BROKER_URL)
            return True
        logger.warning(
            "Redis PING returned falsy response – treating broker as unavailable."
        )
        return False

    except ImportError:
        logger.warning(
            "redis package is not installed – Celery tasks will run synchronously. "
            "Install it with: pip install redis"
        )
        return False

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Redis not available - Celery tasks will run synchronously. Reason: %s",
            exc,
        )
        return False

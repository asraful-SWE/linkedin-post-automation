"""
Posting Scheduler - Manages automated LinkedIn post scheduling with natural timing.

Upgraded to use:
- IntelligentTopicEngine (semantic clustering, series, trending boost)
- IntelligentContentEngine (goal-driven generation, quality scoring)
- ImageFetcher + ImageSelector (Unsplash/Pexels auto images)
- Celery/SyncTaskRunner for background tasks
"""

import asyncio
import logging
import os
import random
from datetime import datetime, time, timedelta
from typing import Any, Dict, Optional, Tuple

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from database.models import DatabaseManager
from services.approval_service import ApprovalService
from services.email_service import EmailService
from services.post_generator import PostGenerator
from services.topic_engine import TopicEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional new-module imports (degrade gracefully if not available)
# ---------------------------------------------------------------------------

# Each block sets an _AVAILABLE flag AND imports the real class.
# The except branch sets the name to None so that the rest of the module
# can reference it without NameError; all call-sites are guarded by the
# corresponding _AVAILABLE flag so None is never actually called at runtime.

try:
    from modules.topic.engine import (
        IntelligentTopicEngine as _IntelligentTopicEngine,  # type: ignore[import]
    )

    _INTELLIGENT_TOPIC_AVAILABLE = True
except ImportError:
    _IntelligentTopicEngine = None  # type: ignore[assignment]
    _INTELLIGENT_TOPIC_AVAILABLE = False
    logger.warning("IntelligentTopicEngine not available – using TopicEngine fallback")

# Convenience alias used throughout the class
IntelligentTopicEngine = _IntelligentTopicEngine  # type: ignore[assignment]

try:
    from ai.openai_provider import (
        OpenAIProvider as _OpenAIProvider,  # type: ignore[import]
    )
    from modules.content.engine import (  # type: ignore[import]
        IntelligentContentEngine as _IntelligentContentEngine,
    )
    from modules.content.engine import PostGoal as _PostGoal
    from modules.content.scorer import (
        ContentScorer as _ContentScorer,  # type: ignore[import]
    )

    _INTELLIGENT_CONTENT_AVAILABLE = True
except ImportError:
    _OpenAIProvider = None  # type: ignore[assignment]
    _IntelligentContentEngine = None  # type: ignore[assignment]
    _PostGoal = None  # type: ignore[assignment]
    _ContentScorer = None  # type: ignore[assignment]
    _INTELLIGENT_CONTENT_AVAILABLE = False
    logger.warning(
        "IntelligentContentEngine not available – using PostGenerator fallback"
    )

OpenAIProvider = _OpenAIProvider  # type: ignore[assignment]
IntelligentContentEngine = _IntelligentContentEngine  # type: ignore[assignment]
PostGoal = _PostGoal  # type: ignore[assignment]
ContentScorer = _ContentScorer  # type: ignore[assignment]

try:
    from modules.image.fetcher import (
        ImageFetcher as _ImageFetcher,  # type: ignore[import]
    )
    from modules.image.selector import (
        ImageSelector as _ImageSelector,  # type: ignore[import]
    )

    _IMAGE_MODULES_AVAILABLE = True
except ImportError:
    _ImageFetcher = None  # type: ignore[assignment]
    _ImageSelector = None  # type: ignore[assignment]
    _IMAGE_MODULES_AVAILABLE = False
    logger.warning("Image modules not available – image auto-selection disabled")

ImageFetcher = _ImageFetcher  # type: ignore[assignment]
ImageSelector = _ImageSelector  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Goal rotation (cycle through goals for variety)
# ---------------------------------------------------------------------------

_GOAL_POOL = ["educational", "viral", "authority", "story", "engagement"]
_goal_index = 0


def _next_goal() -> str:
    global _goal_index
    goal = _GOAL_POOL[_goal_index % len(_GOAL_POOL)]
    _goal_index += 1
    return goal


class PostingScheduler:
    """
    Intelligent posting scheduler with natural human-like behavior.
    Uses intelligent topic/content engines and optional image auto-selection.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

        # --- Core services (always available) ---
        self.topic_engine = TopicEngine(db_manager)
        self.post_generator = PostGenerator()
        self.approval_service = ApprovalService(db_manager)
        self.email_service = EmailService()

        # --- Intelligent topic engine ---
        if _INTELLIGENT_TOPIC_AVAILABLE and IntelligentTopicEngine is not None:
            try:
                self.intelligent_topic_engine = IntelligentTopicEngine(  # type: ignore[misc]
                    db_manager=db_manager,
                    existing_topic_engine=self.topic_engine,
                )
                logger.info("IntelligentTopicEngine initialised")
            except Exception as exc:
                logger.warning(
                    "IntelligentTopicEngine init failed: %s – using fallback", exc
                )
                self.intelligent_topic_engine = None
        else:
            self.intelligent_topic_engine = None

        # --- Intelligent content engine ---
        self.intelligent_content_engine = None
        if (
            _INTELLIGENT_CONTENT_AVAILABLE
            and IntelligentContentEngine is not None
            and OpenAIProvider is not None
            and ContentScorer is not None
        ):
            try:
                _api_key = os.getenv("OPENAI_API_KEY", "")
                if _api_key:
                    _provider = OpenAIProvider(api_key=_api_key)  # type: ignore[misc]
                    _scorer = ContentScorer(  # type: ignore[misc]
                        threshold=float(os.getenv("CONTENT_SCORE_THRESHOLD", "6.0"))
                    )
                    self.intelligent_content_engine = IntelligentContentEngine(  # type: ignore[misc]
                        openai_provider=_provider,
                        scorer=_scorer,
                        max_regeneration_attempts=int(
                            os.getenv("MAX_REGENERATION_ATTEMPTS", "3")
                        ),
                        score_threshold=float(
                            os.getenv("CONTENT_SCORE_THRESHOLD", "6.0")
                        ),
                    )
                    logger.info("IntelligentContentEngine initialised")
                else:
                    logger.warning(
                        "OPENAI_API_KEY not set – IntelligentContentEngine disabled"
                    )
            except Exception as exc:
                logger.warning(
                    "IntelligentContentEngine init failed: %s – using fallback", exc
                )
                self.intelligent_content_engine = None

        # --- Image modules ---
        self.image_selector = None
        self.enable_images = os.getenv("ENABLE_IMAGES", "false").lower() == "true"
        if (
            self.enable_images
            and _IMAGE_MODULES_AVAILABLE
            and ImageFetcher is not None
            and ImageSelector is not None
        ):
            try:
                _fetcher = ImageFetcher()  # type: ignore[misc]
                self.image_selector = ImageSelector(fetcher=_fetcher)  # type: ignore[misc]
                logger.info("ImageSelector initialised for auto image selection")
            except Exception as exc:
                logger.warning(
                    "ImageSelector init failed: %s – image auto-selection disabled", exc
                )

        # --- APScheduler ---
        self.scheduler = AsyncIOScheduler()

        # --- Config ---
        self.timezone = pytz.timezone(os.getenv("TIMEZONE", "Asia/Dhaka"))
        self.max_posts_per_day = int(os.getenv("MAX_POSTS_PER_DAY", "2"))
        self.min_hours_between_posts = float(os.getenv("MIN_HOURS_BETWEEN_POSTS", "4"))
        self.test_mode = os.getenv("TEST_MODE", "false").lower() == "true"

        # Posting windows: Morning 9:30-10:30, Evening 19:30-20:30
        self.posting_windows = [
            {"start": time(9, 30), "end": time(10, 30), "weight": 1},
            {"start": time(19, 30), "end": time(20, 30), "weight": 1},
        ]

        self.skip_probability = float(os.getenv("SKIP_POST_PROBABILITY", "0.15"))
        self.double_post_probability = float(
            os.getenv("DOUBLE_POST_PROBABILITY", "0.05")
        )

    # ------------------------------------------------------------------
    # Scheduler lifecycle
    # ------------------------------------------------------------------

    def start_scheduler(self):
        """Start the automated posting scheduler."""
        try:
            self.scheduler.start()
            self._schedule_daily_posts()
            self._schedule_maintenance_tasks()
            logger.info(
                "event=scheduler_started|jobs=%d", len(self.scheduler.get_jobs())
            )
        except Exception as exc:
            logger.error("event=scheduler_start_failed|error=%s", exc)
            raise

    def stop_scheduler(self):
        """Stop the scheduler gracefully."""
        try:
            self.scheduler.shutdown()
            logger.info("event=scheduler_stopped")
        except Exception as exc:
            logger.error("event=scheduler_stop_failed|error=%s", exc)

    # ------------------------------------------------------------------
    # Job scheduling
    # ------------------------------------------------------------------

    def _schedule_daily_posts(self):
        try:
            if self.test_mode:
                interval_seconds = int(self.min_hours_between_posts * 3600)
                self.scheduler.add_job(
                    self._execute_posting_job,
                    IntervalTrigger(seconds=interval_seconds),
                    id="test_interval_posting",
                    replace_existing=True,
                )
                logger.info("TEST MODE: posting every %ds", interval_seconds)
                return

            for window in self.posting_windows:
                for _ in range(window["weight"]):
                    start_m = window["start"].hour * 60 + window["start"].minute
                    end_m = window["end"].hour * 60 + window["end"].minute
                    rand_m = random.randint(start_m, end_m)
                    hour = rand_m // 60
                    minute = max(0, min(59, rand_m % 60 + random.randint(-5, 5)))
                    days = self._get_natural_posting_days()
                    trigger = CronTrigger(
                        day_of_week=days,
                        hour=hour,
                        minute=minute,
                        timezone=self.timezone,
                        jitter=300,
                    )
                    self.scheduler.add_job(
                        self._execute_posting_job,
                        trigger=trigger,
                        id=f"post_job_{hour}_{minute}_{random.randint(1000, 9999)}",
                        replace_existing=True,
                    )
            logger.info("event=daily_posts_scheduled")
        except Exception as exc:
            logger.error("event=schedule_daily_posts_failed|error=%s", exc)

    def _get_natural_posting_days(self) -> str:
        weekdays = ["mon", "tue", "wed", "thu", "fri"]
        weekends = ["sat", "sun"]
        selected = [d for d in weekdays if random.random() < 0.8]
        selected += [d for d in weekends if random.random() < 0.4]
        if len(selected) < 2:
            selected = random.sample(["mon", "tue", "wed", "thu", "fri"], 3)
        return ",".join(selected)

    def _schedule_maintenance_tasks(self):
        try:
            self.scheduler.add_job(
                self._update_analytics_job,
                CronTrigger(hour=2, minute=30, timezone=self.timezone),
                id="analytics_update",
                replace_existing=True,
            )
            self.scheduler.add_job(
                self._refresh_topic_performance,
                CronTrigger(day_of_week="sun", hour=3, timezone=self.timezone),
                id="topic_performance_refresh",
                replace_existing=True,
            )
            self.scheduler.add_job(
                self._monthly_cleanup,
                CronTrigger(day=1, hour=4, timezone=self.timezone),
                id="monthly_cleanup",
                replace_existing=True,
            )
            logger.info("event=maintenance_tasks_scheduled")
        except Exception as exc:
            logger.error("event=schedule_maintenance_failed|error=%s", exc)

    # ------------------------------------------------------------------
    # Core job execution
    # ------------------------------------------------------------------

    async def _execute_posting_job(self):
        """Execute a single posting job with natural behavior checks."""
        try:
            if random.random() < self.skip_probability:
                logger.info("event=post_skipped|reason=natural_skip")
                return

            posts_today = self.db.get_posts_count_today()
            if posts_today >= self.max_posts_per_day:
                logger.info(
                    "event=post_skipped|reason=daily_limit|count=%d", posts_today
                )
                return

            last_post_time = self.db.get_last_post_time()
            if last_post_time:
                elapsed = (datetime.now() - last_post_time).total_seconds()
                if elapsed < (self.min_hours_between_posts * 3600):
                    logger.info(
                        "event=post_skipped|reason=too_soon|elapsed_hours=%.1f",
                        elapsed / 3600,
                    )
                    return

            await self._generate_and_queue_post_for_approval()

            if random.random() < self.double_post_probability and posts_today == 0:
                delay = random.randint(3 * 3600, 6 * 3600)
                self.scheduler.add_job(
                    self._generate_and_queue_post_for_approval,
                    "date",
                    run_date=datetime.now() + timedelta(seconds=delay),
                    id=f"followup_{random.randint(1000, 9999)}",
                )
                logger.info("event=followup_scheduled|in_hours=%.1f", delay / 3600)

        except Exception as exc:
            logger.error("event=posting_job_failed|error=%s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Post generation & approval pipeline
    # ------------------------------------------------------------------

    def _select_topic(self) -> str:
        """Select the best topic using IntelligentTopicEngine or fallback."""
        if self.intelligent_topic_engine is not None:
            try:
                result = self.intelligent_topic_engine.select_topic_intelligent()
                logger.info("event=topic_selected|engine=intelligent|topic=%s", result)
                return result
            except Exception as exc:
                logger.warning(
                    "event=intelligent_topic_failed|error=%s – using fallback", exc
                )
        topic = self.topic_engine.select_topic()
        logger.info("event=topic_selected|engine=basic|topic=%s", topic)
        return topic

    def _generate_content(self, topic: str, goal: str) -> Tuple[str, float]:
        """
        Generate post content using IntelligentContentEngine or PostGenerator fallback.
        Returns (content, score).
        """
        if self.intelligent_content_engine is not None:
            try:
                from modules.content.engine import PostGoal as PG

                goal_enum = (
                    PG(goal) if goal in [g.value for g in PG] else PG.EDUCATIONAL
                )
                result = self.intelligent_content_engine.generate_post_with_goal(
                    topic=topic, goal=goal_enum
                )
                content = result.get("content", "")
                score_obj = result.get("score")
                score = float(score_obj.overall_score) if score_obj else 0.0
                if content:
                    logger.info(
                        "event=content_generated|engine=intelligent|topic=%s|goal=%s|score=%.2f",
                        topic,
                        goal,
                        score,
                    )
                    return content, score
            except Exception as exc:
                logger.warning(
                    "event=intelligent_content_failed|error=%s – using fallback", exc
                )

        # Fallback
        content = self.post_generator.generate_post(topic)
        logger.info("event=content_generated|engine=basic|topic=%s", topic)
        return content, 0.0

    def _auto_select_image(self, topic: str) -> Optional[str]:
        """Auto-select a relevant image from Unsplash/Pexels if configured."""
        if not self.enable_images or self.image_selector is None:
            return None
        try:
            image = self.image_selector.get_image_for_topic(topic)  # type: ignore[union-attr]
            url = image.get("url") if image is not None else None
            if url and image is not None:
                logger.info(
                    "event=image_selected|topic=%s|source=%s|url=%s",
                    topic,
                    image.get("source"),
                    url,
                )
            return url
        except Exception as exc:
            logger.warning("event=image_selection_failed|topic=%s|error=%s", topic, exc)
            return None

    async def _generate_and_queue_post_for_approval(self):
        """Full pipeline: topic → content → image → DB → email."""
        try:
            logger.info("event=pipeline_start")

            # 1. Select topic
            topic = self._select_topic()

            # 2. Select goal (rotating)
            goal = _next_goal()

            # 3. Generate content
            post_content, content_score = self._generate_content(topic, goal)
            if not post_content:
                logger.error(
                    "event=pipeline_failed|reason=empty_content|topic=%s", topic
                )
                return

            # 4. Auto-select image (optional)
            image_url = self._auto_select_image(topic)

            # 5. Save pending post + approval token
            pending_result = self.approval_service.create_pending_post(
                topic=topic, content=post_content
            )
            post_id = pending_result["post_id"]
            token = pending_result["token"]

            # 6. Persist extra metadata (post_goal, content_score, image_url)
            try:
                self.db.set_post_meta(
                    post_id=post_id,
                    post_goal=goal,
                    content_score=content_score if content_score else None,
                )
                if image_url:
                    self.db.set_post_image_url(post_id, image_url)
            except Exception as meta_exc:
                logger.warning(
                    "event=meta_save_failed|post_id=%d|error=%s", post_id, meta_exc
                )

            # 7. Send approval email
            email_sent = self.email_service.send_post_approval_email(
                post_id=post_id,
                topic=topic,
                content=post_content,
                token=token,
            )
            logger.info(
                "event=pipeline_complete|post_id=%d|topic=%s|goal=%s|score=%.2f"
                "|has_image=%s|email_sent=%s",
                post_id,
                topic,
                goal,
                content_score,
                bool(image_url),
                email_sent,
            )

            await asyncio.sleep(random.randint(5, 20))

        except Exception as exc:
            logger.error("event=pipeline_failed|error=%s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Maintenance jobs
    # ------------------------------------------------------------------

    async def _update_analytics_job(self):
        try:
            logger.info("event=analytics_update_start")
            # Placeholder - real implementation fetches LinkedIn analytics
            logger.info("event=analytics_update_complete")
        except Exception as exc:
            logger.error("event=analytics_update_failed|error=%s", exc)

    async def _refresh_topic_performance(self):
        try:
            logger.info("event=topic_refresh_start")
            self.topic_engine.update_topic_performance()
            logger.info("event=topic_refresh_complete")
        except Exception as exc:
            logger.error("event=topic_refresh_failed|error=%s", exc)

    async def _monthly_cleanup(self):
        try:
            logger.info("event=monthly_cleanup_start")
            # self.db.cleanup_old_data(days=90)
            logger.info("event=monthly_cleanup_complete")
        except Exception as exc:
            logger.error("event=monthly_cleanup_failed|error=%s", exc)

    # ------------------------------------------------------------------
    # Manual post (API-triggered)
    # ------------------------------------------------------------------

    def manual_post(
        self, topic: Optional[str] = None, goal: Optional[str] = None
    ) -> Dict[str, Any]:
        """Manually generate post and trigger approval flow (API-triggered)."""
        try:
            selected_topic = topic or self._select_topic()
            selected_goal = goal or _next_goal()

            post_content, content_score = self._generate_content(
                selected_topic, selected_goal
            )
            if not post_content:
                return {"success": False, "error": "Content generation returned empty"}

            image_url = self._auto_select_image(selected_topic)

            pending_result = self.approval_service.create_pending_post(
                topic=selected_topic, content=post_content
            )
            post_id = pending_result["post_id"]
            token = pending_result["token"]

            try:
                self.db.set_post_meta(
                    post_id=post_id,
                    post_goal=selected_goal,
                    content_score=content_score if content_score else None,
                )
                if image_url:
                    self.db.set_post_image_url(post_id, image_url)
            except Exception as meta_exc:
                logger.warning(
                    "event=manual_meta_failed|post_id=%d|error=%s", post_id, meta_exc
                )

            email_sent = self.email_service.send_post_approval_email(
                post_id=post_id,
                topic=selected_topic,
                content=post_content,
                token=token,
            )

            return {
                "success": True,
                "post_id": post_id,
                "topic": selected_topic,
                "goal": selected_goal,
                "content_score": content_score,
                "has_image": bool(image_url),
                "image_url": image_url,
                "status": "pending",
                "email_sent": email_sent,
                "message": "Post generated and sent for approval",
            }

        except Exception as exc:
            logger.error("event=manual_post_failed|error=%s", exc, exc_info=True)
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Return current scheduler status and configuration."""
        try:
            jobs = self.scheduler.get_jobs()
            return {
                "running": self.scheduler.running,
                "total_jobs": len(jobs),
                "posting_jobs": len([j for j in jobs if j.id.startswith("post_job_")]),
                "next_posts": [
                    {
                        "id": j.id,
                        "next_run": j.next_run_time.isoformat()
                        if j.next_run_time
                        else None,
                    }
                    for j in jobs
                    if j.id.startswith("post_job_")
                ][:5],
                "posts_today": self.db.get_posts_count_today(),
                "max_posts_per_day": self.max_posts_per_day,
                "intelligent_topic_engine": self.intelligent_topic_engine is not None,
                "intelligent_content_engine": self.intelligent_content_engine
                is not None,
                "image_auto_selection": self.image_selector is not None,
            }
        except Exception as exc:
            logger.error("event=scheduler_status_failed|error=%s", exc)
            return {"error": "Unable to get status"}

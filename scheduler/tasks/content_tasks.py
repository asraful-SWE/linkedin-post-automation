"""
Content Generation Tasks - Background post generation with retry and fallback.

Tasks
-----
- generate_post_task                    : Generate a single LinkedIn post with exponential back-off retry.
- generate_and_queue_for_approval_task  : Full pipeline – topic → content → DB → approval email.
- batch_generate_for_ab_test_task       : Generate N variants for an A/B test.
"""

import logging
from typing import Any, Dict, List, Optional

from celery.utils.log import get_task_logger
from scheduler.tasks.celery_app import celery_app

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = get_task_logger(__name__)

# ---------------------------------------------------------------------------
# Optional / NEW module imports – degrade gracefully if not yet implemented
# ---------------------------------------------------------------------------
try:
    from modules.content.engine import IntelligentContentEngine

    _INTELLIGENT_ENGINE_AVAILABLE = True
    logger.debug("IntelligentContentEngine loaded successfully.")
except ImportError:
    IntelligentContentEngine = None  # type: ignore[assignment,misc]
    _INTELLIGENT_ENGINE_AVAILABLE = False
    logger.warning(
        "modules.content.engine not found – generate_post_task will use PostGenerator fallback."
    )

try:
    from modules.analytics.ab_testing import ABTestingManager

    _AB_TESTING_AVAILABLE = True
    logger.debug("ABTestingManager loaded successfully.")
except ImportError:
    ABTestingManager = None  # type: ignore[assignment,misc]
    _AB_TESTING_AVAILABLE = False
    logger.warning(
        "modules.analytics.ab_testing not found – batch A/B test records will not be persisted."
    )


# ===========================================================================
# Task 1 – generate_post_task
# ===========================================================================


@celery_app.task(
    bind=True,
    name="scheduler.tasks.content_tasks.generate_post_task",
    max_retries=3,
    default_retry_delay=120,
    serializer="json",
    acks_late=True,
    track_started=True,
)
def generate_post_task(
    self,
    topic: str,
    goal: str = "educational",
    use_intelligent_engine: bool = True,
) -> Dict[str, Any]:
    """
    Generate a LinkedIn post for *topic*.

    Generation Strategy
    -------------------
    1. **IntelligentContentEngine** (preferred) – richer, goal-aware output with
       a quality score.  Only attempted when ``use_intelligent_engine=True`` and
       the module can be imported.
    2. **PostGenerator** fallback – the existing service that produces
       human-like Bengali/Banglish posts via OpenAI.

    On any unhandled exception the task retries up to ``max_retries`` times
    using exponential back-off (120 s → 240 s → 480 s).

    Returns
    -------
    dict with keys: success, content, topic, goal, score, engine_used, attempts
    """
    attempt_number: int = self.request.retries + 1

    logger.info(
        "task=generate_post|topic=%s|goal=%s|use_intelligent=%s|status=started|attempt=%d",
        topic,
        goal,
        use_intelligent_engine,
        attempt_number,
    )

    content: Optional[str] = None
    score: float = 0.0
    engine_used: str = "none"

    # ------------------------------------------------------------------
    # Step 1 – IntelligentContentEngine (primary)
    # ------------------------------------------------------------------
    if (
        use_intelligent_engine
        and _INTELLIGENT_ENGINE_AVAILABLE
        and IntelligentContentEngine is not None
    ):
        try:
            import os as _os

            from ai.openai_provider import OpenAIProvider as _OAIProvider
            from modules.content.engine import PostGoal as _PG
            from modules.content.scorer import ContentScorer as _Scorer

            _provider = _OAIProvider(api_key=_os.getenv("OPENAI_API_KEY", ""))
            _scorer = _Scorer(
                threshold=float(_os.getenv("CONTENT_SCORE_THRESHOLD", "6.0"))
            )
            engine = IntelligentContentEngine(  # type: ignore[misc]
                openai_provider=_provider,
                scorer=_scorer,
                max_regeneration_attempts=int(
                    _os.getenv("MAX_REGENERATION_ATTEMPTS", "3")
                ),
                score_threshold=float(_os.getenv("CONTENT_SCORE_THRESHOLD", "6.0")),
            )
            # Convert goal string → PostGoal enum
            _valid_goals = {g.value: g for g in _PG}
            _goal_enum = _valid_goals.get(goal, _PG.EDUCATIONAL)
            engine_result: Dict[str, Any] = engine.generate_post_with_goal(
                topic=topic, goal=_goal_enum
            )

            if engine_result and engine_result.get("content"):
                content = str(engine_result["content"]).strip()
                _score_obj = engine_result.get("score")
                score = (
                    float(_score_obj.overall_score)
                    if _score_obj and hasattr(_score_obj, "overall_score")
                    else float(engine_result.get("score", 0.0) or 0.0)
                )
                engine_used = "IntelligentContentEngine"

                logger.info(
                    "task=generate_post|topic=%s|goal=%s|engine=%s|score=%.2f"
                    "|status=engine_success",
                    topic,
                    goal,
                    engine_used,
                    score,
                )
            else:
                logger.warning(
                    "task=generate_post|topic=%s|goal=%s|engine=IntelligentContentEngine"
                    "|status=engine_returned_empty – falling back to PostGenerator",
                    topic,
                    goal,
                )

        except Exception as engine_exc:  # noqa: BLE001
            logger.warning(
                "task=generate_post|topic=%s|goal=%s|engine=IntelligentContentEngine"
                "|status=engine_error|error=%s – falling back to PostGenerator",
                topic,
                goal,
                engine_exc,
            )
            content = None  # ensure we fall through

    # ------------------------------------------------------------------
    # Step 2 – PostGenerator fallback
    # ------------------------------------------------------------------
    if not content:
        try:
            # Local import keeps the task importable even if the service has
            # a top-level import error at worker boot time.
            from services.post_generator import PostGenerator

            generator = PostGenerator()
            raw: str = generator.generate_post(topic)

            if raw and raw.strip():
                content = raw.strip()
                engine_used = "PostGenerator"

                logger.info(
                    "task=generate_post|topic=%s|goal=%s|engine=%s|status=fallback_success",
                    topic,
                    goal,
                    engine_used,
                )
            else:
                raise ValueError(
                    f"PostGenerator returned empty content for topic='{topic}'"
                )

        except Exception as fallback_exc:  # noqa: BLE001
            # Both strategies failed – schedule a retry with exponential back-off.
            countdown: int = 120 * (2**self.request.retries)  # 120, 240, 480 s

            logger.error(
                "task=generate_post|topic=%s|goal=%s|status=failed|error=%s"
                "|attempt=%d|next_retry_in=%ds",
                topic,
                goal,
                fallback_exc,
                attempt_number,
                countdown,
            )

            raise self.retry(exc=fallback_exc, countdown=countdown)

    # ------------------------------------------------------------------
    # Return payload
    # ------------------------------------------------------------------
    payload: Dict[str, Any] = {
        "success": True,
        "content": content,
        "topic": topic,
        "goal": goal,
        "score": score,
        "engine_used": engine_used,
        "attempts": attempt_number,
    }

    logger.info(
        "task=generate_post|topic=%s|goal=%s|engine=%s|score=%.2f"
        "|attempts=%d|status=completed",
        topic,
        goal,
        engine_used,
        score,
        attempt_number,
    )

    return payload


# ===========================================================================
# Task 2 – generate_and_queue_for_approval_task
# ===========================================================================


@celery_app.task(
    bind=True,
    name="scheduler.tasks.content_tasks.generate_and_queue_for_approval_task",
    max_retries=2,
    default_retry_delay=60,
    serializer="json",
    acks_late=True,
    track_started=True,
)
def generate_and_queue_for_approval_task(
    self,
    topic: Optional[str] = None,
    goal: str = "educational",
) -> Dict[str, Any]:
    """
    Full automation pipeline for a single LinkedIn post.

    Pipeline
    --------
    1. **Select topic** – if not supplied, TopicEngine picks one
       (falls back to a random entry from ALL_TOPICS on error).
    2. **Generate content** – calls :func:`generate_post_task` synchronously
       so its result is available for the next steps.
    3. **Persist** – saves the post as *pending* and creates an approval token
       via :class:`ApprovalService`.
    4. **Email** – dispatches an approval email via :class:`EmailService`
       (non-fatal: pipeline succeeds even if the email cannot be sent).

    Returns
    -------
    dict with keys: success, post_id, topic, email_sent
    """
    logger.info(
        "task=generate_and_queue_for_approval|topic=%s|goal=%s|status=started",
        topic or "<auto-select>",
        goal,
    )

    # ------------------------------------------------------------------
    # Step 1 – Topic selection
    # ------------------------------------------------------------------
    if not topic:
        try:
            from database.models import DatabaseManager
            from services.topic_engine import TopicEngine

            _db_topics = DatabaseManager()
            topic_engine = TopicEngine(_db_topics)
            topic = topic_engine.select_topic()

            logger.info(
                "task=generate_and_queue_for_approval|status=topic_selected|topic=%s",
                topic,
            )

        except Exception as topic_exc:  # noqa: BLE001
            import random

            try:
                from services.topics import ALL_TOPICS

                topic = random.choice(ALL_TOPICS)
            except Exception:  # noqa: BLE001
                topic = "Software Engineering Best Practices"

            logger.warning(
                "task=generate_and_queue_for_approval|status=topic_fallback"
                "|topic=%s|reason=%s",
                topic,
                topic_exc,
            )

    # ------------------------------------------------------------------
    # Step 2 – Generate content (run generate_post_task synchronously)
    # ------------------------------------------------------------------
    try:
        # .apply() executes the task in the current process and returns an
        # EagerResult whose .result attribute holds the return value.
        eager = generate_post_task.apply(args=[topic, goal])  # type: ignore[attr-defined]
        content_data: Dict[str, Any] = eager.result

        if not content_data or not content_data.get("success"):
            err = (content_data or {}).get("error", "Unknown generation error")
            logger.error(
                "task=generate_and_queue_for_approval|topic=%s|status=generation_failed"
                "|error=%s",
                topic,
                err,
            )
            return {
                "success": False,
                "topic": topic,
                "post_id": None,
                "email_sent": False,
                "error": err,
            }

        content: str = content_data["content"]

        logger.info(
            "task=generate_and_queue_for_approval|topic=%s|status=content_ready"
            "|engine=%s|score=%.2f",
            topic,
            content_data.get("engine_used", "unknown"),
            content_data.get("score", 0.0),
        )

    except Exception as gen_exc:  # noqa: BLE001
        countdown = 60 * (2**self.request.retries)
        logger.error(
            "task=generate_and_queue_for_approval|topic=%s|status=generation_exception"
            "|error=%s|retrying_in=%ds",
            topic,
            gen_exc,
            countdown,
        )
        raise self.retry(exc=gen_exc, countdown=countdown)

    # ------------------------------------------------------------------
    # Step 3 – Save pending post + create approval token
    # ------------------------------------------------------------------
    post_id: Optional[int] = None
    token: Optional[str] = None

    try:
        from database.models import DatabaseManager
        from services.approval_service import ApprovalService

        db = DatabaseManager()
        approval_service = ApprovalService(db)
        pending: Dict[str, Any] = approval_service.create_pending_post(
            topic=topic, content=content
        )
        post_id = int(pending["post_id"])
        token = str(pending["token"])

        logger.info(
            "task=generate_and_queue_for_approval|topic=%s|post_id=%d|status=saved_pending",
            topic,
            post_id,
        )

    except Exception as db_exc:  # noqa: BLE001
        logger.error(
            "task=generate_and_queue_for_approval|topic=%s|status=db_save_failed"
            "|error=%s",
            topic,
            db_exc,
        )
        return {
            "success": False,
            "topic": topic,
            "post_id": None,
            "email_sent": False,
            "error": f"DB save failed: {db_exc}",
        }

    # ------------------------------------------------------------------
    # Step 4 – Send approval email (non-fatal)
    # ------------------------------------------------------------------
    email_sent: bool = False
    try:
        from services.email_service import EmailService

        email_service = EmailService()
        email_sent = email_service.send_post_approval_email(
            post_id=post_id,
            topic=topic,
            content=content,
            token=token,  # type: ignore[arg-type]
        )

        if email_sent:
            logger.info(
                "task=generate_and_queue_for_approval|post_id=%d|status=email_sent",
                post_id,
            )
        else:
            logger.warning(
                "task=generate_and_queue_for_approval|post_id=%d"
                "|status=email_not_sent|reason=service_returned_false",
                post_id,
            )

    except Exception as email_exc:  # noqa: BLE001
        # Email failure must never abort the pipeline – the post is already saved.
        logger.error(
            "task=generate_and_queue_for_approval|post_id=%d"
            "|status=email_exception|error=%s",
            post_id,
            email_exc,
        )

    logger.info(
        "task=generate_and_queue_for_approval|topic=%s|post_id=%d"
        "|email_sent=%s|status=completed",
        topic,
        post_id,
        email_sent,
    )

    return {
        "success": True,
        "post_id": post_id,
        "topic": topic,
        "email_sent": email_sent,
    }


# ===========================================================================
# Task 3 – batch_generate_for_ab_test_task
# ===========================================================================


@celery_app.task(
    bind=True,
    name="scheduler.tasks.content_tasks.batch_generate_for_ab_test_task",
    max_retries=2,
    default_retry_delay=90,
    serializer="json",
    acks_late=True,
    track_started=True,
)
def batch_generate_for_ab_test_task(
    self,
    topic: str,
    goal: str = "educational",
    count: int = 2,
) -> Dict[str, Any]:
    """
    Generate *count* content variants for an A/B test and persist the test record.

    Generation strategy
    -------------------
    1. Use :meth:`IntelligentContentEngine.batch_generate_for_ab_test` when
       available.
    2. If the engine has no batch method, call
       :meth:`IntelligentContentEngine.generate_post_with_goal` *count* times.
    3. Fall back to :func:`generate_post_task` for any variant the engine
       could not produce.

    Persistence
    -----------
    - If :class:`ABTestingManager` is available, create a formal AB test record.
    - Otherwise log a warning and return variant data directly.

    Returns
    -------
    dict with keys: success, test_id, variants_count, topic, goal, variants
    """
    count = max(1, min(count, 10))  # guard against unreasonable counts

    logger.info(
        "task=batch_generate_ab_test|topic=%s|goal=%s|count=%d|status=started",
        topic,
        goal,
        count,
    )

    variants: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Step 1 – Generate all variants
    # ------------------------------------------------------------------

    # 1a. Try IntelligentContentEngine batch method
    if _INTELLIGENT_ENGINE_AVAILABLE and IntelligentContentEngine is not None:
        try:
            import os as _os2

            from ai.openai_provider import OpenAIProvider as _OAIProvider2
            from modules.content.engine import PostGoal as _PG2
            from modules.content.scorer import ContentScorer as _Scorer2

            _provider2 = _OAIProvider2(api_key=_os2.getenv("OPENAI_API_KEY", ""))
            _scorer2 = _Scorer2(
                threshold=float(_os2.getenv("CONTENT_SCORE_THRESHOLD", "6.0"))
            )
            engine = IntelligentContentEngine(  # type: ignore[misc]
                openai_provider=_provider2,
                scorer=_scorer2,
                max_regeneration_attempts=int(
                    _os2.getenv("MAX_REGENERATION_ATTEMPTS", "3")
                ),
                score_threshold=float(_os2.getenv("CONTENT_SCORE_THRESHOLD", "6.0")),
            )
            _valid_goals2 = {g.value: g for g in _PG2}
            _goal_enum2 = _valid_goals2.get(goal, _PG2.EDUCATIONAL)

            if hasattr(engine, "batch_generate_for_ab_test"):
                raw: List[Dict[str, Any]] = engine.batch_generate_for_ab_test(
                    topic=topic, goal=_goal_enum2, count=count
                )
                if raw:
                    for idx, item in enumerate(raw):
                        if item and item.get("content"):
                            item.setdefault("variant_index", idx)
                            item.setdefault("engine_used", "IntelligentContentEngine")
                            variants.append(item)

                    logger.info(
                        "task=batch_generate_ab_test|topic=%s|status=engine_batch_success"
                        "|variants_generated=%d",
                        topic,
                        len(variants),
                    )

            # Engine exists but has no batch method – generate individually
            if not variants:
                for idx in range(count):
                    try:
                        single: Dict[str, Any] = engine.generate_post_with_goal(
                            topic=topic, goal=_goal_enum2
                        )
                        if single and single.get("content"):
                            single.setdefault("variant_index", idx)
                            single.setdefault("engine_used", "IntelligentContentEngine")
                            variants.append(single)
                    except Exception as single_exc:  # noqa: BLE001
                        logger.warning(
                            "task=batch_generate_ab_test|topic=%s|variant_index=%d"
                            "|status=engine_single_failed|error=%s",
                            topic,
                            idx,
                            single_exc,
                        )

        except Exception as engine_exc:  # noqa: BLE001
            logger.warning(
                "task=batch_generate_ab_test|topic=%s|status=engine_error"
                "|error=%s|falling_back=True",
                topic,
                engine_exc,
            )
            variants = []

    # 1b. Fallback – fill any missing variants via generate_post_task
    missing: int = count - len(variants)
    if missing > 0:
        logger.info(
            "task=batch_generate_ab_test|topic=%s|status=fallback_generation"
            "|missing_variants=%d",
            topic,
            missing,
        )
        for i in range(missing):
            try:
                eager = generate_post_task.apply(args=[topic, goal])  # type: ignore[attr-defined]
                result: Dict[str, Any] = eager.result

                if result and result.get("success") and result.get("content"):
                    variants.append(
                        {
                            "content": result["content"],
                            "score": result.get("score", 0.0),
                            "engine_used": result.get("engine_used", "PostGenerator"),
                            "variant_index": len(variants),
                            "goal": goal,
                        }
                    )
                    logger.debug(
                        "task=batch_generate_ab_test|topic=%s|variant_index=%d"
                        "|status=fallback_variant_ok",
                        topic,
                        len(variants) - 1,
                    )
                else:
                    logger.warning(
                        "task=batch_generate_ab_test|topic=%s|fallback_variant=%d"
                        "|status=generate_returned_failure",
                        topic,
                        i,
                    )

            except Exception as var_exc:  # noqa: BLE001
                logger.warning(
                    "task=batch_generate_ab_test|topic=%s|fallback_variant=%d"
                    "|status=variant_exception|error=%s",
                    topic,
                    i,
                    var_exc,
                )

    # Guard – nothing generated at all
    if not variants:
        logger.error(
            "task=batch_generate_ab_test|topic=%s|goal=%s|status=failed"
            "|reason=no_variants_generated",
            topic,
            goal,
        )
        return {
            "success": False,
            "test_id": None,
            "variants_count": 0,
            "topic": topic,
            "goal": goal,
            "variants": [],
            "error": "No variants could be generated",
        }

    # ------------------------------------------------------------------
    # Step 2 – Persist AB test record
    # ------------------------------------------------------------------
    test_id: Optional[str] = None

    if _AB_TESTING_AVAILABLE and ABTestingManager is not None:
        try:
            from database.models import DatabaseManager as _DBM

            _db_ab = _DBM()
            ab_manager = ABTestingManager(_db_ab)  # type: ignore[misc]
            # create_ab_test expects List[str] (content strings), not List[Dict]
            variant_contents: List[str] = [
                str(v.get("content", "")) for v in variants if v.get("content")
            ]
            test_record = ab_manager.create_ab_test(
                topic=topic,
                goal=goal,
                variants=variant_contents,
            )
            # create_ab_test returns an ABTest dataclass — use attribute access
            test_id = str(test_record.test_id)

            logger.info(
                "task=batch_generate_ab_test|topic=%s|test_id=%s"
                "|variants_count=%d|status=ab_test_saved",
                topic,
                test_id,
                len(variants),
            )

        except Exception as ab_exc:  # noqa: BLE001
            logger.warning(
                "task=batch_generate_ab_test|topic=%s|status=ab_save_failed"
                "|variants_count=%d|error=%s",
                topic,
                len(variants),
                ab_exc,
            )
    else:
        logger.warning(
            "task=batch_generate_ab_test|topic=%s|status=ab_manager_unavailable"
            "|variants_count=%d – test not persisted",
            topic,
            len(variants),
        )

    logger.info(
        "task=batch_generate_ab_test|topic=%s|goal=%s|variants_count=%d"
        "|test_id=%s|status=completed",
        topic,
        goal,
        len(variants),
        test_id or "N/A",
    )

    return {
        "success": True,
        "test_id": test_id,
        "variants_count": len(variants),
        "topic": topic,
        "goal": goal,
        "variants": variants,
    }

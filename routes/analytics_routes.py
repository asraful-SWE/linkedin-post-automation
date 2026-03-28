"""
Analytics Routes - Enhanced analytics and A/B testing endpoints.

All routes are mounted under ``/analytics/v2`` and expose:

* Topic-performance predictions and recommendations.
* Best posting-hour analysis.
* Comprehensive performance summaries.
* Trending-topic rankings.
* Full A/B test lifecycle (create → track → determine winner).
* Aggregated winning-pattern insights.

Mount in main.py::

    from routes.analytics_routes import router as analytics_v2_router
    app.include_router(analytics_v2_router)
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from typing import Any, Dict, List, Optional

from database.models import DatabaseManager, Post
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from services.advanced_analytics import (
    ABTestingManager,
    AdvancedAnalyticsEngine,
    IntelligentContentEngine,
)
from services.post_generator import PostGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/analytics/v2",
    tags=["analytics-v2"],
)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class ABTestCreateRequest(BaseModel):
    """Request body for ``POST /analytics/v2/ab-tests``."""

    topic: str = Field(..., min_length=1, description="LinkedIn post topic.")
    goal: str = Field(
        default="educational",
        description=(
            "Content goal key — one of 'educational', 'thought_leadership', "
            "'engagement', 'storytelling', or 'viral'."
        ),
    )


class ABTestCreateResponse(BaseModel):
    test_id: str
    topic: str
    goal: str
    variants_count: int


class DetermineWinnerResponse(BaseModel):
    test_id: str
    winner_variant_id: str
    winning_pattern: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_db_manager(request: Request) -> DatabaseManager:
    """Extract the initialised :class:`DatabaseManager` from application state.

    Raises:
        HTTPException 500: ``db_manager`` not found on ``app.state``.
    """
    db_manager: Optional[DatabaseManager] = getattr(
        request.app.state, "db_manager", None
    )
    if db_manager is None:
        raise HTTPException(
            status_code=500,
            detail="Database manager is not initialised.",
        )
    return db_manager


def _link_post_to_variant(
    db_path: str,
    post_id: int,
    test_id: str,
    goal: str,
) -> None:
    """Back-fill ``ab_test_id`` and ``post_goal`` on a newly created post.

    ``post_goal`` stores the content goal string (e.g. ``"educational"``).
    ``ab_test_id`` links the post back to its parent experiment row in
    ``ab_tests``.  Both columns are added by migrations 3 and 1 respectively;
    the UPDATE is silently skipped when those columns do not yet exist so that
    the endpoint does not fail on un-migrated databases.

    Args:
        db_path:  Filesystem path to the SQLite database.
        post_id:  ID of the post row to update.
        test_id:  UUID of the parent A/B test.
        goal:     Content goal tag to record in ``post_goal``.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE posts SET ab_test_id = ?, post_goal = ? WHERE id = ?",
                (test_id, goal, post_id),
            )
            conn.commit()
    except sqlite3.OperationalError as exc:
        # Migrations may not have run yet — log a warning but do not abort.
        logger.warning(
            "_link_post_to_variant | could not set ab_test_id/post_goal "
            "(run migrations to enable full A/B tracking): %s",
            exc,
        )
    except Exception as exc:
        logger.error(
            "_link_post_to_variant | unexpected error | post_id=%d | test_id=%s | %s",
            post_id,
            test_id,
            exc,
        )
        raise


# ---------------------------------------------------------------------------
# GET /analytics/v2/predictions
# ---------------------------------------------------------------------------


@router.get("/predictions", response_model=List[Dict[str, Any]])
async def get_topic_predictions(
    request: Request,
    count: int = 5,
) -> List[Dict[str, Any]]:
    """Return AI-driven topic recommendations ranked by predicted engagement.

    Uses :meth:`AdvancedAnalyticsEngine.generate_topic_recommendations` to
    blend historical average engagement with a recency cooldown penalty,
    producing an ordered list of the best topics to post about next.

    Query parameters:
        count: Number of recommendations to return (default: 5, max: 50).

    Returns:
        List of dicts ordered by ``predicted_engagement`` descending::

            [
                {
                    "topic":                 str,
                    "predicted_engagement":  float,
                    "historical_avg":        float,
                    "total_posts":           int,
                    "confidence":            "high" | "medium" | "low",
                    "reason":                str,
                    "recently_used":         bool,
                }
            ]

    Raises:
        500: Unexpected error during analytics processing.
    """
    count = max(1, min(count, 50))
    db_manager = _get_db_manager(request)

    try:
        engine = AdvancedAnalyticsEngine(db_manager)
        predictions = engine.generate_topic_recommendations(count=count)
        logger.info(
            "GET /analytics/v2/predictions | count=%d | returned=%d",
            count,
            len(predictions),
        )
        return predictions

    except Exception as exc:
        logger.error("get_topic_predictions failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate topic predictions: {exc}",
        )


# ---------------------------------------------------------------------------
# GET /analytics/v2/posting-time
# ---------------------------------------------------------------------------


@router.get("/posting-time", response_model=List[Dict[str, Any]])
async def get_best_posting_time(request: Request) -> List[Dict[str, Any]]:
    """Identify which hours of the day yield the highest engagement.

    Aggregates published posts by the hour component of ``created_at`` and
    returns per-hour engagement statistics.  Hours are classified as
    ``"peak"``, ``"good"``, ``"low"``, or ``"no_data"`` relative to the best
    performing hour.

    Returns:
        Serialised list of :class:`PostingTimeInsight` objects ordered by
        ``avg_engagement`` descending::

            [
                {
                    "hour":           int,   // 0-23
                    "avg_engagement": float,
                    "post_count":     int,
                    "recommendation": "peak" | "good" | "low" | "no_data",
                }
            ]

    Raises:
        500: Unexpected error during analytics processing.
    """
    db_manager = _get_db_manager(request)

    try:
        engine = AdvancedAnalyticsEngine(db_manager)
        insights = engine.get_best_posting_hours()
        result = [insight.to_dict() for insight in insights]

        logger.info("GET /analytics/v2/posting-time | hour_buckets=%d", len(result))
        return result

    except Exception as exc:
        logger.error("get_best_posting_time failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve posting-time insights: {exc}",
        )


# ---------------------------------------------------------------------------
# GET /analytics/v2/performance-summary
# ---------------------------------------------------------------------------


@router.get("/performance-summary", response_model=Dict[str, Any])
async def get_performance_summary(request: Request) -> Dict[str, Any]:
    """Return a comprehensive all-time performance snapshot.

    Covers post counts by status, engagement aggregates (avg, max, total),
    a 7-day vs prior-7-day weekly trend, and the top 5 performing topics.

    Returns:
        Dict with keys ``period``, ``generated_at``, ``post_counts``,
        ``engagement``, ``weekly_trend``, ``top_topics``, ``last_post_at``.

    Raises:
        500: Unexpected error during analytics processing.
        503: Summary contained an ``"error"`` key (downstream failure).
    """
    db_manager = _get_db_manager(request)

    try:
        engine = AdvancedAnalyticsEngine(db_manager)
        summary = engine.get_performance_summary()

        if "error" in summary:
            logger.error("get_performance_summary | engine error: %s", summary["error"])
            raise HTTPException(
                status_code=503,
                detail=f"Performance summary unavailable: {summary['error']}",
            )

        logger.info("GET /analytics/v2/performance-summary | ok")
        return summary

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_performance_summary failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate performance summary: {exc}",
        )


# ---------------------------------------------------------------------------
# GET /analytics/v2/trending-topics
# ---------------------------------------------------------------------------


@router.get("/trending-topics", response_model=List[Dict[str, Any]])
async def get_trending_topics(
    request: Request,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Rank topics by average engagement over the past 30 days.

    Falls back to all-time data when fewer than 3 recent posts exist, so the
    endpoint is always useful on databases with limited history.

    Query parameters:
        top_n: Maximum number of topics to return (default: 10, max: 100).

    Returns:
        List of dicts ordered by ``avg_engagement`` descending::

            [
                {
                    "rank":           int,
                    "topic":          str,
                    "avg_engagement": float,
                    "post_count":     int,
                    "last_post_at":   str | null,
                }
            ]

    Raises:
        500: Unexpected error during analytics processing.
    """
    top_n = max(1, min(top_n, 100))
    db_manager = _get_db_manager(request)

    try:
        engine = AdvancedAnalyticsEngine(db_manager)
        trending = engine.get_trending_topics_by_engagement(top_n=top_n)

        logger.info(
            "GET /analytics/v2/trending-topics | top_n=%d | returned=%d",
            top_n,
            len(trending),
        )
        return trending

    except Exception as exc:
        logger.error("get_trending_topics failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve trending topics: {exc}",
        )


# ---------------------------------------------------------------------------
# POST /analytics/v2/ab-tests
# ---------------------------------------------------------------------------


@router.post("/ab-tests", response_model=ABTestCreateResponse, status_code=201)
async def create_ab_test(
    body: ABTestCreateRequest,
    request: Request,
) -> ABTestCreateResponse:
    """Create a new A/B test with two generated content variants.

    Workflow
    ~~~~~~~~
    1. Generate two content variants via
       :meth:`IntelligentContentEngine.batch_generate_for_ab_test`.
    2. Persist each variant as a ``pending`` post in the ``posts`` table.
    3. Link each post to the test via ``ab_test_id`` and record the content
       goal in ``post_goal`` (requires migration 1 & 3).
    4. Back-fill each variant's ``post_id`` in the ``ABTestingManager``.
    5. Persist the test record in the ``ab_tests`` table (migration 5).

    Request body::

        {
            "topic": "Python async programming",
            "goal":  "educational"
        }

    Returns:
        201 Created — ``{test_id, topic, goal, variants_count}``.

    Raises:
        400: Topic string is empty.
        500: Content generation or database persistence failed.
    """
    topic = body.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic must not be empty.")

    goal = body.goal.strip() or "educational"
    db_manager = _get_db_manager(request)
    test_id = str(uuid.uuid4())

    try:
        # ---- 1. Generate content variants ----
        generator = PostGenerator()
        ice = IntelligentContentEngine(post_generator=generator)
        variants = ice.batch_generate_for_ab_test(
            topic=topic, goal=goal, num_variants=2
        )

        if not variants:
            raise ValueError("IntelligentContentEngine returned no variants.")

        logger.info(
            "create_ab_test | test_id=%s | topic=%s | goal=%s | variants=%d",
            test_id,
            topic,
            goal,
            len(variants),
        )

        # ---- 2. Persist each variant as a post ----
        abt_manager = ABTestingManager(db_manager=db_manager)

        # First create the test record (variants without post_ids yet) so the
        # manager's update_variant_post_id call has a row to UPDATE.
        abt_manager.create_test(
            test_id=test_id,
            topic=topic,
            goal=goal,
            variants=variants,
        )

        # Now save each variant's post and back-fill post_id.
        for variant in variants:
            post = Post(
                topic=topic,
                content=variant["content"],
                status="pending",
            )
            post_id: int = db_manager.save_post(post)
            variant["post_id"] = post_id

            # Link post → test via new columns (non-fatal if columns absent).
            _link_post_to_variant(
                db_path=db_manager.db_path,
                post_id=post_id,
                test_id=test_id,
                goal=goal,
            )

            # Update the variant JSON in ab_tests with the real post_id.
            try:
                abt_manager.update_variant_post_id(
                    test_id=test_id,
                    variant_id=variant["variant_id"],
                    post_id=post_id,
                )
            except Exception as upd_exc:
                logger.warning(
                    "create_ab_test | update_variant_post_id failed "
                    "(variant_id=%s, post_id=%d): %s",
                    variant["variant_id"],
                    post_id,
                    upd_exc,
                )

            logger.info(
                "create_ab_test | variant saved | test_id=%s | variant_id=%s | post_id=%d",
                test_id,
                variant["variant_id"],
                post_id,
            )

        return ABTestCreateResponse(
            test_id=test_id,
            topic=topic,
            goal=goal,
            variants_count=len(variants),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("create_ab_test failed | test_id=%s | error=%s", test_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"A/B test creation failed: {exc}",
        )


# ---------------------------------------------------------------------------
# GET /analytics/v2/ab-tests
# ---------------------------------------------------------------------------


@router.get("/ab-tests", response_model=List[Dict[str, Any]])
async def list_ab_tests(request: Request) -> List[Dict[str, Any]]:
    """Return all A/B tests with status ``'active'``, ordered newest-first.

    Variants in each test include any ``post_id`` that was back-filled during
    creation.  Engagement enrichment is **not** performed here for performance
    reasons — use ``GET /analytics/v2/ab-tests/{test_id}`` for per-variant
    engagement stats.

    Returns:
        List of test dicts (may be empty).  Each dict has keys
        ``test_id``, ``topic``, ``goal``, ``variants`` (list),
        ``status``, ``created_at``, ``winner_variant_id``, ``winning_pattern``.

    Raises:
        500: Unexpected database error.
    """
    db_manager = _get_db_manager(request)

    try:
        abt_manager = ABTestingManager(db_manager=db_manager)
        active_tests = abt_manager.get_all_active_tests()

        logger.info("GET /analytics/v2/ab-tests | active=%d", len(active_tests))
        return active_tests

    except Exception as exc:
        logger.error("list_ab_tests failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve active A/B tests: {exc}",
        )


# ---------------------------------------------------------------------------
# GET /analytics/v2/ab-tests/{test_id}
# ---------------------------------------------------------------------------


@router.get("/ab-tests/{test_id}", response_model=Dict[str, Any])
async def get_ab_test(test_id: str, request: Request) -> Dict[str, Any]:
    """Return a detailed summary for a specific A/B test.

    Each variant in the response is enriched with live engagement stats
    (``engagement_score``, ``post_status``) fetched from the linked post row.

    Args:
        test_id: UUID of the test to retrieve.

    Returns:
        Full test dict with enriched ``variants`` list.

    Raises:
        404: No test found with the given *test_id*.
        500: Unexpected database error.
    """
    db_manager = _get_db_manager(request)

    try:
        abt_manager = ABTestingManager(db_manager=db_manager)
        summary = abt_manager.get_test_summary(test_id=test_id)

        if summary is None:
            raise HTTPException(
                status_code=404,
                detail=f"A/B test '{test_id}' not found.",
            )

        logger.info("GET /analytics/v2/ab-tests/%s | ok", test_id)
        return summary

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_ab_test failed | test_id=%s | error=%s", test_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve A/B test '{test_id}': {exc}",
        )


# ---------------------------------------------------------------------------
# POST /analytics/v2/ab-tests/{test_id}/determine-winner
# ---------------------------------------------------------------------------


@router.post(
    "/ab-tests/{test_id}/determine-winner",
    response_model=DetermineWinnerResponse,
)
async def determine_ab_test_winner(
    test_id: str,
    request: Request,
) -> DetermineWinnerResponse:
    """Evaluate variant engagement and mark the A/B test as completed.

    The variant whose linked post has the highest ``engagement_score`` is
    declared the winner.  ``winning_pattern`` is set to that variant's content
    style tag (e.g. ``"tips_practical"``).

    The test row is updated: ``status → 'completed'``,
    ``winner_variant_id``, ``winning_pattern``.

    Args:
        test_id: UUID of the test to evaluate.

    Returns:
        DetermineWinnerResponse with ``test_id``, ``winner_variant_id``,
        ``winning_pattern``.

    Raises:
        404: Test not found.
        409: No engagement data available yet — publish the variant posts and
             let them accumulate engagement before calling this endpoint.
        500: Unexpected error during winner determination.
    """
    db_manager = _get_db_manager(request)

    try:
        abt_manager = ABTestingManager(db_manager=db_manager)
        result = abt_manager.determine_winner(test_id=test_id)

        logger.info(
            "POST /analytics/v2/ab-tests/%s/determine-winner | winner=%s | pattern=%s",
            test_id,
            result["winner_variant_id"],
            result["winning_pattern"],
        )

        return DetermineWinnerResponse(
            test_id=result["test_id"],
            winner_variant_id=result["winner_variant_id"],
            winning_pattern=result["winning_pattern"],
        )

    except ValueError as exc:
        err_msg = str(exc)
        logger.warning(
            "determine_ab_test_winner | test_id=%s | client error: %s",
            test_id,
            err_msg,
        )
        # Distinguish "not found" from "no engagement yet" for correct HTTP codes.
        if "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        raise HTTPException(status_code=409, detail=err_msg)

    except Exception as exc:
        logger.error(
            "determine_ab_test_winner failed | test_id=%s | error=%s", test_id, exc
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to determine winner for test '{test_id}': {exc}",
        )


# ---------------------------------------------------------------------------
# GET /analytics/v2/winning-patterns
# ---------------------------------------------------------------------------


@router.get("/winning-patterns", response_model=List[Dict[str, Any]])
async def get_winning_patterns(request: Request) -> List[Dict[str, Any]]:
    """Aggregate content patterns that have won A/B tests.

    Queries completed tests and groups them by ``winning_pattern``, returning
    each unique pattern alongside a ``times_won`` counter and the list of test
    IDs where it was declared the winner.

    This data can be fed back into content strategy decisions — high-frequency
    winning patterns indicate which content styles resonate most with the
    audience.

    Returns:
        List of pattern dicts ordered by ``times_won`` descending::

            [
                {
                    "pattern":    str,         // e.g. "tips_practical"
                    "times_won":  int,
                    "test_ids":   List[str],
                }
            ]
        Returns an empty list when no tests have been completed yet.

    Raises:
        500: Unexpected database error.
    """
    db_manager = _get_db_manager(request)

    try:
        abt_manager = ABTestingManager(db_manager=db_manager)
        patterns = abt_manager.get_winning_patterns()

        logger.info(
            "GET /analytics/v2/winning-patterns | distinct_patterns=%d", len(patterns)
        )
        return patterns

    except Exception as exc:
        logger.error("get_winning_patterns failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve winning patterns: {exc}",
        )

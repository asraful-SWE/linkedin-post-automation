"""
Advanced Analytics Engine - AI-powered analytics, A/B testing, and content intelligence.

Provides three main service classes:

* :class:`AdvancedAnalyticsEngine`  – topic recommendations, posting-time analysis,
  performance summaries, and trending-topic detection.
* :class:`IntelligentContentEngine` – goal-aware content variant generation for
  A/B experiments, built on top of the existing :class:`PostGenerator`.
* :class:`ABTestingManager`         – full A/B test lifecycle: creation, variant
  tracking, winner determination, and pattern extraction.

All classes use direct ``sqlite3`` connections (via ``db_manager.db_path``) for
queries that go beyond the existing :class:`DatabaseManager` helper methods,
mirroring the pattern already established in ``EngagementEngine``.
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from database.models import DatabaseManager
from services.post_generator import PostGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------


@dataclass
class PostingTimeInsight:
    """Engagement statistics aggregated for a single hour of the day (0-23).

    Attributes:
        hour:           Hour of the day in 24-h format (0-23).
        avg_engagement: Mean engagement score across all published posts at
                        this hour.
        post_count:     Number of published posts that contributed to the
                        aggregate.
        recommendation: One of ``"peak"`` (top 20 %), ``"good"``
                        (top 20-50 %), ``"low"`` (bottom 50 %), or
                        ``"no_data"`` when fewer than 2 posts exist for the
                        hour.
    """

    hour: int
    avg_engagement: float
    post_count: int
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict copy suitable for JSON serialisation."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Goal → preferred style mapping
# Styles are drawn from ai/generator.py POST_STYLES to keep things consistent.
# ---------------------------------------------------------------------------

_GOAL_STYLE_MAP: Dict[str, List[str]] = {
    "educational": [
        "tips_practical",
        "lesson_learned",
        "myth_busting",
        "comparison",
    ],
    "thought_leadership": [
        "opinion",
        "real_talk",
        "observation",
        "rant_honest",
    ],
    "engagement": [
        "question_discussion",
        "challenge",
        "personal_story",
    ],
    "storytelling": [
        "personal_story",
        "experience_sharing",
        "lesson_learned",
    ],
    "viral": [
        "myth_busting",
        "question_discussion",
        "challenge",
        "real_talk",
    ],
}

_FALLBACK_STYLES: List[str] = [
    "tips_practical",
    "opinion",
    "personal_story",
    "lesson_learned",
]


# ---------------------------------------------------------------------------
# AdvancedAnalyticsEngine
# ---------------------------------------------------------------------------


class AdvancedAnalyticsEngine:
    """Extends basic engagement tracking with ML-style recommendations,
    time-series analysis, and rich performance aggregation.

    Every public method is safe to call even when the database is empty;
    it will return an empty collection or a zeroed summary dict rather than
    raising an exception.

    Args:
        db_manager: Initialised :class:`DatabaseManager` instance.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    # ------------------------------------------------------------------
    # Topic recommendations
    # ------------------------------------------------------------------

    def generate_topic_recommendations(self, count: int = 5) -> List[Dict[str, Any]]:
        """Generate smart topic recommendations from historical engagement.

        Scoring formula
        ~~~~~~~~~~~~~~~
        ``predicted_engagement = avg_historic_engagement × recency_multiplier``

        * Topics used within the last 3 days receive a **0.80×** multiplier
          to encourage content variety.
        * Topics with ≥ 5 published posts are rated ``"high"`` confidence;
          2-4 posts → ``"medium"``; 0-1 → ``"low"``.

        Args:
            count: Maximum number of recommendations to return.

        Returns:
            List of dicts, sorted by ``predicted_engagement`` descending::

                {
                    "topic": str,
                    "predicted_engagement": float,
                    "historical_avg": float,
                    "total_posts": int,
                    "confidence": "high" | "medium" | "low",
                    "reason": str,
                    "recently_used": bool,
                }
        """
        try:
            performance_data = self.db.get_topic_performance()
            if not performance_data:
                return []

            recent_topics: set = set(self.db.get_recent_topics(days=3))
            recommendations: List[Dict[str, Any]] = []

            for perf in performance_data:
                recency_multiplier = 0.80 if perf.topic in recent_topics else 1.0
                predicted = round(perf.avg_engagement * recency_multiplier, 2)

                if perf.total_posts >= 5:
                    confidence = "high"
                elif perf.total_posts >= 2:
                    confidence = "medium"
                else:
                    confidence = "low"

                if predicted >= 15:
                    reason = (
                        "Historically high engagement — consistently strong performer"
                    )
                elif predicted >= 8:
                    reason = "Solid average engagement — reliable and safe choice"
                elif predicted < 3:
                    reason = "Low engagement history — use sparingly for variety"
                else:
                    reason = "Moderate engagement — good for content diversification"

                if perf.topic in recent_topics:
                    reason += " (recently used — cooldown penalty applied)"

                recommendations.append(
                    {
                        "topic": perf.topic,
                        "predicted_engagement": predicted,
                        "historical_avg": round(perf.avg_engagement, 2),
                        "total_posts": perf.total_posts,
                        "confidence": confidence,
                        "reason": reason,
                        "recently_used": perf.topic in recent_topics,
                    }
                )

            recommendations.sort(key=lambda x: x["predicted_engagement"], reverse=True)
            return recommendations[:count]

        except Exception as exc:
            logger.error("generate_topic_recommendations failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Best posting hours
    # ------------------------------------------------------------------

    def get_best_posting_hours(self) -> List[PostingTimeInsight]:
        """Identify which hours of the day yield the highest engagement.

        Queries published posts, groups them by ``strftime('%H', created_at)``,
        and returns per-hour aggregates sorted by ``avg_engagement`` descending.

        Recommendation thresholds (relative to the best hour):

        * ``>= 80 %`` → ``"peak"``
        * ``>= 50 %`` → ``"good"``
        * ``< 50 %``  → ``"low"``
        * ``< 2`` posts for the hour → ``"no_data"``

        Returns:
            List of :class:`PostingTimeInsight` objects ordered best-first.
        """
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT
                        CAST(strftime('%H', created_at) AS INTEGER) AS hour,
                        AVG(engagement_score)                        AS avg_eng,
                        COUNT(*)                                     AS post_count
                    FROM  posts
                    WHERE status = 'published'
                      AND engagement_score IS NOT NULL
                    GROUP BY hour
                    ORDER BY avg_eng DESC
                    """
                )
                rows: List[Tuple] = cursor.fetchall()

            if not rows:
                return []

            max_eng = max(float(r[1] or 0) for r in rows) or 1.0

            insights: List[PostingTimeInsight] = []
            for hour_raw, avg_eng_raw, post_count in rows:
                avg_eng = float(avg_eng_raw or 0.0)
                n = int(post_count)
                ratio = avg_eng / max_eng

                if n < 2:
                    rec = "no_data"
                elif ratio >= 0.80:
                    rec = "peak"
                elif ratio >= 0.50:
                    rec = "good"
                else:
                    rec = "low"

                insights.append(
                    PostingTimeInsight(
                        hour=int(hour_raw),
                        avg_engagement=round(avg_eng, 2),
                        post_count=n,
                        recommendation=rec,
                    )
                )

            return insights

        except Exception as exc:
            logger.error("get_best_posting_hours failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Performance summary
    # ------------------------------------------------------------------

    def get_performance_summary(self) -> Dict[str, Any]:
        """Return a comprehensive all-time performance snapshot.

        The summary covers:

        * Post counts by status.
        * Engagement aggregates (avg, max, total).
        * 7-day vs prior-7-day trend (direction + percentage change).
        * Top 5 performing topics.
        * Timestamp of the most recent post.

        Returns:
            A dict with keys ``period``, ``generated_at``, ``post_counts``,
            ``engagement``, ``weekly_trend``, ``top_topics``, ``last_post_at``.
            On failure returns ``{"error": str}``.
        """
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                # ---- status breakdown ----
                cursor.execute("SELECT status, COUNT(*) FROM posts GROUP BY status")
                status_counts: Dict[str, int] = dict(cursor.fetchall())

                # ---- engagement aggregates (published posts only) ----
                cursor.execute(
                    """
                    SELECT
                        COUNT(*)              AS total_published,
                        AVG(engagement_score) AS avg_eng,
                        MAX(engagement_score) AS max_eng,
                        SUM(engagement_score) AS total_eng
                    FROM posts
                    WHERE status = 'published'
                    """
                )
                agg = cursor.fetchone()
                total_published = int(agg[0] or 0)
                avg_engagement = round(float(agg[1] or 0.0), 2)
                max_engagement = round(float(agg[2] or 0.0), 2)
                total_engagement = round(float(agg[3] or 0.0), 2)

                # ---- weekly trend ----
                cursor.execute(
                    """
                    SELECT AVG(engagement_score)
                    FROM posts
                    WHERE status = 'published'
                      AND created_at >= datetime('now', '-7 days')
                    """
                )
                this_week_avg = float((cursor.fetchone() or [0])[0] or 0.0)

                cursor.execute(
                    """
                    SELECT AVG(engagement_score)
                    FROM posts
                    WHERE status = 'published'
                      AND created_at >= datetime('now', '-14 days')
                      AND created_at <  datetime('now', '-7 days')
                    """
                )
                prev_week_avg = float((cursor.fetchone() or [0])[0] or 0.0)

                if prev_week_avg > 0:
                    trend_pct = round(
                        ((this_week_avg - prev_week_avg) / prev_week_avg) * 100, 1
                    )
                    trend_dir = (
                        "up"
                        if trend_pct > 5
                        else "down"
                        if trend_pct < -5
                        else "stable"
                    )
                else:
                    trend_pct = 0.0
                    trend_dir = "no_data"

            # ---- top topics ----
            topic_perf = self.db.get_topic_performance()
            top_topics = [
                {
                    "topic": p.topic,
                    "avg_engagement": round(p.avg_engagement, 2),
                    "total_posts": p.total_posts,
                }
                for p in sorted(
                    topic_perf, key=lambda x: x.avg_engagement, reverse=True
                )[:5]
            ]

            last_post = self.db.get_last_post_time()

            return {
                "period": "all_time",
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "post_counts": {
                    "total": sum(status_counts.values()),
                    "by_status": status_counts,
                },
                "engagement": {
                    "total_published": total_published,
                    "avg_engagement_score": avg_engagement,
                    "max_engagement_score": max_engagement,
                    "total_engagement_points": total_engagement,
                },
                "weekly_trend": {
                    "direction": trend_dir,
                    "change_pct": trend_pct,
                    "this_week_avg": round(this_week_avg, 2),
                    "prev_week_avg": round(prev_week_avg, 2),
                },
                "top_topics": top_topics,
                "last_post_at": last_post.isoformat() if last_post else None,
            }

        except Exception as exc:
            logger.error("get_performance_summary failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Trending topics
    # ------------------------------------------------------------------

    def get_trending_topics_by_engagement(
        self, top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Rank topics by average engagement over the past 30 days.

        Falls back to all-time data when fewer than 3 posts exist in the
        30-day window so that the endpoint is always useful on fresh databases.

        Args:
            top_n: Maximum number of topics to return.

        Returns:
            List of dicts ordered by ``avg_engagement`` descending::

                {
                    "rank": int,
                    "topic": str,
                    "avg_engagement": float,
                    "post_count": int,
                    "last_post_at": str | None,
                }
        """
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT
                        topic,
                        AVG(engagement_score) AS avg_eng,
                        COUNT(*)              AS post_count,
                        MAX(created_at)       AS last_post_at
                    FROM  posts
                    WHERE status = 'published'
                      AND created_at >= datetime('now', '-30 days')
                    GROUP BY topic
                    ORDER BY avg_eng DESC
                    LIMIT ?
                    """,
                    (top_n,),
                )
                rows = cursor.fetchall()

                # Fallback to all-time when recent data is thin.
                if len(rows) < 3:
                    cursor.execute(
                        """
                        SELECT
                            topic,
                            AVG(engagement_score) AS avg_eng,
                            COUNT(*)              AS post_count,
                            MAX(created_at)       AS last_post_at
                        FROM  posts
                        WHERE status = 'published'
                        GROUP BY topic
                        ORDER BY avg_eng DESC
                        LIMIT ?
                        """,
                        (top_n,),
                    )
                    rows = cursor.fetchall()

            return [
                {
                    "rank": idx + 1,
                    "topic": row[0],
                    "avg_engagement": round(float(row[1] or 0.0), 2),
                    "post_count": int(row[2]),
                    "last_post_at": row[3],
                }
                for idx, row in enumerate(rows)
            ]

        except Exception as exc:
            logger.error("get_trending_topics_by_engagement failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# IntelligentContentEngine
# ---------------------------------------------------------------------------


class IntelligentContentEngine:
    """Generates goal-aware content variants for A/B experiments.

    Wraps :class:`PostGenerator` and seeds each variant with a different
    content *style* drawn from :data:`_GOAL_STYLE_MAP`.  Because
    :class:`PostGenerator` already randomises style, mood, and length
    internally, calling it twice with the same topic naturally produces
    diverse outputs; this class provides an additional layer of intentional
    goal-alignment.

    Args:
        post_generator: Initialised :class:`PostGenerator` instance.
    """

    def __init__(self, post_generator: PostGenerator) -> None:
        self.generator = post_generator

    def batch_generate_for_ab_test(
        self,
        topic: str,
        goal: str,
        num_variants: int = 2,
    ) -> List[Dict[str, Any]]:
        """Generate *num_variants* content variants for the given topic and goal.

        Each variant is assigned a distinct style from the goal's preferred
        style pool.  Styles are shuffled and consumed without replacement so
        the two arms of the experiment always differ.

        A placeholder string is used for any variant whose generation fails
        so that the overall A/B test creation request does not abort.

        Args:
            topic:        LinkedIn post topic string.
            goal:         Content-goal key — one of ``"educational"``,
                          ``"thought_leadership"``, ``"engagement"``,
                          ``"storytelling"``, or ``"viral"``.  Unknown values
                          fall back to :data:`_FALLBACK_STYLES`.
            num_variants: Number of variants to create (default: 2).

        Returns:
            List of variant dicts (one per requested variant)::

                {
                    "variant_id":      str,   # UUID v4
                    "style":           str,   # content-style tag
                    "goal":            str,   # echoed from input
                    "content":         str,   # full generated post text
                    "content_preview": str,   # first 150 chars + ellipsis
                    "word_count":      int,
                    "post_id":         None,  # populated by the caller after DB save
                }
        """
        style_pool = list(_GOAL_STYLE_MAP.get(goal, _FALLBACK_STYLES))
        random.shuffle(style_pool)

        variants: List[Dict[str, Any]] = []
        used_styles: List[str] = []

        for i in range(num_variants):
            # Pick the next unused style; cycle if the pool is exhausted.
            style = next(
                (s for s in style_pool if s not in used_styles),
                style_pool[i % len(style_pool)] if style_pool else "opinion",
            )
            used_styles.append(style)

            try:
                content: str = self.generator.generate_post(topic)
                if not content or not content.strip():
                    logger.warning(
                        "Empty content returned for variant %d (topic=%s) — using placeholder",
                        i + 1,
                        topic,
                    )
                    content = (
                        f"[Variant {i + 1} — placeholder] "
                        f"Content for topic '{topic}' could not be generated."
                    )
            except Exception as exc:
                logger.error(
                    "Variant %d generation failed (topic=%s, style=%s): %s — using placeholder",
                    i + 1,
                    topic,
                    style,
                    exc,
                )
                content = (
                    f"[Variant {i + 1} — generation error] "
                    f"Topic: {topic}. Please regenerate."
                )

            preview = content[:150].rstrip()
            if len(content) > 150:
                preview += "…"

            variants.append(
                {
                    "variant_id": str(uuid.uuid4()),
                    "style": style,
                    "goal": goal,
                    "content": content,
                    "content_preview": preview,
                    "word_count": len(content.split()),
                    "post_id": None,  # caller populates after DB save
                }
            )

        logger.info(
            "batch_generate_for_ab_test | topic=%s | goal=%s | variants=%d",
            topic,
            goal,
            len(variants),
        )
        return variants


# ---------------------------------------------------------------------------
# ABTestingManager
# ---------------------------------------------------------------------------


class ABTestingManager:
    """Manages the complete A/B test lifecycle.

    State machine::

        created (status='active')
            └─► determine_winner() ──► completed (status='completed',
                                                  winner_variant_id=...,
                                                  winning_pattern=...)

    Posts linked to an experiment carry ``ab_test_id`` in the ``posts`` table
    (added by migration 3).  Variant metadata — including the back-reference
    ``post_id`` — is serialised as a JSON array in ``ab_tests.variants``.

    Args:
        db_manager: Initialised :class:`DatabaseManager` instance.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def create_test(
        self,
        test_id: str,
        topic: str,
        goal: str,
        variants: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Persist a new A/B test record with status ``"active"``.

        The *variants* list (including any ``post_id`` fields populated by the
        caller) is serialised to JSON and stored verbatim.

        Args:
            test_id:  Unique identifier for this test (caller supplies UUID).
            topic:    Post topic shared across all variants.
            goal:     Content goal tag.
            variants: List of variant dicts as returned by
                      :meth:`IntelligentContentEngine.batch_generate_for_ab_test`
                      *after* the caller has set ``post_id`` on each entry.

        Returns:
            Dict representation of the newly created test.

        Raises:
            ValueError: If *test_id* already exists in the database.
        """
        variants_json = json.dumps(variants, ensure_ascii=False)
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO ab_tests
                        (test_id, topic, goal, variants, status, created_at)
                    VALUES
                        (?, ?, ?, ?, 'active', CURRENT_TIMESTAMP)
                    """,
                    (test_id, topic, goal, variants_json),
                )
                conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"A/B test with ID '{test_id}' already exists.") from exc

        logger.info(
            "ab_test_created | test_id=%s | topic=%s | goal=%s | variants=%d",
            test_id,
            topic,
            goal,
            len(variants),
        )
        return {
            "test_id": test_id,
            "topic": topic,
            "goal": goal,
            "variants": variants,
            "status": "active",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "winner_variant_id": None,
            "winning_pattern": None,
        }

    def determine_winner(self, test_id: str) -> Dict[str, Any]:
        """Evaluate variant performance and mark the test as completed.

        Algorithm
        ~~~~~~~~~
        1. Load the test record and decode the variants JSON.
        2. For each variant that has a ``post_id``, query the linked post's
           ``engagement_score``.
        3. The variant with the highest engagement score is declared the
           winner.  Ties are broken in favour of the first variant.
        4. ``winning_pattern`` is set to the winner's ``style`` field.
        5. The ``ab_tests`` row is updated: ``status → 'completed'``,
           ``winner_variant_id``, ``winning_pattern``.

        Args:
            test_id: ID of the test to evaluate.

        Returns:
            Dict with keys ``test_id``, ``winner_variant_id``,
            ``winning_pattern``, ``winner_avg_engagement``,
            ``variant_stats``::

                {
                    "test_id": str,
                    "winner_variant_id": str,
                    "winning_pattern": str,
                    "winner_avg_engagement": float,
                    "variant_stats": [
                        {"variant_id": str, "post_id": int,
                         "engagement_score": float, "style": str}
                    ],
                }

        Raises:
            ValueError: When *test_id* is not found, or when none of the
                        variants has a linked post with engagement data yet.
        """
        test = self._get_raw_test(test_id)
        if test is None:
            raise ValueError(f"A/B test '{test_id}' not found.")

        variants: List[Dict[str, Any]] = test["variants"]
        if not variants:
            raise ValueError(f"A/B test '{test_id}' has no variants.")

        # Collect engagement score per variant.
        variant_stats: List[Dict[str, Any]] = []
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                for variant in variants:
                    post_id = variant.get("post_id")
                    if post_id is None:
                        logger.warning(
                            "Variant %s in test %s has no post_id — skipping",
                            variant.get("variant_id"),
                            test_id,
                        )
                        continue

                    cursor.execute(
                        "SELECT engagement_score FROM posts WHERE id = ?",
                        (post_id,),
                    )
                    row = cursor.fetchone()
                    score = float(row[0] or 0.0) if row else 0.0
                    variant_stats.append(
                        {
                            "variant_id": variant.get("variant_id"),
                            "post_id": post_id,
                            "engagement_score": score,
                            "style": variant.get("style", "unknown"),
                        }
                    )
        except Exception as exc:
            logger.error(
                "determine_winner: engagement query failed | test_id=%s | error=%s",
                test_id,
                exc,
            )
            raise

        if not variant_stats:
            raise ValueError(
                f"No variants with linked posts found for test '{test_id}'. "
                "Ensure posts have been created and associated with the test."
            )

        # Check that at least one post has non-zero engagement.
        if all(v["engagement_score"] == 0.0 for v in variant_stats):
            raise ValueError(
                f"No engagement data available yet for test '{test_id}'. "
                "Publish the test posts and wait for LinkedIn engagement before "
                "determining a winner."
            )

        # Winner = variant with highest engagement score (first on tie).
        winner = max(variant_stats, key=lambda v: v["engagement_score"])
        winner_variant_id: str = winner["variant_id"]
        winning_pattern: str = winner["style"]
        winner_engagement: float = winner["engagement_score"]

        # Persist result.
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute(
                    """
                    UPDATE ab_tests
                    SET    status            = 'completed',
                           winner_variant_id = ?,
                           winning_pattern   = ?
                    WHERE  test_id           = ?
                    """,
                    (winner_variant_id, winning_pattern, test_id),
                )
                conn.commit()
        except Exception as exc:
            logger.error(
                "determine_winner: DB update failed | test_id=%s | error=%s",
                test_id,
                exc,
            )
            raise

        logger.info(
            "ab_test_completed | test_id=%s | winner=%s | pattern=%s | engagement=%.2f",
            test_id,
            winner_variant_id,
            winning_pattern,
            winner_engagement,
        )

        return {
            "test_id": test_id,
            "winner_variant_id": winner_variant_id,
            "winning_pattern": winning_pattern,
            "winner_avg_engagement": round(winner_engagement, 2),
            "variant_stats": variant_stats,
        }

    def update_variant_post_id(
        self, test_id: str, variant_id: str, post_id: int
    ) -> None:
        """Back-fill ``post_id`` on a variant after the post has been saved.

        Called by the analytics route after each variant post is persisted
        to the ``posts`` table.  Deserialises the variants JSON, patches the
        matching entry, and writes the updated JSON back to ``ab_tests``.

        Args:
            test_id:    The test this variant belongs to.
            variant_id: UUID of the specific variant to update.
            post_id:    Database ID of the newly saved post.

        Raises:
            ValueError: If *test_id* or *variant_id* is not found.
        """
        test = self._get_raw_test(test_id)
        if test is None:
            raise ValueError(f"A/B test '{test_id}' not found.")

        variants: List[Dict[str, Any]] = test["variants"]
        matched = False
        for variant in variants:
            if variant.get("variant_id") == variant_id:
                variant["post_id"] = post_id
                matched = True
                break

        if not matched:
            raise ValueError(f"Variant '{variant_id}' not found in test '{test_id}'.")

        updated_json = json.dumps(variants, ensure_ascii=False)
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute(
                    "UPDATE ab_tests SET variants = ? WHERE test_id = ?",
                    (updated_json, test_id),
                )
                conn.commit()
        except Exception as exc:
            logger.error(
                "update_variant_post_id failed | test_id=%s | variant_id=%s | error=%s",
                test_id,
                variant_id,
                exc,
            )
            raise

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_all_active_tests(self) -> List[Dict[str, Any]]:
        """Return all tests with ``status = 'active'``, newest first.

        Returns:
            List of test dicts (variants decoded from JSON).  Empty list on
            error or when no active tests exist.
        """
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT test_id, topic, goal, variants, status,
                           created_at, winner_variant_id, winning_pattern
                    FROM   ab_tests
                    WHERE  status = 'active'
                    ORDER  BY created_at DESC
                    """
                )
                return [self._row_to_dict(row) for row in cursor.fetchall()]
        except Exception as exc:
            logger.error("get_all_active_tests failed: %s", exc)
            return []

    def get_test_summary(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Return a detailed summary for *test_id*, enriched with per-variant
        engagement stats from the linked posts.

        Args:
            test_id: The test to summarise.

        Returns:
            Dict with full test metadata plus enriched ``variants`` list, or
            ``None`` if the test does not exist.
        """
        test = self._get_raw_test(test_id)
        if test is None:
            return None

        variants: List[Dict[str, Any]] = test["variants"]

        # Enrich each variant with live engagement stats.
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                for variant in variants:
                    post_id = variant.get("post_id")
                    if post_id is None:
                        variant["engagement_score"] = None
                        variant["post_status"] = None
                        continue

                    cursor.execute(
                        """
                        SELECT engagement_score, status
                        FROM   posts
                        WHERE  id = ?
                        """,
                        (post_id,),
                    )
                    row = cursor.fetchone()
                    if row:
                        variant["engagement_score"] = round(float(row[0] or 0.0), 2)
                        variant["post_status"] = row[1]
                    else:
                        variant["engagement_score"] = None
                        variant["post_status"] = "not_found"
        except Exception as exc:
            logger.warning(
                "get_test_summary: could not enrich variants | test_id=%s | error=%s",
                test_id,
                exc,
            )

        return {**test, "variants": variants}

    def get_winning_patterns(self) -> List[Dict[str, Any]]:
        """Aggregate winning patterns from all completed A/B tests.

        Returns:
            List of dicts ordered by ``times_won`` descending::

                {
                    "pattern": str,          # style tag (e.g. "tips_practical")
                    "times_won": int,
                    "test_ids": List[str],   # IDs of the tests where this pattern won
                }
        """
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT
                        winning_pattern,
                        COUNT(*)                     AS times_won,
                        GROUP_CONCAT(test_id, ',')   AS test_ids
                    FROM   ab_tests
                    WHERE  status          = 'completed'
                      AND  winning_pattern IS NOT NULL
                    GROUP  BY winning_pattern
                    ORDER  BY times_won DESC
                    """
                )
                rows = cursor.fetchall()

            return [
                {
                    "pattern": row[0],
                    "times_won": int(row[1]),
                    "test_ids": row[2].split(",") if row[2] else [],
                }
                for row in rows
            ]

        except Exception as exc:
            logger.error("get_winning_patterns failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_raw_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single ``ab_tests`` row as a decoded dict, or ``None``."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT test_id, topic, goal, variants, status,
                           created_at, winner_variant_id, winning_pattern
                    FROM   ab_tests
                    WHERE  test_id = ?
                    """,
                    (test_id,),
                )
                row = cursor.fetchone()
                return self._row_to_dict(row) if row else None
        except Exception as exc:
            logger.error("_get_raw_test failed | test_id=%s | error=%s", test_id, exc)
            return None

    @staticmethod
    def _row_to_dict(row: Tuple) -> Dict[str, Any]:
        """Convert a raw ``ab_tests`` SELECT row to a decoded dict.

        Column order must match the SELECT in all callers:
        ``test_id, topic, goal, variants, status, created_at,
        winner_variant_id, winning_pattern``.
        """
        test_id, topic, goal, variants_json, status, created_at, winner_id, pattern = (
            row
        )
        return {
            "test_id": test_id,
            "topic": topic,
            "goal": goal,
            "variants": json.loads(variants_json) if variants_json else [],
            "status": status,
            "created_at": created_at,
            "winner_variant_id": winner_id,
            "winning_pattern": pattern,
        }

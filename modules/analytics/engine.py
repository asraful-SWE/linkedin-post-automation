"""
Advanced Analytics Engine - Deep performance insights, predictive scoring,
and posting-time optimisation for the LinkedIn AI Poster system.
"""

import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from database.models import DatabaseManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PostingTimeInsight:
    hour: int
    avg_engagement: float
    post_count: int
    recommendation: str  # "best" | "good" | "avoid"


@dataclass
class TopicPrediction:
    topic: str
    predicted_score: float
    confidence: str  # "high" | "medium" | "low"
    reasoning: str
    recommended_goal: str  # "educational" | "story" | "viral" | "authority"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AdvancedAnalyticsEngine:
    """
    Provides deep analytics, trend analysis, and predictive insights to
    continuously optimise LinkedIn post scheduling and content strategy.

    All DB access uses the shared `db_path` from the injected DatabaseManager
    so that every query operates on the same SQLite file as the rest of the
    application.
    """

    # ── Hard-coded fallback hour sets ────────────────────────────────────────
    _DEFAULT_BEST_HOURS: frozenset = frozenset({9, 10, 19, 20})
    _DEFAULT_GOOD_HOURS: frozenset = frozenset({12, 13})
    _DEFAULT_AVOID_HOURS: frozenset = frozenset(range(0, 7))  # midnight → 6 AM

    # ── Goal-classification keyword maps ─────────────────────────────────────
    _EDUCATIONAL_KW = frozenset({"tips", "how", "guide", "learn", "শেখা"})
    _STORY_KW = frozenset({"story", "journey", "experience", "অভিজ্ঞতা"})
    _VIRAL_KW = frozenset({"opinion", "thoughts", "বিতর্ক", "সত্যি"})

    # ── Trend thresholds (10 % swing either way) ─────────────────────────────
    _TREND_UP_THRESHOLD: float = 0.10
    _TREND_DOWN_THRESHOLD: float = -0.10

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    # =========================================================================
    # Public API
    # =========================================================================

    def get_best_posting_hours(
        self, days_lookback: int = 30
    ) -> List[PostingTimeInsight]:
        """
        Return posting-hour insights sorted by average engagement (best first).

        Queries published posts within *days_lookback* days and groups them by
        the hour-of-day extracted from ``created_at``.  When fewer than 3
        data-bearing hours are found the method falls back to a hard-coded
        schedule so callers always receive a usable result.

        Args:
            days_lookback: How many calendar days to look back.  Default 30.

        Returns:
            List[PostingTimeInsight] sorted by avg_engagement descending.
        """
        try:
            rows = self._query_hour_engagement(days_lookback)

            if len(rows) < 3:
                logger.info(
                    "Insufficient posting-hour data (%d rows); using defaults.",
                    len(rows),
                )
                return self._default_hour_insights()

            # Determine dynamic thresholds from the data distribution
            all_avgs: List[float] = [r[1] for r in rows]
            max_avg = max(all_avgs) if all_avgs else 1.0
            threshold_best = max_avg * 0.75
            threshold_good = max_avg * 0.45

            insights: List[PostingTimeInsight] = []
            for hour_str, avg_eng, count in rows:
                hour = int(hour_str)
                rec = self._classify_hour_dynamic(
                    hour, avg_eng, threshold_best, threshold_good
                )
                insights.append(
                    PostingTimeInsight(
                        hour=hour,
                        avg_engagement=round(avg_eng, 2),
                        post_count=count,
                        recommendation=rec,
                    )
                )

            insights.sort(key=lambda x: x.avg_engagement, reverse=True)
            logger.info("Posting-hour insights computed from %d data rows.", len(rows))
            return insights

        except Exception as exc:
            logger.error("get_best_posting_hours failed: %s", exc, exc_info=True)
            return self._default_hour_insights()

    # -------------------------------------------------------------------------

    def predict_topic_performance(self, topic: str) -> TopicPrediction:
        """
        Predict expected engagement for *topic* using available historical data.

        Confidence tiers:
        - **high**   → 3+ direct posts exist for this exact topic
        - **medium** → 1–2 direct posts, or similar-topic proxy data available
        - **low**    → no data at all; falls back to global average

        Args:
            topic: The topic string to evaluate.

        Returns:
            A ``TopicPrediction`` with predicted score, confidence, reasoning,
            and a recommended content goal.
        """
        try:
            topic_rows = self._get_topic_history(topic)
            global_avg = self._global_avg_engagement()
            direct_count = len(topic_rows)
            direct_avg = (
                sum(r[1] for r in topic_rows) / direct_count if topic_rows else 0.0
            )

            similar_rows = self._get_similar_topic_rows(topic, exclude_topic=topic)
            similar_count = len(similar_rows)
            similar_avg = (
                sum(r[1] for r in similar_rows) / similar_count if similar_rows else 0.0
            )

            if direct_count >= 3:
                confidence = "high"
                predicted = direct_avg
                reasoning = (
                    f"Based on {direct_count} historical posts for this exact "
                    f"topic (avg engagement: {direct_avg:.1f}).  High confidence "
                    f"prediction."
                )

            elif direct_count >= 1:
                confidence = "medium"
                # Weight direct data more heavily than the global baseline
                predicted = (direct_avg * 0.65) + (global_avg * 0.35)
                reasoning = (
                    f"Limited direct history ({direct_count} post(s), avg score "
                    f"{direct_avg:.1f}).  Blended with the global average "
                    f"({global_avg:.1f}) for a balanced estimate."
                )

            elif similar_count > 0:
                confidence = "medium"
                predicted = (similar_avg * 0.50) + (global_avg * 0.50)
                reasoning = (
                    f"No direct posts found for this topic.  Inferred from "
                    f"{similar_count} posts on similar topics "
                    f"(avg score {similar_avg:.1f}), averaged with the global "
                    f"baseline ({global_avg:.1f})."
                )

            else:
                confidence = "low"
                predicted = global_avg
                reasoning = (
                    f"No historical data for this topic or semantically similar "
                    f"ones.  Defaulting to the global average engagement "
                    f"({global_avg:.1f}).  Consider this an exploratory post."
                )

            goal = self._classify_goal(topic)

            return TopicPrediction(
                topic=topic,
                predicted_score=round(predicted, 2),
                confidence=confidence,
                reasoning=reasoning,
                recommended_goal=goal,
            )

        except Exception as exc:
            logger.error(
                "predict_topic_performance failed for '%s': %s",
                topic,
                exc,
                exc_info=True,
            )
            return TopicPrediction(
                topic=topic,
                predicted_score=5.0,
                confidence="low",
                reasoning="Prediction unavailable due to an internal error.",
                recommended_goal="authority",
            )

    # -------------------------------------------------------------------------

    def get_trending_topics_by_engagement(
        self, top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Return the top *top_n* topics ordered by average engagement score.

        Each entry includes a ``trend_direction`` field (``"up"`` / ``"down"`` /
        ``"stable"``) that compares the last 14 days against the all-time
        historical average.

        Args:
            top_n: Maximum number of topics to return.  Default 10.

        Returns:
            List of dicts: ``{topic, avg_engagement, post_count, trend_direction}``.
        """
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT topic, avg_engagement, total_posts
                    FROM   topic_performance
                    ORDER  BY avg_engagement DESC
                    LIMIT  ?
                    """,
                    (top_n,),
                )
                rows = cursor.fetchall()

            results: List[Dict[str, Any]] = []
            for topic, avg_eng, post_count in rows:
                trend = self._calculate_topic_trend(topic, avg_eng)
                results.append(
                    {
                        "topic": topic,
                        "avg_engagement": round(avg_eng, 2),
                        "post_count": post_count,
                        "trend_direction": trend,
                    }
                )

            logger.info("Trending topics retrieved: %d result(s).", len(results))
            return results

        except Exception as exc:
            logger.error(
                "get_trending_topics_by_engagement failed: %s", exc, exc_info=True
            )
            return []

    # -------------------------------------------------------------------------

    def auto_adjust_posting_time(self, current_hours: List[int]) -> List[int]:
        """
        Compare *current_hours* against data-driven best hours and suggest
        an optimal schedule of the same length.

        If the current schedule already covers the best hours it is returned
        unchanged.  Otherwise the method builds a replacement list by first
        filling with "best" hours then padding with "good" hours.

        Args:
            current_hours: The caller's existing posting-hour schedule.

        Returns:
            A list of hours (same length as *current_hours*) representing the
            recommended posting schedule.
        """
        try:
            if not current_hours:
                return current_hours

            insights = self.get_best_posting_hours()

            best_hours = [i.hour for i in insights if i.recommendation == "best"]
            good_hours = [i.hour for i in insights if i.recommendation == "good"]

            if not best_hours:
                logger.info(
                    "No 'best' hours identified from data; keeping current schedule."
                )
                return current_hours

            current_set = set(current_hours)
            best_set = set(best_hours)

            # Already optimal – no change needed
            if current_set.issubset(best_set):
                logger.info("Current posting hours already optimal; no change.")
                return sorted(current_hours)

            # Build replacement list: best hours first, then good hours
            candidate_pool = best_hours + [h for h in good_hours if h not in best_set]
            seen: set = set()
            suggested: List[int] = []
            for h in candidate_pool:
                if h not in seen:
                    seen.add(h)
                    suggested.append(h)
                if len(suggested) == len(current_hours):
                    break

            # Pad with remaining good hours if needed (shouldn't normally happen)
            if len(suggested) < len(current_hours):
                remaining = [h for h in range(7, 23) if h not in seen]
                for h in remaining:
                    suggested.append(h)
                    if len(suggested) == len(current_hours):
                        break

            suggested.sort()
            logger.info(
                "Posting hours adjusted: %s → %s",
                sorted(current_hours),
                suggested,
            )
            return suggested

        except Exception as exc:
            logger.error("auto_adjust_posting_time failed: %s", exc, exc_info=True)
            return current_hours

    # -------------------------------------------------------------------------

    def generate_topic_recommendations(self, count: int = 5) -> List[TopicPrediction]:
        """
        Recommend topics that haven't been used in the last 3 days, ranked by
        predicted engagement score.

        To keep the method fast the candidate set is capped at 50 topics
        pulled from the topic_performance table (ordered by avg_engagement so
        the best historical performers are considered first).

        Args:
            count: Number of recommendations to return.  Default 5.

        Returns:
            List[TopicPrediction] sorted by predicted_score descending.
        """
        try:
            recent_topics = set(self.db.get_recent_topics(days=3))
            all_performance = self.db.get_topic_performance()

            # Filter out recently used topics; cap at 50 for speed
            candidates: List[str] = []
            for tp in all_performance:
                if tp.topic not in recent_topics:
                    candidates.append(tp.topic)
                if len(candidates) >= 50:
                    break

            if not candidates:
                logger.info(
                    "All tracked topics used recently; no recommendations available."
                )
                return []

            predictions = [self.predict_topic_performance(t) for t in candidates]
            predictions.sort(key=lambda p: p.predicted_score, reverse=True)

            top = predictions[:count]
            logger.info(
                "Generated %d topic recommendation(s) from %d candidate(s).",
                len(top),
                len(candidates),
            )
            return top

        except Exception as exc:
            logger.error(
                "generate_topic_recommendations failed: %s", exc, exc_info=True
            )
            return []

    # -------------------------------------------------------------------------

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Return a comprehensive performance snapshot dictionary.

        Keys returned:
        - ``total_published_posts``
        - ``avg_engagement_rate``
        - ``best_performing_topic``  → ``{topic, score}``
        - ``worst_performing_topic`` → ``{topic, score}``
        - ``total_likes``
        - ``total_comments``
        - ``total_impressions``
        - ``posting_consistency_score`` (0-100)
        - ``top_hours``  → list of up to 3 best posting hours
        - ``trend``      → ``"improving"`` | ``"declining"`` | ``"stable"``
        """
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                # ── Aggregate stats over all published posts ──────────────────
                cursor.execute(
                    """
                    SELECT
                        COUNT(p.id)             AS total_posts,
                        AVG(p.engagement_score) AS avg_engagement,
                        COALESCE(SUM(a.likes),       0) AS total_likes,
                        COALESCE(SUM(a.comments),    0) AS total_comments,
                        COALESCE(SUM(a.impressions), 0) AS total_impressions
                    FROM  posts p
                    LEFT  JOIN analytics a ON a.post_id = p.id
                    WHERE p.status = 'published'
                    """
                )
                agg = cursor.fetchone() or (0, 0.0, 0, 0, 0)
                (
                    total_posts,
                    avg_eng,
                    total_likes,
                    total_comments,
                    total_impressions,
                ) = agg
                total_posts = total_posts or 0
                avg_eng = avg_eng or 0.0
                total_likes = total_likes or 0
                total_comments = total_comments or 0
                total_impressions = total_impressions or 0

                # ── Best and worst topics ─────────────────────────────────────
                cursor.execute(
                    """
                    SELECT topic, avg_engagement
                    FROM   topic_performance
                    ORDER  BY avg_engagement DESC
                    LIMIT  1
                    """
                )
                best_row = cursor.fetchone()

                cursor.execute(
                    """
                    SELECT topic, avg_engagement
                    FROM   topic_performance
                    WHERE  total_posts > 0
                    ORDER  BY avg_engagement ASC
                    LIMIT  1
                    """
                )
                worst_row = cursor.fetchone()

                # ── Posting consistency over last 30 days ─────────────────────
                cursor.execute(
                    """
                    SELECT date(created_at) AS day, COUNT(*) AS cnt
                    FROM   posts
                    WHERE  created_at >= datetime('now', '-30 days')
                    GROUP  BY day
                    """
                )
                daily_counts = [row[1] for row in cursor.fetchall()]
                consistency = self._calculate_consistency_score(daily_counts)

                # ── Trend: last 7 days vs the previous 7 days ────────────────
                cursor.execute(
                    """
                    SELECT AVG(engagement_score)
                    FROM   posts
                    WHERE  status     = 'published'
                      AND  created_at >= datetime('now', '-7 days')
                    """
                )
                recent_7 = cursor.fetchone()[0] or 0.0

                cursor.execute(
                    """
                    SELECT AVG(engagement_score)
                    FROM   posts
                    WHERE  status     = 'published'
                      AND  created_at >= datetime('now', '-14 days')
                      AND  created_at <  datetime('now', '-7 days')
                    """
                )
                prev_7 = cursor.fetchone()[0] or 0.0

            trend = self._determine_trend(recent_7, prev_7)
            top_hours = [i.hour for i in self.get_best_posting_hours()[:3]]

            return {
                "total_published_posts": total_posts,
                "avg_engagement_rate": round(avg_eng, 2),
                "best_performing_topic": {
                    "topic": best_row[0] if best_row else "N/A",
                    "score": round(best_row[1], 2) if best_row else 0.0,
                },
                "worst_performing_topic": {
                    "topic": worst_row[0] if worst_row else "N/A",
                    "score": round(worst_row[1], 2) if worst_row else 0.0,
                },
                "total_likes": total_likes,
                "total_comments": total_comments,
                "total_impressions": total_impressions,
                "posting_consistency_score": consistency,
                "top_hours": top_hours,
                "trend": trend,
            }

        except Exception as exc:
            logger.error("get_performance_summary failed: %s", exc, exc_info=True)
            return {
                "total_published_posts": 0,
                "avg_engagement_rate": 0.0,
                "best_performing_topic": {"topic": "N/A", "score": 0.0},
                "worst_performing_topic": {"topic": "N/A", "score": 0.0},
                "total_likes": 0,
                "total_comments": 0,
                "total_impressions": 0,
                "posting_consistency_score": 0,
                "top_hours": [],
                "trend": "stable",
            }

    # =========================================================================
    # Private helpers
    # =========================================================================

    # ── DB queries ────────────────────────────────────────────────────────────

    def _query_hour_engagement(
        self, days_lookback: int
    ) -> List[Tuple[str, float, int]]:
        """
        Raw SQL: average engagement grouped by posting hour (00–23).

        Returns:
            List of (hour_str, avg_engagement, post_count) tuples.
        """
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    strftime('%H', p.created_at) AS hour,
                    AVG(p.engagement_score)        AS avg_engagement,
                    COUNT(*)                        AS post_count
                FROM  posts p
                WHERE p.created_at >= datetime('now', ? || ' days')
                  AND p.status = 'published'
                GROUP BY strftime('%H', p.created_at)
                ORDER BY avg_engagement DESC
                """,
                (f"-{days_lookback}",),
            )
            return cursor.fetchall()  # [(hour_str, avg_eng, count), ...]

    def _get_topic_history(self, topic: str) -> List[Tuple[int, float]]:
        """Return (post_id, engagement_score) for all published posts of *topic*."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, engagement_score
                    FROM   posts
                    WHERE  topic  = ?
                      AND  status = 'published'
                    ORDER  BY created_at DESC
                    """,
                    (topic,),
                )
                return cursor.fetchall()
        except Exception as exc:
            logger.warning("_get_topic_history error: %s", exc)
            return []

    def _get_similar_topic_rows(
        self, topic: str, *, exclude_topic: str
    ) -> List[Tuple[int, float]]:
        """
        Return rows for published posts whose topic shares at least one
        meaningful word (length > 2) with *topic*.

        Args:
            topic:         The reference topic to extract keywords from.
            exclude_topic: Exact topic string to exclude (avoids double-counting
                           direct posts already fetched separately).
        """
        try:
            words = {w.lower() for w in topic.split() if len(w) > 2}
            if not words:
                return []

            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, engagement_score, topic
                    FROM   posts
                    WHERE  topic  != ?
                      AND  status = 'published'
                    """,
                    (exclude_topic,),
                )
                rows = cursor.fetchall()

            return [(r[0], r[1]) for r in rows if any(w in r[2].lower() for w in words)]
        except Exception as exc:
            logger.warning("_get_similar_topic_rows error: %s", exc)
            return []

    def _global_avg_engagement(self) -> float:
        """Global average engagement_score across all published posts."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT AVG(engagement_score)
                    FROM   posts
                    WHERE  status = 'published'
                    """
                )
                result = cursor.fetchone()[0]
                return float(result) if result is not None else 5.0
        except Exception as exc:
            logger.warning("_global_avg_engagement error: %s", exc)
            return 5.0

    def _calculate_topic_trend(self, topic: str, historical_avg: float) -> str:
        """
        Compare the recent 14-day average engagement for *topic* against
        its all-time *historical_avg*.

        Returns ``"up"``, ``"down"``, or ``"stable"``.
        """
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT AVG(engagement_score)
                    FROM   posts
                    WHERE  topic      = ?
                      AND  status     = 'published'
                      AND  created_at >= datetime('now', '-14 days')
                    """,
                    (topic,),
                )
                recent_avg = cursor.fetchone()[0] or 0.0

            if historical_avg == 0:
                return "stable"

            change = (recent_avg - historical_avg) / historical_avg
            if change > self._TREND_UP_THRESHOLD:
                return "up"
            if change < self._TREND_DOWN_THRESHOLD:
                return "down"
            return "stable"

        except Exception as exc:
            logger.warning("_calculate_topic_trend error: %s", exc)
            return "stable"

    # ── Classification helpers ────────────────────────────────────────────────

    def _classify_hour_dynamic(
        self,
        hour: int,
        avg_eng: float,
        threshold_best: float,
        threshold_good: float,
    ) -> str:
        """
        Assign a recommendation label to a posting hour using data-derived
        thresholds.  Hours in the hard-coded avoid window are always "avoid"
        regardless of engagement (low-volume nights).
        """
        if hour in self._DEFAULT_AVOID_HOURS:
            return "avoid"
        if avg_eng >= threshold_best:
            return "best"
        if avg_eng >= threshold_good:
            return "good"
        return "avoid"

    def _classify_goal(self, topic: str) -> str:
        """
        Determine the best content goal for *topic* based on keyword matching.
        Evaluated in priority order: educational → story → viral → authority.
        """
        lower = topic.lower()
        if any(kw in lower for kw in self._EDUCATIONAL_KW):
            return "educational"
        if any(kw in lower for kw in self._STORY_KW):
            return "story"
        if any(kw in lower for kw in self._VIRAL_KW):
            return "viral"
        return "authority"

    # ── Metric computations ───────────────────────────────────────────────────

    def _calculate_consistency_score(self, daily_counts: List[int]) -> int:
        """
        Produce a 0-100 score representing how regularly posts were published
        across the last 30 days.

        Combines two factors:
        - **Coverage** (70 %): fraction of days that had at least one post.
        - **Uniformity** (30 %): inverse of the coefficient of variation; lower
          variance in daily post count → higher score.
        """
        if not daily_counts:
            return 0

        active_days = len(daily_counts)
        fraction_active = active_days / 30.0

        if len(daily_counts) > 1:
            mean = sum(daily_counts) / len(daily_counts)
            variance = sum((c - mean) ** 2 for c in daily_counts) / len(daily_counts)
            std_dev = variance**0.5
            # Coefficient of variation; lower = more consistent
            cv = std_dev / mean if mean > 0 else 1.0
            volume_consistency = max(0.0, 1.0 - min(cv, 1.0))
        else:
            volume_consistency = 1.0

        raw = (fraction_active * 0.70 + volume_consistency * 0.30) * 100
        return min(100, int(round(raw)))

    def _determine_trend(self, recent_avg: float, previous_avg: float) -> str:
        """
        Compare two engagement averages and return a trend label.

        Returns ``"improving"``, ``"declining"``, or ``"stable"``.
        """
        if previous_avg == 0:
            return "stable"
        change = (recent_avg - previous_avg) / previous_avg
        if change > self._TREND_UP_THRESHOLD:
            return "improving"
        if change < self._TREND_DOWN_THRESHOLD:
            return "declining"
        return "stable"

    # ── Default fallbacks ─────────────────────────────────────────────────────

    def _default_hour_insights(self) -> List[PostingTimeInsight]:
        """
        Return hard-coded schedule recommendations used when real data is
        insufficient.  Sorted best → good → avoid, then by hour within each
        tier.
        """
        _tier_order = {"best": 0, "good": 1, "avoid": 2}

        def _classify_default(hour: int) -> str:
            if hour in self._DEFAULT_BEST_HOURS:
                return "best"
            if hour in self._DEFAULT_GOOD_HOURS:
                return "good"
            if hour in self._DEFAULT_AVOID_HOURS:
                return "avoid"
            return "good"  # daytime hours not otherwise listed → "good"

        insights = [
            PostingTimeInsight(
                hour=h,
                avg_engagement=0.0,
                post_count=0,
                recommendation=_classify_default(h),
            )
            for h in range(24)
        ]
        insights.sort(key=lambda x: (_tier_order[x.recommendation], x.hour))
        return insights

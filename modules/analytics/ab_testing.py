"""
A/B Testing Manager - Run controlled content experiments, extract winning
patterns, and feed learnings back into the content generation pipeline.

Table ownership
---------------
This module is responsible for two tables that it creates on first use:

  ab_tests          – one row per test, variants stored as serialised JSON
  pattern_learnings – append-only knowledge base of observed winning patterns

Both tables live in the same SQLite file as the rest of the application
(path resolved via the injected DatabaseManager.db_path).
"""

import json
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from database.models import DatabaseManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class ABTest:
    test_id: str
    topic: str
    goal: str
    # Each element: {variant_id, content, score, linkedin_post_id, metrics}
    variants: List[Dict[str, Any]]
    status: str  # "active" | "completed" | "cancelled"
    created_at: str  # ISO-8601 UTC datetime string
    winner_variant_id: Optional[str]
    winning_pattern: Optional[str]  # Human-readable description of winner


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ABTestingManager:
    """
    Manages the full lifecycle of A/B content experiments:

    1. Create test with N content variants
    2. Record which variant was published to LinkedIn
    3. Ingest engagement metrics as they arrive
    4. Determine winner and extract the pattern that drove the win
    5. Persist patterns for future content-generation guidance
    """

    # Engagement formula constants (mirrors EngagementEngine for consistency)
    _COMMENT_WEIGHT: int = 3
    _IMPRESSION_DIVISOR: int = 1_000

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager
        self._ensure_tables()

    # =========================================================================
    # Table initialisation
    # =========================================================================

    def _ensure_tables(self) -> None:
        """
        Idempotently create the ``ab_tests`` and ``pattern_learnings`` tables.
        Safe to call on every startup – uses CREATE TABLE IF NOT EXISTS.
        """
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()

                # ── ab_tests ─────────────────────────────────────────────────
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ab_tests (
                        test_id           TEXT    PRIMARY KEY,
                        topic             TEXT    NOT NULL,
                        goal              TEXT    NOT NULL,
                        variants          TEXT    NOT NULL,  -- JSON array
                        status            TEXT    NOT NULL DEFAULT 'active',
                        created_at        TEXT    NOT NULL,
                        winner_variant_id TEXT,
                        winning_pattern   TEXT
                    )
                    """
                )

                # ── pattern_learnings ─────────────────────────────────────────
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pattern_learnings (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern    TEXT    NOT NULL,
                        context    TEXT    NOT NULL,  -- JSON object
                        created_at TEXT    NOT NULL
                    )
                    """
                )

                conn.commit()
                logger.debug("A/B testing tables verified / created.")

        except Exception as exc:
            logger.error(
                "Failed to initialise A/B testing tables: %s", exc, exc_info=True
            )
            raise

    # =========================================================================
    # Core public API
    # =========================================================================

    def create_ab_test(self, topic: str, goal: str, variants: List[str]) -> ABTest:
        """
        Create and persist a new A/B test.

        Args:
            topic:    LinkedIn post topic being tested.
            goal:     Content goal – e.g. ``"educational"``, ``"viral"``,
                      ``"authority"``.
            variants: Ordered list of content strings; one per variant.
                      At least two variants are expected for a meaningful test.

        Returns:
            The newly created :class:`ABTest` instance.

        Raises:
            sqlite3.Error: If the database write fails.
        """
        test_id = str(uuid.uuid4())[:8]
        created_at = datetime.utcnow().isoformat()

        variant_dicts: List[Dict[str, Any]] = [
            {
                "variant_id": f"v{i + 1}",
                "content": content,
                "score": 0.0,
                "linkedin_post_id": None,
                "metrics": {
                    "likes": 0,
                    "comments": 0,
                    "impressions": 0,
                },
            }
            for i, content in enumerate(variants)
        ]

        test = ABTest(
            test_id=test_id,
            topic=topic,
            goal=goal,
            variants=variant_dicts,
            status="active",
            created_at=created_at,
            winner_variant_id=None,
            winning_pattern=None,
        )

        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO ab_tests
                        (test_id, topic, goal, variants, status, created_at,
                         winner_variant_id, winning_pattern)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        test.test_id,
                        test.topic,
                        test.goal,
                        json.dumps(test.variants),
                        test.status,
                        test.created_at,
                        test.winner_variant_id,
                        test.winning_pattern,
                    ),
                )
                conn.commit()

            logger.info(
                "A/B test created | test_id=%s | topic=%s | variants=%d | goal=%s",
                test_id,
                topic,
                len(variants),
                goal,
            )

        except Exception as exc:
            logger.error(
                "Failed to persist A/B test '%s': %s", test_id, exc, exc_info=True
            )
            raise

        return test

    # -------------------------------------------------------------------------

    def record_publish(
        self, test_id: str, variant_id: str, linkedin_post_id: str
    ) -> None:
        """
        Record that a variant has been published to LinkedIn.

        Stores the LinkedIn post ID on the variant so that metrics can later
        be correlated back to the test.

        Args:
            test_id:          The A/B test identifier.
            variant_id:       e.g. ``"v1"``, ``"v2"``.
            linkedin_post_id: The opaque ID returned by the LinkedIn API.
        """
        try:
            raw = self._load_test_raw(test_id)
            if raw is None:
                logger.warning(
                    "record_publish: test_id '%s' not found – skipping.", test_id
                )
                return

            variants = raw["variants"]
            matched = False
            for v in variants:
                if v["variant_id"] == variant_id:
                    v["linkedin_post_id"] = linkedin_post_id
                    matched = True
                    break

            if not matched:
                logger.warning(
                    "record_publish: variant '%s' not found in test '%s' – skipping.",
                    variant_id,
                    test_id,
                )
                return

            self._persist_variants(test_id, variants)
            logger.info(
                "AB test variant published | test_id=%s | variant_id=%s | post_id=%s",
                test_id,
                variant_id,
                linkedin_post_id,
            )

        except Exception as exc:
            logger.error(
                "record_publish failed for test '%s': %s", test_id, exc, exc_info=True
            )

    # -------------------------------------------------------------------------

    def update_variant_metrics(
        self,
        test_id: str,
        variant_id: str,
        likes: int,
        comments: int,
        impressions: int,
    ) -> None:
        """
        Ingest fresh engagement metrics for a variant and recalculate its score.

        Score formula (consistent with ``EngagementEngine.calculate_engagement_score``):

            score = likes + (comments × 3) + impression_efficiency

        where ``impression_efficiency = (likes + comments) / (impressions / 1000)``
        when impressions > 0, else 0.

        Args:
            test_id:     The A/B test identifier.
            variant_id:  Target variant, e.g. ``"v1"``.
            likes:       Current like count from LinkedIn analytics.
            comments:    Current comment count.
            impressions: Current impression count.
        """
        try:
            raw = self._load_test_raw(test_id)
            if raw is None:
                logger.warning(
                    "update_variant_metrics: test_id '%s' not found – skipping.",
                    test_id,
                )
                return

            variants = raw["variants"]
            matched_variant: Optional[Dict[str, Any]] = None

            for v in variants:
                if v["variant_id"] == variant_id:
                    v["metrics"]["likes"] = likes
                    v["metrics"]["comments"] = comments
                    v["metrics"]["impressions"] = impressions
                    v["score"] = self._calculate_score(likes, comments, impressions)
                    matched_variant = v
                    break

            if matched_variant is None:
                logger.warning(
                    "update_variant_metrics: variant '%s' not found in test '%s' – "
                    "skipping.",
                    variant_id,
                    test_id,
                )
                return

            self._persist_variants(test_id, variants)
            logger.info(
                "Metrics updated | test_id=%s | variant_id=%s | "
                "likes=%d | comments=%d | impressions=%d | score=%.2f",
                test_id,
                variant_id,
                likes,
                comments,
                impressions,
                matched_variant["score"],
            )

        except Exception as exc:
            logger.error(
                "update_variant_metrics failed for test '%s': %s",
                test_id,
                exc,
                exc_info=True,
            )

    # -------------------------------------------------------------------------

    def determine_winner(self, test_id: str) -> Optional[str]:
        """
        Evaluate all variants by engagement score, crown the winner, extract
        the winning pattern, and mark the test as ``"completed"``.

        The winning pattern is also forwarded to :meth:`store_pattern_for_learning`
        so it is available to the content-generation pipeline.

        Args:
            test_id: The A/B test identifier.

        Returns:
            The winning ``variant_id`` string, or ``None`` if the test was not
            found or contained no variants.
        """
        try:
            raw = self._load_test_raw(test_id)
            if raw is None:
                logger.warning("determine_winner: test_id '%s' not found.", test_id)
                return None

            variants: List[Dict[str, Any]] = raw["variants"]
            if not variants:
                logger.warning("determine_winner: test '%s' has no variants.", test_id)
                return None

            # Highest score wins; ties broken by variant order (v1 preferred)
            winner = max(variants, key=lambda v: v["score"])
            losers = [v for v in variants if v["variant_id"] != winner["variant_id"]]

            winning_pattern = self._analyse_winning_pattern(winner, losers)

            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE ab_tests
                    SET    winner_variant_id = ?,
                           winning_pattern   = ?,
                           status            = 'completed'
                    WHERE  test_id = ?
                    """,
                    (winner["variant_id"], winning_pattern, test_id),
                )
                conn.commit()

            # Persist to knowledge base for future content-generation guidance
            self.store_pattern_for_learning(
                pattern=winning_pattern,
                context={
                    "test_id": test_id,
                    "topic": raw["topic"],
                    "goal": raw["goal"],
                    "winner_variant_id": winner["variant_id"],
                    "winner_score": winner["score"],
                    "loser_scores": [v["score"] for v in losers],
                },
            )

            logger.info(
                "A/B test completed | test_id=%s | winner=%s | score=%.2f | pattern=%s",
                test_id,
                winner["variant_id"],
                winner["score"],
                winning_pattern,
            )
            return winner["variant_id"]

        except Exception as exc:
            logger.error(
                "determine_winner failed for test '%s': %s",
                test_id,
                exc,
                exc_info=True,
            )
            return None

    # -------------------------------------------------------------------------

    def get_winning_patterns(self) -> List[Dict[str, Any]]:
        """
        Aggregate winning patterns across all completed tests.

        Computes the frequency each pattern has occurred and the average
        improvement margin (winner score minus average loser score).

        Returns:
            List of ``{pattern, frequency, avg_improvement}`` dicts sorted by
            ``frequency`` descending.
        """
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT winner_variant_id, winning_pattern, variants
                    FROM   ab_tests
                    WHERE  status          = 'completed'
                      AND  winning_pattern IS NOT NULL
                    """
                )
                rows = cursor.fetchall()

            # {pattern: {frequency, total_improvement}}
            stats: Dict[str, Dict[str, Any]] = {}

            for winner_id, pattern, variants_json in rows:
                if not pattern:
                    continue

                variants: List[Dict[str, Any]] = json.loads(variants_json)
                winner = next(
                    (v for v in variants if v["variant_id"] == winner_id), None
                )
                if winner is None:
                    continue

                loser_scores = [
                    v["score"] for v in variants if v["variant_id"] != winner_id
                ]
                avg_loser = (
                    sum(loser_scores) / len(loser_scores) if loser_scores else 0.0
                )
                improvement = winner["score"] - avg_loser

                if pattern not in stats:
                    stats[pattern] = {"frequency": 0, "total_improvement": 0.0}
                stats[pattern]["frequency"] += 1
                stats[pattern]["total_improvement"] += improvement

            result = [
                {
                    "pattern": pat,
                    "frequency": data["frequency"],
                    "avg_improvement": round(
                        data["total_improvement"] / data["frequency"], 2
                    ),
                }
                for pat, data in stats.items()
            ]
            result.sort(key=lambda x: x["frequency"], reverse=True)

            logger.info(
                "Winning patterns aggregated: %d unique pattern(s).", len(result)
            )
            return result

        except Exception as exc:
            logger.error("get_winning_patterns failed: %s", exc, exc_info=True)
            return []

    # -------------------------------------------------------------------------

    def get_test_summary(self, test_id: str) -> Dict[str, Any]:
        """
        Return a complete summary of a specific test including all variants,
        their metrics, and the winner (if determined).

        Args:
            test_id: The A/B test identifier.

        Returns:
            A dict with all test fields, or ``{"error": "..."}`` on failure.
        """
        try:
            raw = self._load_test_raw(test_id)
            if raw is None:
                return {"error": f"Test '{test_id}' not found."}

            # Enrich with computed per-variant stats
            variants = raw["variants"]
            scores = [v["score"] for v in variants]
            max_score = max(scores) if scores else 0.0
            min_score = min(scores) if scores else 0.0

            return {
                "test_id": raw["test_id"],
                "topic": raw["topic"],
                "goal": raw["goal"],
                "status": raw["status"],
                "created_at": raw["created_at"],
                "winner_variant_id": raw["winner_variant_id"],
                "winning_pattern": raw["winning_pattern"],
                "variants": variants,
                "summary_stats": {
                    "variant_count": len(variants),
                    "highest_score": round(max_score, 2),
                    "lowest_score": round(min_score, 2),
                    "score_spread": round(max_score - min_score, 2),
                },
            }

        except Exception as exc:
            logger.error(
                "get_test_summary failed for '%s': %s", test_id, exc, exc_info=True
            )
            return {"error": str(exc)}

    # -------------------------------------------------------------------------

    def list_active_tests(self) -> List[ABTest]:
        """
        Retrieve all tests whose status is ``"active"``, ordered newest first.

        Returns:
            List of :class:`ABTest` instances.
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
                rows = cursor.fetchall()

            tests = [self._row_to_abtest(row) for row in rows]
            logger.info("Active A/B tests fetched: %d.", len(tests))
            return tests

        except Exception as exc:
            logger.error("list_active_tests failed: %s", exc, exc_info=True)
            return []

    # -------------------------------------------------------------------------

    def store_pattern_for_learning(self, pattern: str, context: Dict[str, Any]) -> None:
        """
        Append a discovered pattern to the ``pattern_learnings`` knowledge base.

        This table acts as a long-term memory that content-generation components
        can query to bias future posts towards historically successful patterns.

        Args:
            pattern: Short human-readable description of the pattern, e.g.
                     ``"Starts with question | Uses numbered list"``.
            context: Arbitrary JSON-serialisable dict providing provenance
                     (test_id, topic, goal, scores, etc.).
        """
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO pattern_learnings (pattern, context, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (
                        pattern,
                        json.dumps(context, default=str),
                        datetime.utcnow().isoformat(),
                    ),
                )
                conn.commit()
            logger.debug("Pattern stored for learning: '%s'", pattern)

        except Exception as exc:
            logger.error("store_pattern_for_learning failed: %s", exc, exc_info=True)

    # =========================================================================
    # Private helpers
    # =========================================================================

    # ── Engagement scoring ────────────────────────────────────────────────────

    def _calculate_score(self, likes: int, comments: int, impressions: int) -> float:
        """
        Compute engagement score using the same formula as
        ``EngagementEngine.calculate_engagement_score`` for consistency.

            score = likes + (comments × 3) + impression_efficiency

        ``impression_efficiency = (likes + comments) / (impressions / 1000)``
        when *impressions* > 0, else 0.
        """
        base: float = likes + (comments * self._COMMENT_WEIGHT)
        if impressions > 0:
            impression_efficiency = (likes + comments) / (
                impressions / self._IMPRESSION_DIVISOR
            )
            base += impression_efficiency
        return round(base, 2)

    # ── Pattern analysis ──────────────────────────────────────────────────────

    def _analyse_winning_pattern(
        self,
        winner: Dict[str, Any],
        losers: List[Dict[str, Any]],
    ) -> str:
        """
        Heuristically determine what distinguishes the winning content from the
        losing variants.  Returns a ``" | "``-joined string of observations.

        Checks performed (in order):
        1. Relative content length vs losers
        2. Opening with a question
        3. Presence of a numbered list (≥ 3 numbered lines)
        4. Presence of a bullet list (≥ 3 bullet lines)
        5. Rich emoji usage (≥ 3 emoji characters)
        6. Strong call-to-action keywords
        7. Hashtag density (≥ 3 hashtags)
        8. Generic fallback describing word count and score
        """
        w_content: str = winner.get("content", "")
        w_word_count = len(w_content.split())

        l_contents = [v.get("content", "") for v in losers]
        l_word_counts = [len(c.split()) for c in l_contents]
        avg_loser_words = (
            sum(l_word_counts) / len(l_word_counts) if l_word_counts else w_word_count
        )

        observations: List[str] = []

        # 1. Length comparison
        if avg_loser_words > 0:
            if w_word_count < avg_loser_words * 0.80:
                observations.append(f"Shorter content ({w_word_count} words)")
            elif w_word_count > avg_loser_words * 1.20:
                observations.append(f"Longer content ({w_word_count} words)")

        # 2. Starts with a question
        first_line = (w_content.strip().splitlines() or [""])[0].strip()
        if first_line.endswith("?"):
            observations.append("Starts with question")

        # 3. Numbered list (digit followed by . or ) at line start)
        lines = w_content.splitlines()
        numbered_lines = sum(1 for ln in lines if re.match(r"^\s*\d+[.)]\s", ln))
        if numbered_lines >= 3:
            observations.append("Uses numbered list")

        # 4. Bullet / dash list
        _BULLET_CHARS = frozenset({"•", "-", "*", "→", "▶", "✅", "✔"})
        bulleted_lines = sum(
            1 for ln in lines if ln.strip() and ln.strip()[0] in _BULLET_CHARS
        )
        if bulleted_lines >= 3:
            observations.append("Uses bullet points")

        # 5. Emoji density – code-point range covers most emoji blocks
        emoji_count = sum(
            1
            for ch in w_content
            if (
                0x1F300 <= ord(ch) <= 0x1FAFF  # Misc symbols & pictographs
                or 0x2600 <= ord(ch) <= 0x27BF  # Misc symbols
                or 0xFE00 <= ord(ch) <= 0xFE0F  # Variation selectors
            )
        )
        if emoji_count >= 3:
            observations.append(f"Rich use of emojis ({emoji_count})")

        # 6. Call-to-action
        _CTA_KEYWORDS = {
            "comment",
            "share",
            "thoughts",
            "let me know",
            "what do you think",
            "tag",
            "মন্তব্য",
            "শেয়ার",
        }
        lower_content = w_content.lower()
        if any(kw in lower_content for kw in _CTA_KEYWORDS):
            observations.append("Strong call-to-action")

        # 7. Hashtag density
        hashtag_count = w_content.count("#")
        if hashtag_count >= 3:
            observations.append(f"Uses {hashtag_count} hashtags")

        if observations:
            return " | ".join(observations)

        # 8. Generic fallback
        return (
            f"Higher engagement content "
            f"({w_word_count} words, score {winner.get('score', 0):.1f})"
        )

    # ── DB helpers ─────────────────────────────────────────────────────────────

    def _load_test_raw(self, test_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a test row from the DB and deserialise the ``variants`` JSON.

        Returns:
            A plain dict representing the test, or ``None`` if not found.
        """
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

            if row is None:
                return None

            return {
                "test_id": row[0],
                "topic": row[1],
                "goal": row[2],
                "variants": json.loads(row[3]),
                "status": row[4],
                "created_at": row[5],
                "winner_variant_id": row[6],
                "winning_pattern": row[7],
            }

        except Exception as exc:
            logger.error(
                "_load_test_raw failed for test_id='%s': %s",
                test_id,
                exc,
                exc_info=True,
            )
            return None

    def _persist_variants(self, test_id: str, variants: List[Dict[str, Any]]) -> None:
        """
        Write the updated ``variants`` JSON array back to the database.

        Args:
            test_id:  Row to update.
            variants: The mutated variants list.
        """
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE ab_tests SET variants = ? WHERE test_id = ?",
                (json.dumps(variants), test_id),
            )
            conn.commit()

    def _row_to_abtest(self, row: Tuple) -> ABTest:
        """
        Convert a raw DB row tuple (from a SELECT *) into an :class:`ABTest`.

        Column order must match the SELECT in :meth:`list_active_tests`:
        ``test_id, topic, goal, variants, status, created_at,
        winner_variant_id, winning_pattern``.
        """
        return ABTest(
            test_id=row[0],
            topic=row[1],
            goal=row[2],
            variants=json.loads(row[3]),
            status=row[4],
            created_at=row[5],
            winner_variant_id=row[6],
            winning_pattern=row[7],
        )

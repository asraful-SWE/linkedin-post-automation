"""
Intelligent Topic Engine
========================
Enhanced topic intelligence module for the LinkedIn automation system.

Extends the existing ``TopicEngine`` with:
  - 15 predefined topic clusters (+ automatic "general" catch-all)
  - Four selection strategies: balanced, best_cluster, diversity, trending
  - Multi-part content series management (in-memory)
  - ``get_topic_insights_v2()`` – a superset of the existing insight payload

Backward compatibility
----------------------
Pass an existing ``TopicEngine`` instance to the constructor and it will be
used as the final fallback for every selection path that would otherwise fail.
The existing engine is never modified.

Import path (consistent with rest of project):
    from modules.topic.engine import IntelligentTopicEngine
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from database.models import DatabaseManager, TopicPerformance
from services.topic_engine import TopicEngine
from services.topics import ALL_TOPICS, TOPIC_CATEGORIES, get_topic_count

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class TopicCluster:
    """A logical grouping of related topics that share common keywords.

    Attributes
    ----------
    cluster_id:
        Stable snake_case identifier, e.g. ``"ai_tools"``.
    name:
        Human-readable display name, e.g. ``"AI Tools & LLMs"``.
    topics:
        Topic strings from ALL_TOPICS that belong to this cluster.
    keywords:
        The keyword list used to populate *topics* during ``build_clusters``.
    avg_performance:
        Cached average engagement score — updated by
        ``IntelligentTopicEngine.get_cluster_performance()``.
    """

    cluster_id: str
    name: str
    topics: List[str]
    keywords: List[str]
    avg_performance: float = 0.0


@dataclass
class TopicSeries:
    """A multi-part content series with pre-generated topic titles.

    Attributes
    ----------
    series_id:
        URL-safe slug derived from *title*, used as the registry key.
    title:
        Human-readable title, e.g. ``"AI Tools Mastery"``.
    topic_template:
        A ``str.format``-compatible template with a ``{part}`` placeholder,
        e.g. ``"AI Tools Series: {part}তম পর্ব"``.
    parts:
        Fully-rendered topic title for each part of the series.
    current_part:
        Zero-based index of the **next** part to be delivered.
    total_parts:
        Total number of parts in the series.
    is_active:
        ``False`` once all parts have been consumed or the series is
        explicitly deactivated.
    """

    series_id: str
    title: str
    topic_template: str
    parts: List[str]
    current_part: int
    total_parts: int
    is_active: bool


# ---------------------------------------------------------------------------
# IntelligentTopicEngine
# ---------------------------------------------------------------------------


class IntelligentTopicEngine:
    """Intelligent, cluster-aware topic selection engine.

    Works alongside (not as a replacement for) the existing ``TopicEngine``.
    An existing engine instance may be supplied so that this class can
    delegate to it when needed, preserving all historical weight logic.

    Parameters
    ----------
    db_manager:
        Shared ``DatabaseManager`` instance.
    existing_topic_engine:
        Optional existing ``TopicEngine``; used as a fallback only.
    """

    # ------------------------------------------------------------------
    # Cluster definitions — class-level constant
    # ------------------------------------------------------------------

    #: 15 predefined cluster definitions.  Each entry supplies the three
    #: fields required by ``build_clusters``: ``id``, ``name``, ``keywords``.
    TOPIC_CLUSTERS: List[Dict[str, Any]] = [
        {
            "id": "ai_tools",
            "name": "AI Tools & LLMs",
            "keywords": ["AI", "ChatGPT", "Claude", "Copilot", "GPT"],
        },
        {
            "id": "web_dev",
            "name": "Web Development",
            "keywords": ["React", "Next.js", "Frontend", "CSS", "JavaScript", "Web"],
        },
        {
            "id": "backend",
            "name": "Backend Development",
            "keywords": [
                "FastAPI",
                "Django",
                "API",
                "Database",
                "Backend",
                "Python",
            ],
        },
        {
            "id": "career",
            "name": "Career & Jobs",
            "keywords": [
                "Job",
                "Interview",
                "Career",
                "Salary",
                "Resume",
                "LinkedIn",
                "ক্যারিয়ার",
            ],
        },
        {
            "id": "freelancing",
            "name": "Freelancing & Remote",
            "keywords": ["Freelance", "Upwork", "Client", "Remote", "ফ্রিল্যান্স"],
        },
        {
            "id": "learning",
            "name": "Learning & Education",
            "keywords": ["Learn", "Tutorial", "Course", "শেখা", "Study"],
        },
        {
            "id": "productivity",
            "name": "Tools & Productivity",
            "keywords": ["Productivity", "Tools", "Workflow", "Time"],
        },
        {
            "id": "startup",
            "name": "Startups & Business",
            "keywords": ["Startup", "Business", "Entrepreneur", "MVP"],
        },
        {
            "id": "mobile",
            "name": "Mobile Development",
            "keywords": ["Android", "iOS", "React Native", "Flutter", "Mobile"],
        },
        {
            "id": "devops",
            "name": "DevOps & Cloud",
            "keywords": ["Docker", "AWS", "Cloud", "CI/CD", "DevOps"],
        },
        {
            "id": "security",
            "name": "Cybersecurity",
            "keywords": ["Security", "Cybersecurity", "Vulnerability", "Auth"],
        },
        {
            "id": "open_source",
            "name": "Open Source & GitHub",
            "keywords": ["Open Source", "GitHub", "Contribution"],
        },
        {
            "id": "soft_skills",
            "name": "Soft Skills & Leadership",
            "keywords": ["Communication", "Leadership", "Team", "Soft Skills"],
        },
        {
            "id": "bangladesh_tech",
            "name": "Bangladesh Tech Scene",
            "keywords": ["Bangladesh", "বাংলাদেশ", "BD", "Local"],
        },
        {
            "id": "crypto_web3",
            "name": "Blockchain & Web3",
            "keywords": ["Blockchain", "Web3", "Crypto", "NFT"],
        },
    ]

    # ------------------------------------------------------------------
    # Trending topics
    # ------------------------------------------------------------------

    #: Hand-curated list of currently relevant topics used by the
    #: ``"trending"`` selection strategy.
    TRENDING_TOPICS: List[str] = [
        "Claude Code দিয়ে কোড লেখা",
        "Vibe Coding এর বাস্তবতা",
        "AI Agent দিয়ে automation",
        "GPT-4o Mini কি যথেষ্ট?",
        "Cursor IDE vs VS Code",
        "GitHub Copilot experience",
        "Next.js 15 নতুন features",
        "Vercel free tier limits",
        "Supabase vs Firebase",
        "Python vs JavaScript 2025",
    ]

    # ------------------------------------------------------------------
    # Predefined series recommendations
    # ------------------------------------------------------------------

    _RECOMMENDED_SERIES: List[Dict[str, Any]] = [
        {
            "title": "AI Tools Mastery",
            "parts": 7,
            "template": "AI Tools Series: {part}তম পর্ব",
        },
        {
            "title": "Career Growth Series",
            "parts": 5,
            "template": "Career Growth: Part {part}",
        },
        {
            "title": "Python Advanced Series",
            "parts": 8,
            "template": "Python Advanced: Chapter {part}",
        },
        {
            "title": "Freelancing Guide",
            "parts": 6,
            "template": "Freelancing Guide: Step {part}",
        },
    ]

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        db_manager: DatabaseManager,
        existing_topic_engine: Optional[TopicEngine] = None,
    ) -> None:
        self.db: DatabaseManager = db_manager
        self.existing_engine: Optional[TopicEngine] = existing_topic_engine

        # Build topic → cluster mapping at startup from the live topic list.
        self.clusters: Dict[str, TopicCluster] = self.build_clusters(ALL_TOPICS)

        # In-memory series registry: series_id (slug) → TopicSeries
        self._series_registry: Dict[str, TopicSeries] = {}

        logger.info(
            "IntelligentTopicEngine initialised — %d clusters built "
            "(%d total topics, existing_engine=%s)",
            len(self.clusters),
            len(ALL_TOPICS),
            "attached" if existing_topic_engine is not None else "none",
        )

    # ------------------------------------------------------------------
    # Cluster construction
    # ------------------------------------------------------------------

    def build_clusters(self, all_topics: List[str]) -> Dict[str, TopicCluster]:
        """Assign every topic in *all_topics* to its matching cluster(s).

        A topic is assigned to the **first** cluster whose keyword list
        contains a case-insensitive substring match against the topic string.
        Topics that do not match any cluster are collected into a
        ``"general"`` catch-all cluster.

        Parameters
        ----------
        all_topics:
            The flat list of topic strings to cluster (typically
            ``services.topics.ALL_TOPICS``).

        Returns
        -------
        Dict[str, TopicCluster]
            Mapping of ``cluster_id`` → populated ``TopicCluster``.
            Always includes the ``"general"`` cluster.
        """
        clusters: Dict[str, TopicCluster] = {}
        clustered_topics: set = set()

        for defn in self.TOPIC_CLUSTERS:
            cluster_id: str = defn["id"]
            keywords: List[str] = defn["keywords"]
            matching: List[str] = []

            for topic in all_topics:
                topic_lower = topic.lower()
                if any(kw.lower() in topic_lower for kw in keywords):
                    matching.append(topic)
                    clustered_topics.add(topic)

            clusters[cluster_id] = TopicCluster(
                cluster_id=cluster_id,
                name=defn["name"],
                topics=matching,
                keywords=keywords,
            )
            logger.debug("Cluster '%s' built with %d topics", cluster_id, len(matching))

        # Catch-all for topics that didn't match any defined cluster
        general_topics: List[str] = [t for t in all_topics if t not in clustered_topics]
        clusters["general"] = TopicCluster(
            cluster_id="general",
            name="General Tech",
            topics=general_topics,
            keywords=[],
        )

        logger.info(
            "Cluster build complete: %d named clusters + 1 general "
            "(%d unclustered topics out of %d total)",
            len(self.TOPIC_CLUSTERS),
            len(general_topics),
            len(all_topics),
        )
        return clusters

    # ------------------------------------------------------------------
    # Cluster performance
    # ------------------------------------------------------------------

    def get_cluster_performance(self, cluster_id: str) -> float:
        """Return the mean engagement score for topics in *cluster_id*.

        Queries the database for per-topic engagement records and averages
        the scores for topics that belong to the requested cluster.  Returns
        ``0.0`` when the cluster is unknown, empty, or no database records
        exist for any of its topics.

        Parameters
        ----------
        cluster_id:
            One of the cluster IDs defined in ``TOPIC_CLUSTERS`` or
            ``"general"``.

        Returns
        -------
        float
            Average engagement score (≥ 0.0).
        """
        try:
            cluster = self.clusters.get(cluster_id)
            if not cluster or not cluster.topics:
                logger.debug(
                    "get_cluster_performance: cluster '%s' is empty or unknown",
                    cluster_id,
                )
                return 0.0

            performance_data: List[TopicPerformance] = self.db.get_topic_performance()
            perf_map: Dict[str, float] = {
                p.topic: p.avg_engagement for p in performance_data
            }

            scores: List[float] = [
                perf_map[topic] for topic in cluster.topics if topic in perf_map
            ]

            avg = sum(scores) / len(scores) if scores else 0.0

            # Cache onto the dataclass so callers can read it without a DB hit
            cluster.avg_performance = avg

            logger.debug(
                "Cluster '%s' performance: %.4f (from %d / %d topics with data)",
                cluster_id,
                avg,
                len(scores),
                len(cluster.topics),
            )
            return avg

        except Exception as exc:
            logger.error(
                "Failed to compute performance for cluster '%s': %s",
                cluster_id,
                exc,
                exc_info=True,
            )
            return 0.0

    # ------------------------------------------------------------------
    # Single-cluster topic selection
    # ------------------------------------------------------------------

    def select_topic_from_cluster(
        self,
        cluster_id: str,
        avoid_recent_days: int = 3,
    ) -> Optional[str]:
        """Pick one topic from *cluster_id* using performance-weighted sampling.

        Topics used within the last *avoid_recent_days* days are excluded
        from the candidate pool.  If every topic in the cluster was used
        recently the full cluster list is used as a fallback (no hard block).

        Parameters
        ----------
        cluster_id:
            Target cluster identifier.
        avoid_recent_days:
            How many days of post history to treat as "recent".

        Returns
        -------
        Optional[str]
            A topic string, or ``None`` if the cluster is unknown / empty.
        """
        try:
            cluster = self.clusters.get(cluster_id)
            if not cluster or not cluster.topics:
                logger.warning(
                    "select_topic_from_cluster: cluster '%s' is empty or unknown",
                    cluster_id,
                )
                return None

            recent_topics: set = set(self.db.get_recent_topics(days=avoid_recent_days))

            available: List[str] = [t for t in cluster.topics if t not in recent_topics]

            if not available:
                # Soft fallback: allow all cluster topics to prevent a deadlock
                logger.debug(
                    "All %d topics in cluster '%s' used recently; "
                    "opening full pool as fallback",
                    len(cluster.topics),
                    cluster_id,
                )
                available = cluster.topics[:]

            # Build weights from DB performance data
            performance_data: List[TopicPerformance] = self.db.get_topic_performance()
            perf_map: Dict[str, float] = {
                p.topic: p.avg_engagement for p in performance_data
            }

            # Floor at 0.1 so every topic has a non-zero probability of being
            # selected even when it has never been posted.
            weights: List[float] = [max(0.1, perf_map.get(t, 1.0)) for t in available]

            selected: str = random.choices(available, weights=weights, k=1)[0]
            logger.info("Selected topic from cluster '%s': %s", cluster_id, selected)
            return selected

        except Exception as exc:
            logger.error(
                "Failed to select topic from cluster '%s': %s",
                cluster_id,
                exc,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Strategy-based intelligent topic selection
    # ------------------------------------------------------------------

    def select_topic_intelligent(self, strategy: str = "balanced") -> str:
        """Select a topic using one of four intelligence strategies.

        Parameters
        ----------
        strategy:
            One of the following string literals:

            ``"balanced"``
                Cluster is chosen via weighted-random sampling proportional to
                each cluster's average engagement score.  A topic is then
                chosen within that cluster using the same performance weights.
                This is the default and the fallback target for any strategy
                that cannot fulfil its own selection.

            ``"best_cluster"``
                Deterministically picks the cluster with the highest average
                engagement score and selects a topic from it.

            ``"diversity"``
                Favours the cluster whose topics appear least in recent post
                history (last 7 days), promoting content variety.

            ``"trending"``
                Selects from the hand-curated ``TRENDING_TOPICS`` list,
                excluding topics used in the last 3 days.  Falls back to
                ``"balanced"`` when all trending topics are exhausted.

        Returns
        -------
        str
            A topic string guaranteed to be non-empty.

        Notes
        -----
        If cluster data is unavailable the method delegates to the attached
        ``TopicEngine.select_topic()`` (if present) or falls back to a
        uniformly random choice from ``ALL_TOPICS``.
        """
        try:
            if not self.clusters:
                logger.warning(
                    "No clusters available — delegating to fallback topic selection"
                )
                return self._fallback_topic()

            # ----------------------------------------------------------------
            # Trending strategy
            # ----------------------------------------------------------------
            if strategy == "trending":
                try:
                    recent: set = set(self.db.get_recent_topics(days=3))
                    available_trending: List[str] = [
                        t for t in self.TRENDING_TOPICS if t not in recent
                    ]
                    if available_trending:
                        selected = random.choice(available_trending)
                        logger.info("Strategy='trending' — selected: %s", selected)
                        return selected

                    logger.debug(
                        "All trending topics exhausted; falling back to 'balanced'"
                    )
                except Exception as exc:
                    logger.warning(
                        "Trending strategy encountered an error (%s); "
                        "falling back to 'balanced'",
                        exc,
                    )
                # Treat as balanced when trending is exhausted or errored
                strategy = "balanced"

            # ----------------------------------------------------------------
            # Best-cluster strategy
            # ----------------------------------------------------------------
            if strategy == "best_cluster":
                non_empty: Dict[str, TopicCluster] = {
                    cid: c for cid, c in self.clusters.items() if c.topics
                }
                if non_empty:
                    best_id: str = max(
                        non_empty,
                        key=lambda cid: self.get_cluster_performance(cid),
                    )
                    topic = self.select_topic_from_cluster(best_id)
                    if topic:
                        logger.info(
                            "Strategy='best_cluster' — cluster='%s', topic='%s'",
                            best_id,
                            topic,
                        )
                        return topic

                # Fall through to balanced when best cluster yields nothing
                logger.debug(
                    "'best_cluster' yielded no topic; falling back to 'balanced'"
                )

            # ----------------------------------------------------------------
            # Diversity strategy
            # ----------------------------------------------------------------
            elif strategy == "diversity":
                try:
                    recent_topics: set = set(self.db.get_recent_topics(days=7))

                    def _recency_score(cid: str) -> int:
                        """Count of this cluster's topics that appear in recent history."""
                        return sum(
                            1 for t in self.clusters[cid].topics if t in recent_topics
                        )

                    non_empty_ids: List[str] = [
                        cid for cid, c in self.clusters.items() if c.topics
                    ]
                    if non_empty_ids:
                        least_used_id: str = min(non_empty_ids, key=_recency_score)
                        topic = self.select_topic_from_cluster(least_used_id)
                        if topic:
                            logger.info(
                                "Strategy='diversity' — cluster='%s', topic='%s'",
                                least_used_id,
                                topic,
                            )
                            return topic
                except Exception as exc:
                    logger.warning(
                        "'diversity' strategy failed (%s); falling back to 'balanced'",
                        exc,
                    )

            # ----------------------------------------------------------------
            # Balanced strategy (default + shared fallback)
            # ----------------------------------------------------------------
            cluster_ids: List[str] = [
                cid for cid, c in self.clusters.items() if c.topics
            ]
            if not cluster_ids:
                return self._fallback_topic()

            # Weight each cluster by its average engagement score.
            # Floor at 0.1 so clusters without any data still participate.
            cluster_weights: List[float] = [
                max(0.1, self.get_cluster_performance(cid)) for cid in cluster_ids
            ]

            selected_cluster_id: str = random.choices(
                cluster_ids, weights=cluster_weights, k=1
            )[0]

            topic = self.select_topic_from_cluster(selected_cluster_id)
            if topic:
                logger.info(
                    "Strategy='balanced' — cluster='%s', topic='%s'",
                    selected_cluster_id,
                    topic,
                )
                return topic

            # Absolute last resort
            return self._fallback_topic()

        except Exception as exc:
            logger.error(
                "select_topic_intelligent failed with strategy='%s': %s",
                strategy,
                exc,
                exc_info=True,
            )
            return self._fallback_topic()

    # ------------------------------------------------------------------
    # Series management
    # ------------------------------------------------------------------

    def get_or_create_series(
        self,
        title: str,
        total_parts: int,
        topic_template: str,
    ) -> TopicSeries:
        """Return the series for *title*, creating it if it doesn't exist.

        The series is stored in the in-memory ``_series_registry`` under a
        slug derived from *title*.  Calling this method a second time with
        the same title returns the existing series without modification.

        Parts are rendered as::

            f"{title}: Part {i+1} - {topic_template.format(part=i+1)}"

        Parameters
        ----------
        title:
            Human-readable series name, e.g. ``"AI Tools Mastery"``.
        total_parts:
            Total number of parts to generate.
        topic_template:
            Template string with a ``{part}`` placeholder used to generate
            each part's subtitle, e.g. ``"AI Tools Series: {part}তম পর্ব"``.

        Returns
        -------
        TopicSeries
            The (possibly newly-created) series object.
        """
        series_id: str = self._slugify(title)

        if series_id in self._series_registry:
            logger.debug(
                "get_or_create_series: returning existing series '%s'",
                series_id,
            )
            return self._series_registry[series_id]

        # Render every part title up-front
        parts: List[str] = [
            f"{title}: Part {i + 1} - {topic_template.format(part=i + 1)}"
            for i in range(total_parts)
        ]

        series = TopicSeries(
            series_id=series_id,
            title=title,
            topic_template=topic_template,
            parts=parts,
            current_part=0,
            total_parts=total_parts,
            is_active=True,
        )
        self._series_registry[series_id] = series

        logger.info(
            "Created series '%s' [id=%s, parts=%d]",
            title,
            series_id,
            total_parts,
        )
        return series

    def get_next_series_topic(self, series_id: str) -> Optional[str]:
        """Return the next undelivered topic for *series_id* and advance its pointer.

        The series is automatically marked as inactive once all parts have
        been consumed.

        Parameters
        ----------
        series_id:
            The slug identifier returned by ``get_or_create_series``.

        Returns
        -------
        Optional[str]
            The next topic string, or ``None`` if the series is not found,
            is already inactive, or has no remaining parts.
        """
        series = self._series_registry.get(series_id)
        if series is None:
            logger.warning(
                "get_next_series_topic: series '%s' not found in registry",
                series_id,
            )
            return None

        if not series.is_active:
            logger.info(
                "get_next_series_topic: series '%s' is already completed",
                series_id,
            )
            return None

        if series.current_part >= series.total_parts:
            series.is_active = False
            logger.info(
                "get_next_series_topic: series '%s' marked completed "
                "(pointer=%d reached total=%d)",
                series_id,
                series.current_part,
                series.total_parts,
            )
            return None

        topic: str = series.parts[series.current_part]
        series.current_part += 1

        if series.current_part >= series.total_parts:
            series.is_active = False
            logger.info(
                "Series '%s' fully consumed — final part delivered: %s",
                series_id,
                topic,
            )
        else:
            logger.info(
                "Series '%s' advanced to part %d/%d — delivered: %s",
                series_id,
                series.current_part,
                series.total_parts,
                topic,
            )

        return topic

    def get_recommended_series(self) -> List[Dict[str, Any]]:
        """Return the predefined series recommendation catalogue.

        Each entry is a plain dict with keys ``title``, ``parts``, and
        ``template`` — safe to serialise directly as JSON.

        Returns
        -------
        List[Dict[str, Any]]
            Shallow copies of the ``_RECOMMENDED_SERIES`` class constant.
        """
        return [dict(rec) for rec in self._RECOMMENDED_SERIES]

    # ------------------------------------------------------------------
    # Extended insights
    # ------------------------------------------------------------------

    def get_topic_insights_v2(self) -> Dict[str, Any]:
        """Return an extended topic intelligence report.

        Starts from the base ``TopicEngine.get_topic_insights()`` payload
        (when an existing engine is attached) and augments it with:

        ``cluster_performance``
            ``{cluster_id: avg_engagement_score}`` for every cluster.

        ``trending_topics``
            The current ``TRENDING_TOPICS`` list.

        ``active_series``
            Summary dicts for every in-memory series that is still active.
            Each entry includes: ``series_id``, ``title``, ``current_part``,
            ``total_parts``, ``progress_pct``, ``is_active``, ``next_topic``.

        ``recommended_series``
            The predefined series recommendations from
            ``get_recommended_series()``.

        Returns
        -------
        Dict[str, Any]
            Combined insights payload.  Individual sections are gracefully
            omitted (set to ``{}`` or ``[]``) on DB or computation errors.
        """
        insights: Dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Base payload — from the existing engine when available
        # ------------------------------------------------------------------
        if self.existing_engine is not None:
            try:
                base = self.existing_engine.get_topic_insights()
                if isinstance(base, dict):
                    insights.update(base)
                    logger.debug(
                        "get_topic_insights_v2: merged %d base keys from existing engine",
                        len(base),
                    )
            except Exception as exc:
                logger.error(
                    "Failed to fetch base insights from existing engine: %s",
                    exc,
                    exc_info=True,
                )
        else:
            # Build equivalent base stats from scratch
            try:
                performance_data: List[TopicPerformance] = (
                    self.db.get_topic_performance()
                )
                recent_topics: List[str] = self.db.get_recent_topics(days=7)
                used_set: set = {p.topic for p in performance_data}

                sorted_perf = sorted(
                    performance_data,
                    key=lambda p: p.avg_engagement,
                    reverse=True,
                )
                insights = {
                    "total_available_topics": len(ALL_TOPICS),
                    "total_unique_topics": get_topic_count(),
                    "total_categories": len(TOPIC_CATEGORIES),
                    "total_topics_used": len(performance_data),
                    "topics_used_recently": len(recent_topics),
                    "unused_topics_count": sum(
                        1 for t in ALL_TOPICS if t not in used_set
                    ),
                    "best_performing_topics": [
                        {
                            "topic": p.topic,
                            "avg_engagement": round(p.avg_engagement, 2),
                            "total_posts": p.total_posts,
                        }
                        for p in sorted_perf[:5]
                    ],
                }
                logger.debug(
                    "get_topic_insights_v2: built base stats from scratch "
                    "(%d topics with performance data)",
                    len(performance_data),
                )
            except Exception as exc:
                logger.error(
                    "Failed to build base insights from scratch: %s",
                    exc,
                    exc_info=True,
                )

        # ------------------------------------------------------------------
        # Cluster performance
        # ------------------------------------------------------------------
        try:
            cluster_performance: Dict[str, float] = {
                cluster_id: round(self.get_cluster_performance(cluster_id), 4)
                for cluster_id in self.clusters
            }
            insights["cluster_performance"] = cluster_performance
            logger.debug(
                "get_topic_insights_v2: computed performance for %d clusters",
                len(cluster_performance),
            )
        except Exception as exc:
            logger.error(
                "Failed to compute cluster performance map: %s",
                exc,
                exc_info=True,
            )
            insights["cluster_performance"] = {}

        # ------------------------------------------------------------------
        # Trending topics
        # ------------------------------------------------------------------
        insights["trending_topics"] = self.TRENDING_TOPICS[:]

        # ------------------------------------------------------------------
        # Active series summaries
        # ------------------------------------------------------------------
        try:
            active_series_list: List[Dict[str, Any]] = []
            for series in self._series_registry.values():
                if not series.is_active:
                    continue
                progress_pct = (
                    round(series.current_part / series.total_parts * 100, 1)
                    if series.total_parts > 0
                    else 0.0
                )
                next_topic: Optional[str] = (
                    series.parts[series.current_part]
                    if series.current_part < series.total_parts
                    else None
                )
                active_series_list.append(
                    {
                        "series_id": series.series_id,
                        "title": series.title,
                        "current_part": series.current_part,
                        "total_parts": series.total_parts,
                        "progress_pct": progress_pct,
                        "is_active": series.is_active,
                        "next_topic": next_topic,
                    }
                )
            insights["active_series"] = active_series_list
        except Exception as exc:
            logger.error("Failed to build active series list: %s", exc, exc_info=True)
            insights["active_series"] = []

        # ------------------------------------------------------------------
        # Recommended series
        # ------------------------------------------------------------------
        insights["recommended_series"] = self.get_recommended_series()

        return insights

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _slugify(self, text: str) -> str:
        """Convert *text* into a stable, URL-safe slug used as a registry key.

        Steps applied:
        1. Lowercase the ASCII portion (Unicode letters preserved as-is).
        2. Collapse whitespace runs into a single hyphen.
        3. Strip characters that are neither Unicode word-characters nor
           hyphens (removes punctuation, symbols, etc.).
        4. Collapse consecutive hyphens.
        5. Strip leading/trailing hyphens.

        Parameters
        ----------
        text:
            Arbitrary human-readable string, may contain Bengali Unicode.

        Returns
        -------
        str
            URL-safe slug, e.g. ``"ai-tools-mastery"`` or
            ``"ai-tools-series-1"`` for Bengali-heavy titles.
        """
        # Lowercase ASCII; Unicode letters like Bengali are left unchanged
        slug = text.lower()
        # Whitespace → hyphen
        slug = re.sub(r"\s+", "-", slug)
        # Keep Unicode word characters (\w matches [a-zA-Z0-9_] + Unicode
        # letters/digits in Python 3) and hyphens; strip everything else
        slug = re.sub(r"[^\w\-]", "", slug, flags=re.UNICODE)
        # Collapse consecutive hyphens
        slug = re.sub(r"-{2,}", "-", slug)
        return slug.strip("-")

    def _fallback_topic(self) -> str:
        """Return a topic via the existing engine or uniformly at random.

        This is the ultimate safety net: it should never raise an exception.
        """
        if self.existing_engine is not None:
            try:
                return self.existing_engine.select_topic()
            except Exception as exc:
                logger.error(
                    "_fallback_topic: existing engine raised %s; "
                    "falling back to random.choice",
                    exc,
                )
        fallback = random.choice(ALL_TOPICS)
        logger.debug("_fallback_topic: returning random topic '%s'", fallback)
        return fallback

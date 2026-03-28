"""
Image Selector - Scores and selects the best image from a candidate list
based on resolution, aspect ratio, source quality, and metadata richness.
"""

import logging
from typing import Dict, List, Optional

from modules.image.fetcher import ImageFetcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring thresholds
# ---------------------------------------------------------------------------

# Resolution tiers (width, height) → score
_RESOLUTION_TIERS: List[tuple[int, int, float]] = [
    (1920, 1080, 4.0),  # Full HD +
    (1280, 720, 3.0),  # HD
    (1200, 630, 2.0),  # LinkedIn minimum
]
_RESOLUTION_SCORE_FALLBACK: float = 1.0

# Aspect-ratio windows → score
_RATIO_IDEAL_MIN: float = 1.3  # landscape, ideal for LinkedIn feed
_RATIO_IDEAL_MAX: float = 2.0
_RATIO_SQUARE_MIN: float = 1.0  # squarish — acceptable but not optimal
_RATIO_SQUARE_MAX: float = 1.3
_RATIO_SCORE_IDEAL: float = 3.0
_RATIO_SCORE_SQUARE: float = 2.0
_RATIO_SCORE_OTHER: float = 1.0

# Per-source quality scores
_SOURCE_SCORES: dict[str, float] = {
    "unsplash": 2.0,
    "pexels": 1.5,
}
_SOURCE_SCORE_UNKNOWN: float = 1.0

# Bonus for having a non-empty description
_DESCRIPTION_BONUS: float = 1.0


# ---------------------------------------------------------------------------
# ImageSelector
# ---------------------------------------------------------------------------


class ImageSelector:
    """
    Selects the highest-quality image from a pool of candidates fetched by
    an :class:`~modules.image.fetcher.ImageFetcher`.

    Scoring model (max 10 points)
    ─────────────────────────────
    • Resolution score  0 – 4 pts
    • Aspect-ratio score 0 – 3 pts
    • Source score      0 – 2 pts
    • Description bonus   + 1 pt
    """

    def __init__(self, fetcher: ImageFetcher) -> None:
        self.fetcher: ImageFetcher = fetcher

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_image(self, image: Dict) -> float:
        """
        Return a composite quality score in the range [0, 10] for *image*.

        The dict is expected to contain at minimum the keys produced by
        :class:`~modules.image.fetcher.ImageFetcher`:
        ``width``, ``height``, ``source``, ``description``.

        Parameters
        ----------
        image:
            Normalised image dict as returned by the fetcher.

        Returns
        -------
        float
            Composite score; higher is better.
        """
        try:
            width: int = int(image.get("width", 0) or 0)
            height: int = int(image.get("height", 0) or 0)
            source: str = str(image.get("source", "")).lower().strip()
            description: str = str(image.get("description", "") or "").strip()
        except (TypeError, ValueError) as exc:
            logger.debug("score_image: failed to read image fields — %s", exc)
            return 0.0

        # ── Resolution score ──────────────────────────────────────────
        resolution_score: float = _RESOLUTION_SCORE_FALLBACK
        for min_w, min_h, tier_score in _RESOLUTION_TIERS:
            if width >= min_w and height >= min_h:
                resolution_score = tier_score
                break

        # ── Aspect-ratio score ────────────────────────────────────────
        if height > 0:
            ratio: float = width / height
        else:
            ratio = 0.0

        if _RATIO_IDEAL_MIN <= ratio <= _RATIO_IDEAL_MAX:
            ratio_score: float = _RATIO_SCORE_IDEAL
        elif _RATIO_SQUARE_MIN <= ratio < _RATIO_SQUARE_MAX:
            ratio_score = _RATIO_SCORE_SQUARE
        else:
            ratio_score = _RATIO_SCORE_OTHER

        # ── Source score ──────────────────────────────────────────────
        source_score: float = _SOURCE_SCORES.get(source, _SOURCE_SCORE_UNKNOWN)

        # ── Description bonus ─────────────────────────────────────────
        description_bonus: float = _DESCRIPTION_BONUS if description else 0.0

        total: float = resolution_score + ratio_score + source_score + description_bonus

        logger.debug(
            "score_image | source=%s | %dx%d | ratio=%.2f | "
            "res=%.1f aspect=%.1f src=%.1f desc=%.1f → total=%.2f",
            source,
            width,
            height,
            ratio,
            resolution_score,
            ratio_score,
            source_score,
            description_bonus,
            total,
        )

        return round(total, 4)

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def select_best(self, images: List[Dict]) -> Optional[Dict]:
        """
        Score every image in *images*, attach the score to the dict, sort
        descending and return the winner.

        Returns ``None`` when *images* is empty.

        Parameters
        ----------
        images:
            List of normalised image dicts (may be mutated in-place by
            adding the ``"score"`` key).

        Returns
        -------
        Optional[Dict]
            The highest-scored image dict, or ``None``.
        """
        if not images:
            logger.debug("select_best: received empty image list")
            return None

        for img in images:
            img["score"] = self.score_image(img)

        ranked = sorted(images, key=lambda x: x.get("score", 0.0), reverse=True)

        best = ranked[0]
        logger.debug(
            "select_best: winner | source=%s | score=%.2f | url=%s",
            best.get("source"),
            best.get("score"),
            best.get("url"),
        )
        return best

    # ------------------------------------------------------------------
    # Topic-level entry points
    # ------------------------------------------------------------------

    def get_image_for_topic(
        self,
        topic: str,
        count: int = 5,
    ) -> Optional[Dict]:
        """
        Fetch candidate images for *topic* and return the single best one.

        The returned dict is guaranteed to contain the following keys:
        ``url``, ``thumb_url``, ``width``, ``height``, ``source``,
        ``score``, ``description``, ``photographer``.

        Parameters
        ----------
        topic:
            Free-text topic string (English or Bengali).
        count:
            How many candidate images to fetch before scoring.

        Returns
        -------
        Optional[Dict]
            Best image dict, or ``None`` on failure / no results.
        """
        try:
            images: List[Dict] = self.fetcher.fetch_images(topic, count=count)
        except Exception as exc:
            logger.error(
                "get_image_for_topic: fetch failed | topic='%s' | error=%s",
                topic,
                exc,
                exc_info=True,
            )
            return None

        if not images:
            logger.warning(
                "get_image_for_topic: no images returned | topic='%s'", topic
            )
            return None

        best: Optional[Dict] = self.select_best(images)
        if best is None:
            return None

        # Ensure the canonical key set is always present
        result: Dict = {
            "url": best.get("url", ""),
            "thumb_url": best.get("thumb_url", ""),
            "width": best.get("width", 0),
            "height": best.get("height", 0),
            "source": best.get("source", ""),
            "score": best.get("score", 0.0),
            "description": best.get("description", ""),
            "photographer": best.get("photographer", ""),
        }

        logger.info(
            "Selected image | topic=%s | source=%s | score=%.2f | url=%s",
            topic,
            result["source"],
            result["score"],
            result["url"],
        )

        return result

    # ------------------------------------------------------------------

    def get_top_images(
        self,
        topic: str,
        count: int = 3,
    ) -> List[Dict]:
        """
        Fetch candidates for *topic*, score them all and return the top
        *count* images sorted by score (highest first).

        Parameters
        ----------
        topic:
            Free-text topic string.
        count:
            Number of top images to return.  The fetcher is asked for a
            larger pool (``count * 2``) to increase scoring diversity.

        Returns
        -------
        List[Dict]
            Up to *count* scored image dicts; empty on failure.
        """
        fetch_count: int = max(count * 2, 10)  # wider pool → better selection

        try:
            images: List[Dict] = self.fetcher.fetch_images(topic, count=fetch_count)
        except Exception as exc:
            logger.error(
                "get_top_images: fetch failed | topic='%s' | error=%s",
                topic,
                exc,
                exc_info=True,
            )
            return []

        if not images:
            logger.warning("get_top_images: no images returned | topic='%s'", topic)
            return []

        for img in images:
            img["score"] = self.score_image(img)

        ranked: List[Dict] = sorted(
            images,
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )

        top: List[Dict] = ranked[:count]

        logger.info(
            "Top images selected | topic='%s' | requested=%d | pool=%d | returned=%d",
            topic,
            count,
            len(images),
            len(top),
        )

        return top

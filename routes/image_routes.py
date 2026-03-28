"""
Image Routes - Endpoints for fetching and selecting LinkedIn post images.
Uses Unsplash and Pexels APIs. No AI-generated images.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/images", tags=["images"])


def _make_fetcher_and_selector():
    """Lazy-initialize ImageFetcher and ImageSelector."""
    from modules.image.fetcher import ImageFetcher
    from modules.image.selector import ImageSelector

    fetcher = ImageFetcher()
    selector = ImageSelector(fetcher=fetcher)
    return fetcher, selector


@router.get("/best", response_model=Dict[str, Any])
async def get_best_image(
    topic: str = Query(..., description="Topic to find image for"),
    count: int = Query(default=5, ge=1, le=20, description="Candidate pool size"),
):
    """
    Fetch candidate images from Unsplash/Pexels for *topic* and return the
    single best image based on resolution, aspect ratio and source quality.
    """
    try:
        _, selector = _make_fetcher_and_selector()
        result = selector.get_image_for_topic(topic=topic, count=count)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"No suitable image found for topic '{topic}'. "
                "Ensure UNSPLASH_ACCESS_KEY or PEXELS_API_KEY is configured.",
            )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"get_best_image | topic={topic} | error={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image fetch failed: {exc}")


@router.get("/top", response_model=List[Dict[str, Any]])
async def get_top_images(
    topic: str = Query(..., description="Topic to find images for"),
    count: int = Query(default=3, ge=1, le=10, description="Number of top images"),
):
    """
    Return the top *count* images for *topic* ranked by quality score.
    """
    try:
        _, selector = _make_fetcher_and_selector()
        results = selector.get_top_images(topic=topic, count=count)
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No suitable images found for topic '{topic}'.",
            )
        return results
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"get_top_images | topic={topic} | error={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image fetch failed: {exc}")


@router.get("/keywords", response_model=Dict[str, Any])
async def extract_keywords(
    topic: str = Query(..., description="Topic to extract keywords from"),
):
    """Extract searchable English keywords from a topic string (handles Bengali->English)."""
    try:
        fetcher, _ = _make_fetcher_and_selector()
        keywords = fetcher.extract_keywords(topic)
        return {"topic": topic, "keywords": keywords}
    except Exception as exc:
        logger.error(f"extract_keywords | topic={topic} | error={exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

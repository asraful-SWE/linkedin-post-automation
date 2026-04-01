"""
Hacker News Collector - Fetches top stories from Hacker News API.

Uses the official Firebase API:
- https://hacker-news.firebaseio.com/v0/topstories.json
- https://hacker-news.firebaseio.com/v0/item/{id}.json
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from .base_collector import BaseCollector, CollectedItem, CollectorError

logger = logging.getLogger(__name__)


class HackerNewsCollector(BaseCollector):
    """
    Collector for Hacker News using the Firebase API.
    
    Fetches top stories and normalizes them to CollectedItem format.
    """
    
    BASE_URL = "https://hacker-news.firebaseio.com/v0"
    
    def __init__(self, max_items: int = 30, **kwargs):
        super().__init__(max_items=max_items, **kwargs)
        # Concurrent requests limit to avoid overwhelming the API
        self._semaphore = asyncio.Semaphore(10)
    
    def get_source_name(self) -> str:
        return "hackernews"
    
    async def fetch_items(self) -> List[CollectedItem]:
        """Fetch top stories from Hacker News."""
        # Get top story IDs
        top_ids = await self._fetch_json(f"{self.BASE_URL}/topstories.json")
        
        if not top_ids:
            logger.warning("No top stories returned from Hacker News")
            return []
        
        # Limit to max_items
        story_ids = top_ids[:self.max_items]
        
        # Fetch story details concurrently
        tasks = [self._fetch_story(story_id) for story_id in story_ids]
        stories = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors and None results
        items = []
        for story in stories:
            if isinstance(story, Exception):
                logger.debug(f"Failed to fetch story: {story}")
                continue
            if story is not None:
                items.append(story)
        
        logger.info(f"Fetched {len(items)} stories from Hacker News")
        return items
    
    async def _fetch_story(self, story_id: int) -> Optional[CollectedItem]:
        """Fetch a single story by ID."""
        async with self._semaphore:
            try:
                story = await self._fetch_json(f"{self.BASE_URL}/item/{story_id}.json")
                
                if not story:
                    return None
                
                # Skip non-story items (comments, polls, etc.)
                if story.get("type") != "story":
                    return None
                
                # Skip items without URL (Ask HN, Show HN text posts)
                url = story.get("url")
                if not url:
                    # For text posts, use HN URL
                    url = f"https://news.ycombinator.com/item?id={story_id}"
                
                return CollectedItem(
                    title=story.get("title", ""),
                    url=url,
                    source=self.get_source_name(),
                    published_at=self._parse_datetime(story.get("time")),
                    external_score=story.get("score", 0),
                    author=story.get("by"),
                )
                
            except Exception as e:
                logger.debug(f"Error fetching story {story_id}: {e}")
                return None

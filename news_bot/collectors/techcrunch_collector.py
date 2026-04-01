"""
TechCrunch Collector - Fetches articles from TechCrunch RSS feed.

Uses the official RSS feed:
- https://techcrunch.com/feed/
"""

import logging
from datetime import datetime
from typing import List, Optional

import feedparser

from .base_collector import BaseCollector, CollectedItem

logger = logging.getLogger(__name__)


class TechCrunchCollector(BaseCollector):
    """
    Collector for TechCrunch articles using RSS feed.
    
    Fetches latest tech news from TechCrunch.
    """
    
    RSS_FEED = "https://techcrunch.com/feed/"
    
    def __init__(self, max_items: int = 30, **kwargs):
        super().__init__(max_items=max_items, **kwargs)
    
    def get_source_name(self) -> str:
        return "techcrunch"
    
    async def fetch_items(self) -> List[CollectedItem]:
        """Fetch articles from TechCrunch RSS feed."""
        try:
            response_text = await self._fetch_text(self.RSS_FEED)
            feed = feedparser.parse(response_text)
            
            items = []
            for entry in feed.entries[:self.max_items]:
                item = self._parse_entry(entry)
                if item:
                    items.append(item)
            
            logger.info(f"Fetched {len(items)} articles from TechCrunch")
            return items
            
        except Exception as e:
            logger.error(f"Error fetching TechCrunch feed: {e}")
            return []
    
    def _parse_entry(self, entry: dict) -> Optional[CollectedItem]:
        """Parse RSS entry to CollectedItem."""
        try:
            title = entry.get("title", "")
            link = entry.get("link", "")
            
            if not link:
                return None
            
            # Parse published date
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6])
            
            # Extract author
            author = None
            if hasattr(entry, "author"):
                author = entry.author
            elif hasattr(entry, "authors") and entry.authors:
                author = entry.authors[0].get("name")
            
            # Extract tags from categories
            tags = []
            if hasattr(entry, "tags"):
                tags = [tag.get("term", "") for tag in entry.tags if tag.get("term")]
            
            return CollectedItem(
                title=title,
                url=link,
                source=self.get_source_name(),
                published_at=published,
                external_score=0,
                author=author,
                tags=tags,
            )
            
        except Exception as e:
            logger.debug(f"Error parsing TechCrunch entry: {e}")
            return None

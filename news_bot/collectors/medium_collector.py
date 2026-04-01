"""
Medium Collector - Fetches articles from Medium RSS feeds.

Uses RSS feeds from Medium publications:
- https://medium.com/feed/tag/programming
- https://medium.com/feed/tag/technology
"""

import logging
from datetime import datetime
from typing import List, Optional

import feedparser

from .base_collector import BaseCollector, CollectedItem

logger = logging.getLogger(__name__)


class MediumCollector(BaseCollector):
    """
    Collector for Medium articles using RSS feeds.
    
    Fetches from popular tech tags and publications.
    """
    
    # RSS feed URLs for tech content
    RSS_FEEDS = [
        "https://medium.com/feed/tag/programming",
        "https://medium.com/feed/tag/software-engineering",
        "https://medium.com/feed/tag/technology",
        "https://medium.com/feed/tag/web-development",
        "https://medium.com/feed/tag/artificial-intelligence",
        "https://medium.com/feed/tag/machine-learning",
        "https://medium.com/feed/tag/javascript",
        "https://medium.com/feed/tag/python",
        "https://medium.com/feed/tag/devops",
        "https://medium.com/feed/tag/software-development",
    ]
    
    def __init__(
        self,
        feeds: Optional[List[str]] = None,
        max_items: int = 50,
        **kwargs
    ):
        super().__init__(max_items=max_items, **kwargs)
        self.feeds = feeds or self.RSS_FEEDS
    
    def get_source_name(self) -> str:
        return "medium"
    
    async def fetch_items(self) -> List[CollectedItem]:
        """Fetch articles from Medium RSS feeds."""
        all_items = []
        
        for feed_url in self.feeds:
            try:
                items = await self._fetch_feed(feed_url)
                all_items.extend(items)
            except Exception as e:
                logger.warning(f"Failed to fetch Medium feed {feed_url}: {e}")
                continue
        
        # Deduplicate by URL
        seen_urls = set()
        unique_items = []
        for item in all_items:
            # Clean Medium URL (remove query params)
            clean_url = item.url.split("?")[0]
            if clean_url not in seen_urls:
                seen_urls.add(clean_url)
                item.url = clean_url
                unique_items.append(item)
        
        # Sort by published date (newest first)
        unique_items.sort(
            key=lambda x: x.published_at or datetime.min,
            reverse=True
        )
        
        return unique_items[:self.max_items]
    
    async def _fetch_feed(self, feed_url: str) -> List[CollectedItem]:
        """Fetch and parse a single RSS feed."""
        try:
            response_text = await self._fetch_text(feed_url)
            feed = feedparser.parse(response_text)
            
            items = []
            for entry in feed.entries:
                item = self._parse_entry(entry, feed_url)
                if item:
                    items.append(item)
            
            logger.debug(f"Fetched {len(items)} articles from {feed_url}")
            return items
            
        except Exception as e:
            logger.warning(f"Error fetching Medium feed {feed_url}: {e}")
            return []
    
    def _parse_entry(self, entry: dict, feed_url: str) -> Optional[CollectedItem]:
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
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published = datetime(*entry.updated_parsed[:6])
            
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
            
            # Add tag from feed URL
            if "/tag/" in feed_url:
                tag = feed_url.split("/tag/")[-1].replace("-", " ")
                tags.append(tag)
            
            return CollectedItem(
                title=title,
                url=link,
                source=self.get_source_name(),
                published_at=published,
                external_score=0,  # Medium RSS doesn't include claps
                author=author,
                tags=tags,
            )
            
        except Exception as e:
            logger.debug(f"Error parsing Medium entry: {e}")
            return None

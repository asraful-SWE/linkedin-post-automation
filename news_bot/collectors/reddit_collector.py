"""
Reddit Collector - Fetches posts from tech subreddits.

Supports both:
- Reddit API (with credentials)
- RSS fallback (no auth required)

Subreddits:
- r/programming
- r/webdev
- r/machinelearning
- r/devops
- r/Python
- r/javascript
- r/artificial
"""

import logging
import os
from datetime import datetime
from typing import List, Optional
from urllib.parse import urljoin

import feedparser

from .base_collector import BaseCollector, CollectedItem

logger = logging.getLogger(__name__)


class RedditCollector(BaseCollector):
    """
    Collector for Reddit tech subreddits.
    
    Uses RSS feeds for simplicity and to avoid API rate limits.
    Falls back to JSON API if needed.
    """
    
    DEFAULT_SUBREDDITS = [
        "programming",
        "webdev",
        "machinelearning",
        "devops",
        "Python",
        "javascript",
        "artificial",
        "coding",
        "learnprogramming",
    ]
    
    RSS_BASE = "https://www.reddit.com/r/{subreddit}/top/.rss?t=day&limit=25"
    JSON_BASE = "https://www.reddit.com/r/{subreddit}/top.json?t=day&limit=25"
    
    def __init__(
        self,
        subreddits: Optional[List[str]] = None,
        max_items: int = 50,
        **kwargs
    ):
        super().__init__(max_items=max_items, **kwargs)
        self.subreddits = subreddits or self.DEFAULT_SUBREDDITS
    
    def get_source_name(self) -> str:
        return "reddit"
    
    async def fetch_items(self) -> List[CollectedItem]:
        """Fetch posts from configured subreddits."""
        all_items = []
        
        for subreddit in self.subreddits:
            try:
                items = await self._fetch_subreddit(subreddit)
                all_items.extend(items)
            except Exception as e:
                logger.warning(f"Failed to fetch r/{subreddit}: {e}")
                continue
        
        # Sort by score and limit
        all_items.sort(key=lambda x: x.external_score, reverse=True)
        return all_items[:self.max_items]
    
    async def _fetch_subreddit(self, subreddit: str) -> List[CollectedItem]:
        """Fetch posts from a single subreddit using RSS."""
        url = self.RSS_BASE.format(subreddit=subreddit)
        
        try:
            # Fetch RSS feed
            response_text = await self._fetch_text(url)
            feed = feedparser.parse(response_text)
            
            items = []
            for entry in feed.entries:
                item = self._parse_rss_entry(entry, subreddit)
                if item:
                    items.append(item)
            
            logger.debug(f"Fetched {len(items)} posts from r/{subreddit}")
            return items
            
        except Exception as e:
            logger.debug(f"RSS failed for r/{subreddit}, trying JSON: {e}")
            return await self._fetch_subreddit_json(subreddit)
    
    async def _fetch_subreddit_json(self, subreddit: str) -> List[CollectedItem]:
        """Fallback: fetch from Reddit JSON API."""
        url = self.JSON_BASE.format(subreddit=subreddit)
        
        try:
            data = await self._fetch_json(url)
            posts = data.get("data", {}).get("children", [])
            
            items = []
            for post in posts:
                item = self._parse_json_post(post.get("data", {}), subreddit)
                if item:
                    items.append(item)
            
            return items
            
        except Exception as e:
            logger.warning(f"JSON fallback also failed for r/{subreddit}: {e}")
            return []
    
    def _parse_rss_entry(self, entry: dict, subreddit: str) -> Optional[CollectedItem]:
        """Parse RSS entry to CollectedItem."""
        try:
            title = entry.get("title", "")
            link = entry.get("link", "")
            
            # Skip self posts without useful content
            if not link or "reddit.com" not in link:
                return None
            
            # Get actual link (not Reddit discussion)
            # RSS entries link to Reddit, but we want the actual article
            # For now, use Reddit URL; article extractor will handle it
            
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6])
            
            return CollectedItem(
                title=title,
                url=link,
                source=self.get_source_name(),
                published_at=published,
                external_score=0,  # RSS doesn't include score
                author=entry.get("author"),
                tags=[f"r/{subreddit}"],
            )
            
        except Exception as e:
            logger.debug(f"Error parsing RSS entry: {e}")
            return None
    
    def _parse_json_post(self, post: dict, subreddit: str) -> Optional[CollectedItem]:
        """Parse JSON post data to CollectedItem."""
        try:
            # Skip stickied posts
            if post.get("stickied"):
                return None
            
            title = post.get("title", "")
            url = post.get("url", "")
            
            # For self posts, use Reddit URL
            if post.get("is_self"):
                url = f"https://reddit.com{post.get('permalink', '')}"
            
            if not url:
                return None
            
            return CollectedItem(
                title=title,
                url=url,
                source=self.get_source_name(),
                published_at=self._parse_datetime(post.get("created_utc")),
                external_score=post.get("score", 0),
                author=post.get("author"),
                tags=[f"r/{subreddit}"],
            )
            
        except Exception as e:
            logger.debug(f"Error parsing JSON post: {e}")
            return None

"""
Dev.to Collector - Fetches articles from Dev.to API.

Uses the public API:
- https://dev.to/api/articles
- https://dev.to/api/articles?tag=webdev&top=1
"""

import logging
from datetime import datetime
from typing import List, Optional

from .base_collector import BaseCollector, CollectedItem

logger = logging.getLogger(__name__)


class DevtoCollector(BaseCollector):
    """
    Collector for Dev.to articles using the public API.
    
    Fetches latest and trending articles from various tags.
    """
    
    BASE_URL = "https://dev.to/api/articles"
    
    # Tech-focused tags to fetch
    DEFAULT_TAGS = [
        "webdev",
        "programming",
        "javascript",
        "python",
        "devops",
        "machinelearning",
        "beginners",
        "career",
        "productivity",
        "tutorial",
    ]
    
    def __init__(
        self,
        tags: Optional[List[str]] = None,
        max_items: int = 50,
        **kwargs
    ):
        super().__init__(max_items=max_items, **kwargs)
        self.tags = tags or self.DEFAULT_TAGS
    
    def get_source_name(self) -> str:
        return "devto"
    
    async def fetch_items(self) -> List[CollectedItem]:
        """Fetch articles from Dev.to."""
        all_items = []
        
        # Fetch trending articles (no tag filter)
        trending = await self._fetch_articles(top=1, per_page=25)
        all_items.extend(trending)
        
        # Fetch from specific tags
        for tag in self.tags[:5]:  # Limit tag queries to avoid rate limits
            try:
                items = await self._fetch_articles(tag=tag, per_page=10)
                all_items.extend(items)
            except Exception as e:
                logger.warning(f"Failed to fetch Dev.to tag '{tag}': {e}")
                continue
        
        # Deduplicate by URL
        seen_urls = set()
        unique_items = []
        for item in all_items:
            if item.url not in seen_urls:
                seen_urls.add(item.url)
                unique_items.append(item)
        
        # Sort by external_score (reactions) and limit
        unique_items.sort(key=lambda x: x.external_score, reverse=True)
        return unique_items[:self.max_items]
    
    async def _fetch_articles(
        self,
        tag: Optional[str] = None,
        top: Optional[int] = None,
        per_page: int = 25,
    ) -> List[CollectedItem]:
        """Fetch articles with optional filters."""
        params = {"per_page": per_page}
        
        if tag:
            params["tag"] = tag
        if top:
            params["top"] = top
        
        try:
            articles = await self._fetch_json(self.BASE_URL, params=params)
            
            items = []
            for article in articles:
                item = self._parse_article(article)
                if item:
                    items.append(item)
            
            return items
            
        except Exception as e:
            logger.warning(f"Error fetching Dev.to articles: {e}")
            return []
    
    def _parse_article(self, article: dict) -> Optional[CollectedItem]:
        """Parse Dev.to article to CollectedItem."""
        try:
            url = article.get("url") or article.get("canonical_url")
            if not url:
                return None
            
            # Parse published date
            published_at = None
            if article.get("published_at"):
                published_at = self._parse_datetime(article["published_at"])
            elif article.get("published_timestamp"):
                published_at = self._parse_datetime(article["published_timestamp"])
            
            # Collect tags
            tags = article.get("tag_list", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",")]
            
            return CollectedItem(
                title=article.get("title", ""),
                url=url,
                source=self.get_source_name(),
                published_at=published_at,
                external_score=article.get("positive_reactions_count", 0),
                author=article.get("user", {}).get("username"),
                tags=tags,
            )
            
        except Exception as e:
            logger.debug(f"Error parsing Dev.to article: {e}")
            return None

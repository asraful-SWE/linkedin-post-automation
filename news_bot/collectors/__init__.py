"""
Collectors module - Data source collectors for tech news and articles.

Collectors:
- HackerNewsCollector: Hacker News API
- RedditCollector: Reddit tech subreddits
- DevtoCollector: Dev.to API
- MediumCollector: Medium RSS feeds
- TechCrunchCollector: TechCrunch RSS feed
"""

from .base_collector import BaseCollector, CollectedItem
from .hackernews_collector import HackerNewsCollector
from .reddit_collector import RedditCollector
from .devto_collector import DevtoCollector
from .medium_collector import MediumCollector
from .techcrunch_collector import TechCrunchCollector

__all__ = [
    "BaseCollector",
    "CollectedItem",
    "HackerNewsCollector",
    "RedditCollector",
    "DevtoCollector",
    "MediumCollector",
    "TechCrunchCollector",
]

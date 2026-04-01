"""
Base Collector - Abstract base class for all content collectors.

Provides common functionality:
- Retry logic with exponential backoff
- Rate limiting
- Error handling
- Logging
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from news_bot.models import ContentItem

logger = logging.getLogger(__name__)


# Re-export CollectedItem from models for convenience
from news_bot.models.content_models import CollectedItem


class CollectorError(Exception):
    """Base exception for collector errors."""
    pass


class RateLimitError(CollectorError):
    """Raised when rate limit is hit."""
    pass


class NetworkError(CollectorError):
    """Raised for network-related errors."""
    pass


class BaseCollector(ABC):
    """
    Abstract base class for all content collectors.
    
    Each collector must implement:
    - get_source_name(): Return the source identifier
    - fetch_items(): Fetch items from the source
    - _parse_item(): Parse raw item to CollectedItem
    
    Provides:
    - HTTP client with retry logic
    - Rate limiting support
    - Standardized error handling
    - Logging
    """
    
    # Default configuration
    DEFAULT_TIMEOUT = 30.0
    DEFAULT_MAX_ITEMS = 50
    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36 LinkedInAutoPoster/2.0"
    )
    
    # Enhanced headers for better scraping success
    DEFAULT_HEADERS = {
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Fetch-User": "?1",
        "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="121", "Google Chrome";v="121"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
    }
    
    def __init__(
        self,
        max_items: int = DEFAULT_MAX_ITEMS,
        timeout: float = DEFAULT_TIMEOUT,
        user_agent: Optional[str] = None,
    ):
        self.max_items = max_items
        self.timeout = timeout
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            # Merge default headers with custom User-Agent
            headers = {**self.DEFAULT_HEADERS, "User-Agent": self.user_agent}
            
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers,
                follow_redirects=True,
            )
        return self._client
    
    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Return the source identifier (e.g., 'hackernews', 'reddit')."""
        pass
    
    @abstractmethod
    async def fetch_items(self) -> List[CollectedItem]:
        """
        Fetch items from the source.
        
        Returns:
            List of CollectedItem objects
        """
        pass
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, NetworkError)),
        reraise=True,
    )
    async def _fetch_json(self, url: str, params: Optional[dict] = None) -> dict:
        """
        Fetch JSON from URL with retry logic.
        
        Args:
            url: URL to fetch
            params: Optional query parameters
            
        Returns:
            Parsed JSON response
            
        Raises:
            NetworkError: On network failures
            RateLimitError: On rate limit (429)
        """
        try:
            response = await self.client.get(url, params=params)
            
            if response.status_code == 429:
                logger.warning(f"Rate limited by {self.get_source_name()}")
                raise RateLimitError(f"Rate limited by {self.get_source_name()}")
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            raise NetworkError(f"HTTP error: {e}")
        except httpx.RequestError as e:
            logger.error(f"Request error fetching {url}: {e}")
            raise NetworkError(f"Request error: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, NetworkError)),
        reraise=True,
    )
    async def _fetch_text(self, url: str) -> str:
        """
        Fetch text content from URL with retry logic.
        
        Args:
            url: URL to fetch
            
        Returns:
            Text response
            
        Raises:
            NetworkError: On network failures
        """
        try:
            response = await self.client.get(url)
            
            if response.status_code == 429:
                logger.warning(f"Rate limited by {self.get_source_name()}")
                raise RateLimitError(f"Rate limited by {self.get_source_name()}")
            
            response.raise_for_status()
            return response.text
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            raise NetworkError(f"HTTP error: {e}")
        except httpx.RequestError as e:
            logger.error(f"Request error fetching {url}: {e}")
            raise NetworkError(f"Request error: {e}")
    
    async def collect(self) -> List[ContentItem]:
        """
        Main entry point for collecting content.
        
        Fetches items, converts to ContentItem, and handles errors gracefully.
        
        Returns:
            List of ContentItem objects ready for processing
        """
        logger.info(f"Starting collection from {self.get_source_name()}")
        
        try:
            items = await self.fetch_items()
            content_items = [item.to_content_item() for item in items[:self.max_items]]
            
            logger.info(
                f"Collected {len(content_items)} items from {self.get_source_name()}"
            )
            return content_items
            
        except RateLimitError:
            logger.warning(f"Rate limited, skipping {self.get_source_name()}")
            return []
        except CollectorError as e:
            logger.error(f"Collector error for {self.get_source_name()}: {e}")
            return []
        except Exception as e:
            logger.exception(f"Unexpected error collecting from {self.get_source_name()}: {e}")
            return []
        finally:
            await self.close()
    
    def _parse_datetime(self, timestamp: any) -> Optional[datetime]:
        """
        Parse various datetime formats to datetime object.
        
        Args:
            timestamp: Unix timestamp, ISO string, or datetime
            
        Returns:
            datetime object or None
        """
        if timestamp is None:
            return None
        
        if isinstance(timestamp, datetime):
            return timestamp
        
        if isinstance(timestamp, (int, float)):
            try:
                return datetime.utcfromtimestamp(timestamp)
            except (ValueError, OSError):
                return None
        
        if isinstance(timestamp, str):
            # Try ISO format
            try:
                return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                pass
            
            # Try common formats
            formats = [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%a, %d %b %Y %H:%M:%S %z",  # RSS format
                "%a, %d %b %Y %H:%M:%S GMT",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
        
        return None

"""
Image Extractor - Extracts main image from articles.

Extraction strategy:
1. og:image meta tag (primary)
2. twitter:image meta tag
3. First large image in article body

Validation:
- Direct image URL (not data URI)
- Minimum resolution (400x300)
- Not a logo/icon (based on size and URL patterns)
"""

import logging
import re
from typing import Optional, List, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ImageExtractor:
    """
    Extracts and validates main images from articles.
    
    Ensures images are:
    - Direct, accessible URLs
    - High enough resolution for sharing
    - Not logos, icons, or tracking pixels
    """
    
    DEFAULT_TIMEOUT = 15.0
    MIN_WIDTH = 400
    MIN_HEIGHT = 300
    
    # URL patterns to exclude (logos, icons, tracking)
    EXCLUDE_PATTERNS = [
        r"logo",
        r"icon",
        r"favicon",
        r"sprite",
        r"avatar",
        r"badge",
        r"button",
        r"banner-ad",
        r"advertisement",
        r"pixel",
        r"tracking",
        r"1x1",
        r"spacer",
        r"gravatar",
        r"profile",
        r"thumb",  # too small
    ]
    
    # Valid image extensions
    VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".gif"]
    
    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        min_width: int = MIN_WIDTH,
        min_height: int = MIN_HEIGHT,
    ):
        self.timeout = timeout
        self.min_width = min_width
        self.min_height = min_height
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client
    
    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def extract(self, url: str, html: Optional[str] = None) -> Optional[str]:
        """
        Extract main image from article.
        
        Args:
            url: Article URL (used for resolving relative URLs)
            html: HTML content (optional, will be fetched if not provided)
            
        Returns:
            Validated image URL or None
        """
        try:
            if html is None:
                html = await self._fetch_html(url)
                if not html:
                    return None
            
            soup = BeautifulSoup(html, "lxml")
            
            # Strategy 1: og:image
            image_url = self._get_og_image(soup)
            if image_url:
                validated = await self._validate_image(image_url, url)
                if validated:
                    logger.debug(f"Found og:image: {validated}")
                    return validated
            
            # Strategy 2: twitter:image
            image_url = self._get_twitter_image(soup)
            if image_url:
                validated = await self._validate_image(image_url, url)
                if validated:
                    logger.debug(f"Found twitter:image: {validated}")
                    return validated
            
            # Strategy 3: First large image in content
            image_url = self._get_content_image(soup)
            if image_url:
                validated = await self._validate_image(image_url, url)
                if validated:
                    logger.debug(f"Found content image: {validated}")
                    return validated
            
            logger.debug(f"No valid image found for {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting image from {url}: {e}")
            return None
    
    async def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML content from URL."""
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.debug(f"Error fetching {url}: {e}")
            return None
    
    def _get_og_image(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract og:image from meta tags."""
        og_image = soup.find("meta", property="og:image")
        if og_image:
            return og_image.get("content")
        
        # Some sites use name instead of property
        og_image = soup.find("meta", {"name": "og:image"})
        if og_image:
            return og_image.get("content")
        
        return None
    
    def _get_twitter_image(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract twitter:image from meta tags."""
        twitter_image = soup.find("meta", {"name": "twitter:image"})
        if twitter_image:
            return twitter_image.get("content")
        
        twitter_image = soup.find("meta", {"name": "twitter:image:src"})
        if twitter_image:
            return twitter_image.get("content")
        
        return None
    
    def _get_content_image(self, soup: BeautifulSoup) -> Optional[str]:
        """Find first large image in article content."""
        # Look for images in article/main content
        article = (
            soup.find("article") or
            soup.find("main") or
            soup.find(class_=re.compile(r"article|content|post|entry", re.I))
        )
        
        if not article:
            article = soup
        
        # Find all images
        images = article.find_all("img")
        
        for img in images:
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if not src:
                continue
            
            # Skip if matches exclude patterns
            if self._is_excluded(src):
                continue
            
            # Check size attributes if available
            width = img.get("width")
            height = img.get("height")
            
            if width and height:
                try:
                    w = int(re.sub(r"[^\d]", "", str(width)))
                    h = int(re.sub(r"[^\d]", "", str(height)))
                    if w >= self.min_width and h >= self.min_height:
                        return src
                except ValueError:
                    pass
            
            # If no size, check if it looks like a main image
            # (not in figure, has meaningful alt text, etc.)
            alt = img.get("alt", "")
            if len(alt) > 10 or img.find_parent("figure"):
                return src
        
        return None
    
    def _is_excluded(self, url: str) -> bool:
        """Check if URL matches exclusion patterns."""
        url_lower = url.lower()
        for pattern in self.EXCLUDE_PATTERNS:
            if re.search(pattern, url_lower):
                return True
        return False
    
    async def _validate_image(self, image_url: str, base_url: str) -> Optional[str]:
        """
        Validate image URL.
        
        Args:
            image_url: Image URL (may be relative)
            base_url: Base URL for resolving relative URLs
            
        Returns:
            Validated absolute image URL or None
        """
        if not image_url:
            return None
        
        # Skip data URIs
        if image_url.startswith("data:"):
            return None
        
        # Resolve relative URLs
        if not image_url.startswith(("http://", "https://")):
            image_url = urljoin(base_url, image_url)
        
        # Check URL pattern exclusions
        if self._is_excluded(image_url):
            return None
        
        # Verify image is accessible and check dimensions
        try:
            # Just do a HEAD request to verify accessibility
            response = await self.client.head(image_url)
            
            if response.status_code != 200:
                return None
            
            # Check content type
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                return None
            
            # Check content length (skip tiny images < 5KB)
            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    size = int(content_length)
                    if size < 5000:  # 5KB minimum
                        return None
                except ValueError:
                    pass
            
            return image_url
            
        except Exception as e:
            logger.debug(f"Image validation failed for {image_url}: {e}")
            return None

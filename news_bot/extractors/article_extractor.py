"""
Article Extractor - Extracts full article text from URLs.

Uses multiple extraction strategies with robust retry and fallback mechanisms:
1. newspaper3k (primary)
2. readability-lxml (fallback)
3. BeautifulSoup (last resort)
4. RSS summary fallback
"""

import asyncio
import logging
import re
from typing import Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# Try importing extraction libraries
try:
    from newspaper import Article, ArticleException
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    logger.info("newspaper3k not available - using fallback extraction")

try:
    from readability import Document
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    logger.info("readability-lxml not available - using basic extraction")


class ExtractionError(Exception):
    """Raised when article extraction fails."""
    pass


class ArticleExtractor:
    """
    Extracts full article content from URLs with robust retry and fallback mechanisms.
    
    Features:
    - Automatic retry with exponential backoff
    - Multiple extraction strategies
    - Intelligent header spoofing
    - Domain-specific handling
    - Graceful error handling
    """
    
    DEFAULT_TIMEOUT = 20.0
    MAX_RETRIES = 3
    BATCH_SIZE = 10
    
    # Enhanced headers for better success rates
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Cache-Control": "no-cache",
    }
    
    # Domains with special handling
    MEDIUM_DOMAINS = {"medium.com", "towardsdatascience.com"}
    SKIP_DOMAINS = {"youtube.com", "twitter.com", "github.com"}
    
    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        enable_fallback: bool = True,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_fallback = enable_fallback
        self._client: Optional[httpx.AsyncClient] = None
        
        # Track statistics
        self.stats = {
            "total_requests": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "403_errors": 0,
            "timeouts": 0,
            "fallback_used": 0,
        }
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with optimized settings."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                headers=self.DEFAULT_HEADERS.copy(),
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            )
        return self._client
    
    async def close(self):
        """Close HTTP client and log statistics."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        
        # Log final statistics
        if self.stats["total_requests"] > 0:
            success_rate = (self.stats["successful_extractions"] / self.stats["total_requests"]) * 100
            logger.info(f"Article extraction stats: {self.stats} (success rate: {success_rate:.1f}%)")
    
    async def extract(self, url: str, fallback_summary: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract article content from URL with robust retry and fallback mechanisms.
        
        Args:
            url: Article URL to extract
            fallback_summary: RSS summary to use if extraction fails
            
        Returns:
            Dictionary with extraction results and metadata
        """
        self.stats["total_requests"] += 1
        
        result = {
            "full_text": None,
            "title": None,
            "author": None,
            "published_at": None,
            "reading_time": 0,
            "word_count": 0,
            "success": False,
            "error": None,
            "extraction_method": None,
        }
        
        # Check if domain should be skipped
        domain = self._get_domain(url)
        if domain in self.SKIP_DOMAINS:
            result["error"] = f"Skipping {domain} - not suitable for article extraction"
            if fallback_summary:
                result["full_text"] = fallback_summary
                result["success"] = True
                result["extraction_method"] = "rss_fallback"
                self.stats["fallback_used"] += 1
            return result
        
        try:
            # Fetch HTML with retry
            html = await self._fetch_html_with_retry(url)
            if not html:
                return self._handle_extraction_failure(result, "Failed to fetch URL", fallback_summary)
            
            # Try extraction methods in order of preference
            for method_name, method in [
                ("newspaper3k", self._extract_with_newspaper),
                ("readability", self._extract_with_readability),
                ("beautifulsoup", self._extract_with_beautifulsoup),
            ]:
                try:
                    if method_name == "newspaper3k" and not NEWSPAPER_AVAILABLE:
                        continue
                    if method_name == "readability" and not READABILITY_AVAILABLE:
                        continue
                    
                    if method_name == "newspaper3k":
                        extracted = method(url, html)
                    else:
                        extracted = method(html)
                    
                    if extracted.get("success") and extracted.get("full_text"):
                        extracted["extraction_method"] = method_name
                        self.stats["successful_extractions"] += 1
                        logger.debug(f"Successfully extracted article from {url} using {method_name}")
                        return extracted
                        
                except Exception as e:
                    logger.debug(f"Extraction method {method_name} failed for {url}: {e}")
                    continue
            
            # All extraction methods failed
            return self._handle_extraction_failure(
                result, "All extraction methods failed", fallback_summary
            )
            
        except Exception as e:
            return self._handle_extraction_failure(result, str(e), fallback_summary)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        reraise=True,
    )
    async def _fetch_html_with_retry(self, url: str) -> Optional[str]:
        """Fetch HTML content with retry logic."""
        try:
            # Add domain-specific headers if needed
            headers = self.DEFAULT_HEADERS.copy()
            domain = self._get_domain(url)
            
            if domain in self.MEDIUM_DOMAINS:
                # Medium-specific headers
                headers.update({
                    "Referer": "https://www.google.com/",
                    "Sec-Fetch-User": "?1",
                })
            
            response = await self.client.get(url, headers=headers)
            
            # Handle specific status codes
            if response.status_code == 403:
                self.stats["403_errors"] += 1
                logger.warning(f"403 Forbidden for {url} - may need different headers or cookies")
                return None
            elif response.status_code == 429:
                logger.warning(f"Rate limited for {url} - backing off")
                await asyncio.sleep(2)
                return None
            elif response.status_code >= 400:
                logger.warning(f"HTTP {response.status_code} for {url}")
                return None
            
            response.raise_for_status()
            return response.text
            
        except httpx.TimeoutException:
            self.stats["timeouts"] += 1
            logger.warning(f"Timeout fetching {url}")
            raise
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                self.stats["403_errors"] += 1
                logger.warning(f"403 Forbidden for {url}")
            else:
                logger.warning(f"HTTP error {e.response.status_code} for {url}")
            return None
        except Exception as e:
            logger.debug(f"Error fetching {url}: {e}")
            raise
    
    def _extract_with_newspaper(self, url: str, html: str) -> dict:
        """Extract using newspaper3k library."""
        result = {
            "full_text": None,
            "title": None,
            "author": None,
            "published_at": None,
            "reading_time": 0,
            "word_count": 0,
            "success": False,
            "error": None,
        }
        
        try:
            article = Article(url)
            article.set_html(html)
            article.parse()
            
            text = article.text
            if not text or len(text) < 100:
                result["error"] = "Article text too short"
                return result
            
            result["full_text"] = self._clean_text(text)
            result["title"] = article.title
            result["author"] = ", ".join(article.authors) if article.authors else None
            result["published_at"] = article.publish_date
            result["word_count"] = len(text.split())
            result["reading_time"] = max(1, result["word_count"] // 200)
            result["success"] = True
            
            return result
            
        except Exception as e:
            logger.debug(f"newspaper3k extraction failed: {e}")
            result["error"] = str(e)
            return result
    
    def _extract_with_readability(self, html: str) -> dict:
        """Extract using readability-lxml library."""
        result = {
            "full_text": None,
            "title": None,
            "author": None,
            "published_at": None,
            "reading_time": 0,
            "word_count": 0,
            "success": False,
            "error": None,
        }
        
        try:
            doc = Document(html)
            
            # Get clean HTML
            clean_html = doc.summary()
            soup = BeautifulSoup(clean_html, "lxml")
            
            # Extract text
            text = soup.get_text(separator="\n", strip=True)
            if not text or len(text) < 100:
                result["error"] = "Article text too short"
                return result
            
            result["full_text"] = self._clean_text(text)
            result["title"] = doc.title()
            result["word_count"] = len(text.split())
            result["reading_time"] = max(1, result["word_count"] // 200)
            result["success"] = True
            
            return result
            
        except Exception as e:
            logger.debug(f"readability extraction failed: {e}")
            result["error"] = str(e)
            return result
    
    def _extract_with_beautifulsoup(self, html: str) -> dict:
        """Extract using BeautifulSoup (fallback)."""
        result = {
            "full_text": None,
            "title": None,
            "author": None,
            "published_at": None,
            "reading_time": 0,
            "word_count": 0,
            "success": False,
            "error": None,
        }
        
        try:
            soup = BeautifulSoup(html, "lxml")
            
            # Remove unwanted elements
            for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside", "iframe", "noscript"]):
                tag.decompose()
            
            # Try to find article content
            article = (
                soup.find("article") or
                soup.find("main") or
                soup.find(class_=re.compile(r"article|content|post|entry", re.I)) or
                soup.find(id=re.compile(r"article|content|post|entry", re.I))
            )
            
            if article:
                text = article.get_text(separator="\n", strip=True)
            else:
                # Fall back to body
                body = soup.find("body")
                text = body.get_text(separator="\n", strip=True) if body else ""
            
            if not text or len(text) < 100:
                result["error"] = "Could not extract meaningful content"
                return result
            
            # Get title
            title = None
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)
            
            # Get author from meta tags
            author = None
            author_meta = soup.find("meta", {"name": "author"})
            if author_meta:
                author = author_meta.get("content")
            
            result["full_text"] = self._clean_text(text)
            result["title"] = title
            result["author"] = author
            result["word_count"] = len(text.split())
            result["reading_time"] = max(1, result["word_count"] // 200)
            result["success"] = True
            
            return result
            
        except Exception as e:
            logger.debug(f"BeautifulSoup extraction failed: {e}")
            result["error"] = str(e)
            return result
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        
        # Remove common junk
        junk_patterns = [
            r"Share on (Facebook|Twitter|LinkedIn|WhatsApp)\.?",
            r"Follow us on \w+",
            r"Subscribe to our newsletter",
            r"Read more:.*$",
            r"Related articles?:.*$",
            r"Advertisement",
            r"Sponsored content",
        ]
        
        for pattern in junk_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Strip and return
        return text.strip()
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            return urlparse(url).netloc.lower()
        except:
            return ""
    
    def _handle_extraction_failure(self, result: Dict[str, Any], error: str, fallback_summary: Optional[str]) -> Dict[str, Any]:
        """Handle extraction failure with optional fallback."""
        self.stats["failed_extractions"] += 1
        result["error"] = error
        
        if fallback_summary and self.enable_fallback:
            result["full_text"] = fallback_summary
            result["success"] = True
            result["extraction_method"] = "rss_fallback"
            self.stats["fallback_used"] += 1
            logger.debug(f"Using RSS fallback for failed extraction: {error}")
        else:
            logger.debug(f"Extraction failed: {error}")
        
        return result

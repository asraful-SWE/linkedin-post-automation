"""
Image Fetcher - Retrieves relevant images from Unsplash and Pexels APIs
based on extracted keywords from a given topic string.
"""

import logging
import os
import re
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"
PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"

STOP_WORDS: set[str] = {
    # English
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    # Bengali
    "নিয়ে",
    "এবং",
    "বা",
    "থেকে",
    "হলো",
    "করা",
}

# Basic Bengali → English keyword mapping (extend as needed)
BENGALI_TO_ENGLISH: dict[str, str] = {
    "প্রোগ্রামিং": "programming",
    "প্রযুক্তি": "technology",
    "ডেভেলপমেন্ট": "development",
    "সফটওয়্যার": "software",
    "ইন্টারনেট": "internet",
    "ডেটা": "data",
    "কৃত্রিম": "artificial",
    "বুদ্ধিমত্তা": "intelligence",
    "নেটওয়ার্ক": "network",
    "সাইবার": "cyber",
    "ক্লাউড": "cloud",
    "অ্যাপ": "app",
    "ওয়েব": "web",
    "মোবাইল": "mobile",
    "ডিজাইন": "design",
    "উদ্যোক্তা": "entrepreneur",
    "ব্যবসা": "business",
    "শিক্ষা": "education",
    "স্বাস্থ্য": "health",
    "বিজ্ঞান": "science",
}

FALLBACK_KEYWORDS: List[str] = ["technology", "programming"]


# ---------------------------------------------------------------------------
# ImageFetcher
# ---------------------------------------------------------------------------


class ImageFetcher:
    """
    Fetches relevant, high-resolution images from Unsplash and Pexels based
    on keywords derived from a topic string.

    LinkedIn recommended minimum dimensions: 1200 × 630 px.
    """

    def __init__(
        self,
        min_width: int = 1200,
        min_height: int = 630,
        timeout: int = 15,
    ) -> None:
        self.unsplash_key: Optional[str] = os.getenv("UNSPLASH_ACCESS_KEY") or None
        self.pexels_key: Optional[str] = os.getenv("PEXELS_API_KEY") or None
        self.min_width = min_width
        self.min_height = min_height
        self.timeout = timeout

        self._log_key_status()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_key_status(self) -> None:
        if not self.unsplash_key:
            logger.warning(
                "UNSPLASH_ACCESS_KEY not configured — Unsplash fetching disabled"
            )
        else:
            logger.debug("Unsplash API key loaded successfully")

        if not self.pexels_key:
            logger.warning("PEXELS_API_KEY not configured — Pexels fetching disabled")
        else:
            logger.debug("Pexels API key loaded successfully")

    @staticmethod
    def _translate_bengali(word: str) -> str:
        """Return the English equivalent for a Bengali word, or the word itself."""
        return BENGALI_TO_ENGLISH.get(word, word)

    @staticmethod
    def _is_ascii_word(word: str) -> bool:
        """Return True if the word consists solely of ASCII characters."""
        try:
            word.encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_keywords(self, topic: str) -> List[str]:
        """
        Extract 3-5 meaningful keywords from a topic string.

        Strategy
        --------
        1. Split on whitespace, commas and forward-slashes.
        2. Lowercase and strip punctuation from every token.
        3. Translate known Bengali tokens to English.
        4. Remove stop-words and tokens shorter than 3 characters.
        5. Return the first 5 surviving tokens.
        6. Fall back to ``["technology", "programming"]`` when nothing survives.
        """
        # Tokenise on spaces, commas, slashes
        raw_tokens: List[str] = re.split(r"[\s,/]+", topic.strip())

        keywords: List[str] = []
        for token in raw_tokens:
            # Strip leading/trailing punctuation (preserve internal hyphens)
            cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", token, flags=re.UNICODE)
            if not cleaned:
                continue

            lower = cleaned.lower()

            # Translate Bengali → English when possible
            translated = self._translate_bengali(lower)

            # After translation, strip non-ASCII residue that can't be used
            # as a search query keyword
            if not self._is_ascii_word(translated):
                # Try the original token's translation once more with the
                # non-lowered form (some mappings are case-sensitive)
                translated = self._translate_bengali(cleaned)
                if not self._is_ascii_word(translated):
                    # Skip untranslatable non-ASCII tokens
                    continue

            # Apply stop-word filter and minimum length gate
            if translated in STOP_WORDS or lower in STOP_WORDS:
                continue
            if len(translated) < 3:
                continue

            keywords.append(translated)
            if len(keywords) == 5:
                break

        if not keywords:
            logger.debug(
                "No usable keywords extracted from topic '%s'; using fallback %s",
                topic,
                FALLBACK_KEYWORDS,
            )
            return list(FALLBACK_KEYWORDS)

        logger.debug("Extracted keywords from topic '%s': %s", topic, keywords)
        return keywords

    # ------------------------------------------------------------------

    def fetch_from_unsplash(
        self,
        keywords: List[str],
        count: int = 5,
    ) -> List[Dict]:
        """
        Search Unsplash for landscape images matching *keywords*.

        Returns a list of normalised image dicts.  Returns an empty list
        when the API key is missing or the request fails.
        """
        if not self.unsplash_key:
            logger.debug("Unsplash fetch skipped — no API key")
            return []

        query = " ".join(keywords)
        params = {
            "query": query,
            "per_page": count,
            "orientation": "landscape",
            "client_id": self.unsplash_key,
        }

        try:
            logger.debug("Fetching Unsplash images | query='%s' count=%d", query, count)
            response = requests.get(
                UNSPLASH_SEARCH_URL,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            logger.warning(
                "Unsplash API HTTP error | status=%s | detail=%s",
                exc.response.status_code if exc.response is not None else "N/A",
                exc,
            )
            return []
        except requests.exceptions.Timeout:
            logger.warning("Unsplash API request timed out after %ds", self.timeout)
            return []
        except requests.exceptions.RequestException as exc:
            logger.warning("Unsplash API request failed: %s", exc)
            return []

        try:
            data = response.json()
            results = data.get("results", [])
        except ValueError as exc:
            logger.warning("Failed to parse Unsplash JSON response: %s", exc)
            return []

        images: List[Dict] = []
        for item in results:
            try:
                urls = item.get("urls", {})
                user = item.get("user", {})
                description = (
                    item.get("description") or item.get("alt_description") or ""
                )
                images.append(
                    {
                        "url": urls.get("regular", ""),
                        "thumb_url": urls.get("small", ""),
                        "width": item.get("width", 0),
                        "height": item.get("height", 0),
                        "description": description,
                        "photographer": user.get("name", "Unknown"),
                        "source": "unsplash",
                        "relevance_score": 1.0,
                    }
                )
            except (KeyError, TypeError) as exc:
                logger.debug("Skipping malformed Unsplash result item: %s", exc)
                continue

        logger.info(
            "Unsplash fetch complete | query='%s' | returned=%d/%d",
            query,
            len(images),
            count,
        )
        return images

    # ------------------------------------------------------------------

    def fetch_from_pexels(
        self,
        keywords: List[str],
        count: int = 5,
    ) -> List[Dict]:
        """
        Search Pexels for landscape images matching *keywords*.

        Returns a list of normalised image dicts.  Returns an empty list
        when the API key is missing or the request fails.
        """
        if not self.pexels_key:
            logger.debug("Pexels fetch skipped — no API key")
            return []

        query = " ".join(keywords)
        headers = {"Authorization": self.pexels_key}
        params = {
            "query": query,
            "per_page": count,
            "orientation": "landscape",
        }

        try:
            logger.debug("Fetching Pexels images | query='%s' count=%d", query, count)
            response = requests.get(
                PEXELS_SEARCH_URL,
                headers=headers,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            logger.warning(
                "Pexels API HTTP error | status=%s | detail=%s",
                exc.response.status_code if exc.response is not None else "N/A",
                exc,
            )
            return []
        except requests.exceptions.Timeout:
            logger.warning("Pexels API request timed out after %ds", self.timeout)
            return []
        except requests.exceptions.RequestException as exc:
            logger.warning("Pexels API request failed: %s", exc)
            return []

        try:
            data = response.json()
            results = data.get("photos", [])
        except ValueError as exc:
            logger.warning("Failed to parse Pexels JSON response: %s", exc)
            return []

        images: List[Dict] = []
        for item in results:
            try:
                src = item.get("src", {})
                images.append(
                    {
                        "url": src.get("large", ""),
                        "thumb_url": src.get("medium", ""),
                        "width": item.get("width", 0),
                        "height": item.get("height", 0),
                        "description": item.get("alt", ""),
                        "photographer": item.get("photographer", "Unknown"),
                        "source": "pexels",
                        "relevance_score": 1.0,
                    }
                )
            except (KeyError, TypeError) as exc:
                logger.debug("Skipping malformed Pexels result item: %s", exc)
                continue

        logger.info(
            "Pexels fetch complete | query='%s' | returned=%d/%d",
            query,
            len(images),
            count,
        )
        return images

    # ------------------------------------------------------------------

    def fetch_images(self, topic: str, count: int = 5) -> List[Dict]:
        """
        Fetch up to *count* images relevant to *topic* from all configured
        providers (Unsplash and/or Pexels).

        Results from both providers are combined; only the first *count*
        entries are returned so the caller always receives a predictable
        upper bound.
        """
        keywords = self.extract_keywords(topic)
        logger.info(
            "Fetching images | topic='%s' | keywords=%s | count=%d",
            topic,
            keywords,
            count,
        )

        unsplash_images = self.fetch_from_unsplash(keywords, count=count)
        pexels_images = self.fetch_from_pexels(keywords, count=count)

        # Interleave results so neither source dominates the top-N slice
        combined: List[Dict] = []
        for pair in zip(unsplash_images, pexels_images):
            combined.extend(pair)

        # Append any remaining items from the longer list
        longer = (
            unsplash_images
            if len(unsplash_images) > len(pexels_images)
            else pexels_images
        )
        combined.extend(longer[min(len(unsplash_images), len(pexels_images)) :])

        selected = combined[:count]

        logger.info(
            "Image fetch summary | topic='%s' | unsplash=%d | pexels=%d | combined=%d | returning=%d",
            topic,
            len(unsplash_images),
            len(pexels_images),
            len(combined),
            len(selected),
        )
        return selected

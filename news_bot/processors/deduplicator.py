"""
Deduplicator - Removes duplicate content items.

Deduplication strategies:
1. URL hash (exact match)
2. Title similarity (fuzzy match for near-duplicates)
"""

import hashlib
import logging
import re
import sqlite3
from typing import List, Optional, Set

from news_bot.models import ContentItem

logger = logging.getLogger(__name__)


class Deduplicator:
    """
    Removes duplicate content items before processing.
    
    Uses URL hash for exact duplicates and title similarity
    for near-duplicates (same article from different sources).
    """
    
    # Minimum similarity ratio to consider titles as duplicates
    TITLE_SIMILARITY_THRESHOLD = 0.85
    
    def __init__(self, db_path: str):
        """
        Initialize deduplicator with database path.
        
        Args:
            db_path: Path to SQLite database containing content_items
        """
        self.db_path = db_path
    
    def filter_duplicates(self, items: List[ContentItem]) -> List[ContentItem]:
        """
        Filter out duplicate items.
        
        Removes items that already exist in the database
        or are duplicates within the current batch.
        
        Args:
            items: List of content items to filter
            
        Returns:
            List of unique items
        """
        if not items:
            return []
        
        # Get existing URL hashes from database
        existing_hashes = self._get_existing_hashes()
        
        # Track seen items in this batch
        seen_hashes: Set[str] = set()
        seen_titles: List[str] = []
        
        unique_items = []
        duplicate_count = 0
        
        for item in items:
            # Check URL hash
            if item.url_hash in existing_hashes:
                duplicate_count += 1
                continue
            
            if item.url_hash in seen_hashes:
                duplicate_count += 1
                continue
            
            # Check title similarity
            if self._is_similar_title(item.title, seen_titles):
                duplicate_count += 1
                continue
            
            # Mark as seen
            seen_hashes.add(item.url_hash)
            seen_titles.append(item.title)
            unique_items.append(item)
        
        logger.info(
            f"Deduplication: {len(items)} items → {len(unique_items)} unique "
            f"({duplicate_count} duplicates removed)"
        )
        
        return unique_items
    
    def _get_existing_hashes(self) -> Set[str]:
        """Get all existing URL hashes from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT url_hash FROM content_items")
                return {row[0] for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return set()
        except Exception as e:
            logger.error(f"Error getting existing hashes: {e}")
            return set()
    
    def _is_similar_title(
        self,
        title: str,
        seen_titles: List[str],
    ) -> bool:
        """
        Check if title is similar to any previously seen title.
        
        Uses a simple token-based similarity check.
        """
        if not title or not seen_titles:
            return False
        
        # Normalize title
        normalized = self._normalize_title(title)
        if not normalized:
            return False
        
        for seen_title in seen_titles:
            seen_normalized = self._normalize_title(seen_title)
            if not seen_normalized:
                continue
            
            similarity = self._calculate_similarity(normalized, seen_normalized)
            if similarity >= self.TITLE_SIMILARITY_THRESHOLD:
                logger.debug(f"Similar titles found: '{title[:50]}...' ~ '{seen_title[:50]}...'")
                return True
        
        return False
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        # Lowercase
        title = title.lower()
        
        # Remove special characters
        title = re.sub(r"[^\w\s]", " ", title)
        
        # Remove extra whitespace
        title = " ".join(title.split())
        
        return title
    
    def _calculate_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate similarity ratio between two titles.
        
        Uses Jaccard similarity of word tokens.
        """
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def is_duplicate(self, item: ContentItem) -> bool:
        """
        Check if a single item is a duplicate.
        
        Args:
            item: Content item to check
            
        Returns:
            True if duplicate, False otherwise
        """
        existing_hashes = self._get_existing_hashes()
        return item.url_hash in existing_hashes

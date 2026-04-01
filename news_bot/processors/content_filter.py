"""
Content Filter - Filters content based on relevance and recency.

Filtering criteria:
1. Age: Content from last 24-48 hours
2. Keywords: Relevant to tech/software engineering
3. Quality: Minimum content length, not spam
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Set

from news_bot.models import ContentItem

logger = logging.getLogger(__name__)


class ContentFilter:
    """
    Filters content items based on relevance and quality.
    
    Keeps only high-quality, relevant content for processing.
    """
    
    # Relevant keywords for tech content
    RELEVANT_KEYWORDS = {
        # Programming & Development
        "programming", "coding", "developer", "software", "engineering",
        "code", "algorithm", "data structure", "debug", "testing",
        
        # Web Development
        "web", "frontend", "backend", "fullstack", "javascript", "typescript",
        "react", "vue", "angular", "node", "css", "html", "api", "rest", "graphql",
        
        # Languages
        "python", "java", "golang", "rust", "c++", "kotlin", "swift",
        "php", "ruby", "scala", "elixir",
        
        # AI & ML
        "ai", "artificial intelligence", "machine learning", "ml", "deep learning",
        "neural network", "gpt", "llm", "chatgpt", "openai", "claude", "gemini",
        "nlp", "computer vision", "tensorflow", "pytorch",
        
        # Cloud & DevOps
        "cloud", "aws", "azure", "gcp", "kubernetes", "docker", "devops",
        "ci/cd", "terraform", "ansible", "microservices", "serverless",
        
        # Database
        "database", "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
        
        # Security
        "security", "cybersecurity", "encryption", "authentication", "oauth",
        
        # Career & Industry
        "tech", "startup", "career", "interview", "hiring", "job",
        "productivity", "remote work", "engineering culture",
        
        # Trends
        "blockchain", "crypto", "web3", "metaverse", "iot", "edge computing",
    }
    
    # Spam/low-quality indicators
    SPAM_PATTERNS = [
        r"buy\s+now",
        r"limited\s+time\s+offer",
        r"click\s+here",
        r"free\s+download",
        r"\$\d+\s+off",
        r"discount\s+code",
        r"affiliate",
        r"sponsored\s+content",
    ]
    
    def __init__(
        self,
        max_age_hours: int = 48,
        min_relevance_score: float = 0.1,
        min_text_length: int = 100,
    ):
        """
        Initialize content filter.
        
        Args:
            max_age_hours: Maximum age of content in hours (default 48)
            min_relevance_score: Minimum keyword relevance score (0-1)
            min_text_length: Minimum article text length
        """
        self.max_age_hours = max_age_hours
        self.min_relevance_score = min_relevance_score
        self.min_text_length = min_text_length
        self._compiled_spam = [re.compile(p, re.I) for p in self.SPAM_PATTERNS]
    
    def filter_items(self, items: List[ContentItem]) -> List[ContentItem]:
        """
        Filter content items based on relevance and quality.
        
        Args:
            items: List of content items to filter
            
        Returns:
            List of filtered items
        """
        if not items:
            return []
        
        filtered = []
        stats = {"age": 0, "relevance": 0, "spam": 0, "length": 0, "passed": 0}
        
        for item in items:
            # Check age
            if not self._check_age(item):
                stats["age"] += 1
                continue
            
            # Check relevance
            if not self._check_relevance(item):
                stats["relevance"] += 1
                continue
            
            # Check for spam
            if self._is_spam(item):
                stats["spam"] += 1
                continue
            
            # Check content length (if full_text is available)
            if item.full_text and len(item.full_text) < self.min_text_length:
                stats["length"] += 1
                continue
            
            filtered.append(item)
            stats["passed"] += 1
        
        logger.info(
            f"Content filter: {len(items)} items → {len(filtered)} passed | "
            f"Rejected: age={stats['age']}, relevance={stats['relevance']}, "
            f"spam={stats['spam']}, length={stats['length']}"
        )
        
        return filtered
    
    def _check_age(self, item: ContentItem) -> bool:
        """Check if content is within age limit."""
        if not item.published_at:
            # If no publish date, assume it's recent
            return True
        
        # Use timezone-aware datetime for comparison
        now_utc = datetime.now(timezone.utc)
        
        # Ensure published_at is timezone-aware
        if item.published_at.tzinfo is None:
            # If naive, assume UTC
            published_at_utc = item.published_at.replace(tzinfo=timezone.utc)
        else:
            # Convert to UTC if needed
            published_at_utc = item.published_at.astimezone(timezone.utc)
        
        age = now_utc - published_at_utc
        max_age = timedelta(hours=self.max_age_hours)
        
        return age <= max_age
    
    def _check_relevance(self, item: ContentItem) -> bool:
        """Check if content is relevant based on keywords."""
        # Combine title, tags, and partial text for analysis
        text_to_check = item.title.lower()
        
        if item.tags:
            text_to_check += " " + " ".join(item.tags).lower()
        
        if item.full_text:
            # Check first 500 chars of content
            text_to_check += " " + item.full_text[:500].lower()
        
        # Count matching keywords
        matches = 0
        for keyword in self.RELEVANT_KEYWORDS:
            if keyword in text_to_check:
                matches += 1
        
        # Calculate relevance score
        relevance_score = min(1.0, matches / 3)  # 3+ keywords = max score
        
        return relevance_score >= self.min_relevance_score
    
    def _is_spam(self, item: ContentItem) -> bool:
        """Check if content appears to be spam."""
        text_to_check = item.title
        if item.full_text:
            text_to_check += " " + item.full_text[:200]
        
        for pattern in self._compiled_spam:
            if pattern.search(text_to_check):
                return True
        
        return False
    
    def calculate_relevance_score(self, item: ContentItem) -> float:
        """
        Calculate relevance score for an item.
        
        Returns:
            Score between 0 and 1
        """
        text = item.title.lower()
        if item.tags:
            text += " " + " ".join(item.tags).lower()
        if item.full_text:
            text += " " + item.full_text.lower()
        
        matches = sum(1 for kw in self.RELEVANT_KEYWORDS if kw in text)
        
        # Normalize: 5+ keywords = 1.0
        return min(1.0, matches / 5)

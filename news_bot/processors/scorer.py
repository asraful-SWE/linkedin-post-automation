"""
Content Scorer - Calculates importance scores for content items.

Scoring factors:
1. Recency (newer = higher)
2. Keyword relevance
3. Content length/quality
4. External engagement signals
5. Source reputation
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from news_bot.models import ContentItem

logger = logging.getLogger(__name__)


class ContentScorer:
    """
    Calculates and normalizes importance scores for content items.
    
    Combines multiple signals into a single 0-100 score.
    """
    
    # Source reputation weights (based on general quality)
    SOURCE_WEIGHTS: Dict[str, float] = {
        "hackernews": 1.2,      # High-quality curation
        "techcrunch": 1.15,     # Professional news
        "devto": 1.0,           # Community content
        "medium": 0.95,         # Variable quality
        "reddit": 0.9,          # Community voting helps
    }
    
    # Weights for score components
    WEIGHTS = {
        "recency": 0.25,        # How new the content is
        "relevance": 0.25,      # Keyword relevance
        "engagement": 0.2,      # External signals (upvotes, etc.)
        "quality": 0.2,         # Content length/quality
        "source": 0.1,          # Source reputation
    }
    
    # Relevant keywords (same as filter, used for scoring)
    KEYWORDS = {
        "ai", "machine learning", "deep learning", "gpt", "llm", "chatgpt",
        "programming", "software", "developer", "engineering", "code",
        "javascript", "python", "typescript", "react", "node",
        "cloud", "aws", "kubernetes", "docker", "devops",
        "startup", "career", "interview", "hiring",
        "security", "performance", "architecture", "microservices",
        "database", "api", "backend", "frontend", "fullstack",
    }
    
    def __init__(
        self,
        recency_decay_hours: int = 24,
        max_engagement: int = 1000,
        ideal_content_length: int = 1500,
    ):
        """
        Initialize scorer.
        
        Args:
            recency_decay_hours: Hours until recency score reaches 50%
            max_engagement: External score considered "maximum"
            ideal_content_length: Word count for maximum quality score
        """
        self.recency_decay_hours = recency_decay_hours
        self.max_engagement = max_engagement
        self.ideal_content_length = ideal_content_length
    
    def score(self, item: ContentItem) -> float:
        """
        Calculate overall score for a content item.
        
        Args:
            item: Content item to score
            
        Returns:
            Score between 0 and 100
        """
        # Calculate component scores (all 0-1)
        recency = self._score_recency(item)
        relevance = self._score_relevance(item)
        engagement = self._score_engagement(item)
        quality = self._score_quality(item)
        source = self._score_source(item)
        
        # Weighted average
        weighted_score = (
            self.WEIGHTS["recency"] * recency +
            self.WEIGHTS["relevance"] * relevance +
            self.WEIGHTS["engagement"] * engagement +
            self.WEIGHTS["quality"] * quality +
            self.WEIGHTS["source"] * source
        )
        
        # Scale to 0-100
        final_score = round(weighted_score * 100, 1)
        
        logger.debug(
            f"Scored '{item.title[:40]}...': {final_score} "
            f"(rec={recency:.2f}, rel={relevance:.2f}, eng={engagement:.2f}, "
            f"qual={quality:.2f}, src={source:.2f})"
        )
        
        return final_score
    
    def score_batch(self, items: List[ContentItem]) -> List[ContentItem]:
        """
        Score multiple items and sort by score.
        
        Args:
            items: List of content items
            
        Returns:
            Sorted list with updated scores
        """
        for item in items:
            # Don't overwrite AI-generated score if available
            if item.score == 0 or item.status != "processed":
                item.score = self.score(item)
        
        # Sort by score (highest first)
        items.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Scored {len(items)} items, top score: {items[0].score if items else 0}")
        
        return items
    
    def _score_recency(self, item: ContentItem) -> float:
        """Score based on content age (exponential decay)."""
        if not item.published_at:
            return 0.5  # Unknown age gets middle score
        
        # Handle both naive and aware datetimes
        now = datetime.now(timezone.utc)
        published = item.published_at
        
        # If published_at is naive, assume UTC
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        
        age_hours = (now - published).total_seconds() / 3600
        
        # Exponential decay: score = 0.5^(age / decay_hours)
        decay_factor = 0.5 ** (age_hours / self.recency_decay_hours)
        
        return max(0.1, min(1.0, decay_factor))
    
    def _score_relevance(self, item: ContentItem) -> float:
        """Score based on keyword relevance."""
        text = item.title.lower()
        
        if item.tags:
            text += " " + " ".join(item.tags).lower()
        
        if item.full_text:
            text += " " + item.full_text[:1000].lower()
        
        # Count keyword matches
        matches = sum(1 for kw in self.KEYWORDS if kw in text)
        
        # Normalize: 5+ keywords = 1.0
        return min(1.0, matches / 5)
    
    def _score_engagement(self, item: ContentItem) -> float:
        """Score based on external engagement signals."""
        if not item.external_score:
            return 0.3  # No data gets low-middle score
        
        # Logarithmic scaling for engagement
        # 10 = 0.5, 100 = 0.75, 1000 = 1.0
        import math
        score = math.log10(max(1, item.external_score)) / math.log10(self.max_engagement)
        
        return max(0.1, min(1.0, score))
    
    def _score_quality(self, item: ContentItem) -> float:
        """Score based on content quality indicators."""
        if not item.full_text:
            return 0.3  # No content yet
        
        word_count = len(item.full_text.split())
        
        # Ideal length scoring
        if word_count < 200:
            return 0.3
        elif word_count < 500:
            return 0.5
        elif word_count < 1000:
            return 0.7
        elif word_count < 2000:
            return 0.9
        else:
            return 1.0
    
    def _score_source(self, item: ContentItem) -> float:
        """Score based on source reputation."""
        weight = self.SOURCE_WEIGHTS.get(item.source, 0.8)
        return min(1.0, weight)
    
    def combine_with_ai_score(
        self,
        item: ContentItem,
        ai_score: float,
        ai_weight: float = 0.6,
    ) -> float:
        """
        Combine calculated score with AI-generated score.
        
        Args:
            item: Content item
            ai_score: AI-generated importance score (0-100)
            ai_weight: Weight given to AI score (0-1)
            
        Returns:
            Combined score (0-100)
        """
        calculated_score = self.score(item)
        
        combined = (ai_score * ai_weight) + (calculated_score * (1 - ai_weight))
        
        return round(combined, 1)

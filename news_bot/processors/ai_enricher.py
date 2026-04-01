"""
AI Enricher - Uses OpenAI to enrich content with summaries and insights.

Generates:
1. Short summary (2-3 lines)
2. Key insights (3-5 points)
3. Relevant tags
4. Importance score (0-100)
"""

import json
import logging
import os
from typing import List, Optional, Tuple

from openai import OpenAI

from news_bot.models import ContentItem, ContentStatus

logger = logging.getLogger(__name__)


class AIEnricher:
    """
    Enriches content items with AI-generated analysis.
    
    Uses OpenAI API to generate summaries, insights, and scores.
    """
    
    SYSTEM_PROMPT = """You are a tech content analyst. Your job is to analyze tech articles and provide:
1. A concise summary (2-3 sentences)
2. Key insights (3-5 bullet points)
3. Relevant tags for categorization
4. An importance score (0-100)

Respond in JSON format only:
{
    "summary": "Brief summary of the article",
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "tags": ["tag1", "tag2", "tag3"],
    "importance_score": 75
}

Scoring guidelines:
- 80-100: Breaking news, major announcements, highly impactful
- 60-79: Important updates, useful tutorials, notable insights
- 40-59: General tech news, standard tutorials
- 20-39: Opinion pieces, minor updates
- 0-19: Low value, outdated, or too niche"""

    USER_PROMPT_TEMPLATE = """Analyze this tech article:

Title: {title}
Source: {source}

Content:
{content}

Provide analysis in JSON format."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_content_length: int = 4000,
    ):
        """
        Initialize AI enricher.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: Model to use for analysis
            max_content_length: Maximum content length to send to API
        """
        self.api_key = api_key or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_content_length = max_content_length
        self._client: Optional[OpenAI] = None
    
    @property
    def client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("OpenAI API key not configured")
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def enrich(self, item: ContentItem) -> ContentItem:
        """
        Enrich a single content item with AI analysis.
        
        Args:
            item: Content item to enrich
            
        Returns:
            Enriched content item with summary, key_points, tags, and score
        """
        try:
            # Prepare content for analysis
            content = self._prepare_content(item)
            
            # Call OpenAI API
            result = self._call_api(item.title, item.source, content)
            
            if result:
                item.summary = result.get("summary")
                item.key_points = result.get("key_points", [])
                item.tags = result.get("tags", [])
                item.score = float(result.get("importance_score", 50))
                item.status = ContentStatus.PROCESSED.value
                
                logger.debug(f"Enriched '{item.title[:50]}...' with score {item.score}")
            else:
                # Fallback if API fails
                item.summary = self._generate_fallback_summary(item)
                item.score = 50.0
                item.status = ContentStatus.PROCESSED.value
            
            return item
            
        except Exception as e:
            logger.error(f"Error enriching content: {e}")
            item.error_message = str(e)
            item.status = ContentStatus.FAILED.value
            return item
    
    def enrich_batch(
        self,
        items: List[ContentItem],
        batch_size: int = 10,
    ) -> List[ContentItem]:
        """
        Enrich multiple content items.
        
        Args:
            items: List of content items to enrich
            batch_size: Number of items to process per batch
            
        Returns:
            List of enriched content items
        """
        enriched = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            for item in batch:
                enriched_item = self.enrich(item)
                enriched.append(enriched_item)
            
            logger.info(f"Enriched batch {i // batch_size + 1}: {len(batch)} items")
        
        return enriched
    
    def _prepare_content(self, item: ContentItem) -> str:
        """Prepare content for API call."""
        content = item.full_text or item.title
        
        # Truncate if too long
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "..."
        
        return content
    
    def _call_api(
        self,
        title: str,
        source: str,
        content: str,
    ) -> Optional[dict]:
        """Call OpenAI API for analysis."""
        try:
            user_prompt = self.USER_PROMPT_TEMPLATE.format(
                title=title,
                source=source,
                content=content,
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # Validate result
            if not isinstance(result.get("summary"), str):
                return None
            if not isinstance(result.get("importance_score"), (int, float)):
                result["importance_score"] = 50
            
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse API response as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
    
    def _generate_fallback_summary(self, item: ContentItem) -> str:
        """Generate a simple fallback summary if API fails."""
        if item.full_text:
            # Use first 200 characters as summary
            text = item.full_text[:200].strip()
            if len(item.full_text) > 200:
                text += "..."
            return text
        return item.title

"""
Post Generator Service - Generates LinkedIn posts from content items.

Uses existing AI generation system to create engaging posts
from collected and enriched content.
"""

import logging
import os
import random
from typing import Optional, Tuple

from openai import OpenAI

from news_bot.models import ContentItem
from news_bot.services.content_service import ContentService
from modules.image.fetcher import ImageFetcher
from modules.image.selector import ImageSelector

logger = logging.getLogger(__name__)


# Post generation prompt for tech news content
NEWS_POST_PROMPT = """তুমি একজন বাংলাদেশী experienced software developer। তুমি LinkedIn এ tech news এবং interesting articles share করো নিজের perspective সহ।

তোমাকে এখন একটা tech article থেকে LinkedIn post বানাতে হবে।

===== SOURCE ARTICLE =====
Title: {title}
Source: {source}

Summary:
{summary}

Key Points:
{key_points}

===== WRITING RULES =====

1. BANGLISH STYLE: বাংলা আর English mix করে লিখবে
   GOOD: "এই article টা পড়ে আমার চোখ খুলে গেছে AI adoption নিয়ে"
   BAD: "এই নিবন্ধটি পাঠ করিয়া জ্ঞান লাভ করিলাম"

2. HOOK FIRST: Strong opening দিয়ে শুরু করবে
   - Interesting observation
   - Bold statement
   - Relatable scenario

3. ADD YOUR PERSPECTIVE:
   - শুধু article summarize করবে না
   - নিজের opinion/experience add করবে
   - Why this matters বলবে

4. CALL TO ACTION (optional):
   - Question ask করতে পারো
   - Discussion start করতে পারো

5. FORMAT:
   - Short paragraphs (1-3 sentences)
   - 150-300 words
   - Occasional emoji (1-3 total)
   - No formal Bengali

6. FORBIDDEN:
   - "আজকে আমরা জানবো", "এই পোস্টে" দিয়ে শুরু করো না
   - "ধন্যবাদ সবাইকে", "আশা করি কাজে লাগবে" দিও না
   - Article এর সবকিছু copy করো না
   - Generic motivational quotes দিও না

===== OUTPUT =====

শুধু post content দাও, কোনো explanation না।
Article link share করার দরকার নেই - admin পরে add করবে।
"""


class PostGeneratorService:
    """
    Generates LinkedIn posts from content items.
    
    Uses the content summary and key points to create
    engaging posts with a personal perspective.
    """
    
    def __init__(
        self,
        db_path: str,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        """
        Initialize post generator service.
        
        Args:
            db_path: Path to database
            api_key: OpenAI API key
            model: Model to use for generation
        """
        self.db_path = db_path
        self.api_key = api_key or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.content_service = ContentService(db_path)
        self._client: Optional[OpenAI] = None
        
        # Initialize image fetcher and selector
        try:
            self.image_fetcher = ImageFetcher()
            self.image_selector = ImageSelector(self.image_fetcher)
        except Exception as e:
            logger.warning(f"Could not initialize image fetcher: {e}")
            self.image_fetcher = None
            self.image_selector = None
    
    @property
    def client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("OpenAI API key not configured")
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def generate_post(
        self,
        content_id: int,
        mark_as_used: bool = True,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate a LinkedIn post from a content item.
        
        Args:
            content_id: ID of content item to use
            mark_as_used: Whether to mark content as used
            
        Returns:
            Tuple of (post_content, image_url) or (None, None) on error
        """
        # Get content item
        content = self.content_service.get_by_id(content_id)
        if not content:
            logger.error(f"Content item {content_id} not found")
            return None, None
        
        try:
            # Generate post
            post_content = self._generate(content)
            
            if not post_content:
                logger.error(f"Failed to generate post for content {content_id}")
                return None, None
            
            # Clean up post
            post_content = self._clean_post(post_content)
            
            # Get image URL
            image_url = content.image_url
            
            # If no image, try to search for one
            if not image_url and self.image_fetcher:
                image_url = self._search_image_for_content(content)
            
            # If still no image, use a placeholder
            if not image_url:
                image_url = self._get_placeholder_image(content)
            
            if image_url:
                # Update content with found image if it was searched
                if not content.image_url and self.image_fetcher:
                    content.image_url = image_url
                    try:
                        self.content_service.update(content)
                    except Exception as e:
                        logger.warning(f"Could not update content with image: {e}")
            
            # Mark as used
            if mark_as_used:
                self.content_service.mark_as_used(content_id)
            
            logger.info(f"Generated post from content {content_id}: {len(post_content)} chars")
            
            return post_content, image_url
            
        except Exception as e:
            logger.error(f"Error generating post: {e}")
            return None, None
    
    def generate_from_top_content(
        self,
        min_score: float = 60.0,
    ) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """
        Generate a post from the highest-scored unused content.
        
        Args:
            min_score: Minimum score for content selection
            
        Returns:
            Tuple of (post_content, image_url, content_id) or (None, None, None)
        """
        # Get top unused content
        top_content = self.content_service.list_top_scored(
            limit=1,
            min_score=min_score,
            unused_only=True,
        )
        
        if not top_content:
            logger.warning(f"No unused content with score >= {min_score}")
            return None, None, None
        
        content = top_content[0]
        post_content, image_url = self.generate_post(content.id)
        
        return post_content, image_url, content.id
    
    def _generate(self, content: ContentItem) -> Optional[str]:
        """Generate post content using OpenAI."""
        # Format key points
        key_points_text = ""
        if content.key_points:
            key_points_text = "\n".join(f"- {point}" for point in content.key_points)
        else:
            key_points_text = "(No key points available)"
        
        # Build prompt
        prompt = NEWS_POST_PROMPT.format(
            title=content.title,
            source=content.source,
            summary=content.summary or content.full_text[:500] if content.full_text else content.title,
            key_points=key_points_text,
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "তুমি একজন বাংলাদেশী software developer যে LinkedIn এ tech content share করে। তুমি human এর মতো naturally লেখো, AI এর মতো না।"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=random.uniform(0.85, 0.95),
                max_tokens=700,
                top_p=0.92,
                frequency_penalty=0.3,
                presence_penalty=0.2,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
    
    def _clean_post(self, content: str) -> str:
        """Clean up generated post content."""
        # Replace long dashes
        content = content.replace("—", "-")
        content = content.replace("–", "-")
        
        # Remove meta-text
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            line_lower = line.strip().lower()
            if any(skip in line_lower for skip in [
                'here\'s', 'here is', 'post:', 'linkedin post:',
            ]):
                continue
            cleaned_lines.append(line)
        
        content = '\n'.join(cleaned_lines).strip()
        
        # Remove quotes if wrapped
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1].strip()
        if content.startswith("'") and content.endswith("'"):
            content = content[1:-1].strip()
        
        # Remove trailing generic closers
        generic_closers = [
            "ধন্যবাদ সবাইকে।",
            "ধন্যবাদ।",
            "আশা করি কাজে লাগবে।",
        ]
        for closer in generic_closers:
            if content.endswith(closer):
                content = content[:-len(closer)].strip()
        
        return content
    
    def _search_image_for_content(self, content: ContentItem) -> Optional[str]:
        """
        Search for a relevant image for the content using title and keywords.
        
        Args:
            content: Content item
            
        Returns:
            Image URL or None
        """
        if not self.image_fetcher or not self.image_selector:
            logger.debug("Image fetcher not available")
            return None
        
        try:
            # Check if API keys are configured
            if not self.image_fetcher.unsplash_key and not self.image_fetcher.pexels_key:
                logger.warning("No image API keys configured (UNSPLASH_ACCESS_KEY or PEXELS_API_KEY)")
                return None
            
            # Use title and tags to search for images
            search_query = content.title
            if content.tags:
                search_query += " " + " ".join(content.tags[:3])
            
            logger.info(f"Searching for image: {search_query[:60]}...")
            images = self.image_fetcher.fetch(search_query)
            
            if not images:
                logger.debug(f"No images found for: {search_query}")
                return None
            
            # Select best image
            best_image = None
            best_score = 0
            
            for image in images:
                score = self.image_selector.score_image(image)
                if score > best_score:
                    best_score = score
                    best_image = image
            
            if best_image and best_score > 0:
                image_url = best_image.get("url")
                if image_url:
                    logger.info(f"Found image for content: {image_url[:80]}...")
                    return image_url
            
            return None
            
        except Exception as e:
            logger.warning(f"Error searching for image: {e}")
            return None
    
    def _get_placeholder_image(self, content: ContentItem) -> Optional[str]:
        """
        Get a placeholder/generic image URL based on content keywords.
        Uses Picsum.photos for beautiful random but real-looking images.
        
        Args:
            content: Content item
            
        Returns:
            Placeholder image URL
        """
        import random
        pic_id = random.randint(1, 1000)
        
        # Use Picsum.photos for beautiful random images
        placeholder_url = f"https://picsum.photos/1200/630?random={pic_id}"
        
        logger.info(f"Using placeholder image for content: {placeholder_url[:80]}")
        return placeholder_url
    
    def get_suggested_content(
        self,
        limit: int = 5,
        min_score: float = 50.0,
    ) -> list:
        """
        Get suggested content items for post generation.
        
        Returns list of content items sorted by score.
        """
        return self.content_service.list_top_scored(
            limit=limit,
            min_score=min_score,
            unused_only=True,
        )

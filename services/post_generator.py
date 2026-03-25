"""
Post Generator - Creates humanized Bengali LinkedIn posts (TEXT ONLY)
Uses AI to generate 100% human-sounding posts with 1000+ topic variety
"""

import logging
import random
from typing import Dict
from dotenv import load_dotenv
from ai.openai_provider import OpenAIProvider
from ai.generator import POST_STYLES, POST_MOODS, POST_LENGTHS, HUMANIZED_PROMPT, _clean_post

load_dotenv()
logger = logging.getLogger(__name__)


class PostGenerator:
    """
    Generates human-like text-only LinkedIn posts in Bengali
    Focuses on developer experiences and insights
    """
    
    def __init__(self):
        self.ai_provider = OpenAIProvider()
    
    def generate_post(self, topic: str) -> str:
        """
        Generate a humanized Bengali/Banglish post for the given topic.
        Uses the new human-like prompt system with varied styles, moods, and lengths.
        
        Args:
            topic: The topic to generate content about
            
        Returns:
            Generated post content (80-500 words depending on length selection)
        """
        try:
            logger.info(f"Generating human-like post for topic: {topic}")
            
            # Random style, mood, length for variety
            style = random.choice(POST_STYLES)
            mood = random.choice(POST_MOODS)
            length = random.choices(
                POST_LENGTHS,
                weights=[25, 50, 25],
                k=1
            )[0]
            
            # Format prompt with parameters
            prompt = HUMANIZED_PROMPT.format(style=style, mood=mood, length=length)
            prompt += f"\nTopic: {topic}"
            
            # Set max tokens based on length
            max_tokens_map = {"short": 400, "medium": 700, "long": 1200}
            max_tokens = max_tokens_map.get(length, 700)
            
            # Use AI provider to generate
            post_content = self.ai_provider.generate_completion(
                prompt,
                system_message="তুমি একজন বাংলাদেশী software developer। তুমি LinkedIn এ regularly পোস্ট করো নিজের experience, opinion, আর industry observations নিয়ে। তোমার writing style সম্পূর্ণ natural এবং human। তুমি AI না, তুমি real মানুষ।",
                max_tokens=max_tokens,
                temperature=random.uniform(0.85, 0.98),
            )
            
            # Clean AI artifacts
            post_content = _clean_post(post_content)
            
            logger.info(f"Generated post | Topic: {topic} | Style: {style} | Mood: {mood} | Length: {length}")
            return post_content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate post for topic {topic}: {e}")
            return self._get_fallback_post(topic)

    def _get_fallback_post(self, topic: str) -> str:
        """Fallback post if generation fails"""
        return f"""
{topic} নিয়ে আজ কাজ করতে গিয়ে কিছু নতুন জিনিস শিখেছি।

Development এর journey টা এমনই - প্রতিদিন কিছু না কিছু নতুন শেখার থাকে। 

আপনাদের কি মনে হয় এ বিষয়ে?
        """.strip()
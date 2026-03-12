"""
Post Generator - Creates humanized Bengali LinkedIn posts (TEXT ONLY)
Uses AI to generate 100% human-sounding posts with 1000+ topic variety
"""

import os
import logging
import random
from typing import Optional, Dict, Any
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
        self.post_templates = self._load_post_templates()
        self.topic_contexts = self._load_topic_contexts()
    
    def _load_post_templates(self) -> Dict[str, str]:
        """Load different post structure templates for variety"""
        return {
            "personal_experience": """
আজ {topic} নিয়ে কাজ করতে গিয়ে একটা interesting জিনিস বুঝলাম।

{main_content}

আপনার কি এরকম experience হয়েছে?
            """.strip(),
            
            "lesson_learned": """
{topic} নিয়ে কাজ করার সময় একটা mistake করেছিলাম যেটা থেকে ভালো lesson পেয়েছি।

{main_content}

আপনি কি এরকম কোনো mistake থেকে শিখেছেন?
            """.strip(),
            
            "tips_sharing": """
{topic} এ কাজ করার সময় কিছু tips যেগুলো সত্যিই helpful:

{main_content}

আপনার কোন favorite tip আছে এ বিষয়ে?
            """.strip(),
            
            "problem_solution": """
গতকাল {topic} এর সাথে একটা problem এ stuck হয়েছিলাম। 

{main_content}

এরকম situation এ আপনি কী করেন?
            """.strip(),
            
            "observation": """
কিছুদিন ধরে {topic} observe করছি আর একটা pattern notice করেছি।

{main_content}

আপনার কি মনে হয়?
            """.strip(),
            
            "simple_sharing": """
{main_content}

কী মনে হয় আপনার?
            """.strip()
        }
    
    def _load_topic_contexts(self) -> Dict[str, Dict]:
        """Context and keywords for better content generation"""
        return {
            "Artificial Intelligence": {
                "keywords": ["AI", "Machine Learning", "মডেল", "ডেটা", "algorithm"],
                "context": "AI/ML development এবং implementation"
            },
            "Software Engineering": {
                "keywords": ["clean code", "architecture", "design pattern", "সফটওয়্যার"],
                "context": "সফটওয়্যার development best practices"
            },
            "Web Development": {
                "keywords": ["frontend", "backend", "API", "database", "ওয়েব"],
                "context": "web application development"
            },
            "Programming": {
                "keywords": ["coding", "প্রোগ্রামিং", "bugs", "debugging", "logic"],
                "context": "প্রোগ্রামিং এর বিভিন্ন দিক"
            },
            "Backend Development": {
                "keywords": ["server", "API", "database", "backend", "সার্ভার"],
                "context": "backend systems এবং architecture"
            },
            "DevOps": {
                "keywords": ["deployment", "CI/CD", "containers", "monitoring", "DevOps"],
                "context": "development operations এবং automation"
            },
            "Debugging Lessons": {
                "keywords": ["bug", "debugging", "troubleshooting", "fix", "সমস্যা"],
                "context": "debugging experience এবং শেখার বিষয়"
            },
            "Developer Productivity": {
                "keywords": ["productivity", "time management", "efficiency", "tools", "কাজের গতি"],
                "context": "developer productivity এবং workflow improvement"
            },
            "Coding Mistakes": {
                "keywords": ["mistakes", "ভুল", "learning", "experience", "lesson"],
                "context": "coding এ করা ভুল এবং তা থেকে শেখা"
            },
            "Developer Career Advice": {
                "keywords": ["career", "ক্যারিয়ার", "growth", "skills", "job"],
                "context": "developer career growth এবং advice"
            },
            "System Design": {
                "keywords": ["architecture", "scalability", "design", "system", "সিস্টেম"],
                "context": "system architecture এবং design principles"
            },
            "Open Source": {
                "keywords": ["open source", "contribution", "community", "projects", "ওপেন সোর্স"],
                "context": "open source contribution এবং community"
            }
        }
    
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
    
    def _generate_main_content(self, topic: str, topic_info: Dict) -> str:
        """Generate main content using AI"""
        try:
            prompt = f"""
Write a SHORT Bengali paragraph (30-70 words) about {topic} from a developer's perspective.

Context: {topic_info.get('context', topic)}
Keywords to include: {', '.join(topic_info.get('keywords', [topic]))}

Requirements:
- Write in Bengali (mix with English technical terms)
- Sound like a developer sharing personal experience  
- Be conversational and authentic
- Use short sentences
- Include 1-2 bullet points if helpful
- Do NOT use long dash (—) character
- Keep it concise but meaningful

Write only the main content paragraph, no questions or conclusions.
            """
            
            response = self.ai_provider.generate_completion(prompt)
            return response.strip()
            
        except Exception as e:
            logger.error(f"AI content generation failed: {e}")
            return self._get_simple_content(topic)
    
    def _get_simple_content(self, topic: str) -> str:
        """Simple fallback content generation"""
        simple_contents = {
            "Programming": "Programming এ নতুন কিছু শিখতে হলে practice আর patience দুটাই লাগে। মাঝে মাঝে frustrating লাগে কিন্তু যখন code কাজ করে তখন feeling টা অন্যরকম।",
            "Debugging Lessons": "Bug fixing এর সময় সবচেয়ে বড় lesson হলো ধৈর্য রাখা। অনেক সময় simple mistake হয়ে থাকে যেটা আমরা overlook করে ফেলি।",
            "Web Development": "Web development এ responsive design খুবই important। Mobile first approach follow করলে better result পাওয়া যায়।",
            "Backend Development": "API design করার সময় scalability এর কথা মাথায় রাখা জরুরি। Future এ যেন easily extend করা যায়।"
        }
        
        return simple_contents.get(topic, f"{topic} নিয়ে কাজ করতে গিয়ে নতুন কিছু শিখেছি। Experience share করতে চাই।")
    
    def _trim_content(self, content: str, max_words: int) -> str:
        """Trim content to fit word limit"""
        words = content.split()
        if len(words) <= max_words:
            return content
        
        # Find a good breaking point
        trimmed = []
        current_count = 0
        
        for word in words:
            if current_count >= max_words - 10:  # Leave buffer for ending
                break
            trimmed.append(word)
            current_count += 1
        
        # Add proper ending if needed
        trimmed_text = ' '.join(trimmed)
        if not trimmed_text.endswith(('?', '।', '!')):
            trimmed_text += '।'
        
        return trimmed_text
    
    def _expand_content(self, content: str, topic_info: Dict) -> str:
        """Expand content if too short"""
        expansion_phrases = [
            "এটা নিয়ে আরো কাজ করার plan আছে।",
            "আপনাদের experience কেমন?", 
            "কী মনে হয় এ বিষয়ে?",
            "Share করলে ভালো লাগবে।"
        ]
        
        expansion = random.choice(expansion_phrases)
        return f"{content}\n\n{expansion}"
    
    def _get_fallback_post(self, topic: str) -> str:
        """Fallback post if generation fails"""
        return f"""
{topic} নিয়ে আজ কাজ করতে গিয়ে কিছু নতুন জিনিস শিখেছি।

Development এর journey টা এমনই - প্রতিদিন কিছু না কিছু নতুন শেখার থাকে। 

আপনাদের কি মনে হয় এ বিষয়ে?
        """.strip()
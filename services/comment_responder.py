"""
Comment Responder - Auto-replies to comments on our own LinkedIn posts using AI
"""

import logging
import random
from typing import Optional, Dict, Any, List

from database.models import DatabaseManager
from services.linkedin_publisher import LinkedInPublisher
from ai.generator import _get_client, _clean_post

logger = logging.getLogger(__name__)


# Prompt template for generating replies to comments
REPLY_PROMPT = """You are a tech professional on LinkedIn. Someone commented on your post and you need to reply naturally.

Your post topic: {topic}
Your original post: {post_content}
Comment to reply to: {comment_text}

Write a SHORT, natural, human-like reply (1-3 sentences max). 
Rules:
- Mix Bangla/English just like Bangladeshi tech people do (Banglish)
- Sound genuine, warm, and conversational — NOT corporate
- If the comment is positive, appreciate it naturally (e.g., "Thanks bhai!", "Haha sত্যি বলেছ!", "Exactly!")
- If it's a question, give a brief helpful answer
- If it's a disagreement, acknowledge their point respectfully
- Never start with "Great question!" or "Thank you for your comment"
- Don't use hashtags or emojis unless very natural
- Keep it short — max 2-3 sentences
- Write ONLY the reply text, nothing else

Reply:"""


REPLY_MOODS = [
    "casual and warm",
    "enthusiastic and friendly",
    "thoughtful and brief",
    "relatable and funny",
    "appreciative and genuine",
]


class CommentResponder:
    """Fetches new comments from LinkedIn posts and posts AI-generated replies"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.publisher = LinkedInPublisher()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_new_comments(self) -> Dict[str, Any]:
        """
        Main entry point — checks all tracked posts for new comments and replies.
        Returns a summary dict.
        """
        if self.publisher.mock_mode:
            logger.info("MOCK MODE: Simulating comment processing")
            return {"checked": 0, "replied": 0, "skipped": 0, "errors": 0, "mock": True}

        posts = self.db.get_tracked_posts()
        if not posts:
            logger.info("No tracked posts found for comment checking")
            return {"checked": 0, "replied": 0, "skipped": 0, "errors": 0}

        total_replied = 0
        total_skipped = 0
        total_errors = 0

        for post in posts:
            linkedin_post_id = post["linkedin_post_id"]
            # Skip mock/fallback post IDs
            if not linkedin_post_id or linkedin_post_id.startswith("mock_") or linkedin_post_id.startswith("linkedin_post_"):
                continue

            try:
                comments = self.publisher.get_post_comments(linkedin_post_id)
                if not comments:
                    continue

                for comment in comments:
                    comment_id = comment.get("id")
                    comment_text = comment.get("text", "")

                    if not comment_id or not comment_text:
                        continue

                    # Skip if already replied
                    if self.db.is_comment_replied(comment_id):
                        total_skipped += 1
                        continue

                    # Generate and post reply
                    reply_text = self._generate_reply(
                        topic=post.get("topic", "software development"),
                        post_content=post.get("content", ""),
                        comment_text=comment_text,
                    )

                    if not reply_text:
                        total_errors += 1
                        continue

                    success = self.publisher.post_comment_reply(
                        post_urn=linkedin_post_id,
                        comment_id=comment_id,
                        reply_text=reply_text,
                    )

                    if success:
                        self.db.mark_comment_replied(
                            comment_id=comment_id,
                            post_id=linkedin_post_id,
                            comment_text=comment_text,
                            reply_text=reply_text,
                        )
                        total_replied += 1
                        logger.info(f"Replied to comment {comment_id} on post {linkedin_post_id}")
                    else:
                        total_errors += 1

            except Exception as e:
                logger.error(f"Error processing comments for post {linkedin_post_id}: {e}")
                total_errors += 1

        return {
            "checked": len(posts),
            "replied": total_replied,
            "skipped": total_skipped,
            "errors": total_errors,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_reply(self, topic: str, post_content: str, comment_text: str) -> Optional[str]:
        """Use AI to generate a short, natural reply to a comment"""
        try:
            client = _get_client()
            if client is None:
                logger.warning("OpenAI client not available — cannot generate reply")
                return None

            mood = random.choice(REPLY_MOODS)
            prompt = REPLY_PROMPT.format(
                topic=topic,
                post_content=post_content[:400],   # keep prompt short
                comment_text=comment_text[:300],
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a Bangladeshi tech professional. Be {mood}. "
                            "Reply naturally in Banglish (mix of Bangla + English). "
                            "Keep replies very short (1-3 sentences)."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=120,
                frequency_penalty=0.4,
                presence_penalty=0.3,
            )

            raw_reply = response.choices[0].message.content.strip()
            return _clean_post(raw_reply)

        except Exception as e:
            logger.error(f"Reply generation failed: {e}")
            return None

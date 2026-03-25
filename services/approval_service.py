"""
Approval Service - Handles secure approval/rejection workflow and publishing
"""

import logging
import os
from typing import Any, Dict, Optional

from database.models import DatabaseManager, Post
from services.linkedin_publisher import LinkedInPublisher


logger = logging.getLogger(__name__)


class ApprovalService:
    """Coordinates post approval state transitions and guarded publishing"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.publisher = LinkedInPublisher()
        self.secret = os.getenv("APPROVAL_SECRET", "change-this-secret")
        self.token_expires_hours = int(os.getenv("APPROVAL_TOKEN_EXPIRES_HOURS", "24"))

    def create_pending_post(self, topic: str, content: str) -> Dict[str, Any]:
        """Create pending post + token for email approval"""
        post = Post(topic=topic, content=content, status="pending")
        post_id = self.db.save_post(post)
        token = self.db.create_approval_token(
            post_id=post_id,
            secret_key=self.secret,
            expires_hours=self.token_expires_hours,
        )
        logger.info(f"post generated | post_id={post_id} | status=pending | topic={topic}")
        return {"post_id": post_id, "token": token}

    def _publish_approved_post(self, post_id: int, image_url: Optional[str] = None) -> Dict[str, Any]:
        """Shared publish flow after a post is marked approved."""
        post = self.db.get_post_by_id(post_id)
        if not post:
            return {"success": False, "error": "Post not found"}

        if post["status"] in {"rejected", "published"}:
            return {"success": False, "error": f"Post already {post['status']}"}

        if image_url:
            self.db.set_post_image_url(post_id, image_url)

        self.db.update_post_status(post_id, "approved")
        logger.info(f"post approved | post_id={post_id}")

        publish_result = self.publisher.publish_to_linkedin(
            post_text=post["content"],
            image_url=image_url or post.get("image_url"),
        )
        if not publish_result.get("success"):
            return {"success": False, "error": publish_result.get("error", "Publish failed")}

        linkedin_post_id = publish_result.get("linkedin_post_id")
        if linkedin_post_id:
            self.db.set_linkedin_post_id(post_id, linkedin_post_id)

        self.db.update_post_status(post_id, "published")
        logger.info(f"post published | post_id={post_id} | linkedin_post_id={linkedin_post_id}")

        return {
            "success": True,
            "post_id": post_id,
            "status": "published",
            "linkedin_post_id": linkedin_post_id,
        }

    def approve_post(self, post_id: int, token: str, image_url: Optional[str] = None) -> Dict[str, Any]:
        """Approve + publish post after token validation"""
        if not self.db.validate_approval_token(post_id=post_id, token=token, secret_key=self.secret):
            return {"success": False, "error": "Invalid or expired token"}

        result = self._publish_approved_post(post_id=post_id, image_url=image_url)
        if result.get("success"):
            self.db.mark_approval_token_used(post_id)
        return result

    def approve_post_without_token(self, post_id: int, image_url: Optional[str] = None) -> Dict[str, Any]:
        """Dashboard approval flow (token-less) for internal UI operations."""
        return self._publish_approved_post(post_id=post_id, image_url=image_url)

    def reject_post(self, post_id: int, token: str) -> Dict[str, Any]:
        """Reject post after token validation"""
        if not self.db.validate_approval_token(post_id=post_id, token=token, secret_key=self.secret):
            return {"success": False, "error": "Invalid or expired token"}

        result = self.reject_post_without_token(post_id)
        if result.get("success"):
            self.db.mark_approval_token_used(post_id)
        return result

    def reject_post_without_token(self, post_id: int) -> Dict[str, Any]:
        """Dashboard reject flow (token-less) for internal UI operations."""
        post = self.db.get_post_by_id(post_id)
        if not post:
            return {"success": False, "error": "Post not found"}

        if post["status"] == "published":
            return {"success": False, "error": "Post already published"}

        if post["status"] == "rejected":
            return {"success": False, "error": "Post already rejected"}

        self.db.update_post_status(post_id, "rejected")
        logger.info(f"post rejected | post_id={post_id}")
        return {"success": True, "post_id": post_id, "status": "rejected"}

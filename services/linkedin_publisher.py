"""
LinkedIn Publisher - Handles TEXT-ONLY posting to LinkedIn via API
"""

import io
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class LinkedInPublisher:
    """
    Publishes text-only posts to LinkedIn using the LinkedIn API
    """

    def __init__(self):
        self.access_token = os.getenv("LINKEDIN_ACCESS_TOKEN", "")
        self.person_id = os.getenv("LINKEDIN_PERSON_ID", "")  # LinkedIn person URN
        self.mock_mode = os.getenv("MOCK_LINKEDIN_POSTING", "false").lower() == "true"

        if self.mock_mode:
            logger.info(
                "LinkedIn Publisher running in MOCK MODE - posts will be simulated"
            )
        elif not self.access_token:
            logger.warning("LinkedIn access token not configured")

        self.base_url = "https://api.linkedin.com/v2"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "X-Restli-Protocol-Version": "2.0.0",
            }
        )

    def publish_text_post(self, content: str) -> Optional[str]:
        """
        Publish a text-only post to LinkedIn

        Args:
            content: Post content text

        Returns:
            LinkedIn post ID if successful, None otherwise
        """
        try:
            # Mock mode - simulate posting
            if self.mock_mode:
                fake_post_id = (
                    f"mock_text_post_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                logger.info(
                    f"MOCK MODE: Simulated text post publishing - ID: {fake_post_id}"
                )
                # Avoid logging Bengali content that causes encoding issues
                content_preview = content[:50].encode("ascii", "ignore").decode("ascii")
                logger.info(
                    f"MOCK POST CONTENT: {content_preview}... [Bengali content]"
                )
                return fake_post_id

            if not self.access_token or not self.person_id:
                logger.error("LinkedIn credentials not properly configured")
                return None

            post_data = {
                "author": f"urn:li:person:{self.person_id}",
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {"text": content},
                        "shareMediaCategory": "NONE",
                    }
                },
                "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
            }

            response = self.session.post(
                f"{self.base_url}/ugcPosts", json=post_data, timeout=30
            )

            if response.status_code == 201:
                # Try multiple ways to get post ID
                post_id = response.headers.get("x-linkedin-id")

                # If not in headers, try response body
                if not post_id:
                    try:
                        response_data = response.json()
                        post_id = response_data.get("id")
                    except:
                        pass

                # If still not found, use LinkedIn's activity ID pattern
                if not post_id:
                    post_id = (
                        response.headers.get("location", "").split("/")[-1]
                        if response.headers.get("location")
                        else None
                    )

                # Generate fallback ID if LinkedIn doesn't provide one
                if not post_id:
                    post_id = (
                        f"linkedin_post_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )

                logger.info(f"Successfully published text post: {post_id}")
                return post_id
            else:
                logger.error(
                    f"Failed to publish text post: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"Text post publishing failed: {e}")
            return None

    def publish_post(self, content: str) -> Optional[str]:
        """
        Main method to publish a text-only post

        Args:
            content: Post content text

        Returns:
            LinkedIn post ID if successful, None otherwise
        """
        try:
            if not self.validate_credentials():
                logger.error("Cannot publish - LinkedIn credentials invalid")
                return None

            logger.info("Publishing text-only post")
            return self.publish_text_post(content)

        except Exception as e:
            logger.error(f"Post publishing failed: {e}")
            return None

    def publish_to_linkedin(
        self, post_text: str, image_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Publish to LinkedIn with optional image URL.

        If image_url exists, attempts media publishing flow.
        Otherwise publishes a text-only post.
        """
        try:
            if not self.validate_credentials():
                return {"success": False, "error": "LinkedIn credentials invalid"}

            if image_url:
                linkedin_post_id = self._publish_image_post(
                    post_text=post_text, image_url=image_url
                )
            else:
                linkedin_post_id = self.publish_text_post(post_text)

            if linkedin_post_id:
                return {"success": True, "linkedin_post_id": linkedin_post_id}

            return {"success": False, "error": "LinkedIn publish failed"}
        except Exception as e:
            logger.error(f"publish_to_linkedin failed: {e}")
            return {"success": False, "error": str(e)}

    def _register_image_asset(self, image_url: str) -> Optional[str]:
        """
        Register an image asset with LinkedIn and get back its URN.

        Args:
            image_url: URL of the image to upload

        Returns:
            Asset URN if successful, None otherwise
        """
        try:
            if self.mock_mode:
                return f"urn:li:digitalmediaAsset:mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if not self.access_token or not self.person_id:
                logger.error("LinkedIn credentials not properly configured")
                return None

            # Step 1: Register the upload with LinkedIn
            register_upload_data = {
                "registerUploadRequest": {
                    "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
                    "owner": f"urn:li:person:{self.person_id}",
                    "serviceRelationships": [
                        {
                            "relationshipType": "OWNER",
                            "identifier": "urn:li:userGeneratedContent",
                        }
                    ],
                }
            }

            register_response = self.session.post(
                f"{self.base_url}/assets?action=registerUpload",
                json=register_upload_data,
                timeout=30,
            )

            if register_response.status_code != 200:
                logger.error(
                    f"Failed to register upload: {register_response.status_code} - {register_response.text}"
                )
                return None

            register_data = register_response.json()
            upload_url = (
                register_data.get("value", {})
                .get("uploadMechanism", {})
                .get("com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest", {})
                .get("uploadUrl")
            )
            asset_urn = register_data.get("value", {}).get("asset")

            if not upload_url or not asset_urn:
                logger.error(
                    "Failed to get upload URL or asset URN from registration response"
                )
                return None

            # Step 2: Download the image from the provided URL
            try:
                image_response = requests.get(image_url, timeout=30)
                if image_response.status_code != 200:
                    logger.error(
                        f"Failed to download image from {image_url}: {image_response.status_code}"
                    )
                    return None
                image_data = image_response.content
            except Exception as e:
                logger.error(f"Failed to download image: {e}")
                return None

            # Step 3: Upload the image to the provided upload URL
            upload_headers = {
                "Authorization": f"Bearer {self.access_token}",
                "X-Restli-Protocol-Version": "2.0.0",
            }

            upload_response = requests.put(
                upload_url, data=image_data, headers=upload_headers, timeout=30
            )

            if upload_response.status_code not in [200, 201]:
                logger.error(
                    f"Failed to upload image: {upload_response.status_code} - {upload_response.text}"
                )
                return None

            logger.info(f"Successfully registered image asset: {asset_urn}")
            return asset_urn

        except Exception as e:
            logger.error(f"Image asset registration failed: {e}")
            return None

    def _publish_image_post(self, post_text: str, image_url: str) -> Optional[str]:
        """Publish a post with image using registered asset URN."""
        try:
            if self.mock_mode:
                fake_post_id = (
                    f"mock_image_post_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                logger.info(
                    f"MOCK MODE: Simulated image post publishing - ID: {fake_post_id}"
                )
                return fake_post_id

            if not self.access_token or not self.person_id:
                logger.error("LinkedIn credentials not properly configured")
                return None

            # Step 1: Register and upload the image to get its URN
            asset_urn = self._register_image_asset(image_url)
            if not asset_urn:
                logger.error("Failed to register image asset")
                return None

            # Step 2: Publish post with the registered asset
            post_data = {
                "author": f"urn:li:person:{self.person_id}",
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {"text": post_text},
                        "shareMediaCategory": "IMAGE",
                        "media": [
                            {
                                "status": "READY",
                                "media": asset_urn,
                            }
                        ],
                    }
                },
                "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
            }

            response = self.session.post(
                f"{self.base_url}/ugcPosts",
                json=post_data,
                timeout=30,
            )

            if response.status_code == 201:
                post_id = response.headers.get("x-linkedin-id")
                if not post_id:
                    try:
                        post_id = response.json().get("id")
                    except Exception:
                        post_id = None
                if not post_id:
                    post_id = (
                        response.headers.get("location", "").split("/")[-1]
                        if response.headers.get("location")
                        else None
                    )
                if not post_id:
                    post_id = f"linkedin_image_post_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.info(f"Successfully published image post: {post_id}")
                return post_id

            logger.error(
                f"Failed to publish image post: {response.status_code} - {response.text}"
            )
            return None
        except Exception as e:
            logger.error(f"Image post publishing failed: {e}")
            return None

    def get_post_analytics(self, post_id: str) -> Optional[Dict[str, Any]]:
        """
        Get analytics data for a specific post

        Args:
            post_id: LinkedIn post ID

        Returns:
            Analytics data if successful, None otherwise
        """
        try:
            if not self.access_token or not post_id:
                return None

            # Get post insights (this requires additional permissions)
            response = self.session.get(
                f"{self.base_url}/socialActions/{post_id}", timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                analytics = {
                    "likes": data.get("numLikes", 0),
                    "comments": data.get("numComments", 0),
                    "impressions": data.get("numViews", 0),
                }

                logger.info(f"Retrieved analytics for post {post_id}")
                return analytics
            else:
                logger.warning(f"Failed to get analytics: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Analytics retrieval failed: {e}")
            return None

    def validate_credentials(self) -> bool:
        """
        Validate LinkedIn API credentials

        Returns:
            True if credentials are valid, False otherwise
        """
        try:
            # Mock mode - always return True to bypass validation
            if self.mock_mode:
                logger.info("MOCK MODE: LinkedIn credentials validation bypassed")
                return True

            if not self.access_token:
                logger.error("LinkedIn access token not configured")
                return False

            # Test API call to get user profile using working endpoint
            response = self.session.get(f"{self.base_url}/userinfo", timeout=10)

            if response.status_code == 200:
                user_data = response.json()
                logger.info(
                    f"LinkedIn API credentials validated for user: {user_data.get('given_name', 'Unknown')} {user_data.get('family_name', '')}"
                )
                return True
            else:
                logger.error(f"LinkedIn API validation failed: {response.status_code}")
                logger.debug(f"Response: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Credential validation failed: {e}")
            return False

    def get_publishing_status(self) -> Dict[str, Any]:
        """Get status of LinkedIn publishing capabilities"""
        return {
            "credentials_configured": bool(self.access_token and self.person_id),
            "api_accessible": self.validate_credentials()
            if self.access_token
            else False,
            "mock_mode": self.mock_mode,
            "last_check": datetime.now().isoformat(),
        }

"""
LinkedIn Publisher - Handles TEXT-ONLY posting to LinkedIn via API
"""

import os
import requests
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import json


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
            logger.info("LinkedIn Publisher running in MOCK MODE - posts will be simulated")
        elif not self.access_token:
            logger.warning("LinkedIn access token not configured")
        
        self.base_url = "https://api.linkedin.com/v2"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'X-Restli-Protocol-Version': '2.0.0'
        })
    
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
                fake_post_id = f"mock_text_post_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.info(f"MOCK MODE: Simulated text post publishing - ID: {fake_post_id}")
                # Avoid logging Bengali content that causes encoding issues  
                content_preview = content[:50].encode('ascii', 'ignore').decode('ascii')
                logger.info(f"MOCK POST CONTENT: {content_preview}... [Bengali content]")
                return fake_post_id
            
            if not self.access_token or not self.person_id:
                logger.error("LinkedIn credentials not properly configured")
                return None
            
            post_data = {
                "author": f"urn:li:person:{self.person_id}",
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {
                            "text": content
                        },
                        "shareMediaCategory": "NONE"
                    }
                },
                "visibility": {
                    "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/ugcPosts",
                json=post_data,
                timeout=30
            )
            
            if response.status_code == 201:
                # Try multiple ways to get post ID
                post_id = response.headers.get('x-linkedin-id')
                
                # If not in headers, try response body
                if not post_id:
                    try:
                        response_data = response.json()
                        post_id = response_data.get('id')
                    except:
                        pass
                
                # If still not found, use LinkedIn's activity ID pattern
                if not post_id:
                    post_id = response.headers.get('location', '').split('/')[-1] if response.headers.get('location') else None
                
                # Generate fallback ID if LinkedIn doesn't provide one
                if not post_id:
                    post_id = f"linkedin_post_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                logger.info(f"Successfully published text post: {post_id}")
                return post_id
            else:
                logger.error(f"Failed to publish text post: {response.status_code} - {response.text}")
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
                f"{self.base_url}/socialActions/{post_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                analytics = {
                    "likes": data.get("numLikes", 0),
                    "comments": data.get("numComments", 0), 
                    "impressions": data.get("numViews", 0)
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
            response = self.session.get(
                f"{self.base_url}/userinfo",
                timeout=10
            )
            
            if response.status_code == 200:
                user_data = response.json()
                logger.info(f"LinkedIn API credentials validated for user: {user_data.get('given_name', 'Unknown')} {user_data.get('family_name', '')}")
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
            "api_accessible": self.validate_credentials() if self.access_token else False,
            "mock_mode": self.mock_mode,
            "last_check": datetime.now().isoformat()
        }
    
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
                f"{self.base_url}/socialActions/{post_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                analytics = {
                    "likes": data.get("numLikes", 0),
                    "comments": data.get("numComments", 0), 
                    "shares": data.get("numShares", 0),
                    "impressions": data.get("numViews", 0),
                    "clicks": 0  # Not available in basic API
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
            response = self.session.get(
                f"{self.base_url}/userinfo",
                timeout=10
            )
            
            if response.status_code == 200:
                user_data = response.json()
                logger.info(f"LinkedIn API credentials validated for user: {user_data.get('given_name', 'Unknown')} {user_data.get('family_name', '')}")
                return True
            else:
                logger.error(f"LinkedIn API validation failed: {response.status_code}")
                logger.debug(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Credential validation failed: {e}")
            return False
    
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
    
    def get_publishing_status(self) -> Dict[str, Any]:
        """Get status of LinkedIn publishing capabilities"""
        return {
            "credentials_configured": bool(self.access_token and self.person_id),
            "api_accessible": self.validate_credentials() if self.access_token else False,
            "last_check": datetime.now().isoformat()
        }

    # ------------------------------------------------------------------
    # Comment APIs
    # ------------------------------------------------------------------

    def get_post_comments(self, post_urn: str) -> List[Dict[str, Any]]:
        """
        Fetch comments on a LinkedIn post.

        Args:
            post_urn: The LinkedIn post URN (e.g. urn:li:share:xxx or urn:li:ugcPost:xxx)

        Returns:
            List of dicts with keys: id, text, author
        """
        try:
            if self.mock_mode:
                logger.info(f"MOCK MODE: Simulating get_post_comments for {post_urn}")
                return []

            if not self.access_token:
                logger.error("LinkedIn access token not configured")
                return []

            from urllib.parse import quote
            # Use the URN as-is — do NOT double-wrap it
            encoded_urn = quote(post_urn, safe="")

            # LinkedIn versioned REST API (requires LinkedIn-Version header)
            response = self.session.get(
                f"https://api.linkedin.com/rest/socialActions/{encoded_urn}/comments",
                params={"count": 50},
                headers={"LinkedIn-Version": "202501"},
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                elements = data.get("elements", [])
                comments = []
                for elem in elements:
                    comment_id = elem.get("id", "")
                    message = elem.get("message", {})
                    text = message.get("text", "")
                    author = elem.get("actor", "")
                    if comment_id and text:
                        comments.append({
                            "id": comment_id,
                            "text": text,
                            "author": author,
                        })
                logger.info(f"Fetched {len(comments)} comments for post {post_urn}")
                return comments
            elif response.status_code == 403:
                logger.warning(
                    "LinkedIn API returned 403 for comments — the access token may lack "
                    "r_organization_social or Marketing Developer Platform permissions. "
                    "Comment auto-reply will be skipped until permissions are granted."
                )
                return []
            else:
                logger.warning(
                    f"Failed to fetch comments for {post_urn}: "
                    f"{response.status_code} - {response.text[:200]}"
                )
                return []

        except Exception as e:
            logger.error(f"get_post_comments failed: {e}")
            return []

    def post_comment_reply(self, post_urn: str, comment_id: str, reply_text: str) -> bool:
        """
        Post a reply to an existing comment on a LinkedIn post.

        Args:
            post_urn: The LinkedIn post URN
            comment_id: The comment ID to reply to
            reply_text: Text of the reply

        Returns:
            True if the reply was posted successfully, False otherwise
        """
        try:
            if self.mock_mode:
                logger.info(
                    f"MOCK MODE: Simulated reply to comment {comment_id} "
                    f"on post {post_urn}: {reply_text[:60]}..."
                )
                return True

            if not self.access_token or not self.person_id:
                logger.error("LinkedIn credentials not configured")
                return False

            from urllib.parse import quote
            encoded_urn = quote(post_urn, safe="")

            payload = {
                "actor": f"urn:li:person:{self.person_id}",
                "message": {"text": reply_text},
                "parentComment": comment_id,
            }

            response = self.session.post(
                f"https://api.linkedin.com/rest/socialActions/{encoded_urn}/comments",
                json=payload,
                headers={"LinkedIn-Version": "202501"},
                timeout=30,
            )

            if response.status_code in (200, 201):
                logger.info(f"Posted reply to comment {comment_id}")
                return True
            else:
                logger.error(
                    f"Failed to post reply to comment {comment_id}: "
                    f"{response.status_code} - {response.text[:200]}"
                )
                return False

        except Exception as e:
            logger.error(f"post_comment_reply failed: {e}")
            return False
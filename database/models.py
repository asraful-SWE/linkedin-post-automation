"""
Database models for LinkedIn Auto Poster
"""

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class Post:
    id: Optional[int] = None
    topic: str = ""
    content: str = ""
    status: str = "pending"
    image_url: Optional[str] = None
    created_at: Optional[datetime] = None
    linkedin_post_id: Optional[str] = None
    engagement_score: float = 0.0


@dataclass
class Analytics:
    id: Optional[int] = None
    post_id: int = 0
    likes: int = 0
    comments: int = 0
    impressions: int = 0
    updated_at: Optional[datetime] = None


@dataclass
class TopicPerformance:
    topic: str
    total_posts: int
    avg_engagement: float
    last_used: Optional[datetime]


class DatabaseManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", "linkedin_ai_poster.db")
            # On Railway, use /data volume for persistence if available
            railway_volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
            if railway_volume and os.path.isdir(railway_volume):
                db_path = os.path.join(railway_volume, db_path)
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Posts table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS posts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic TEXT NOT NULL,
                        content TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        image_url TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        linkedin_post_id TEXT,
                        engagement_score REAL DEFAULT 0.0
                    )
                """)
                
                # Check if topic column exists, if not add it (for legacy databases)
                cursor.execute("PRAGMA table_info(posts)")
                columns = [column[1] for column in cursor.fetchall()]
                if 'topic' not in columns:
                    logger.info("Adding missing 'topic' column to posts table")
                    cursor.execute("ALTER TABLE posts ADD COLUMN topic TEXT DEFAULT 'General'")

                if 'status' not in columns:
                    logger.info("Adding missing 'status' column to posts table")
                    cursor.execute("ALTER TABLE posts ADD COLUMN status TEXT DEFAULT 'pending'")

                if 'image_url' not in columns:
                    logger.info("Adding missing 'image_url' column to posts table")
                    cursor.execute("ALTER TABLE posts ADD COLUMN image_url TEXT")
                
                # Analytics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        post_id INTEGER REFERENCES posts(id),
                        likes INTEGER DEFAULT 0,
                        comments INTEGER DEFAULT 0,
                        impressions INTEGER DEFAULT 0,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Topic performance tracking
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS topic_performance (
                        topic TEXT PRIMARY KEY,
                        total_posts INTEGER DEFAULT 0,
                        total_engagement REAL DEFAULT 0.0,
                        avg_engagement REAL DEFAULT 0.0,
                        last_used TIMESTAMP
                    )
                """)

                # Replied comments tracking (to avoid double-replying)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS replied_comments (
                        comment_id TEXT PRIMARY KEY,
                        post_id TEXT NOT NULL,
                        comment_text TEXT,
                        reply_text TEXT,
                        replied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS post_approval_tokens (
                        post_id INTEGER PRIMARY KEY,
                        token_hash TEXT NOT NULL,
                        expires_at TIMESTAMP,
                        used_at TIMESTAMP,
                        FOREIGN KEY(post_id) REFERENCES posts(id)
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def save_post(self, post: Post) -> int:
        """Save a new post and return its ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO posts (topic, content, status, image_url, linkedin_post_id)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        post.topic,
                        post.content,
                        post.status,
                        post.image_url,
                        post.linkedin_post_id,
                    ),
                )
                
                post_id = cursor.lastrowid
                
                # Initialize analytics record
                cursor.execute("""
                    INSERT INTO analytics (post_id) VALUES (?)
                """, (post_id,))
                
                conn.commit()
                logger.info(f"Post saved with ID: {post_id}")
                return post_id
                
        except Exception as e:
            logger.error(f"Failed to save post: {e}")
            raise

    def update_analytics(self, post_id: int, likes: int = 0, comments: int = 0, 
                        impressions: int = 0):
        """Update analytics for a post"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update analytics
                cursor.execute("""
                    UPDATE analytics 
                    SET likes = ?, comments = ?, impressions = ?, 
                        updated_at = CURRENT_TIMESTAMP
                    WHERE post_id = ?
                """, (likes, comments, impressions, post_id))
                
                # Calculate and update engagement score
                engagement_score = likes + (comments * 3)
                cursor.execute("""
                    UPDATE posts SET engagement_score = ? WHERE id = ?
                """, (engagement_score, post_id))
                
                conn.commit()
                logger.info(f"Analytics updated for post {post_id}")
                
        except Exception as e:
            logger.error(f"Failed to update analytics: {e}")

    def get_topic_performance(self) -> List[TopicPerformance]:
        """Get performance metrics for all topics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        topic,
                        COUNT(*) as total_posts,
                        AVG(engagement_score) as avg_engagement,
                        MAX(created_at) as last_used
                    FROM posts 
                    GROUP BY topic
                    ORDER BY avg_engagement DESC
                """)
                
                results = []
                for row in cursor.fetchall():
                    results.append(TopicPerformance(
                        topic=row[0],
                        total_posts=row[1],
                        avg_engagement=row[2] or 0.0,
                        last_used=datetime.fromisoformat(row[3]) if row[3] else None
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get topic performance: {e}")
            return []

    def get_recent_topics(self, days: int = 7) -> List[str]:
        """Get topics used in the last N days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT topic 
                    FROM posts 
                    WHERE created_at > datetime('now', ? || ' days')
                """, (f"-{days}",))
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get recent topics: {e}")
            return []

    def get_posts_count_today(self) -> int:
        """Get number of posts created today"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM posts 
                    WHERE date(created_at) = date('now')
                """)
                
                return cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Failed to get today's post count: {e}")
            return 0

    def get_last_post_time(self) -> Optional[datetime]:
        """Get the timestamp of the last post"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT created_at 
                    FROM posts 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                
                result = cursor.fetchone()
                if result:
                    return datetime.fromisoformat(result[0])
                return None
                
        except Exception as e:
            logger.error(f"Failed to get last post time: {e}")
            return None

    def cleanup_old_data(self, days: int = 90):
        """Clean up data older than specified days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old posts and their analytics
                cursor.execute("""
                    DELETE FROM analytics 
                    WHERE post_id IN (
                        SELECT id FROM posts 
                        WHERE created_at < datetime('now', '-{} days')
                    )
                """.format(days))
                
                cursor.execute("""
                    DELETE FROM posts 
                    WHERE created_at < datetime('now', '-{} days')
                """.format(days))
                
                conn.commit()
                logger.info(f"Cleaned up data older than {days} days")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

    def get_tracked_posts(self) -> List[Dict[str, Any]]:
        """Return all posts that have a LinkedIn post ID (for comment checking)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, linkedin_post_id, topic, content
                    FROM posts
                    WHERE linkedin_post_id IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 50
                """)
                return [
                    {"db_id": row[0], "linkedin_post_id": row[1],
                     "topic": row[2], "content": row[3]}
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Failed to get tracked posts: {e}")
            return []

    def list_posts(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List posts with optional status filter for dashboard UI"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if status:
                    cursor.execute(
                        """
                        SELECT id, topic, content, status, image_url, created_at, linkedin_post_id
                        FROM posts
                        WHERE status = ?
                        ORDER BY created_at DESC
                        """,
                        (status,),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT id, topic, content, status, image_url, created_at, linkedin_post_id
                        FROM posts
                        ORDER BY created_at DESC
                        """
                    )

                return [
                    {
                        "id": row[0],
                        "topic": row[1],
                        "content": row[2],
                        "status": row[3],
                        "image_url": row[4],
                        "created_at": row[5],
                        "linkedin_post_id": row[6],
                    }
                    for row in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"Failed to list posts: {e}")
            return []

    def get_post_by_id(self, post_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single post by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, topic, content, status, image_url, created_at, linkedin_post_id
                    FROM posts
                    WHERE id = ?
                    """,
                    (post_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "topic": row[1],
                    "content": row[2],
                    "status": row[3],
                    "image_url": row[4],
                    "created_at": row[5],
                    "linkedin_post_id": row[6],
                }
        except Exception as e:
            logger.error(f"Failed to fetch post {post_id}: {e}")
            return None

    def update_post_status(self, post_id: int, status: str):
        """Update workflow status for a post"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE posts SET status = ? WHERE id = ?", (status, post_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update post status: {e}")

    def set_post_image_url(self, post_id: int, image_url: Optional[str]):
        """Set optional image URL for a post"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE posts SET image_url = ? WHERE id = ?", (image_url, post_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to set image URL for post {post_id}: {e}")

    def set_linkedin_post_id(self, post_id: int, linkedin_post_id: str):
        """Set LinkedIn post id after successful publish"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE posts SET linkedin_post_id = ? WHERE id = ?",
                    (linkedin_post_id, post_id),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to set linkedin_post_id for post {post_id}: {e}")

    def create_approval_token(self, post_id: int, secret_key: str, expires_hours: int = 24) -> str:
        """Create and store a hashed approval token"""
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(f"{token}:{secret_key}".encode("utf-8")).hexdigest()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO post_approval_tokens(post_id, token_hash, expires_at, used_at)
                    VALUES (?, ?, datetime('now', ? || ' hours'), NULL)
                    ON CONFLICT(post_id)
                    DO UPDATE SET token_hash = excluded.token_hash, expires_at = excluded.expires_at, used_at = NULL
                    """,
                    (post_id, token_hash, expires_hours),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to create approval token for post {post_id}: {e}")
            raise
        return token

    def validate_approval_token(self, post_id: int, token: str, secret_key: str) -> bool:
        """Validate token hash and expiry"""
        token_hash = hashlib.sha256(f"{token}:{secret_key}".encode("utf-8")).hexdigest()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT 1
                    FROM post_approval_tokens
                    WHERE post_id = ?
                      AND token_hash = ?
                      AND used_at IS NULL
                      AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                    """,
                    (post_id, token_hash),
                )
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to validate approval token for post {post_id}: {e}")
            return False

    def mark_approval_token_used(self, post_id: int):
        """Mark token as used to enforce one-time action"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE post_approval_tokens SET used_at = CURRENT_TIMESTAMP WHERE post_id = ?",
                    (post_id,),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to mark approval token as used for post {post_id}: {e}")

    def is_comment_replied(self, comment_id: str) -> bool:
        """Check if we have already replied to this comment"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT 1 FROM replied_comments WHERE comment_id = ?",
                    (comment_id,)
                )
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to check reply status: {e}")
            return False

    def mark_comment_replied(self, comment_id: str, post_id: str,
                              comment_text: str, reply_text: str):
        """Record that we have replied to a comment"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO replied_comments
                        (comment_id, post_id, comment_text, reply_text)
                    VALUES (?, ?, ?, ?)
                """, (comment_id, post_id, comment_text, reply_text))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to mark comment as replied: {e}")
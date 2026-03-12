"""
Database models for LinkedIn Auto Poster
"""

import os
import sqlite3
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
                cursor.execute("""
                    INSERT INTO posts (topic, content, linkedin_post_id)
                    VALUES (?, ?, ?)
                """, (post.topic, post.content, post.linkedin_post_id))
                
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
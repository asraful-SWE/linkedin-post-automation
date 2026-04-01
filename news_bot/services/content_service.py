"""
Content Service - CRUD operations for content items.

Provides database operations for storing, retrieving, and managing
content items in the content_items table.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from news_bot.models import ContentItem, ContentStatus

logger = logging.getLogger(__name__)


class ContentService:
    """
    Service for managing content items in the database.
    
    Handles all CRUD operations for the content_items table.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize content service.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._ensure_table()
    
    def _ensure_table(self):
        """Ensure content_items table exists."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS content_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        url TEXT NOT NULL,
                        url_hash TEXT UNIQUE NOT NULL,
                        source TEXT NOT NULL,
                        author TEXT,
                        full_text TEXT,
                        summary TEXT,
                        key_points TEXT,
                        image_url TEXT,
                        tags TEXT,
                        score REAL DEFAULT 0,
                        external_score INTEGER DEFAULT 0,
                        published_at TIMESTAMP,
                        fetched_at TIMESTAMP,
                        processed_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        used_for_post BOOLEAN DEFAULT 0,
                        status TEXT DEFAULT 'pending',
                        error_message TEXT,
                        retry_count INTEGER DEFAULT 0
                    )
                """)
                
                # Create indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_content_url_hash
                    ON content_items(url_hash)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_content_score
                    ON content_items(score DESC)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_content_created
                    ON content_items(created_at DESC)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_content_status
                    ON content_items(status)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_content_source
                    ON content_items(source)
                """)
                
                conn.commit()
                logger.debug("content_items table ready")
                
        except Exception as e:
            logger.error(f"Error creating content_items table: {e}")
            raise
    
    def save(self, item: ContentItem) -> int:
        """
        Save a content item to the database.
        
        Args:
            item: Content item to save
            
        Returns:
            ID of saved item
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO content_items (
                        title, url, url_hash, source, author,
                        full_text, summary, key_points, image_url, tags,
                        score, external_score, published_at, fetched_at,
                        processed_at, status, error_message, retry_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item.title,
                    item.url,
                    item.url_hash,
                    item.source,
                    item.author,
                    item.full_text,
                    item.summary,
                    item.key_points_json(),
                    item.image_url,
                    item.tags_json(),
                    item.score,
                    item.external_score,
                    item.published_at.isoformat() if item.published_at else None,
                    item.fetched_at.isoformat() if item.fetched_at else None,
                    item.processed_at.isoformat() if item.processed_at else None,
                    item.status,
                    item.error_message,
                    item.retry_count,
                ))
                
                item_id = cursor.lastrowid
                conn.commit()
                
                logger.debug(f"Saved content item {item_id}: {item.title[:50]}...")
                return item_id or 0
                
        except sqlite3.IntegrityError:
            logger.debug(f"Duplicate content item: {item.url_hash}")
            return 0
        except Exception as e:
            logger.error(f"Error saving content item: {e}")
            raise
    
    def save_batch(self, items: List[ContentItem]) -> int:
        """
        Save multiple content items.
        
        Args:
            items: List of content items to save
            
        Returns:
            Number of items saved
        """
        saved = 0
        for item in items:
            try:
                item_id = self.save(item)
                if item_id > 0:
                    saved += 1
            except Exception as e:
                logger.warning(f"Failed to save item: {e}")
                continue
        
        logger.info(f"Saved {saved}/{len(items)} content items")
        return saved
    
    def update(self, item: ContentItem) -> bool:
        """
        Update an existing content item.
        
        Args:
            item: Content item to update (must have id)
            
        Returns:
            True if updated, False otherwise
        """
        if not item.id:
            logger.warning("Cannot update item without id")
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE content_items SET
                        title = ?, full_text = ?, summary = ?, key_points = ?,
                        image_url = ?, tags = ?, score = ?, processed_at = ?,
                        status = ?, error_message = ?, retry_count = ?
                    WHERE id = ?
                """, (
                    item.title,
                    item.full_text,
                    item.summary,
                    item.key_points_json(),
                    item.image_url,
                    item.tags_json(),
                    item.score,
                    datetime.utcnow().isoformat(),
                    item.status,
                    item.error_message,
                    item.retry_count,
                    item.id,
                ))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error updating content item {item.id}: {e}")
            return False
    
    def get_by_id(self, item_id: int) -> Optional[ContentItem]:
        """
        Get a content item by ID.
        
        Args:
            item_id: Content item ID
            
        Returns:
            ContentItem or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT * FROM content_items WHERE id = ?",
                    (item_id,)
                )
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_item(dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Error getting content item {item_id}: {e}")
            return None
    
    def get_by_url_hash(self, url_hash: str) -> Optional[ContentItem]:
        """Get content item by URL hash."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT * FROM content_items WHERE url_hash = ?",
                    (url_hash,)
                )
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_item(dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Error getting content by hash: {e}")
            return None
    
    def list_recent(
        self,
        limit: int = 50,
        source: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ContentItem]:
        """
        List recent content items.
        
        Args:
            limit: Maximum number of items
            source: Filter by source
            status: Filter by status
            
        Returns:
            List of content items
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = "SELECT * FROM content_items WHERE 1=1"
                params = []
                
                if source:
                    query += " AND source = ?"
                    params.append(source)
                
                if status:
                    query += " AND status = ?"
                    params.append(status)
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                
                return [self._row_to_item(dict(row)) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error listing content items: {e}")
            return []
    
    def list_top_scored(
        self,
        limit: int = 20,
        min_score: float = 50.0,
        unused_only: bool = True,
    ) -> List[ContentItem]:
        """
        List top-scored content items.
        
        Args:
            limit: Maximum number of items
            min_score: Minimum score threshold
            unused_only: Only return items not used for posts
            
        Returns:
            List of content items sorted by score
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = """
                    SELECT * FROM content_items
                    WHERE score >= ? AND status = 'processed'
                """
                params = [min_score]
                
                if unused_only:
                    query += " AND used_for_post = 0"
                
                query += " ORDER BY score DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                
                return [self._row_to_item(dict(row)) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error listing top content: {e}")
            return []
    
    def list_pending(self, limit: int = 100) -> List[ContentItem]:
        """Get pending items for processing."""
        return self.list_recent(
            limit=limit,
            status=ContentStatus.PENDING.value,
        )
    
    def mark_as_used(self, item_id: int) -> bool:
        """Mark content item as used for post generation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE content_items SET used_for_post = 1 WHERE id = ?",
                    (item_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error marking item as used: {e}")
            return False
    
    def update_status(
        self,
        item_id: int,
        status: str,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update status of a content item."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if error_message:
                    cursor.execute(
                        """UPDATE content_items
                        SET status = ?, error_message = ?, retry_count = retry_count + 1
                        WHERE id = ?""",
                        (status, error_message, item_id)
                    )
                else:
                    cursor.execute(
                        "UPDATE content_items SET status = ? WHERE id = ?",
                        (status, item_id)
                    )
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            return False
    
    def cleanup_old(self, days: int = 7) -> int:
        """
        Remove content items older than specified days.
        
        Args:
            days: Delete items older than this many days
            
        Returns:
            Number of items deleted
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    """DELETE FROM content_items
                    WHERE created_at < datetime('now', '-' || ? || ' days')""",
                    (days,)
                )
                
                deleted = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted} old content items")
                return deleted
                
        except Exception as e:
            logger.error(f"Error cleaning up old content: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get content statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total count
                cursor.execute("SELECT COUNT(*) FROM content_items")
                total = cursor.fetchone()[0]
                
                # By status
                cursor.execute("""
                    SELECT status, COUNT(*) FROM content_items
                    GROUP BY status
                """)
                by_status = dict(cursor.fetchall())
                
                # By source
                cursor.execute("""
                    SELECT source, COUNT(*) FROM content_items
                    GROUP BY source
                """)
                by_source = dict(cursor.fetchall())
                
                # Average score
                cursor.execute("""
                    SELECT AVG(score) FROM content_items
                    WHERE status = 'processed'
                """)
                avg_score = cursor.fetchone()[0] or 0
                
                return {
                    "total": total,
                    "by_status": by_status,
                    "by_source": by_source,
                    "average_score": round(avg_score, 1),
                }
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def _row_to_item(self, row: Dict[str, Any]) -> ContentItem:
        """Convert database row to ContentItem."""
        # Parse JSON fields
        key_points = []
        if row.get("key_points"):
            try:
                key_points = json.loads(row["key_points"])
            except json.JSONDecodeError:
                pass
        
        tags = []
        if row.get("tags"):
            try:
                tags = json.loads(row["tags"])
            except json.JSONDecodeError:
                pass
        
        # Parse datetime fields
        def parse_dt(val):
            if val and isinstance(val, str):
                try:
                    return datetime.fromisoformat(val)
                except ValueError:
                    return None
            return val
        
        return ContentItem(
            id=row["id"],
            title=row["title"],
            url=row["url"],
            url_hash=row["url_hash"],
            source=row["source"],
            author=row.get("author"),
            full_text=row.get("full_text"),
            summary=row.get("summary"),
            key_points=key_points,
            image_url=row.get("image_url"),
            tags=tags,
            score=row.get("score", 0),
            external_score=row.get("external_score", 0),
            published_at=parse_dt(row.get("published_at")),
            fetched_at=parse_dt(row.get("fetched_at")),
            processed_at=parse_dt(row.get("processed_at")),
            created_at=parse_dt(row.get("created_at")),
            status=row.get("status", "pending"),
            used_for_post=bool(row.get("used_for_post")),
            error_message=row.get("error_message"),
            retry_count=row.get("retry_count", 0),
        )

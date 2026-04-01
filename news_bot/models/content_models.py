"""
Content data models for the News Bot module.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ContentStatus(str, Enum):
    """Status of content item in processing pipeline."""
    PENDING = "pending"           # Just fetched, waiting for processing
    EXTRACTING = "extracting"     # Article extraction in progress
    EXTRACTED = "extracted"       # Article extracted, waiting for AI
    ENRICHING = "enriching"       # AI enrichment in progress
    PROCESSED = "processed"       # Fully processed and ready
    FAILED = "failed"             # Processing failed
    SKIPPED = "skipped"           # Skipped (duplicate, irrelevant, etc.)


class ContentSource(str, Enum):
    """Supported content sources."""
    HACKERNEWS = "hackernews"
    REDDIT = "reddit"
    DEVTO = "devto"
    MEDIUM = "medium"
    TECHCRUNCH = "techcrunch"
    OTHER = "other"


@dataclass
class ContentItem:
    """
    Represents a content item from any source.
    
    This is the core data model used throughout the pipeline:
    collection → extraction → processing → storage → post generation
    """
    # Identity
    id: Optional[int] = None
    url: str = ""
    url_hash: str = ""
    
    # Basic info
    title: str = ""
    source: str = ""
    author: Optional[str] = None
    
    # Content
    full_text: Optional[str] = None
    summary: Optional[str] = None
    key_points: List[str] = field(default_factory=list)
    
    # Media
    image_url: Optional[str] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    score: float = 0.0
    external_score: int = 0  # upvotes, likes, etc.
    
    # Timestamps
    published_at: Optional[datetime] = None
    fetched_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    
    # Status
    status: str = ContentStatus.PENDING.value
    used_for_post: bool = False
    
    # Error tracking
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def __post_init__(self):
        """Generate URL hash if not provided."""
        if self.url and not self.url_hash:
            self.url_hash = self._generate_url_hash(self.url)
        if not self.fetched_at:
            self.fetched_at = datetime.utcnow()
    
    @staticmethod
    def _generate_url_hash(url: str) -> str:
        """Generate MD5 hash of URL for fast deduplication."""
        normalized_url = url.lower().strip().rstrip("/")
        return hashlib.md5(normalized_url.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "url": self.url,
            "url_hash": self.url_hash,
            "title": self.title,
            "source": self.source,
            "author": self.author,
            "full_text": self.full_text,
            "summary": self.summary,
            "key_points": self.key_points,
            "image_url": self.image_url,
            "tags": self.tags,
            "score": self.score,
            "external_score": self.external_score,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "status": self.status,
            "used_for_post": self.used_for_post,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentItem":
        """Create ContentItem from dictionary."""
        # Parse datetime fields
        for date_field in ["published_at", "fetched_at", "processed_at", "created_at"]:
            if data.get(date_field) and isinstance(data[date_field], str):
                data[date_field] = datetime.fromisoformat(data[date_field])
        
        # Ensure lists
        if isinstance(data.get("key_points"), str):
            try:
                data["key_points"] = json.loads(data["key_points"])
            except json.JSONDecodeError:
                data["key_points"] = []
        
        if isinstance(data.get("tags"), str):
            try:
                data["tags"] = json.loads(data["tags"])
            except json.JSONDecodeError:
                data["tags"] = []
        
        return cls(**data)
    
    def key_points_json(self) -> str:
        """Serialize key_points to JSON string for DB storage."""
        return json.dumps(self.key_points, ensure_ascii=False)
    
    def tags_json(self) -> str:
        """Serialize tags to JSON string for DB storage."""
        return json.dumps(self.tags, ensure_ascii=False)


@dataclass
class CollectedItem:
    """
    Lightweight item returned by collectors before full extraction.
    Contains only the minimal info needed to identify and fetch content.
    """
    title: str
    url: str
    source: str
    published_at: Optional[datetime] = None
    external_score: int = 0
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_content_item(self) -> ContentItem:
        """Convert to full ContentItem for further processing."""
        return ContentItem(
            title=self.title,
            url=self.url,
            source=self.source,
            published_at=self.published_at,
            external_score=self.external_score,
            author=self.author,
            tags=self.tags,
        )

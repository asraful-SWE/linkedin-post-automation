"""
Content Routes - API endpoints for content management.

Endpoints:
- GET /content: List content items
- GET /content/top: Top-scored content
- GET /content/stats: Content statistics
- GET /content/{id}: Get content by ID
- POST /content/generate-post/{id}: Generate LinkedIn post
- POST /content/pipeline/run: Trigger pipeline run
- DELETE /content/cleanup: Clean up old content
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

from news_bot.services.content_service import ContentService
from news_bot.services.post_generator_service import PostGeneratorService
from news_bot.services.pipeline_service import run_content_pipeline
from news_bot.models import ContentItem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/content", tags=["content"])


# Request/Response Models
class ContentResponse(BaseModel):
    """Content item response."""
    id: int
    title: str
    url: str
    source: str
    author: Optional[str] = None
    summary: Optional[str] = None
    key_points: List[str] = []
    image_url: Optional[str] = None
    tags: List[str] = []
    score: float
    external_score: int
    published_at: Optional[str] = None
    created_at: Optional[str] = None
    status: str
    used_for_post: bool


class ContentListResponse(BaseModel):
    """List of content items response."""
    items: List[ContentResponse]
    total: int


class GeneratedPostResponse(BaseModel):
    """Generated post response."""
    success: bool
    post_content: Optional[str] = None
    image_url: Optional[str] = None
    content_id: int
    error: Optional[str] = None


class StatsResponse(BaseModel):
    """Content statistics response."""
    total: int
    by_status: Dict[str, int]
    by_source: Dict[str, int]
    average_score: float


class PipelineResponse(BaseModel):
    """Pipeline run response."""
    success: bool
    message: str
    stats: Optional[Dict[str, int]] = None


# Global service instances (initialized in main.py)
_content_service: Optional[ContentService] = None
_post_generator: Optional[PostGeneratorService] = None


def init_services(db_path: str):
    """Initialize services with database path."""
    global _content_service, _post_generator
    _content_service = ContentService(db_path)
    _post_generator = PostGeneratorService(db_path)


def get_content_service() -> ContentService:
    """Get content service instance."""
    if _content_service is None:
        raise HTTPException(500, "Content service not initialized")
    return _content_service


def get_post_generator() -> PostGeneratorService:
    """Get post generator service instance."""
    if _post_generator is None:
        raise HTTPException(500, "Post generator service not initialized")
    return _post_generator


def _item_to_response(item: ContentItem) -> ContentResponse:
    """Convert ContentItem to response model."""
    return ContentResponse(
        id=item.id or 0,
        title=item.title,
        url=item.url,
        source=item.source,
        author=item.author,
        summary=item.summary,
        key_points=item.key_points,
        image_url=item.image_url,
        tags=item.tags,
        score=item.score,
        external_score=item.external_score,
        published_at=item.published_at.isoformat() if item.published_at else None,
        created_at=item.created_at.isoformat() if item.created_at else None,
        status=item.status,
        used_for_post=item.used_for_post,
    )


# Endpoints

@router.get("", response_model=ContentListResponse)
async def list_content(
    limit: int = Query(50, ge=1, le=200),
    source: Optional[str] = None,
    status: Optional[str] = None,
):
    """
    List content items.
    
    - **limit**: Maximum number of items (1-200)
    - **source**: Filter by source (hackernews, reddit, devto, medium, techcrunch)
    - **status**: Filter by status (pending, extracted, processed, failed, skipped)
    """
    service = get_content_service()
    items = service.list_recent(limit=limit, source=source, status=status)
    
    return ContentListResponse(
        items=[_item_to_response(item) for item in items],
        total=len(items),
    )


@router.get("/top", response_model=ContentListResponse)
async def list_top_content(
    limit: int = Query(20, ge=1, le=100),
    min_score: float = Query(50.0, ge=0, le=100),
    unused_only: bool = True,
):
    """
    List top-scored content items.
    
    - **limit**: Maximum number of items (1-100)
    - **min_score**: Minimum score threshold (0-100)
    - **unused_only**: Only return items not used for posts
    """
    service = get_content_service()
    items = service.list_top_scored(
        limit=limit,
        min_score=min_score,
        unused_only=unused_only,
    )
    
    return ContentListResponse(
        items=[_item_to_response(item) for item in items],
        total=len(items),
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get content statistics."""
    service = get_content_service()
    stats = service.get_stats()
    
    return StatsResponse(
        total=stats.get("total", 0),
        by_status=stats.get("by_status", {}),
        by_source=stats.get("by_source", {}),
        average_score=stats.get("average_score", 0),
    )


@router.get("/{content_id}", response_model=ContentResponse)
async def get_content(content_id: int):
    """
    Get a content item by ID.
    
    Returns full content details including full_text.
    """
    service = get_content_service()
    item = service.get_by_id(content_id)
    
    if not item:
        raise HTTPException(404, f"Content item {content_id} not found")
    
    return _item_to_response(item)


@router.get("/{content_id}/full")
async def get_content_full(content_id: int):
    """
    Get full content item including article text.
    
    Returns all fields including full_text.
    """
    service = get_content_service()
    item = service.get_by_id(content_id)
    
    if not item:
        raise HTTPException(404, f"Content item {content_id} not found")
    
    return item.to_dict()


@router.post("/{content_id}/generate-post", response_model=GeneratedPostResponse)
async def generate_post(
    content_id: int,
    mark_as_used: bool = True,
):
    """
    Generate a LinkedIn post from a content item.
    
    - **content_id**: ID of content item to use
    - **mark_as_used**: Whether to mark content as used (prevents reuse)
    
    Returns generated post content and image URL.
    """
    generator = get_post_generator()
    
    post_content, image_url = generator.generate_post(
        content_id=content_id,
        mark_as_used=mark_as_used,
    )
    
    if not post_content:
        return GeneratedPostResponse(
            success=False,
            content_id=content_id,
            error="Failed to generate post",
        )
    
    return GeneratedPostResponse(
        success=True,
        post_content=post_content,
        image_url=image_url,
        content_id=content_id,
    )


@router.post("/generate-post/auto", response_model=GeneratedPostResponse)
async def generate_post_auto(
    min_score: float = Query(60.0, ge=0, le=100),
):
    """
    Generate a LinkedIn post from the top-scored unused content.
    
    Automatically selects the best available content.
    
    - **min_score**: Minimum score for content selection
    """
    generator = get_post_generator()
    
    post_content, image_url, content_id = generator.generate_from_top_content(
        min_score=min_score,
    )
    
    if not post_content or content_id is None:
        return GeneratedPostResponse(
            success=False,
            content_id=0,
            error=f"No unused content with score >= {min_score}",
        )
    
    return GeneratedPostResponse(
        success=True,
        post_content=post_content,
        image_url=image_url,
        content_id=content_id,
    )


@router.post("/pipeline/run", response_model=PipelineResponse)
async def trigger_pipeline(
    background_tasks: BackgroundTasks,
    run_in_background: bool = True,
):
    """
    Trigger a content pipeline run.
    
    - **run_in_background**: If true, runs async and returns immediately
    
    Collects content from all sources, extracts articles,
    enriches with AI, and stores in database.
    """
    service = get_content_service()
    db_path = service.db_path
    
    if run_in_background:
        # Run in background
        background_tasks.add_task(run_content_pipeline, db_path)
        
        return PipelineResponse(
            success=True,
            message="Pipeline started in background",
        )
    else:
        # Run synchronously (may timeout for large runs)
        try:
            stats = await run_content_pipeline(db_path)
            
            return PipelineResponse(
                success=True,
                message="Pipeline completed",
                stats=stats,
            )
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return PipelineResponse(
                success=False,
                message=f"Pipeline failed: {str(e)}",
            )


class SaveGeneratedPostRequest(BaseModel):
    """Request to save a generated post to pending."""
    content_id: int
    post_content: str
    image_url: Optional[str] = None
    topic: Optional[str] = None


class SaveGeneratedPostResponse(BaseModel):
    """Response from saving a generated post."""
    success: bool
    post_id: Optional[int] = None
    error: Optional[str] = None


@router.post("/save-generated-post", response_model=SaveGeneratedPostResponse)
async def save_generated_post(
    request: SaveGeneratedPostRequest,
):
    """
    Save a generated post to pending approval.
    
    Takes the generated content and saves it to the posts table
    with 'pending' status so it can be reviewed and approved.
    
    - **content_id**: ID of content item used
    - **post_content**: Generated post text
    - **image_url**: Optional image URL
    - **topic**: Optional topic label
    """
    try:
        from database.models import DatabaseManager, Post
        from datetime import datetime
        
        # Get database manager from app state - we need to use a different approach
        # since this is in news_bot routes
        db_path = "linkedin_ai_poster.db"  # Default path
        db_manager = DatabaseManager(db_path=db_path)
        
        # Create post object with only supported fields
        post = Post(
            topic=request.topic or "General",
            content=request.post_content,
            image_url=request.image_url,
            status="pending",
            created_at=datetime.utcnow(),
        )
        
        # Save to database
        post_id = db_manager.save_post(post)
        
        if not post_id:
            return SaveGeneratedPostResponse(
                success=False,
                error="Failed to save post to database",
            )
        
        logger.info(f"Saved generated post {post_id} from content {request.content_id}")
        
        return SaveGeneratedPostResponse(
            success=True,
            post_id=post_id,
        )
        
    except Exception as e:
        logger.error(f"Error saving generated post: {e}")
        return SaveGeneratedPostResponse(
            success=False,
            error=str(e),
        )


@router.delete("/cleanup")
async def cleanup_old_content(
    days: int = Query(7, ge=1, le=90),
):
    """
    Clean up content older than specified days.
    
    - **days**: Delete content older than this many days (1-90)
    """
    service = get_content_service()
    deleted = service.cleanup_old(days=days)
    
    return {
        "success": True,
        "deleted": deleted,
        "message": f"Deleted {deleted} items older than {days} days",
    }


@router.post("/{content_id}/mark-used")
async def mark_content_used(content_id: int):
    """Mark a content item as used for post generation."""
    service = get_content_service()
    
    if not service.mark_as_used(content_id):
        raise HTTPException(404, f"Content item {content_id} not found")
    
    return {"success": True, "content_id": content_id}


@router.get("/suggestions/posts")
async def get_post_suggestions(
    limit: int = Query(5, ge=1, le=20),
    min_score: float = Query(50.0, ge=0, le=100),
):
    """
    Get suggested content for post generation.
    
    Returns top-scored unused content items.
    """
    generator = get_post_generator()
    suggestions = generator.get_suggested_content(
        limit=limit,
        min_score=min_score,
    )
    
    return {
        "suggestions": [_item_to_response(item) for item in suggestions],
        "total": len(suggestions),
    }

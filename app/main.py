"""
Main FastAPI Application - LinkedIn Auto Poster
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from database.migrations import run_migrations
from database.models import DatabaseManager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from routes.admin_routes import router as admin_router
from routes.analytics_routes import router as analytics_v2_router
from routes.approval_routes import router as approval_router
from routes.image_routes import router as image_router
from scheduler.posting_scheduler import PostingScheduler
from services.engagement_engine import EngagementEngine
from services.linkedin_publisher import LinkedInPublisher
from services.post_generator import PostGenerator
from services.topic_engine import TopicEngine
from utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global instances
db_manager: Optional[DatabaseManager] = None
scheduler: Optional[PostingScheduler] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global db_manager, scheduler

    try:
        # Initialize database
        db_manager = DatabaseManager()
        app.state.db_manager = db_manager
        logger.info("Database initialized")

        # Apply schema migrations (idempotent — safe to run on every startup)
        run_migrations(db_manager.db_path)
        logger.info("Database migrations applied")

        # Initialize and start scheduler
        scheduler = PostingScheduler(db_manager)
        app.state.scheduler = scheduler

        # Only start automatic scheduling if enabled
        if os.getenv("AUTO_SCHEDULE_ENABLED", "true").lower() == "true":
            scheduler.start_scheduler()
            logger.info("Automatic posting scheduler started")
        else:
            logger.info("Automatic scheduling disabled")

        yield

    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    finally:
        # Cleanup
        if scheduler:
            scheduler.stop_scheduler()
        logger.info("Application shutdown completed")


# Initialize FastAPI app
app = FastAPI(
    title="LinkedIn Auto Poster",
    description="Automated LinkedIn posting system with AI-generated content",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(approval_router)
app.include_router(admin_router)
app.include_router(analytics_v2_router)
app.include_router(image_router)
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# Pydantic models for API
class PostRequest(BaseModel):
    topic: Optional[str] = None
    goal: Optional[str] = None  # e.g. "educational", "viral", "authority"
    use_image: Optional[bool] = None


class AutoImagesUpdate(BaseModel):
    enabled: bool


class AnalyticsUpdate(BaseModel):
    post_id: int
    likes: int = 0
    comments: int = 0
    impressions: int = 0


class PostResponse(BaseModel):
    success: bool
    message: str
    post_id: Optional[int] = None
    linkedin_post_id: Optional[str] = None
    topic: Optional[str] = None
    status: Optional[str] = None
    email_sent: Optional[bool] = None
    error: Optional[str] = None


# ─── Publisher status helper ──────────────────────────────────────────────────


def _get_publisher_status() -> dict:
    """Return publisher status, preferring V2 if available."""
    try:
        from modules.publishing.publisher import LinkedInPublisherV2

        return LinkedInPublisherV2().get_publishing_status()
    except Exception:
        return LinkedInPublisher().get_publishing_status()


# Health check endpoint
@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected" if db_manager else "not_initialized",
            "scheduler": "running"
            if scheduler and scheduler.scheduler.running
            else "stopped",
            "services": {
                "linkedin_publisher": _get_publisher_status(),
                "post_generator": "available",
                "topic_engine": "available",
                "engagement_engine": "available",
            },
        }

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/")
def root():
    return {"message": "LinkedIn AI Poster is running!", "status": "healthy"}


# Dashboard endpoint
@app.get("/dashboard", response_model=Dict[str, Any])
async def get_dashboard():
    """Get comprehensive dashboard data"""
    try:
        if not db_manager or not scheduler:
            raise HTTPException(status_code=500, detail="Services not initialized")

        engagement_engine = EngagementEngine(db_manager)
        topic_engine = TopicEngine(db_manager)

        dashboard = {
            "overview": engagement_engine.get_engagement_dashboard(),
            "scheduler": scheduler.get_scheduler_status(),
            "topic_insights": topic_engine.get_topic_insights(),
            "recent_activity": await _get_recent_activity(),
            "system_status": await _get_system_status(),
        }

        return dashboard

    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        raise HTTPException(status_code=500, detail="Dashboard unavailable")


@app.get("/posts", response_model=List[Dict[str, Any]])
async def list_posts(status: Optional[str] = None):
    """List posts for dashboard with optional status filter"""
    try:
        if not db_manager:
            raise HTTPException(status_code=500, detail="Database not initialized")

        allowed_statuses = {"pending", "approved", "published", "rejected"}
        if status and status not in allowed_statuses:
            raise HTTPException(status_code=400, detail="Invalid status filter")

        return db_manager.list_posts(status=status)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List posts failed: {e}")
        raise HTTPException(status_code=500, detail="Unable to list posts")


# Manual generation endpoint
@app.post("/generate-post", response_model=PostResponse)
async def generate_post_for_approval(request: PostRequest):
    """Generate a post, send approval email, and wait for approval"""
    try:
        if not scheduler:
            raise HTTPException(status_code=500, detail="Scheduler not initialized")

        result = scheduler.manual_post(
            topic=request.topic,
            goal=request.goal,
            use_image=request.use_image,
        )

        if result.get("success"):
            return PostResponse(
                success=True,
                message=result.get("message", "Post generated and sent for approval"),
                post_id=result.get("post_id"),
                topic=result.get("topic"),
                status=result.get("status"),
                email_sent=result.get("email_sent"),
            )
        else:
            return PostResponse(
                success=False,
                message="Post generation failed",
                error=result.get("error"),
            )

    except Exception as e:
        logger.error(f"Manual posting failed: {e}")
        return PostResponse(
            success=False, message="Internal server error", error=str(e)
        )


@app.post("/post/manual", response_model=PostResponse)
async def manual_post_compat(request: PostRequest):
    """Backward-compatible route mapped to approval workflow"""
    return await generate_post_for_approval(request)


# Analytics update endpoint
@app.post("/analytics/update")
async def update_analytics(analytics: AnalyticsUpdate):
    """Update engagement analytics for a post"""
    try:
        if not db_manager:
            raise HTTPException(status_code=500, detail="Database not initialized")

        engagement_engine = EngagementEngine(db_manager)
        engagement_engine.update_post_engagement(
            post_id=analytics.post_id,
            likes=analytics.likes,
            comments=analytics.comments,
            impressions=analytics.impressions,
        )

        return {"message": "Analytics updated successfully"}

    except Exception as e:
        logger.error(f"Analytics update failed: {e}")
        raise HTTPException(status_code=500, detail="Analytics update failed")


# Topic management endpoints
@app.get("/topics/insights", response_model=Dict[str, Any])
async def get_topic_insights():
    """Get topic performance insights"""
    try:
        if not db_manager:
            raise HTTPException(status_code=500, detail="Database not initialized")

        topic_engine = TopicEngine(db_manager)
        return topic_engine.get_topic_insights()

    except Exception as e:
        logger.error(f"Topic insights failed: {e}")
        raise HTTPException(status_code=500, detail="Unable to get topic insights")


@app.get("/topics/recommended", response_model=List[str])
async def get_recommended_topics(count: int = 5):
    """Get recommended topics for upcoming posts"""
    try:
        if not db_manager:
            raise HTTPException(status_code=500, detail="Database not initialized")

        topic_engine = TopicEngine(db_manager)
        return topic_engine.get_next_recommended_topics(count)

    except Exception as e:
        logger.error(f"Recommended topics failed: {e}")
        raise HTTPException(status_code=500, detail="Unable to get recommended topics")


# Scheduler management endpoints
@app.get("/scheduler/status", response_model=Dict[str, Any])
async def get_scheduler_status():
    """Get scheduler status"""
    try:
        if not scheduler:
            raise HTTPException(status_code=500, detail="Scheduler not initialized")

        return scheduler.get_scheduler_status()

    except Exception as e:
        logger.error(f"Scheduler status failed: {e}")
        raise HTTPException(status_code=500, detail="Unable to get scheduler status")


@app.post("/scheduler/start")
async def start_scheduler():
    """Start the posting scheduler"""
    try:
        if not scheduler:
            raise HTTPException(status_code=500, detail="Scheduler not initialized")

        if not scheduler.scheduler.running:
            scheduler.start_scheduler()
            return {"message": "Scheduler started successfully"}
        else:
            return {"message": "Scheduler is already running"}

    except Exception as e:
        logger.error(f"Scheduler start failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to start scheduler")


@app.post("/scheduler/stop")
async def stop_scheduler():
    """Stop the posting scheduler"""
    try:
        if not scheduler:
            raise HTTPException(status_code=500, detail="Scheduler not initialized")

        if scheduler.scheduler.running:
            scheduler.stop_scheduler()
            return {"message": "Scheduler stopped successfully"}
        else:
            return {"message": "Scheduler is already stopped"}

    except Exception as e:
        logger.error(f"Scheduler stop failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop scheduler")


@app.get("/settings/auto-images", response_model=Dict[str, Any])
async def get_auto_images_setting():
    """Get current server-side auto-images runtime setting."""
    try:
        if not scheduler:
            raise HTTPException(status_code=500, detail="Scheduler not initialized")

        return {
            "enabled": bool(getattr(scheduler, "enable_images", False)),
            "active": bool(getattr(scheduler, "image_selector", None) is not None),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get auto-images setting failed: {e}")
        raise HTTPException(status_code=500, detail="Unable to get auto-images setting")


@app.post("/settings/auto-images", response_model=Dict[str, Any])
async def set_auto_images_setting(payload: AutoImagesUpdate):
    """Enable/disable auto-images at runtime and persist it across restarts."""
    try:
        if not scheduler:
            raise HTTPException(status_code=500, detail="Scheduler not initialized")

        result = scheduler.set_auto_images_enabled(payload.enabled, persist=True)
        return {
            "success": True,
            "enabled": result.get("enabled", False),
            "active": result.get("active", False),
            "persisted": result.get("persisted", True),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Set auto-images setting failed: {e}")
        raise HTTPException(status_code=500, detail="Unable to set auto-images setting")


# Analytics and reporting endpoints
@app.get("/analytics/engagement", response_model=Dict[str, Any])
async def get_engagement_analytics():
    """Get engagement analytics dashboard"""
    try:
        if not db_manager:
            raise HTTPException(status_code=500, detail="Database not initialized")

        engagement_engine = EngagementEngine(db_manager)
        return engagement_engine.get_engagement_dashboard()

    except Exception as e:
        logger.error(f"Engagement analytics failed: {e}")
        raise HTTPException(
            status_code=500, detail="Unable to get engagement analytics"
        )


# ─── Topic V2 endpoints (Intelligent Topic Engine) ───────────────────────────


@app.get("/topics/v2/clusters", response_model=Dict[str, Any])
async def get_topic_clusters():
    """Get topic semantic clusters with performance data."""
    try:
        if not db_manager:
            raise HTTPException(status_code=500, detail="Database not initialized")
        from modules.topic.engine import IntelligentTopicEngine
        from services.topic_engine import TopicEngine

        base_engine = TopicEngine(db_manager)
        engine = IntelligentTopicEngine(
            db_manager=db_manager, existing_topic_engine=base_engine
        )
        clusters = {}
        for cname, cluster in engine.clusters.items():
            clusters[cname] = {
                "name": cluster.name,
                "topics": cluster.topics[:10],  # first 10 for brevity
                "total_topics": len(cluster.topics),
            }
        return {"clusters": clusters, "total_clusters": len(clusters)}
    except ImportError:
        raise HTTPException(
            status_code=503, detail="IntelligentTopicEngine not available"
        )
    except Exception as e:
        logger.error(f"Topic clusters failed: {e}")
        raise HTTPException(status_code=500, detail="Unable to get topic clusters")


@app.get("/topics/v2/series", response_model=Dict[str, Any])
async def get_topic_series(topic: str, count: int = 10):
    """Get or create a content series for a topic (e.g. 'AI Series Part 1-10')."""
    try:
        if not db_manager:
            raise HTTPException(status_code=500, detail="Database not initialized")
        from modules.topic.engine import IntelligentTopicEngine

        engine = IntelligentTopicEngine(db_manager=db_manager)
        series = engine.get_or_create_series(
            title=topic,
            total_parts=count,
            topic_template=f"{topic}: Part {{part}}",
        )
        return {
            "series_id": series.series_id,
            "title": series.title,
            "total_parts": series.total_parts,
            "parts": series.parts,
            "current_part": series.current_part,
            "is_active": series.is_active,
        }
    except ImportError:
        raise HTTPException(
            status_code=503, detail="IntelligentTopicEngine not available"
        )
    except Exception as e:
        logger.error(f"Topic series failed: {e}")
        raise HTTPException(status_code=500, detail="Unable to create topic series")


@app.get("/topics/v2/next-series-topic", response_model=Dict[str, Any])
async def get_next_series_topic(series_id: str):
    """Get the next topic in a series."""
    try:
        if not db_manager:
            raise HTTPException(status_code=500, detail="Database not initialized")
        from modules.topic.engine import IntelligentTopicEngine

        engine = IntelligentTopicEngine(db_manager=db_manager)
        next_topic = engine.get_next_series_topic(series_id=series_id)
        if not next_topic:
            raise HTTPException(
                status_code=404, detail=f"Series '{series_id}' not found or complete"
            )
        return {"series_id": series_id, "next_topic": next_topic}
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=503, detail="IntelligentTopicEngine not available"
        )
    except Exception as e:
        logger.error(f"Next series topic failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/topics/v2/insights", response_model=Dict[str, Any])
async def get_topic_insights_v2():
    """Enhanced topic insights with cluster performance and series data."""
    try:
        if not db_manager:
            raise HTTPException(status_code=500, detail="Database not initialized")
        from modules.topic.engine import IntelligentTopicEngine

        engine = IntelligentTopicEngine(db_manager=db_manager)
        return engine.get_topic_insights_v2()
    except ImportError:
        raise HTTPException(
            status_code=503, detail="IntelligentTopicEngine not available"
        )
    except Exception as e:
        logger.error(f"Topic insights v2 failed: {e}")
        raise HTTPException(status_code=500, detail="Unable to get topic insights v2")


# Utility functions
async def _get_recent_activity() -> List[Dict[str, Any]]:
    """Get recent posting activity"""
    try:
        # This would query recent posts from database
        # For now, return placeholder
        return [
            {
                "timestamp": datetime.now().isoformat(),
                "action": "post_published",
                "topic": "Sample Topic",
                "success": True,
            }
        ]
    except Exception:
        return []


async def _get_system_status() -> Dict[str, Any]:
    """Get system status information"""
    try:
        import importlib.util

        # Check intelligent engines via find_spec (no actual import needed)
        intelligent_topic = importlib.util.find_spec("modules.topic.engine") is not None
        intelligent_content = (
            importlib.util.find_spec("modules.content.engine") is not None
        )
        image_enabled = bool(scheduler and getattr(scheduler, "enable_images", False))

        return {
            "database_connected": bool(db_manager),
            "scheduler_running": scheduler and scheduler.scheduler.running,
            "linkedin_configured": bool(os.getenv("LINKEDIN_ACCESS_TOKEN")),
            "ai_provider_configured": bool(os.getenv("OPENAI_API_KEY")),
            "intelligent_topic_engine": intelligent_topic,
            "intelligent_content_engine": intelligent_content,
            "image_auto_selection": image_enabled,
            "image_auto_selection_active": bool(
                scheduler and getattr(scheduler, "image_selector", None) is not None
            ),
            "unsplash_configured": bool(os.getenv("UNSPLASH_ACCESS_KEY")),
            "pexels_configured": bool(os.getenv("PEXELS_API_KEY")),
            "posting_mode": "image_and_text" if image_enabled else "text_only",
        }
    except Exception:
        return {"error": "Unable to get system status"}


# Development endpoints (only in debug mode)
if os.getenv("DEBUG", "false").lower() == "true":

    @app.get("/debug/generate-post")
    async def debug_generate_post(topic: str = "Programming"):
        """Debug endpoint to generate a text-only post without publishing"""
        try:
            post_generator = PostGenerator()
            content = post_generator.generate_post(topic)

            return {
                "topic": topic,
                "content": content,
                "mode": "text_only",
                "word_count": len(content.split()),
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the application
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "false").lower() == "true"

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="debug" if debug else "info",
    )

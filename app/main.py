"""
Main FastAPI Application - LinkedIn Auto Poster
"""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from database.models import DatabaseManager
from scheduler.posting_scheduler import PostingScheduler
from services.topic_engine import TopicEngine
from services.post_generator import PostGenerator
from services.engagement_engine import EngagementEngine
from services.linkedin_publisher import LinkedInPublisher
from services.comment_responder import CommentResponder
from utils.logger import setup_logging
from routes.approval_routes import router as approval_router


# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global instances
db_manager: DatabaseManager = None
scheduler: PostingScheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global db_manager, scheduler
    
    try:
        # Initialize database
        db_manager = DatabaseManager()
        app.state.db_manager = db_manager
        logger.info("Database initialized")
        
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
    lifespan=lifespan
)

app.include_router(approval_router)
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# Pydantic models for API
class PostRequest(BaseModel):
    topic: Optional[str] = None


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


# Health check endpoint
@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected" if db_manager else "not_initialized",
            "scheduler": "running" if scheduler and scheduler.scheduler.running else "stopped",
            "services": {
                "linkedin_publisher": LinkedInPublisher().get_publishing_status(),
                "post_generator": "available",
                "topic_engine": "available",
                "engagement_engine": "available"
            }
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
            "system_status": await _get_system_status()
        }
        
        return dashboard
        
    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        raise HTTPException(status_code=500, detail="Dashboard unavailable")


# Manual generation endpoint
@app.post("/generate-post", response_model=PostResponse)
async def generate_post_for_approval(request: PostRequest):
    """Generate a post, send approval email, and wait for approval"""
    try:
        if not scheduler:
            raise HTTPException(status_code=500, detail="Scheduler not initialized")
        
        result = scheduler.manual_post(topic=request.topic)
        
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
                error=result.get("error")
            )
            
    except Exception as e:
        logger.error(f"Manual posting failed: {e}")
        return PostResponse(
            success=False,
            message="Internal server error",
            error=str(e)
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
            impressions=analytics.impressions
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
        raise HTTPException(status_code=500, detail="Unable to get engagement analytics")


# Comment responder endpoints
@app.post("/comments/check", response_model=Dict[str, Any])
async def check_and_reply_comments(background_tasks: BackgroundTasks):
    """Manually trigger comment checking and AI auto-reply"""
    try:
        if not db_manager:
            raise HTTPException(status_code=500, detail="Database not initialized")

        def run_check():
            responder = CommentResponder(db_manager)
            return responder.process_new_comments()

        background_tasks.add_task(run_check)
        return {"message": "Comment check started in background"}

    except Exception as e:
        logger.error(f"Comment check trigger failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to start comment check")


@app.get("/comments/status", response_model=Dict[str, Any])
async def get_comment_reply_status():
    """Get stats about auto-replied comments"""
    try:
        if not db_manager:
            raise HTTPException(status_code=500, detail="Database not initialized")

        import sqlite3
        with sqlite3.connect(db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM replied_comments")
            total_replied = cursor.fetchone()[0]
            cursor.execute(
                "SELECT COUNT(*) FROM replied_comments "
                "WHERE replied_at > datetime('now', '-1 day')"
            )
            replied_today = cursor.fetchone()[0]

        return {
            "total_comments_replied": total_replied,
            "replied_in_last_24h": replied_today,
            "auto_reply_enabled": True,
            "check_interval_hours": 2,
        }

    except Exception as e:
        logger.error(f"Comment status failed: {e}")
        raise HTTPException(status_code=500, detail="Unable to get comment status")


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
                "success": True
            }
        ]
    except Exception:
        return []


async def _get_system_status() -> Dict[str, Any]:
    """Get system status information"""
    try:
        return {
            "database_connected": bool(db_manager),
            "scheduler_running": scheduler and scheduler.scheduler.running,
            "linkedin_configured": bool(os.getenv("LINKEDIN_ACCESS_TOKEN")),
            "ai_provider_configured": bool(os.getenv("OPENAI_API_KEY")),
            "posting_mode": "text_only"
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
                "word_count": len(content.split())
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
        log_level="debug" if debug else "info"
    )

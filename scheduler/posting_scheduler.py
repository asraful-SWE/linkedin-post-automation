"""
Posting Scheduler - Manages automated LinkedIn post scheduling with natural timing
"""

import os
import random
import logging
from datetime import datetime, timedelta, time
from typing import Dict, Any, Optional, List
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import pytz

from database.models import DatabaseManager, Post
from services.topic_engine import TopicEngine
from services.post_generator import PostGenerator
from services.linkedin_publisher import LinkedInPublisher
from services.engagement_engine import EngagementEngine
from services.comment_responder import CommentResponder


logger = logging.getLogger(__name__)


class PostingScheduler:
    """
    Intelligent posting scheduler that mimics human behavior:
    - Random posting times within specified windows
    - Varied content and format
    - Natural delays and pauses
    - Anti-bot detection measures
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.scheduler = AsyncIOScheduler()
        
        # Initialize services
        self.topic_engine = TopicEngine(db_manager)
        self.post_generator = PostGenerator()
        self.linkedin_publisher = LinkedInPublisher()
        self.engagement_engine = EngagementEngine(db_manager)
        self.comment_responder = CommentResponder(db_manager)
        
        # Configuration
        self.timezone = pytz.timezone(os.getenv("TIMEZONE", "Asia/Dhaka"))
        self.max_posts_per_day = int(os.getenv("MAX_POSTS_PER_DAY", "2"))
        self.min_hours_between_posts = float(os.getenv("MIN_HOURS_BETWEEN_POSTS", "4"))
        self.test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
        
        # Posting windows (as per requirements: 9:30-10:30 and 19:30-20:30)
        self.posting_windows = [
            {"start": time(9, 30), "end": time(10, 30), "weight": 1},   # Morning window
            {"start": time(19, 30), "end": time(20, 30), "weight": 1},  # Evening window
        ]
        
        # Natural behavior settings
        self.skip_probability = float(os.getenv("SKIP_POST_PROBABILITY", "0.15"))
        self.double_post_probability = float(os.getenv("DOUBLE_POST_PROBABILITY", "0.05"))
    
    def start_scheduler(self):
        """Start the automated posting scheduler"""
        try:
            self.scheduler.start()
            
            # Schedule daily posting jobs with randomized times
            self._schedule_daily_posts()
            
            # Schedule maintenance tasks
            self._schedule_maintenance_tasks()
            
            logger.info("Posting scheduler started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        try:
            self.scheduler.shutdown()
            logger.info("Posting scheduler stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {e}")
    
    def _schedule_daily_posts(self):
        """Schedule daily posting jobs with natural variation"""
        try:
            # Test mode: rapid interval posting
            if self.test_mode:
                logger.info(f"TEST MODE: Scheduling posts every {self.min_hours_between_posts} hours")
                
                # Calculate interval in seconds
                interval_seconds = int(self.min_hours_between_posts * 3600)
                
                # Add interval-based job for testing
                self.scheduler.add_job(
                    self._execute_posting_job,
                    IntervalTrigger(seconds=interval_seconds),
                    id="test_interval_posting",
                    replace_existing=True
                )
                
                logger.info(f"Test mode: Posting every {interval_seconds} seconds")
                return
            
            # Normal mode: Schedule posts for different time windows
            for window in self.posting_windows:
                # Create randomized cron jobs for this window
                for _ in range(window["weight"]):
                    # Random minute within the window
                    start_minutes = window["start"].hour * 60 + window["start"].minute
                    end_minutes = window["end"].hour * 60 + window["end"].minute
                    random_minutes = random.randint(start_minutes, end_minutes)
                    
                    hour = random_minutes // 60
                    minute = random_minutes % 60
                    
                    # Add some jitter (±5 minutes)
                    minute_jitter = random.randint(-5, 5)
                    minute = max(0, min(59, minute + minute_jitter))
                    
                    # Schedule job with random days (not every day to seem natural)
                    days_of_week = self._get_natural_posting_days()
                    
                    trigger = CronTrigger(
                        day_of_week=days_of_week,
                        hour=hour,
                        minute=minute,
                        timezone=self.timezone,
                        jitter=300  # ±5 minutes additional jitter
                    )
                    
                    self.scheduler.add_job(
                        self._execute_posting_job,
                        trigger=trigger,
                        id=f"post_job_{hour}_{minute}_{random.randint(1000, 9999)}",
                        replace_existing=True
                    )
            
            logger.info("Daily posting jobs scheduled")
            
        except Exception as e:
            logger.error(f"Failed to schedule daily posts: {e}")
    
    def _get_natural_posting_days(self) -> str:
        """Get natural posting days (not every single day)"""
        # Most active on weekdays, less on weekends
        all_days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        weekdays = ["mon", "tue", "wed", "thu", "fri"]
        weekends = ["sat", "sun"]
        
        # 80% chance to include weekdays, 40% chance for weekends
        selected_days = []
        
        for day in weekdays:
            if random.random() < 0.8:  # 80% chance for weekdays
                selected_days.append(day)
        
        for day in weekends:
            if random.random() < 0.4:  # 40% chance for weekends
                selected_days.append(day)
        
        # Ensure at least 2 days are selected
        if len(selected_days) < 2:
            selected_days = random.sample(all_days, 3)
        
        return ",".join(selected_days)
    
    def _schedule_maintenance_tasks(self):
        """Schedule maintenance and optimization tasks"""
        try:
            # Comment reply check — every 2 hours
            self.scheduler.add_job(
                self._check_and_reply_comments,
                IntervalTrigger(hours=2),
                id="comment_reply_check",
                replace_existing=True
            )

            # Daily analytics update (late night)
            self.scheduler.add_job(
                self._update_analytics_job,
                CronTrigger(hour=2, minute=30, timezone=self.timezone),
                id="analytics_update",
                replace_existing=True
            )
            
            # Weekly topic performance refresh
            self.scheduler.add_job(
                self._refresh_topic_performance,
                CronTrigger(day_of_week="sun", hour=3, timezone=self.timezone),
                id="topic_performance_refresh",
                replace_existing=True
            )
            
            # Monthly cleanup
            self.scheduler.add_job(
                self._monthly_cleanup,
                CronTrigger(day=1, hour=4, timezone=self.timezone),
                id="monthly_cleanup",
                replace_existing=True
            )
            
            logger.info("Maintenance tasks scheduled")
            
        except Exception as e:
            logger.error(f"Failed to schedule maintenance tasks: {e}")
    
    async def _execute_posting_job(self):
        """Execute a single posting job with natural behavior"""
        try:
            # Check if we should skip this post (natural behavior)
            if random.random() < self.skip_probability:
                logger.info("Skipping post naturally (random skip)")
                return
            
            # Check daily post limit
            posts_today = self.db.get_posts_count_today()
            if posts_today >= self.max_posts_per_day:
                logger.info(f"Daily post limit reached ({posts_today}/{self.max_posts_per_day})")
                return
            
            # Check minimum time between posts
            last_post_time = self.db.get_last_post_time()
            if last_post_time:
                time_since_last = datetime.now() - last_post_time
                if time_since_last.total_seconds() < (self.min_hours_between_posts * 3600):
                    logger.info("Too soon since last post, skipping")
                    return
            
            # Generate and publish post
            await self._generate_and_publish_post()
            
            # Small chance for a follow-up post later
            if random.random() < self.double_post_probability and posts_today == 0:
                # Schedule a follow-up post 3-6 hours later
                delay_seconds = random.randint(3 * 3600, 6 * 3600)
                self.scheduler.add_job(
                    self._generate_and_publish_post,
                    'date',
                    run_date=datetime.now() + timedelta(seconds=delay_seconds),
                    id=f"followup_post_{random.randint(1000, 9999)}"
                )
                logger.info(f"Scheduled follow-up post in {delay_seconds/3600:.1f} hours")
            
        except Exception as e:
            logger.error(f"Posting job execution failed: {e}")
    
    async def _generate_and_publish_post(self):
        """Generate and publish a single text-only post"""
        try:
            logger.info("Starting post generation and publishing process")
            
            # Step 1: Select topic
            topic = self.topic_engine.select_topic()
            logger.info(f"Selected topic: {topic}")
            
            # Step 2: Generate post content  
            post_content = self.post_generator.generate_post(topic)
            logger.info("Post content generated")
            
            # Step 3: Publish to LinkedIn (text-only)
            linkedin_post_id = self.linkedin_publisher.publish_post(post_content)
            
            if linkedin_post_id:
                # Step 4: Save to database
                post = Post(
                    topic=topic,
                    content=post_content,
                    linkedin_post_id=linkedin_post_id
                )
                
                post_id = self.db.save_post(post)
                logger.info(f"Text post published successfully: LinkedIn ID {linkedin_post_id}, DB ID {post_id}")
                
                # Update topic performance (if engagement data is available later)
                # For now, just log the successful post
                
                # Add natural delay before next action (anti-bot measure)
                await asyncio.sleep(random.randint(30, 120))
                
            else:
                logger.error("Failed to publish post to LinkedIn")
            
        except Exception as e:
            logger.error(f"Post generation and publishing failed: {e}")
    
    async def _update_analytics_job(self):
        """Update analytics for recent posts"""
        try:
            logger.info("Starting analytics update job")
            
            # This would typically fetch analytics from LinkedIn API
            # For now, we'll simulate with random data for demonstration
            
            # In a real implementation, you would:
            # 1. Get recent posts from database
            # 2. Fetch analytics from LinkedIn API
            # 3. Update engagement_engine with real data
            
            logger.info("Analytics update completed")
            
        except Exception as e:
            logger.error(f"Analytics update job failed: {e}")
    
    async def _refresh_topic_performance(self):
        """Refresh topic performance weights"""
        try:
            logger.info("Refreshing topic performance")
            self.topic_engine.update_topic_performance()
            logger.info("Topic performance refresh completed")
            
        except Exception as e:
            logger.error(f"Topic performance refresh failed: {e}")
    
    async def _monthly_cleanup(self):
        """Monthly database cleanup"""
        try:
            logger.info("Starting monthly cleanup")
            
            # Cleanup old database records (keep 3 months of data)
            # This would be implemented in database manager
            # self.db.cleanup_old_data(days=90)
            
            logger.info("Monthly cleanup completed")
            
        except Exception as e:
            logger.error(f"Monthly cleanup failed: {e}")

    async def _check_and_reply_comments(self):
        """Check for new comments on our posts and reply with AI"""
        try:
            logger.info("Starting comment reply check")
            result = self.comment_responder.process_new_comments()
            logger.info(
                f"Comment check done — checked: {result['checked']}, "
                f"replied: {result['replied']}, skipped: {result['skipped']}, "
                f"errors: {result['errors']}"
            )
        except Exception as e:
            logger.error(f"Comment reply check failed: {e}")
    
    def manual_post(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """Manually trigger a text-only post (for testing)"""
        try:
            if topic:
                # Use provided topic
                selected_topic = topic
            else:
                # Select topic normally
                selected_topic = self.topic_engine.select_topic()
            
            # Generate post content 
            post_content = self.post_generator.generate_post(selected_topic)
            
            # Publish text-only post
            linkedin_post_id = self.linkedin_publisher.publish_post(post_content)
            
            if linkedin_post_id:
                # Save to database
                post = Post(
                    topic=selected_topic,
                    content=post_content,
                    linkedin_post_id=linkedin_post_id
                )
                
                post_id = self.db.save_post(post)
                
                return {
                    "success": True,
                    "post_id": post_id,
                    "linkedin_post_id": linkedin_post_id,
                    "topic": selected_topic
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to publish to LinkedIn"
                }
                
        except Exception as e:
            logger.error(f"Manual post failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        try:
            jobs = self.scheduler.get_jobs()
            
            return {
                "running": self.scheduler.running,
                "total_jobs": len(jobs),
                "posting_jobs": len([job for job in jobs if job.id.startswith("post_job_")]),
                "next_posts": [
                    {
                        "id": job.id,
                        "next_run": job.next_run_time.isoformat() if job.next_run_time else None
                    }
                    for job in jobs if job.id.startswith("post_job_")
                ][:5],
                "posts_today": self.db.get_posts_count_today(),
                "max_posts_per_day": self.max_posts_per_day
            }
            
        except Exception as e:
            logger.error(f"Failed to get scheduler status: {e}") 
            return {"error": "Unable to get status"}
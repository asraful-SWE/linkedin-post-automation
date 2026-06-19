"""
Message Formatter - সুন্দর ফরম্যাটে Telegram মেসেজ তৈরি করুন
"""

from datetime import datetime
from typing import Dict, List, Optional


class MessageFormatter:
    """Format messages for Telegram"""
    
    @staticmethod
    def dashboard_summary(overview: Dict, system_status: Dict, next_run: Optional[str] = None) -> str:
        """ড্যাশবোর্ড সামারি ফরম্যাট করুন"""
        
        message = "📊 <b>LinkedIn Auto Posting Dashboard</b>\n\n"
        
        if overview:
            message += f"📈 <b>Statistics:</b>\n"
            message += f"├ Total Posts: <code>{overview.get('total_posts', 0)}</code>\n"
            message += f"├ Total Engagement: <code>{overview.get('total_engagement', 0):.0f}</code> likes\n"
            message += f"├ Active Topics: <code>{overview.get('unique_topics', 0)}</code>\n"
            message += f"└ Avg Engagement: <code>{overview.get('avg_engagement_score', 0):.2f}/10</code>\n\n"
        
        if system_status:
            message += f"🔧 <b>System Status:</b>\n"
            message += f"├ Database: {'✅ Online' if system_status.get('database_connected') else '❌ Offline'}\n"
            message += f"├ Scheduler: {'✅ Running' if system_status.get('scheduler_running') else '⏹️ Stopped'}\n"
            message += f"├ LinkedIn: {'✅ Configured' if system_status.get('linkedin_configured') else '❌ Not Configured'}\n"
            message += f"└ AI Engine: {'✅ Ready' if system_status.get('ai_provider_configured') else '❌ Not Ready'}\n\n"
        
        if next_run:
            message += f"⏰ <b>Next Run:</b> <code>{next_run}</code>\n\n"
        
        message += "━━━━━━━━━━━━━━━━━━━━━━━\n"
        message += "Select an option below:"
        
        return message
    
    @staticmethod
    def topic_performance(topics: List[Dict]) -> str:
        """Topic performance দেখান"""
        
        message = "📊 <b>Topic Performance</b>\n\n"
        
        for i, topic in enumerate(topics[:10], 1):
            avg_score = topic.get('avg_score', 0)
            posts = topic.get('posts', 0)
            emoji = "🔥" if avg_score >= 8 else "👍" if avg_score >= 6 else "📌"
            
            message += f"{i}. {emoji} <b>{topic.get('topic', 'Unknown')}</b>\n"
            message += f"   Posts: <code>{posts}</code> | Score: <code>{avg_score:.2f}/10</code>\n"
        
        return message
    
    @staticmethod
    def post_preview(topic: str, content: str, score: float = 0) -> str:
        """পোস্ট preview দেখান"""
        
        message = f"✨ <b>Post Preview - {topic}</b>\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        message += f"{content}\n\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        if score:
            message += f"📊 Engagement Score: <code>{score:.2f}/10</code>\n"
        
        return message
    
    @staticmethod
    def statistics(stats: Dict) -> str:
        """বিস্তারিত statistics দেখান"""
        
        message = "📈 <b>Detailed Statistics</b>\n\n"
        
        message += f"<b>Posts:</b>\n"
        message += f"├ Total: <code>{stats.get('total_posts', 0)}</code>\n"
        message += f"├ This Week: <code>{stats.get('weekly_posts', 0)}</code>\n"
        message += f"└ This Month: <code>{stats.get('monthly_posts', 0)}</code>\n\n"
        
        message += f"<b>Engagement:</b>\n"
        message += f"├ Total Likes: <code>{stats.get('total_likes', 0)}</code>\n"
        message += f"├ Total Comments: <code>{stats.get('total_comments', 0)}</code>\n"
        message += f"└ Avg per Post: <code>{stats.get('avg_engagement', 0):.2f}</code>\n\n"
        
        message += f"<b>Performance:</b>\n"
        message += f"├ Best Topic: <code>{stats.get('best_topic', 'N/A')}</code>\n"
        message += f"└ Best Score: <code>{stats.get('best_score', 0):.2f}/10</code>\n"
        
        return message
    
    @staticmethod
    def schedule_info(next_posts: List[Dict]) -> str:
        """Schedule information দেখান"""
        
        if not next_posts:
            return "📅 <b>Schedule</b>\n\nNo scheduled posts yet."
        
        message = "📅 <b>Next Scheduled Posts</b>\n\n"
        
        for i, post in enumerate(next_posts[:5], 1):
            next_run = post.get('next_run', 'Unknown')
            message += f"{i}. ⏰ <code>{next_run}</code>\n"
        
        return message
    
    @staticmethod
    def error_message(error: str, context: str = "") -> str:
        """Error message ফরম্যাট করুন"""
        
        message = f"❌ <b>Error</b>\n\n"
        message += f"<code>{error}</code>\n"
        
        if context:
            message += f"\n<b>Context:</b> {context}"
        
        return message
    
    @staticmethod
    def success_message(title: str, details: str = "") -> str:
        """Success message ফরম্যাট করুন"""
        
        message = f"✅ <b>{title}</b>\n\n"
        
        if details:
            message += details
        
        return message

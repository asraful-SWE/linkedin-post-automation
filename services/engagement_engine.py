"""
Engagement Engine - Tracks and learns from post performance
"""

import logging
import sqlite3
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from database.models import DatabaseManager


logger = logging.getLogger(__name__)


@dataclass
class EngagementMetrics:
    post_id: int
    likes: int
    comments: int
    impressions: int
    engagement_rate: float
    performance_score: float


@dataclass
class TopicInsight:
    topic: str
    avg_engagement_score: float
    best_performing_post: Optional[int]
    worst_performing_post: Optional[int] 
    post_count: int
    performance_trend: str  # "improving", "declining", "stable"


class EngagementEngine:
    """
    Analyzes post engagement patterns and provides insights for optimization
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def calculate_engagement_score(self, likes: int, comments: int, 
                                 impressions: int) -> float:
        """
        Calculate engagement score for text-only posts
        
        Formula: likes + (comments * 3) + impression_ratio
        """
        try:
            # Base engagement score (comments weighted more heavily)
            base_score = likes + (comments * 3)
            
            # Add impression efficiency (engagement per 1000 impressions)
            if impressions > 0:
                impression_efficiency = (likes + comments) / (impressions / 1000)
                base_score += impression_efficiency
            
            return round(base_score, 2)
            
        except Exception as e:
            logger.error(f"Engagement score calculation failed: {e}")
            return 0.0
    
    def update_post_engagement(self, post_id: int, likes: int = 0, comments: int = 0,
                             impressions: int = 0):
        """Update engagement data for a text-only post"""
        try:
            # Calculate engagement score
            engagement_score = self.calculate_engagement_score(
                likes, comments, impressions
            )
            
            # Update in database
            self.db.update_analytics(post_id, likes, comments, impressions)
            
            logger.info(f"Updated engagement for post {post_id}: score={engagement_score}")
            
        except Exception as e:
            logger.error(f"Failed to update post engagement: {e}")
    
    def get_topic_performance_analysis(self) -> List[TopicInsight]:
        """Analyze performance across all topics"""
        try:
            performance_data = self.db.get_topic_performance()
            insights = []
            
            for topic_perf in performance_data:
                # Get detailed post data for this topic
                with sqlite3.connect(self.db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT id, engagement_score, created_at
                        FROM posts
                        WHERE topic = ?
                        ORDER BY created_at DESC
                        """,
                        (topic_perf.topic,),
                    )
                    posts = cursor.fetchall()
                    
                if not posts:
                    continue
                
                # Calculate trends
                scores = [post[1] for post in posts if post[1] is not None]
                trend = self._calculate_performance_trend(scores)
                
                # Find best and worst posts
                best_post = max(posts, key=lambda x: x[1] if x[1] else 0)[0] if posts else None
                worst_post = min(posts, key=lambda x: x[1] if x[1] else float('inf'))[0] if posts else None
                
                insight = TopicInsight(
                    topic=topic_perf.topic,
                    avg_engagement_score=topic_perf.avg_engagement,
                    best_performing_post=best_post,
                    worst_performing_post=worst_post,
                    post_count=len(posts),
                    performance_trend=trend
                )
                
                insights.append(insight)
            
            # Sort by average engagement score
            insights.sort(key=lambda x: x.avg_engagement_score, reverse=True)
            
            logger.info(f"Generated performance analysis for {len(insights)} topics")
            return insights
            
        except Exception as e:
            logger.error(f"Topic performance analysis failed: {e}")
            return []
    
    def _calculate_performance_trend(self, scores: List[float]) -> str:
        """Calculate if performance is improving, declining, or stable"""
        if len(scores) < 3:
            return "insufficient_data"
        
        try:
            # Compare recent performance vs older performance
            recent_scores = scores[:len(scores)//2]  # First half (most recent)
            older_scores = scores[len(scores)//2:]   # Second half (older)
            
            recent_avg = sum(recent_scores) / len(recent_scores)
            older_avg = sum(older_scores) / len(older_scores)
            
            if recent_avg > older_avg * 1.1:  # 10% better
                return "improving"
            elif recent_avg < older_avg * 0.9:  # 10% worse
                return "declining"
            else:
                return "stable"
                
        except Exception:
            return "stable"
    
    def get_optimal_posting_insights(self) -> Dict[str, Any]:
        """Analyze optimal posting patterns"""
        try:
            # This would require timestamp analysis of high-performing posts
            # For now, return general insights
            
            topic_insights = self.get_topic_performance_analysis()
            
            insights = {
                "best_performing_topics": [
                    {"topic": t.topic, "avg_score": t.avg_engagement_score}
                    for t in topic_insights[:5]
                ],
                "underperforming_topics": [
                    {"topic": t.topic, "avg_score": t.avg_engagement_score}
                    for t in topic_insights[-3:]
                ],
                "improvement_opportunities": [],
                "posting_frequency_recommendation": self._analyze_posting_frequency()
            }
            
            # Add improvement recommendations
            for topic in topic_insights:
                if topic.performance_trend == "declining":
                    insights["improvement_opportunities"].append({
                        "topic": topic.topic,
                        "issue": "declining_performance",
                        "recommendation": "Consider changing approach or taking a break from this topic"
                    })
                elif topic.avg_engagement_score < 5.0:
                    insights["improvement_opportunities"].append({
                        "topic": topic.topic,
                        "issue": "low_engagement",
                        "recommendation": "Try different content formats or focus on more specific aspects"
                    })
            
            return insights
            
        except Exception as e:
            logger.error(f"Optimal posting insights failed: {e}")
            return {}
    
    def _analyze_posting_frequency(self) -> Dict[str, Any]:
        """Analyze current posting frequency and suggest improvements"""
        try:
            # Get posting frequency over last 30 days
            today_posts = self.db.get_posts_count_today()
            
            # Simple frequency analysis
            if today_posts >= 3:
                recommendation = "Consider reducing frequency - too many posts per day might reduce engagement"
            elif today_posts == 0:
                recommendation = "No posts today - maintaining consistent posting schedule is important"
            else:
                recommendation = "Good posting frequency - maintain current schedule"
            
            return {
                "today_posts": today_posts,
                "recommendation": recommendation,
                "optimal_daily_posts": 2
            }
            
        except Exception as e:
            logger.error(f"Posting frequency analysis failed: {e}")
            return {"optimal_daily_posts": 2, "recommendation": "Maintain consistent posting schedule"}
    
    def get_engagement_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive engagement dashboard data"""
        try:
            topic_analysis = self.get_topic_performance_analysis()
            posting_insights = self.get_optimal_posting_insights()
            
            # Calculate overall stats
            total_posts = sum(t.post_count for t in topic_analysis)
            avg_engagement = sum(t.avg_engagement_score * t.post_count for t in topic_analysis) / max(total_posts, 1)
            
            dashboard = {
                "overview": {
                    "total_posts": total_posts,
                    "average_engagement_score": round(avg_engagement, 2),
                    "total_topics_used": len(topic_analysis),
                    "posts_today": self.db.get_posts_count_today()
                },
                "top_performing_topics": posting_insights.get("best_performing_topics", []),
                "improvement_opportunities": posting_insights.get("improvement_opportunities", []),
                "posting_frequency": posting_insights.get("posting_frequency_recommendation", {}),
                "topic_breakdown": [
                    {
                        "topic": t.topic,
                        "posts": t.post_count,
                        "avg_score": t.avg_engagement_score,
                        "trend": t.performance_trend
                    }
                    for t in topic_analysis[:10]  # Top 10 topics
                ]
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Engagement dashboard generation failed: {e}")
            return {"error": "Unable to generate dashboard"}
    
    def predict_post_performance(self, topic: str) -> Dict[str, Any]:
        """Predict expected performance for a post based on historical data"""
        try:
            # Get historical performance for this topic
            topic_insights = self.get_topic_performance_analysis()
            topic_data = next((t for t in topic_insights if t.topic.lower() == topic.lower()), None)
            
            if topic_data:
                base_score = topic_data.avg_engagement_score
            else:
                # Use overall average for new topics
                all_scores = [t.avg_engagement_score for t in topic_insights]
                base_score = sum(all_scores) / len(all_scores) if all_scores else 5.0
            
            predicted_score = base_score
            
            # Categorize expected performance
            if predicted_score >= 15:
                performance_category = "high"
            elif predicted_score >= 8:
                performance_category = "medium"
            else:
                performance_category = "low"
            
            return {
                "predicted_engagement_score": round(predicted_score, 2),
                "performance_category": performance_category,
                "confidence": "high" if topic_data and topic_data.post_count >= 3 else "medium",
                "historical_posts": topic_data.post_count if topic_data else 0
            }
            
        except Exception as e:
            logger.error(f"Performance prediction failed: {e}")
            return {"predicted_engagement_score": 5.0, "performance_category": "medium", "confidence": "low"}
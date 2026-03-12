"""
Topic Engine - Intelligently selects topics for text-only LinkedIn posts
"""

import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from database.models import DatabaseManager, TopicPerformance
from services.topics import ALL_TOPICS, TOPIC_CATEGORIES, get_random_topic, get_topic_count


logger = logging.getLogger(__name__)


class TopicEngine:
    """
    Topic selection system for Bengali LinkedIn posts that:
    1. Selects from 1000+ diverse topics
    2. Learns from engagement performance
    3. Avoids recent repetition
    4. Prioritizes high-performing topic categories
    """
    
    # Use the massive topic list from topics.py
    TOPICS = ALL_TOPICS
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.topic_weights = self._initialize_topic_weights()
    
    def _initialize_topic_weights(self) -> Dict[str, float]:
        """Initialize topic weights based on historical performance"""
        # Start with equal weights
        weights = {topic: 1.0 for topic in self.TOPICS}
        
        try:
            performance_data = self.db.get_topic_performance()
            
            if performance_data:
                logger.info(f"Loading performance data for {len(performance_data)} topics")
                
                # Calculate average engagement
                total_engagement = sum(p.avg_engagement for p in performance_data)
                avg_engagement = total_engagement / len(performance_data) if performance_data else 0
                
                # Update weights based on performance
                for perf in performance_data:
                    if perf.topic in weights and avg_engagement > 0:
                        # Topics with higher engagement get higher weights
                        # But limit extreme differences (0.5x to 3x range)
                        multiplier = max(0.5, min(3.0, perf.avg_engagement / avg_engagement))
                        weights[perf.topic] = multiplier
                        logger.debug(f"Topic '{perf.topic}' weight: {multiplier:.2f}")
                
                logger.info("Topic weights initialized from performance data")
            else:
                logger.info("No performance data found, using equal weights")
                
        except Exception as e:
            logger.error(f"Failed to initialize topic weights: {e}")
        
        return weights
    
    def select_topic(self, avoid_recent_days: int = 3) -> str:
        """
        Select the next topic intelligently
        
        Args:
            avoid_recent_days: Number of days to avoid repeating topics
            
        Returns:
            Selected topic string
        """
        try:
            # Get recently used topics
            recent_topics = set(self.db.get_recent_topics(days=avoid_recent_days))
            logger.debug(f"Recent topics to avoid: {recent_topics}")
            
            # Available topics (not used recently)
            available_topics = [topic for topic in self.TOPICS if topic not in recent_topics]
            
            # If all topics were used recently, allow all but with penalty
            if not available_topics:
                available_topics = self.TOPICS.copy()
                recent_penalty = 0.3  # Reduce weight of recently used topics
                logger.info(f"All topics used recently, applying {recent_penalty} penalty")
            else:
                recent_penalty = 1.0  # No penalty
            
            # Build weighted selection pool
            weighted_topics = []
            for topic in available_topics:
                base_weight = self.topic_weights.get(topic, 1.0)
                
                # Apply recent usage penalty if needed
                if topic in recent_topics:
                    final_weight = base_weight * recent_penalty
                else:
                    final_weight = base_weight
                
                # Add multiple instances based on weight (for random selection)
                instances = max(1, int(final_weight * 10))
                weighted_topics.extend([topic] * instances)
            
            # Select topic randomly from weighted pool
            selected_topic = random.choice(weighted_topics)
            
            logger.info(f"Selected topic: {selected_topic}")
            return selected_topic
            
        except Exception as e:
            logger.error(f"Topic selection failed: {e}")
            # Fallback to random topic
            return random.choice(self.TOPICS)
    
    def update_topic_performance(self, topic: str, engagement_score: float):
        """
        Update topic performance and adjust weights
        
        Args:
            topic: Topic that was posted
            engagement_score: Calculated engagement score
        """
        try:
            if topic not in self.TOPICS:
                logger.warning(f"Unknown topic for performance update: {topic}")
                return
            
            # Update performance in database (this will be handled by the database manager)
            # For now, just update local weights
            
            # Get current performance data
            performance_data = self.db.get_topic_performance()
            topic_performance = next((p for p in performance_data if p.topic == topic), None)
            
            if topic_performance:
                # Calculate running average
                total_posts = topic_performance.total_posts
                old_total = topic_performance.avg_engagement * total_posts
                new_total = old_total + engagement_score
                new_average = new_total / (total_posts + 1)
                
                # Update weight based on new performance
                overall_avg = sum(p.avg_engagement for p in performance_data) / len(performance_data)
                if overall_avg > 0:
                    new_weight = max(0.5, min(3.0, new_average / overall_avg))
                    self.topic_weights[topic] = new_weight
                    
                logger.info(f"Updated performance for topic '{topic}': score={engagement_score:.2f}, new_weight={new_weight:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to update topic performance: {e}")
    
    def get_topic_stats(self) -> Dict[str, any]:
        """
        Get statistics about topic usage and performance
        
        Returns:
            Dictionary with topic statistics
        """
        try:
            performance_data = self.db.get_topic_performance()
            recent_topics = self.db.get_recent_topics(days=7)
            
            stats = {
                "total_topics": len(self.TOPICS),
                "topics_with_data": len(performance_data),
                "recent_usage": len(recent_topics),
                "topic_weights": dict(sorted(self.topic_weights.items(), key=lambda x: x[1], reverse=True)),
                "top_performers": []
            }
            
            # Add top performing topics
            if performance_data:
                sorted_performance = sorted(performance_data, key=lambda x: x.avg_engagement, reverse=True)
                stats["top_performers"] = [
                    {"topic": p.topic, "avg_engagement": p.avg_engagement, "total_posts": p.total_posts}
                    for p in sorted_performance[:5]
                ]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get topic stats: {e}")
            return {"error": str(e)}
    
    def get_next_recommended_topics(self, count: int = 3) -> List[str]:
        """
        Get next recommended topics without selecting them
        
        Args:
            count: Number of topic recommendations to return
            
        Returns:
            List of recommended topic strings
        """
        try:
            recent_topics = set(self.db.get_recent_topics(days=3))
            available_topics = [topic for topic in self.TOPICS if topic not in recent_topics]
            
            if not available_topics:
                available_topics = self.TOPICS.copy()
            
            # Sort by weight (descending)
            sorted_topics = sorted(
                available_topics, 
                key=lambda t: self.topic_weights.get(t, 1.0), 
                reverse=True
            )
            
            return sorted_topics[:count]
            
        except Exception as e:
            logger.error(f"Failed to get recommended topics: {e}")
            return self.TOPICS[:count]
    
    def update_topic_performance(self):
        """
        Update topic weights based on latest performance data
        Call this periodically to adapt to changing engagement patterns
        """
        try:
            performance_data = self.db.get_topic_performance()
            
            if not performance_data:
                return
            
            # Calculate new weights
            avg_engagement = sum(p.avg_engagement for p in performance_data) / len(performance_data)
            
            for perf in performance_data:
                if perf.topic in self.topic_weights:
                    # Adaptive weight calculation
                    base_multiplier = perf.avg_engagement / max(avg_engagement, 1.0) if avg_engagement > 0 else 1.0
                    
                    # Apply smoothing to avoid extreme changes
                    old_weight = self.topic_weights[perf.topic]
                    new_weight = (old_weight * 0.7) + (base_multiplier * 0.3)
                    
                    # Keep weights within reasonable bounds
                    self.topic_weights[perf.topic] = max(0.2, min(3.0, new_weight))
            
            logger.info("Topic weights updated based on performance")
            
        except Exception as e:
            logger.error(f"Failed to update topic performance: {e}")
    
    def get_topic_insights(self) -> Dict[str, any]:
        """Get insights about topic performance and usage"""
        try:
            performance_data = self.db.get_topic_performance()
            recent_topics = self.db.get_recent_topics(days=7)
            
            insights = {
                "total_available_topics": len(self.TOPICS),
                "total_unique_topics": get_topic_count(),
                "total_categories": len(TOPIC_CATEGORIES),
                "total_topics_used": len(performance_data),
                "topics_used_recently": len(recent_topics),
                "best_performing_topics": [],
                "topic_weights": self.topic_weights.copy()
            }
            
            # Best performing topics
            if performance_data:
                sorted_performance = sorted(performance_data, key=lambda x: x.avg_engagement, reverse=True)
                insights["best_performing_topics"] = [
                    {
                        "topic": p.topic,
                        "avg_engagement": round(p.avg_engagement, 2),
                        "total_posts": p.total_posts
                    }
                    for p in sorted_performance[:5]
                ]
            
            # Unused topics count (not listing all - too many)
            used_topics = {p.topic for p in performance_data}
            insights["unused_topics_count"] = len([topic for topic in self.TOPICS if topic not in used_topics])
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get topic insights: {e}")
            return {}
    
    def get_next_recommended_topics(self, count: int = 5) -> List[str]:
        """Get recommended topics for upcoming posts"""
        try:
            recent_topics = set(self.db.get_recent_topics(days=2))
            available_topics = [topic for topic in self.TOPICS if topic not in recent_topics]
            
            if len(available_topics) < count:
                available_topics = self.TOPICS
            
            # Sort by weight (descending)
            weighted_topics = sorted(
                available_topics, 
                key=lambda t: self.topic_weights.get(t, 1.0), 
                reverse=True
            )
            
            return weighted_topics[:count]
            
        except Exception as e:
            logger.error(f"Failed to get recommended topics: {e}")
            return self.TOPICS[:count]
    
    def force_topic_refresh(self):
        """Reset topic weights and force fresh analysis"""
        try:
            logger.info("Forcing topic refresh...")
            self.topic_weights = self._initialize_topic_weights()
            self.update_topic_performance()
            logger.info("Topic refresh completed")
            
        except Exception as e:
            logger.error(f"Topic refresh failed: {e}")
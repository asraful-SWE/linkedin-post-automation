"""
News Bot Configuration.

Centralizes all configuration options for the news bot system.
"""

import os
from typing import Dict, List

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class CollectorConfig(BaseModel):
    """Configuration for individual collectors."""
    enabled: bool = True
    max_items: int = 30
    rate_limit_delay: float = 1.0


class ExtractionConfig(BaseModel):
    """Configuration for content extraction."""
    enabled: bool = True
    timeout: int = 30
    max_retries: int = 3
    concurrency: int = 5
    enable_fallback: bool = True
    skip_domains: List[str] = Field(default_factory=lambda: ["medium.com", "paywalled-site.com"])


class ImageConfig(BaseModel):
    """Configuration for image extraction."""
    enabled: bool = True  # Enabled to search images for generated posts
    min_width: int = 300
    min_height: int = 200
    timeout: int = 10


class AIConfig(BaseModel):
    """Configuration for AI enrichment."""
    enabled: bool = True
    batch_size: int = 5
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.3


class FilterConfig(BaseModel):
    """Configuration for content filtering."""
    max_age_hours: int = 48
    min_word_count: int = 50
    keywords: List[str] = Field(default_factory=lambda: [
        "ai", "artificial intelligence", "machine learning", "ml", "deep learning",
        "programming", "software", "development", "coding", "python", "javascript",
        "backend", "frontend", "api", "web", "mobile", "cloud", "aws", "azure",
        "docker", "kubernetes", "devops", "microservices", "database", "sql",
        "nosql", "react", "vue", "angular", "node", "typescript", "java",
        "tech", "technology", "startup", "saas", "framework", "library",
        "open source", "github", "automation", "testing", "security", "privacy"
    ])


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    enable_structured: bool = True


class NewsConfig(BaseSettings):
    """Main news bot configuration."""
    
    # Collector configurations
    collectors: Dict[str, CollectorConfig] = Field(default_factory=lambda: {
        "hackernews": CollectorConfig(enabled=True, max_items=50),
        "reddit": CollectorConfig(enabled=True, max_items=30),
        "devto": CollectorConfig(enabled=True, max_items=40),
        "medium": CollectorConfig(enabled=False, max_items=20),  # Disabled due to 403s
        "techcrunch": CollectorConfig(enabled=True, max_items=25),
    })
    
    # Feature configurations
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    images: ImageConfig = Field(default_factory=ImageConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    filtering: FilterConfig = Field(default_factory=FilterConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Pipeline settings
    pipeline_interval_minutes: int = 60
    cleanup_old_content_days: int = 7
    max_pipeline_duration_minutes: int = 30
    
    # Performance settings
    database_batch_size: int = 100
    request_timeout: int = 30
    
    # API Keys (from environment)
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    reddit_client_id: str = Field(default="", env="REDDIT_CLIENT_ID")
    reddit_client_secret: str = Field(default="", env="REDDIT_CLIENT_SECRET")
    
    class Config:
        env_file = ".env"
        env_prefix = "NEWS_BOT_"
        extra = "ignore"  # Ignore extra fields from .env


# Global config instance
config = NewsConfig()


def get_collector_config(name: str) -> CollectorConfig:
    """Get configuration for a specific collector."""
    return config.collectors.get(name, CollectorConfig())


def is_collector_enabled(name: str) -> bool:
    """Check if a collector is enabled."""
    return get_collector_config(name).enabled


def update_config(**kwargs):
    """Update configuration at runtime."""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)


# Logging setup
import logging

def setup_logging():
    """Setup logging based on configuration."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format=config.logging.format,
        force=True
    )
    
    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("feedparser").setLevel(logging.WARNING)
    
    # Custom logger for our system
    logger = logging.getLogger("news_bot")
    return logger


# Initialize logging
setup_logging()
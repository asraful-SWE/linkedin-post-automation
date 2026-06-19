"""
Telegram Bot Configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Bot configuration"""
    
    # Telegram Bot
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    ADMIN_ID = int(os.getenv("TELEGRAM_ADMIN_ID", "0")) if os.getenv("TELEGRAM_ADMIN_ID") else None
    
    # API
    API_URL = os.getenv("API_URL", "http://localhost:8000")
    
    # Database
    DATABASE_PATH = os.getenv("DATABASE_PATH", "linkedin_ai_poster.db")
    
    # AI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Timezone
    TIMEZONE = os.getenv("TIMEZONE", "Asia/Dhaka")
    
    # Features
    MAX_POSTS_PER_DAY = int(os.getenv("MAX_POSTS_PER_DAY", "2"))
    MIN_HOURS_BETWEEN_POSTS = int(os.getenv("MIN_HOURS_BETWEEN_POSTS", "4"))
    AUTO_SCHEDULE_ENABLED = os.getenv("AUTO_SCHEDULE_ENABLED", "true").lower() == "true"


# Verify configuration
if not Config.BOT_TOKEN:
    raise ValueError("❌ TELEGRAM_BOT_TOKEN not set in .env file")
if not Config.ADMIN_ID:
    raise ValueError("❌ TELEGRAM_ADMIN_ID not set in .env file")

print(f"✅ Config loaded successfully")

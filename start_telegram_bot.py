"""
Start Telegram Bot Script
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from telegram_bot.bot import LinkedInTelegramBot
from telegram_bot.config import Config

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 LinkedIn Auto Posting Telegram Bot")
    print("=" * 60)
    print(f"\n✅ Configuration loaded:")
    print(f"   Bot Token: {Config.BOT_TOKEN[:20]}...")
    print(f"   Admin ID: {Config.ADMIN_ID}")
    print(f"   API URL: {Config.API_URL}")
    print(f"   Database: {Config.DATABASE_PATH}")
    print(f"\n🚀 Starting bot...\n")
    
    try:
        bot = LinkedInTelegramBot()
        bot.run()
    except KeyboardInterrupt:
        print("\n\n⏹️  Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

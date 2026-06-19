"""
Start Handler - /start command এবং মেইন মেনু
"""

import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from telegram_bot.services.formatter import MessageFormatter
from telegram_bot.services.telegram_publisher import TelegramPublisher

logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /start command handler
    ব্যবহারকারীকে স্বাগত জানান এবং মেনু দেখান
    """
    
    user = update.effective_user
    logger.info(f"User {user.id} started bot")
    
    # স্বাগত বার্তা
    welcome_message = f"""
🤖 <b>Welcome to LinkedIn Auto Posting Bot!</b>

Hello <b>{user.first_name}</b>! 👋

I'm your assistant for managing LinkedIn posts automatically. You can:

📊 View dashboard and statistics
🤖 Generate new posts instantly
📅 Manage posting schedule
⚙️ Configure settings

Let's get started! 🚀
"""
    
    # মেনু বাটন
    keyboard = [
        [
            InlineKeyboardButton("📊 Dashboard", callback_data="menu_dashboard"),
            InlineKeyboardButton("🤖 Generate", callback_data="menu_generate"),
        ],
        [
            InlineKeyboardButton("📈 Stats", callback_data="menu_stats"),
            InlineKeyboardButton("📅 Schedule", callback_data="menu_schedule"),
        ],
        [
            InlineKeyboardButton("⚙️ Settings", callback_data="menu_settings"),
            InlineKeyboardButton("❓ Help", callback_data="menu_help"),
        ],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(welcome_message, reply_markup=reply_markup, parse_mode="HTML")


async def show_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """সাহায্য দেখান"""
    
    help_text = """
<b>🆘 Available Commands:</b>

<b>Main Menu:</b>
├ /start - মেইন মেনু দেখান
├ /help - এই সাহায্য দেখান
└ /about - বট সম্পর্কে জানুন

<b>Quick Actions:</b>
├ /dashboard - ড্যাশবোর্ড দেখান
├ /generate - পোস্ট generate করুন
├ /stats - statistics দেখান
├ /schedule - পরবর্তী পোস্ট দেখান
└ /settings - settings configure করুন

<b>Features:</b>
📊 Real-time dashboard monitoring
🤖 AI-powered content generation
📅 Smart scheduling
📈 Detailed analytics
⚙️ Configuration management

Type any command to get started!
"""
    
    query = update.callback_query
    if query:
        await query.edit_message_text(help_text, parse_mode="HTML")
    else:
        await update.message.reply_text(help_text, parse_mode="HTML")


async def show_about(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """বট সম্পর্কে তথ্য দেখান"""
    
    about_text = """
<b>🤖 LinkedIn Auto Posting Bot</b>

Version: 1.0.0
Status: ✅ Active

This bot helps you manage your LinkedIn auto-posting system directly from Telegram!

<b>Features:</b>
✨ AI-powered Bengali content generation
🎯 Smart topic rotation
📊 Real-time analytics
⏰ Intelligent scheduling
🛡️ Anti-bot detection evasion

<b>Built with:</b>
🐍 Python + FastAPI
🤖 OpenAI GPT
📱 Telegram Bot API
💾 SQLite Database

<b>Created by:</b> Your AI Assistant
<b>Last Updated:</b> April 2026

Made with ❤️ for content creators
"""
    
    query = update.callback_query
    if query:
        await query.edit_message_text(about_text, parse_mode="HTML")
    else:
        await update.message.reply_text(about_text, parse_mode="HTML")

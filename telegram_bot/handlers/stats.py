"""
Stats Handler - বিস্তারিত statistics দেখান
"""

import logging
import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from telegram_bot.config import Config
from telegram_bot.services.formatter import MessageFormatter

logger = logging.getLogger(__name__)


async def show_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """বিস্তারিত statistics দেখান"""
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{Config.API_URL}/dashboard")
            
            if response.status_code == 200:
                dashboard_data = response.json()
                overview = dashboard_data.get("overview", {}).get("overview", {})
                
                # Statistics তৈরি করুন
                stats = {
                    'total_posts': overview.get('total_posts', 0),
                    'weekly_posts': overview.get('posts_this_week', 0),
                    'monthly_posts': overview.get('posts_this_month', 0),
                    'total_likes': overview.get('total_engagement', 0),
                    'total_comments': overview.get('total_comments', 0),
                    'avg_engagement': overview.get('avg_engagement_score', 0),
                    'best_topic': overview.get('best_performing_topic', 'N/A'),
                    'best_score': overview.get('best_score', 0),
                }
                
                message = MessageFormatter.statistics(stats)
                
                keyboard = [
                    [
                        InlineKeyboardButton("📊 Topic Performance", callback_data="topic_performance"),
                        InlineKeyboardButton("🔄 Refresh", callback_data="stats_refresh"),
                    ],
                    [
                        InlineKeyboardButton("📈 Weekly Report", callback_data="weekly_report"),
                        InlineKeyboardButton("📉 Monthly Report", callback_data="monthly_report"),
                    ],
                    [
                        InlineKeyboardButton("⬅️ Back", callback_data="menu_back"),
                    ],
                ]
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                query = update.callback_query
                if query:
                    await query.edit_message_text(message, reply_markup=reply_markup, parse_mode="HTML")
                else:
                    await update.message.reply_text(message, reply_markup=reply_markup, parse_mode="HTML")
            else:
                await update.callback_query.edit_message_text("❌ Failed to fetch statistics")
    
    except Exception as e:
        logger.error(f"Error showing stats: {e}")
        query = update.callback_query
        if query:
            await query.edit_message_text(f"❌ Error: {str(e)}")


async def show_topic_performance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Topic performance দেখান"""
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{Config.API_URL}/dashboard")
            
            if response.status_code == 200:
                dashboard_data = response.json()
                topics = dashboard_data.get("overview", {}).get("topic_breakdown", [])
                
                message = MessageFormatter.topic_performance(topics)
                
                keyboard = [
                    [
                        InlineKeyboardButton("🔄 Refresh", callback_data="topic_performance"),
                        InlineKeyboardButton("📊 All Stats", callback_data="menu_stats"),
                    ],
                    [
                        InlineKeyboardButton("⬅️ Back", callback_data="menu_back"),
                    ],
                ]
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                query = update.callback_query
                await query.edit_message_text(message, reply_markup=reply_markup, parse_mode="HTML")
            else:
                query = update.callback_query
                await query.edit_message_text("❌ Failed to fetch topic performance")
    
    except Exception as e:
        logger.error(f"Error showing topic performance: {e}")
        query = update.callback_query
        await query.edit_message_text(f"❌ Error: {str(e)}")

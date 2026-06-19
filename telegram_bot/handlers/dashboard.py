"""
Dashboard Handler - ড্যাশবোর্ড এবং overview দেখান
"""

import logging
import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from telegram_bot.config import Config
from telegram_bot.services.formatter import MessageFormatter
from telegram_bot.services.telegram_publisher import TelegramPublisher

logger = logging.getLogger(__name__)


async def show_dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """ড্যাশবোর্ড দেখান"""
    
    try:
        # Backend API থেকে ডেটা fetch করুন
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{Config.API_URL}/dashboard")
            
            if response.status_code == 200:
                dashboard_data = response.json()
                
                overview = dashboard_data.get("overview", {}).get("overview", {})
                system_status = dashboard_data.get("system_status", {})
                next_run = dashboard_data.get("scheduler", {}).get("next_posts", [{}])[0].get("next_run")
                
                # মেসেজ তৈরি করুন
                message = MessageFormatter.dashboard_summary(overview, system_status, next_run)
                
                # বাটন তৈরি করুন
                keyboard = [
                    [
                        InlineKeyboardButton("🔄 Refresh", callback_data="dashboard_refresh"),
                        InlineKeyboardButton("📈 More Stats", callback_data="menu_stats"),
                    ],
                    [
                        InlineKeyboardButton("🤖 Generate", callback_data="menu_generate"),
                        InlineKeyboardButton("📅 Schedule", callback_data="menu_schedule"),
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
                error_msg = "❌ Failed to fetch dashboard data"
                query = update.callback_query
                if query:
                    await query.edit_message_text(error_msg, parse_mode="HTML")
                else:
                    await update.message.reply_text(error_msg, parse_mode="HTML")
    
    except Exception as e:
        logger.error(f"Error showing dashboard: {e}")
        error_msg = MessageFormatter.error_message(str(e), "Failed to load dashboard")
        
        query = update.callback_query
        if query:
            await query.edit_message_text(error_msg, parse_mode="HTML")
        else:
            await update.message.reply_text(error_msg, parse_mode="HTML")


async def handle_dashboard_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """ড্যাশবোর্ড মেনু হ্যান্ডেল করুন"""
    
    query = update.callback_query
    
    if query.data == "dashboard_refresh":
        await show_dashboard(update, context)
    elif query.data == "menu_dashboard":
        await show_dashboard(update, context)

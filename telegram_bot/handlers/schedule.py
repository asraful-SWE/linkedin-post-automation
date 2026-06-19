"""
Schedule Handler - পোস্ট শিডিউল ম্যানেজ করুন
"""

import logging
import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from telegram_bot.config import Config
from telegram_bot.services.formatter import MessageFormatter

logger = logging.getLogger(__name__)


async def show_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """পরবর্তী শিডিউল করা পোস্ট দেখান"""
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{Config.API_URL}/dashboard")
            
            if response.status_code == 200:
                dashboard_data = response.json()
                next_posts = dashboard_data.get("scheduler", {}).get("next_posts", [])
                scheduler_status = dashboard_data.get("scheduler", {}).get("status", "unknown")
                
                message = MessageFormatter.schedule_info(next_posts)
                message += f"\n\n⏰ <b>Scheduler Status:</b> {scheduler_status}\n"
                
                keyboard = [
                    [
                        InlineKeyboardButton(
                            "⏸️ Pause" if scheduler_status == "running" else "▶️ Resume",
                            callback_data=f"toggle_scheduler_{scheduler_status}"
                        ),
                        InlineKeyboardButton("🔄 Refresh", callback_data="schedule_refresh"),
                    ],
                    [
                        InlineKeyboardButton("📅 Set Schedule", callback_data="set_schedule"),
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
                query = update.callback_query
                await query.edit_message_text("❌ Failed to fetch schedule")
    
    except Exception as e:
        logger.error(f"Error showing schedule: {e}")
        query = update.callback_query
        if query:
            await query.edit_message_text(f"❌ Error: {str(e)}")


async def pause_scheduler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scheduler pause করুন"""
    
    query = update.callback_query
    await query.answer()
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(f"{Config.API_URL}/scheduler/pause")
            
            if response.status_code == 200:
                await query.edit_message_text("⏸️ <b>Scheduler Paused</b>\n\nAuto-posting is now disabled.", parse_mode="HTML")
            else:
                await query.edit_message_text("❌ Failed to pause scheduler")
    
    except Exception as e:
        logger.error(f"Error pausing scheduler: {e}")
        await query.edit_message_text(f"❌ Error: {str(e)}")


async def resume_scheduler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scheduler resume করুন"""
    
    query = update.callback_query
    await query.answer()
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(f"{Config.API_URL}/scheduler/resume")
            
            if response.status_code == 200:
                await query.edit_message_text("▶️ <b>Scheduler Resumed</b>\n\nAuto-posting is now enabled.", parse_mode="HTML")
            else:
                await query.edit_message_text("❌ Failed to resume scheduler")
    
    except Exception as e:
        logger.error(f"Error resuming scheduler: {e}")
        await query.edit_message_text(f"❌ Error: {str(e)}")

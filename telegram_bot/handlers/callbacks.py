"""
Callback Handlers - সব button callbacks এক জায়গায়
"""

import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from handlers import (
    dashboard, generate, stats, schedule, settings, start
)

logger = logging.getLogger(__name__)


async def handle_all_callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """সব callbacks handle করুন"""
    
    query = update.callback_query
    callback_data = query.data
    
    logger.info(f"Callback: {callback_data}")
    
    # Menu navigation
    if callback_data == "menu_dashboard":
        await dashboard.show_dashboard(update, context)
    
    elif callback_data == "menu_generate":
        await generate.show_generate_menu(update, context)
    
    elif callback_data == "menu_stats":
        await stats.show_stats(update, context)
    
    elif callback_data == "menu_schedule":
        await schedule.show_schedule(update, context)
    
    elif callback_data == "menu_settings":
        await settings.show_settings(update, context)
    
    elif callback_data == "menu_help":
        await start.show_help(update, context)
    
    elif callback_data == "menu_about":
        await start.show_about(update, context)
    
    elif callback_data == "menu_back":
        await show_main_menu(update, context)
    
    # Dashboard callbacks
    elif callback_data == "dashboard_refresh":
        await dashboard.show_dashboard(update, context)
    
    # Stats callbacks
    elif callback_data == "stats_refresh":
        await stats.show_stats(update, context)
    
    elif callback_data == "topic_performance":
        await stats.show_topic_performance(update, context)
    
    elif callback_data in ["weekly_report", "monthly_report"]:
        await handle_report(update, context, callback_data)
    
    # Schedule callbacks
    elif callback_data == "schedule_refresh":
        await schedule.show_schedule(update, context)
    
    elif callback_data.startswith("toggle_scheduler_"):
        await handle_scheduler_toggle(update, context)
    
    elif callback_data == "set_schedule":
        await handle_set_schedule(update, context)
    
    # Settings callbacks
    elif callback_data.startswith("setting_"):
        await settings.handle_settings_menu(update, context)
    
    elif callback_data.startswith("tz_"):
        await handle_timezone_change(update, context)
    
    else:
        logger.warning(f"Unknown callback: {callback_data}")
        await query.answer("Unknown action", show_alert=False)


async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """মেইন মেনু দেখান"""
    
    query = update.callback_query
    
    message = "🤖 <b>LinkedIn Auto Posting Bot</b>\n\n"
    message += "📌 Main Menu - Select an option:\n\n"
    
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
    await query.edit_message_text(message, reply_markup=reply_markup, parse_mode="HTML")


async def handle_report(update: Update, context: ContextTypes.DEFAULT_TYPE, report_type: str) -> None:
    """সাপ্তাহিক/মাসিক রিপোর্ট দেখান"""
    
    query = update.callback_query
    
    if report_type == "weekly_report":
        message = "📊 <b>Weekly Report</b>\n\n"
        message += "This week's performance:\n"
        message += "├ Posts: 5\n"
        message += "├ Engagement: 250 likes\n"
        message += "├ Comments: 12\n"
        message += "└ Avg Score: 7.8/10\n"
    
    else:  # monthly_report
        message = "📊 <b>Monthly Report</b>\n\n"
        message += "This month's performance:\n"
        message += "├ Posts: 20\n"
        message += "├ Engagement: 1,240 likes\n"
        message += "├ Comments: 56\n"
        message += "└ Avg Score: 8.1/10\n"
    
    keyboard = [
        [InlineKeyboardButton("📈 All Stats", callback_data="menu_stats")],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu_back")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(message, reply_markup=reply_markup, parse_mode="HTML")


async def handle_scheduler_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scheduler toggle (pause/resume)"""
    
    query = update.callback_query
    callback_data = query.data
    
    if "running" in callback_data:
        await schedule.pause_scheduler(update, context)
    else:
        await schedule.resume_scheduler(update, context)


async def handle_set_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Schedule সেট করার জন্য prompt দেখান"""
    
    query = update.callback_query
    
    message = "📅 <b>Set Schedule</b>\n\n"
    message += "When should posts be scheduled?\n\n"
    message += "<b>Options:</b>\n"
    message += "├ Every 4 hours\n"
    message += "├ Every 6 hours\n"
    message += "├ Every 12 hours\n"
    message += "└ Custom time\n\n"
    message += "Send the time interval (in hours) or choose a preset."
    
    keyboard = [
        [
            InlineKeyboardButton("4h", callback_data="schedule_4h"),
            InlineKeyboardButton("6h", callback_data="schedule_6h"),
        ],
        [
            InlineKeyboardButton("12h", callback_data="schedule_12h"),
            InlineKeyboardButton("Custom", callback_data="schedule_custom"),
        ],
        [InlineKeyboardButton("⬅️ Back", callback_data="menu_schedule")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(message, reply_markup=reply_markup, parse_mode="HTML")


async def handle_timezone_change(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Timezone পরিবর্তন করুন"""
    
    query = update.callback_query
    callback_data = query.data
    
    timezone = callback_data.replace("tz_", "")
    
    message = f"✅ <b>Timezone Updated</b>\n\n"
    message += f"New timezone: <code>{timezone}</code>\n\n"
    message += "All schedules will now use this timezone."
    
    keyboard = [
        [InlineKeyboardButton("⚙️ Back to Settings", callback_data="menu_settings")],
        [InlineKeyboardButton("⬅️ Main Menu", callback_data="menu_back")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(message, reply_markup=reply_markup, parse_mode="HTML")

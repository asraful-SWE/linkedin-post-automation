"""
Settings Handler - সেটিংস কনফিগার করুন
"""

import logging
import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from telegram_bot.config import Config
from telegram_bot.services.formatter import MessageFormatter

logger = logging.getLogger(__name__)


async def show_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """সেটিংস মেনু দেখান"""
    
    message = """
⚙️ <b>Settings</b>

Configure your LinkedIn Auto Posting system:

<b>Current Configuration:</b>
├ Max Posts/Day: <code>2</code>
├ Min Hours Between Posts: <code>4</code>
├ Timezone: <code>Asia/Dhaka</code>
├ Auto Schedule: <code>Enabled</code>
└ Test Mode: <code>Disabled</code>

Select an option to modify:
"""
    
    keyboard = [
        [
            InlineKeyboardButton("📝 Max Posts/Day", callback_data="setting_max_posts"),
            InlineKeyboardButton("⏱️ Min Hours", callback_data="setting_min_hours"),
        ],
        [
            InlineKeyboardButton("🌍 Timezone", callback_data="setting_timezone"),
            InlineKeyboardButton("🔑 API Keys", callback_data="setting_api_keys"),
        ],
        [
            InlineKeyboardButton("🧪 Test Mode", callback_data="setting_test_mode"),
            InlineKeyboardButton("🔄 Reset", callback_data="setting_reset"),
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


async def handle_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """সেটিংস মেনু অপশন হ্যান্ডেল করুন"""
    
    query = update.callback_query
    callback_data = query.data
    
    if callback_data == "setting_max_posts":
        await query.edit_message_text(
            "📝 <b>Max Posts Per Day</b>\n\n"
            "Current: 2\n\n"
            "Send a number between 1-5",
            parse_mode="HTML"
        )
    
    elif callback_data == "setting_min_hours":
        await query.edit_message_text(
            "⏱️ <b>Min Hours Between Posts</b>\n\n"
            "Current: 4\n\n"
            "Send a number between 1-12",
            parse_mode="HTML"
        )
    
    elif callback_data == "setting_timezone":
        message = "🌍 <b>Select Timezone:</b>\n\n"
        
        timezones = [
            "Asia/Dhaka",
            "Asia/Kolkata",
            "Asia/Bangkok",
            "Asia/Manila",
            "Europe/London",
            "America/New_York",
            "America/Los_Angeles",
        ]
        
        keyboard = []
        for tz in timezones:
            keyboard.append([InlineKeyboardButton(tz, callback_data=f"tz_{tz}")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode="HTML")
    
    elif callback_data == "setting_api_keys":
        await query.edit_message_text(
            "🔑 <b>API Keys</b>\n\n"
            "You can update your API keys here.\n\n"
            "Available keys:\n"
            "• OpenAI Key\n"
            "• LinkedIn Token\n\n"
            "Send the key name to update",
            parse_mode="HTML"
        )
    
    elif callback_data == "setting_test_mode":
        await query.edit_message_text(
            "🧪 <b>Test Mode</b>\n\n"
            "Current: Disabled\n\n"
            "[✅ Enable] [❌ Keep Disabled]",
            parse_mode="HTML"
        )
    
    elif callback_data == "setting_reset":
        await query.edit_message_text(
            "🔄 <b>Reset Settings</b>\n\n"
            "Are you sure? This will reset all settings to defaults.\n\n"
            "[✅ Confirm Reset] [❌ Cancel]",
            parse_mode="HTML"
        )

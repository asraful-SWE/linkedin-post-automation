"""
Telegram Bot - Main Entry Point
LinkedIn Auto Posting Bot for Telegram
"""

import logging
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from dotenv import load_dotenv

# Import handlers
from handlers.start import start, show_help, show_about
from handlers.dashboard import show_dashboard, handle_dashboard_menu
from handlers.generate import show_generate_menu, handle_topic_selection, handle_generate_confirmation, SELECT_TOPIC, CONFIRM_PUBLISH
from handlers.stats import show_stats, show_topic_performance
from handlers.schedule import show_schedule, pause_scheduler, resume_scheduler
from handlers.settings import show_settings, handle_settings_menu

# Import config
from config import Config

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class LinkedInTelegramBot:
    """Main Telegram Bot class"""
    
    def __init__(self):
        self.application = None
    
    def setup_handlers(self):
        """সব handlers setup করুন"""
        
        # /start command
        self.application.add_handler(CommandHandler("start", start))
        self.application.add_handler(CommandHandler("help", show_help))
        self.application.add_handler(CommandHandler("about", show_about))
        
        # Quick commands
        self.application.add_handler(CommandHandler("dashboard", show_dashboard))
        self.application.add_handler(CommandHandler("stats", show_stats))
        self.application.add_handler(CommandHandler("schedule", show_schedule))
        self.application.add_handler(CommandHandler("settings", show_settings))
        
        # Generate conversation handler
        generate_conversation = ConversationHandler(
            entry_points=[
                CommandHandler("generate", show_generate_menu),
                CallbackQueryHandler(show_generate_menu, pattern="^menu_generate$")
            ],
            states={
                SELECT_TOPIC: [
                    CallbackQueryHandler(handle_topic_selection, pattern="^topic_"),
                ],
                CONFIRM_PUBLISH: [
                    CallbackQueryHandler(handle_generate_confirmation, pattern="^(publish_|menu_generate|topic_)"),
                ],
            },
            fallbacks=[
                CallbackQueryHandler(self.handle_menu_back, pattern="^menu_back$")
            ],
        )
        self.application.add_handler(generate_conversation)
        
        # Callback queries for menus
        self.application.add_handler(CallbackQueryHandler(
            self.handle_main_menu,
            pattern="^menu_(dashboard|stats|schedule|settings)$"
        ))
        
        # Dashboard callbacks
        self.application.add_handler(CallbackQueryHandler(
            show_dashboard,
            pattern="^(menu_dashboard|dashboard_refresh)$"
        ))
        
        # Stats callbacks
        self.application.add_handler(CallbackQueryHandler(
            show_stats,
            pattern="^(menu_stats|stats_refresh)$"
        ))
        
        self.application.add_handler(CallbackQueryHandler(
            show_topic_performance,
            pattern="^topic_performance$"
        ))
        
        # Schedule callbacks
        self.application.add_handler(CallbackQueryHandler(
            show_schedule,
            pattern="^(menu_schedule|schedule_refresh)$"
        ))
        
        self.application.add_handler(CallbackQueryHandler(
            self.toggle_scheduler,
            pattern="^toggle_scheduler_"
        ))
        
        # Settings callbacks
        self.application.add_handler(CallbackQueryHandler(
            show_settings,
            pattern="^menu_settings$"
        ))
        
        self.application.add_handler(CallbackQueryHandler(
            handle_settings_menu,
            pattern="^setting_"
        ))
        
        # Back to main menu
        self.application.add_handler(CallbackQueryHandler(
            self.handle_menu_back,
            pattern="^menu_back$"
        ))
        
        # Help callback
        self.application.add_handler(CallbackQueryHandler(
            show_help,
            pattern="^menu_help$"
        ))
        
        logger.info("✅ All handlers registered")
    
    async def handle_main_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """মূল মেনু অপশন হ্যান্ডেল করুন"""
        
        query = update.callback_query
        callback_data = query.data
        
        if callback_data == "menu_dashboard":
            await show_dashboard(update, context)
        elif callback_data == "menu_stats":
            await show_stats(update, context)
        elif callback_data == "menu_schedule":
            await show_schedule(update, context)
        elif callback_data == "menu_settings":
            await show_settings(update, context)
    
    async def handle_menu_back(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """মূল মেনুতে ফিরে যান"""
        
        query = update.callback_query
        
        # Reshow start
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
        message = "🤖 <b>LinkedIn Auto Posting Bot</b>\n\nMain Menu - Select an option:"
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode="HTML")
    
    async def toggle_scheduler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Scheduler toggle করুন (pause/resume)"""
        
        query = update.callback_query
        callback_data = query.data
        
        if "running" in callback_data:
            await pause_scheduler(update, context)
        else:
            await resume_scheduler(update, context)
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Error handle করুন"""
        
        logger.error(f"Exception while handling an update: {context.error}")
        
        if update and update.callback_query:
            try:
                await update.callback_query.answer(
                    "❌ An error occurred. Please try again.",
                    show_alert=True
                )
            except:
                pass
    
    def run(self):
        """Bot চালান"""
        
        logger.info("🚀 Starting LinkedIn Auto Posting Telegram Bot...")
        logger.info(f"Bot Token: {Config.BOT_TOKEN[:20]}...")
        logger.info(f"Admin ID: {Config.ADMIN_ID}")
        
        # Create application
        self.application = Application.builder().token(Config.BOT_TOKEN).build()
        
        # Setup handlers
        self.setup_handlers()
        
        # Add error handler
        self.application.add_error_handler(self.error_handler)
        
        # Start bot
        logger.info("✅ Bot started successfully!")
        logger.info("Listening for messages...")
        
        self.application.run_polling()


def main():
    """Main entry point"""
    
    try:
        bot = LinkedInTelegramBot()
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise


if __name__ == "__main__":
    main()

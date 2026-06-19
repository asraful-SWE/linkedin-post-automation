"""
Generate Handler - পোস্ট জেনারেট এবং প্রকাশ করুন
"""

import logging
import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ConversationHandler

from telegram_bot.config import Config
from telegram_bot.services.formatter import MessageFormatter

logger = logging.getLogger(__name__)

# Conversation states
SELECT_TOPIC, CONFIRM_PUBLISH = range(2)


async def show_generate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Generate মেনু দেখান এবং topic সিলেক্ট করতে বলুন"""
    
    try:
        # Backend থেকে topics fetch করুন
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{Config.API_URL}/topics")
            
            if response.status_code == 200:
                topics = response.json().get("topics", [])[:10]  # top 10
                
                # Store topics in context for later retrieval
                context.user_data['topics'] = topics
                
                message = "🤖 <b>Generate Post</b>\n\n"
                message += "Select a topic to generate a post:\n\n"
                
                # Topic buttons তৈরি করুন (use index to keep callback_data short)
                keyboard = []
                for i in range(0, len(topics), 2):
                    row = []
                    for j in range(2):
                        if i + j < len(topics):
                            topic = topics[i + j]
                            # Use index instead of full topic name to avoid 64-byte limit
                            idx = i + j
                            row.append(InlineKeyboardButton(
                                f"📌 {topic[:20]}...",
                                callback_data=f"topic_{idx}"
                            ))
                    if row:
                        keyboard.append(row)
                
                # Back button
                keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="menu_back")])
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                query = update.callback_query
                if query:
                    await query.edit_message_text(message, reply_markup=reply_markup, parse_mode="HTML")
                else:
                    await update.message.reply_text(message, reply_markup=reply_markup, parse_mode="HTML")
                
                return SELECT_TOPIC
            else:
                if update.callback_query:
                    await update.callback_query.edit_message_text("❌ Failed to fetch topics")
                else:
                    await update.message.reply_text("❌ Failed to fetch topics")
                return ConversationHandler.END
    
    except Exception as e:
        logger.error(f"Error in show_generate_menu: {e}")
        if update.callback_query:
            await update.callback_query.edit_message_text(f"❌ Error: {str(e)}")
        else:
            await update.message.reply_text(f"❌ Error: {str(e)}")
        return ConversationHandler.END


async def handle_topic_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Topic সিলেক্ট হলে পোস্ট জেনারেট করুন"""
    
    query = update.callback_query
    await query.answer()
    
    # Get topic index from callback_data and retrieve from context
    callback_data = query.data
    topic_idx = int(callback_data.replace("topic_", ""))
    
    # Get topic from stored list
    topics = context.user_data.get('topics', [])
    if topic_idx >= len(topics):
        await query.edit_message_text("❌ Invalid topic selection")
        return ConversationHandler.END
    
    topic = topics[topic_idx]
    
    # Store the topic for later use
    context.user_data['selected_topic'] = topic
    
    # Generating মেসেজ দেখান
    await query.edit_message_text(f"✨ Generating and publishing post for: <b>{topic}</b>...\n\n⏳ Please wait...", parse_mode="HTML")
    
    try:
        # Backend এ generate request পাঠান (generate-post endpoint generates AND publishes)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{Config.API_URL}/generate-post",
                json={"topic": topic}
            )
            
            if response.status_code == 200:
                post_data = response.json()
                
                # Check if successful
                if post_data.get("success", False):
                    post_id = post_data.get("post_id", "")
                    linkedin_url = post_data.get("linkedin_url", "")
                    status = post_data.get("status", "pending")
                    
                    message = f"✅ <b>Post Generated!</b>\n\n"
                    message += f"📌 <b>Topic:</b> {topic}\n"
                    message += f"📊 <b>Status:</b> {status}\n"
                    message += f"🆔 <b>Post ID:</b> {post_id}\n"
                    
                    if linkedin_url:
                        message += f"\n<a href='{linkedin_url}'>View on LinkedIn →</a>"
                    
                    keyboard = [
                        [
                            InlineKeyboardButton("🤖 Generate Another", callback_data="menu_generate"),
                            InlineKeyboardButton("📊 Dashboard", callback_data="menu_dashboard"),
                        ],
                    ]
                    
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await query.edit_message_text(message, reply_markup=reply_markup, parse_mode="HTML")
                else:
                    error_msg = post_data.get("message", "Unknown error")
                    await query.edit_message_text(f"❌ Failed:\n{error_msg[:100]}", parse_mode="HTML")
                
                return ConversationHandler.END
            else:
                error_msg = response.text if response.text else f"HTTP {response.status_code}"
                logger.error(f"Generate endpoint returned {response.status_code}: {error_msg}")
                await query.edit_message_text(f"❌ Error:\n{error_msg[:100]}", parse_mode="HTML")
                return ConversationHandler.END
    
    except Exception as e:
        logger.error(f"Error generating post: {e}")
        await query.edit_message_text(f"❌ Error: {str(e)[:100]}", parse_mode="HTML")
        return ConversationHandler.END


async def handle_generate_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """পোস্ট confirm করে publish করুন"""
    
    query = update.callback_query
    await query.answer()
    
    callback_data = query.data
    
    if callback_data.startswith("publish_"):
        # Get topic from context (not from callback_data which is now just an index)
        topic = context.user_data.get('generated_topic', '')
        content = context.user_data.get('generated_content', '')
        
        if not topic or not content:
            await query.edit_message_text("❌ Session expired. Please try again.")
            return ConversationHandler.END
        
        # Publish করার মেসেজ
        await query.edit_message_text("🚀 Publishing post...\n\n⏳ Please wait...", parse_mode="HTML")
        
        try:
            # Backend এ publish request পাঠান
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{Config.API_URL}/publish",
                    json={
                        "topic": topic,
                        "content": content
                    }
                )
                
                if response.status_code == 200:
                    post_data = response.json()
                    linkedin_url = post_data.get("linkedin_url", "")
                    
                    message = f"✅ <b>Post Published Successfully!</b>\n\n"
                    message += f"📌 <b>Topic:</b> {topic}\n"
                    message += f"📊 <b>Status:</b> Live\n"
                    
                    if linkedin_url:
                        message += f"\n<a href='{linkedin_url}'>View on LinkedIn →</a>"
                    
                    keyboard = [
                        [
                            InlineKeyboardButton("🤖 Generate Another", callback_data="menu_generate"),
                            InlineKeyboardButton("📊 Back to Dashboard", callback_data="menu_dashboard"),
                        ],
                    ]
                    
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await query.edit_message_text(message, reply_markup=reply_markup, parse_mode="HTML")
                else:
                    error_msg = response.text if response.text else "Unknown error"
                    logger.error(f"Publish failed: {error_msg}")
                    await query.edit_message_text(f"❌ Failed to publish post:\n{error_msg[:100]}", parse_mode="HTML")
        
        except Exception as e:
            logger.error(f"Error publishing post: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)[:100]}", parse_mode="HTML")
        
        return ConversationHandler.END
    
    return CONFIRM_PUBLISH

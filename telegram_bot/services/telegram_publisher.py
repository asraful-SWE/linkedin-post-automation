"""
Telegram Publisher - Telegram channel এ পোস্ট করুন
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramPublisher:
    """Telegram এ পোস্ট publish করার জন্য service"""
    
    def __init__(self):
        pass
    
    @staticmethod
    async def send_message(bot, chat_id: int, message: str, parse_mode: str = "HTML", 
                          reply_markup=None) -> Optional[int]:
        """Telegram এ মেসেজ পাঠান"""
        
        try:
            msg = await bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                disable_web_page_preview=False
            )
            logger.info(f"Message sent to {chat_id}")
            return msg.message_id
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return None
    
    @staticmethod
    async def send_photo(bot, chat_id: int, photo_url: str, caption: str = "",
                        parse_mode: str = "HTML", reply_markup=None) -> Optional[int]:
        """Telegram এ ছবি সহ মেসেজ পাঠান"""
        
        try:
            msg = await bot.send_photo(
                chat_id=chat_id,
                photo=photo_url,
                caption=caption,
                parse_mode=parse_mode,
                reply_markup=reply_markup
            )
            logger.info(f"Photo sent to {chat_id}")
            return msg.message_id
        except Exception as e:
            logger.error(f"Failed to send photo: {e}")
            return None
    
    @staticmethod
    async def edit_message(bot, chat_id: int, message_id: int, text: str,
                          parse_mode: str = "HTML", reply_markup=None) -> bool:
        """Telegram এ মেসেজ edit করুন"""
        
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup
            )
            logger.info(f"Message {message_id} edited in {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to edit message: {e}")
            return False
    
    @staticmethod
    async def delete_message(bot, chat_id: int, message_id: int) -> bool:
        """Telegram এ মেসেজ delete করুন"""
        
        try:
            await bot.delete_message(chat_id=chat_id, message_id=message_id)
            logger.info(f"Message {message_id} deleted from {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete message: {e}")
            return False
    
    @staticmethod
    def create_keyboard(buttons: list, columns: int = 2) -> 'InlineKeyboardMarkup':
        """Inline keyboard তৈরি করুন"""
        
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        
        keyboard = []
        for i in range(0, len(buttons), columns):
            keyboard.append(buttons[i:i + columns])
        
        return InlineKeyboardMarkup(keyboard)

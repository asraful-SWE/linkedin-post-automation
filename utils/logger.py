"""
Enhanced logging configuration for LinkedIn Auto Poster
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def setup_logging():
    """Setup comprehensive logging configuration"""
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure log level from environment
    log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Define log format
    detailed_format = logging.Formatter(
        '%(asctime)s | %(name)-20s | %(levelname)-8s | %(funcName)-15s | %(message)s'
    )
    
    simple_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'  
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if not debug_mode else logging.DEBUG)
    console_handler.setFormatter(simple_format if not debug_mode else detailed_format)
    logger.addHandler(console_handler)
    
    # File handler for general logs
    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, "linkedin_poster.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_format)
    logger.addHandler(file_handler)
    
    # Separate error log file
    error_handler = RotatingFileHandler(
        os.path.join(logs_dir, "errors.log"),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_format)
    logger.addHandler(error_handler)
    
    # Debug file handler (only in debug mode)
    if debug_mode:
        debug_handler = RotatingFileHandler(
            os.path.join(logs_dir, "debug.log"),
            maxBytes=20*1024*1024,  # 20MB
            backupCount=2
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(detailed_format)
        logger.addHandler(debug_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    
    # Log startup info
    startup_logger = logging.getLogger("startup")
    startup_logger.info("Logging system initialized")
    startup_logger.info(f"Log level: {logging.getLevelName(log_level)}")
    startup_logger.info(f"Debug mode: {debug_mode}")
    startup_logger.info(f"Log directory: {os.path.abspath(logs_dir)}")


def get_logger(name: str) -> logging.Logger:
    """Get a named logger with proper configuration"""
    return logging.getLogger(name)


class ContextualLogger:
    """Logger that adds contextual information to log messages"""
    
    def __init__(self, name: str, context: dict = None):
        self.logger = logging.getLogger(name)
        self.context = context or {}
    
    def _format_message(self, message: str) -> str:
        """Add context to log message"""
        if self.context:
            context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            return f"[{context_str}] {message}"
        return message
    
    def debug(self, message: str):
        self.logger.debug(self._format_message(message))
    
    def info(self, message: str):
        self.logger.info(self._format_message(message))
    
    def warning(self, message: str):
        self.logger.warning(self._format_message(message))
    
    def error(self, message: str):
        self.logger.error(self._format_message(message))
    
    def critical(self, message: str):
        self.logger.critical(self._format_message(message))
    
    def add_context(self, **kwargs):
        """Add context information"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context"""
        self.context.clear()

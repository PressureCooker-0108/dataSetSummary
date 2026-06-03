import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from config.settings import settings

def setup_logging() -> None:
    """Configures global logging settings with console and rotating file handlers."""
    # Ensure logs directory exists
    log_file = Path(settings.LOG_FILE_PATH)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if root logger already has handlers configured to avoid duplicates
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
        
    # Get configuration log level
    log_level_name = settings.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    root_logger.setLevel(log_level)
    
    # Format definition
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Rotating File Handler
    try:
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to setup file logging: {e}", file=sys.stderr)

def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the given name, ensuring setup is run."""
    setup_logging()
    return logging.getLogger(name)

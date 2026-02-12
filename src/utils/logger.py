import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# Directory where logs will be stored
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Format: [Date Time] [Log Level] [Module:Line] - Message
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(logger_name, log_file_name="application.log"):
    """
    Returns a professional logger instance with both rotating file and console handlers.
    Ensures no duplicate logs if called multiple times.
    """

    # 1. Get the logger object (Use named logger to isolate from root/external libs)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 2. Prevent duplicate logs: Check if handlers already exist
    # If we don't check this, logs will be duplicated every time this function is called.
    if logger.hasHandlers():
        return logger

    # 3. File Handler (Rotating)
    # Rotates after 10MB, keeps 5 backup files to save disk space.
    log_file_path = os.path.join(LOG_DIR, log_file_name)
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    file_handler.setLevel(logging.INFO)

    # 4. Console Handler (Stream to stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    console_handler.setLevel(logging.INFO)

    # 5. Add Handlers to Logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logger(name: str, log_file: str = None, level: str = "INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Format untuk logging
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler untuk console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler untuk file (jika ada path file)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Logger global
app_logger = setup_logger("indonesian_rag")
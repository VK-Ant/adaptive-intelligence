"""Shared utilities for Adaptive Intelligence."""

import uuid
import logging
import time
from typing import Optional
from functools import wraps


def generate_query_id() -> str:
    """Generate a unique query ID."""
    return uuid.uuid4().hex[:12]


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging for the library."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Root logger for the library
    lib_logger = logging.getLogger("adaptive_intelligence")
    lib_logger.setLevel(log_level)

    # Console handler
    if not lib_logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        lib_logger.addHandler(console)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        lib_logger.addHandler(file_handler)

    return lib_logger


def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.getLogger("adaptive_intelligence").debug(
            f"{func.__name__} took {elapsed:.3f}s"
        )
        return result
    return wrapper


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."

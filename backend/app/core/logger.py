# Logging utilities

import logging
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: The name for the logger, typically __name__

    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)
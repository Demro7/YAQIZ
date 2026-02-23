"""
YAQIZ Structured Logger
"""

import logging
import sys


def setup_logging(level: str = "INFO"):
    """Configure structured logging for YAQIZ"""
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Root YAQIZ logger
    logger = logging.getLogger("yaqiz")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(handler)

    # Suppress noisy loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logger

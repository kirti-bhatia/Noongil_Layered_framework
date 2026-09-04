"""
============================================================
NOONGIL-X
LLM Logger
============================================================

Author : NOONGIL-X

Purpose:
Central logging utility for the LLM subsystem.

Features
--------
✓ Console logging
✓ File logging
✓ Timestamped logs
✓ Automatic log directory creation
✓ Singleton logger
✓ Colored console output
✓ Thread-safe
✓ Reusable across all layers
============================================================
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    from colorama import Fore, Style, init

    init(autoreset=True)
    COLOR_ENABLED = True
except ImportError:
    COLOR_ENABLED = False

from .config import LOG_DIR


# ============================================================
# Ensure Log Directory Exists
# ============================================================

LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Colored Formatter
# ============================================================

class ColoredFormatter(logging.Formatter):

    COLORS = {
        logging.DEBUG: Fore.CYAN if COLOR_ENABLED else "",
        logging.INFO: Fore.GREEN if COLOR_ENABLED else "",
        logging.WARNING: Fore.YELLOW if COLOR_ENABLED else "",
        logging.ERROR: Fore.RED if COLOR_ENABLED else "",
        logging.CRITICAL: Fore.MAGENTA if COLOR_ENABLED else "",
    }

    RESET = Style.RESET_ALL if COLOR_ENABLED else ""

    def format(self, record):

        message = super().format(record)

        color = self.COLORS.get(record.levelno, "")

        return f"{color}{message}{self.RESET}"


# ============================================================
# Logger Factory
# ============================================================

class LLMLogger:

    """
    Singleton Logger Manager
    """

    _loggers = {}

    @staticmethod
    def get_logger(name: str = "NOONGIL-LLM") -> logging.Logger:

        if name in LLMLogger._loggers:
            return LLMLogger._loggers[name]

        logger = logging.getLogger(name)

        logger.setLevel(logging.DEBUG)

        logger.propagate = False

        # Prevent duplicate handlers
        if logger.handlers:
            return logger

        # ----------------------------------------------------
        # Log File
        # ----------------------------------------------------

        filename = datetime.now().strftime("%Y-%m-%d") + ".log"

        logfile = LOG_DIR / filename

        file_handler = logging.FileHandler(
            logfile,
            encoding="utf-8"
        )

        file_handler.setLevel(logging.DEBUG)

        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler.setFormatter(file_formatter)

        # ----------------------------------------------------
        # Console
        # ----------------------------------------------------

        console = logging.StreamHandler(sys.stdout)

        console.setLevel(logging.INFO)

        console_formatter = ColoredFormatter(
            fmt="%(levelname)-8s | %(message)s"
        )

        console.setFormatter(console_formatter)

        # ----------------------------------------------------

        logger.addHandler(file_handler)
        logger.addHandler(console)

        LLMLogger._loggers[name] = logger

        return logger


# ============================================================
# Convenience Functions
# ============================================================

logger = LLMLogger.get_logger()


def debug(message: str):
    logger.debug(message)


def info(message: str):
    logger.info(message)


def warning(message: str):
    logger.warning(message)


def error(message: str):
    logger.error(message)


def critical(message: str):
    logger.critical(message)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    log = LLMLogger.get_logger("LOGGER TEST")

    log.debug("Debug message")

    log.info("Information message")

    log.warning("Warning message")

    log.error("Error message")

    log.critical("Critical message")

    print("\nLogger initialized successfully.")
"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Logging and Execution Timing
File    : layer2/utils/logger.py
============================================================

Purpose
-------
Provides:

- Console logging
- Rotating file logging
- Structured JSON event logging
- Packet and module context
- Exception logging
- Module execution timing
- Duplicate-handler prevention

This logger is independent of Layer 1 logging.

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import math
import time
import traceback
import uuid

from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TextIO


# ============================================================
# CONSTANTS
# ============================================================

LOGGER_VERSION = "1.0"
ROOT_LOGGER_NAME = "noongil.layer2"

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_MAX_LOG_BYTES = 5 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 5

VALID_LOG_LEVELS = {
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
}


# ============================================================
# EXCEPTIONS
# ============================================================

class Layer2LoggerError(Exception):
    """Base exception for Layer 2 logging."""


class LoggerConfigurationError(
    Layer2LoggerError
):
    """Raised when logger configuration is invalid."""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def utc_now_iso() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="milliseconds"
    )


def make_json_safe(
    value: Any,
) -> Any:
    """Convert values to JSON-safe representations."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
        return value

    if isinstance(value, BaseException):
        return {
            "type": value.__class__.__name__,
            "message": str(value),
        }

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def normalize_log_level(
    level: str | int,
) -> int:
    """Convert a level name or number to logging level."""

    if isinstance(level, bool):
        raise LoggerConfigurationError(
            "Log level cannot be boolean."
        )

    if isinstance(level, int):
        if level < 0:
            raise LoggerConfigurationError(
                "Numeric log level cannot be negative."
            )

        return level

    if not isinstance(level, str):
        raise LoggerConfigurationError(
            "Log level must be a string or integer."
        )

    normalized = level.strip().upper()

    if normalized not in VALID_LOG_LEVELS:
        raise LoggerConfigurationError(
            f"Unsupported log level: {level!r}"
        )

    return getattr(logging, normalized)


def build_logger_name(
    module_name: Optional[str],
) -> str:
    """Build a Layer 2 logger name."""

    if module_name is None:
        return ROOT_LOGGER_NAME

    normalized = module_name.strip().replace(
        " ",
        "_",
    )

    if not normalized:
        return ROOT_LOGGER_NAME

    if normalized.startswith(
        ROOT_LOGGER_NAME
    ):
        return normalized

    return f"{ROOT_LOGGER_NAME}.{normalized}"


# ============================================================
# LOG FORMATTERS
# ============================================================

class UTCFormatter(logging.Formatter):
    """Human-readable formatter using UTC timestamps."""

    converter = time.gmtime


class StructuredJSONFormatter(logging.Formatter):
    """Convert one log record to a JSON object."""

    RESERVED_FIELDS = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
    }

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:

        record_message = record.getMessage()

        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                timezone.utc,
            ).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record_message,
            "module": getattr(
                record,
                "layer2_module",
                None,
            ),
            "packet_id": getattr(
                record,
                "packet_id",
                None,
            ),
            "result_id": getattr(
                record,
                "result_id",
                None,
            ),
            "scenario": getattr(
                record,
                "scenario",
                None,
            ),
            "event": getattr(
                record,
                "event",
                None,
            ),
        }

        extra_fields = {}

        for key, value in record.__dict__.items():
            if key in self.RESERVED_FIELDS:
                continue

            if key in {
                "layer2_module",
                "packet_id",
                "result_id",
                "scenario",
                "event",
            }:
                continue

            extra_fields[key] = make_json_safe(
                value
            )

        if extra_fields:
            payload["details"] = extra_fields

        if record.exc_info:
            payload["exception"] = {
                "type": (
                    record.exc_info[0].__name__
                    if record.exc_info[0]
                    else None
                ),
                "message": (
                    str(record.exc_info[1])
                    if record.exc_info[1]
                    else None
                ),
                "traceback": self.formatException(
                    record.exc_info
                ),
            }

        return json.dumps(
            make_json_safe(payload),
            ensure_ascii=False,
            separators=(",", ":"),
        )


# ============================================================
# CONTEXT LOGGER
# ============================================================

class Layer2LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter containing Layer 2 context."""

    def process(
        self,
        msg: Any,
        kwargs: Dict[str, Any],
    ) -> tuple[Any, Dict[str, Any]]:

        supplied_extra = kwargs.get(
            "extra",
            {},
        )

        merged_extra = dict(self.extra)
        merged_extra.update(supplied_extra)

        kwargs["extra"] = merged_extra

        return msg, kwargs

    def bind(
        self,
        **context: Any,
    ) -> "Layer2LoggerAdapter":
        """Create a new adapter with extra context."""

        merged = dict(self.extra)
        merged.update(
            {
                key: value
                for key, value in context.items()
                if value is not None
            }
        )

        return Layer2LoggerAdapter(
            self.logger,
            merged,
        )


# ============================================================
# LOGGER CONFIGURATION
# ============================================================

def configure_logger(
    *,
    module_name: Optional[str] = None,
    level: str | int = DEFAULT_LOG_LEVEL,
    log_file: Optional[Path | str] = None,
    json_log_file: Optional[Path | str] = None,
    enable_console: bool = True,
    max_log_bytes: int = DEFAULT_MAX_LOG_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    force_reconfigure: bool = False,
    stream: Optional[TextIO] = None,
) -> logging.Logger:
    """Configure and return a Layer 2 logger."""

    logger_name = build_logger_name(
        module_name
    )

    logger = logging.getLogger(
        logger_name
    )

    resolved_level = normalize_log_level(
        level
    )

    if (
        isinstance(max_log_bytes, bool)
        or not isinstance(max_log_bytes, int)
        or max_log_bytes <= 0
    ):
        raise LoggerConfigurationError(
            "max_log_bytes must be a "
            "positive integer."
        )

    if (
        isinstance(backup_count, bool)
        or not isinstance(backup_count, int)
        or backup_count < 0
    ):
        raise LoggerConfigurationError(
            "backup_count must be a "
            "non-negative integer."
        )

    if force_reconfigure:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    logger.setLevel(resolved_level)
    logger.propagate = False

    existing_handler_keys = {
        getattr(
            handler,
            "_layer2_handler_key",
            None,
        )
        for handler in logger.handlers
    }

    human_formatter = UTCFormatter(
        fmt=(
            "%(asctime)sZ | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if enable_console:
        console_key = "console"

        if console_key not in existing_handler_keys:
            console_handler = logging.StreamHandler(
                stream
            )

            console_handler.setLevel(
                resolved_level
            )

            console_handler.setFormatter(
                human_formatter
            )

            setattr(
                console_handler,
                "_layer2_handler_key",
                console_key,
            )

            logger.addHandler(
                console_handler
            )

    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_key = (
            f"human_file:{file_path.resolve()}"
        )

        if file_key not in existing_handler_keys:
            file_handler = (
                logging.handlers
                .RotatingFileHandler(
                    file_path,
                    maxBytes=max_log_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
            )

            file_handler.setLevel(
                resolved_level
            )

            file_handler.setFormatter(
                human_formatter
            )

            setattr(
                file_handler,
                "_layer2_handler_key",
                file_key,
            )

            logger.addHandler(
                file_handler
            )

    if json_log_file is not None:
        json_path = Path(json_log_file)
        json_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_key = (
            f"json_file:{json_path.resolve()}"
        )

        if json_key not in existing_handler_keys:
            json_handler = (
                logging.handlers
                .RotatingFileHandler(
                    json_path,
                    maxBytes=max_log_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
            )

            json_handler.setLevel(
                resolved_level
            )

            json_handler.setFormatter(
                StructuredJSONFormatter()
            )

            setattr(
                json_handler,
                "_layer2_handler_key",
                json_key,
            )

            logger.addHandler(
                json_handler
            )

    if not logger.handlers:
        logger.addHandler(
            logging.NullHandler()
        )

    return logger


def get_logger(
    module_name: Optional[str] = None,
    *,
    packet_id: Optional[str] = None,
    result_id: Optional[str] = None,
    scenario: Optional[str] = None,
    level: str | int = DEFAULT_LOG_LEVEL,
    log_file: Optional[Path | str] = None,
    json_log_file: Optional[Path | str] = None,
    enable_console: bool = True,
) -> Layer2LoggerAdapter:
    """Return a context-aware Layer 2 logger."""

    logger = configure_logger(
        module_name=module_name,
        level=level,
        log_file=log_file,
        json_log_file=json_log_file,
        enable_console=enable_console,
    )

    context = {
        "layer2_module": module_name,
        "packet_id": packet_id,
        "result_id": result_id,
        "scenario": scenario,
    }

    return Layer2LoggerAdapter(
        logger,
        {
            key: value
            for key, value in context.items()
            if value is not None
        },
    )


# ============================================================
# EVENT LOGGING
# ============================================================

def log_event(
    logger: logging.Logger | Layer2LoggerAdapter,
    *,
    event: str,
    message: str,
    level: str | int = "INFO",
    details: Optional[
        Mapping[str, Any]
    ] = None,
) -> None:
    """Write a structured Layer 2 event."""

    event_name = (
        event.strip()
        if isinstance(event, str)
        else ""
    )

    if not event_name:
        raise Layer2LoggerError(
            "event must be a non-empty string."
        )

    resolved_level = normalize_log_level(
        level
    )

    extra: Dict[str, Any] = {
        "event": event_name,
    }

    if details:
        extra.update(
            make_json_safe(
                dict(details)
            )
        )

    logger.log(
        resolved_level,
        message,
        extra=extra,
    )


def log_exception(
    logger: logging.Logger | Layer2LoggerAdapter,
    error: BaseException,
    *,
    event: str = "module_exception",
    message: Optional[str] = None,
    details: Optional[
        Mapping[str, Any]
    ] = None,
) -> None:
    """Log an exception with traceback and context."""

    extra: Dict[str, Any] = {
        "event": event,
        "exception_type": (
            error.__class__.__name__
        ),
    }

    if details:
        extra.update(
            make_json_safe(
                dict(details)
            )
        )

    logger.error(
        message or str(error),
        exc_info=(
            type(error),
            error,
            error.__traceback__,
        ),
        extra=extra,
    )


# ============================================================
# EXECUTION TIMER
# ============================================================

class ModuleTimer(AbstractContextManager):
    """
    Measure and optionally log module execution time.

    Example
    -------
    with ModuleTimer(
        "object_detector",
        logger=logger
    ) as timer:
        run_detection()

    print(timer.elapsed_ms)
    """

    def __init__(
        self,
        module_name: str,
        *,
        logger: Optional[
            logging.Logger
            | Layer2LoggerAdapter
        ] = None,
        packet_id: Optional[str] = None,
        log_start: bool = True,
        log_completion: bool = True,
    ) -> None:

        if (
            not isinstance(module_name, str)
            or not module_name.strip()
        ):
            raise Layer2LoggerError(
                "module_name must be a "
                "non-empty string."
            )

        self.module_name = (
            module_name.strip()
        )

        self.logger = logger
        self.packet_id = packet_id
        self.log_start = log_start
        self.log_completion = (
            log_completion
        )

        self.timer_id = (
            "TIMER_"
            f"{uuid.uuid4().hex[:12].upper()}"
        )

        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

        self.success: Optional[bool] = None
        self.error: Optional[
            BaseException
        ] = None

    def __enter__(self) -> "ModuleTimer":

        self.started_at = time.perf_counter()
        self.finished_at = None
        self.success = None
        self.error = None

        if (
            self.logger is not None
            and self.log_start
        ):
            log_event(
                self.logger,
                event="module_started",
                message=(
                    f"{self.module_name} started."
                ),
                details={
                    "timer_id": self.timer_id,
                    "module_name": (
                        self.module_name
                    ),
                    "packet_id": self.packet_id,
                },
            )

        return self

    def __exit__(
        self,
        exception_type: Any,
        exception: Any,
        traceback_object: Any,
    ) -> bool:

        self.finished_at = time.perf_counter()
        self.error = exception
        self.success = exception is None

        if (
            self.logger is not None
            and self.log_completion
        ):
            if exception is None:
                log_event(
                    self.logger,
                    event="module_completed",
                    message=(
                        f"{self.module_name} "
                        "completed."
                    ),
                    details={
                        "timer_id": self.timer_id,
                        "module_name": (
                            self.module_name
                        ),
                        "packet_id": (
                            self.packet_id
                        ),
                        "processing_time_ms": (
                            self.elapsed_ms
                        ),
                    },
                )
            else:
                log_exception(
                    self.logger,
                    exception,
                    event="module_failed",
                    message=(
                        f"{self.module_name} "
                        "failed."
                    ),
                    details={
                        "timer_id": self.timer_id,
                        "module_name": (
                            self.module_name
                        ),
                        "packet_id": (
                            self.packet_id
                        ),
                        "processing_time_ms": (
                            self.elapsed_ms
                        ),
                    },
                )

        return False

    @property
    def elapsed_seconds(self) -> float:

        if self.started_at is None:
            return 0.0

        endpoint = (
            self.finished_at
            if self.finished_at is not None
            else time.perf_counter()
        )

        return max(
            0.0,
            endpoint - self.started_at,
        )

    @property
    def elapsed_ms(self) -> float:

        milliseconds = (
            self.elapsed_seconds * 1000.0
        )

        if not math.isfinite(milliseconds):
            return 0.0

        return round(
            milliseconds,
            3,
        )

    def details(self) -> Dict[str, Any]:
        """Return timer state."""

        return {
            "timer_id": self.timer_id,
            "module_name": self.module_name,
            "packet_id": self.packet_id,
            "elapsed_ms": self.elapsed_ms,
            "success": self.success,
            "error_type": (
                self.error.__class__.__name__
                if self.error is not None
                else None
            ),
        }


# ============================================================
# HANDLER CLEANUP
# ============================================================

def close_logger(
    logger: (
        logging.Logger
        | Layer2LoggerAdapter
    ),
) -> None:
    """Flush and close all handlers."""

    underlying_logger = (
        logger.logger
        if isinstance(
            logger,
            logging.LoggerAdapter,
        )
        else logger
    )

    for handler in list(
        underlying_logger.handlers
    ):
        try:
            handler.flush()
            handler.close()
        finally:
            underlying_logger.removeHandler(
                handler
            )


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 72)
    print("NOONGIL-X | LAYER 2 LOGGER SELF-TEST")
    print("=" * 72)

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    log_directory = (
        project_root
        / "output"
        / "layer2"
        / "logger_self_test"
    )

    human_log_path = (
        log_directory
        / "layer2_test.log"
    )

    json_log_path = (
        log_directory
        / "layer2_test.jsonl"
    )

    try:
        logger_name = (
            "logger_self_test_"
            f"{uuid.uuid4().hex[:8]}"
        )

        logger = get_logger(
            logger_name,
            packet_id="MSP_TEST_007",
            scenario="park_walking",
            level="DEBUG",
            log_file=human_log_path,
            json_log_file=json_log_path,
            enable_console=True,
        )

        print("[PASS] Logger configured")

        logger.debug(
            "Debug logging is active."
        )

        logger.info(
            "Layer 2 logger self-test started."
        )

        logger.warning(
            "This is a test warning."
        )

        print("[PASS] Standard log levels written")

        log_event(
            logger,
            event="packet_received",
            message=(
                "Layer 1 packet received."
            ),
            details={
                "available_modalities": [
                    "vision",
                    "audio",
                    "motion",
                ],
                "confidence": 0.935,
            },
        )

        print("[PASS] Structured event logged")

        with ModuleTimer(
            "scene_classifier",
            logger=logger,
            packet_id="MSP_TEST_007",
        ) as timer:
            time.sleep(0.01)

        if timer.elapsed_ms <= 0.0:
            raise AssertionError(
                "Module timer did not record time."
            )

        print("[PASS] Module timer measured execution")

        try:
            raise ValueError(
                "Simulated model inference error."
            )
        except ValueError as error:
            log_exception(
                logger,
                error,
                event="self_test_exception",
                message=(
                    "Expected test exception."
                ),
                details={
                    "recoverable": True
                },
            )

        print("[PASS] Exception logged")

        handler_count_before = len(
            logger.logger.handlers
        )

        duplicate_logger = get_logger(
            logger_name,
            packet_id="MSP_TEST_007",
            scenario="park_walking",
            level="DEBUG",
            log_file=human_log_path,
            json_log_file=json_log_path,
            enable_console=True,
        )

        handler_count_after = len(
            duplicate_logger.logger.handlers
        )

        if (
            handler_count_before
            != handler_count_after
        ):
            raise AssertionError(
                "Duplicate logging handlers "
                "were created."
            )

        print("[PASS] Duplicate handlers prevented")

        for handler in logger.logger.handlers:
            handler.flush()

        if (
            not human_log_path.exists()
            or human_log_path.stat().st_size == 0
        ):
            raise AssertionError(
                "Human-readable log was not created."
            )

        print("[PASS] Human-readable log verified")

        if (
            not json_log_path.exists()
            or json_log_path.stat().st_size == 0
        ):
            raise AssertionError(
                "JSON log was not created."
            )

        json_lines = [
            line
            for line in json_log_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if not json_lines:
            raise AssertionError(
                "JSON log contains no events."
            )

        for line in json_lines:
            parsed = json.loads(line)

            if "timestamp" not in parsed:
                raise AssertionError(
                    "JSON log timestamp is missing."
                )

            if "message" not in parsed:
                raise AssertionError(
                    "JSON log message is missing."
                )

        print("[PASS] Structured JSON log verified")

        print("\nLogger summary:")
        print(f"  logger: {logger.logger.name}")
        print(f"  handlers: {handler_count_after}")
        print(f"  timer: {timer.elapsed_ms} ms")
        print(f"  human log: {human_log_path}")
        print(f"  JSON log: {json_log_path}")

        close_logger(logger)

        print("[PASS] Logger handlers closed")

        print("\n" + "=" * 72)
        print("[PASSED] LAYER 2 LOGGER IS WORKING")
        print("=" * 72)

        return True

    except (
        Layer2LoggerError,
        AssertionError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        print(f"\n[FAILED] {error}")
        print("=" * 72)

        return False


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:

    return argparse.ArgumentParser(
        description=(
            "Run the NOONGIL-X Layer 2 "
            "logger self-test."
        )
    )


def main() -> int:

    build_argument_parser().parse_args()

    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
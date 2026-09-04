"""
NOONGIL-X Layer 1 centralized logging utilities.

File: layer1/utils/logger.py
Python: 3.10+
Dependencies: standard library only.
"""

from __future__ import annotations

import functools
import json
import logging
import logging.handlers
import os
import sys
import time
import traceback

from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Mapping, Optional, TypeVar, cast

from layer1.config.paths import (
    LAYER1_ERROR_LOG_PATH,
    LAYER1_LOG_PATH,
    LAYER1_SENSOR_LOG_PATH,
    ensure_parent_directory,
)
from layer1.config.settings import (
    Layer1Settings,
    LogLevel,
    create_default_settings,
)

F = TypeVar("F", bound=Callable[..., Any])

LOGGER_NAMESPACE = "noongil.layer1"
TEXT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(threadName)s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

ANSI_RESET = "\033[0m"
ANSI_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def level_to_int(level: LogLevel | str | int) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, LogLevel):
        level = level.value
    if not isinstance(level, str):
        raise TypeError("level must be int, str, or LogLevel")
    result = logging.getLevelName(level.strip().upper())
    if not isinstance(result, int):
        raise ValueError(f"Unsupported log level: {level!r}")
    return result


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if isinstance(value, Mapping):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return make_json_safe(asdict(value))
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


class MinLevelFilter(logging.Filter):
    def __init__(self, minimum: int) -> None:
        super().__init__()
        self.minimum = minimum

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.minimum


class MaxLevelFilter(logging.Filter):
    def __init__(self, maximum: int) -> None:
        super().__init__()
        self.maximum = maximum

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.maximum


class TextFormatter(logging.Formatter):
    RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "message",
        "asctime", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        extras = {
            k: make_json_safe(v)
            for k, v in record.__dict__.items()
            if k not in self.RESERVED and not k.startswith("_")
        }
        if extras:
            text += " | extra=" + json.dumps(
                extras, ensure_ascii=False, sort_keys=True
            )
        return text


class ColorFormatter(TextFormatter):
    def __init__(self, *, use_color: bool) -> None:
        super().__init__(TEXT_FORMAT, DATE_FORMAT)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self.use_color:
            return text
        color = ANSI_COLORS.get(record.levelno, "")
        return f"{color}{text}{ANSI_RESET}" if color else text


class JsonFormatter(logging.Formatter):
    RESERVED = TextFormatter.RESERVED

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.threadName,
            "message": record.getMessage(),
        }
        extras = {
            k: make_json_safe(v)
            for k, v in record.__dict__.items()
            if k not in self.RESERVED and not k.startswith("_")
        }
        if extras:
            payload["extra"] = extras
        if record.exc_info:
            payload["exception"] = {
                "type": (
                    record.exc_info[0].__name__
                    if record.exc_info[0]
                    else "Exception"
                ),
                "message": str(record.exc_info[1]),
                "traceback": "".join(
                    traceback.format_exception(*record.exc_info)
                ),
            }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


@dataclass
class LogStatistics:
    debug: int = 0
    info: int = 0
    warning: int = 0
    error: int = 0
    critical: int = 0
    sensor_events: int = 0
    timing_events: int = 0
    exception_events: int = 0
    started_at: str = field(default_factory=utc_now_iso)
    last_event_at: Optional[str] = None

    def record(self, level: int) -> None:
        if level >= logging.CRITICAL:
            self.critical += 1
        elif level >= logging.ERROR:
            self.error += 1
        elif level >= logging.WARNING:
            self.warning += 1
        elif level >= logging.INFO:
            self.info += 1
        else:
            self.debug += 1
        self.last_event_at = utc_now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StatisticsHandler(logging.Handler):
    def __init__(self, statistics: LogStatistics) -> None:
        super().__init__(logging.NOTSET)
        self.statistics = statistics

    def emit(self, record: logging.LogRecord) -> None:
        self.statistics.record(record.levelno)


@dataclass
class LoggerOptions:
    use_color: bool = True
    json_file_logs: bool = False
    sensor_log_enabled: bool = True
    propagate: bool = False


class Layer1LogManager:
    _instance: Optional["Layer1LogManager"] = None
    _instance_lock = RLock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "Layer1LogManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(
        self,
        settings: Optional[Layer1Settings] = None,
        options: Optional[LoggerOptions] = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._lock = RLock()
        self.settings = settings or create_default_settings()
        self.options = options or LoggerOptions()
        self.statistics = LogStatistics()
        self.root = logging.getLogger(LOGGER_NAMESPACE)
        self.sensor = logging.getLogger(f"{LOGGER_NAMESPACE}.sensor")
        self._configured = False
        self.configure(self.settings, self.options, force=True)

    @staticmethod
    def _clear_handlers(logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            try:
                handler.flush()
                handler.close()
            finally:
                logger.removeHandler(handler)

    @staticmethod
    def _color_supported() -> bool:
        if os.getenv("NO_COLOR"):
            return False
        return bool(
            getattr(sys.stdout, "isatty", lambda: False)()
            or os.getenv("WT_SESSION")
            or os.getenv("ANSICON")
        )

    def _file_handler(
        self,
        path: Path,
        *,
        minimum: Optional[int] = None,
        maximum: Optional[int] = None,
    ) -> logging.Handler:
        ensure_parent_directory(path)
        cfg = self.settings.logging
        handler = logging.handlers.RotatingFileHandler(
            str(path),
            maxBytes=cfg.maximum_log_size_bytes,
            backupCount=cfg.backup_count,
            encoding="utf-8",
            delay=True,
        )
        handler.setLevel(level_to_int(cfg.level))
        if minimum is not None:
            handler.addFilter(MinLevelFilter(minimum))
        if maximum is not None:
            handler.addFilter(MaxLevelFilter(maximum))
        handler.setFormatter(
            JsonFormatter()
            if self.options.json_file_logs
            else TextFormatter(TEXT_FORMAT, DATE_FORMAT)
        )
        return handler

    def configure(
        self,
        settings: Optional[Layer1Settings] = None,
        options: Optional[LoggerOptions] = None,
        *,
        force: bool = False,
    ) -> logging.Logger:
        with self._lock:
            if settings is not None:
                settings.validate()
                self.settings = settings
            if options is not None:
                self.options = options
            if self._configured and not force:
                return self.root

            self._clear_handlers(self.root)
            self._clear_handlers(self.sensor)

            cfg = self.settings.logging
            level = level_to_int(cfg.level)

            self.root.setLevel(level)
            self.root.propagate = self.options.propagate
            self.sensor.setLevel(level)
            self.sensor.propagate = False

            self.root.addHandler(StatisticsHandler(self.statistics))

            if cfg.enabled and cfg.log_to_console:
                console = logging.StreamHandler(sys.stdout)
                console.setLevel(level)
                console.setFormatter(
                    ColorFormatter(
                        use_color=(
                            self.options.use_color
                            and self._color_supported()
                        )
                    )
                )
                self.root.addHandler(console)

            if cfg.enabled and cfg.log_to_file:
                self.root.addHandler(
                    self._file_handler(
                        LAYER1_LOG_PATH,
                        maximum=logging.WARNING,
                    )
                )
                self.root.addHandler(
                    self._file_handler(
                        LAYER1_ERROR_LOG_PATH,
                        minimum=logging.ERROR,
                    )
                )

            if cfg.enabled and self.options.sensor_log_enabled:
                sensor_handler = self._file_handler(
                    LAYER1_SENSOR_LOG_PATH
                )
                sensor_handler.setLevel(logging.INFO)
                self.sensor.addHandler(sensor_handler)

            self._configured = True
            return self.root

    def get_logger(self, module_name: Optional[str] = None) -> logging.Logger:
        if not module_name:
            return self.root
        normalized = (
            module_name.strip()
            .replace(" ", "_")
            .replace("/", ".")
            .replace("\\", ".")
        )
        if normalized.startswith(LOGGER_NAMESPACE):
            return logging.getLogger(normalized)
        return logging.getLogger(f"{LOGGER_NAMESPACE}.{normalized}")

    def log_sensor_event(
        self,
        *,
        modality: str,
        event: str,
        level: int | str | LogLevel = logging.INFO,
        device_id: Optional[str] = None,
        sensor_type: Optional[str] = None,
        packet_id: Optional[str] = None,
        sequence_number: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not modality.strip() or not event.strip():
            raise ValueError("modality and event cannot be empty")
        self.sensor.log(
            level_to_int(level),
            event,
            extra={
                "event_type": "sensor_event",
                "modality": modality,
                "device_id": device_id,
                "sensor_type": sensor_type,
                "packet_id": packet_id,
                "sequence_number": sequence_number,
                "details": make_json_safe(details or {}),
            },
        )
        self.statistics.sensor_events += 1
        self.statistics.last_event_at = utc_now_iso()

    def log_exception(
        self,
        logger: logging.Logger,
        message: str,
        *,
        error: Optional[BaseException] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        extra = {
            "event_type": "exception",
            "details": make_json_safe(details or {}),
        }
        if error is None:
            logger.exception(message, extra=extra)
        else:
            logger.error(
                message,
                exc_info=(type(error), error, error.__traceback__),
                extra=extra,
            )
        self.statistics.exception_events += 1
        self.statistics.last_event_at = utc_now_iso()

    def record_timing(self) -> None:
        self.statistics.timing_events += 1
        self.statistics.last_event_at = utc_now_iso()

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": self._configured,
            "namespace": LOGGER_NAMESPACE,
            "level": logging.getLevelName(self.root.level),
            "root_handler_count": len(self.root.handlers),
            "sensor_handler_count": len(self.sensor.handlers),
            "main_log_path": str(LAYER1_LOG_PATH),
            "error_log_path": str(LAYER1_ERROR_LOG_PATH),
            "sensor_log_path": str(LAYER1_SENSOR_LOG_PATH),
            "statistics": self.statistics.to_dict(),
        }

    def shutdown(self) -> None:
        for logger in (self.root, self.sensor):
            for handler in list(logger.handlers):
                try:
                    handler.flush()
                    handler.close()
                except Exception:
                    pass


_MANAGER: Optional[Layer1LogManager] = None
_MANAGER_LOCK = RLock()


def get_log_manager(
    settings: Optional[Layer1Settings] = None,
    *,
    options: Optional[LoggerOptions] = None,
    force: bool = False,
) -> Layer1LogManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = Layer1LogManager(settings, options)
        elif settings is not None or options is not None:
            _MANAGER.configure(settings, options, force=force)
        return _MANAGER


def configure_logging(
    settings: Optional[Layer1Settings] = None,
    *,
    force: bool = False,
    use_color: bool = True,
    json_file_logs: bool = False,
    sensor_log_enabled: bool = True,
) -> logging.Logger:
    manager = get_log_manager(
        settings,
        options=LoggerOptions(
            use_color=use_color,
            json_file_logs=json_file_logs,
            sensor_log_enabled=sensor_log_enabled,
        ),
        force=force,
    )
    return manager.root


def get_logger(
    module_name: Optional[str] = None,
    *,
    settings: Optional[Layer1Settings] = None,
) -> logging.Logger:
    return get_log_manager(settings).get_logger(module_name)


def log_sensor_event(**kwargs: Any) -> None:
    get_log_manager().log_sensor_event(**kwargs)


def log_exception(
    logger: logging.Logger,
    message: str,
    *,
    error: Optional[BaseException] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> None:
    get_log_manager().log_exception(
        logger,
        message,
        error=error,
        details=details,
    )


@dataclass
class TimingResult:
    operation: str
    started_at: str
    ended_at: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    success: bool = False
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineTimer(AbstractContextManager["PipelineTimer"]):
    def __init__(
        self,
        operation: str,
        *,
        logger: Optional[logging.Logger] = None,
        level: int | str | LogLevel = logging.INFO,
        metadata: Optional[Mapping[str, Any]] = None,
        log_start: bool = False,
    ) -> None:
        if not operation.strip():
            raise ValueError("operation cannot be empty")
        self.operation = operation
        self.logger = logger or get_logger("timing")
        self.level = level_to_int(level)
        self.metadata = dict(metadata or {})
        self.log_start = log_start
        self._started: Optional[float] = None
        self.result = TimingResult(
            operation=operation,
            started_at=utc_now_iso(),
            metadata=self.metadata,
        )

    def __enter__(self) -> "PipelineTimer":
        self._started = time.perf_counter()
        if self.log_start:
            self.logger.log(
                self.level,
                f"Started: {self.operation}",
                extra={
                    "event_type": "timing_start",
                    "operation": self.operation,
                },
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._started is None:
            raise RuntimeError("PipelineTimer was not entered")
        elapsed = time.perf_counter() - self._started
        self.result.ended_at = utc_now_iso()
        self.result.elapsed_seconds = elapsed
        self.result.success = exc_type is None
        self.result.error_type = exc_type.__name__ if exc_type else None

        extra = {
            "event_type": "timing_complete",
            "operation": self.operation,
            "elapsed_seconds": round(elapsed, 6),
            "success": self.result.success,
            "metadata": make_json_safe(self.metadata),
        }

        if exc_type is None:
            self.logger.log(
                self.level,
                f"Completed: {self.operation} in {elapsed:.6f}s",
                extra=extra,
            )
        else:
            self.logger.error(
                f"Failed: {self.operation} after {elapsed:.6f}s",
                exc_info=(exc_type, exc, tb),
                extra=extra,
            )

        get_log_manager().record_timing()
        return False


def timed(
    operation: Optional[str] = None,
    *,
    logger_name: str = "timing",
    level: int | str | LogLevel = logging.INFO,
) -> Callable[[F], F]:
    def decorator(function: F) -> F:
        name = operation or f"{function.__module__}.{function.__qualname__}"

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with PipelineTimer(
                name,
                logger=get_logger(logger_name),
                level=level,
            ):
                return function(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def run_logger_self_test() -> bool:
    print("\n" + "=" * 72)
    print("NOONGIL-X | LAYER 1 LOGGER TEST")
    print("=" * 72)

    try:
        settings = create_default_settings()
        settings.logging.enabled = True
        settings.logging.level = LogLevel.DEBUG
        settings.logging.log_to_console = True
        settings.logging.log_to_file = True

        print("[1/6] Configuring logger...")
        configure_logging(settings, force=True)
        logger = get_logger("self_test")
        print("[SUCCESS] Logger configured.")

        print("[2/6] Writing standard messages...")
        logger.debug("Debug self-test message.")
        logger.info("Info self-test message.")
        logger.warning("Warning self-test message.")
        print("[SUCCESS] Standard messages written.")

        print("[3/6] Writing sensor event...")
        log_sensor_event(
            modality="vision",
            event="Simulated camera packet accepted",
            device_id="PHONE_TEST_001",
            sensor_type="rgb_camera",
            packet_id="PACKET_TEST_001",
            sequence_number=1,
            details={"width": 640, "height": 480},
        )
        print("[SUCCESS] Sensor event written.")

        print("[4/6] Testing pipeline timer...")
        with PipelineTimer("logger_self_test", logger=logger) as timer:
            time.sleep(0.01)
        if not timer.result.elapsed_seconds:
            raise AssertionError("Timer did not measure elapsed time.")
        print("[SUCCESS] Pipeline timer works.")

        print("[5/6] Testing exception logging...")
        try:
            raise RuntimeError("Expected logger self-test exception.")
        except RuntimeError as error:
            log_exception(
                logger,
                "Expected logger self-test exception",
                error=error,
                details={"expected": True},
            )
        print("[SUCCESS] Exception logging works.")

        print("[6/6] Checking files and health...")
        manager = get_log_manager()
        for handler in manager.root.handlers:
            try:
                handler.flush()
            except Exception:
                pass
        for handler in manager.sensor.handlers:
            try:
                handler.flush()
            except Exception:
                pass

        for path in (
            LAYER1_LOG_PATH,
            LAYER1_ERROR_LOG_PATH,
            LAYER1_SENSOR_LOG_PATH,
        ):
            if not path.exists():
                raise AssertionError(f"Log file not created: {path}")

        health = manager.health_check()
        if not health["healthy"]:
            raise AssertionError("Logger health check failed.")

        print("[SUCCESS] Log files and health are valid.")
        print("\nLogger health:")
        print(json.dumps(health, indent=2, ensure_ascii=False))

        print("\n" + "=" * 72)
        print("[PASSED] LAYER 1 LOGGER IS WORKING")
        print("=" * 72)
        return True

    except Exception as error:
        print("\n" + "=" * 72)
        print("[FAILED] LAYER 1 LOGGER TEST")
        print("=" * 72)
        print(f"[ERROR] {type(error).__name__}: {error}")
        return False


if __name__ == "__main__":
    if not run_logger_self_test():
        raise SystemExit(1)
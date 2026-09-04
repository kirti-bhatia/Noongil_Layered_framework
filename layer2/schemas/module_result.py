"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Standard Perception Module Result
File    : layer2/schemas/module_result.py
============================================================

Purpose
-------
Defines a common output structure for every Layer 2 module:

- Vision processing
- Scene classification
- Object detection
- Activity recognition
- OCR and text understanding
- Speech recognition
- Sound-event detection
- Depth estimation
- Obstacle detection
- Motion analysis
- Confidence estimation
- Multimodal fusion

Every Layer 2 module should return a ModuleResult object.

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


# ============================================================
# CONSTANTS
# ============================================================

MODULE_RESULT_SCHEMA_VERSION = "1.0"

SUPPORTED_MODALITIES = {
    "vision",
    "objects",
    "text",
    "audio",
    "speech",
    "sound",
    "spatial",
    "depth",
    "motion",
    "activity",
    "fusion",
    "output",
    "system",
}


# ============================================================
# EXCEPTIONS
# ============================================================

class ModuleResultError(Exception):
    """Base exception for module-result operations."""


class ModuleResultValidationError(ModuleResultError):
    """Raised when a module result is invalid."""


class ModuleResultSerializationError(ModuleResultError):
    """Raised when a module result cannot be serialized."""


# ============================================================
# ENUMERATIONS
# ============================================================

class ModuleStatus(str, Enum):
    """Execution status of a Layer 2 module."""

    SUCCESS = "success"
    PARTIAL = "partial"
    DEGRADED = "degraded"
    SKIPPED = "skipped"
    FAILED = "failed"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""

    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="milliseconds"
    )


def generate_result_id(
    module_name: str,
) -> str:
    """Generate a unique result identifier."""

    normalized_name = (
        module_name.strip()
        .upper()
        .replace(" ", "_")
        .replace("-", "_")
    )

    return (
        f"L2_{normalized_name}_"
        f"{uuid.uuid4().hex[:12].upper()}"
    )


def require_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    """Validate and return a non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise ModuleResultValidationError(
            f"{field_name} must be a non-empty string."
        )

    return value.strip()


def validate_iso_timestamp(
    value: Any,
    field_name: str,
) -> str:
    """Validate an ISO-8601 timestamp."""

    timestamp = require_non_empty_string(
        value,
        field_name,
    )

    normalized = timestamp.replace(
        "Z",
        "+00:00",
    )

    try:
        datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ModuleResultValidationError(
            f"{field_name} is not a valid "
            f"ISO-8601 timestamp: {timestamp!r}"
        ) from error

    return timestamp


def validate_confidence(
    value: Any,
    field_name: str = "confidence",
) -> Optional[float]:
    """Validate an optional confidence score."""

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ModuleResultValidationError(
            f"{field_name} must be numeric."
        )

    confidence = float(value)

    if not math.isfinite(confidence):
        raise ModuleResultValidationError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= confidence <= 1.0:
        raise ModuleResultValidationError(
            f"{field_name} must be between 0 and 1."
        )

    return confidence


def validate_processing_time(
    value: Any,
) -> float:
    """Validate processing duration."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ModuleResultValidationError(
            "processing_time_ms must be numeric."
        )

    duration = float(value)

    if not math.isfinite(duration):
        raise ModuleResultValidationError(
            "processing_time_ms must be finite."
        )

    if duration < 0.0:
        raise ModuleResultValidationError(
            "processing_time_ms cannot be negative."
        )

    return duration


def make_json_safe(
    value: Any,
) -> Any:
    """Convert common Python values to JSON-safe values."""

    if isinstance(value, Enum):
        return value.value

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

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


# ============================================================
# MODULE RESULT
# ============================================================

@dataclass
class ModuleResult:
    """
    Standard output generated by one Layer 2 module.

    Parameters
    ----------
    module_name:
        Name of the module producing the result.

    modality:
        Primary modality handled by the module.

    status:
        Execution status.

    data:
        Module-specific result data.

    confidence:
        Prediction confidence between 0 and 1.

    processing_time_ms:
        Module execution duration in milliseconds.
    """

    module_name: str
    modality: str
    status: ModuleStatus

    data: Dict[str, Any] = field(
        default_factory=dict
    )

    confidence: Optional[float] = None
    processing_time_ms: float = 0.0

    result_id: str = ""
    timestamp: str = field(
        default_factory=utc_now_iso
    )

    source_packet_id: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    schema_version: str = (
        MODULE_RESULT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:

        if isinstance(self.status, str):
            try:
                self.status = ModuleStatus(
                    self.status.lower()
                )
            except ValueError as error:
                raise ModuleResultValidationError(
                    f"Unsupported module status: "
                    f"{self.status!r}"
                ) from error

        if not self.result_id:
            self.result_id = generate_result_id(
                self.module_name
            )

        self.validate()

    @property
    def succeeded(self) -> bool:
        """Return True for fully successful results."""

        return self.status == ModuleStatus.SUCCESS

    @property
    def usable(self) -> bool:
        """Return whether downstream modules may use the result."""

        return self.status in {
            ModuleStatus.SUCCESS,
            ModuleStatus.PARTIAL,
            ModuleStatus.DEGRADED,
        }

    @property
    def failed(self) -> bool:
        """Return whether module execution failed."""

        return self.status == ModuleStatus.FAILED

    def validate(self) -> None:
        """Validate the complete result."""

        self.module_name = require_non_empty_string(
            self.module_name,
            "module_name",
        )

        self.modality = require_non_empty_string(
            self.modality,
            "modality",
        ).lower()

        if self.modality not in SUPPORTED_MODALITIES:
            raise ModuleResultValidationError(
                f"Unsupported modality: "
                f"{self.modality!r}. "
                f"Supported modalities: "
                f"{sorted(SUPPORTED_MODALITIES)}"
            )

        if not isinstance(
            self.status,
            ModuleStatus,
        ):
            raise ModuleResultValidationError(
                "status must be a ModuleStatus."
            )

        self.result_id = require_non_empty_string(
            self.result_id,
            "result_id",
        )

        self.schema_version = (
            require_non_empty_string(
                self.schema_version,
                "schema_version",
            )
        )

        self.timestamp = validate_iso_timestamp(
            self.timestamp,
            "timestamp",
        )

        if not isinstance(self.data, dict):
            raise ModuleResultValidationError(
                "data must be a dictionary."
            )

        if not isinstance(self.metadata, dict):
            raise ModuleResultValidationError(
                "metadata must be a dictionary."
            )

        self.confidence = validate_confidence(
            self.confidence
        )

        self.processing_time_ms = (
            validate_processing_time(
                self.processing_time_ms
            )
        )

        if self.source_packet_id is not None:
            self.source_packet_id = (
                require_non_empty_string(
                    self.source_packet_id,
                    "source_packet_id",
                )
            )

        if self.model_name is not None:
            self.model_name = (
                require_non_empty_string(
                    self.model_name,
                    "model_name",
                )
            )

        if self.model_version is not None:
            self.model_version = (
                require_non_empty_string(
                    self.model_version,
                    "model_version",
                )
            )

        if not isinstance(self.warnings, list):
            raise ModuleResultValidationError(
                "warnings must be a list."
            )

        if not isinstance(self.errors, list):
            raise ModuleResultValidationError(
                "errors must be a list."
            )

        self.warnings = [
            str(item)
            for item in self.warnings
            if str(item).strip()
        ]

        self.errors = [
            str(item)
            for item in self.errors
            if str(item).strip()
        ]

        if (
            self.status == ModuleStatus.FAILED
            and not self.errors
        ):
            raise ModuleResultValidationError(
                "A failed result must contain at least "
                "one error message."
            )

        if (
            self.status == ModuleStatus.SKIPPED
            and self.confidence is not None
        ):
            raise ModuleResultValidationError(
                "A skipped result cannot have confidence."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result into a JSON-safe dictionary."""

        self.validate()

        payload = asdict(self)
        payload["status"] = self.status.value

        return make_json_safe(payload)

    def to_json(
        self,
        *,
        indent: int = 2,
    ) -> str:
        """Serialize the result into JSON text."""

        try:
            return json.dumps(
                self.to_dict(),
                indent=indent,
                ensure_ascii=False,
            )
        except (TypeError, ValueError) as error:
            raise ModuleResultSerializationError(
                "Unable to serialize module result."
            ) from error

    def write_json(
        self,
        file_path: Path | str,
    ) -> Path:
        """Write the result to a JSON file."""

        output_path = Path(file_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            output_path.write_text(
                self.to_json() + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            raise ModuleResultSerializationError(
                f"Unable to write result: {output_path}"
            ) from error

        return output_path

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ModuleResult":
        """Create a ModuleResult from a dictionary."""

        if not isinstance(payload, Mapping):
            raise ModuleResultValidationError(
                "Module result payload must be "
                "a dictionary."
            )

        return cls(
            module_name=payload.get(
                "module_name",
                "",
            ),
            modality=payload.get(
                "modality",
                "",
            ),
            status=payload.get(
                "status",
                ModuleStatus.FAILED,
            ),
            data=dict(
                payload.get("data", {})
            ),
            confidence=payload.get(
                "confidence"
            ),
            processing_time_ms=payload.get(
                "processing_time_ms",
                0.0,
            ),
            result_id=payload.get(
                "result_id",
                "",
            ),
            timestamp=payload.get(
                "timestamp",
                utc_now_iso(),
            ),
            source_packet_id=payload.get(
                "source_packet_id"
            ),
            model_name=payload.get(
                "model_name"
            ),
            model_version=payload.get(
                "model_version"
            ),
            warnings=list(
                payload.get("warnings", [])
            ),
            errors=list(
                payload.get("errors", [])
            ),
            metadata=dict(
                payload.get("metadata", {})
            ),
            schema_version=payload.get(
                "schema_version",
                MODULE_RESULT_SCHEMA_VERSION,
            ),
        )

    @classmethod
    def success(
        cls,
        *,
        module_name: str,
        modality: str,
        data: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
        processing_time_ms: float = 0.0,
        source_packet_id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ModuleResult":
        """Create a successful module result."""

        return cls(
            module_name=module_name,
            modality=modality,
            status=ModuleStatus.SUCCESS,
            data=data or {},
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            source_packet_id=source_packet_id,
            model_name=model_name,
            model_version=model_version,
            warnings=warnings or [],
            metadata=metadata or {},
        )

    @classmethod
    def partial(
        cls,
        *,
        module_name: str,
        modality: str,
        data: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
        processing_time_ms: float = 0.0,
        source_packet_id: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ModuleResult":
        """Create a partially successful result."""

        return cls(
            module_name=module_name,
            modality=modality,
            status=ModuleStatus.PARTIAL,
            data=data or {},
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            source_packet_id=source_packet_id,
            warnings=warnings or [
                "Module returned a partial result."
            ],
            metadata=metadata or {},
        )

    @classmethod
    def degraded(
        cls,
        *,
        module_name: str,
        modality: str,
        data: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
        processing_time_ms: float = 0.0,
        source_packet_id: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ModuleResult":
        """Create a degraded but usable result."""

        return cls(
            module_name=module_name,
            modality=modality,
            status=ModuleStatus.DEGRADED,
            data=data or {},
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            source_packet_id=source_packet_id,
            warnings=warnings or [
                "Module returned a degraded result."
            ],
            metadata=metadata or {},
        )

    @classmethod
    def skipped(
        cls,
        *,
        module_name: str,
        modality: str,
        reason: str,
        source_packet_id: Optional[str] = None,
        processing_time_ms: float = 0.0,
    ) -> "ModuleResult":
        """Create a skipped module result."""

        reason = require_non_empty_string(
            reason,
            "reason",
        )

        return cls(
            module_name=module_name,
            modality=modality,
            status=ModuleStatus.SKIPPED,
            data={},
            confidence=None,
            processing_time_ms=processing_time_ms,
            source_packet_id=source_packet_id,
            warnings=[reason],
        )

    @classmethod
    def failure(
        cls,
        *,
        module_name: str,
        modality: str,
        error: str,
        processing_time_ms: float = 0.0,
        source_packet_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ModuleResult":
        """Create a failed module result."""

        error = require_non_empty_string(
            error,
            "error",
        )

        return cls(
            module_name=module_name,
            modality=modality,
            status=ModuleStatus.FAILED,
            data={},
            confidence=None,
            processing_time_ms=processing_time_ms,
            source_packet_id=source_packet_id,
            errors=[error],
            metadata=metadata or {},
        )


# ============================================================
# EXECUTION TIMER
# ============================================================

class ModuleTimer:
    """
    Context manager for measuring module execution time.

    Example
    -------
    with ModuleTimer() as timer:
        run_model()

    print(timer.elapsed_ms)
    """

    def __init__(self) -> None:
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    def __enter__(self) -> "ModuleTimer":
        self.started_at = time.perf_counter()
        self.finished_at = None
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception: Any,
        traceback: Any,
    ) -> None:
        self.finished_at = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:

        if self.started_at is None:
            return 0.0

        endpoint = (
            self.finished_at
            if self.finished_at is not None
            else time.perf_counter()
        )

        return round(
            (endpoint - self.started_at) * 1000.0,
            3,
        )


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 72)
    print("NOONGIL-X | MODULE RESULT SCHEMA SELF-TEST")
    print("=" * 72)

    try:
        with ModuleTimer() as timer:
            time.sleep(0.01)

        result = ModuleResult.success(
            module_name="scene_classifier",
            modality="vision",
            data={
                "scene": {
                    "type": "park",
                    "confidence": 0.95,
                }
            },
            confidence=0.95,
            processing_time_ms=timer.elapsed_ms,
            source_packet_id="MSP_TEST_007",
            model_name="test_scene_classifier",
            model_version="1.0",
        )

        print("[PASS] Successful result created")
        print("[PASS] Result validation completed")

        serialized = result.to_json()
        restored_payload = json.loads(serialized)

        print("[PASS] Result serialized to JSON")

        restored = ModuleResult.from_dict(
            restored_payload
        )

        if restored.to_dict() != result.to_dict():
            raise AssertionError(
                "Restored result does not match "
                "the original result."
            )

        print("[PASS] Result restored from dictionary")

        partial_result = ModuleResult.partial(
            module_name="ocr_engine",
            modality="text",
            data={"recognized_text": []},
            confidence=0.42,
            warnings=[
                "No clearly readable text was detected."
            ],
        )

        if not partial_result.usable:
            raise AssertionError(
                "Partial result should be usable."
            )

        print("[PASS] Partial result created")

        skipped_result = ModuleResult.skipped(
            module_name="sound_event_detector",
            modality="sound",
            reason="Audio modality was unavailable.",
        )

        if skipped_result.usable:
            raise AssertionError(
                "Skipped result should not be usable."
            )

        print("[PASS] Skipped result created")

        failed_result = ModuleResult.failure(
            module_name="depth_estimator",
            modality="depth",
            error="Depth model could not be loaded.",
        )

        if not failed_result.failed:
            raise AssertionError(
                "Failure status was not preserved."
            )

        print("[PASS] Failed result created")

        try:
            ModuleResult.success(
                module_name="invalid_confidence_test",
                modality="vision",
                confidence=1.5,
            )
        except ModuleResultValidationError:
            print(
                "[PASS] Invalid confidence rejected"
            )
        else:
            raise AssertionError(
                "Invalid confidence was accepted."
            )

        print("\nExample result:")
        print(result.to_json())

        print("\n" + "=" * 72)
        print("[PASSED] MODULE RESULT SCHEMA IS WORKING")
        print("=" * 72)

        return True

    except (
        ModuleResultError,
        AssertionError,
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
            "Run the Layer 2 ModuleResult "
            "schema self-test."
        )
    )


def main() -> int:

    build_argument_parser().parse_args()

    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
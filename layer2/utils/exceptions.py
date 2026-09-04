"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Centralized Exception Hierarchy
File    : layer2/utils/exceptions.py
============================================================

Purpose
-------
Provides consistent exceptions for all Layer 2 components:

- Input reception
- Packet validation
- Media loading
- Model loading and inference
- Vision, OCR and audio processing
- Depth and motion processing
- Confidence calibration
- Multimodal fusion
- Output construction
- Pipeline execution

Every exception provides:
- A stable error code
- The responsible module
- Recoverability status
- Structured diagnostic details
- The original underlying exception, when available

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import argparse
import json
import traceback

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


# ============================================================
# ERROR CODES
# ============================================================

class ErrorCode(str, Enum):
    """Stable error codes used throughout Layer 2."""

    UNKNOWN = "L2_UNKNOWN"

    INPUT_RECEPTION = "L2_INPUT_RECEPTION"
    PACKET_VALIDATION = "L2_PACKET_VALIDATION"
    MISSING_MODALITY = "L2_MISSING_MODALITY"
    MEDIA_INPUT = "L2_MEDIA_INPUT"

    DEPENDENCY_MISSING = "L2_DEPENDENCY_MISSING"
    MODEL_LOADING = "L2_MODEL_LOADING"
    MODEL_INFERENCE = "L2_MODEL_INFERENCE"

    VISION_PROCESSING = "L2_VISION_PROCESSING"
    SCENE_CLASSIFICATION = "L2_SCENE_CLASSIFICATION"
    OBJECT_DETECTION = "L2_OBJECT_DETECTION"
    OBJECT_TRACKING = "L2_OBJECT_TRACKING"
    ACTIVITY_RECOGNITION = "L2_ACTIVITY_RECOGNITION"

    OCR_PROCESSING = "L2_OCR_PROCESSING"
    TEXT_INTERPRETATION = "L2_TEXT_INTERPRETATION"

    AUDIO_PROCESSING = "L2_AUDIO_PROCESSING"
    SPEECH_RECOGNITION = "L2_SPEECH_RECOGNITION"
    SOUND_EVENT_DETECTION = (
        "L2_SOUND_EVENT_DETECTION"
    )

    SPATIAL_PROCESSING = "L2_SPATIAL_PROCESSING"
    DEPTH_ESTIMATION = "L2_DEPTH_ESTIMATION"
    OBSTACLE_DETECTION = "L2_OBSTACLE_DETECTION"

    MOTION_PROCESSING = "L2_MOTION_PROCESSING"

    CONFIDENCE_CALIBRATION = (
        "L2_CONFIDENCE_CALIBRATION"
    )

    MULTIMODAL_FUSION = "L2_MULTIMODAL_FUSION"

    OUTPUT_BUILDING = "L2_OUTPUT_BUILDING"
    OUTPUT_VALIDATION = "L2_OUTPUT_VALIDATION"

    MODULE_TIMEOUT = "L2_MODULE_TIMEOUT"
    PIPELINE_EXECUTION = "L2_PIPELINE_EXECUTION"


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
    """Convert diagnostic values to JSON-safe data."""

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
# BASE EXCEPTION
# ============================================================

class Layer2Error(Exception):
    """
    Base exception for every Layer 2 error.
    """

    default_error_code = ErrorCode.UNKNOWN
    default_recoverable = False

    def __init__(
        self,
        message: str,
        *,
        module: Optional[str] = None,
        error_code: Optional[ErrorCode | str] = None,
        recoverable: Optional[bool] = None,
        details: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:

        if not isinstance(message, str) or not message.strip():
            message = self.__class__.__name__

        self.message = message.strip()
        self.module = (
            module.strip()
            if isinstance(module, str)
            and module.strip()
            else None
        )

        selected_code = (
            error_code
            if error_code is not None
            else self.default_error_code
        )

        if isinstance(selected_code, ErrorCode):
            self.error_code = selected_code
        else:
            try:
                self.error_code = ErrorCode(
                    selected_code
                )
            except ValueError:
                self.error_code = (
                    ErrorCode.UNKNOWN
                )

        self.recoverable = (
            self.default_recoverable
            if recoverable is None
            else bool(recoverable)
        )

        self.details = make_json_safe(
            dict(details or {})
        )

        self.cause = cause
        self.timestamp = utc_now_iso()

        super().__init__(self.message)

    @property
    def cause_type(self) -> Optional[str]:
        """Return the underlying exception type."""

        if self.cause is None:
            return None

        return self.cause.__class__.__name__

    @property
    def cause_message(self) -> Optional[str]:
        """Return the underlying exception message."""

        if self.cause is None:
            return None

        return str(self.cause)

    def to_dict(
        self,
        *,
        include_traceback: bool = False,
    ) -> Dict[str, Any]:
        """Convert the exception into structured data."""

        payload: Dict[str, Any] = {
            "error_type": (
                self.__class__.__name__
            ),
            "error_code": self.error_code.value,
            "message": self.message,
            "module": self.module,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp,
            "details": self.details,
            "cause": (
                {
                    "type": self.cause_type,
                    "message": self.cause_message,
                }
                if self.cause is not None
                else None
            ),
        }

        if include_traceback:
            payload["traceback"] = (
                self.format_traceback()
            )

        return make_json_safe(payload)

    def to_json(
        self,
        *,
        indent: int = 2,
        include_traceback: bool = False,
    ) -> str:
        """Serialize the exception to JSON."""

        return json.dumps(
            self.to_dict(
                include_traceback=(
                    include_traceback
                )
            ),
            indent=indent,
            ensure_ascii=False,
        )

    def format_traceback(self) -> Optional[str]:
        """Return the underlying traceback."""

        if self.cause is None:
            return None

        return "".join(
            traceback.format_exception(
                type(self.cause),
                self.cause,
                self.cause.__traceback__,
            )
        )

    def __str__(self) -> str:

        prefix = f"[{self.error_code.value}]"

        if self.module:
            prefix += f"[{self.module}]"

        return f"{prefix} {self.message}"


# ============================================================
# INPUT EXCEPTIONS
# ============================================================

class InputReceptionError(Layer2Error):
    default_error_code = ErrorCode.INPUT_RECEPTION
    default_recoverable = True


class PacketValidationError(Layer2Error):
    default_error_code = ErrorCode.PACKET_VALIDATION
    default_recoverable = False


class MissingModalityError(Layer2Error):
    default_error_code = ErrorCode.MISSING_MODALITY
    default_recoverable = True


class MediaInputError(Layer2Error):
    default_error_code = ErrorCode.MEDIA_INPUT
    default_recoverable = True


# ============================================================
# DEPENDENCY AND MODEL EXCEPTIONS
# ============================================================

class DependencyMissingError(Layer2Error):
    default_error_code = ErrorCode.DEPENDENCY_MISSING
    default_recoverable = False


class ModelLoadingError(Layer2Error):
    default_error_code = ErrorCode.MODEL_LOADING
    default_recoverable = True


class ModelInferenceError(Layer2Error):
    default_error_code = ErrorCode.MODEL_INFERENCE
    default_recoverable = True


# ============================================================
# VISION EXCEPTIONS
# ============================================================

class VisionProcessingError(Layer2Error):
    default_error_code = ErrorCode.VISION_PROCESSING
    default_recoverable = True


class SceneClassificationError(Layer2Error):
    default_error_code = ErrorCode.SCENE_CLASSIFICATION
    default_recoverable = True


class ObjectDetectionError(Layer2Error):
    default_error_code = ErrorCode.OBJECT_DETECTION
    default_recoverable = True


class ObjectTrackingError(Layer2Error):
    default_error_code = ErrorCode.OBJECT_TRACKING
    default_recoverable = True


class ActivityRecognitionError(Layer2Error):
    default_error_code = ErrorCode.ACTIVITY_RECOGNITION
    default_recoverable = True


# ============================================================
# TEXT EXCEPTIONS
# ============================================================

class OCRProcessingError(Layer2Error):
    default_error_code = ErrorCode.OCR_PROCESSING
    default_recoverable = True


class TextInterpretationError(Layer2Error):
    default_error_code = ErrorCode.TEXT_INTERPRETATION
    default_recoverable = True


# ============================================================
# AUDIO EXCEPTIONS
# ============================================================

class AudioProcessingError(Layer2Error):
    default_error_code = ErrorCode.AUDIO_PROCESSING
    default_recoverable = True


class SpeechRecognitionError(Layer2Error):
    default_error_code = ErrorCode.SPEECH_RECOGNITION
    default_recoverable = True


class SoundEventDetectionError(Layer2Error):
    default_error_code = (
        ErrorCode.SOUND_EVENT_DETECTION
    )
    default_recoverable = True


# ============================================================
# SPATIAL AND MOTION EXCEPTIONS
# ============================================================

class SpatialProcessingError(Layer2Error):
    default_error_code = ErrorCode.SPATIAL_PROCESSING
    default_recoverable = True


class DepthEstimationError(Layer2Error):
    default_error_code = ErrorCode.DEPTH_ESTIMATION
    default_recoverable = True


class ObstacleDetectionError(Layer2Error):
    default_error_code = ErrorCode.OBSTACLE_DETECTION
    default_recoverable = True


class MotionProcessingError(Layer2Error):
    default_error_code = ErrorCode.MOTION_PROCESSING
    default_recoverable = True


# ============================================================
# CONFIDENCE AND FUSION EXCEPTIONS
# ============================================================

class ConfidenceCalibrationError(Layer2Error):
    default_error_code = (
        ErrorCode.CONFIDENCE_CALIBRATION
    )
    default_recoverable = True


class MultimodalFusionError(Layer2Error):
    default_error_code = ErrorCode.MULTIMODAL_FUSION
    default_recoverable = True


# ============================================================
# OUTPUT EXCEPTIONS
# ============================================================

class OutputBuildingError(Layer2Error):
    default_error_code = ErrorCode.OUTPUT_BUILDING
    default_recoverable = True


class OutputValidationError(Layer2Error):
    default_error_code = ErrorCode.OUTPUT_VALIDATION
    default_recoverable = False


# ============================================================
# PIPELINE EXCEPTIONS
# ============================================================

class ModuleTimeoutError(Layer2Error):
    default_error_code = ErrorCode.MODULE_TIMEOUT
    default_recoverable = True


class PipelineExecutionError(Layer2Error):
    default_error_code = ErrorCode.PIPELINE_EXECUTION
    default_recoverable = False


# ============================================================
# EXCEPTION CONVERSION
# ============================================================

def wrap_exception(
    error: BaseException,
    *,
    message: Optional[str] = None,
    module: Optional[str] = None,
    error_class: type[Layer2Error] = Layer2Error,
    recoverable: Optional[bool] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> Layer2Error:
    """
    Convert an arbitrary exception into a Layer2Error.

    Existing Layer2Error instances are returned unchanged
    unless a different message or additional context is needed.
    """

    if (
        isinstance(error, Layer2Error)
        and message is None
        and module is None
        and details is None
        and recoverable is None
    ):
        return error

    selected_message = (
        message
        or str(error)
        or error.__class__.__name__
    )

    merged_details: Dict[str, Any] = {
        "original_exception_type": (
            error.__class__.__name__
        )
    }

    if isinstance(error, Layer2Error):
        merged_details[
            "original_error_code"
        ] = error.error_code.value

        merged_details.update(
            error.details
        )

    if details:
        merged_details.update(
            dict(details)
        )

    return error_class(
        selected_message,
        module=module,
        recoverable=recoverable,
        details=merged_details,
        cause=error,
    )


def exception_to_dict(
    error: BaseException,
    *,
    module: Optional[str] = None,
    include_traceback: bool = False,
) -> Dict[str, Any]:
    """Convert any exception into structured data."""

    wrapped = (
        error
        if isinstance(error, Layer2Error)
        else wrap_exception(
            error,
            module=module,
        )
    )

    return wrapped.to_dict(
        include_traceback=include_traceback
    )


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 72)
    print("NOONGIL-X | LAYER 2 EXCEPTION SELF-TEST")
    print("=" * 72)

    try:
        packet_error = PacketValidationError(
            "packet_id is missing.",
            module="packet_validator",
            details={
                "field": "metadata.packet_id",
                "packet_path": Path(
                    "layer1_sensor_packet.json"
                ),
            },
        )

        if (
            packet_error.error_code
            != ErrorCode.PACKET_VALIDATION
        ):
            raise AssertionError(
                "Packet error code is incorrect."
            )

        print("[PASS] Packet validation error created")

        if packet_error.recoverable:
            raise AssertionError(
                "Packet validation error should "
                "not be recoverable."
            )

        print("[PASS] Recoverability preserved")

        serialized = packet_error.to_json()
        restored_payload = json.loads(
            serialized
        )

        if (
            restored_payload["error_code"]
            != "L2_PACKET_VALIDATION"
        ):
            raise AssertionError(
                "Serialized error code is incorrect."
            )

        print("[PASS] Exception serialized to JSON")

        try:
            open(
                "file_that_does_not_exist.wav",
                "rb",
            )
        except OSError as original_error:
            wrapped_error = wrap_exception(
                original_error,
                message=(
                    "Unable to open audio input."
                ),
                module="audio_processor",
                error_class=AudioProcessingError,
                details={
                    "modality": "audio"
                },
            )

        if not isinstance(
            wrapped_error,
            AudioProcessingError,
        ):
            raise AssertionError(
                "Exception wrapping failed."
            )

        print("[PASS] Native exception wrapped")

        if (
            wrapped_error.cause_type
            != "FileNotFoundError"
        ):
            raise AssertionError(
                "Original exception was not preserved."
            )

        print("[PASS] Original cause preserved")

        missing_modality_error = (
            MissingModalityError(
                "Audio modality is unavailable.",
                module="modality_router",
                details={
                    "modality": "audio",
                    "available_modalities": [
                        "vision",
                        "motion",
                    ],
                },
            )
        )

        if not missing_modality_error.recoverable:
            raise AssertionError(
                "Missing modality should be "
                "recoverable."
            )

        print("[PASS] Recoverable error created")

        failure_payload = exception_to_dict(
            wrapped_error
        )

        required_fields = {
            "error_type",
            "error_code",
            "message",
            "module",
            "recoverable",
            "timestamp",
            "details",
            "cause",
        }

        if not required_fields.issubset(
            failure_payload
        ):
            raise AssertionError(
                "Structured error fields are missing."
            )

        print("[PASS] Structured error payload created")

        print("\nExample error:")
        print(packet_error.to_json())

        print("\n" + "=" * 72)
        print(
            "[PASSED] LAYER 2 EXCEPTIONS ARE WORKING"
        )
        print("=" * 72)

        return True

    except AssertionError as error:
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
            "exception hierarchy self-test."
        )
    )


def main() -> int:

    build_argument_parser().parse_args()

    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
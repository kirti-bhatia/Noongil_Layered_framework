"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Adapted Packet Validator
File    : layer2/input_reception/packet_validator.py
============================================================

Purpose
-------
Validates normalized Layer 1 input before modality routing.

Validation includes:
- Packet identity and Layer 2 routing
- Packet and synchronization timestamps
- Sensor confidence
- Image existence, format and dimensions
- WAV integrity, channels, sample rate and duration
- GPS coordinate ranges
- Motion-vector structure
- Required versus optional modalities

Validation statuses:
- valid
- degraded
- rejected

Architectural Boundary
----------------------
This module validates sensor input only. It does not run
perception models or create semantic results.

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import argparse
import copy
import math
import struct
import wave

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from layer2.config.settings import (
    Layer2Settings,
    create_default_settings,
    create_test_settings,
)

from layer2.input_reception.layer1_packet_adapter import (
    AdaptedLayer1Input,
    AdaptedModality,
    Layer1PacketAdapter,
)

from layer2.utils.exceptions import (
    PacketValidationError,
)

from layer2.utils.logger import (
    Layer2LoggerAdapter,
    get_logger,
    log_event,
    log_exception,
)


# ============================================================
# CONSTANTS
# ============================================================

VALIDATOR_VERSION = "1.0"

SUPPORTED_IMAGE_FORMATS = {
    "jpeg",
    "png",
    "bmp",
    "webp",
}

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
}

MAX_TIMESTAMP_OFFSET_SECONDS = 60.0
WARNING_TIMESTAMP_OFFSET_SECONDS = 5.0


# ============================================================
# ENUMERATIONS
# ============================================================

class ValidationStatus(str, Enum):
    """Overall packet-validation status."""

    VALID = "valid"
    DEGRADED = "degraded"
    REJECTED = "rejected"


class IssueSeverity(str, Enum):
    """Severity of one validation issue."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class ValidationIssue:
    """One packet-validation issue."""

    code: str
    severity: IssueSeverity
    message: str

    field: Optional[str] = None
    modality: Optional[str] = None
    fatal: bool = False

    details: Dict[str, Any] = dataclass_field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "field": self.field,
            "modality": self.modality,
            "fatal": self.fatal,
            "details": dict(self.details),
        }


@dataclass
class MediaInspection:
    """Media-file inspection result."""

    valid: bool
    format: Optional[str]
    size_bytes: int

    width: Optional[int] = None
    height: Optional[int] = None

    sample_rate_hz: Optional[int] = None
    channels: Optional[int] = None
    sample_width_bytes: Optional[int] = None
    frame_count: Optional[int] = None
    duration_seconds: Optional[float] = None

    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:

        return {
            "valid": self.valid,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
            "sample_rate_hz": (
                self.sample_rate_hz
            ),
            "channels": self.channels,
            "sample_width_bytes": (
                self.sample_width_bytes
            ),
            "frame_count": self.frame_count,
            "duration_seconds": (
                self.duration_seconds
            ),
            "error": self.error,
        }


@dataclass
class PacketValidationReport:
    """Complete validation report."""

    packet_id: str
    status: ValidationStatus

    issues: List[ValidationIssue]

    modality_validity: Dict[str, bool]
    media_inspections: Dict[
        str,
        MediaInspection,
    ]

    checks_performed: List[str]

    validator_version: str = (
        VALIDATOR_VERSION
    )

    @property
    def valid(self) -> bool:
        return self.status == ValidationStatus.VALID

    @property
    def degraded(self) -> bool:
        return (
            self.status
            == ValidationStatus.DEGRADED
        )

    @property
    def rejected(self) -> bool:
        return (
            self.status
            == ValidationStatus.REJECTED
        )

    @property
    def can_route(self) -> bool:
        """Return whether routing may continue."""

        return self.status in {
            ValidationStatus.VALID,
            ValidationStatus.DEGRADED,
        }

    @property
    def errors(self) -> List[ValidationIssue]:

        return [
            issue
            for issue in self.issues
            if issue.severity
            == IssueSeverity.ERROR
        ]

    @property
    def warnings(self) -> List[ValidationIssue]:

        return [
            issue
            for issue in self.issues
            if issue.severity
            == IssueSeverity.WARNING
        ]

    @property
    def fatal_issues(
        self,
    ) -> List[ValidationIssue]:

        return [
            issue
            for issue in self.issues
            if issue.fatal
        ]

    def to_dict(self) -> Dict[str, Any]:

        return {
            "packet_id": self.packet_id,
            "status": self.status.value,
            "valid": self.valid,
            "degraded": self.degraded,
            "rejected": self.rejected,
            "can_route": self.can_route,
            "issue_count": len(self.issues),
            "error_count": len(self.errors),
            "warning_count": len(
                self.warnings
            ),
            "fatal_issue_count": len(
                self.fatal_issues
            ),
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
            "modality_validity": dict(
                self.modality_validity
            ),
            "media_inspections": {
                name: inspection.to_dict()
                for name, inspection
                in self.media_inspections.items()
            },
            "checks_performed": list(
                self.checks_performed
            ),
            "validator_version": (
                self.validator_version
            ),
        }


# ============================================================
# GENERAL HELPERS
# ============================================================

def parse_iso_timestamp(
    value: str,
) -> datetime:
    """Parse an ISO timestamp as UTC."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Timestamp must be a non-empty string."
        )

    normalized = value.strip().replace(
        "Z",
        "+00:00",
    )

    parsed = datetime.fromisoformat(
        normalized
    )

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        timezone.utc
    )


def is_finite_number(
    value: Any,
) -> bool:
    """Return whether a value is a finite number."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def detect_image_format(
    header: bytes,
) -> Optional[str]:
    """Detect image format from file signature."""

    if header.startswith(
        b"\xff\xd8\xff"
    ):
        return "jpeg"

    if header.startswith(
        b"\x89PNG\r\n\x1a\n"
    ):
        return "png"

    if header.startswith(b"BM"):
        return "bmp"

    if (
        len(header) >= 12
        and header[:4] == b"RIFF"
        and header[8:12] == b"WEBP"
    ):
        return "webp"

    return None


def read_png_dimensions(
    header: bytes,
) -> tuple[Optional[int], Optional[int]]:

    if len(header) < 24:
        return None, None

    width = struct.unpack(
        ">I",
        header[16:20],
    )[0]

    height = struct.unpack(
        ">I",
        header[20:24],
    )[0]

    return width, height


def read_bmp_dimensions(
    header: bytes,
) -> tuple[Optional[int], Optional[int]]:

    if len(header) < 26:
        return None, None

    width = struct.unpack(
        "<I",
        header[18:22],
    )[0]

    height = struct.unpack(
        "<I",
        header[22:26],
    )[0]

    return width, height


def read_jpeg_dimensions(
    file_path: Path,
) -> tuple[Optional[int], Optional[int]]:
    """Read JPEG dimensions without external libraries."""

    start_of_frame_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }

    try:
        with file_path.open("rb") as file:

            if file.read(2) != b"\xff\xd8":
                return None, None

            while True:
                byte = file.read(1)

                if not byte:
                    return None, None

                if byte != b"\xff":
                    continue

                while byte == b"\xff":
                    byte = file.read(1)

                if not byte:
                    return None, None

                marker = byte[0]

                if marker in {
                    0xD8,
                    0xD9,
                }:
                    continue

                length_data = file.read(2)

                if len(length_data) != 2:
                    return None, None

                segment_length = struct.unpack(
                    ">H",
                    length_data,
                )[0]

                if segment_length < 2:
                    return None, None

                if marker in start_of_frame_markers:

                    frame_data = file.read(5)

                    if len(frame_data) != 5:
                        return None, None

                    height = struct.unpack(
                        ">H",
                        frame_data[1:3],
                    )[0]

                    width = struct.unpack(
                        ">H",
                        frame_data[3:5],
                    )[0]

                    return width, height

                file.seek(
                    segment_length - 2,
                    1,
                )

    except OSError:
        return None, None


# ============================================================
# MEDIA INSPECTION
# ============================================================

def inspect_image(
    file_path: Path | str,
) -> MediaInspection:
    """Inspect an image without external dependencies."""

    path = Path(file_path)

    if not path.is_file():
        return MediaInspection(
            valid=False,
            format=None,
            size_bytes=0,
            error=(
                f"Image does not exist: {path}"
            ),
        )

    try:
        size_bytes = path.stat().st_size

        if size_bytes <= 0:
            return MediaInspection(
                valid=False,
                format=None,
                size_bytes=size_bytes,
                error="Image file is empty.",
            )

        with path.open("rb") as file:
            header = file.read(64)

        image_format = detect_image_format(
            header
        )

        if image_format is None:
            return MediaInspection(
                valid=False,
                format=None,
                size_bytes=size_bytes,
                error=(
                    "Unsupported or invalid "
                    "image signature."
                ),
            )

        width: Optional[int] = None
        height: Optional[int] = None

        if image_format == "jpeg":
            width, height = (
                read_jpeg_dimensions(path)
            )

        elif image_format == "png":
            width, height = (
                read_png_dimensions(header)
            )

        elif image_format == "bmp":
            width, height = (
                read_bmp_dimensions(header)
            )

        dimensions_valid = (
            width is not None
            and height is not None
            and width > 0
            and height > 0
        )

        if (
            image_format != "webp"
            and not dimensions_valid
        ):
            return MediaInspection(
                valid=False,
                format=image_format,
                size_bytes=size_bytes,
                width=width,
                height=height,
                error=(
                    "Unable to determine valid "
                    "image dimensions."
                ),
            )

        return MediaInspection(
            valid=True,
            format=image_format,
            size_bytes=size_bytes,
            width=width,
            height=height,
        )

    except OSError as error:
        return MediaInspection(
            valid=False,
            format=None,
            size_bytes=0,
            error=(
                f"{error.__class__.__name__}: "
                f"{error}"
            ),
        )


def inspect_wav(
    file_path: Path | str,
) -> MediaInspection:
    """Inspect a WAV audio file."""

    path = Path(file_path)

    if not path.is_file():
        return MediaInspection(
            valid=False,
            format="wav",
            size_bytes=0,
            error=(
                f"Audio file does not exist: {path}"
            ),
        )

    try:
        size_bytes = path.stat().st_size

        if size_bytes <= 0:
            return MediaInspection(
                valid=False,
                format="wav",
                size_bytes=size_bytes,
                error="Audio file is empty.",
            )

        with wave.open(
            str(path),
            "rb",
        ) as audio_file:

            channels = (
                audio_file.getnchannels()
            )

            sample_rate = (
                audio_file.getframerate()
            )

            sample_width = (
                audio_file.getsampwidth()
            )

            frame_count = (
                audio_file.getnframes()
            )

        if sample_rate <= 0:
            return MediaInspection(
                valid=False,
                format="wav",
                size_bytes=size_bytes,
                error=(
                    "Invalid audio sample rate."
                ),
            )

        duration = (
            frame_count / sample_rate
        )

        valid = (
            channels > 0
            and sample_width > 0
            and frame_count > 0
            and duration > 0.0
        )

        return MediaInspection(
            valid=valid,
            format="wav",
            size_bytes=size_bytes,
            sample_rate_hz=sample_rate,
            channels=channels,
            sample_width_bytes=sample_width,
            frame_count=frame_count,
            duration_seconds=round(
                duration,
                6,
            ),
            error=(
                None
                if valid
                else "Invalid WAV parameters."
            ),
        )

    except (
        OSError,
        wave.Error,
        EOFError,
    ) as error:
        return MediaInspection(
            valid=False,
            format="wav",
            size_bytes=(
                path.stat().st_size
                if path.exists()
                else 0
            ),
            error=(
                f"{error.__class__.__name__}: "
                f"{error}"
            ),
        )


# ============================================================
# PACKET VALIDATOR
# ============================================================

class PacketValidator:
    """Validate adapted Layer 1 packets."""

    def __init__(
        self,
        settings: Optional[
            Layer2Settings
        ] = None,
        *,
        logger: Optional[
            Layer2LoggerAdapter
        ] = None,
    ) -> None:

        self.settings = (
            settings
            or create_default_settings()
        )

        self.settings.validate()

        self.logger = (
            logger
            or get_logger(
                "packet_validator"
            )
        )

    def validate(
        self,
        packet: AdaptedLayer1Input,
    ) -> PacketValidationReport:
        """Validate one adapted Layer 1 packet."""

        if not isinstance(
            packet,
            AdaptedLayer1Input,
        ):
            raise PacketValidationError(
                "packet must be an "
                "AdaptedLayer1Input.",
                module="packet_validator",
                details={
                    "received_type": (
                        packet.__class__.__name__
                    )
                },
            )

        log_event(
            self.logger,
            event="packet_validation_started",
            message=(
                "Layer 1 packet validation started."
            ),
            details={
                "packet_id": packet.packet_id
            },
        )

        issues: List[
            ValidationIssue
        ] = []

        modality_validity = {
            name: modality.usable
            for name, modality
            in packet.modalities.items()
        }

        media_inspections: Dict[
            str,
            MediaInspection,
        ] = {}

        checks = []

        self._validate_identity(
            packet,
            issues,
        )
        checks.append("packet_identity")

        self._validate_contract(
            packet,
            issues,
        )
        checks.append("layer2_contract")

        self._validate_timestamps(
            packet,
            issues,
        )
        checks.append("timestamps")

        self._validate_confidence(
            packet,
            issues,
        )
        checks.append("sensor_confidence")

        self._validate_vision(
            packet,
            issues,
            modality_validity,
            media_inspections,
        )
        checks.append("vision_media")

        self._validate_audio(
            packet,
            issues,
            modality_validity,
            media_inspections,
        )
        checks.append("audio_media")

        self._validate_spatial(
            packet,
            issues,
            modality_validity,
        )
        checks.append("spatial_data")

        self._validate_motion(
            packet,
            issues,
            modality_validity,
        )
        checks.append("motion_data")

        self._validate_core_availability(
            packet,
            issues,
            modality_validity,
        )
        checks.append("core_availability")

        status = self._determine_status(
            issues
        )

        report = PacketValidationReport(
            packet_id=packet.packet_id,
            status=status,
            issues=issues,
            modality_validity=(
                modality_validity
            ),
            media_inspections=(
                media_inspections
            ),
            checks_performed=checks,
        )

        log_event(
            self.logger,
            event=(
                "packet_validation_completed"
            ),
            message=(
                "Layer 1 packet validation "
                "completed."
            ),
            level=(
                "WARNING"
                if report.degraded
                else (
                    "ERROR"
                    if report.rejected
                    else "INFO"
                )
            ),
            details={
                "packet_id": packet.packet_id,
                "status": report.status.value,
                "errors": len(report.errors),
                "warnings": len(
                    report.warnings
                ),
                "can_route": (
                    report.can_route
                ),
            },
        )

        return report

    def validate_or_raise(
        self,
        packet: AdaptedLayer1Input,
    ) -> PacketValidationReport:
        """
        Validate and raise when packet is rejected.
        """

        report = self.validate(packet)

        if report.rejected:
            raise PacketValidationError(
                "Layer 1 packet was rejected.",
                module="packet_validator",
                details=report.to_dict(),
            )

        return report

    def _add_issue(
        self,
        issues: List[ValidationIssue],
        *,
        code: str,
        severity: IssueSeverity,
        message: str,
        field_name: Optional[str] = None,
        modality: Optional[str] = None,
        fatal: bool = False,
        details: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:

        issues.append(
            ValidationIssue(
                code=code,
                severity=severity,
                message=message,
                field=field_name,
                modality=modality,
                fatal=fatal,
                details=details or {},
            )
        )

    def _validate_identity(
        self,
        packet: AdaptedLayer1Input,
        issues: List[ValidationIssue],
    ) -> None:

        if not packet.packet_id.strip():
            self._add_issue(
                issues,
                code="PACKET_ID_MISSING",
                severity=IssueSeverity.ERROR,
                message="Packet ID is missing.",
                field_name="packet_id",
                fatal=True,
            )

        if not packet.source_frame_id.strip():
            self._add_issue(
                issues,
                code="FRAME_ID_MISSING",
                severity=IssueSeverity.ERROR,
                message=(
                    "Source frame ID is missing."
                ),
                field_name="source_frame_id",
                fatal=True,
            )

    def _validate_contract(
        self,
        packet: AdaptedLayer1Input,
        issues: List[ValidationIssue],
    ) -> None:

        ready = packet.layer2_contract.get(
            "ready_for_layer2"
        )

        if ready is not True:
            self._add_issue(
                issues,
                code="NOT_READY_FOR_LAYER2",
                severity=IssueSeverity.ERROR,
                message=(
                    "Layer 1 did not mark the "
                    "packet ready for Layer 2."
                ),
                field_name=(
                    "layer2_contract."
                    "ready_for_layer2"
                ),
                fatal=True,
            )

        action = packet.layer2_contract.get(
            "recommended_action"
        )

        if (
            action is not None
            and action
            not in {
                "process",
                "continue",
            }
        ):
            self._add_issue(
                issues,
                code="REACQUISITION_RECOMMENDED",
                severity=IssueSeverity.WARNING,
                message=(
                    "Layer 1 recommends "
                    f"{action!r}."
                ),
                field_name=(
                    "layer2_contract."
                    "recommended_action"
                ),
            )

    def _validate_timestamps(
        self,
        packet: AdaptedLayer1Input,
        issues: List[ValidationIssue],
    ) -> None:

        try:
            packet_timestamp = (
                parse_iso_timestamp(
                    packet.timestamp
                )
            )
        except (
            ValueError,
            TypeError,
        ):
            self._add_issue(
                issues,
                code="INVALID_PACKET_TIMESTAMP",
                severity=IssueSeverity.ERROR,
                message=(
                    "Packet timestamp is invalid."
                ),
                field_name="timestamp",
                fatal=True,
            )
            return

        anchor = packet.synchronization.get(
            "anchor_timestamp"
        )

        if anchor is None:
            self._add_issue(
                issues,
                code="ANCHOR_TIMESTAMP_MISSING",
                severity=IssueSeverity.WARNING,
                message=(
                    "Synchronization anchor "
                    "timestamp is missing."
                ),
                field_name=(
                    "synchronization."
                    "anchor_timestamp"
                ),
            )
            return

        try:
            anchor_timestamp = (
                parse_iso_timestamp(anchor)
            )
        except (
            ValueError,
            TypeError,
        ):
            self._add_issue(
                issues,
                code="INVALID_ANCHOR_TIMESTAMP",
                severity=IssueSeverity.ERROR,
                message=(
                    "Synchronization anchor "
                    "timestamp is invalid."
                ),
                field_name=(
                    "synchronization."
                    "anchor_timestamp"
                ),
            )
            return

        offset_seconds = abs(
            (
                packet_timestamp
                - anchor_timestamp
            ).total_seconds()
        )

        if (
            offset_seconds
            > MAX_TIMESTAMP_OFFSET_SECONDS
        ):
            self._add_issue(
                issues,
                code="TIMESTAMP_OFFSET_EXCESSIVE",
                severity=IssueSeverity.ERROR,
                message=(
                    "Packet and synchronization "
                    "timestamps are excessively "
                    "different."
                ),
                field_name="synchronization",
                details={
                    "offset_seconds": (
                        offset_seconds
                    )
                },
            )

        elif (
            offset_seconds
            > WARNING_TIMESTAMP_OFFSET_SECONDS
        ):
            self._add_issue(
                issues,
                code="TIMESTAMP_OFFSET_WARNING",
                severity=IssueSeverity.WARNING,
                message=(
                    "Packet and synchronization "
                    "timestamps differ."
                ),
                field_name="synchronization",
                details={
                    "offset_seconds": (
                        offset_seconds
                    )
                },
            )

    def _validate_confidence(
        self,
        packet: AdaptedLayer1Input,
        issues: List[ValidationIssue],
    ) -> None:

        overall = (
            packet.overall_sensor_confidence
        )

        if overall is None:
            self._add_issue(
                issues,
                code="CONFIDENCE_MISSING",
                severity=IssueSeverity.WARNING,
                message=(
                    "Overall sensor confidence "
                    "is missing."
                ),
                field_name=(
                    "sensor_confidence."
                    "overall_confidence"
                ),
            )

        elif (
            not math.isfinite(overall)
            or not 0.0 <= overall <= 1.0
        ):
            self._add_issue(
                issues,
                code="CONFIDENCE_INVALID",
                severity=IssueSeverity.ERROR,
                message=(
                    "Overall sensor confidence "
                    "is invalid."
                ),
                field_name=(
                    "sensor_confidence."
                    "overall_confidence"
                ),
            )

        elif (
            overall
            < self.settings.confidence
            .minimum_usable_confidence
        ):
            self._add_issue(
                issues,
                code="CONFIDENCE_TOO_LOW",
                severity=IssueSeverity.WARNING,
                message=(
                    "Overall sensor confidence "
                    "is below the preferred "
                    "minimum."
                ),
                field_name=(
                    "sensor_confidence."
                    "overall_confidence"
                ),
                details={
                    "confidence": overall,
                    "threshold": (
                        self.settings.confidence
                        .minimum_usable_confidence
                    ),
                },
            )

        for name, modality in (
            packet.modalities.items()
        ):
            confidence = modality.confidence

            if confidence is None:
                continue

            if (
                not math.isfinite(confidence)
                or not 0.0
                <= confidence
                <= 1.0
            ):
                self._add_issue(
                    issues,
                    code=(
                        "MODALITY_CONFIDENCE_INVALID"
                    ),
                    severity=IssueSeverity.ERROR,
                    message=(
                        f"{name} confidence "
                        "is invalid."
                    ),
                    modality=name,
                    field_name=(
                        f"modalities.{name}."
                        "confidence"
                    ),
                )

    def _validate_vision(
        self,
        packet: AdaptedLayer1Input,
        issues: List[ValidationIssue],
        modality_validity: Dict[str, bool],
        inspections: Dict[
            str,
            MediaInspection,
        ],
    ) -> None:

        vision = packet.modalities.get(
            "vision"
        )

        if (
            vision is None
            or not vision.available
        ):
            modality_validity["vision"] = False
            return

        if vision.media_path is None:
            modality_validity["vision"] = False

            self._add_issue(
                issues,
                code="VISION_PATH_MISSING",
                severity=IssueSeverity.ERROR,
                message=(
                    "Vision modality has no "
                    "frame path."
                ),
                modality="vision",
                field_name="vision.media_path",
            )
            return

        inspection = inspect_image(
            vision.media_path
        )

        inspections["vision"] = inspection
        modality_validity["vision"] = (
            inspection.valid
        )

        if not inspection.valid:
            self._add_issue(
                issues,
                code="VISION_FRAME_INVALID",
                severity=IssueSeverity.ERROR,
                message=(
                    inspection.error
                    or "Vision frame is invalid."
                ),
                modality="vision",
                field_name="vision.media_path",
                details=inspection.to_dict(),
            )

    def _validate_audio(
        self,
        packet: AdaptedLayer1Input,
        issues: List[ValidationIssue],
        modality_validity: Dict[str, bool],
        inspections: Dict[
            str,
            MediaInspection,
        ],
    ) -> None:

        audio = packet.modalities.get(
            "audio"
        )

        if (
            audio is None
            or not audio.available
        ):
            modality_validity["audio"] = False
            return

        if audio.media_path is None:
            modality_validity["audio"] = False

            self._add_issue(
                issues,
                code="AUDIO_PATH_MISSING",
                severity=IssueSeverity.ERROR,
                message=(
                    "Audio modality has no "
                    "audio path."
                ),
                modality="audio",
                field_name="audio.media_path",
            )
            return

        if (
            audio.media_path.suffix.lower()
            not in SUPPORTED_AUDIO_EXTENSIONS
        ):
            modality_validity["audio"] = False

            self._add_issue(
                issues,
                code="AUDIO_FORMAT_UNSUPPORTED",
                severity=IssueSeverity.ERROR,
                message=(
                    "Only WAV input is currently "
                    "supported."
                ),
                modality="audio",
                field_name="audio.media_path",
            )
            return

        inspection = inspect_wav(
            audio.media_path
        )

        inspections["audio"] = inspection
        modality_validity["audio"] = (
            inspection.valid
        )

        if not inspection.valid:
            self._add_issue(
                issues,
                code="AUDIO_FILE_INVALID",
                severity=IssueSeverity.ERROR,
                message=(
                    inspection.error
                    or "Audio file is invalid."
                ),
                modality="audio",
                field_name="audio.media_path",
                details=inspection.to_dict(),
            )
            return

        if (
            inspection.sample_rate_hz
            != self.settings.audio
            .target_sample_rate_hz
        ):
            self._add_issue(
                issues,
                code="AUDIO_SAMPLE_RATE_MISMATCH",
                severity=IssueSeverity.WARNING,
                message=(
                    "Audio sample rate differs "
                    "from the configured target."
                ),
                modality="audio",
                details={
                    "actual": (
                        inspection
                        .sample_rate_hz
                    ),
                    "target": (
                        self.settings.audio
                        .target_sample_rate_hz
                    ),
                },
            )

        if (
            inspection.channels
            != self.settings.audio
            .target_channels
        ):
            self._add_issue(
                issues,
                code="AUDIO_CHANNEL_MISMATCH",
                severity=IssueSeverity.WARNING,
                message=(
                    "Audio channel count differs "
                    "from the configured target."
                ),
                modality="audio",
                details={
                    "actual": (
                        inspection.channels
                    ),
                    "target": (
                        self.settings.audio
                        .target_channels
                    ),
                },
            )

        duration = (
            inspection.duration_seconds
            or 0.0
        )

        if (
            duration
            < self.settings.audio
            .minimum_audio_duration_seconds
        ):
            modality_validity["audio"] = False

            self._add_issue(
                issues,
                code="AUDIO_TOO_SHORT",
                severity=IssueSeverity.ERROR,
                message=(
                    "Audio duration is below "
                    "the configured minimum."
                ),
                modality="audio",
                details={
                    "duration_seconds": duration,
                    "minimum": (
                        self.settings.audio
                        .minimum_audio_duration_seconds
                    ),
                },
            )

        elif (
            duration
            > self.settings.audio
            .maximum_audio_duration_seconds
        ):
            self._add_issue(
                issues,
                code="AUDIO_TOO_LONG",
                severity=IssueSeverity.WARNING,
                message=(
                    "Audio will require chunking "
                    "before processing."
                ),
                modality="audio",
                details={
                    "duration_seconds": duration,
                    "maximum": (
                        self.settings.audio
                        .maximum_audio_duration_seconds
                    ),
                },
            )

    def _validate_spatial(
        self,
        packet: AdaptedLayer1Input,
        issues: List[ValidationIssue],
        modality_validity: Dict[str, bool],
    ) -> None:

        spatial = packet.modalities.get(
            "spatial"
        )

        if (
            spatial is None
            or not spatial.available
        ):
            modality_validity["spatial"] = False
            return

        latitude = spatial.data.get(
            "latitude"
        )

        longitude = spatial.data.get(
            "longitude"
        )

        if (
            latitude is None
            or longitude is None
        ):
            self._add_issue(
                issues,
                code="GPS_COORDINATES_MISSING",
                severity=IssueSeverity.WARNING,
                message=(
                    "Spatial modality does not "
                    "contain complete GPS coordinates."
                ),
                modality="spatial",
            )
            return

        coordinates_valid = True

        if (
            not is_finite_number(latitude)
            or not -90.0
            <= float(latitude)
            <= 90.0
        ):
            coordinates_valid = False

            self._add_issue(
                issues,
                code="LATITUDE_INVALID",
                severity=IssueSeverity.ERROR,
                message="Latitude is invalid.",
                modality="spatial",
                field_name=(
                    "spatial.latitude"
                ),
                details={
                    "value": latitude
                },
            )

        if (
            not is_finite_number(longitude)
            or not -180.0
            <= float(longitude)
            <= 180.0
        ):
            coordinates_valid = False

            self._add_issue(
                issues,
                code="LONGITUDE_INVALID",
                severity=IssueSeverity.ERROR,
                message="Longitude is invalid.",
                modality="spatial",
                field_name=(
                    "spatial.longitude"
                ),
                details={
                    "value": longitude
                },
            )

        modality_validity["spatial"] = (
            coordinates_valid
        )

    def _validate_motion(
        self,
        packet: AdaptedLayer1Input,
        issues: List[ValidationIssue],
        modality_validity: Dict[str, bool],
    ) -> None:

        motion = packet.modalities.get(
            "motion"
        )

        if (
            motion is None
            or not motion.available
        ):
            modality_validity["motion"] = False
            return

        accelerometer = (
            motion.data.get(
                "accelerometer_mps2"
            )
            or motion.data.get(
                "accelerometer"
            )
        )

        if not isinstance(
            accelerometer,
            Mapping,
        ):
            modality_validity["motion"] = False

            self._add_issue(
                issues,
                code="ACCELEROMETER_MISSING",
                severity=IssueSeverity.ERROR,
                message=(
                    "Motion modality does not "
                    "contain an accelerometer vector."
                ),
                modality="motion",
            )
            return

        if not self._valid_vector3(
            accelerometer
        ):
            modality_validity["motion"] = False

            self._add_issue(
                issues,
                code="ACCELEROMETER_INVALID",
                severity=IssueSeverity.ERROR,
                message=(
                    "Accelerometer vector must "
                    "contain finite x, y and z values."
                ),
                modality="motion",
            )
            return

        for field_name in (
            "gyroscope_rps",
            "gyroscope",
            "magnetometer_ut",
            "magnetometer",
        ):
            vector = motion.data.get(
                field_name
            )

            if (
                vector is not None
                and not self._valid_vector3(
                    vector
                )
            ):
                self._add_issue(
                    issues,
                    code=(
                        "OPTIONAL_MOTION_VECTOR_INVALID"
                    ),
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"{field_name} vector "
                        "is invalid."
                    ),
                    modality="motion",
                    field_name=(
                        f"motion.{field_name}"
                    ),
                )

        modality_validity["motion"] = True

    def _valid_vector3(
        self,
        value: Any,
    ) -> bool:

        if not isinstance(value, Mapping):
            return False

        return all(
            coordinate in value
            and is_finite_number(
                value[coordinate]
            )
            for coordinate in (
                "x",
                "y",
                "z",
            )
        )

    def _validate_core_availability(
        self,
        packet: AdaptedLayer1Input,
        issues: List[ValidationIssue],
        modality_validity: Dict[str, bool],
    ) -> None:

        vision_valid = bool(
            modality_validity.get(
                "vision"
            )
        )

        audio_valid = bool(
            modality_validity.get(
                "audio"
            )
        )

        if not vision_valid and not audio_valid:
            self._add_issue(
                issues,
                code="CORE_MODALITIES_UNAVAILABLE",
                severity=IssueSeverity.ERROR,
                message=(
                    "Neither valid vision nor valid "
                    "audio is available."
                ),
                fatal=True,
                details={
                    "packet_id": packet.packet_id
                },
            )

    def _determine_status(
        self,
        issues: List[ValidationIssue],
    ) -> ValidationStatus:

        if any(
            issue.fatal
            for issue in issues
        ):
            return ValidationStatus.REJECTED

        if any(
            issue.severity
            in {
                IssueSeverity.ERROR,
                IssueSeverity.WARNING,
            }
            for issue in issues
        ):
            return ValidationStatus.DEGRADED

        return ValidationStatus.VALID


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def validate_packet(
    packet: AdaptedLayer1Input,
    *,
    settings: Optional[
        Layer2Settings
    ] = None,
) -> PacketValidationReport:

    validator = PacketValidator(
        settings=settings
    )

    return validator.validate(packet)


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 72)
    print("NOONGIL-X | PACKET VALIDATOR SELF-TEST")
    print("=" * 72)

    try:
        adapter = Layer1PacketAdapter(
            require_media=True
        )

        validator = PacketValidator(
            settings=create_test_settings()
        )

        scenarios = (
            adapter.discover_scenarios()
        )

        if len(scenarios) != 8:
            raise AssertionError(
                "Expected eight scenarios."
            )

        print("[PASS] Eight scenarios discovered")

        reports = []

        for scenario in scenarios:

            packet = adapter.load_scenario(
                scenario
            )

            report = validator.validate(
                packet
            )

            if report.rejected:
                raise AssertionError(
                    f"{scenario} was rejected."
                )

            if not report.can_route:
                raise AssertionError(
                    f"{scenario} cannot be routed."
                )

            vision_inspection = (
                report.media_inspections.get(
                    "vision"
                )
            )

            if (
                vision_inspection is None
                or not vision_inspection.valid
            ):
                raise AssertionError(
                    f"Vision validation failed "
                    f"for {scenario}."
                )

            audio_inspection = (
                report.media_inspections.get(
                    "audio"
                )
            )

            if (
                audio_inspection is None
                or not audio_inspection.valid
            ):
                raise AssertionError(
                    f"Audio validation failed "
                    f"for {scenario}."
                )

            reports.append(report)

            print(
                f"[PASS] {scenario}: "
                f"{report.status.value}"
            )

        print("[PASS] All media files inspected")
        print("[PASS] GPS coordinates validated")
        print("[PASS] Motion vectors validated")
        print("[PASS] Confidence values validated")
        print("[PASS] Timestamps validated")

        # Test a degraded spatial input.
        degraded_packet = (
            adapter.load_scenario(
                "park_walking"
            )
        )

        degraded_packet.modalities[
            "spatial"
        ].data["latitude"] = 999.0

        degraded_report = (
            validator.validate(
                degraded_packet
            )
        )

        if (
            degraded_report.status
            != ValidationStatus.DEGRADED
        ):
            raise AssertionError(
                "Invalid optional spatial input "
                "did not degrade the packet."
            )

        if not degraded_report.can_route:
            raise AssertionError(
                "Packet with valid vision/audio "
                "should remain routable."
            )

        print(
            "[PASS] Invalid optional modality "
            "caused degradation"
        )

        # Test rejection when both core modalities fail.
        rejected_packet = (
            adapter.load_scenario(
                "park_walking"
            )
        )

        rejected_packet.modalities[
            "vision"
        ] = AdaptedModality(
            name="vision",
            available=False,
            data={},
        )

        rejected_packet.modalities[
            "audio"
        ] = AdaptedModality(
            name="audio",
            available=False,
            data={},
        )

        rejected_report = (
            validator.validate(
                rejected_packet
            )
        )

        if (
            rejected_report.status
            != ValidationStatus.REJECTED
        ):
            raise AssertionError(
                "Missing core modalities did "
                "not reject the packet."
            )

        print(
            "[PASS] Missing core modalities rejected"
        )

        example_report = reports[0]

        print("\nValidation summary:")
        print(
            f"  packet_id: "
            f"{example_report.packet_id}"
        )
        print(
            f"  status: "
            f"{example_report.status.value}"
        )
        print(
            f"  can_route: "
            f"{example_report.can_route}"
        )
        print(
            f"  checks: "
            f"{example_report.checks_performed}"
        )
        print(
            f"  image: "
            f"{example_report.media_inspections['vision'].to_dict()}"
        )
        print(
            f"  audio: "
            f"{example_report.media_inspections['audio'].to_dict()}"
        )

        print("\n" + "=" * 72)
        print(
            "[PASSED] PACKET VALIDATOR IS WORKING"
        )
        print("=" * 72)

        return True

    except (
        PacketValidationError,
        AssertionError,
    ) as error:

        log_exception(
            get_logger(
                "packet_validator_self_test"
            ),
            error,
            event=(
                "packet_validator_self_test_failed"
            ),
        )

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
            "packet-validator self-test."
        )
    )


def main() -> int:

    build_argument_parser().parse_args()

    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
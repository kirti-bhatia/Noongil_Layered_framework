"""
============================================================
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : Sensor Packet Schema
File    : layer1/schemas/sensor_packet.py
============================================================

Purpose
-------
Defines the standard data structures used by Layer 1 to represent:

1. Source-device information
2. Vision input
3. Audio input
4. Spatial input
5. Motion input
6. User-interaction input
7. Wearable/device input
8. Environmental context
9. Synchronization metadata
10. Sensor-confidence information
11. Missing-modality recovery information
12. Layer-routing information
13. The final Multimodal Sensor Packet

Architectural Boundary
----------------------
This module represents sensor-level information only.

It does NOT perform:
- object detection;
- OCR;
- speech recognition;
- activity recognition;
- scene understanding;
- hazard detection;
- intent reasoning;
- LLM processing.

Those operations belong to higher NOONGIL layers.

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import json
import math
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar


# ============================================================
# TYPE VARIABLES
# ============================================================

T = TypeVar("T")


# ============================================================
# CONSTANTS
# ============================================================

SCHEMA_VERSION = "1.0"

SUPPORTED_MODALITIES = {
    "vision",
    "audio",
    "spatial",
    "motion",
    "interaction",
    "wearable",
    "environment",
}


# ============================================================
# ENUMERATIONS
# ============================================================

class AcquisitionMode(str, Enum):
    """
    Supported Layer 1 acquisition modes.

    These modes control sensor activation and sampling behaviour.
    """

    IDLE = "idle"
    AWARENESS = "awareness"
    READING = "reading"
    NAVIGATION = "navigation"
    EMERGENCY = "emergency"
    LOW_POWER = "low_power"
    DEGRADED_NETWORK = "degraded_network"


class ModalityStatus(str, Enum):
    """
    Indicates how modality data was obtained.
    """

    OBSERVED = "observed"
    INTERPOLATED = "interpolated"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"
    NOT_REQUESTED = "not_requested"


class ReliabilityLevel(str, Enum):
    """
    Human-readable classification for a confidence value.
    """

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNRELIABLE = "unreliable"
    UNKNOWN = "unknown"


class SynchronizationStatus(str, Enum):
    """
    State of multimodal temporal synchronization.
    """

    SYNCHRONIZED = "synchronized"
    PARTIALLY_SYNCHRONIZED = "partially_synchronized"
    UNSYNCHRONIZED = "unsynchronized"
    NOT_PERFORMED = "not_performed"


class InteractionType(str, Enum):
    """
    Explicit user/device interaction categories.
    """

    TOUCH = "touch"
    BUTTON = "button"
    VOICE_TRIGGER = "voice_trigger"
    EMERGENCY_TRIGGER = "emergency_trigger"
    GESTURE_TRIGGER = "gesture_trigger"
    SYSTEM_EVENT = "system_event"
    NONE = "none"


class NetworkType(str, Enum):
    """
    Supported phone-to-laptop communication states.
    """

    WIFI = "wifi"
    MOBILE_DATA = "mobile_data"
    BLUETOOTH = "bluetooth"
    USB = "usb"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


# ============================================================
# GENERAL HELPER FUNCTIONS
# ============================================================

def utc_now_iso() -> str:
    """
    Return the current UTC timestamp in ISO 8601 format.
    """

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def generate_identifier(prefix: str) -> str:
    """
    Generate a short globally unique identifier.

    Example
    -------
    MSP_20260806T001210_4F8A9B21
    """

    normalized_prefix = prefix.strip().upper() or "ID"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    token = uuid.uuid4().hex[:8].upper()

    return f"{normalized_prefix}_{timestamp}_{token}"


def require_non_empty_string(value: str, field_name: str) -> None:
    """
    Validate that a value is a non-empty string.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string."
        )


def require_probability(value: float, field_name: str) -> None:
    """
    Validate a probability or normalized score in the range [0, 1].
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{field_name} must be a numeric value."
        )

    if not math.isfinite(float(value)):
        raise ValueError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0.0 and 1.0."
        )


def require_non_negative(
    value: float,
    field_name: str,
) -> None:
    """
    Validate that a numeric value is finite and non-negative.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{field_name} must be numeric."
        )

    if not math.isfinite(float(value)):
        raise ValueError(
            f"{field_name} must be finite."
        )

    if float(value) < 0:
        raise ValueError(
            f"{field_name} cannot be negative."
        )


def require_positive(
    value: float,
    field_name: str,
) -> None:
    """
    Validate that a numeric value is finite and greater than zero.
    """

    require_non_negative(value, field_name)

    if float(value) == 0:
        raise ValueError(
            f"{field_name} must be greater than zero."
        )


def require_iso_timestamp(
    value: Optional[str],
    field_name: str,
) -> None:
    """
    Validate an ISO 8601 timestamp when a value is provided.
    """

    if value is None:
        return

    require_non_empty_string(value, field_name)

    normalized = value.replace("Z", "+00:00")

    try:
        datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(
            f"{field_name} must use ISO 8601 format. "
            f"Received: {value!r}"
        ) from error


def enum_value(value: Any) -> Any:
    """
    Convert Enum values into their serializable string values.
    """

    if isinstance(value, Enum):
        return value.value

    return value


def make_json_safe(value: Any) -> Any:
    """
    Recursively convert supported Python values into JSON-safe values.
    """

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

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

    if hasattr(value, "__dataclass_fields__"):
        return make_json_safe(asdict(value))

    return value


def reliability_level_from_score(
    score: Optional[float],
) -> ReliabilityLevel:
    """
    Convert a numerical confidence value into a reliability level.
    """

    if score is None:
        return ReliabilityLevel.UNKNOWN

    require_probability(score, "score")

    if score >= 0.80:
        return ReliabilityLevel.HIGH

    if score >= 0.55:
        return ReliabilityLevel.MODERATE

    if score >= 0.30:
        return ReliabilityLevel.LOW

    return ReliabilityLevel.UNRELIABLE


# ============================================================
# BASE VALIDATION MIXIN
# ============================================================

class Validatable:
    """
    Base interface for schema objects that support validation.
    """

    def validate(self) -> None:
        """
        Validate the current schema object.

        Child classes should override this method.
        """

        raise NotImplementedError


# ============================================================
# DEVICE SCHEMA
# ============================================================

@dataclass
class SourceDevice(Validatable):
    """
    Information about the smartphone or sensor device producing data.
    """

    device_id: str
    device_type: str = "android_smartphone"
    device_name: Optional[str] = None
    operating_system: Optional[str] = None

    network_type: NetworkType = NetworkType.UNKNOWN
    network_strength: Optional[float] = None
    network_latency_ms: Optional[float] = None

    battery_level: Optional[float] = None
    is_charging: Optional[bool] = None

    application_version: Optional[str] = None
    available_sensors: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty_string(self.device_id, "device_id")
        require_non_empty_string(self.device_type, "device_type")

        if self.network_strength is not None:
            require_probability(
                self.network_strength,
                "network_strength",
            )

        if self.network_latency_ms is not None:
            require_non_negative(
                self.network_latency_ms,
                "network_latency_ms",
            )

        if self.battery_level is not None:
            require_probability(
                self.battery_level,
                "battery_level",
            )

        for sensor_name in self.available_sensors:
            require_non_empty_string(
                sensor_name,
                "available_sensors item",
            )


# ============================================================
# COMMON MODALITY METADATA
# ============================================================

@dataclass
class ModalityMetadata(Validatable):
    """
    Metadata shared by all sensor modalities.
    """

    modality: str
    status: ModalityStatus

    source_timestamp: Optional[str] = None
    arrival_timestamp: Optional[str] = None

    sequence_number: Optional[int] = None
    sampling_rate_hz: Optional[float] = None
    latency_ms: Optional[float] = None

    source_device_id: Optional[str] = None
    data_reference: Optional[str] = None

    preprocessing_steps: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty_string(self.modality, "modality")

        if self.modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported modality: {self.modality!r}. "
                f"Supported modalities: "
                f"{sorted(SUPPORTED_MODALITIES)}"
            )

        require_iso_timestamp(
            self.source_timestamp,
            "source_timestamp",
        )

        require_iso_timestamp(
            self.arrival_timestamp,
            "arrival_timestamp",
        )

        if self.sequence_number is not None:
            if (
                isinstance(self.sequence_number, bool)
                or not isinstance(self.sequence_number, int)
                or self.sequence_number < 0
            ):
                raise ValueError(
                    "sequence_number must be a non-negative integer."
                )

        if self.sampling_rate_hz is not None:
            require_positive(
                self.sampling_rate_hz,
                "sampling_rate_hz",
            )

        if self.latency_ms is not None:
            require_non_negative(
                self.latency_ms,
                "latency_ms",
            )

        for step in self.preprocessing_steps:
            require_non_empty_string(
                step,
                "preprocessing_steps item",
            )

        for limitation in self.limitations:
            require_non_empty_string(
                limitation,
                "limitations item",
            )


# ============================================================
# VISION SCHEMA
# ============================================================

@dataclass
class VisionData(Validatable):
    """
    Signal-level camera input prepared by Layer 1.

    No semantic vision result is stored here.
    """

    metadata: ModalityMetadata = field(
        default_factory=lambda: ModalityMetadata(
            modality="vision",
            status=ModalityStatus.NOT_REQUESTED,
        )
    )

    frame_id: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None

    encoding: Optional[str] = None
    color_space: Optional[str] = None
    frame_rate_fps: Optional[float] = None

    brightness_score: Optional[float] = None
    sharpness_score: Optional[float] = None
    contrast_score: Optional[float] = None
    frame_integrity_score: Optional[float] = None

    frame_path: Optional[str] = None
    encoded_frame: Optional[str] = None

    def validate(self) -> None:
        self.metadata.validate()

        if self.metadata.modality != "vision":
            raise ValueError(
                "VisionData metadata modality must be 'vision'."
            )

        for name, value in (
            ("width", self.width),
            ("height", self.height),
            ("channels", self.channels),
        ):
            if value is not None:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                ):
                    raise ValueError(
                        f"{name} must be a positive integer."
                    )

        if self.frame_rate_fps is not None:
            require_positive(
                self.frame_rate_fps,
                "frame_rate_fps",
            )

        for name, score in (
            ("brightness_score", self.brightness_score),
            ("sharpness_score", self.sharpness_score),
            ("contrast_score", self.contrast_score),
            (
                "frame_integrity_score",
                self.frame_integrity_score,
            ),
        ):
            if score is not None:
                require_probability(score, name)

        if (
            self.metadata.status == ModalityStatus.OBSERVED
            and not any(
                (
                    self.frame_id,
                    self.frame_path,
                    self.encoded_frame,
                    self.metadata.data_reference,
                )
            )
        ):
            raise ValueError(
                "Observed vision data requires a frame identifier, "
                "path, encoded frame, or data reference."
            )


# ============================================================
# AUDIO SCHEMA
# ============================================================

@dataclass
class AudioData(Validatable):
    """
    Signal-level microphone input prepared by Layer 1.
    """

    metadata: ModalityMetadata = field(
        default_factory=lambda: ModalityMetadata(
            modality="audio",
            status=ModalityStatus.NOT_REQUESTED,
        )
    )

    chunk_id: Optional[str] = None
    sample_rate_hz: Optional[int] = None
    channels: Optional[int] = None
    sample_width_bits: Optional[int] = None
    duration_ms: Optional[float] = None

    encoding: Optional[str] = None

    amplitude_score: Optional[float] = None
    signal_to_noise_score: Optional[float] = None
    clipping_ratio: Optional[float] = None
    silence_ratio: Optional[float] = None
    packet_integrity_score: Optional[float] = None

    audio_path: Optional[str] = None
    encoded_audio: Optional[str] = None

    feature_type: Optional[str] = None
    feature_reference: Optional[str] = None

    def validate(self) -> None:
        self.metadata.validate()

        if self.metadata.modality != "audio":
            raise ValueError(
                "AudioData metadata modality must be 'audio'."
            )

        if self.sample_rate_hz is not None:
            if (
                isinstance(self.sample_rate_hz, bool)
                or not isinstance(self.sample_rate_hz, int)
                or self.sample_rate_hz <= 0
            ):
                raise ValueError(
                    "sample_rate_hz must be a positive integer."
                )

        if self.channels is not None:
            if (
                isinstance(self.channels, bool)
                or not isinstance(self.channels, int)
                or self.channels <= 0
            ):
                raise ValueError(
                    "channels must be a positive integer."
                )

        if self.sample_width_bits is not None:
            if self.sample_width_bits not in {8, 16, 24, 32}:
                raise ValueError(
                    "sample_width_bits must be one of "
                    "{8, 16, 24, 32}."
                )

        if self.duration_ms is not None:
            require_non_negative(
                self.duration_ms,
                "duration_ms",
            )

        for name, score in (
            ("amplitude_score", self.amplitude_score),
            (
                "signal_to_noise_score",
                self.signal_to_noise_score,
            ),
            ("clipping_ratio", self.clipping_ratio),
            ("silence_ratio", self.silence_ratio),
            (
                "packet_integrity_score",
                self.packet_integrity_score,
            ),
        ):
            if score is not None:
                require_probability(score, name)

        if (
            self.metadata.status == ModalityStatus.OBSERVED
            and not any(
                (
                    self.chunk_id,
                    self.audio_path,
                    self.encoded_audio,
                    self.metadata.data_reference,
                )
            )
        ):
            raise ValueError(
                "Observed audio data requires a chunk identifier, "
                "path, encoded audio, or data reference."
            )


# ============================================================
# SPATIAL SCHEMA
# ============================================================

@dataclass
class SpatialData(Validatable):
    """
    GPS and orientation data acquired from the smartphone.
    """

    metadata: ModalityMetadata = field(
        default_factory=lambda: ModalityMetadata(
            modality="spatial",
            status=ModalityStatus.NOT_REQUESTED,
        )
    )

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_meters: Optional[float] = None

    horizontal_accuracy_meters: Optional[float] = None
    vertical_accuracy_meters: Optional[float] = None

    heading_degrees: Optional[float] = None
    heading_accuracy_degrees: Optional[float] = None

    speed_meters_per_second: Optional[float] = None
    provider: Optional[str] = None

    def validate(self) -> None:
        self.metadata.validate()

        if self.metadata.modality != "spatial":
            raise ValueError(
                "SpatialData metadata modality must be 'spatial'."
            )

        if self.latitude is not None:
            if not -90.0 <= float(self.latitude) <= 90.0:
                raise ValueError(
                    "latitude must be between -90 and 90 degrees."
                )

        if self.longitude is not None:
            if not -180.0 <= float(self.longitude) <= 180.0:
                raise ValueError(
                    "longitude must be between "
                    "-180 and 180 degrees."
                )

        if self.horizontal_accuracy_meters is not None:
            require_non_negative(
                self.horizontal_accuracy_meters,
                "horizontal_accuracy_meters",
            )

        if self.vertical_accuracy_meters is not None:
            require_non_negative(
                self.vertical_accuracy_meters,
                "vertical_accuracy_meters",
            )

        if self.heading_degrees is not None:
            if not 0.0 <= float(self.heading_degrees) < 360.0:
                raise ValueError(
                    "heading_degrees must be in the range [0, 360)."
                )

        if self.heading_accuracy_degrees is not None:
            require_non_negative(
                self.heading_accuracy_degrees,
                "heading_accuracy_degrees",
            )

        if self.speed_meters_per_second is not None:
            require_non_negative(
                self.speed_meters_per_second,
                "speed_meters_per_second",
            )

        if (
            self.metadata.status == ModalityStatus.OBSERVED
            and self.latitude is None
            and self.longitude is None
            and self.heading_degrees is None
        ):
            raise ValueError(
                "Observed spatial data must contain a location "
                "or heading value."
            )


# ============================================================
# MOTION SCHEMA
# ============================================================

@dataclass
class Vector3(Validatable):
    """
    Three-axis sensor vector.
    """

    x: float
    y: float
    z: float
    unit: str

    def validate(self) -> None:
        for name, value in (
            ("x", self.x),
            ("y", self.y),
            ("z", self.z),
        ):
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    f"{name} must be numeric."
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{name} must be finite."
                )

        require_non_empty_string(self.unit, "unit")


@dataclass
class MotionData(Validatable):
    """
    Smartphone accelerometer, gyroscope, and motion metadata.
    """

    metadata: ModalityMetadata = field(
        default_factory=lambda: ModalityMetadata(
            modality="motion",
            status=ModalityStatus.NOT_REQUESTED,
        )
    )

    accelerometer: Optional[Vector3] = None
    gyroscope: Optional[Vector3] = None
    magnetometer: Optional[Vector3] = None
    linear_acceleration: Optional[Vector3] = None

    motion_intensity: Optional[float] = None
    orientation_change_score: Optional[float] = None
    sampling_continuity_score: Optional[float] = None
    sensor_saturation_score: Optional[float] = None

    def validate(self) -> None:
        self.metadata.validate()

        if self.metadata.modality != "motion":
            raise ValueError(
                "MotionData metadata modality must be 'motion'."
            )

        for vector in (
            self.accelerometer,
            self.gyroscope,
            self.magnetometer,
            self.linear_acceleration,
        ):
            if vector is not None:
                vector.validate()

        for name, score in (
            ("motion_intensity", self.motion_intensity),
            (
                "orientation_change_score",
                self.orientation_change_score,
            ),
            (
                "sampling_continuity_score",
                self.sampling_continuity_score,
            ),
            (
                "sensor_saturation_score",
                self.sensor_saturation_score,
            ),
        ):
            if score is not None:
                require_probability(score, name)

        if (
            self.metadata.status == ModalityStatus.OBSERVED
            and not any(
                (
                    self.accelerometer,
                    self.gyroscope,
                    self.magnetometer,
                    self.linear_acceleration,
                )
            )
        ):
            raise ValueError(
                "Observed motion data must contain at least "
                "one sensor vector."
            )


# ============================================================
# INTERACTION SCHEMA
# ============================================================

@dataclass
class InteractionData(Validatable):
    """
    Explicit user interaction recorded by Layer 1.
    """

    metadata: ModalityMetadata = field(
        default_factory=lambda: ModalityMetadata(
            modality="interaction",
            status=ModalityStatus.NOT_REQUESTED,
        )
    )

    interaction_id: Optional[str] = None
    interaction_type: InteractionType = InteractionType.NONE
    action: Optional[str] = None

    value: Any = None
    emergency_flag: bool = False

    def validate(self) -> None:
        self.metadata.validate()

        if self.metadata.modality != "interaction":
            raise ValueError(
                "InteractionData metadata modality must be "
                "'interaction'."
            )

        if (
            self.metadata.status == ModalityStatus.OBSERVED
            and self.interaction_type == InteractionType.NONE
        ):
            raise ValueError(
                "Observed interaction data requires an "
                "interaction_type."
            )

        if self.action is not None:
            require_non_empty_string(self.action, "action")


# ============================================================
# WEARABLE / CONNECTED DEVICE SCHEMA
# ============================================================

@dataclass
class WearableData(Validatable):
    """
    Connected wearable or earphone metadata.

    For NOONGIL v1 this can represent the Realme Buds T200.
    """

    metadata: ModalityMetadata = field(
        default_factory=lambda: ModalityMetadata(
            modality="wearable",
            status=ModalityStatus.NOT_REQUESTED,
        )
    )

    device_id: Optional[str] = None
    device_name: Optional[str] = None
    device_type: Optional[str] = None

    connected: bool = False
    connection_type: Optional[str] = None

    battery_level: Optional[float] = None
    capabilities: List[str] = field(default_factory=list)

    microphone_available: Optional[bool] = None
    audio_output_available: Optional[bool] = None
    haptic_output_available: Optional[bool] = None

    def validate(self) -> None:
        self.metadata.validate()

        if self.metadata.modality != "wearable":
            raise ValueError(
                "WearableData metadata modality must be 'wearable'."
            )

        if self.battery_level is not None:
            require_probability(
                self.battery_level,
                "battery_level",
            )

        for capability in self.capabilities:
            require_non_empty_string(
                capability,
                "capabilities item",
            )

        if self.connected and not any(
            (
                self.device_id,
                self.device_name,
                self.device_type,
            )
        ):
            raise ValueError(
                "A connected wearable must include device identity."
            )


# ============================================================
# ENVIRONMENTAL CONTEXT SCHEMA
# ============================================================

@dataclass
class EnvironmentalData(Validatable):
    """
    Raw environmental context obtained from external services.

    Layer 1 stores the acquired values without reasoning about them.
    """

    metadata: ModalityMetadata = field(
        default_factory=lambda: ModalityMetadata(
            modality="environment",
            status=ModalityStatus.NOT_REQUESTED,
        )
    )

    weather: Dict[str, Any] = field(default_factory=dict)
    map_context: Dict[str, Any] = field(default_factory=dict)
    traffic_context: Dict[str, Any] = field(default_factory=dict)
    transport_context: Dict[str, Any] = field(default_factory=dict)

    provider_names: List[str] = field(default_factory=list)
    retrieved_at: Optional[str] = None
    cache_age_seconds: Optional[float] = None

    def validate(self) -> None:
        self.metadata.validate()

        if self.metadata.modality != "environment":
            raise ValueError(
                "EnvironmentalData metadata modality must be "
                "'environment'."
            )

        require_iso_timestamp(
            self.retrieved_at,
            "retrieved_at",
        )

        if self.cache_age_seconds is not None:
            require_non_negative(
                self.cache_age_seconds,
                "cache_age_seconds",
            )

        for provider in self.provider_names:
            require_non_empty_string(
                provider,
                "provider_names item",
            )


# ============================================================
# SYNCHRONIZATION SCHEMA
# ============================================================

@dataclass
class ModalityAlignment(Validatable):
    """
    Temporal-alignment result for one modality.
    """

    modality: str
    source_timestamp: Optional[str] = None
    offset_ms: Optional[float] = None
    tolerance_ms: Optional[float] = None
    alignment_score: Optional[float] = None
    within_window: bool = False

    def validate(self) -> None:
        require_non_empty_string(self.modality, "modality")

        if self.modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported alignment modality: {self.modality!r}."
            )

        require_iso_timestamp(
            self.source_timestamp,
            "source_timestamp",
        )

        if self.offset_ms is not None:
            require_non_negative(
                abs(float(self.offset_ms)),
                "absolute offset_ms",
            )

        if self.tolerance_ms is not None:
            require_non_negative(
                self.tolerance_ms,
                "tolerance_ms",
            )

        if self.alignment_score is not None:
            require_probability(
                self.alignment_score,
                "alignment_score",
            )


@dataclass
class SynchronizationData(Validatable):
    """
    Temporal synchronization metadata for the final packet.
    """

    synchronization_id: str = field(
        default_factory=lambda: generate_identifier("SYNC")
    )

    status: SynchronizationStatus = (
        SynchronizationStatus.NOT_PERFORMED
    )

    reference_timestamp: Optional[str] = None
    window_ms: Optional[float] = None

    aligned_modalities: List[str] = field(default_factory=list)
    missing_modalities: List[str] = field(default_factory=list)
    stale_modalities: List[str] = field(default_factory=list)

    modality_alignments: Dict[str, ModalityAlignment] = field(
        default_factory=dict
    )

    overall_alignment_score: Optional[float] = None

    def validate(self) -> None:
        require_non_empty_string(
            self.synchronization_id,
            "synchronization_id",
        )

        require_iso_timestamp(
            self.reference_timestamp,
            "reference_timestamp",
        )

        if self.window_ms is not None:
            require_non_negative(
                self.window_ms,
                "window_ms",
            )

        if self.overall_alignment_score is not None:
            require_probability(
                self.overall_alignment_score,
                "overall_alignment_score",
            )

        for group_name, modalities in (
            ("aligned_modalities", self.aligned_modalities),
            ("missing_modalities", self.missing_modalities),
            ("stale_modalities", self.stale_modalities),
        ):
            for modality in modalities:
                if modality not in SUPPORTED_MODALITIES:
                    raise ValueError(
                        f"{group_name} contains unsupported "
                        f"modality {modality!r}."
                    )

        for modality, alignment in self.modality_alignments.items():
            if modality not in SUPPORTED_MODALITIES:
                raise ValueError(
                    f"Unsupported modality alignment key: "
                    f"{modality!r}."
                )

            alignment.validate()

            if alignment.modality != modality:
                raise ValueError(
                    "Modality-alignment dictionary key must match "
                    "ModalityAlignment.modality."
                )


# ============================================================
# CONFIDENCE SCHEMA
# ============================================================

@dataclass
class ModalityConfidence(Validatable):
    """
    Confidence and reliability breakdown for one modality.
    """

    modality: str

    signal_quality: Optional[float] = None
    temporal_alignment: Optional[float] = None
    sensor_health: Optional[float] = None
    freshness: Optional[float] = None
    cross_modal_agreement: Optional[float] = None

    conflict_score: Optional[float] = None
    final_confidence: Optional[float] = None

    reliability_level: ReliabilityLevel = ReliabilityLevel.UNKNOWN

    limitations: List[str] = field(default_factory=list)
    calculation_details: Dict[str, Any] = field(
        default_factory=dict
    )

    def calculate_reliability_level(self) -> ReliabilityLevel:
        """
        Update and return the reliability classification.
        """

        self.reliability_level = reliability_level_from_score(
            self.final_confidence
        )

        return self.reliability_level

    def validate(self) -> None:
        require_non_empty_string(self.modality, "modality")

        if self.modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported confidence modality: {self.modality!r}."
            )

        for name, score in (
            ("signal_quality", self.signal_quality),
            ("temporal_alignment", self.temporal_alignment),
            ("sensor_health", self.sensor_health),
            ("freshness", self.freshness),
            (
                "cross_modal_agreement",
                self.cross_modal_agreement,
            ),
            ("conflict_score", self.conflict_score),
            ("final_confidence", self.final_confidence),
        ):
            if score is not None:
                require_probability(score, name)

        expected_level = reliability_level_from_score(
            self.final_confidence
        )

        if (
            self.final_confidence is not None
            and self.reliability_level
            not in {
                ReliabilityLevel.UNKNOWN,
                expected_level,
            }
        ):
            raise ValueError(
                f"reliability_level {self.reliability_level.value!r} "
                f"does not match final_confidence "
                f"{self.final_confidence}."
            )

        for limitation in self.limitations:
            require_non_empty_string(
                limitation,
                "limitations item",
            )


@dataclass
class SensorConfidenceData(Validatable):
    """
    Confidence data for all modalities in one sensor packet.
    """

    confidence_id: str = field(
        default_factory=lambda: generate_identifier("CONF")
    )

    modality_confidences: Dict[str, ModalityConfidence] = field(
        default_factory=dict
    )

    normalized_weights: Dict[str, float] = field(
        default_factory=dict
    )

    overall_confidence: Optional[float] = None
    generated_at: str = field(default_factory=utc_now_iso)

    def validate(self) -> None:
        require_non_empty_string(
            self.confidence_id,
            "confidence_id",
        )

        require_iso_timestamp(
            self.generated_at,
            "generated_at",
        )

        if self.overall_confidence is not None:
            require_probability(
                self.overall_confidence,
                "overall_confidence",
            )

        for modality, confidence in self.modality_confidences.items():
            if modality not in SUPPORTED_MODALITIES:
                raise ValueError(
                    f"Unsupported confidence key: {modality!r}."
                )

            confidence.validate()

            if confidence.modality != modality:
                raise ValueError(
                    "Confidence dictionary key must match "
                    "ModalityConfidence.modality."
                )

        total_weight = 0.0

        for modality, weight in self.normalized_weights.items():
            if modality not in SUPPORTED_MODALITIES:
                raise ValueError(
                    f"Unsupported weight modality: {modality!r}."
                )

            require_probability(
                weight,
                f"normalized_weights[{modality!r}]",
            )

            total_weight += float(weight)

        if self.normalized_weights:
            if not math.isclose(
                total_weight,
                1.0,
                rel_tol=1e-5,
                abs_tol=1e-5,
            ):
                raise ValueError(
                    "normalized_weights must sum to 1.0. "
                    f"Current total: {total_weight}"
                )


# ============================================================
# RECOVERY SCHEMA
# ============================================================

@dataclass
class ModalityRecovery(Validatable):
    """
    Recovery result for an unavailable or degraded modality.
    """

    modality: str
    original_status: ModalityStatus
    final_status: ModalityStatus

    recovery_used: bool = False
    recovery_method: Optional[str] = None
    source_modalities: List[str] = field(default_factory=list)
    recovery_confidence: Optional[float] = None

    notes: List[str] = field(default_factory=list)

    def validate(self) -> None:
        require_non_empty_string(self.modality, "modality")

        if self.modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported recovery modality: {self.modality!r}."
            )

        if self.recovery_confidence is not None:
            require_probability(
                self.recovery_confidence,
                "recovery_confidence",
            )

        if self.recovery_used and not self.recovery_method:
            raise ValueError(
                "recovery_method is required when recovery_used "
                "is True."
            )

        if not self.recovery_used and self.recovery_method is not None:
            raise ValueError(
                "recovery_method must be None when recovery_used "
                "is False."
            )

        if self.recovery_used and self.final_status not in {
            ModalityStatus.INTERPOLATED,
            ModalityStatus.ESTIMATED,
            ModalityStatus.OBSERVED,
        }:
            raise ValueError(
                "Recovered modality must end as observed, "
                "interpolated, or estimated."
            )

        for source_modality in self.source_modalities:
            if source_modality not in SUPPORTED_MODALITIES:
                raise ValueError(
                    f"Unsupported recovery source modality: "
                    f"{source_modality!r}."
                )


@dataclass
class RecoveryData(Validatable):
    """
    Packet-level missing-modality recovery report.
    """

    recovery_used: bool = False
    recovered_modalities: List[str] = field(default_factory=list)
    unavailable_modalities: List[str] = field(default_factory=list)

    modality_recoveries: Dict[str, ModalityRecovery] = field(
        default_factory=dict
    )

    generated_at: str = field(default_factory=utc_now_iso)

    def validate(self) -> None:
        require_iso_timestamp(
            self.generated_at,
            "generated_at",
        )

        for group_name, modalities in (
            ("recovered_modalities", self.recovered_modalities),
            (
                "unavailable_modalities",
                self.unavailable_modalities,
            ),
        ):
            for modality in modalities:
                if modality not in SUPPORTED_MODALITIES:
                    raise ValueError(
                        f"{group_name} contains unsupported "
                        f"modality {modality!r}."
                    )

        if self.recovered_modalities and not self.recovery_used:
            raise ValueError(
                "recovery_used must be True when "
                "recovered_modalities is not empty."
            )

        for modality, recovery in self.modality_recoveries.items():
            if modality not in SUPPORTED_MODALITIES:
                raise ValueError(
                    f"Unsupported recovery key: {modality!r}."
                )

            recovery.validate()

            if recovery.modality != modality:
                raise ValueError(
                    "Recovery dictionary key must match "
                    "ModalityRecovery.modality."
                )


# ============================================================
# ROUTING SCHEMA
# ============================================================

@dataclass
class RoutingData(Validatable):
    """
    Layer-to-layer packet routing information.
    """

    source_layer: str = "layer1"
    destination_layer: str = "layer2"

    semantic_processing_performed: bool = False
    packet_priority: str = "normal"

    dispatch_status: str = "pending"
    dispatched_at: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty_string(
            self.source_layer,
            "source_layer",
        )

        require_non_empty_string(
            self.destination_layer,
            "destination_layer",
        )

        require_non_empty_string(
            self.packet_priority,
            "packet_priority",
        )

        require_non_empty_string(
            self.dispatch_status,
            "dispatch_status",
        )

        require_iso_timestamp(
            self.dispatched_at,
            "dispatched_at",
        )

        if self.source_layer.lower() != "layer1":
            raise ValueError(
                "Layer 1 sensor packets must use source_layer='layer1'."
            )

        if self.semantic_processing_performed:
            raise ValueError(
                "Layer 1 must not mark semantic processing as performed."
            )


# ============================================================
# COMPLETE MODALITY COLLECTION
# ============================================================

@dataclass
class Modalities(Validatable):
    """
    Collection of all supported Layer 1 modalities.
    """

    vision: VisionData = field(default_factory=VisionData)
    audio: AudioData = field(default_factory=AudioData)
    spatial: SpatialData = field(default_factory=SpatialData)
    motion: MotionData = field(default_factory=MotionData)
    interaction: InteractionData = field(
        default_factory=InteractionData
    )
    wearable: WearableData = field(default_factory=WearableData)
    environment: EnvironmentalData = field(
        default_factory=EnvironmentalData
    )

    def validate(self) -> None:
        self.vision.validate()
        self.audio.validate()
        self.spatial.validate()
        self.motion.validate()
        self.interaction.validate()
        self.wearable.validate()
        self.environment.validate()

    def status_map(self) -> Dict[str, str]:
        """
        Return the current status of every modality.
        """

        return {
            "vision": self.vision.metadata.status.value,
            "audio": self.audio.metadata.status.value,
            "spatial": self.spatial.metadata.status.value,
            "motion": self.motion.metadata.status.value,
            "interaction": self.interaction.metadata.status.value,
            "wearable": self.wearable.metadata.status.value,
            "environment": self.environment.metadata.status.value,
        }

    def available_modalities(self) -> List[str]:
        """
        Return modalities containing usable data.
        """

        usable_statuses = {
            ModalityStatus.OBSERVED,
            ModalityStatus.INTERPOLATED,
            ModalityStatus.ESTIMATED,
        }

        modality_objects = {
            "vision": self.vision,
            "audio": self.audio,
            "spatial": self.spatial,
            "motion": self.motion,
            "interaction": self.interaction,
            "wearable": self.wearable,
            "environment": self.environment,
        }

        return [
            name
            for name, modality_object in modality_objects.items()
            if modality_object.metadata.status in usable_statuses
        ]


# ============================================================
# FINAL MULTIMODAL SENSOR PACKET
# ============================================================

@dataclass
class MultimodalSensorPacket(Validatable):
    """
    Final Layer 1 output forwarded to Layer 2.
    """

    packet_id: str = field(
        default_factory=lambda: generate_identifier("MSP")
    )

    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now_iso)

    source_device: SourceDevice = field(
        default_factory=lambda: SourceDevice(
            device_id="UNKNOWN_DEVICE"
        )
    )

    acquisition_mode: AcquisitionMode = AcquisitionMode.IDLE

    modalities: Modalities = field(default_factory=Modalities)

    synchronization: SynchronizationData = field(
        default_factory=SynchronizationData
    )

    sensor_confidence: SensorConfidenceData = field(
        default_factory=SensorConfidenceData
    )

    recovery: RecoveryData = field(default_factory=RecoveryData)
    routing: RoutingData = field(default_factory=RoutingData)

    packet_sequence_number: Optional[int] = None
    scenario_name: Optional[str] = None

    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate the complete Layer 1 packet.

        Raises
        ------
        TypeError
            When a field has an invalid type.

        ValueError
            When a field contains an invalid value.
        """

        require_non_empty_string(
            self.packet_id,
            "packet_id",
        )

        require_non_empty_string(
            self.schema_version,
            "schema_version",
        )

        require_iso_timestamp(
            self.created_at,
            "created_at",
        )

        if self.packet_sequence_number is not None:
            if (
                isinstance(self.packet_sequence_number, bool)
                or not isinstance(self.packet_sequence_number, int)
                or self.packet_sequence_number < 0
            ):
                raise ValueError(
                    "packet_sequence_number must be a "
                    "non-negative integer."
                )

        if self.scenario_name is not None:
            require_non_empty_string(
                self.scenario_name,
                "scenario_name",
            )

        self.source_device.validate()
        self.modalities.validate()
        self.synchronization.validate()
        self.sensor_confidence.validate()
        self.recovery.validate()
        self.routing.validate()

        for warning in self.warnings:
            require_non_empty_string(
                warning,
                "warnings item",
            )

        self._validate_cross_section_consistency()

    def _validate_cross_section_consistency(self) -> None:
        """
        Validate relationships between different packet sections.
        """

        modality_statuses = self.modalities.status_map()

        available_modalities = set(
            self.modalities.available_modalities()
        )

        synchronized_modalities = set(
            self.synchronization.aligned_modalities
        )

        impossible_synchronized = (
            synchronized_modalities - available_modalities
        )

        if impossible_synchronized:
            raise ValueError(
                "Synchronization lists unusable modalities as aligned: "
                f"{sorted(impossible_synchronized)}"
            )

        for modality in self.recovery.recovered_modalities:
            status = modality_statuses.get(modality)

            if status not in {
                ModalityStatus.OBSERVED.value,
                ModalityStatus.INTERPOLATED.value,
                ModalityStatus.ESTIMATED.value,
            }:
                raise ValueError(
                    f"Recovered modality {modality!r} has "
                    f"incompatible status {status!r}."
                )

        for modality in self.recovery.unavailable_modalities:
            status = modality_statuses.get(modality)

            if status != ModalityStatus.UNAVAILABLE.value:
                raise ValueError(
                    f"Unavailable modality {modality!r} must use "
                    "status='unavailable'."
                )

        confidence_modalities = set(
            self.sensor_confidence.modality_confidences.keys()
        )

        unknown_confidences = (
            confidence_modalities - SUPPORTED_MODALITIES
        )

        if unknown_confidences:
            raise ValueError(
                "Confidence data contains unknown modalities: "
                f"{sorted(unknown_confidences)}"
            )

    def to_dict(
        self,
        *,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert the packet into a JSON-compatible dictionary.
        """

        if validate:
            self.validate()

        return make_json_safe(asdict(self))

    def to_json(
        self,
        *,
        indent: int = 4,
        validate: bool = True,
    ) -> str:
        """
        Serialize the packet as formatted JSON.
        """

        return json.dumps(
            self.to_dict(validate=validate),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )

    def save_json(
        self,
        file_path: str | Path,
        *,
        indent: int = 4,
        validate: bool = True,
    ) -> Path:
        """
        Validate and save the packet to a JSON file.
        """

        output_path = Path(file_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        payload = self.to_json(
            indent=indent,
            validate=validate,
        )

        try:
            temporary_path.write_text(
                payload,
                encoding="utf-8",
            )

            temporary_path.replace(output_path)

        except OSError as error:
            if temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

            raise OSError(
                f"Unable to save sensor packet to "
                f"{output_path}: {error}"
            ) from error

        return output_path

    def summary(self) -> Dict[str, Any]:
        """
        Return a compact diagnostic summary of the packet.
        """

        return {
            "packet_id": self.packet_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "acquisition_mode": self.acquisition_mode.value,
            "device_id": self.source_device.device_id,
            "available_modalities": (
                self.modalities.available_modalities()
            ),
            "modality_statuses": self.modalities.status_map(),
            "synchronization_status": (
                self.synchronization.status.value
            ),
            "overall_confidence": (
                self.sensor_confidence.overall_confidence
            ),
            "recovery_used": self.recovery.recovery_used,
            "destination_layer": (
                self.routing.destination_layer
            ),
        }


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_empty_sensor_packet(
    *,
    device_id: str,
    acquisition_mode: AcquisitionMode = AcquisitionMode.IDLE,
    device_name: Optional[str] = None,
    scenario_name: Optional[str] = None,
) -> MultimodalSensorPacket:
    """
    Create a valid initial packet containing no observed modalities.

    This factory is useful at the start of every Layer 1 cycle.
    """

    packet = MultimodalSensorPacket(
        source_device=SourceDevice(
            device_id=device_id,
            device_name=device_name,
        ),
        acquisition_mode=acquisition_mode,
        scenario_name=scenario_name,
    )

    packet.validate()
    return packet


def create_demo_sensor_packet() -> MultimodalSensorPacket:
    """
    Create a realistic sample packet for local schema testing.
    """

    current_time = utc_now_iso()

    vision = VisionData(
        metadata=ModalityMetadata(
            modality="vision",
            status=ModalityStatus.OBSERVED,
            source_timestamp=current_time,
            arrival_timestamp=current_time,
            sequence_number=101,
            sampling_rate_hz=15.0,
            latency_ms=18.4,
            source_device_id="PHONE_001",
            data_reference="FRAME_000101",
            preprocessing_steps=[
                "decoded",
                "resized",
                "normalized",
            ],
        ),
        frame_id="FRAME_000101",
        width=640,
        height=480,
        channels=3,
        encoding="jpeg",
        color_space="RGB",
        frame_rate_fps=15.0,
        brightness_score=0.82,
        sharpness_score=0.76,
        contrast_score=0.71,
        frame_integrity_score=0.98,
        frame_path="output/layer1/raw/vision/frame_000101.jpg",
    )

    audio = AudioData(
        metadata=ModalityMetadata(
            modality="audio",
            status=ModalityStatus.OBSERVED,
            source_timestamp=current_time,
            arrival_timestamp=current_time,
            sequence_number=52,
            sampling_rate_hz=16_000.0,
            latency_ms=24.0,
            source_device_id="PHONE_001",
            data_reference="AUDIO_000052",
            preprocessing_steps=[
                "decoded",
                "mono_conversion",
                "amplitude_normalization",
            ],
        ),
        chunk_id="AUDIO_000052",
        sample_rate_hz=16_000,
        channels=1,
        sample_width_bits=16,
        duration_ms=1000.0,
        encoding="pcm_s16le",
        amplitude_score=0.69,
        signal_to_noise_score=0.64,
        clipping_ratio=0.01,
        silence_ratio=0.12,
        packet_integrity_score=0.99,
        audio_path="output/layer1/raw/audio/audio_000052.wav",
    )

    spatial = SpatialData(
        metadata=ModalityMetadata(
            modality="spatial",
            status=ModalityStatus.OBSERVED,
            source_timestamp=current_time,
            arrival_timestamp=current_time,
            sequence_number=18,
            sampling_rate_hz=1.0,
            latency_ms=42.0,
            source_device_id="PHONE_001",
        ),
        latitude=31.6340,
        longitude=74.8720,
        altitude_meters=234.0,
        horizontal_accuracy_meters=8.4,
        heading_degrees=83.2,
        speed_meters_per_second=0.9,
        provider="gps",
    )

    motion = MotionData(
        metadata=ModalityMetadata(
            modality="motion",
            status=ModalityStatus.OBSERVED,
            source_timestamp=current_time,
            arrival_timestamp=current_time,
            sequence_number=320,
            sampling_rate_hz=50.0,
            latency_ms=7.5,
            source_device_id="PHONE_001",
        ),
        accelerometer=Vector3(
            x=0.18,
            y=9.72,
            z=0.41,
            unit="m/s^2",
        ),
        gyroscope=Vector3(
            x=0.02,
            y=0.11,
            z=-0.04,
            unit="rad/s",
        ),
        motion_intensity=0.43,
        orientation_change_score=0.18,
        sampling_continuity_score=0.97,
        sensor_saturation_score=0.02,
    )

    interaction = InteractionData(
        metadata=ModalityMetadata(
            modality="interaction",
            status=ModalityStatus.OBSERVED,
            source_timestamp=current_time,
            arrival_timestamp=current_time,
            sequence_number=4,
            source_device_id="PHONE_001",
        ),
        interaction_id="INTERACTION_000004",
        interaction_type=InteractionType.BUTTON,
        action="navigation_mode_requested",
        emergency_flag=False,
    )

    wearable = WearableData(
        metadata=ModalityMetadata(
            modality="wearable",
            status=ModalityStatus.OBSERVED,
            source_timestamp=current_time,
            arrival_timestamp=current_time,
            source_device_id="PHONE_001",
        ),
        device_id="EARPHONE_001",
        device_name="realme Buds T200",
        device_type="wireless_earphones",
        connected=True,
        connection_type="bluetooth",
        capabilities=[
            "audio_output",
            "microphone_input",
        ],
        microphone_available=True,
        audio_output_available=True,
        haptic_output_available=False,
    )

    environment = EnvironmentalData(
        metadata=ModalityMetadata(
            modality="environment",
            status=ModalityStatus.NOT_REQUESTED,
        )
    )

    modality_confidences = {
        "vision": ModalityConfidence(
            modality="vision",
            signal_quality=0.82,
            temporal_alignment=0.95,
            sensor_health=0.98,
            freshness=0.97,
            cross_modal_agreement=0.91,
            conflict_score=0.09,
            final_confidence=0.86,
            reliability_level=ReliabilityLevel.HIGH,
        ),
        "audio": ModalityConfidence(
            modality="audio",
            signal_quality=0.64,
            temporal_alignment=0.92,
            sensor_health=0.96,
            freshness=0.95,
            cross_modal_agreement=0.84,
            conflict_score=0.16,
            final_confidence=0.67,
            reliability_level=ReliabilityLevel.MODERATE,
            limitations=["moderate_background_noise"],
        ),
        "spatial": ModalityConfidence(
            modality="spatial",
            signal_quality=0.74,
            temporal_alignment=0.88,
            sensor_health=0.97,
            freshness=0.90,
            cross_modal_agreement=0.89,
            conflict_score=0.11,
            final_confidence=0.75,
            reliability_level=ReliabilityLevel.MODERATE,
        ),
        "motion": ModalityConfidence(
            modality="motion",
            signal_quality=0.93,
            temporal_alignment=0.98,
            sensor_health=0.98,
            freshness=0.99,
            cross_modal_agreement=0.94,
            conflict_score=0.06,
            final_confidence=0.91,
            reliability_level=ReliabilityLevel.HIGH,
        ),
        "interaction": ModalityConfidence(
            modality="interaction",
            signal_quality=0.99,
            temporal_alignment=0.99,
            sensor_health=0.99,
            freshness=0.99,
            cross_modal_agreement=1.0,
            conflict_score=0.0,
            final_confidence=0.99,
            reliability_level=ReliabilityLevel.HIGH,
        ),
    }

    packet = MultimodalSensorPacket(
        source_device=SourceDevice(
            device_id="PHONE_001",
            device_type="android_smartphone",
            device_name="NOONGIL Phone",
            operating_system="Android",
            network_type=NetworkType.WIFI,
            network_strength=0.88,
            network_latency_ms=21.0,
            battery_level=0.72,
            is_charging=False,
            application_version="1.0.0",
            available_sensors=[
                "camera",
                "microphone",
                "gps",
                "accelerometer",
                "gyroscope",
                "magnetometer",
                "touchscreen",
            ],
        ),
        acquisition_mode=AcquisitionMode.NAVIGATION,
        modalities=Modalities(
            vision=vision,
            audio=audio,
            spatial=spatial,
            motion=motion,
            interaction=interaction,
            wearable=wearable,
            environment=environment,
        ),
        synchronization=SynchronizationData(
            status=SynchronizationStatus.SYNCHRONIZED,
            reference_timestamp=current_time,
            window_ms=250.0,
            aligned_modalities=[
                "vision",
                "audio",
                "spatial",
                "motion",
                "interaction",
                "wearable",
            ],
            missing_modalities=[],
            stale_modalities=[],
            modality_alignments={
                modality: ModalityAlignment(
                    modality=modality,
                    source_timestamp=current_time,
                    offset_ms=0.0,
                    tolerance_ms=250.0,
                    alignment_score=0.95,
                    within_window=True,
                )
                for modality in [
                    "vision",
                    "audio",
                    "spatial",
                    "motion",
                    "interaction",
                    "wearable",
                ]
            },
            overall_alignment_score=0.95,
        ),
        sensor_confidence=SensorConfidenceData(
            modality_confidences=modality_confidences,
            normalized_weights={
                "vision": 0.21,
                "audio": 0.16,
                "spatial": 0.18,
                "motion": 0.22,
                "interaction": 0.23,
            },
            overall_confidence=0.84,
        ),
        recovery=RecoveryData(
            recovery_used=False,
            recovered_modalities=[],
            unavailable_modalities=[],
        ),
        routing=RoutingData(
            source_layer="layer1",
            destination_layer="layer2",
            semantic_processing_performed=False,
            packet_priority="high",
            dispatch_status="pending",
        ),
        packet_sequence_number=1,
        scenario_name="navigation_demo",
    )

    packet.validate()
    return packet


# ============================================================
# COMMAND-LINE SELF-TEST
# ============================================================

def run_schema_self_test() -> bool:
    """
    Run a standalone test of the Layer 1 sensor-packet schema.

    Returns
    -------
    bool
        True when all checks pass.
    """

    print("\n" + "=" * 68)
    print("NOONGIL-X | LAYER 1 SENSOR PACKET SCHEMA TEST")
    print("=" * 68)

    try:
        print("[1/5] Creating empty packet...")
        empty_packet = create_empty_sensor_packet(
            device_id="PHONE_TEST_001",
            acquisition_mode=AcquisitionMode.IDLE,
            device_name="Test Android Phone",
            scenario_name="schema_empty_test",
        )

        print("[SUCCESS] Empty packet created and validated.")

        print("[2/5] Creating realistic demonstration packet...")
        demo_packet = create_demo_sensor_packet()

        print("[SUCCESS] Demonstration packet created.")

        print("[3/5] Validating complete packet...")
        demo_packet.validate()

        print("[SUCCESS] Complete packet is valid.")

        print("[4/5] Testing JSON serialization...")
        serialized = demo_packet.to_json(indent=2)

        parsed = json.loads(serialized)

        if parsed["packet_id"] != demo_packet.packet_id:
            raise AssertionError(
                "Serialized packet ID does not match."
            )

        if (
            parsed["routing"]["semantic_processing_performed"]
            is not False
        ):
            raise AssertionError(
                "Layer 1 semantic-processing boundary was violated."
            )

        print("[SUCCESS] JSON serialization is valid.")

        print("[5/5] Saving demonstration packet...")
        test_output_path = (
            Path(__file__).resolve().parent
            / "_sensor_packet_schema_test.json"
        )

        saved_path = demo_packet.save_json(test_output_path)

        if not saved_path.exists():
            raise AssertionError(
                "Test JSON file was not created."
            )

        print(f"[SUCCESS] Test packet saved: {saved_path}")

        print("\nPacket summary:")
        print(
            json.dumps(
                demo_packet.summary(),
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n" + "=" * 68)
        print("[PASSED] SENSOR PACKET SCHEMA IS WORKING CORRECTLY")
        print("=" * 68)

        return True

    except Exception as error:
        print("\n" + "=" * 68)
        print("[FAILED] SENSOR PACKET SCHEMA TEST")
        print("=" * 68)
        print(f"[ERROR] {type(error).__name__}: {error}")

        return False


if __name__ == "__main__":
    success = run_schema_self_test()

    if not success:
        raise SystemExit(1)
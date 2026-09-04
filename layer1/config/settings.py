"""
============================================================
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : Runtime Settings
File    : layer1/config/settings.py
============================================================

Purpose
-------
This module defines all configurable runtime settings used by
Layer 1 of NOONGIL-X.

It controls:

1. Execution mode
2. Smartphone-to-laptop communication
3. Vision acquisition
4. Audio acquisition
5. Spatial acquisition
6. Motion acquisition
7. User interaction
8. Wearable integration
9. Environmental-context acquisition
10. NAMARA adaptive acquisition
11. Multimodal synchronization
12. Sensor-confidence estimation
13. Missing-modality recovery
14. Packet generation
15. Logging and debugging

Architectural Boundary
----------------------
This module contains configuration only.

It does not:
- capture sensor data;
- preprocess sensor data;
- perform semantic interpretation;
- run perception models;
- use an LLM;
- perform reasoning.

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import json
import math
import os

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from layer1.schemas.sensor_packet import AcquisitionMode


# ============================================================
# ENUMERATIONS
# ============================================================

class ExecutionMode(str, Enum):
    """
    Defines the source of Layer 1 sensor inputs.
    """

    SIMULATION = "simulation"
    LIVE = "live"
    REPLAY = "replay"
    TEST = "test"


class TransportProtocol(str, Enum):
    """
    Supported phone-to-laptop transport protocols.
    """

    WEBSOCKET = "websocket"
    HTTP = "http"
    UDP = "udp"
    TCP = "tcp"
    LOCAL = "local"


class LogLevel(str, Enum):
    """
    Supported logging levels.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class FrameEncoding(str, Enum):
    """
    Supported camera-frame encodings.
    """

    JPEG = "jpeg"
    PNG = "png"
    RAW = "raw"


class AudioEncoding(str, Enum):
    """
    Supported audio encodings.
    """

    PCM_S16LE = "pcm_s16le"
    WAV = "wav"
    OPUS = "opus"
    AAC = "aac"


class RecoveryPolicy(str, Enum):
    """
    Defines how Layer 1 handles missing modality data.
    """

    STRICT = "strict"
    INTERPOLATE = "interpolate"
    FALLBACK = "fallback"
    BEST_EFFORT = "best_effort"


# ============================================================
# GENERAL VALIDATION HELPERS
# ============================================================

def validate_probability(
    value: float,
    field_name: str,
) -> None:
    """
    Validate a normalized value in the range [0, 1].
    """

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{field_name} must be numeric."
        )

    if not math.isfinite(float(value)):
        raise ValueError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0.0 and 1.0."
        )


def validate_positive(
    value: float,
    field_name: str,
) -> None:
    """
    Validate that a numeric value is greater than zero.
    """

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{field_name} must be numeric."
        )

    if not math.isfinite(float(value)):
        raise ValueError(
            f"{field_name} must be finite."
        )

    if float(value) <= 0:
        raise ValueError(
            f"{field_name} must be greater than zero."
        )


def validate_non_negative(
    value: float,
    field_name: str,
) -> None:
    """
    Validate that a numeric value is zero or greater.
    """

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
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


def validate_port(
    value: int,
    field_name: str,
) -> None:
    """
    Validate a network port number.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be an integer."
        )

    if not 1 <= value <= 65535:
        raise ValueError(
            f"{field_name} must be between 1 and 65535."
        )


def validate_non_empty_string(
    value: str,
    field_name: str,
) -> None:
    """
    Validate a required string.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string."
        )


def enum_value(value: Any) -> Any:
    """
    Recursively convert enum values into JSON-safe values.
    """

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): enum_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            enum_value(item)
            for item in value
        ]

    return value


# ============================================================
# EXECUTION SETTINGS
# ============================================================

@dataclass
class RuntimeSettings:
    """
    General Layer 1 execution configuration.
    """

    execution_mode: ExecutionMode = ExecutionMode.SIMULATION
    default_acquisition_mode: AcquisitionMode = (
        AcquisitionMode.AWARENESS
    )

    continuous_operation: bool = False
    cycle_interval_seconds: float = 1.0
    maximum_cycles: Optional[int] = 1

    enable_validation: bool = True
    fail_fast: bool = False
    save_intermediate_outputs: bool = True

    use_utc_timestamps: bool = True
    schema_version: str = "1.0"

    def validate(self) -> None:
        validate_positive(
            self.cycle_interval_seconds,
            "runtime.cycle_interval_seconds",
        )

        if self.maximum_cycles is not None:
            if (
                isinstance(self.maximum_cycles, bool)
                or not isinstance(self.maximum_cycles, int)
                or self.maximum_cycles <= 0
            ):
                raise ValueError(
                    "runtime.maximum_cycles must be a positive "
                    "integer or None."
                )

        validate_non_empty_string(
            self.schema_version,
            "runtime.schema_version",
        )


# ============================================================
# NETWORK SETTINGS
# ============================================================

@dataclass
class NetworkSettings:
    """
    Smartphone-to-laptop connection configuration.
    """

    enabled: bool = False
    protocol: TransportProtocol = TransportProtocol.WEBSOCKET

    host: str = "0.0.0.0"
    port: int = 8765
    endpoint: str = "/noongil/layer1"

    connection_timeout_seconds: float = 10.0
    receive_timeout_seconds: float = 5.0
    heartbeat_interval_seconds: float = 10.0

    reconnect_enabled: bool = True
    maximum_reconnect_attempts: int = 5
    reconnect_delay_seconds: float = 2.0

    maximum_packet_size_bytes: int = 10 * 1024 * 1024

    authentication_enabled: bool = False
    authentication_token_env: str = "NOONGIL_PHONE_TOKEN"

    compression_enabled: bool = True
    verify_device_id: bool = True

    def validate(self) -> None:
        validate_non_empty_string(
            self.host,
            "network.host",
        )

        validate_port(
            self.port,
            "network.port",
        )

        validate_non_empty_string(
            self.endpoint,
            "network.endpoint",
        )

        validate_positive(
            self.connection_timeout_seconds,
            "network.connection_timeout_seconds",
        )

        validate_positive(
            self.receive_timeout_seconds,
            "network.receive_timeout_seconds",
        )

        validate_positive(
            self.heartbeat_interval_seconds,
            "network.heartbeat_interval_seconds",
        )

        if (
            isinstance(self.maximum_reconnect_attempts, bool)
            or not isinstance(
                self.maximum_reconnect_attempts,
                int,
            )
            or self.maximum_reconnect_attempts < 0
        ):
            raise ValueError(
                "network.maximum_reconnect_attempts must be a "
                "non-negative integer."
            )

        validate_non_negative(
            self.reconnect_delay_seconds,
            "network.reconnect_delay_seconds",
        )

        if (
            isinstance(self.maximum_packet_size_bytes, bool)
            or not isinstance(
                self.maximum_packet_size_bytes,
                int,
            )
            or self.maximum_packet_size_bytes <= 0
        ):
            raise ValueError(
                "network.maximum_packet_size_bytes must be a "
                "positive integer."
            )

        if self.authentication_enabled:
            validate_non_empty_string(
                self.authentication_token_env,
                "network.authentication_token_env",
            )

    def get_authentication_token(self) -> Optional[str]:
        """
        Read the phone authentication token from the environment.
        """

        if not self.authentication_enabled:
            return None

        value = os.getenv(self.authentication_token_env)

        if not value:
            raise RuntimeError(
                "Phone authentication is enabled, but environment "
                f"variable {self.authentication_token_env!r} "
                "is not set."
            )

        return value


# ============================================================
# VISION SETTINGS
# ============================================================

@dataclass
class VisionSettings:
    """
    Smartphone-camera acquisition and preprocessing settings.
    """

    enabled: bool = True

    target_width: int = 640
    target_height: int = 480
    target_channels: int = 3

    encoding: FrameEncoding = FrameEncoding.JPEG
    jpeg_quality: int = 80

    minimum_fps: float = 2.0
    default_fps: float = 10.0
    maximum_fps: float = 30.0

    normalize_pixels: bool = True
    resize_frames: bool = True
    denoise_frames: bool = False
    preserve_aspect_ratio: bool = True

    calculate_brightness: bool = True
    calculate_sharpness: bool = True
    calculate_contrast: bool = True
    calculate_frame_integrity: bool = True

    low_brightness_threshold: float = 0.20
    minimum_sharpness_threshold: float = 0.25
    minimum_frame_integrity: float = 0.70

    maximum_frame_age_ms: float = 500.0
    frame_buffer_size: int = 30

    save_raw_frames: bool = True
    save_preprocessed_frames: bool = False

    def validate(self) -> None:
        for name, value in (
            ("target_width", self.target_width),
            ("target_height", self.target_height),
            ("target_channels", self.target_channels),
            ("frame_buffer_size", self.frame_buffer_size),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"vision.{name} must be a positive integer."
                )

        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(
                "vision.jpeg_quality must be between 1 and 100."
            )

        for name, value in (
            ("minimum_fps", self.minimum_fps),
            ("default_fps", self.default_fps),
            ("maximum_fps", self.maximum_fps),
        ):
            validate_positive(
                value,
                f"vision.{name}",
            )

        if not (
            self.minimum_fps
            <= self.default_fps
            <= self.maximum_fps
        ):
            raise ValueError(
                "Vision FPS must satisfy: "
                "minimum_fps <= default_fps <= maximum_fps."
            )

        for name, value in (
            (
                "low_brightness_threshold",
                self.low_brightness_threshold,
            ),
            (
                "minimum_sharpness_threshold",
                self.minimum_sharpness_threshold,
            ),
            (
                "minimum_frame_integrity",
                self.minimum_frame_integrity,
            ),
        ):
            validate_probability(
                value,
                f"vision.{name}",
            )

        validate_positive(
            self.maximum_frame_age_ms,
            "vision.maximum_frame_age_ms",
        )


# ============================================================
# AUDIO SETTINGS
# ============================================================

@dataclass
class AudioSettings:
    """
    Phone or earphone microphone acquisition settings.
    """

    enabled: bool = True

    sample_rate_hz: int = 16_000
    channels: int = 1
    sample_width_bits: int = 16

    encoding: AudioEncoding = AudioEncoding.PCM_S16LE
    chunk_duration_ms: float = 1000.0
    overlap_duration_ms: float = 100.0

    convert_to_mono: bool = True
    normalize_amplitude: bool = True
    basic_noise_reduction: bool = False

    calculate_amplitude: bool = True
    calculate_signal_to_noise: bool = True
    calculate_clipping_ratio: bool = True
    calculate_silence_ratio: bool = True
    calculate_packet_integrity: bool = True

    generate_mel_spectrogram: bool = False
    generate_mfcc: bool = False
    number_of_mfcc_coefficients: int = 13

    silence_threshold: float = 0.02
    maximum_clipping_ratio: float = 0.10
    minimum_signal_to_noise_score: float = 0.30

    maximum_audio_age_ms: float = 1500.0
    audio_buffer_chunks: int = 10

    save_raw_audio: bool = True
    save_preprocessed_audio: bool = False

    def validate(self) -> None:
        for name, value in (
            ("sample_rate_hz", self.sample_rate_hz),
            ("channels", self.channels),
            (
                "number_of_mfcc_coefficients",
                self.number_of_mfcc_coefficients,
            ),
            (
                "audio_buffer_chunks",
                self.audio_buffer_chunks,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"audio.{name} must be a positive integer."
                )

        if self.sample_width_bits not in {8, 16, 24, 32}:
            raise ValueError(
                "audio.sample_width_bits must be one of "
                "{8, 16, 24, 32}."
            )

        validate_positive(
            self.chunk_duration_ms,
            "audio.chunk_duration_ms",
        )

        validate_non_negative(
            self.overlap_duration_ms,
            "audio.overlap_duration_ms",
        )

        if self.overlap_duration_ms >= self.chunk_duration_ms:
            raise ValueError(
                "audio.overlap_duration_ms must be smaller than "
                "audio.chunk_duration_ms."
            )

        for name, value in (
            ("silence_threshold", self.silence_threshold),
            (
                "maximum_clipping_ratio",
                self.maximum_clipping_ratio,
            ),
            (
                "minimum_signal_to_noise_score",
                self.minimum_signal_to_noise_score,
            ),
        ):
            validate_probability(
                value,
                f"audio.{name}",
            )

        validate_positive(
            self.maximum_audio_age_ms,
            "audio.maximum_audio_age_ms",
        )


# ============================================================
# SPATIAL SETTINGS
# ============================================================

@dataclass
class SpatialSettings:
    """
    GPS and compass acquisition settings.
    """

    enabled: bool = True

    minimum_sampling_rate_hz: float = 0.2
    default_sampling_rate_hz: float = 1.0
    maximum_sampling_rate_hz: float = 5.0

    enable_gps: bool = True
    enable_compass: bool = True

    maximum_horizontal_accuracy_meters: float = 50.0
    preferred_horizontal_accuracy_meters: float = 10.0

    maximum_location_age_ms: float = 5000.0
    maximum_heading_age_ms: float = 2000.0

    minimum_speed_meters_per_second: float = 0.0
    maximum_reasonable_speed_meters_per_second: float = 60.0

    use_last_known_location: bool = True
    maximum_last_known_location_age_seconds: float = 30.0

    def validate(self) -> None:
        for name, value in (
            (
                "minimum_sampling_rate_hz",
                self.minimum_sampling_rate_hz,
            ),
            (
                "default_sampling_rate_hz",
                self.default_sampling_rate_hz,
            ),
            (
                "maximum_sampling_rate_hz",
                self.maximum_sampling_rate_hz,
            ),
        ):
            validate_positive(
                value,
                f"spatial.{name}",
            )

        if not (
            self.minimum_sampling_rate_hz
            <= self.default_sampling_rate_hz
            <= self.maximum_sampling_rate_hz
        ):
            raise ValueError(
                "Spatial sampling rates must satisfy: minimum "
                "<= default <= maximum."
            )

        for name, value in (
            (
                "maximum_horizontal_accuracy_meters",
                self.maximum_horizontal_accuracy_meters,
            ),
            (
                "preferred_horizontal_accuracy_meters",
                self.preferred_horizontal_accuracy_meters,
            ),
            (
                "maximum_location_age_ms",
                self.maximum_location_age_ms,
            ),
            (
                "maximum_heading_age_ms",
                self.maximum_heading_age_ms,
            ),
            (
                "maximum_reasonable_speed_meters_per_second",
                self.maximum_reasonable_speed_meters_per_second,
            ),
            (
                "maximum_last_known_location_age_seconds",
                self.maximum_last_known_location_age_seconds,
            ),
        ):
            validate_positive(
                value,
                f"spatial.{name}",
            )

        validate_non_negative(
            self.minimum_speed_meters_per_second,
            "spatial.minimum_speed_meters_per_second",
        )

        if (
            self.preferred_horizontal_accuracy_meters
            > self.maximum_horizontal_accuracy_meters
        ):
            raise ValueError(
                "Preferred GPS accuracy cannot be worse than "
                "maximum accepted GPS accuracy."
            )


# ============================================================
# MOTION SETTINGS
# ============================================================

@dataclass
class MotionSettings:
    """
    Accelerometer, gyroscope, and magnetometer settings.
    """

    enabled: bool = True

    enable_accelerometer: bool = True
    enable_gyroscope: bool = True
    enable_magnetometer: bool = True

    minimum_sampling_rate_hz: float = 10.0
    default_sampling_rate_hz: float = 25.0
    maximum_sampling_rate_hz: float = 100.0

    motion_buffer_seconds: float = 3.0
    maximum_motion_age_ms: float = 250.0

    calculate_motion_intensity: bool = True
    calculate_orientation_change: bool = True
    calculate_sampling_continuity: bool = True
    calculate_sensor_saturation: bool = True

    low_motion_threshold: float = 0.15
    high_motion_threshold: float = 0.70

    accelerometer_maximum_absolute_value: float = 80.0
    gyroscope_maximum_absolute_value: float = 35.0
    magnetometer_maximum_absolute_value: float = 5000.0

    def validate(self) -> None:
        for name, value in (
            (
                "minimum_sampling_rate_hz",
                self.minimum_sampling_rate_hz,
            ),
            (
                "default_sampling_rate_hz",
                self.default_sampling_rate_hz,
            ),
            (
                "maximum_sampling_rate_hz",
                self.maximum_sampling_rate_hz,
            ),
            (
                "motion_buffer_seconds",
                self.motion_buffer_seconds,
            ),
            (
                "maximum_motion_age_ms",
                self.maximum_motion_age_ms,
            ),
            (
                "accelerometer_maximum_absolute_value",
                self.accelerometer_maximum_absolute_value,
            ),
            (
                "gyroscope_maximum_absolute_value",
                self.gyroscope_maximum_absolute_value,
            ),
            (
                "magnetometer_maximum_absolute_value",
                self.magnetometer_maximum_absolute_value,
            ),
        ):
            validate_positive(
                value,
                f"motion.{name}",
            )

        if not (
            self.minimum_sampling_rate_hz
            <= self.default_sampling_rate_hz
            <= self.maximum_sampling_rate_hz
        ):
            raise ValueError(
                "Motion sampling rates must satisfy: minimum "
                "<= default <= maximum."
            )

        validate_probability(
            self.low_motion_threshold,
            "motion.low_motion_threshold",
        )

        validate_probability(
            self.high_motion_threshold,
            "motion.high_motion_threshold",
        )

        if self.low_motion_threshold >= self.high_motion_threshold:
            raise ValueError(
                "motion.low_motion_threshold must be smaller than "
                "motion.high_motion_threshold."
            )


# ============================================================
# INTERACTION SETTINGS
# ============================================================

@dataclass
class InteractionSettings:
    """
    Explicit smartphone interaction settings.
    """

    enabled: bool = True

    enable_touch_input: bool = True
    enable_button_input: bool = True
    enable_voice_trigger: bool = True
    enable_emergency_trigger: bool = True

    emergency_button_press_count: int = 3
    emergency_press_window_seconds: float = 3.0

    allow_volume_button_shortcuts: bool = True
    interaction_event_timeout_ms: float = 1000.0

    supported_actions: List[str] = field(
        default_factory=lambda: [
            "capture_request",
            "reading_mode_requested",
            "navigation_mode_requested",
            "awareness_mode_requested",
            "emergency_mode_requested",
            "pause_requested",
            "resume_requested",
            "stop_requested",
        ]
    )

    def validate(self) -> None:
        if (
            isinstance(self.emergency_button_press_count, bool)
            or not isinstance(
                self.emergency_button_press_count,
                int,
            )
            or self.emergency_button_press_count <= 0
        ):
            raise ValueError(
                "interaction.emergency_button_press_count must "
                "be a positive integer."
            )

        validate_positive(
            self.emergency_press_window_seconds,
            "interaction.emergency_press_window_seconds",
        )

        validate_positive(
            self.interaction_event_timeout_ms,
            "interaction.interaction_event_timeout_ms",
        )

        if not self.supported_actions:
            raise ValueError(
                "interaction.supported_actions cannot be empty."
            )

        for action in self.supported_actions:
            validate_non_empty_string(
                action,
                "interaction.supported_actions item",
            )


# ============================================================
# DEVICE / WEARABLE SETTINGS
# ============================================================

@dataclass
class DeviceSettings:
    """
    Smartphone and connected-device configuration.
    """

    expected_phone_device_type: str = "android_smartphone"
    allow_unknown_phone: bool = True

    enable_earphone_detection: bool = True
    preferred_earphone_name: str = "realme Buds T200"

    allow_phone_microphone_fallback: bool = True
    allow_earphone_microphone: bool = True

    record_battery_state: bool = True
    record_network_state: bool = True
    record_available_sensors: bool = True

    minimum_supported_battery_level: float = 0.05

    def validate(self) -> None:
        validate_non_empty_string(
            self.expected_phone_device_type,
            "device.expected_phone_device_type",
        )

        validate_non_empty_string(
            self.preferred_earphone_name,
            "device.preferred_earphone_name",
        )

        validate_probability(
            self.minimum_supported_battery_level,
            "device.minimum_supported_battery_level",
        )


# ============================================================
# ENVIRONMENT SETTINGS
# ============================================================

@dataclass
class EnvironmentSettings:
    """
    External environmental-context acquisition settings.
    """

    enabled: bool = False

    enable_weather: bool = False
    enable_map_context: bool = False
    enable_traffic_context: bool = False
    enable_transport_context: bool = False

    request_timeout_seconds: float = 5.0
    cache_enabled: bool = True
    maximum_cache_age_seconds: float = 300.0

    allow_network_failure: bool = True

    weather_api_key_env: str = "NOONGIL_WEATHER_API_KEY"
    maps_api_key_env: str = "NOONGIL_MAPS_API_KEY"

    def validate(self) -> None:
        validate_positive(
            self.request_timeout_seconds,
            "environment.request_timeout_seconds",
        )

        validate_non_negative(
            self.maximum_cache_age_seconds,
            "environment.maximum_cache_age_seconds",
        )

        validate_non_empty_string(
            self.weather_api_key_env,
            "environment.weather_api_key_env",
        )

        validate_non_empty_string(
            self.maps_api_key_env,
            "environment.maps_api_key_env",
        )


# ============================================================
# NAMARA SETTINGS
# ============================================================

@dataclass
class NAMARAWeights:
    """
    Weights used by NAMARA for sensor-activation decisions.

    Activation score:
        A_i = reliability_weight * reliability
            + context_weight * context_relevance
            + urgency_weight * urgency
            - energy_weight * energy_cost
    """

    reliability_weight: float = 0.35
    context_weight: float = 0.35
    urgency_weight: float = 0.20
    energy_weight: float = 0.10

    def validate(self) -> None:
        values = {
            "reliability_weight": self.reliability_weight,
            "context_weight": self.context_weight,
            "urgency_weight": self.urgency_weight,
            "energy_weight": self.energy_weight,
        }

        for name, value in values.items():
            validate_probability(
                value,
                f"namara.weights.{name}",
            )

        total = sum(values.values())

        if not math.isclose(
            total,
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "NAMARA activation weights must sum to 1.0. "
                f"Current total: {total}"
            )


@dataclass
class NAMARASettings:
    """
    NOONGIL Adaptive Multimodal Acquisition and Reliability
    Algorithm configuration.
    """

    enabled: bool = True

    activation_threshold: float = 0.45
    emergency_activation_threshold: float = 0.20

    weights: NAMARAWeights = field(
        default_factory=NAMARAWeights
    )

    enable_context_adaptive_activation: bool = True
    enable_dynamic_sampling: bool = True
    enable_energy_awareness: bool = True
    enable_network_awareness: bool = True

    battery_low_threshold: float = 0.20
    battery_critical_threshold: float = 0.10

    network_degraded_strength_threshold: float = 0.35
    network_high_latency_ms: float = 250.0

    motion_sampling_multiplier: float = 1.50
    emergency_sampling_multiplier: float = 2.00
    low_power_sampling_multiplier: float = 0.50
    degraded_network_sampling_multiplier: float = 0.70

    sampling_adjustment_interval_seconds: float = 2.0
    minimum_mode_duration_seconds: float = 2.0

    mode_sensor_priorities: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            AcquisitionMode.IDLE.value: {
                "vision": 0.20,
                "audio": 0.30,
                "spatial": 0.10,
                "motion": 0.30,
                "interaction": 1.00,
                "wearable": 0.40,
                "environment": 0.00,
            },
            AcquisitionMode.AWARENESS.value: {
                "vision": 0.85,
                "audio": 0.75,
                "spatial": 0.50,
                "motion": 0.65,
                "interaction": 1.00,
                "wearable": 0.60,
                "environment": 0.20,
            },
            AcquisitionMode.READING.value: {
                "vision": 1.00,
                "audio": 0.70,
                "spatial": 0.10,
                "motion": 0.30,
                "interaction": 1.00,
                "wearable": 0.60,
                "environment": 0.00,
            },
            AcquisitionMode.NAVIGATION.value: {
                "vision": 1.00,
                "audio": 0.85,
                "spatial": 1.00,
                "motion": 1.00,
                "interaction": 1.00,
                "wearable": 0.80,
                "environment": 0.50,
            },
            AcquisitionMode.EMERGENCY.value: {
                "vision": 1.00,
                "audio": 1.00,
                "spatial": 1.00,
                "motion": 1.00,
                "interaction": 1.00,
                "wearable": 1.00,
                "environment": 0.50,
            },
            AcquisitionMode.LOW_POWER.value: {
                "vision": 0.35,
                "audio": 0.40,
                "spatial": 0.30,
                "motion": 0.40,
                "interaction": 1.00,
                "wearable": 0.50,
                "environment": 0.00,
            },
            AcquisitionMode.DEGRADED_NETWORK.value: {
                "vision": 0.50,
                "audio": 0.55,
                "spatial": 0.80,
                "motion": 0.85,
                "interaction": 1.00,
                "wearable": 0.60,
                "environment": 0.10,
            },
        }
    )

    def validate(self) -> None:
        validate_probability(
            self.activation_threshold,
            "namara.activation_threshold",
        )

        validate_probability(
            self.emergency_activation_threshold,
            "namara.emergency_activation_threshold",
        )

        if (
            self.emergency_activation_threshold
            > self.activation_threshold
        ):
            raise ValueError(
                "Emergency activation threshold should not be "
                "greater than the normal activation threshold."
            )

        self.weights.validate()

        for name, value in (
            (
                "battery_low_threshold",
                self.battery_low_threshold,
            ),
            (
                "battery_critical_threshold",
                self.battery_critical_threshold,
            ),
            (
                "network_degraded_strength_threshold",
                self.network_degraded_strength_threshold,
            ),
        ):
            validate_probability(
                value,
                f"namara.{name}",
            )

        if (
            self.battery_critical_threshold
            >= self.battery_low_threshold
        ):
            raise ValueError(
                "namara.battery_critical_threshold must be "
                "lower than namara.battery_low_threshold."
            )

        for name, value in (
            (
                "network_high_latency_ms",
                self.network_high_latency_ms,
            ),
            (
                "motion_sampling_multiplier",
                self.motion_sampling_multiplier,
            ),
            (
                "emergency_sampling_multiplier",
                self.emergency_sampling_multiplier,
            ),
            (
                "low_power_sampling_multiplier",
                self.low_power_sampling_multiplier,
            ),
            (
                "degraded_network_sampling_multiplier",
                self.degraded_network_sampling_multiplier,
            ),
            (
                "sampling_adjustment_interval_seconds",
                self.sampling_adjustment_interval_seconds,
            ),
            (
                "minimum_mode_duration_seconds",
                self.minimum_mode_duration_seconds,
            ),
        ):
            validate_positive(
                value,
                f"namara.{name}",
            )

        expected_modes = {
            mode.value
            for mode in AcquisitionMode
        }

        configured_modes = set(
            self.mode_sensor_priorities.keys()
        )

        missing_modes = expected_modes - configured_modes

        if missing_modes:
            raise ValueError(
                "NAMARA sensor priorities are missing modes: "
                f"{sorted(missing_modes)}"
            )

        supported_modalities = {
            "vision",
            "audio",
            "spatial",
            "motion",
            "interaction",
            "wearable",
            "environment",
        }

        for mode, priorities in (
            self.mode_sensor_priorities.items()
        ):
            unknown_modalities = (
                set(priorities.keys())
                - supported_modalities
            )

            if unknown_modalities:
                raise ValueError(
                    f"NAMARA mode {mode!r} contains unsupported "
                    f"modalities: {sorted(unknown_modalities)}"
                )

            for modality, priority in priorities.items():
                validate_probability(
                    priority,
                    (
                        "namara.mode_sensor_priorities"
                        f"[{mode!r}][{modality!r}]"
                    ),
                )


# ============================================================
# SYNCHRONIZATION SETTINGS
# ============================================================

@dataclass
class SynchronizationSettings:
    """
    Delay-aware multimodal synchronization settings.
    """

    enabled: bool = True

    default_window_ms: float = 250.0
    minimum_required_modalities: int = 1

    reference_modality_priority: List[str] = field(
        default_factory=lambda: [
            "interaction",
            "vision",
            "audio",
            "motion",
            "spatial",
            "wearable",
            "environment",
        ]
    )

    modality_tolerance_ms: Dict[str, float] = field(
        default_factory=lambda: {
            "vision": 100.0,
            "audio": 150.0,
            "spatial": 1000.0,
            "motion": 50.0,
            "interaction": 100.0,
            "wearable": 500.0,
            "environment": 300_000.0,
        }
    )

    allow_partial_synchronization: bool = True
    minimum_alignment_score: float = 0.50

    drop_stale_data: bool = True
    preserve_unaligned_data_in_metadata: bool = True

    maximum_buffer_age_ms: float = 5000.0
    maximum_buffer_items_per_modality: int = 1000

    def validate(self) -> None:
        validate_positive(
            self.default_window_ms,
            "synchronization.default_window_ms",
        )

        if (
            isinstance(self.minimum_required_modalities, bool)
            or not isinstance(
                self.minimum_required_modalities,
                int,
            )
            or self.minimum_required_modalities <= 0
        ):
            raise ValueError(
                "synchronization.minimum_required_modalities "
                "must be a positive integer."
            )

        supported_modalities = {
            "vision",
            "audio",
            "spatial",
            "motion",
            "interaction",
            "wearable",
            "environment",
        }

        for modality in self.reference_modality_priority:
            if modality not in supported_modalities:
                raise ValueError(
                    "Synchronization reference priority contains "
                    f"unsupported modality {modality!r}."
                )

        if len(set(self.reference_modality_priority)) != len(
            self.reference_modality_priority
        ):
            raise ValueError(
                "Synchronization reference priority contains "
                "duplicate modalities."
            )

        for modality, tolerance in (
            self.modality_tolerance_ms.items()
        ):
            if modality not in supported_modalities:
                raise ValueError(
                    f"Unsupported synchronization modality: "
                    f"{modality!r}."
                )

            validate_positive(
                tolerance,
                (
                    "synchronization.modality_tolerance_ms"
                    f"[{modality!r}]"
                ),
            )

        validate_probability(
            self.minimum_alignment_score,
            "synchronization.minimum_alignment_score",
        )

        validate_positive(
            self.maximum_buffer_age_ms,
            "synchronization.maximum_buffer_age_ms",
        )

        if (
            isinstance(
                self.maximum_buffer_items_per_modality,
                bool,
            )
            or not isinstance(
                self.maximum_buffer_items_per_modality,
                int,
            )
            or self.maximum_buffer_items_per_modality <= 0
        ):
            raise ValueError(
                "synchronization.maximum_buffer_items_per_modality "
                "must be a positive integer."
            )


# ============================================================
# CONFIDENCE SETTINGS
# ============================================================

@dataclass
class ConfidenceWeights:
    """
    Weight distribution used to calculate modality confidence.
    """

    signal_quality: float = 0.30
    temporal_alignment: float = 0.20
    sensor_health: float = 0.20
    freshness: float = 0.15
    cross_modal_agreement: float = 0.15

    def validate(self) -> None:
        values = {
            "signal_quality": self.signal_quality,
            "temporal_alignment": self.temporal_alignment,
            "sensor_health": self.sensor_health,
            "freshness": self.freshness,
            "cross_modal_agreement": (
                self.cross_modal_agreement
            ),
        }

        for name, value in values.items():
            validate_probability(
                value,
                f"confidence.weights.{name}",
            )

        total = sum(values.values())

        if not math.isclose(
            total,
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "Confidence weights must sum to 1.0. "
                f"Current total: {total}"
            )


@dataclass
class ConfidenceSettings:
    """
    Sensor-confidence and conflict-estimation settings.
    """

    enabled: bool = True

    weights: ConfidenceWeights = field(
        default_factory=ConfidenceWeights
    )

    high_threshold: float = 0.80
    moderate_threshold: float = 0.55
    low_threshold: float = 0.30

    minimum_usable_confidence: float = 0.20
    conflict_penalty_strength: float = 1.0

    enable_cross_modal_conflict: bool = True
    enable_confidence_normalization: bool = True

    default_unknown_signal_quality: float = 0.50
    default_unknown_sensor_health: float = 0.75
    default_unknown_freshness: float = 0.50
    default_unknown_alignment: float = 0.50
    default_unknown_agreement: float = 0.50

    exclude_unavailable_modalities_from_weights: bool = True

    def validate(self) -> None:
        self.weights.validate()

        for name, value in (
            ("high_threshold", self.high_threshold),
            (
                "moderate_threshold",
                self.moderate_threshold,
            ),
            ("low_threshold", self.low_threshold),
            (
                "minimum_usable_confidence",
                self.minimum_usable_confidence,
            ),
            (
                "default_unknown_signal_quality",
                self.default_unknown_signal_quality,
            ),
            (
                "default_unknown_sensor_health",
                self.default_unknown_sensor_health,
            ),
            (
                "default_unknown_freshness",
                self.default_unknown_freshness,
            ),
            (
                "default_unknown_alignment",
                self.default_unknown_alignment,
            ),
            (
                "default_unknown_agreement",
                self.default_unknown_agreement,
            ),
        ):
            validate_probability(
                value,
                f"confidence.{name}",
            )

        if not (
            self.high_threshold
            > self.moderate_threshold
            > self.low_threshold
        ):
            raise ValueError(
                "Confidence thresholds must satisfy: "
                "high > moderate > low."
            )

        if (
            self.minimum_usable_confidence
            > self.low_threshold
        ):
            raise ValueError(
                "confidence.minimum_usable_confidence should not "
                "be greater than confidence.low_threshold."
            )

        validate_non_negative(
            self.conflict_penalty_strength,
            "confidence.conflict_penalty_strength",
        )


# ============================================================
# RECOVERY SETTINGS
# ============================================================

@dataclass
class RecoverySettings:
    """
    Missing-modality recovery configuration.
    """

    enabled: bool = True
    policy: RecoveryPolicy = RecoveryPolicy.BEST_EFFORT

    enable_short_gap_interpolation: bool = True
    enable_cross_sensor_fallback: bool = True
    allow_last_known_value: bool = True

    maximum_interpolation_gap_ms: float = 500.0
    maximum_last_known_value_age_ms: float = 3000.0

    minimum_recovery_confidence: float = 0.30

    fallback_map: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "vision": [],
            "audio": ["wearable"],
            "spatial": ["motion"],
            "motion": ["spatial"],
            "interaction": [],
            "wearable": [],
            "environment": [],
        }
    )

    preserve_original_status: bool = True
    mark_recovered_data_explicitly: bool = True

    def validate(self) -> None:
        validate_positive(
            self.maximum_interpolation_gap_ms,
            "recovery.maximum_interpolation_gap_ms",
        )

        validate_positive(
            self.maximum_last_known_value_age_ms,
            "recovery.maximum_last_known_value_age_ms",
        )

        validate_probability(
            self.minimum_recovery_confidence,
            "recovery.minimum_recovery_confidence",
        )

        supported_modalities = {
            "vision",
            "audio",
            "spatial",
            "motion",
            "interaction",
            "wearable",
            "environment",
        }

        for target, sources in self.fallback_map.items():
            if target not in supported_modalities:
                raise ValueError(
                    f"Unsupported recovery target: {target!r}."
                )

            for source in sources:
                if source not in supported_modalities:
                    raise ValueError(
                        f"Unsupported recovery source: {source!r}."
                    )

                if source == target:
                    raise ValueError(
                        f"Recovery source for {target!r} cannot "
                        "reference itself."
                    )


# ============================================================
# OUTPUT SETTINGS
# ============================================================

@dataclass
class OutputSettings:
    """
    Layer 1 output and Layer 2 dispatch settings.
    """

    save_final_packet: bool = True
    save_packet_history: bool = False

    final_packet_filename: str = (
        "multimodal_sensor_packet.json"
    )

    layer2_packet_filename: str = (
        "layer1_sensor_packet.json"
    )

    pretty_print_json: bool = True
    json_indent: int = 4

    atomic_file_write: bool = True
    validate_before_save: bool = True
    validate_before_dispatch: bool = True

    dispatch_to_layer2: bool = True
    overwrite_latest_packet: bool = True

    packet_history_limit: int = 100

    def validate(self) -> None:
        validate_non_empty_string(
            self.final_packet_filename,
            "output.final_packet_filename",
        )

        validate_non_empty_string(
            self.layer2_packet_filename,
            "output.layer2_packet_filename",
        )

        if (
            isinstance(self.json_indent, bool)
            or not isinstance(self.json_indent, int)
            or self.json_indent < 0
        ):
            raise ValueError(
                "output.json_indent must be a non-negative integer."
            )

        if (
            isinstance(self.packet_history_limit, bool)
            or not isinstance(
                self.packet_history_limit,
                int,
            )
            or self.packet_history_limit <= 0
        ):
            raise ValueError(
                "output.packet_history_limit must be a positive "
                "integer."
            )


# ============================================================
# LOGGING SETTINGS
# ============================================================

@dataclass
class LoggingSettings:
    """
    Layer 1 logging configuration.
    """

    enabled: bool = True
    level: LogLevel = LogLevel.INFO

    log_to_console: bool = True
    log_to_file: bool = True

    log_filename: str = "layer1.log"
    error_log_filename: str = "layer1_errors.log"

    include_timestamp: bool = True
    include_module_name: bool = True

    maximum_log_size_bytes: int = 5 * 1024 * 1024
    backup_count: int = 3

    def validate(self) -> None:
        validate_non_empty_string(
            self.log_filename,
            "logging.log_filename",
        )

        validate_non_empty_string(
            self.error_log_filename,
            "logging.error_log_filename",
        )

        if (
            isinstance(self.maximum_log_size_bytes, bool)
            or not isinstance(
                self.maximum_log_size_bytes,
                int,
            )
            or self.maximum_log_size_bytes <= 0
        ):
            raise ValueError(
                "logging.maximum_log_size_bytes must be a "
                "positive integer."
            )

        if (
            isinstance(self.backup_count, bool)
            or not isinstance(self.backup_count, int)
            or self.backup_count < 0
        ):
            raise ValueError(
                "logging.backup_count must be a non-negative integer."
            )


# ============================================================
# SIMULATION SETTINGS
# ============================================================

@dataclass
class SimulationSettings:
    """
    Simulated smartphone-input configuration.
    """

    enabled: bool = True

    scenario_name: str = "navigation_demo"
    random_seed: int = 42

    simulate_vision: bool = True
    simulate_audio: bool = True
    simulate_spatial: bool = True
    simulate_motion: bool = True
    simulate_interaction: bool = True
    simulate_wearable: bool = True
    simulate_environment: bool = False

    introduce_random_latency: bool = True
    minimum_latency_ms: float = 5.0
    maximum_latency_ms: float = 80.0

    introduce_missing_modalities: bool = False
    modality_missing_probability: float = 0.05

    introduce_degraded_quality: bool = False
    degraded_quality_probability: float = 0.10

    def validate(self) -> None:
        validate_non_empty_string(
            self.scenario_name,
            "simulation.scenario_name",
        )

        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
        ):
            raise ValueError(
                "simulation.random_seed must be an integer."
            )

        validate_non_negative(
            self.minimum_latency_ms,
            "simulation.minimum_latency_ms",
        )

        validate_non_negative(
            self.maximum_latency_ms,
            "simulation.maximum_latency_ms",
        )

        if (
            self.minimum_latency_ms
            > self.maximum_latency_ms
        ):
            raise ValueError(
                "simulation.minimum_latency_ms cannot exceed "
                "simulation.maximum_latency_ms."
            )

        validate_probability(
            self.modality_missing_probability,
            "simulation.modality_missing_probability",
        )

        validate_probability(
            self.degraded_quality_probability,
            "simulation.degraded_quality_probability",
        )


# ============================================================
# COMPLETE LAYER 1 SETTINGS
# ============================================================

@dataclass
class Layer1Settings:
    """
    Complete configuration container for NOONGIL Layer 1.
    """

    runtime: RuntimeSettings = field(
        default_factory=RuntimeSettings
    )

    network: NetworkSettings = field(
        default_factory=NetworkSettings
    )

    vision: VisionSettings = field(
        default_factory=VisionSettings
    )

    audio: AudioSettings = field(
        default_factory=AudioSettings
    )

    spatial: SpatialSettings = field(
        default_factory=SpatialSettings
    )

    motion: MotionSettings = field(
        default_factory=MotionSettings
    )

    interaction: InteractionSettings = field(
        default_factory=InteractionSettings
    )

    device: DeviceSettings = field(
        default_factory=DeviceSettings
    )

    environment: EnvironmentSettings = field(
        default_factory=EnvironmentSettings
    )

    namara: NAMARASettings = field(
        default_factory=NAMARASettings
    )

    synchronization: SynchronizationSettings = field(
        default_factory=SynchronizationSettings
    )

    confidence: ConfidenceSettings = field(
        default_factory=ConfidenceSettings
    )

    recovery: RecoverySettings = field(
        default_factory=RecoverySettings
    )

    output: OutputSettings = field(
        default_factory=OutputSettings
    )

    logging: LoggingSettings = field(
        default_factory=LoggingSettings
    )

    simulation: SimulationSettings = field(
        default_factory=SimulationSettings
    )

    def validate(self) -> None:
        """
        Validate every Layer 1 configuration group.
        """

        self.runtime.validate()
        self.network.validate()
        self.vision.validate()
        self.audio.validate()
        self.spatial.validate()
        self.motion.validate()
        self.interaction.validate()
        self.device.validate()
        self.environment.validate()
        self.namara.validate()
        self.synchronization.validate()
        self.confidence.validate()
        self.recovery.validate()
        self.output.validate()
        self.logging.validate()
        self.simulation.validate()

        self._validate_cross_section_consistency()

    def _validate_cross_section_consistency(self) -> None:
        """
        Validate settings that depend on multiple sections.
        """

        if (
            self.runtime.execution_mode
            == ExecutionMode.SIMULATION
            and not self.simulation.enabled
        ):
            raise ValueError(
                "Simulation execution mode requires "
                "simulation.enabled=True."
            )

        if (
            self.runtime.execution_mode
            == ExecutionMode.LIVE
            and not self.network.enabled
        ):
            raise ValueError(
                "Live execution mode requires network.enabled=True."
            )

        if (
            self.runtime.default_acquisition_mode
            == AcquisitionMode.EMERGENCY
            and not self.interaction.enable_emergency_trigger
        ):
            raise ValueError(
                "Emergency acquisition mode requires emergency "
                "interaction support."
            )

        if (
            self.audio.enabled
            and not (
                self.device.allow_phone_microphone_fallback
                or self.device.allow_earphone_microphone
            )
        ):
            raise ValueError(
                "Audio is enabled, but no microphone source is "
                "permitted."
            )

        enabled_modalities = self.enabled_modalities()

        if not enabled_modalities:
            raise ValueError(
                "At least one Layer 1 modality must be enabled."
            )

        if (
            self.synchronization.minimum_required_modalities
            > len(enabled_modalities)
        ):
            raise ValueError(
                "synchronization.minimum_required_modalities "
                "cannot exceed the number of enabled modalities."
            )

    def enabled_modalities(self) -> List[str]:
        """
        Return all currently enabled Layer 1 modalities.
        """

        modalities: List[str] = []

        if self.vision.enabled:
            modalities.append("vision")

        if self.audio.enabled:
            modalities.append("audio")

        if self.spatial.enabled:
            modalities.append("spatial")

        if self.motion.enabled:
            modalities.append("motion")

        if self.interaction.enabled:
            modalities.append("interaction")

        if self.device.enable_earphone_detection:
            modalities.append("wearable")

        if self.environment.enabled:
            modalities.append("environment")

        return modalities

    def to_dict(
        self,
        *,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert settings into a JSON-compatible dictionary.
        """

        if validate:
            self.validate()

        return enum_value(asdict(self))

    def to_json(
        self,
        *,
        indent: int = 4,
        validate: bool = True,
    ) -> str:
        """
        Serialize settings to formatted JSON.
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
        validate: bool = True,
    ) -> Path:
        """
        Save settings to a JSON configuration file.
        """

        output_path = Path(file_path).expanduser().resolve()
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = self.to_json(validate=validate)

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
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
                f"Unable to save Layer 1 settings to "
                f"{output_path}: {error}"
            ) from error

        return output_path

    def summary(self) -> Dict[str, Any]:
        """
        Return a compact runtime summary.
        """

        return {
            "execution_mode": self.runtime.execution_mode.value,
            "default_acquisition_mode": (
                self.runtime.default_acquisition_mode.value
            ),
            "continuous_operation": (
                self.runtime.continuous_operation
            ),
            "network_enabled": self.network.enabled,
            "namara_enabled": self.namara.enabled,
            "synchronization_enabled": (
                self.synchronization.enabled
            ),
            "confidence_enabled": self.confidence.enabled,
            "recovery_enabled": self.recovery.enabled,
            "enabled_modalities": self.enabled_modalities(),
            "simulation_enabled": self.simulation.enabled,
            "dispatch_to_layer2": (
                self.output.dispatch_to_layer2
            ),
        }


# ============================================================
# SETTINGS FACTORIES
# ============================================================

def create_default_settings() -> Layer1Settings:
    """
    Create the recommended NOONGIL Layer 1 development settings.
    """

    settings = Layer1Settings()
    settings.validate()

    return settings


def create_simulation_settings() -> Layer1Settings:
    """
    Create settings for laptop-only simulated phone input.
    """

    settings = Layer1Settings()

    settings.runtime.execution_mode = ExecutionMode.SIMULATION
    settings.runtime.continuous_operation = False
    settings.runtime.maximum_cycles = 1

    settings.network.enabled = False
    settings.simulation.enabled = True

    settings.environment.enabled = False

    settings.validate()

    return settings


def create_live_phone_settings(
    *,
    host: str = "0.0.0.0",
    port: int = 8765,
) -> Layer1Settings:
    """
    Create settings for live smartphone-to-laptop input.
    """

    settings = Layer1Settings()

    settings.runtime.execution_mode = ExecutionMode.LIVE
    settings.runtime.continuous_operation = True
    settings.runtime.maximum_cycles = None

    settings.network.enabled = True
    settings.network.host = host
    settings.network.port = port

    settings.simulation.enabled = False

    settings.validate()

    return settings


def create_test_settings() -> Layer1Settings:
    """
    Create fast deterministic settings for automated tests.
    """

    settings = create_simulation_settings()

    settings.runtime.execution_mode = ExecutionMode.TEST
    settings.runtime.cycle_interval_seconds = 0.01
    settings.runtime.maximum_cycles = 1

    settings.runtime.save_intermediate_outputs = False

    settings.output.save_final_packet = False
    settings.output.dispatch_to_layer2 = False

    settings.logging.log_to_file = False
    settings.logging.level = LogLevel.DEBUG

    settings.simulation.random_seed = 1
    settings.simulation.introduce_random_latency = False
    settings.simulation.introduce_missing_modalities = False
    settings.simulation.introduce_degraded_quality = False

    settings.validate()

    return settings


# ============================================================
# GLOBAL DEFAULT SETTINGS
# ============================================================

DEFAULT_SETTINGS = create_default_settings()


# ============================================================
# COMMAND-LINE SELF-TEST
# ============================================================

def run_settings_self_test() -> bool:
    """
    Validate the settings module and its main configuration modes.
    """

    print("\n" + "=" * 68)
    print("NOONGIL-X | LAYER 1 SETTINGS TEST")
    print("=" * 68)

    test_output_path = (
        Path(__file__).resolve().parent
        / "_layer1_settings_test.json"
    )

    try:
        print("[1/6] Creating default settings...")
        default_settings = create_default_settings()
        print("[SUCCESS] Default settings are valid.")

        print("[2/6] Creating simulation settings...")
        simulation_settings = create_simulation_settings()

        if (
            simulation_settings.runtime.execution_mode
            != ExecutionMode.SIMULATION
        ):
            raise AssertionError(
                "Simulation execution mode was not configured."
            )

        print("[SUCCESS] Simulation settings are valid.")

        print("[3/6] Creating live-phone settings...")
        live_settings = create_live_phone_settings(
            host="0.0.0.0",
            port=8765,
        )

        if not live_settings.network.enabled:
            raise AssertionError(
                "Live settings did not enable networking."
            )

        print("[SUCCESS] Live-phone settings are valid.")

        print("[4/6] Creating automated-test settings...")
        test_settings = create_test_settings()

        if test_settings.output.dispatch_to_layer2:
            raise AssertionError(
                "Test settings should not dispatch to Layer 2."
            )

        print("[SUCCESS] Test settings are valid.")

        print("[5/6] Testing JSON serialization...")
        payload = default_settings.to_json(indent=2)
        parsed = json.loads(payload)

        if (
            parsed["runtime"]["execution_mode"]
            != ExecutionMode.SIMULATION.value
        ):
            raise AssertionError(
                "Serialized execution mode is incorrect."
            )

        print("[SUCCESS] Settings JSON is valid.")

        print("[6/6] Saving settings JSON...")
        saved_path = default_settings.save_json(
            test_output_path
        )

        if not saved_path.exists():
            raise AssertionError(
                "Settings test file was not created."
            )

        print(f"[SUCCESS] Test settings saved: {saved_path}")

        print("\nSettings summary:")
        print(
            json.dumps(
                default_settings.summary(),
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n" + "=" * 68)
        print("[PASSED] LAYER 1 SETTINGS ARE WORKING CORRECTLY")
        print("=" * 68)

        return True

    except Exception as error:
        print("\n" + "=" * 68)
        print("[FAILED] LAYER 1 SETTINGS TEST")
        print("=" * 68)
        print(f"[ERROR] {type(error).__name__}: {error}")

        return False


if __name__ == "__main__":
    success = run_settings_self_test()

    if not success:
        raise SystemExit(1)
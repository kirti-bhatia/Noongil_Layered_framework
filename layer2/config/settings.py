"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Layer 2 Runtime Settings
File    : layer2/config/settings.py
============================================================

Purpose
-------
Provides centralized, validated configuration for:

- Vision perception
- Object detection and tracking
- OCR and text perception
- Audio and speech perception
- Depth and obstacle perception
- Motion and activity perception
- Prediction-confidence calibration
- Multimodal fusion
- Pipeline execution

Model names and model paths are intentionally excluded. They
will be defined separately in config/model_config.py.

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
import os

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


# ============================================================
# CONSTANTS
# ============================================================

SETTINGS_VERSION = "1.0"

SUPPORTED_DEVICES = {
    "auto",
    "cpu",
    "cuda",
    "mps",
}

SUPPORTED_PRECISIONS = {
    "auto",
    "float32",
    "float16",
    "bfloat16",
    "int8",
}


# ============================================================
# EXCEPTIONS
# ============================================================

class Layer2SettingsError(Exception):
    """Base exception for Layer 2 settings."""


class Layer2SettingsValidationError(
    Layer2SettingsError
):
    """Raised when a setting is invalid."""


class Layer2SettingsSerializationError(
    Layer2SettingsError
):
    """Raised when settings cannot be serialized."""


# ============================================================
# ENUMERATIONS
# ============================================================

class RuntimeMode(str, Enum):
    """Layer 2 runtime mode."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_probability(
    value: Any,
    field_name: str,
) -> float:
    """Validate a number between zero and one."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise Layer2SettingsValidationError(
            f"{field_name} must be numeric."
        )

    number = float(value)

    if not math.isfinite(number):
        raise Layer2SettingsValidationError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= number <= 1.0:
        raise Layer2SettingsValidationError(
            f"{field_name} must be between 0 and 1."
        )

    return number


def validate_positive_number(
    value: Any,
    field_name: str,
    *,
    allow_zero: bool = False,
) -> float:
    """Validate a positive numerical value."""

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise Layer2SettingsValidationError(
            f"{field_name} must be numeric."
        )

    number = float(value)

    if not math.isfinite(number):
        raise Layer2SettingsValidationError(
            f"{field_name} must be finite."
        )

    if allow_zero:
        valid = number >= 0.0
    else:
        valid = number > 0.0

    if not valid:
        condition = (
            "non-negative"
            if allow_zero
            else "greater than zero"
        )

        raise Layer2SettingsValidationError(
            f"{field_name} must be {condition}."
        )

    return number


def validate_positive_integer(
    value: Any,
    field_name: str,
    *,
    allow_zero: bool = False,
) -> int:
    """Validate a positive integer."""

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise Layer2SettingsValidationError(
            f"{field_name} must be an integer."
        )

    if allow_zero:
        valid = value >= 0
    else:
        valid = value > 0

    if not valid:
        condition = (
            "non-negative"
            if allow_zero
            else "greater than zero"
        )

        raise Layer2SettingsValidationError(
            f"{field_name} must be {condition}."
        )

    return value


def require_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    """Validate and return a non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise Layer2SettingsValidationError(
            f"{field_name} must be a non-empty string."
        )

    return value.strip()


# ============================================================
# MODULE ENABLEMENT
# ============================================================

@dataclass
class ModuleSettings:
    """Enable or disable Layer 2 modules."""

    vision_processing: bool = True
    scene_classification: bool = True
    object_detection: bool = True
    object_tracking: bool = True
    activity_recognition: bool = True

    ocr: bool = True
    text_interpretation: bool = True

    audio_processing: bool = True
    speech_recognition: bool = True
    sound_event_detection: bool = True

    depth_estimation: bool = True
    obstacle_detection: bool = True

    motion_analysis: bool = True
    confidence_calibration: bool = True
    multimodal_fusion: bool = True

    def validate(self) -> None:

        for name, value in asdict(self).items():
            if not isinstance(value, bool):
                raise Layer2SettingsValidationError(
                    f"modules.{name} must be boolean."
                )


# ============================================================
# VISION SETTINGS
# ============================================================

@dataclass
class VisionSettings:
    """Vision and object-perception settings."""

    scene_confidence_threshold: float = 0.40
    object_confidence_threshold: float = 0.35
    activity_confidence_threshold: float = 0.40

    object_iou_threshold: float = 0.45
    tracking_iou_threshold: float = 0.30

    maximum_detections: int = 100
    input_width: int = 640
    input_height: int = 640

    preserve_aspect_ratio: bool = True
    enable_object_prioritization: bool = True

    navigation_priority_boost: float = 0.15
    hazard_candidate_threshold: float = 0.65

    def validate(self) -> None:

        probability_fields = (
            "scene_confidence_threshold",
            "object_confidence_threshold",
            "activity_confidence_threshold",
            "object_iou_threshold",
            "tracking_iou_threshold",
            "navigation_priority_boost",
            "hazard_candidate_threshold",
        )

        for field_name in probability_fields:
            setattr(
                self,
                field_name,
                validate_probability(
                    getattr(self, field_name),
                    f"vision.{field_name}",
                ),
            )

        self.maximum_detections = (
            validate_positive_integer(
                self.maximum_detections,
                "vision.maximum_detections",
            )
        )

        self.input_width = (
            validate_positive_integer(
                self.input_width,
                "vision.input_width",
            )
        )

        self.input_height = (
            validate_positive_integer(
                self.input_height,
                "vision.input_height",
            )
        )


# ============================================================
# TEXT SETTINGS
# ============================================================

@dataclass
class TextSettings:
    """OCR and text-understanding settings."""

    detection_confidence_threshold: float = 0.40
    recognition_confidence_threshold: float = 0.45
    critical_text_threshold: float = 0.70

    minimum_text_length: int = 1
    maximum_text_regions: int = 50

    language: str = "en"
    use_orientation_detection: bool = True
    preserve_text_case: bool = True
    enable_text_prioritization: bool = True

    def validate(self) -> None:

        self.detection_confidence_threshold = (
            validate_probability(
                self.detection_confidence_threshold,
                (
                    "text."
                    "detection_confidence_threshold"
                ),
            )
        )

        self.recognition_confidence_threshold = (
            validate_probability(
                self.recognition_confidence_threshold,
                (
                    "text."
                    "recognition_confidence_threshold"
                ),
            )
        )

        self.critical_text_threshold = (
            validate_probability(
                self.critical_text_threshold,
                "text.critical_text_threshold",
            )
        )

        self.minimum_text_length = (
            validate_positive_integer(
                self.minimum_text_length,
                "text.minimum_text_length",
            )
        )

        self.maximum_text_regions = (
            validate_positive_integer(
                self.maximum_text_regions,
                "text.maximum_text_regions",
            )
        )

        self.language = require_non_empty_string(
            self.language,
            "text.language",
        )


# ============================================================
# AUDIO SETTINGS
# ============================================================

@dataclass
class AudioSettings:
    """Audio, speech and sound-event settings."""

    target_sample_rate_hz: int = 16000
    target_channels: int = 1

    speech_confidence_threshold: float = 0.45
    sound_confidence_threshold: float = 0.45
    emergency_sound_threshold: float = 0.75

    minimum_audio_duration_seconds: float = 0.10
    maximum_audio_duration_seconds: float = 30.0

    language: str = "en"
    enable_noise_reduction: bool = True
    enable_voice_activity_detection: bool = True
    enable_emergency_sound_priority: bool = True

    def validate(self) -> None:

        self.target_sample_rate_hz = (
            validate_positive_integer(
                self.target_sample_rate_hz,
                "audio.target_sample_rate_hz",
            )
        )

        self.target_channels = (
            validate_positive_integer(
                self.target_channels,
                "audio.target_channels",
            )
        )

        for field_name in (
            "speech_confidence_threshold",
            "sound_confidence_threshold",
            "emergency_sound_threshold",
        ):
            setattr(
                self,
                field_name,
                validate_probability(
                    getattr(self, field_name),
                    f"audio.{field_name}",
                ),
            )

        self.minimum_audio_duration_seconds = (
            validate_positive_number(
                self.minimum_audio_duration_seconds,
                (
                    "audio."
                    "minimum_audio_duration_seconds"
                ),
            )
        )

        self.maximum_audio_duration_seconds = (
            validate_positive_number(
                self.maximum_audio_duration_seconds,
                (
                    "audio."
                    "maximum_audio_duration_seconds"
                ),
            )
        )

        if (
            self.minimum_audio_duration_seconds
            > self.maximum_audio_duration_seconds
        ):
            raise Layer2SettingsValidationError(
                "audio.minimum_audio_duration_seconds "
                "cannot exceed "
                "audio.maximum_audio_duration_seconds."
            )

        self.language = require_non_empty_string(
            self.language,
            "audio.language",
        )


# ============================================================
# SPATIAL SETTINGS
# ============================================================

@dataclass
class SpatialSettings:
    """Depth, distance and obstacle settings."""

    minimum_depth_m: float = 0.10
    maximum_depth_m: float = 50.0

    obstacle_distance_threshold_m: float = 3.0
    critical_obstacle_distance_m: float = 1.0

    depth_confidence_threshold: float = 0.40
    obstacle_confidence_threshold: float = 0.50

    enable_depth_normalization: bool = True
    enable_free_space_estimation: bool = True

    def validate(self) -> None:

        self.minimum_depth_m = (
            validate_positive_number(
                self.minimum_depth_m,
                "spatial.minimum_depth_m",
            )
        )

        self.maximum_depth_m = (
            validate_positive_number(
                self.maximum_depth_m,
                "spatial.maximum_depth_m",
            )
        )

        if self.minimum_depth_m >= self.maximum_depth_m:
            raise Layer2SettingsValidationError(
                "spatial.minimum_depth_m must be "
                "smaller than spatial.maximum_depth_m."
            )

        self.obstacle_distance_threshold_m = (
            validate_positive_number(
                self.obstacle_distance_threshold_m,
                (
                    "spatial."
                    "obstacle_distance_threshold_m"
                ),
            )
        )

        self.critical_obstacle_distance_m = (
            validate_positive_number(
                self.critical_obstacle_distance_m,
                (
                    "spatial."
                    "critical_obstacle_distance_m"
                ),
            )
        )

        if (
            self.critical_obstacle_distance_m
            > self.obstacle_distance_threshold_m
        ):
            raise Layer2SettingsValidationError(
                "spatial.critical_obstacle_distance_m "
                "cannot exceed "
                "spatial.obstacle_distance_threshold_m."
            )

        self.depth_confidence_threshold = (
            validate_probability(
                self.depth_confidence_threshold,
                (
                    "spatial."
                    "depth_confidence_threshold"
                ),
            )
        )

        self.obstacle_confidence_threshold = (
            validate_probability(
                self.obstacle_confidence_threshold,
                (
                    "spatial."
                    "obstacle_confidence_threshold"
                ),
            )
        )


# ============================================================
# MOTION SETTINGS
# ============================================================

@dataclass
class MotionSettings:
    """Motion and user-activity settings."""

    stationary_threshold_mps2: float = 0.20
    walking_threshold_mps2: float = 0.80
    running_threshold_mps2: float = 2.00

    activity_confidence_threshold: float = 0.40
    smoothing_window_size: int = 5

    use_visual_activity_support: bool = True
    use_gps_speed_support: bool = True

    def validate(self) -> None:

        self.stationary_threshold_mps2 = (
            validate_positive_number(
                self.stationary_threshold_mps2,
                (
                    "motion."
                    "stationary_threshold_mps2"
                ),
                allow_zero=True,
            )
        )

        self.walking_threshold_mps2 = (
            validate_positive_number(
                self.walking_threshold_mps2,
                "motion.walking_threshold_mps2",
            )
        )

        self.running_threshold_mps2 = (
            validate_positive_number(
                self.running_threshold_mps2,
                "motion.running_threshold_mps2",
            )
        )

        if not (
            self.stationary_threshold_mps2
            < self.walking_threshold_mps2
            < self.running_threshold_mps2
        ):
            raise Layer2SettingsValidationError(
                "Motion thresholds must satisfy: "
                "stationary < walking < running."
            )

        self.activity_confidence_threshold = (
            validate_probability(
                self.activity_confidence_threshold,
                (
                    "motion."
                    "activity_confidence_threshold"
                ),
            )
        )

        self.smoothing_window_size = (
            validate_positive_integer(
                self.smoothing_window_size,
                "motion.smoothing_window_size",
            )
        )


# ============================================================
# CONFIDENCE SETTINGS
# ============================================================

@dataclass
class ConfidenceSettings:
    """Prediction-confidence calibration settings."""

    minimum_usable_confidence: float = 0.35
    trusted_confidence_threshold: float = 0.75
    degraded_confidence_threshold: float = 0.50

    calibration_temperature: float = 1.0
    sensor_confidence_weight: float = 0.30
    model_confidence_weight: float = 0.70

    reject_below_minimum: bool = False

    def validate(self) -> None:

        for field_name in (
            "minimum_usable_confidence",
            "trusted_confidence_threshold",
            "degraded_confidence_threshold",
            "sensor_confidence_weight",
            "model_confidence_weight",
        ):
            setattr(
                self,
                field_name,
                validate_probability(
                    getattr(self, field_name),
                    f"confidence.{field_name}",
                ),
            )

        if not (
            self.minimum_usable_confidence
            <= self.degraded_confidence_threshold
            <= self.trusted_confidence_threshold
        ):
            raise Layer2SettingsValidationError(
                "Confidence thresholds must satisfy: "
                "minimum usable <= degraded <= trusted."
            )

        total_weight = (
            self.sensor_confidence_weight
            + self.model_confidence_weight
        )

        if not math.isclose(
            total_weight,
            1.0,
            abs_tol=1e-6,
        ):
            raise Layer2SettingsValidationError(
                "confidence sensor and model weights "
                "must sum to 1.0."
            )

        self.calibration_temperature = (
            validate_positive_number(
                self.calibration_temperature,
                (
                    "confidence."
                    "calibration_temperature"
                ),
            )
        )


# ============================================================
# FUSION SETTINGS
# ============================================================

@dataclass
class FusionSettings:
    """Confidence-aware multimodal fusion settings."""

    vision_weight: float = 0.35
    audio_weight: float = 0.20
    spatial_weight: float = 0.20
    motion_weight: float = 0.10
    text_weight: float = 0.15

    minimum_modalities: int = 1
    minimum_fusion_confidence: float = 0.40

    normalize_features: bool = True
    use_confidence_weighting: bool = True
    allow_partial_fusion: bool = True

    conflict_threshold: float = 0.35
    epsilon: float = 1e-8

    def weights(self) -> Dict[str, float]:
        """Return modality fusion weights."""

        return {
            "vision": self.vision_weight,
            "audio": self.audio_weight,
            "spatial": self.spatial_weight,
            "motion": self.motion_weight,
            "text": self.text_weight,
        }

    def validate(self) -> None:

        for name, value in self.weights().items():
            validated = validate_probability(
                value,
                f"fusion.{name}_weight",
            )

            setattr(
                self,
                f"{name}_weight",
                validated,
            )

        weight_total = sum(
            self.weights().values()
        )

        if not math.isclose(
            weight_total,
            1.0,
            abs_tol=1e-6,
        ):
            raise Layer2SettingsValidationError(
                "Fusion weights must sum to 1.0. "
                f"Current sum: {weight_total}"
            )

        self.minimum_modalities = (
            validate_positive_integer(
                self.minimum_modalities,
                "fusion.minimum_modalities",
            )
        )

        if self.minimum_modalities > 5:
            raise Layer2SettingsValidationError(
                "fusion.minimum_modalities cannot "
                "exceed five fusion modalities."
            )

        self.minimum_fusion_confidence = (
            validate_probability(
                self.minimum_fusion_confidence,
                (
                    "fusion."
                    "minimum_fusion_confidence"
                ),
            )
        )

        self.conflict_threshold = (
            validate_probability(
                self.conflict_threshold,
                "fusion.conflict_threshold",
            )
        )

        self.epsilon = validate_positive_number(
            self.epsilon,
            "fusion.epsilon",
        )


# ============================================================
# EXECUTION SETTINGS
# ============================================================

@dataclass
class ExecutionSettings:
    """Pipeline execution and resource settings."""

    device: str = "auto"
    precision: str = "auto"

    enable_parallel_processing: bool = True
    maximum_workers: int = 4

    module_timeout_seconds: float = 60.0
    continue_on_module_failure: bool = True

    save_intermediate_outputs: bool = True
    validate_every_result: bool = True

    log_level: str = "INFO"
    random_seed: int = 42

    def validate(self) -> None:

        self.device = require_non_empty_string(
            self.device,
            "execution.device",
        ).lower()

        if self.device not in SUPPORTED_DEVICES:
            raise Layer2SettingsValidationError(
                f"Unsupported device: {self.device!r}"
            )

        self.precision = require_non_empty_string(
            self.precision,
            "execution.precision",
        ).lower()

        if self.precision not in SUPPORTED_PRECISIONS:
            raise Layer2SettingsValidationError(
                f"Unsupported precision: "
                f"{self.precision!r}"
            )

        self.maximum_workers = (
            validate_positive_integer(
                self.maximum_workers,
                "execution.maximum_workers",
            )
        )

        self.module_timeout_seconds = (
            validate_positive_number(
                self.module_timeout_seconds,
                (
                    "execution."
                    "module_timeout_seconds"
                ),
            )
        )

        self.log_level = (
            require_non_empty_string(
                self.log_level,
                "execution.log_level",
            ).upper()
        )

        if self.log_level not in {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }:
            raise Layer2SettingsValidationError(
                f"Unsupported log level: "
                f"{self.log_level!r}"
            )

        if isinstance(
            self.random_seed,
            bool,
        ) or not isinstance(
            self.random_seed,
            int,
        ):
            raise Layer2SettingsValidationError(
                "execution.random_seed must be "
                "an integer."
            )


# ============================================================
# COMPLETE LAYER 2 SETTINGS
# ============================================================

@dataclass
class Layer2Settings:
    """Complete validated Layer 2 configuration."""

    runtime_mode: RuntimeMode = (
        RuntimeMode.DEVELOPMENT
    )

    modules: ModuleSettings = field(
        default_factory=ModuleSettings
    )

    vision: VisionSettings = field(
        default_factory=VisionSettings
    )

    text: TextSettings = field(
        default_factory=TextSettings
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

    confidence: ConfidenceSettings = field(
        default_factory=ConfidenceSettings
    )

    fusion: FusionSettings = field(
        default_factory=FusionSettings
    )

    execution: ExecutionSettings = field(
        default_factory=ExecutionSettings
    )

    settings_version: str = SETTINGS_VERSION

    def __post_init__(self) -> None:

        if isinstance(
            self.runtime_mode,
            str,
        ):
            try:
                self.runtime_mode = RuntimeMode(
                    self.runtime_mode.lower()
                )
            except ValueError as error:
                raise Layer2SettingsValidationError(
                    f"Unsupported runtime mode: "
                    f"{self.runtime_mode!r}"
                ) from error

        self.validate()

    def validate(self) -> None:
        """Validate all settings groups."""

        if not isinstance(
            self.runtime_mode,
            RuntimeMode,
        ):
            raise Layer2SettingsValidationError(
                "runtime_mode must be a RuntimeMode."
            )

        self.settings_version = (
            require_non_empty_string(
                self.settings_version,
                "settings_version",
            )
        )

        self.modules.validate()
        self.vision.validate()
        self.text.validate()
        self.audio.validate()
        self.spatial.validate()
        self.motion.validate()
        self.confidence.validate()
        self.fusion.validate()
        self.execution.validate()

        if (
            self.modules.object_tracking
            and not self.modules.object_detection
        ):
            raise Layer2SettingsValidationError(
                "Object tracking requires "
                "object detection."
            )

        if (
            self.modules.text_interpretation
            and not self.modules.ocr
        ):
            raise Layer2SettingsValidationError(
                "Text interpretation requires OCR."
            )

        if (
            self.modules.obstacle_detection
            and not self.modules.depth_estimation
        ):
            raise Layer2SettingsValidationError(
                "Obstacle detection requires "
                "depth estimation."
            )

        if (
            self.modules.multimodal_fusion
            and not any(
                (
                    self.modules.vision_processing,
                    self.modules.audio_processing,
                    self.modules.motion_analysis,
                    self.modules.ocr,
                    self.modules.depth_estimation,
                )
            )
        ):
            raise Layer2SettingsValidationError(
                "Multimodal fusion requires at "
                "least one perception module."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe settings."""

        self.validate()

        payload = asdict(self)
        payload["runtime_mode"] = (
            self.runtime_mode.value
        )

        return payload

    def to_json(
        self,
        *,
        indent: int = 2,
    ) -> str:
        """Serialize settings to JSON."""

        try:
            return json.dumps(
                self.to_dict(),
                indent=indent,
                ensure_ascii=False,
            )
        except (TypeError, ValueError) as error:
            raise Layer2SettingsSerializationError(
                "Unable to serialize Layer 2 settings."
            ) from error

    def write_json(
        self,
        file_path: Path | str,
    ) -> Path:
        """Write settings to a JSON file."""

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
            raise Layer2SettingsSerializationError(
                f"Unable to write settings: "
                f"{output_path}"
            ) from error

        return output_path

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Layer2Settings":
        """Create settings from a dictionary."""

        if not isinstance(payload, Mapping):
            raise Layer2SettingsValidationError(
                "Settings payload must be a dictionary."
            )

        return cls(
            runtime_mode=payload.get(
                "runtime_mode",
                RuntimeMode.DEVELOPMENT,
            ),
            modules=ModuleSettings(
                **dict(
                    payload.get("modules", {})
                )
            ),
            vision=VisionSettings(
                **dict(
                    payload.get("vision", {})
                )
            ),
            text=TextSettings(
                **dict(
                    payload.get("text", {})
                )
            ),
            audio=AudioSettings(
                **dict(
                    payload.get("audio", {})
                )
            ),
            spatial=SpatialSettings(
                **dict(
                    payload.get("spatial", {})
                )
            ),
            motion=MotionSettings(
                **dict(
                    payload.get("motion", {})
                )
            ),
            confidence=ConfidenceSettings(
                **dict(
                    payload.get("confidence", {})
                )
            ),
            fusion=FusionSettings(
                **dict(
                    payload.get("fusion", {})
                )
            ),
            execution=ExecutionSettings(
                **dict(
                    payload.get("execution", {})
                )
            ),
            settings_version=payload.get(
                "settings_version",
                SETTINGS_VERSION,
            ),
        )


# ============================================================
# SETTINGS FACTORIES
# ============================================================

def create_default_settings() -> Layer2Settings:
    """Create development settings."""

    device = os.getenv(
        "NOONGIL_LAYER2_DEVICE",
        "auto",
    )

    return Layer2Settings(
        runtime_mode=RuntimeMode.DEVELOPMENT,
        execution=ExecutionSettings(
            device=device,
            precision="auto",
            enable_parallel_processing=True,
            maximum_workers=4,
            save_intermediate_outputs=True,
            validate_every_result=True,
            log_level="INFO",
        ),
    )


def create_test_settings() -> Layer2Settings:
    """Create deterministic test settings."""

    return Layer2Settings(
        runtime_mode=RuntimeMode.TEST,
        execution=ExecutionSettings(
            device="cpu",
            precision="float32",
            enable_parallel_processing=False,
            maximum_workers=1,
            module_timeout_seconds=30.0,
            continue_on_module_failure=False,
            save_intermediate_outputs=True,
            validate_every_result=True,
            log_level="DEBUG",
            random_seed=42,
        ),
    )


def create_production_settings() -> Layer2Settings:
    """Create production-oriented settings."""

    device = os.getenv(
        "NOONGIL_LAYER2_DEVICE",
        "auto",
    )

    return Layer2Settings(
        runtime_mode=RuntimeMode.PRODUCTION,
        execution=ExecutionSettings(
            device=device,
            precision="auto",
            enable_parallel_processing=True,
            maximum_workers=4,
            module_timeout_seconds=60.0,
            continue_on_module_failure=True,
            save_intermediate_outputs=False,
            validate_every_result=True,
            log_level="INFO",
            random_seed=42,
        ),
    )


def load_settings(
    file_path: Path | str,
) -> Layer2Settings:
    """Load settings from JSON."""

    path = Path(file_path)

    if not path.exists():
        raise Layer2SettingsError(
            f"Settings file does not exist: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise Layer2SettingsSerializationError(
            f"Invalid JSON in {path}: "
            f"line {error.lineno}, "
            f"column {error.colno}."
        ) from error
    except OSError as error:
        raise Layer2SettingsSerializationError(
            f"Unable to read settings: {path}"
        ) from error

    return Layer2Settings.from_dict(payload)


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 72)
    print("NOONGIL-X | LAYER 2 SETTINGS SELF-TEST")
    print("=" * 72)

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    test_output = (
        project_root
        / "output"
        / "layer2"
        / "config_self_test"
        / "layer2_settings.json"
    )

    try:
        default_settings = (
            create_default_settings()
        )

        print("[PASS] Default settings created")
        print("[PASS] Default settings validated")

        test_settings = create_test_settings()

        print("[PASS] Test settings created")
        print("[PASS] Test settings validated")

        production_settings = (
            create_production_settings()
        )

        print("[PASS] Production settings created")
        print("[PASS] Production settings validated")

        written_path = test_settings.write_json(
            test_output
        )

        print(
            f"[PASS] Settings serialized: "
            f"{written_path}"
        )

        restored = load_settings(
            written_path
        )

        if (
            restored.to_dict()
            != test_settings.to_dict()
        ):
            raise AssertionError(
                "Restored settings do not match "
                "the original settings."
            )

        print("[PASS] Settings restored from JSON")

        weight_total = sum(
            restored.fusion.weights().values()
        )

        if not math.isclose(
            weight_total,
            1.0,
            abs_tol=1e-6,
        ):
            raise AssertionError(
                "Fusion weights do not sum to 1.0."
            )

        print("[PASS] Fusion weights validated")

        try:
            invalid_settings = (
                create_test_settings()
            )

            invalid_settings.vision\
                .object_confidence_threshold = 1.5

            invalid_settings.validate()

        except Layer2SettingsValidationError:
            print(
                "[PASS] Invalid threshold rejected"
            )
        else:
            raise AssertionError(
                "Invalid threshold was accepted."
            )

        print("\nSettings summary:")
        print(
            f"  mode: "
            f"{restored.runtime_mode.value}"
        )
        print(
            f"  device: "
            f"{restored.execution.device}"
        )
        print(
            f"  precision: "
            f"{restored.execution.precision}"
        )
        print(
            f"  parallel: "
            f"{restored.execution.enable_parallel_processing}"
        )
        print(
            f"  fusion weights: "
            f"{restored.fusion.weights()}"
        )
        print(
            f"  object threshold: "
            f"{restored.vision.object_confidence_threshold}"
        )
        print(
            f"  speech threshold: "
            f"{restored.audio.speech_confidence_threshold}"
        )
        print(
            f"  obstacle distance: "
            f"{restored.spatial.obstacle_distance_threshold_m} m"
        )

        print("\n" + "=" * 72)
        print("[PASSED] LAYER 2 SETTINGS ARE WORKING")
        print("=" * 72)

        return True

    except (
        Layer2SettingsError,
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
            "Run the NOONGIL-X Layer 2 "
            "settings self-test."
        )
    )


def main() -> int:

    build_argument_parser().parse_args()

    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
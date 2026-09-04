"""
============================================================
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : Confidence Estimator
File    : layer1/processing/confidence_estimator.py
============================================================

Purpose
-------
Estimate confidence for every selected Layer 1 modality and
produce one overall confidence report for the synchronized frame.

Confidence model
----------------
For modality i:

    C_i =
        w_q * Q_i
      + w_f * F_i
      + w_s * S_i
      + w_h * H_i
      + w_d * D_i
      - P_i

where:

    Q_i = signal quality
    F_i = freshness
    S_i = synchronization quality
    H_i = sensor/integrity health
    D_i = device/network support
    P_i = limitation and degradation penalty

Overall confidence:

    C_overall =
        sum(alpha_i * C_i) / sum(alpha_i)

The result is clamped to [0, 1].

Architectural Boundary
----------------------
This module does NOT:
- repair missing modalities;
- modify sensor values;
- perform semantic fusion;
- perform perception;
- perform reasoning;
- run an LLM.

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import json
import math
import time
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional

from layer1.config.settings import (
    Layer1Settings,
    create_default_settings,
    create_test_settings,
)
from layer1.processing.multimodal_synchronizer import (
    ModalitySynchronizationRecord,
    SynchronizedMultimodalFrame,
    SynchronizationStatus,
)
from layer1.utils.logger import (
    PipelineTimer,
    get_logger,
    log_exception,
    log_sensor_event,
)


# ============================================================
# CONSTANTS
# ============================================================

CONFIDENCE_ESTIMATOR_VERSION = "1.0"

SUPPORTED_MODALITIES = {
    "vision",
    "audio",
    "spatial",
    "motion",
    "interaction",
    "wearable",
    "source_device",
    "environment",
}

DEFAULT_COMPONENT_WEIGHTS: Dict[str, float] = {
    "quality": 0.30,
    "freshness": 0.20,
    "synchronization": 0.20,
    "health": 0.20,
    "device": 0.10,
}

DEFAULT_MODALITY_WEIGHTS: Dict[str, float] = {
    "vision": 0.22,
    "audio": 0.16,
    "spatial": 0.18,
    "motion": 0.16,
    "interaction": 0.10,
    "wearable": 0.08,
    "source_device": 0.07,
    "environment": 0.03,
}

DEFAULT_LIMITATION_PENALTIES: Dict[str, float] = {
    "low_brightness": 0.12,
    "low_sharpness": 0.15,
    "low_frame_integrity": 0.20,
    "simulated_degraded_quality": 0.20,

    "low_signal_to_noise": 0.18,
    "high_clipping": 0.18,
    "mostly_silent": 0.10,

    "poor_horizontal_accuracy": 0.25,
    "reduced_horizontal_accuracy": 0.10,
    "stale_spatial_data": 0.25,
    "unreasonable_speed_value": 0.20,

    "incomplete_motion_sensor_group": 0.20,
    "sensor_saturation_detected": 0.25,
    "stale_motion_data": 0.20,

    "stale_interaction_event": 0.15,

    "wearable_not_connected": 0.35,
    "audio_output_unavailable": 0.15,
    "microphone_unavailable": 0.15,

    "outside_synchronization_window": 0.30,
    "stale_modality_sample": 0.30,
    "modality_missing": 1.00,
}

DEFAULT_UNKNOWN_LIMITATION_PENALTY = 0.05


# ============================================================
# EXCEPTIONS
# ============================================================

class ConfidenceEstimatorError(Exception):
    """Base exception for confidence estimation."""


class ConfidenceValidationError(ConfidenceEstimatorError):
    """Raised when confidence input is invalid."""


class ConfidenceProcessingError(ConfidenceEstimatorError):
    """Raised when a confidence report cannot be generated."""


# ============================================================
# ENUMERATIONS
# ============================================================

class ConfidenceLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ConfidenceComponents:
    """
    Component scores used for one modality.
    """

    quality: float
    freshness: float
    synchronization: float
    health: float
    device: float
    penalty: float

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ConfidenceValidationError(
                    f"{name} must be between 0 and 1."
                )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass
class ModalityConfidence:
    """
    Confidence result for one modality.
    """

    modality: str
    available: bool
    selected: bool

    confidence_score: float
    confidence_level: ConfidenceLevel

    components: ConfidenceComponents

    limitation_codes: List[str] = field(
        default_factory=list
    )
    reason_codes: List[str] = field(
        default_factory=list
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        if self.modality not in SUPPORTED_MODALITIES:
            raise ConfidenceValidationError(
                f"Unsupported modality: {self.modality!r}"
            )

        if (
            isinstance(self.confidence_score, bool)
            or not isinstance(
                self.confidence_score,
                (int, float),
            )
            or not math.isfinite(
                float(self.confidence_score)
            )
            or not 0.0 <= self.confidence_score <= 1.0
        ):
            raise ConfidenceValidationError(
                "confidence_score must be between 0 and 1."
            )

        self.components.validate()

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["confidence_level"] = (
            self.confidence_level.value
        )
        payload["components"] = (
            self.components.to_dict()
        )
        return payload


@dataclass
class ConfidenceReport:
    """
    Complete confidence report for one synchronized frame.
    """

    report_id: str
    generated_at: str
    frame_id: str

    modality_confidences: Dict[
        str,
        ModalityConfidence,
    ]

    overall_confidence: float
    overall_level: ConfidenceLevel

    synchronization_confidence: float
    completeness_confidence: float
    device_support_confidence: float

    trusted_modalities: List[str]
    uncertain_modalities: List[str]
    unavailable_modalities: List[str]

    warnings: List[str] = field(
        default_factory=list
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        if not self.report_id.strip():
            raise ConfidenceValidationError(
                "report_id cannot be empty."
            )

        if not self.frame_id.strip():
            raise ConfidenceValidationError(
                "frame_id cannot be empty."
            )

        for name, value in (
            ("overall_confidence", self.overall_confidence),
            (
                "synchronization_confidence",
                self.synchronization_confidence,
            ),
            (
                "completeness_confidence",
                self.completeness_confidence,
            ),
            (
                "device_support_confidence",
                self.device_support_confidence,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise ConfidenceValidationError(
                    f"{name} must be between 0 and 1."
                )

        for result in self.modality_confidences.values():
            result.validate()

    def to_dict(self) -> Dict[str, Any]:
        self.validate()

        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "frame_id": self.frame_id,
            "modality_confidences": {
                modality: confidence.to_dict()
                for modality, confidence
                in self.modality_confidences.items()
            },
            "overall_confidence": self.overall_confidence,
            "overall_level": self.overall_level.value,
            "synchronization_confidence": (
                self.synchronization_confidence
            ),
            "completeness_confidence": (
                self.completeness_confidence
            ),
            "device_support_confidence": (
                self.device_support_confidence
            ),
            "trusted_modalities": (
                self.trusted_modalities
            ),
            "uncertain_modalities": (
                self.uncertain_modalities
            ),
            "unavailable_modalities": (
                self.unavailable_modalities
            ),
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


@dataclass
class ConfidenceEstimatorStatistics:
    """
    Runtime statistics for ConfidenceEstimator.
    """

    total_reports: int = 0
    high_confidence_reports: int = 0
    medium_confidence_reports: int = 0
    low_confidence_reports: int = 0

    total_unavailable_modalities: int = 0
    total_uncertain_modalities: int = 0

    cumulative_processing_seconds: float = 0.0

    last_report_id: Optional[str] = None
    last_frame_id: Optional[str] = None
    last_overall_confidence: Optional[float] = None
    last_error: Optional[str] = None

    @property
    def average_processing_seconds(self) -> float:
        if self.total_reports == 0:
            return 0.0

        return (
            self.cumulative_processing_seconds
            / self.total_reports
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["average_processing_seconds"] = (
            self.average_processing_seconds
        )
        return payload


# ============================================================
# HELPERS
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="milliseconds"
    )


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(minimum, min(maximum, value))


def safe_float(
    value: Any,
    default: float,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(parsed):
        return default

    return parsed


def average(
    values: Iterable[Optional[float]],
    *,
    default: float = 0.0,
) -> float:
    valid = [
        float(value)
        for value in values
        if value is not None
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]

    if not valid:
        return default

    return sum(valid) / len(valid)


def confidence_level(
    score: float,
) -> ConfidenceLevel:
    if score >= 0.90:
        return ConfidenceLevel.VERY_HIGH

    if score >= 0.75:
        return ConfidenceLevel.HIGH

    if score >= 0.55:
        return ConfidenceLevel.MEDIUM

    if score >= 0.35:
        return ConfidenceLevel.LOW

    return ConfidenceLevel.VERY_LOW


def get_metadata_limitations(
    value: Any,
) -> List[str]:
    metadata = getattr(value, "metadata", None)

    if metadata is None:
        return []

    limitations = getattr(
        metadata,
        "limitations",
        [],
    )

    if not isinstance(limitations, list):
        return []

    return [
        str(item)
        for item in limitations
    ]


# ============================================================
# CONFIDENCE ESTIMATOR
# ============================================================

class ConfidenceEstimator:
    """
    Estimate per-modality and overall Layer 1 confidence.
    """

    def __init__(
        self,
        settings: Optional[Layer1Settings] = None,
    ) -> None:
        self.settings = settings or create_default_settings()
        self.settings.validate()

        self.logger = get_logger(
            "processing.confidence_estimator"
        )

        self.statistics = (
            ConfidenceEstimatorStatistics()
        )

        self._last_report: Optional[
            ConfidenceReport
        ] = None

        self.component_weights = (
            self._get_component_weights()
        )
        self.modality_weights = (
            self._get_modality_weights()
        )
        self.trusted_threshold = (
            self._get_trusted_threshold()
        )
        self.uncertain_threshold = (
            self._get_uncertain_threshold()
        )

    # ========================================================
    # SETTINGS COMPATIBILITY
    # ========================================================

    def _confidence_settings(self) -> Any:
        return getattr(
            self.settings,
            "confidence",
            None,
        )

    def _get_component_weights(
        self,
    ) -> Dict[str, float]:
        confidence_settings = (
            self._confidence_settings()
        )

        weights = dict(
            DEFAULT_COMPONENT_WEIGHTS
        )

        if confidence_settings is None:
            return weights

        aliases = {
            "quality": (
                "quality_weight",
                "signal_quality_weight",
            ),
            "freshness": (
                "freshness_weight",
                "temporal_freshness_weight",
            ),
            "synchronization": (
                "synchronization_weight",
                "temporal_alignment_weight",
            ),
            "health": (
                "health_weight",
                "integrity_weight",
                "sensor_health_weight",
            ),
            "device": (
                "device_weight",
                "device_support_weight",
                "network_weight",
            ),
        }

        for component, names in aliases.items():
            for name in names:
                value = getattr(
                    confidence_settings,
                    name,
                    None,
                )

                if value is None:
                    continue

                parsed = safe_float(
                    value,
                    weights[component],
                )

                if parsed < 0:
                    raise ConfidenceValidationError(
                        f"{name} cannot be negative."
                    )

                weights[component] = parsed
                break

        total = sum(weights.values())

        if total <= 0:
            raise ConfidenceValidationError(
                "Confidence component weights must "
                "sum to a positive value."
            )

        return {
            key: value / total
            for key, value in weights.items()
        }

    def _get_modality_weights(
        self,
    ) -> Dict[str, float]:
        confidence_settings = (
            self._confidence_settings()
        )

        if confidence_settings is None:
            return dict(DEFAULT_MODALITY_WEIGHTS)

        candidate = getattr(
            confidence_settings,
            "modality_weights",
            None,
        )

        if isinstance(candidate, Mapping):
            weights = dict(
                DEFAULT_MODALITY_WEIGHTS
            )

            for modality, value in candidate.items():
                normalized = str(
                    modality
                ).strip().lower()

                if normalized not in SUPPORTED_MODALITIES:
                    continue

                parsed = safe_float(
                    value,
                    weights[normalized],
                )

                if parsed < 0:
                    raise ConfidenceValidationError(
                        "Modality weights cannot be negative."
                    )

                weights[normalized] = parsed

            return weights

        return dict(DEFAULT_MODALITY_WEIGHTS)

    def _get_trusted_threshold(self) -> float:
        settings = self._confidence_settings()

        if settings is None:
            return 0.70

        for name in (
            "trusted_threshold",
            "high_confidence_threshold",
            "minimum_trusted_confidence",
        ):
            value = getattr(settings, name, None)

            if value is not None:
                return clamp(
                    safe_float(value, 0.70),
                    0.0,
                    1.0,
                )

        return 0.70

    def _get_uncertain_threshold(self) -> float:
        settings = self._confidence_settings()

        if settings is None:
            return 0.40

        for name in (
            "uncertain_threshold",
            "minimum_usable_confidence",
            "low_confidence_threshold",
        ):
            value = getattr(settings, name, None)

            if value is not None:
                return clamp(
                    safe_float(value, 0.40),
                    0.0,
                    1.0,
                )

        return 0.40

    # ========================================================
    # PUBLIC API
    # ========================================================

    def estimate(
        self,
        frame: SynchronizedMultimodalFrame,
        *,
        raise_on_error: Optional[bool] = None,
    ) -> ConfidenceReport:
        """
        Estimate confidence for one synchronized frame.
        """

        should_raise = (
            self.settings.runtime.fail_fast
            if raise_on_error is None
            else raise_on_error
        )

        started = time.perf_counter()

        try:
            with PipelineTimer(
                "confidence_estimator.estimate",
                logger=self.logger,
                metadata={
                    "frame_id": frame.frame_id,
                    "frame_status": frame.status.value,
                },
            ):
                frame.validate()

                device_support = (
                    self._calculate_device_support(
                        frame
                    )
                )

                modality_results: Dict[
                    str,
                    ModalityConfidence,
                ] = {}

                for modality in sorted(
                    SUPPORTED_MODALITIES
                ):
                    result = (
                        self._estimate_modality(
                            modality=modality,
                            frame=frame,
                            device_support=device_support,
                        )
                    )

                    modality_results[
                        modality
                    ] = result

                overall = self._calculate_overall(
                    modality_results,
                    frame=frame,
                )

                trusted = sorted(
                    modality
                    for modality, result
                    in modality_results.items()
                    if result.available
                    and result.selected
                    and result.confidence_score
                    >= self.trusted_threshold
                )

                uncertain = sorted(
                    modality
                    for modality, result
                    in modality_results.items()
                    if result.available
                    and result.selected
                    and result.confidence_score
                    < self.trusted_threshold
                )

                unavailable = sorted(
                    modality
                    for modality, result
                    in modality_results.items()
                    if not result.available
                )

                warnings: List[str] = []

                if uncertain:
                    warnings.append(
                        "uncertain_modalities_present"
                    )

                if unavailable:
                    warnings.append(
                        "unavailable_modalities_present"
                    )

                if overall < (
                    self.uncertain_threshold
                ):
                    warnings.append(
                        "overall_confidence_low"
                    )

                report = ConfidenceReport(
                    report_id=(
                        "CONF_"
                        f"{uuid.uuid4().hex[:12].upper()}"
                    ),
                    generated_at=utc_now_iso(),
                    frame_id=frame.frame_id,
                    modality_confidences=(
                        modality_results
                    ),
                    overall_confidence=round(
                        overall,
                        6,
                    ),
                    overall_level=(
                        confidence_level(overall)
                    ),
                    synchronization_confidence=(
                        frame.synchronization_score
                    ),
                    completeness_confidence=(
                        frame.completeness_score
                    ),
                    device_support_confidence=round(
                        device_support,
                        6,
                    ),
                    trusted_modalities=trusted,
                    uncertain_modalities=uncertain,
                    unavailable_modalities=unavailable,
                    warnings=warnings,
                    metadata={
                        "estimator_version": (
                            CONFIDENCE_ESTIMATOR_VERSION
                        ),
                        "frame_status": (
                            frame.status.value
                        ),
                        "component_weights": (
                            self.component_weights
                        ),
                        "modality_weights": (
                            self.modality_weights
                        ),
                        "trusted_threshold": (
                            self.trusted_threshold
                        ),
                        "uncertain_threshold": (
                            self.uncertain_threshold
                        ),
                    },
                )

                report.validate()

                elapsed = time.perf_counter() - started

                self._register_report(
                    report,
                    elapsed,
                )

                log_sensor_event(
                    modality="interaction",
                    event="Confidence report generated",
                    sensor_type="confidence_estimator",
                    packet_id=report.report_id,
                    details={
                        "frame_id": frame.frame_id,
                        "overall_confidence": (
                            report.overall_confidence
                        ),
                        "overall_level": (
                            report.overall_level.value
                        ),
                        "trusted_modalities": (
                            report.trusted_modalities
                        ),
                        "uncertain_modalities": (
                            report.uncertain_modalities
                        ),
                        "unavailable_modalities": (
                            report.unavailable_modalities
                        ),
                    },
                )

                return report

        except Exception as error:
            self.statistics.last_error = (
                f"{type(error).__name__}: {error}"
            )

            log_exception(
                self.logger,
                "Confidence estimation failed",
                error=error,
                details={
                    "frame_id": getattr(
                        frame,
                        "frame_id",
                        None,
                    ),
                },
            )

            if should_raise:
                raise

            raise ConfidenceProcessingError(
                f"Confidence estimation failed: {error}"
            ) from error

    # ========================================================
    # MODALITY ESTIMATION
    # ========================================================

    def _estimate_modality(
        self,
        *,
        modality: str,
        frame: SynchronizedMultimodalFrame,
        device_support: float,
    ) -> ModalityConfidence:
        value = getattr(frame, modality, None)

        record = frame.synchronization_records.get(
            modality
        )

        available = bool(
            record.available
            if record is not None
            else value is not None
        )

        selected = bool(
            record.selected
            if record is not None
            else value is not None
        )

        if not available:
            components = ConfidenceComponents(
                quality=0.0,
                freshness=0.0,
                synchronization=0.0,
                health=0.0,
                device=0.0,
                penalty=1.0,
            )

            return ModalityConfidence(
                modality=modality,
                available=False,
                selected=False,
                confidence_score=0.0,
                confidence_level=(
                    ConfidenceLevel.VERY_LOW
                ),
                components=components,
                limitation_codes=[
                    "modality_missing"
                ],
                reason_codes=[
                    "modality_unavailable"
                ],
            )

        limitations = (
            self._collect_limitations(
                value=value,
                record=record,
            )
        )

        quality = self._quality_score(
            modality,
            value,
        )

        freshness = self._freshness_score(
            record
        )

        synchronization = (
            self._synchronization_score(
                record,
                frame,
            )
        )

        health = self._health_score(
            modality,
            value,
        )

        modality_device_support = (
            self._modality_device_support(
                modality=modality,
                frame=frame,
                base_device_support=device_support,
            )
        )

        penalty = self._penalty_score(
            limitations
        )

        weighted = (
            self.component_weights["quality"]
            * quality
            + self.component_weights["freshness"]
            * freshness
            + self.component_weights["synchronization"]
            * synchronization
            + self.component_weights["health"]
            * health
            + self.component_weights["device"]
            * modality_device_support
        )

        confidence = clamp(
            weighted - penalty,
            0.0,
            1.0,
        )

        reason_codes = [
            "quality_evaluated",
            "freshness_evaluated",
            "synchronization_evaluated",
            "health_evaluated",
            "device_support_evaluated",
        ]

        if not selected:
            reason_codes.append(
                "modality_not_selected"
            )
            confidence *= 0.50

        if limitations:
            reason_codes.append(
                "limitation_penalty_applied"
            )

        confidence = clamp(
            confidence,
            0.0,
            1.0,
        )

        components = ConfidenceComponents(
            quality=round(quality, 6),
            freshness=round(freshness, 6),
            synchronization=round(
                synchronization,
                6,
            ),
            health=round(health, 6),
            device=round(
                modality_device_support,
                6,
            ),
            penalty=round(penalty, 6),
        )

        return ModalityConfidence(
            modality=modality,
            available=available,
            selected=selected,
            confidence_score=round(
                confidence,
                6,
            ),
            confidence_level=(
                confidence_level(confidence)
            ),
            components=components,
            limitation_codes=limitations,
            reason_codes=reason_codes,
            metadata={
                "value_type": (
                    type(value).__name__
                    if value is not None
                    else None
                ),
            },
        )

    # ========================================================
    # COMPONENT SCORES
    # ========================================================

    def _quality_score(
        self,
        modality: str,
        value: Any,
    ) -> float:
        if value is None:
            return 0.0

        if modality == "vision":
            return clamp(
                average(
                    [
                        getattr(
                            value,
                            "brightness_score",
                            None,
                        ),
                        getattr(
                            value,
                            "sharpness_score",
                            None,
                        ),
                        getattr(
                            value,
                            "contrast_score",
                            None,
                        ),
                    ],
                    default=0.50,
                ),
                0.0,
                1.0,
            )

        if modality == "audio":
            amplitude = safe_float(
                getattr(
                    value,
                    "amplitude_score",
                    0.50,
                ),
                0.50,
            )

            snr = safe_float(
                getattr(
                    value,
                    "signal_to_noise_score",
                    0.50,
                ),
                0.50,
            )

            clipping = safe_float(
                getattr(
                    value,
                    "clipping_ratio",
                    0.0,
                ),
                0.0,
            )

            silence = safe_float(
                getattr(
                    value,
                    "silence_ratio",
                    0.0,
                ),
                0.0,
            )

            return clamp(
                0.20 * amplitude
                + 0.50 * snr
                + 0.15 * (1.0 - clipping)
                + 0.15 * (1.0 - silence),
                0.0,
                1.0,
            )

        if modality == "spatial":
            accuracy = getattr(
                value,
                "horizontal_accuracy_meters",
                None,
            )

            if accuracy is None:
                return 0.50

            accuracy = max(
                0.0,
                safe_float(accuracy, 50.0),
            )

            return clamp(
                1.0 - accuracy / 100.0,
                0.0,
                1.0,
            )

        if modality == "motion":
            continuity = safe_float(
                getattr(
                    value,
                    "sampling_continuity_score",
                    0.50,
                ),
                0.50,
            )

            saturation = safe_float(
                getattr(
                    value,
                    "sensor_saturation_score",
                    0.0,
                ),
                0.0,
            )

            return clamp(
                0.80 * continuity
                + 0.20 * (1.0 - saturation),
                0.0,
                1.0,
            )

        if modality == "interaction":
            return 1.0

        if modality == "wearable":
            connected = bool(
                getattr(
                    value,
                    "connected",
                    False,
                )
            )

            return 1.0 if connected else 0.20

        if modality == "source_device":
            return self._calculate_device_support_from_value(
                value
            )

        if modality == "environment":
            return 0.75

        return 0.50

    def _health_score(
        self,
        modality: str,
        value: Any,
    ) -> float:
        if value is None:
            return 0.0

        if modality == "vision":
            return clamp(
                safe_float(
                    getattr(
                        value,
                        "frame_integrity_score",
                        1.0,
                    ),
                    1.0,
                ),
                0.0,
                1.0,
            )

        if modality == "audio":
            return clamp(
                safe_float(
                    getattr(
                        value,
                        "packet_integrity_score",
                        1.0,
                    ),
                    1.0,
                ),
                0.0,
                1.0,
            )

        if modality == "spatial":
            return 1.0

        if modality == "motion":
            saturation = safe_float(
                getattr(
                    value,
                    "sensor_saturation_score",
                    0.0,
                ),
                0.0,
            )

            return clamp(
                1.0 - saturation,
                0.0,
                1.0,
            )

        if modality == "interaction":
            return 1.0

        if modality == "wearable":
            connected = bool(
                getattr(
                    value,
                    "connected",
                    False,
                )
            )

            return 1.0 if connected else 0.0

        if modality == "source_device":
            return self._calculate_device_support_from_value(
                value
            )

        return 0.75

    def _freshness_score(
        self,
        record: Optional[
            ModalitySynchronizationRecord
        ],
    ) -> float:
        if record is None:
            return 0.50

        if not record.available:
            return 0.0

        if record.stale:
            return 0.0

        age_ms = record.source_age_ms

        if age_ms is None:
            return 1.0

        maximum_age = self._get_maximum_age_ms()

        return clamp(
            1.0 - age_ms / max(
                maximum_age,
                1.0,
            ),
            0.0,
            1.0,
        )

    def _synchronization_score(
        self,
        record: Optional[
            ModalitySynchronizationRecord
        ],
        frame: SynchronizedMultimodalFrame,
    ) -> float:
        if record is None:
            return frame.synchronization_score

        if not record.available:
            return 0.0

        if not record.within_window:
            return 0.0

        offset = record.offset_from_anchor_ms

        if offset is None:
            return 1.0

        window = self._get_synchronization_window_ms()

        return clamp(
            1.0 - offset / max(
                window,
                1.0,
            ),
            0.0,
            1.0,
        )

    def _calculate_device_support(
        self,
        frame: SynchronizedMultimodalFrame,
    ) -> float:
        source_device = frame.source_device

        if source_device is None:
            return 0.60

        return self._calculate_device_support_from_value(
            source_device
        )

    def _calculate_device_support_from_value(
        self,
        source_device: Any,
    ) -> float:
        battery = getattr(
            source_device,
            "battery_level",
            None,
        )

        network_strength = getattr(
            source_device,
            "network_strength",
            None,
        )

        network_latency = getattr(
            source_device,
            "network_latency_ms",
            None,
        )

        battery_score = (
            clamp(
                safe_float(battery, 0.75),
                0.0,
                1.0,
            )
            if battery is not None
            else 0.75
        )

        network_score = (
            clamp(
                safe_float(
                    network_strength,
                    0.75,
                ),
                0.0,
                1.0,
            )
            if network_strength is not None
            else 0.75
        )

        latency_score = (
            clamp(
                1.0
                - safe_float(
                    network_latency,
                    0.0,
                )
                / 1000.0,
                0.0,
                1.0,
            )
            if network_latency is not None
            else 0.75
        )

        return clamp(
            0.40 * battery_score
            + 0.35 * network_score
            + 0.25 * latency_score,
            0.0,
            1.0,
        )

    def _modality_device_support(
        self,
        *,
        modality: str,
        frame: SynchronizedMultimodalFrame,
        base_device_support: float,
    ) -> float:
        if modality == "wearable":
            wearable = frame.wearable

            if wearable is None:
                return 0.0

            return (
                1.0
                if bool(
                    getattr(
                        wearable,
                        "connected",
                        False,
                    )
                )
                else 0.0
            )

        if modality == "source_device":
            return base_device_support

        return base_device_support

    # ========================================================
    # PENALTIES
    # ========================================================

    def _collect_limitations(
        self,
        *,
        value: Any,
        record: Optional[
            ModalitySynchronizationRecord
        ],
    ) -> List[str]:
        limitations = get_metadata_limitations(
            value
        )

        if record is not None:
            limitations.extend(
                record.limitation_codes
            )

        return sorted(
            set(limitations)
        )

    def _penalty_score(
        self,
        limitations: Iterable[str],
    ) -> float:
        total = 0.0

        for limitation in limitations:
            total += DEFAULT_LIMITATION_PENALTIES.get(
                limitation,
                DEFAULT_UNKNOWN_LIMITATION_PENALTY,
            )

        return clamp(
            total,
            0.0,
            1.0,
        )

    # ========================================================
    # OVERALL CONFIDENCE
    # ========================================================

    def _calculate_overall(
        self,
        modality_results: Mapping[
            str,
            ModalityConfidence,
        ],
        *,
        frame: SynchronizedMultimodalFrame,
    ) -> float:
        weighted_sum = 0.0
        weight_sum = 0.0

        for modality, result in (
            modality_results.items()
        ):
            if not result.available:
                continue

            weight = self.modality_weights.get(
                modality,
                0.0,
            )

            if weight <= 0:
                continue

            weighted_sum += (
                weight
                * result.confidence_score
            )
            weight_sum += weight

        modality_confidence = (
            weighted_sum / weight_sum
            if weight_sum > 0
            else 0.0
        )

        status_factor = {
            SynchronizationStatus.COMPLETE: 1.0,
            SynchronizationStatus.PARTIAL: 0.85,
            SynchronizationStatus.DEGRADED: 0.65,
            SynchronizationStatus.EMPTY: 0.0,
        }[frame.status]

        overall = (
            0.75 * modality_confidence
            + 0.15 * frame.synchronization_score
            + 0.10 * frame.completeness_score
        )

        return clamp(
            overall * status_factor,
            0.0,
            1.0,
        )

    # ========================================================
    # SYNCHRONIZER SETTINGS COMPATIBILITY
    # ========================================================

    def _get_synchronization_window_ms(
        self,
    ) -> float:
        sync_settings = self.settings.synchronization

        for name in (
            "maximum_time_difference_ms",
            "synchronization_window_ms",
            "maximum_timestamp_difference_ms",
            "max_time_difference_ms",
            "max_time_skew_ms",
            "tolerance_ms",
        ):
            value = getattr(
                sync_settings,
                name,
                None,
            )

            if value is not None:
                parsed = safe_float(value, 500.0)

                if parsed > 0:
                    return parsed

        return 500.0

    def _get_maximum_age_ms(
        self,
    ) -> float:
        sync_settings = self.settings.synchronization

        for name in (
            "maximum_modality_age_ms",
            "max_modality_age_ms",
            "maximum_sample_age_ms",
            "stale_after_ms",
            "freshness_timeout_ms",
        ):
            value = getattr(
                sync_settings,
                name,
                None,
            )

            if value is not None:
                parsed = safe_float(value, 5000.0)

                if parsed > 0:
                    return parsed

        return 5000.0

    # ========================================================
    # STATISTICS AND DIAGNOSTICS
    # ========================================================

    def _register_report(
        self,
        report: ConfidenceReport,
        elapsed_seconds: float,
    ) -> None:
        self.statistics.total_reports += 1
        self.statistics.cumulative_processing_seconds += (
            elapsed_seconds
        )

        self.statistics.last_report_id = (
            report.report_id
        )
        self.statistics.last_frame_id = (
            report.frame_id
        )
        self.statistics.last_overall_confidence = (
            report.overall_confidence
        )
        self.statistics.last_error = None

        if report.overall_confidence >= 0.75:
            self.statistics.high_confidence_reports += 1
        elif report.overall_confidence >= 0.55:
            self.statistics.medium_confidence_reports += 1
        else:
            self.statistics.low_confidence_reports += 1

        self.statistics.total_unavailable_modalities += len(
            report.unavailable_modalities
        )
        self.statistics.total_uncertain_modalities += len(
            report.uncertain_modalities
        )

        self._last_report = report

    def get_last_report(
        self,
    ) -> Optional[ConfidenceReport]:
        return self._last_report

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "last_report_available": (
                self._last_report is not None
            ),
            "last_report_id": (
                self._last_report.report_id
                if self._last_report
                else None
            ),
            "last_overall_confidence": (
                self._last_report
                .overall_confidence
                if self._last_report
                else None
            ),
            "trusted_threshold": (
                self.trusted_threshold
            ),
            "uncertain_threshold": (
                self.uncertain_threshold
            ),
            "statistics": (
                self.statistics.to_dict()
            ),
        }


# ============================================================
# SELF-TEST
# ============================================================

def run_confidence_estimator_self_test() -> bool:
    print("\n" + "=" * 72)
    print("NOONGIL-X | CONFIDENCE ESTIMATOR TEST")
    print("=" * 72)

    try:
        print("[1/7] Creating test settings...")

        settings = create_test_settings()
        estimator = ConfidenceEstimator(settings)

        print("[SUCCESS] Confidence estimator initialized.")

        print("[2/7] Building synchronized frame...")

        from layer1.acquisition.multimodal_receiver import (
            MultimodalReceiver,
        )
        from layer1.acquisition.phone_sensor_simulator import (
            PhoneSensorSimulator,
            PhoneSimulatorConfig,
            SimulationScenario,
        )
        from layer1.modalities.audio_input import (
            AudioInputProcessor,
        )
        from layer1.modalities.device_input import (
            DeviceInputProcessor,
        )
        from layer1.modalities.interaction_input import (
            InteractionInputProcessor,
        )
        from layer1.modalities.motion_input import (
            MotionInputProcessor,
        )
        from layer1.modalities.spatial_input import (
            SpatialInputProcessor,
        )
        from layer1.modalities.vision_input import (
            VisionInputProcessor,
        )
        from layer1.processing.multimodal_synchronizer import (
            MultimodalSynchronizer,
        )

        receiver = MultimodalReceiver(settings)
        receiver.start()

        simulator = PhoneSensorSimulator(
            PhoneSimulatorConfig(
                scenario=(
                    SimulationScenario.NAVIGATION
                ),
                random_seed=42,
            )
        )

        receipts = receiver.receive_batch(
            simulator.generate_cycle(),
            raise_on_error=True,
        )

        if not all(
            receipt.accepted
            for receipt in receipts
        ):
            raise AssertionError(
                "Simulator packets were not accepted."
            )

        vision_result = (
            VisionInputProcessor(settings)
            .process_latest_from_receiver(
                receiver,
                raise_on_error=True,
            )
        )

        audio_result = (
            AudioInputProcessor(settings)
            .process_latest_from_receiver(
                receiver,
                raise_on_error=True,
            )
        )

        spatial_result = (
            SpatialInputProcessor(settings)
            .process_latest_from_receiver(
                receiver,
                raise_on_error=True,
            )
        )

        motion_result = (
            MotionInputProcessor(settings)
            .process_receiver_queue(
                receiver,
                raise_on_error=True,
            )
        )

        interaction_result = (
            InteractionInputProcessor(settings)
            .process_latest_from_receiver(
                receiver,
                raise_on_error=True,
            )
        )

        device_results = (
            DeviceInputProcessor(settings)
            .process_receiver_queue(
                receiver,
                raise_on_error=True,
            )
        )

        synchronizer = MultimodalSynchronizer(
            settings
        )

        frame = (
            synchronizer
            .synchronize_from_results(
                vision_result=vision_result,
                audio_result=audio_result,
                spatial_result=spatial_result,
                motion_result=motion_result,
                interaction_result=(
                    interaction_result
                ),
                device_results=device_results,
                include_cached_values=True,
                raise_on_error=True,
            )
        )

        print("[SUCCESS] Synchronized frame created.")

        print("[3/7] Estimating confidence...")

        report = estimator.estimate(
            frame,
            raise_on_error=True,
        )

        print("[SUCCESS] Confidence report created.")

        print("[4/7] Validating report...")

        report.validate()

        if report.frame_id != frame.frame_id:
            raise AssertionError(
                "Confidence report frame ID mismatch."
            )

        if not 0.0 <= (
            report.overall_confidence
        ) <= 1.0:
            raise AssertionError(
                "Overall confidence is out of range."
            )

        if (
            report.modality_confidences[
                "vision"
            ].confidence_score
            <= 0.0
        ):
            raise AssertionError(
                "Vision confidence was not calculated."
            )

        if (
            report.modality_confidences[
                "audio"
            ].confidence_score
            <= 0.0
        ):
            raise AssertionError(
                "Audio confidence was not calculated."
            )

        print("[SUCCESS] Confidence report is valid.")

        print("[5/7] Testing unavailable modality...")

        environment_result = (
            report.modality_confidences[
                "environment"
            ]
        )

        if environment_result.available:
            raise AssertionError(
                "Environment should be unavailable "
                "in this test."
            )

        if (
            environment_result.confidence_score
            != 0.0
        ):
            raise AssertionError(
                "Unavailable modality confidence "
                "must be zero."
            )

        print(
            "[SUCCESS] Unavailable modality handling works."
        )

        print("[6/7] Checking trusted modalities...")

        if not report.trusted_modalities:
            raise AssertionError(
                "Expected at least one trusted modality."
            )

        print("[SUCCESS] Trusted modalities identified.")

        print("[7/7] Checking diagnostics...")

        health = estimator.health_check()

        if not health["healthy"]:
            raise AssertionError(
                "Estimator health check failed."
            )

        if (
            health["statistics"]["total_reports"]
            != 1
        ):
            raise AssertionError(
                "Report count is incorrect."
            )

        print("[SUCCESS] Diagnostics are correct.")

        print("\nConfidence summary:")
        print(
            json.dumps(
                {
                    "report_id": report.report_id,
                    "frame_id": report.frame_id,
                    "overall_confidence": (
                        report.overall_confidence
                    ),
                    "overall_level": (
                        report.overall_level.value
                    ),
                    "synchronization_confidence": (
                        report
                        .synchronization_confidence
                    ),
                    "completeness_confidence": (
                        report
                        .completeness_confidence
                    ),
                    "device_support_confidence": (
                        report
                        .device_support_confidence
                    ),
                    "trusted_modalities": (
                        report.trusted_modalities
                    ),
                    "uncertain_modalities": (
                        report.uncertain_modalities
                    ),
                    "unavailable_modalities": (
                        report.unavailable_modalities
                    ),
                    "modality_scores": {
                        modality: result.confidence_score
                        for modality, result
                        in report
                        .modality_confidences.items()
                    },
                    "warnings": report.warnings,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nEstimator health:")
        print(
            json.dumps(
                health,
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n" + "=" * 72)
        print(
            "[PASSED] CONFIDENCE ESTIMATOR IS WORKING"
        )
        print("=" * 72)

        return True

    except Exception as error:
        print("\n" + "=" * 72)
        print(
            "[FAILED] CONFIDENCE ESTIMATOR TEST"
        )
        print("=" * 72)
        print(
            f"[ERROR] {type(error).__name__}: {error}"
        )

        return False


if __name__ == "__main__":
    if not run_confidence_estimator_self_test():
        raise SystemExit(1)

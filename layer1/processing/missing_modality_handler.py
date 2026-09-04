"""
============================================================
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : Missing Modality Handler
File    : layer1/processing/missing_modality_handler.py
============================================================

Purpose
-------
Handle missing, stale, excluded, and low-confidence modalities
after synchronization and confidence estimation.

Safety Principle
----------------
This module never fabricates sensor observations.

A missing modality may only be handled by:
1. Reusing a previously observed cached value when it is still
   within the configured recovery age;
2. Marking the modality unavailable;
3. Continuing with a partial frame;
4. Requesting reacquisition;
5. Escalating to a degraded or blocked state when a required
   modality is unavailable.

Responsibilities
----------------
1. Inspect synchronized-frame modality availability
2. Inspect per-modality confidence
3. Reuse valid cached observations
4. Mark recovered observations transparently
5. Generate reacquisition requests
6. Determine whether Layer 1 can continue safely
7. Produce a structured recovery report
8. Preserve all original modality values and provenance
9. Provide diagnostics and a standalone self-test

Architectural Boundary
----------------------
This module does NOT:
- synthesize camera frames;
- synthesize audio;
- invent GPS coordinates;
- infer semantic content;
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

import copy
import json
import math
import time
import uuid

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from layer1.config.settings import (
    Layer1Settings,
    create_default_settings,
    create_test_settings,
)
from layer1.processing.confidence_estimator import (
    ConfidenceReport,
)
from layer1.processing.multimodal_synchronizer import (
    ModalitySynchronizationRecord,
    SynchronizedMultimodalFrame,
    SynchronizationStatus,
    parse_iso_timestamp,
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

HANDLER_VERSION = "1.0"

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

DEFAULT_REQUIRED_MODALITIES = {
    "vision",
    "audio",
    "spatial",
    "motion",
}

DEFAULT_CACHE_AGE_LIMITS_MS: Dict[str, float] = {
    "vision": 1500.0,
    "audio": 2500.0,
    "spatial": 10000.0,
    "motion": 1000.0,
    "interaction": 5000.0,
    "wearable": 30000.0,
    "source_device": 30000.0,
    "environment": 60000.0,
}

DEFAULT_REACQUISITION_PRIORITIES: Dict[str, int] = {
    "vision": 90,
    "audio": 75,
    "spatial": 85,
    "motion": 88,
    "interaction": 95,
    "wearable": 55,
    "source_device": 60,
    "environment": 35,
}


# ============================================================
# EXCEPTIONS
# ============================================================

class MissingModalityError(Exception):
    """Base exception for missing-modality handling."""


class MissingModalityValidationError(MissingModalityError):
    """Raised when input data is invalid."""


class MissingModalityProcessingError(MissingModalityError):
    """Raised when recovery handling fails."""


# ============================================================
# ENUMERATIONS
# ============================================================

class RecoveryStrategy(str, Enum):
    KEEP_CURRENT = "keep_current"
    REUSE_CACHED = "reuse_cached"
    MARK_UNAVAILABLE = "mark_unavailable"
    REQUEST_REACQUISITION = "request_reacquisition"
    CONTINUE_PARTIAL = "continue_partial"
    BLOCK_PIPELINE = "block_pipeline"


class RecoveryStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class ModalityAvailability(str, Enum):
    OBSERVED = "observed"
    RECOVERED_CACHED = "recovered_cached"
    UNAVAILABLE = "unavailable"
    LOW_CONFIDENCE = "low_confidence"
    STALE = "stale"
    EXCLUDED = "excluded"


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


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if is_dataclass(value):
        return make_json_safe(asdict(value))

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

    return value


def extract_source_timestamp(value: Any) -> Optional[str]:
    metadata = getattr(value, "metadata", None)

    if metadata is None:
        return None

    source_timestamp = getattr(
        metadata,
        "source_timestamp",
        None,
    )

    if source_timestamp is None:
        return None

    parse_iso_timestamp(source_timestamp)

    return source_timestamp


def calculate_age_ms(
    source_timestamp: Optional[str],
) -> Optional[float]:
    if source_timestamp is None:
        return None

    source = parse_iso_timestamp(source_timestamp)
    now = datetime.now(timezone.utc)

    return max(
        0.0,
        (
            now - source
        ).total_seconds() * 1000.0,
    )


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ReacquisitionRequest:
    """
    Structured request to reacquire one modality.
    """

    request_id: str
    modality: str
    priority: int
    reason_codes: List[str]
    requested_at: str
    target_deadline_ms: float
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        if self.modality not in SUPPORTED_MODALITIES:
            raise MissingModalityValidationError(
                f"Unsupported modality: {self.modality!r}"
            )

        if not 0 <= self.priority <= 100:
            raise MissingModalityValidationError(
                "priority must be between 0 and 100."
            )

        if (
            isinstance(self.target_deadline_ms, bool)
            or not isinstance(
                self.target_deadline_ms,
                (int, float),
            )
            or not math.isfinite(
                float(self.target_deadline_ms)
            )
            or self.target_deadline_ms <= 0
        ):
            raise MissingModalityValidationError(
                "target_deadline_ms must be positive."
            )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return make_json_safe(asdict(self))


@dataclass
class ModalityRecoveryDecision:
    """
    Recovery decision for one modality.
    """

    modality: str
    required: bool

    original_availability: ModalityAvailability
    final_availability: ModalityAvailability

    strategy: RecoveryStrategy

    original_confidence: float
    final_confidence: float

    cache_age_ms: Optional[float] = None
    cache_age_limit_ms: Optional[float] = None

    recovered: bool = False
    usable: bool = False

    reason_codes: List[str] = field(
        default_factory=list
    )
    warnings: List[str] = field(
        default_factory=list
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        if self.modality not in SUPPORTED_MODALITIES:
            raise MissingModalityValidationError(
                f"Unsupported modality: {self.modality!r}"
            )

        for name, value in (
            ("original_confidence", self.original_confidence),
            ("final_confidence", self.final_confidence),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise MissingModalityValidationError(
                    f"{name} must be between 0 and 1."
                )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["original_availability"] = (
            self.original_availability.value
        )
        payload["final_availability"] = (
            self.final_availability.value
        )
        payload["strategy"] = self.strategy.value
        return make_json_safe(payload)


@dataclass
class MissingModalityRecoveryReport:
    """
    Complete missing-modality recovery result.
    """

    report_id: str
    generated_at: str
    original_frame_id: str
    recovered_frame_id: str

    status: RecoveryStatus
    safe_to_continue: bool

    decisions: Dict[
        str,
        ModalityRecoveryDecision,
    ]

    recovered_modalities: List[str]
    unavailable_modalities: List[str]
    low_confidence_modalities: List[str]
    required_unavailable_modalities: List[str]

    reacquisition_requests: List[
        ReacquisitionRequest
    ]

    original_overall_confidence: float
    adjusted_overall_confidence: float

    warnings: List[str] = field(
        default_factory=list
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        if not self.report_id.strip():
            raise MissingModalityValidationError(
                "report_id cannot be empty."
            )

        if not self.original_frame_id.strip():
            raise MissingModalityValidationError(
                "original_frame_id cannot be empty."
            )

        if not self.recovered_frame_id.strip():
            raise MissingModalityValidationError(
                "recovered_frame_id cannot be empty."
            )

        for name, value in (
            (
                "original_overall_confidence",
                self.original_overall_confidence,
            ),
            (
                "adjusted_overall_confidence",
                self.adjusted_overall_confidence,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise MissingModalityValidationError(
                    f"{name} must be between 0 and 1."
                )

        for decision in self.decisions.values():
            decision.validate()

        for request in self.reacquisition_requests:
            request.validate()

    def to_dict(self) -> Dict[str, Any]:
        self.validate()

        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "original_frame_id": self.original_frame_id,
            "recovered_frame_id": self.recovered_frame_id,
            "status": self.status.value,
            "safe_to_continue": self.safe_to_continue,
            "decisions": {
                modality: decision.to_dict()
                for modality, decision
                in self.decisions.items()
            },
            "recovered_modalities": (
                self.recovered_modalities
            ),
            "unavailable_modalities": (
                self.unavailable_modalities
            ),
            "low_confidence_modalities": (
                self.low_confidence_modalities
            ),
            "required_unavailable_modalities": (
                self.required_unavailable_modalities
            ),
            "reacquisition_requests": [
                request.to_dict()
                for request in self.reacquisition_requests
            ],
            "original_overall_confidence": (
                self.original_overall_confidence
            ),
            "adjusted_overall_confidence": (
                self.adjusted_overall_confidence
            ),
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


@dataclass
class RecoveryResult:
    """
    Returned by MissingModalityHandler.handle().
    """

    frame: SynchronizedMultimodalFrame
    report: MissingModalityRecoveryReport


@dataclass
class MissingModalityStatistics:
    """
    Runtime statistics.
    """

    total_runs: int = 0
    complete_runs: int = 0
    partial_runs: int = 0
    degraded_runs: int = 0
    blocked_runs: int = 0

    total_cached_recoveries: int = 0
    total_reacquisition_requests: int = 0
    total_required_failures: int = 0

    cumulative_processing_seconds: float = 0.0

    last_report_id: Optional[str] = None
    last_status: Optional[str] = None
    last_error: Optional[str] = None

    @property
    def average_processing_seconds(self) -> float:
        if self.total_runs == 0:
            return 0.0

        return (
            self.cumulative_processing_seconds
            / self.total_runs
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["average_processing_seconds"] = (
            self.average_processing_seconds
        )
        return payload


# ============================================================
# MISSING MODALITY HANDLER
# ============================================================

class MissingModalityHandler:
    """
    Safely recover or explicitly mark missing modalities.
    """

    def __init__(
        self,
        settings: Optional[Layer1Settings] = None,
    ) -> None:
        self.settings = settings or create_default_settings()
        self.settings.validate()

        self.logger = get_logger(
            "processing.missing_modality_handler"
        )

        self.statistics = MissingModalityStatistics()

        self._cache: Dict[str, Any] = {}
        self._last_report: Optional[
            MissingModalityRecoveryReport
        ] = None

    # ========================================================
    # SETTINGS COMPATIBILITY
    # ========================================================

    def _recovery_settings(self) -> Any:
        return getattr(
            self.settings,
            "recovery",
            None,
        )

    def _get_required_modalities(self) -> set[str]:
        recovery_settings = self._recovery_settings()
        sync_settings = getattr(
            self.settings,
            "synchronization",
            None,
        )

        for owner in (
            recovery_settings,
            sync_settings,
        ):
            if owner is None:
                continue

            for name in (
                "required_modalities",
                "mandatory_modalities",
                "core_modalities",
            ):
                value = getattr(owner, name, None)

                if value is None:
                    continue

                if not isinstance(
                    value,
                    (list, tuple, set),
                ):
                    raise MissingModalityValidationError(
                        f"{name} must be a list, tuple, or set."
                    )

                normalized = {
                    str(item).strip().lower()
                    for item in value
                    if str(item).strip()
                }

                unsupported = (
                    normalized - SUPPORTED_MODALITIES
                )

                if unsupported:
                    raise MissingModalityValidationError(
                        "Unsupported required modalities: "
                        f"{sorted(unsupported)}"
                    )

                if normalized:
                    return normalized

        return set(DEFAULT_REQUIRED_MODALITIES)

    def _get_cache_age_limit_ms(
        self,
        modality: str,
    ) -> float:
        recovery_settings = self._recovery_settings()

        if recovery_settings is not None:
            mapping = getattr(
                recovery_settings,
                "maximum_cache_age_ms",
                None,
            )

            if isinstance(mapping, Mapping):
                value = mapping.get(modality)

                if value is not None:
                    parsed = float(value)

                    if (
                        math.isfinite(parsed)
                        and parsed > 0
                    ):
                        return parsed

            for name in (
                f"{modality}_maximum_cache_age_ms",
                f"{modality}_cache_age_ms",
            ):
                value = getattr(
                    recovery_settings,
                    name,
                    None,
                )

                if value is not None:
                    parsed = float(value)

                    if (
                        math.isfinite(parsed)
                        and parsed > 0
                    ):
                        return parsed

            generic = getattr(
                recovery_settings,
                "default_maximum_cache_age_ms",
                None,
            )

            if generic is not None:
                parsed = float(generic)

                if (
                    math.isfinite(parsed)
                    and parsed > 0
                ):
                    return parsed

        return DEFAULT_CACHE_AGE_LIMITS_MS[
            modality
        ]

    def _get_low_confidence_threshold(self) -> float:
        recovery_settings = self._recovery_settings()
        confidence_settings = getattr(
            self.settings,
            "confidence",
            None,
        )

        for owner in (
            recovery_settings,
            confidence_settings,
        ):
            if owner is None:
                continue

            for name in (
                "low_confidence_threshold",
                "minimum_usable_confidence",
                "uncertain_threshold",
            ):
                value = getattr(owner, name, None)

                if value is not None:
                    parsed = float(value)

                    if (
                        math.isfinite(parsed)
                        and 0.0 <= parsed <= 1.0
                    ):
                        return parsed

        return 0.40

    def _get_reuse_confidence_multiplier(self) -> float:
        recovery_settings = self._recovery_settings()

        if recovery_settings is not None:
            for name in (
                "cached_reuse_confidence_multiplier",
                "cache_confidence_multiplier",
                "reuse_penalty_multiplier",
            ):
                value = getattr(
                    recovery_settings,
                    name,
                    None,
                )

                if value is not None:
                    parsed = float(value)

                    if (
                        math.isfinite(parsed)
                        and 0.0 <= parsed <= 1.0
                    ):
                        return parsed

        return 0.75

    def _allow_cached_reuse(self) -> bool:
        recovery_settings = self._recovery_settings()

        if recovery_settings is None:
            return True

        for name in (
            "allow_cached_reuse",
            "reuse_cached_modalities",
            "enable_cache_recovery",
        ):
            value = getattr(
                recovery_settings,
                name,
                None,
            )

            if value is not None:
                return bool(value)

        return True

    def _block_on_required_missing(self) -> bool:
        recovery_settings = self._recovery_settings()

        if recovery_settings is None:
            return False

        for name in (
            "block_on_required_missing",
            "strict_required_modalities",
            "fail_when_required_missing",
        ):
            value = getattr(
                recovery_settings,
                name,
                None,
            )

            if value is not None:
                return bool(value)

        return False

    # ========================================================
    # PUBLIC API
    # ========================================================

    def update_cache(
        self,
        frame: SynchronizedMultimodalFrame,
    ) -> None:
        """
        Store observed selected modalities for future recovery.
        """

        frame.validate()

        for modality in SUPPORTED_MODALITIES:
            value = getattr(frame, modality, None)
            record = frame.synchronization_records.get(
                modality
            )

            selected = bool(
                record.selected
                if record is not None
                else value is not None
            )

            if value is not None and selected:
                self._cache[modality] = copy.deepcopy(
                    value
                )

    def handle(
        self,
        frame: SynchronizedMultimodalFrame,
        confidence_report: ConfidenceReport,
        *,
        raise_on_error: Optional[bool] = None,
    ) -> RecoveryResult:
        """
        Handle missing, stale, excluded, and low-confidence
        modalities.

        Returns a copied frame. The original frame is not mutated.
        """

        should_raise = (
            self.settings.runtime.fail_fast
            if raise_on_error is None
            else raise_on_error
        )

        started = time.perf_counter()

        try:
            with PipelineTimer(
                "missing_modality_handler.handle",
                logger=self.logger,
                metadata={
                    "frame_id": frame.frame_id,
                    "confidence_report_id": (
                        confidence_report.report_id
                    ),
                },
            ):
                frame.validate()
                confidence_report.validate()

                if (
                    confidence_report.frame_id
                    != frame.frame_id
                ):
                    raise MissingModalityValidationError(
                        "Confidence report does not match "
                        "the synchronized frame."
                    )

                recovered_frame = copy.deepcopy(frame)

                original_frame_id = frame.frame_id
                recovered_frame.frame_id = (
                    "RECOVERED_"
                    f"{uuid.uuid4().hex[:10].upper()}"
                )

                required_modalities = (
                    self._get_required_modalities()
                )
                low_threshold = (
                    self._get_low_confidence_threshold()
                )

                decisions: Dict[
                    str,
                    ModalityRecoveryDecision,
                ] = {}

                reacquisition_requests: List[
                    ReacquisitionRequest
                ] = []

                recovered_modalities: List[str] = []
                unavailable_modalities: List[str] = []
                low_confidence_modalities: List[str] = []

                for modality in sorted(
                    SUPPORTED_MODALITIES
                ):
                    confidence_result = (
                        confidence_report
                        .modality_confidences[
                            modality
                        ]
                    )

                    decision, request = (
                        self._handle_modality(
                            modality=modality,
                            frame=recovered_frame,
                            confidence_result=(
                                confidence_result
                            ),
                            required=(
                                modality
                                in required_modalities
                            ),
                            low_threshold=low_threshold,
                        )
                    )

                    decisions[modality] = decision

                    if request is not None:
                        reacquisition_requests.append(
                            request
                        )

                    if decision.recovered:
                        recovered_modalities.append(
                            modality
                        )

                    if (
                        decision.final_availability
                        == ModalityAvailability.UNAVAILABLE
                    ):
                        unavailable_modalities.append(
                            modality
                        )

                    if (
                        decision.final_availability
                        == ModalityAvailability.LOW_CONFIDENCE
                    ):
                        low_confidence_modalities.append(
                            modality
                        )

                self._rebuild_frame_lists(
                    recovered_frame,
                    decisions,
                )

                required_unavailable = sorted(
                    modality
                    for modality in required_modalities
                    if not decisions[modality].usable
                )

                status, safe_to_continue = (
                    self._determine_recovery_status(
                        required_unavailable=(
                            required_unavailable
                        ),
                        unavailable_modalities=(
                            unavailable_modalities
                        ),
                        recovered_modalities=(
                            recovered_modalities
                        ),
                    )
                )

                adjusted_confidence = (
                    self._calculate_adjusted_confidence(
                        confidence_report,
                        decisions,
                    )
                )

                warnings: List[str] = []

                if recovered_modalities:
                    warnings.append(
                        "cached_modalities_reused"
                    )

                if unavailable_modalities:
                    warnings.append(
                        "unavailable_modalities_remain"
                    )

                if required_unavailable:
                    warnings.append(
                        "required_modalities_unavailable"
                    )

                if reacquisition_requests:
                    warnings.append(
                        "reacquisition_requested"
                    )

                recovered_frame.status = (
                    self._map_recovery_to_sync_status(
                        status
                    )
                )

                recovered_frame.warnings = sorted(
                    set(
                        list(recovered_frame.warnings)
                        + warnings
                    )
                )

                recovered_frame.metadata = dict(
                    recovered_frame.metadata
                )
                recovered_frame.metadata.update(
                    {
                        "recovery_applied": True,
                        "recovery_handler_version": (
                            HANDLER_VERSION
                        ),
                        "original_frame_id": (
                            original_frame_id
                        ),
                        "recovered_modalities": (
                            sorted(recovered_modalities)
                        ),
                        "required_unavailable_modalities": (
                            required_unavailable
                        ),
                    }
                )

                report = MissingModalityRecoveryReport(
                    report_id=(
                        "RECOVERY_"
                        f"{uuid.uuid4().hex[:12].upper()}"
                    ),
                    generated_at=utc_now_iso(),
                    original_frame_id=original_frame_id,
                    recovered_frame_id=(
                        recovered_frame.frame_id
                    ),
                    status=status,
                    safe_to_continue=safe_to_continue,
                    decisions=decisions,
                    recovered_modalities=sorted(
                        recovered_modalities
                    ),
                    unavailable_modalities=sorted(
                        unavailable_modalities
                    ),
                    low_confidence_modalities=sorted(
                        low_confidence_modalities
                    ),
                    required_unavailable_modalities=(
                        required_unavailable
                    ),
                    reacquisition_requests=(
                        reacquisition_requests
                    ),
                    original_overall_confidence=(
                        confidence_report
                        .overall_confidence
                    ),
                    adjusted_overall_confidence=round(
                        adjusted_confidence,
                        6,
                    ),
                    warnings=warnings,
                    metadata={
                        "handler_version": (
                            HANDLER_VERSION
                        ),
                        "required_modalities": sorted(
                            required_modalities
                        ),
                        "low_confidence_threshold": (
                            low_threshold
                        ),
                        "cached_reuse_enabled": (
                            self._allow_cached_reuse()
                        ),
                    },
                )

                report.validate()
                recovered_frame.validate()

                elapsed = time.perf_counter() - started

                self._register_report(
                    report,
                    elapsed,
                )

                self.update_cache(
                    recovered_frame
                )

                log_sensor_event(
                    modality="interaction",
                    event="Missing modality recovery completed",
                    sensor_type=(
                        "missing_modality_handler"
                    ),
                    packet_id=report.report_id,
                    details={
                        "original_frame_id": (
                            original_frame_id
                        ),
                        "recovered_frame_id": (
                            recovered_frame.frame_id
                        ),
                        "status": report.status.value,
                        "safe_to_continue": (
                            report.safe_to_continue
                        ),
                        "recovered_modalities": (
                            report.recovered_modalities
                        ),
                        "unavailable_modalities": (
                            report.unavailable_modalities
                        ),
                        "required_unavailable_modalities": (
                            report
                            .required_unavailable_modalities
                        ),
                        "reacquisition_request_count": (
                            len(
                                report
                                .reacquisition_requests
                            )
                        ),
                    },
                )

                return RecoveryResult(
                    frame=recovered_frame,
                    report=report,
                )

        except Exception as error:
            self.statistics.last_error = (
                f"{type(error).__name__}: {error}"
            )

            log_exception(
                self.logger,
                "Missing modality handling failed",
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

            raise MissingModalityProcessingError(
                f"Missing modality handling failed: {error}"
            ) from error

    # ========================================================
    # MODALITY HANDLING
    # ========================================================

    def _handle_modality(
        self,
        *,
        modality: str,
        frame: SynchronizedMultimodalFrame,
        confidence_result: Any,
        required: bool,
        low_threshold: float,
    ) -> Tuple[
        ModalityRecoveryDecision,
        Optional[ReacquisitionRequest],
    ]:
        value = getattr(frame, modality, None)
        record = frame.synchronization_records.get(
            modality
        )

        original_confidence = clamp(
            float(
                confidence_result
                .confidence_score
            ),
            0.0,
            1.0,
        )

        original_availability = (
            self._classify_original_availability(
                value=value,
                record=record,
                confidence=original_confidence,
                low_threshold=low_threshold,
            )
        )

        if (
            original_availability
            == ModalityAvailability.OBSERVED
        ):
            return (
                ModalityRecoveryDecision(
                    modality=modality,
                    required=required,
                    original_availability=(
                        original_availability
                    ),
                    final_availability=(
                        ModalityAvailability.OBSERVED
                    ),
                    strategy=(
                        RecoveryStrategy.KEEP_CURRENT
                    ),
                    original_confidence=(
                        original_confidence
                    ),
                    final_confidence=(
                        original_confidence
                    ),
                    recovered=False,
                    usable=True,
                    reason_codes=[
                        "current_modality_valid"
                    ],
                ),
                None,
            )

        cache_value = self._cache.get(modality)
        cache_age = (
            calculate_age_ms(
                extract_source_timestamp(
                    cache_value
                )
            )
            if cache_value is not None
            else None
        )
        cache_limit = (
            self._get_cache_age_limit_ms(
                modality
            )
        )

        cache_usable = bool(
            self._allow_cached_reuse()
            and cache_value is not None
            and cache_age is not None
            and cache_age <= cache_limit
        )

        if cache_usable:
            copied_value = copy.deepcopy(
                cache_value
            )

            self._mark_cached_recovery(
                value=copied_value,
                original_frame_id=frame.frame_id,
                cache_age_ms=cache_age,
            )

            setattr(
                frame,
                modality,
                copied_value,
            )

            record = (
                self._create_recovered_record(
                    modality=modality,
                    value=copied_value,
                    cache_age_ms=cache_age,
                )
            )
            frame.synchronization_records[
                modality
            ] = record

            final_confidence = clamp(
                max(
                    original_confidence,
                    low_threshold,
                )
                * self._get_reuse_confidence_multiplier(),
                0.0,
                1.0,
            )

            request = self._create_reacquisition_request(
                modality=modality,
                reason_codes=[
                    "cached_value_reused",
                    "fresh_observation_requested",
                ],
                required=required,
            )

            return (
                ModalityRecoveryDecision(
                    modality=modality,
                    required=required,
                    original_availability=(
                        original_availability
                    ),
                    final_availability=(
                        ModalityAvailability
                        .RECOVERED_CACHED
                    ),
                    strategy=(
                        RecoveryStrategy.REUSE_CACHED
                    ),
                    original_confidence=(
                        original_confidence
                    ),
                    final_confidence=round(
                        final_confidence,
                        6,
                    ),
                    cache_age_ms=round(
                        cache_age,
                        6,
                    ),
                    cache_age_limit_ms=(
                        cache_limit
                    ),
                    recovered=True,
                    usable=True,
                    reason_codes=[
                        "valid_cached_sample_available",
                        "cached_sample_reused",
                        "confidence_penalty_applied",
                    ],
                    warnings=[
                        "recovered_value_is_not_current"
                    ],
                ),
                request,
            )

        setattr(
            frame,
            modality,
            None,
        )

        frame.synchronization_records[
            modality
        ] = ModalitySynchronizationRecord(
            modality=modality,
            available=False,
            within_window=False,
            stale=False,
            selected=False,
            limitation_codes=[
                "modality_unavailable_after_recovery"
            ],
            metadata={
                "recovery_attempted": True,
                "cache_available": (
                    cache_value is not None
                ),
                "cache_age_ms": cache_age,
                "cache_age_limit_ms": (
                    cache_limit
                ),
            },
        )

        request = self._create_reacquisition_request(
            modality=modality,
            reason_codes=[
                "modality_unavailable",
                "valid_cache_unavailable",
            ],
            required=required,
        )

        strategy = (
            RecoveryStrategy.BLOCK_PIPELINE
            if (
                required
                and self._block_on_required_missing()
            )
            else RecoveryStrategy.MARK_UNAVAILABLE
        )

        return (
            ModalityRecoveryDecision(
                modality=modality,
                required=required,
                original_availability=(
                    original_availability
                ),
                final_availability=(
                    ModalityAvailability.UNAVAILABLE
                ),
                strategy=strategy,
                original_confidence=(
                    original_confidence
                ),
                final_confidence=0.0,
                cache_age_ms=(
                    round(cache_age, 6)
                    if cache_age is not None
                    else None
                ),
                cache_age_limit_ms=cache_limit,
                recovered=False,
                usable=False,
                reason_codes=[
                    "no_valid_current_observation",
                    "no_valid_cached_observation",
                    "modality_marked_unavailable",
                ],
                warnings=[
                    "downstream_processing_must_handle_missing"
                ],
            ),
            request,
        )

    def _classify_original_availability(
        self,
        *,
        value: Any,
        record: Optional[
            ModalitySynchronizationRecord
        ],
        confidence: float,
        low_threshold: float,
    ) -> ModalityAvailability:
        if value is None:
            return ModalityAvailability.UNAVAILABLE

        if record is not None:
            if record.stale:
                return ModalityAvailability.STALE

            if record.available and not record.selected:
                return ModalityAvailability.EXCLUDED

            if not record.available:
                return ModalityAvailability.UNAVAILABLE

        if confidence < low_threshold:
            return (
                ModalityAvailability.LOW_CONFIDENCE
            )

        return ModalityAvailability.OBSERVED

    def _mark_cached_recovery(
        self,
        *,
        value: Any,
        original_frame_id: str,
        cache_age_ms: float,
    ) -> None:
        metadata = getattr(value, "metadata", None)

        if metadata is None:
            return

        limitations = getattr(
            metadata,
            "limitations",
            None,
        )

        if isinstance(limitations, list):
            if (
                "recovered_from_cached_observation"
                not in limitations
            ):
                limitations.append(
                    "recovered_from_cached_observation"
                )

        nested_metadata = getattr(
            metadata,
            "metadata",
            None,
        )

        if isinstance(nested_metadata, dict):
            nested_metadata.update(
                {
                    "recovered_from_cache": True,
                    "recovery_target_frame_id": (
                        original_frame_id
                    ),
                    "cache_age_ms": round(
                        cache_age_ms,
                        6,
                    ),
                }
            )

    def _create_recovered_record(
        self,
        *,
        modality: str,
        value: Any,
        cache_age_ms: float,
    ) -> ModalitySynchronizationRecord:
        source_timestamp = (
            extract_source_timestamp(value)
        )

        return ModalitySynchronizationRecord(
            modality=modality,
            available=True,
            source_timestamp=source_timestamp,
            arrival_timestamp=source_timestamp,
            offset_from_anchor_ms=None,
            source_age_ms=round(
                cache_age_ms,
                6,
            ),
            within_window=False,
            stale=False,
            selected=True,
            limitation_codes=[
                "recovered_from_cached_observation"
            ],
            metadata={
                "recovered": True,
                "recovery_strategy": (
                    RecoveryStrategy.REUSE_CACHED.value
                ),
            },
        )

    # ========================================================
    # REACQUISITION
    # ========================================================

    def _create_reacquisition_request(
        self,
        *,
        modality: str,
        reason_codes: List[str],
        required: bool,
    ) -> ReacquisitionRequest:
        base_priority = (
            DEFAULT_REACQUISITION_PRIORITIES[
                modality
            ]
        )

        priority = min(
            100,
            base_priority
            + (10 if required else 0),
        )

        deadline = {
            "vision": 500.0,
            "audio": 1000.0,
            "spatial": 3000.0,
            "motion": 300.0,
            "interaction": 250.0,
            "wearable": 5000.0,
            "source_device": 5000.0,
            "environment": 10000.0,
        }[modality]

        return ReacquisitionRequest(
            request_id=(
                "REACQ_"
                f"{uuid.uuid4().hex[:10].upper()}"
            ),
            modality=modality,
            priority=priority,
            reason_codes=reason_codes,
            requested_at=utc_now_iso(),
            target_deadline_ms=deadline,
            metadata={
                "required_modality": required,
            },
        )

    # ========================================================
    # FRAME REBUILDING
    # ========================================================

    def _rebuild_frame_lists(
        self,
        frame: SynchronizedMultimodalFrame,
        decisions: Mapping[
            str,
            ModalityRecoveryDecision,
        ],
    ) -> None:
        frame.available_modalities = sorted(
            modality
            for modality, decision
            in decisions.items()
            if decision.usable
        )

        frame.selected_modalities = list(
            frame.available_modalities
        )

        frame.missing_modalities = sorted(
            modality
            for modality, decision
            in decisions.items()
            if not decision.usable
        )

        frame.stale_modalities = sorted(
            modality
            for modality, decision
            in decisions.items()
            if (
                decision.original_availability
                == ModalityAvailability.STALE
                and not decision.recovered
            )
        )

        frame.excluded_modalities = sorted(
            modality
            for modality, decision
            in decisions.items()
            if (
                decision.original_availability
                == ModalityAvailability.EXCLUDED
                and not decision.recovered
            )
        )

        required_modalities = (
            self._get_required_modalities()
        )

        usable_required = sum(
            1
            for modality in required_modalities
            if decisions[modality].usable
        )

        frame.completeness_score = round(
            (
                usable_required
                / len(required_modalities)
            )
            if required_modalities
            else 1.0,
            6,
        )

    def _determine_recovery_status(
        self,
        *,
        required_unavailable: List[str],
        unavailable_modalities: List[str],
        recovered_modalities: List[str],
    ) -> Tuple[RecoveryStatus, bool]:
        if required_unavailable:
            if self._block_on_required_missing():
                return RecoveryStatus.BLOCKED, False

            return RecoveryStatus.DEGRADED, True

        if unavailable_modalities:
            return RecoveryStatus.PARTIAL, True

        if recovered_modalities:
            return RecoveryStatus.PARTIAL, True

        return RecoveryStatus.COMPLETE, True

    def _map_recovery_to_sync_status(
        self,
        status: RecoveryStatus,
    ) -> SynchronizationStatus:
        mapping = {
            RecoveryStatus.COMPLETE: (
                SynchronizationStatus.COMPLETE
            ),
            RecoveryStatus.PARTIAL: (
                SynchronizationStatus.PARTIAL
            ),
            RecoveryStatus.DEGRADED: (
                SynchronizationStatus.DEGRADED
            ),
            RecoveryStatus.BLOCKED: (
                SynchronizationStatus.DEGRADED
            ),
        }

        return mapping[status]

    # ========================================================
    # CONFIDENCE ADJUSTMENT
    # ========================================================

    def _calculate_adjusted_confidence(
        self,
        confidence_report: ConfidenceReport,
        decisions: Mapping[
            str,
            ModalityRecoveryDecision,
        ],
    ) -> float:
        weighted_sum = 0.0
        weight_sum = 0.0

        modality_weights = (
            confidence_report.metadata.get(
                "modality_weights",
                {},
            )
        )

        for modality, decision in decisions.items():
            weight = float(
                modality_weights.get(
                    modality,
                    1.0,
                )
            )

            if weight <= 0:
                continue

            weighted_sum += (
                weight
                * decision.final_confidence
            )
            weight_sum += weight

        if weight_sum <= 0:
            return 0.0

        base = weighted_sum / weight_sum

        required_unusable = sum(
            1
            for modality in self._get_required_modalities()
            if not decisions[modality].usable
        )

        required_count = max(
            1,
            len(
                self._get_required_modalities()
            ),
        )

        required_factor = (
            1.0
            - 0.40
            * (
                required_unusable
                / required_count
            )
        )

        return clamp(
            base * required_factor,
            0.0,
            1.0,
        )

    # ========================================================
    # STATISTICS AND DIAGNOSTICS
    # ========================================================

    def _register_report(
        self,
        report: MissingModalityRecoveryReport,
        elapsed_seconds: float,
    ) -> None:
        self.statistics.total_runs += 1
        self.statistics.cumulative_processing_seconds += (
            elapsed_seconds
        )

        self.statistics.last_report_id = (
            report.report_id
        )
        self.statistics.last_status = (
            report.status.value
        )
        self.statistics.last_error = None

        if report.status == RecoveryStatus.COMPLETE:
            self.statistics.complete_runs += 1

        elif report.status == RecoveryStatus.PARTIAL:
            self.statistics.partial_runs += 1

        elif report.status == RecoveryStatus.DEGRADED:
            self.statistics.degraded_runs += 1

        elif report.status == RecoveryStatus.BLOCKED:
            self.statistics.blocked_runs += 1

        self.statistics.total_cached_recoveries += len(
            report.recovered_modalities
        )

        self.statistics.total_reacquisition_requests += len(
            report.reacquisition_requests
        )

        self.statistics.total_required_failures += len(
            report.required_unavailable_modalities
        )

        self._last_report = report

    def get_cached_modalities(
        self,
    ) -> List[str]:
        return sorted(self._cache.keys())

    def get_last_report(
        self,
    ) -> Optional[
        MissingModalityRecoveryReport
    ]:
        return self._last_report

    def clear_cache(self) -> None:
        self._cache.clear()

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "cached_modalities": (
                self.get_cached_modalities()
            ),
            "last_report_available": (
                self._last_report is not None
            ),
            "last_report_id": (
                self._last_report.report_id
                if self._last_report
                else None
            ),
            "last_status": (
                self._last_report.status.value
                if self._last_report
                else None
            ),
            "cached_reuse_enabled": (
                self._allow_cached_reuse()
            ),
            "statistics": (
                self.statistics.to_dict()
            ),
        }


# ============================================================
# SELF-TEST
# ============================================================

def run_missing_modality_handler_self_test() -> bool:
    print("\n" + "=" * 72)
    print("NOONGIL-X | MISSING MODALITY HANDLER TEST")
    print("=" * 72)

    try:
        print("[1/8] Creating test settings...")

        settings = create_test_settings()
        handler = MissingModalityHandler(settings)

        print("[SUCCESS] Handler initialized.")

        print("[2/8] Building complete synchronized frame...")

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
        from layer1.processing.confidence_estimator import (
            ConfidenceEstimator,
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

        complete_frame = (
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

        confidence_estimator = (
            ConfidenceEstimator(settings)
        )

        complete_confidence = (
            confidence_estimator.estimate(
                complete_frame,
                raise_on_error=True,
            )
        )

        handler.update_cache(
            complete_frame
        )

        print("[SUCCESS] Complete frame and cache created.")

        print("[3/8] Creating a frame with missing vision...")

        missing_frame = copy.deepcopy(
            complete_frame
        )

        missing_frame.frame_id = (
            "SYNC_TEST_MISSING_VISION"
        )
        missing_frame.vision = None
        missing_frame.selected_modalities = [
            modality
            for modality
            in missing_frame.selected_modalities
            if modality != "vision"
        ]
        missing_frame.available_modalities = [
            modality
            for modality
            in missing_frame.available_modalities
            if modality != "vision"
        ]

        if "vision" not in (
            missing_frame.missing_modalities
        ):
            missing_frame.missing_modalities.append(
                "vision"
            )

        missing_frame.synchronization_records[
            "vision"
        ] = ModalitySynchronizationRecord(
            modality="vision",
            available=False,
            selected=False,
            limitation_codes=[
                "modality_missing"
            ],
        )

        missing_confidence = copy.deepcopy(
            complete_confidence
        )
        missing_confidence.frame_id = (
            missing_frame.frame_id
        )
        missing_confidence.modality_confidences[
            "vision"
        ].available = False
        missing_confidence.modality_confidences[
            "vision"
        ].selected = False
        missing_confidence.modality_confidences[
            "vision"
        ].confidence_score = 0.0

        print("[SUCCESS] Missing-vision frame created.")

        print("[4/8] Recovering missing vision from cache...")

        recovered = handler.handle(
            missing_frame,
            missing_confidence,
            raise_on_error=True,
        )

        if recovered.frame.vision is None:
            raise AssertionError(
                "Vision was not recovered from cache."
            )

        if "vision" not in (
            recovered.report.recovered_modalities
        ):
            raise AssertionError(
                "Recovered vision was not recorded."
            )

        vision_decision = (
            recovered.report.decisions["vision"]
        )

        if (
            vision_decision.strategy
            != RecoveryStrategy.REUSE_CACHED
        ):
            raise AssertionError(
                "Unexpected vision recovery strategy."
            )

        print("[SUCCESS] Cached recovery works.")

        print("[5/8] Testing unavailable environment...")

        environment_decision = (
            recovered.report.decisions[
                "environment"
            ]
        )

        if (
            environment_decision
            .final_availability
            != ModalityAvailability.UNAVAILABLE
        ):
            raise AssertionError(
                "Environment should remain unavailable."
            )

        print(
            "[SUCCESS] Unavailable modality remains explicit."
        )

        print("[6/8] Testing reacquisition requests...")

        requested_modalities = {
            request.modality
            for request
            in recovered.report
            .reacquisition_requests
        }

        if "vision" not in requested_modalities:
            raise AssertionError(
                "Fresh vision reacquisition was not requested."
            )

        if "environment" not in requested_modalities:
            raise AssertionError(
                "Environment reacquisition was not requested."
            )

        print("[SUCCESS] Reacquisition requests created.")

        print("[7/8] Validating recovery report...")

        recovered.report.validate()
        recovered.frame.validate()

        if not recovered.report.safe_to_continue:
            raise AssertionError(
                "Recovered frame should be safe to continue."
            )

        print("[SUCCESS] Recovery report is valid.")

        print("[8/8] Checking diagnostics...")

        health = handler.health_check()

        if not health["healthy"]:
            raise AssertionError(
                "Handler health check failed."
            )

        if (
            health["statistics"]["total_runs"]
            != 1
        ):
            raise AssertionError(
                "Recovery run count is incorrect."
            )

        if (
            health["statistics"]
            ["total_cached_recoveries"]
            < 1
        ):
            raise AssertionError(
                "Cached recovery count is incorrect."
            )

        print("[SUCCESS] Diagnostics are correct.")

        print("\nRecovery summary:")
        print(
            json.dumps(
                {
                    "report_id": (
                        recovered.report.report_id
                    ),
                    "original_frame_id": (
                        recovered.report
                        .original_frame_id
                    ),
                    "recovered_frame_id": (
                        recovered.report
                        .recovered_frame_id
                    ),
                    "status": (
                        recovered.report.status.value
                    ),
                    "safe_to_continue": (
                        recovered.report
                        .safe_to_continue
                    ),
                    "recovered_modalities": (
                        recovered.report
                        .recovered_modalities
                    ),
                    "unavailable_modalities": (
                        recovered.report
                        .unavailable_modalities
                    ),
                    "required_unavailable_modalities": (
                        recovered.report
                        .required_unavailable_modalities
                    ),
                    "reacquisition_modalities": [
                        request.modality
                        for request
                        in recovered.report
                        .reacquisition_requests
                    ],
                    "original_overall_confidence": (
                        recovered.report
                        .original_overall_confidence
                    ),
                    "adjusted_overall_confidence": (
                        recovered.report
                        .adjusted_overall_confidence
                    ),
                    "warnings": (
                        recovered.report.warnings
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nHandler health:")
        print(
            json.dumps(
                health,
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n" + "=" * 72)
        print(
            "[PASSED] MISSING MODALITY HANDLER IS WORKING"
        )
        print("=" * 72)

        return True

    except Exception as error:
        print("\n" + "=" * 72)
        print(
            "[FAILED] MISSING MODALITY HANDLER TEST"
        )
        print("=" * 72)
        print(
            f"[ERROR] {type(error).__name__}: {error}"
        )

        return False


if __name__ == "__main__":
    if not run_missing_modality_handler_self_test():
        raise SystemExit(1)

"""
============================================================
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : Multimodal Synchronizer
File    : layer1/processing/multimodal_synchronizer.py
============================================================

Purpose
-------
Align processed Layer 1 modality outputs into one coherent,
timestamped multimodal frame for downstream confidence
estimation and Layer 2 dispatch.

Responsibilities
----------------
1. Accept processed modality outputs
2. Select one synchronization anchor timestamp
3. Measure temporal offset for every modality
4. Reject or mark stale modality samples
5. Calculate cross-modal temporal skew
6. Calculate synchronization completeness
7. Preserve modality-specific data objects
8. Produce one SynchronizedMultimodalFrame
9. Maintain latest modality state for asynchronous streams
10. Provide diagnostics and a standalone self-test

Architectural Boundary
----------------------
This module does NOT:
- calculate final modality confidence;
- repair missing modalities;
- perform perception;
- perform sensor fusion;
- perform semantic reasoning;
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

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from layer1.config.settings import (
    Layer1Settings,
    create_default_settings,
    create_test_settings,
)
from layer1.schemas.sensor_packet import (
    AudioData,
    InteractionData,
    MotionData,
    SourceDevice,
    SpatialData,
    VisionData,
    WearableData,
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

SYNCHRONIZER_VERSION = "1.0"

CORE_MODALITIES = {
    "vision",
    "audio",
    "spatial",
    "motion",
}

OPTIONAL_MODALITIES = {
    "interaction",
    "wearable",
    "source_device",
    "environment",
}

ALL_MODALITIES = CORE_MODALITIES | OPTIONAL_MODALITIES


# ============================================================
# EXCEPTIONS
# ============================================================

class SynchronizationError(Exception):
    """Base exception for multimodal synchronization."""


class SynchronizationValidationError(SynchronizationError):
    """Raised when synchronization input is invalid."""


class SynchronizationProcessingError(SynchronizationError):
    """Raised when a synchronized frame cannot be produced."""


# ============================================================
# ENUMERATIONS
# ============================================================

class SynchronizationStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    DEGRADED = "degraded"
    EMPTY = "empty"


class AnchorStrategy(str, Enum):
    LATEST = "latest"
    EARLIEST = "earliest"
    MEDIAN = "median"
    VISION = "vision"
    EXPLICIT = "explicit"


# ============================================================
# HELPERS
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="milliseconds"
    )


def parse_iso_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise SynchronizationValidationError(
            "Timestamp must be a non-empty ISO 8601 string."
        )

    normalized = value.strip().replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise SynchronizationValidationError(
            f"Invalid ISO 8601 timestamp: {value!r}"
        ) from error

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def timestamp_difference_ms(
    first: str,
    second: str,
) -> float:
    first_dt = parse_iso_timestamp(first)
    second_dt = parse_iso_timestamp(second)

    return abs(
        (first_dt - second_dt).total_seconds()
        * 1000.0
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


def extract_source_timestamp(value: Any) -> str:
    """
    Extract the canonical source timestamp from a modality object.
    """

    metadata = getattr(value, "metadata", None)

    if metadata is None:
        raise SynchronizationValidationError(
            f"{type(value).__name__} has no metadata field."
        )

    source_timestamp = getattr(
        metadata,
        "source_timestamp",
        None,
    )

    if source_timestamp is None:
        raise SynchronizationValidationError(
            f"{type(value).__name__} metadata has no "
            "source_timestamp."
        )

    parse_iso_timestamp(source_timestamp)

    return source_timestamp


def extract_arrival_timestamp(value: Any) -> str:
    metadata = getattr(value, "metadata", None)

    if metadata is None:
        raise SynchronizationValidationError(
            f"{type(value).__name__} has no metadata field."
        )

    arrival_timestamp = getattr(
        metadata,
        "arrival_timestamp",
        None,
    )

    if arrival_timestamp is None:
        return extract_source_timestamp(value)

    parse_iso_timestamp(arrival_timestamp)

    return arrival_timestamp


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ModalitySynchronizationRecord:
    """
    Temporal alignment information for one modality.
    """

    modality: str
    available: bool

    source_timestamp: Optional[str] = None
    arrival_timestamp: Optional[str] = None

    offset_from_anchor_ms: Optional[float] = None
    source_age_ms: Optional[float] = None

    within_window: bool = False
    stale: bool = False
    selected: bool = False

    limitation_codes: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return make_json_safe(asdict(self))


@dataclass
class SynchronizedMultimodalFrame:
    """
    One temporally aligned Layer 1 multimodal frame.
    """

    frame_id: str
    created_at: str

    anchor_timestamp: str
    anchor_strategy: AnchorStrategy

    status: SynchronizationStatus

    vision: Optional[VisionData] = None
    audio: Optional[AudioData] = None
    spatial: Optional[SpatialData] = None
    motion: Optional[MotionData] = None
    interaction: Optional[InteractionData] = None
    wearable: Optional[WearableData] = None
    source_device: Optional[SourceDevice] = None
    environment: Optional[Any] = None

    synchronization_records: Dict[
        str,
        ModalitySynchronizationRecord,
    ] = field(default_factory=dict)

    available_modalities: List[str] = field(
        default_factory=list
    )
    selected_modalities: List[str] = field(
        default_factory=list
    )
    missing_modalities: List[str] = field(
        default_factory=list
    )
    stale_modalities: List[str] = field(
        default_factory=list
    )
    excluded_modalities: List[str] = field(
        default_factory=list
    )

    temporal_span_ms: float = 0.0
    maximum_offset_ms: float = 0.0
    average_offset_ms: float = 0.0

    completeness_score: float = 0.0
    synchronization_score: float = 0.0

    warnings: List[str] = field(
        default_factory=list
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        if not self.frame_id.strip():
            raise SynchronizationValidationError(
                "frame_id cannot be empty."
            )

        parse_iso_timestamp(self.created_at)
        parse_iso_timestamp(self.anchor_timestamp)

        for field_name, value in (
            ("temporal_span_ms", self.temporal_span_ms),
            ("maximum_offset_ms", self.maximum_offset_ms),
            ("average_offset_ms", self.average_offset_ms),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise SynchronizationValidationError(
                    f"{field_name} must be finite and "
                    "non-negative."
                )

        for field_name, value in (
            ("completeness_score", self.completeness_score),
            ("synchronization_score", self.synchronization_score),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise SynchronizationValidationError(
                    f"{field_name} must be between 0 and 1."
                )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return make_json_safe(asdict(self))


@dataclass
class SynchronizerStatistics:
    """
    Runtime statistics for MultimodalSynchronizer.
    """

    total_frames: int = 0
    complete_frames: int = 0
    partial_frames: int = 0
    degraded_frames: int = 0
    empty_frames: int = 0

    total_stale_modalities: int = 0
    total_excluded_modalities: int = 0
    total_missing_modalities: int = 0

    cumulative_processing_seconds: float = 0.0

    last_frame_id: Optional[str] = None
    last_status: Optional[str] = None
    last_error: Optional[str] = None

    @property
    def average_processing_seconds(self) -> float:
        if self.total_frames == 0:
            return 0.0

        return (
            self.cumulative_processing_seconds
            / self.total_frames
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["average_processing_seconds"] = (
            self.average_processing_seconds
        )
        return payload


# ============================================================
# MULTIMODAL SYNCHRONIZER
# ============================================================

class MultimodalSynchronizer:
    """
    Synchronize asynchronous Layer 1 modality outputs.

    The synchronizer maintains the most recent sample for each
    modality. A new frame may be created from a complete set of
    fresh samples or from a partial set when some streams are
    unavailable.
    """

    def __init__(
        self,
        settings: Optional[Layer1Settings] = None,
        *,
        anchor_strategy: AnchorStrategy = (
            AnchorStrategy.LATEST
        ),
    ) -> None:
        self.settings = settings or create_default_settings()
        self.settings.validate()

        self.anchor_strategy = anchor_strategy

        self.logger = get_logger(
            "processing.multimodal_synchronizer"
        )

        self.statistics = SynchronizerStatistics()

        self._latest_values: Dict[str, Any] = {}
        self._frame_counter = 0
        self._last_frame: Optional[
            SynchronizedMultimodalFrame
        ] = None

    # ========================================================
    # PUBLIC API
    # ========================================================

    def update(
        self,
        *,
        vision: Optional[VisionData] = None,
        audio: Optional[AudioData] = None,
        spatial: Optional[SpatialData] = None,
        motion: Optional[MotionData] = None,
        interaction: Optional[InteractionData] = None,
        wearable: Optional[WearableData] = None,
        source_device: Optional[SourceDevice] = None,
        environment: Optional[Any] = None,
    ) -> None:
        """
        Update the latest cached values without creating a frame.
        """

        values = {
            "vision": vision,
            "audio": audio,
            "spatial": spatial,
            "motion": motion,
            "interaction": interaction,
            "wearable": wearable,
            "source_device": source_device,
            "environment": environment,
        }

        for modality, value in values.items():
            if value is None:
                continue

            self._validate_modality_value(
                modality,
                value,
            )
            self._latest_values[modality] = value

    def synchronize(
        self,
        *,
        vision: Optional[VisionData] = None,
        audio: Optional[AudioData] = None,
        spatial: Optional[SpatialData] = None,
        motion: Optional[MotionData] = None,
        interaction: Optional[InteractionData] = None,
        wearable: Optional[WearableData] = None,
        source_device: Optional[SourceDevice] = None,
        environment: Optional[Any] = None,
        explicit_anchor_timestamp: Optional[str] = None,
        include_cached_values: bool = True,
        raise_on_error: Optional[bool] = None,
    ) -> SynchronizedMultimodalFrame:
        """
        Create one synchronized multimodal frame.
        """

        should_raise = (
            self.settings.runtime.fail_fast
            if raise_on_error is None
            else raise_on_error
        )

        started = time.perf_counter()

        try:
            with PipelineTimer(
                "multimodal_synchronizer.synchronize",
                logger=self.logger,
                metadata={
                    "anchor_strategy": (
                        self.anchor_strategy.value
                    ),
                    "include_cached_values": (
                        include_cached_values
                    ),
                },
            ):
                incoming_values = {
                    "vision": vision,
                    "audio": audio,
                    "spatial": spatial,
                    "motion": motion,
                    "interaction": interaction,
                    "wearable": wearable,
                    "source_device": source_device,
                    "environment": environment,
                }

                for modality, value in (
                    incoming_values.items()
                ):
                    if value is None:
                        continue

                    self._validate_modality_value(
                        modality,
                        value,
                    )

                if include_cached_values:
                    working_values = dict(
                        self._latest_values
                    )
                    working_values.update(
                        {
                            modality: value
                            for modality, value
                            in incoming_values.items()
                            if value is not None
                        }
                    )
                else:
                    working_values = {
                        modality: value
                        for modality, value
                        in incoming_values.items()
                        if value is not None
                    }

                if not working_values:
                    frame = self._build_empty_frame(
                        explicit_anchor_timestamp
                    )
                    self._register_frame(
                        frame,
                        time.perf_counter() - started,
                    )
                    return frame

                self._latest_values.update(
                    working_values
                )

                anchor_timestamp = (
                    self._select_anchor_timestamp(
                        working_values,
                        explicit_anchor_timestamp=(
                            explicit_anchor_timestamp
                        ),
                    )
                )

                records = (
                    self._create_synchronization_records(
                        working_values,
                        anchor_timestamp=anchor_timestamp,
                    )
                )

                selected_values = {
                    modality: working_values[modality]
                    for modality, record
                    in records.items()
                    if record.selected
                }

                frame = self._build_frame(
                    anchor_timestamp=anchor_timestamp,
                    working_values=working_values,
                    selected_values=selected_values,
                    records=records,
                )

                frame.validate()

                elapsed = time.perf_counter() - started

                self._register_frame(
                    frame,
                    elapsed,
                )

                log_sensor_event(
                    modality="interaction",
                    event="Multimodal frame synchronized",
                    sensor_type="multimodal_synchronizer",
                    packet_id=frame.frame_id,
                    details={
                        "status": frame.status.value,
                        "anchor_timestamp": (
                            frame.anchor_timestamp
                        ),
                        "selected_modalities": (
                            frame.selected_modalities
                        ),
                        "missing_modalities": (
                            frame.missing_modalities
                        ),
                        "stale_modalities": (
                            frame.stale_modalities
                        ),
                        "temporal_span_ms": (
                            frame.temporal_span_ms
                        ),
                        "completeness_score": (
                            frame.completeness_score
                        ),
                        "synchronization_score": (
                            frame.synchronization_score
                        ),
                    },
                )

                return frame

        except Exception as error:
            self.statistics.last_error = (
                f"{type(error).__name__}: {error}"
            )

            log_exception(
                self.logger,
                "Multimodal synchronization failed",
                error=error,
                details={
                    "anchor_strategy": (
                        self.anchor_strategy.value
                    ),
                },
            )

            if should_raise:
                raise

            raise SynchronizationProcessingError(
                f"Synchronization failed: {error}"
            ) from error

    def synchronize_from_results(
        self,
        *,
        vision_result: Optional[Any] = None,
        audio_result: Optional[Any] = None,
        spatial_result: Optional[Any] = None,
        motion_result: Optional[Any] = None,
        interaction_result: Optional[Any] = None,
        device_results: Optional[
            Iterable[Any]
        ] = None,
        include_cached_values: bool = True,
        raise_on_error: Optional[bool] = None,
    ) -> SynchronizedMultimodalFrame:
        """
        Synchronize directly from modality processing-result objects.
        """

        vision = (
            getattr(
                vision_result,
                "vision_data",
                None,
            )
            if vision_result is not None
            else None
        )

        audio = (
            getattr(
                audio_result,
                "audio_data",
                None,
            )
            if audio_result is not None
            else None
        )

        spatial = (
            getattr(
                spatial_result,
                "spatial_data",
                None,
            )
            if spatial_result is not None
            else None
        )

        motion = (
            getattr(
                motion_result,
                "motion_data",
                None,
            )
            if motion_result is not None
            else None
        )

        interaction = (
            getattr(
                interaction_result,
                "interaction_data",
                None,
            )
            if interaction_result is not None
            else None
        )

        source_device: Optional[
            SourceDevice
        ] = None
        wearable: Optional[
            WearableData
        ] = None

        if device_results is not None:
            for result in device_results:
                candidate_source = getattr(
                    result,
                    "source_device",
                    None,
                )
                candidate_wearable = getattr(
                    result,
                    "wearable_data",
                    None,
                )

                if candidate_source is not None:
                    source_device = candidate_source

                if candidate_wearable is not None:
                    wearable = candidate_wearable

        return self.synchronize(
            vision=vision,
            audio=audio,
            spatial=spatial,
            motion=motion,
            interaction=interaction,
            wearable=wearable,
            source_device=source_device,
            include_cached_values=include_cached_values,
            raise_on_error=raise_on_error,
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_modality_value(
        self,
        modality: str,
        value: Any,
    ) -> None:
        if modality not in ALL_MODALITIES:
            raise SynchronizationValidationError(
                f"Unsupported modality: {modality!r}"
            )

        expected_types = {
            "vision": VisionData,
            "audio": AudioData,
            "spatial": SpatialData,
            "motion": MotionData,
            "interaction": InteractionData,
            "wearable": WearableData,
            "source_device": SourceDevice,
        }

        expected = expected_types.get(modality)

        if (
            expected is not None
            and not isinstance(value, expected)
        ):
            raise SynchronizationValidationError(
                f"{modality!r} must be "
                f"{expected.__name__}, not "
                f"{type(value).__name__}."
            )

        validate = getattr(value, "validate", None)

        if callable(validate):
            validate()

        if modality != "source_device":
            extract_source_timestamp(value)

    # ========================================================
    # ANCHOR SELECTION
    # ========================================================

    def _select_anchor_timestamp(
        self,
        values: Mapping[str, Any],
        *,
        explicit_anchor_timestamp: Optional[str],
    ) -> str:
        if (
            explicit_anchor_timestamp is not None
        ):
            parse_iso_timestamp(
                explicit_anchor_timestamp
            )
            return explicit_anchor_timestamp

        timestamps = {
            modality: extract_source_timestamp(value)
            for modality, value in values.items()
            if modality != "source_device"
        }

        if not timestamps:
            return utc_now_iso()

        if self.anchor_strategy == AnchorStrategy.VISION:
            if "vision" in timestamps:
                return timestamps["vision"]

        if self.anchor_strategy == AnchorStrategy.EARLIEST:
            return min(
                timestamps.values(),
                key=parse_iso_timestamp,
            )

        if self.anchor_strategy == AnchorStrategy.MEDIAN:
            ordered = sorted(
                timestamps.values(),
                key=parse_iso_timestamp,
            )
            return ordered[len(ordered) // 2]

        return max(
            timestamps.values(),
            key=parse_iso_timestamp,
        )

    # ========================================================
    # SYNCHRONIZATION SETTINGS COMPATIBILITY
    # ========================================================

    def _get_synchronization_window_ms(self) -> float:
        """
        Return the maximum allowed timestamp difference between
        modality samples.

        Multiple attribute names are supported so this module can
        run with older and newer versions of settings.py.
        """

        sync_settings = self.settings.synchronization

        candidate_names = (
            "maximum_time_difference_ms",
            "synchronization_window_ms",
            "maximum_timestamp_difference_ms",
            "max_time_difference_ms",
            "max_time_skew_ms",
            "tolerance_ms",
        )

        for name in candidate_names:
            value = getattr(sync_settings, name, None)

            if value is None:
                continue

            try:
                parsed = float(value)
            except (TypeError, ValueError) as error:
                raise SynchronizationValidationError(
                    f"{name} must be numeric."
                ) from error

            if not math.isfinite(parsed) or parsed <= 0:
                raise SynchronizationValidationError(
                    f"{name} must be finite and greater than zero."
                )

            return parsed

        return 500.0

    def _get_maximum_modality_age_ms(self) -> float:
        """
        Return the maximum age allowed for a cached modality sample.
        """

        sync_settings = self.settings.synchronization

        candidate_names = (
            "maximum_modality_age_ms",
            "max_modality_age_ms",
            "maximum_sample_age_ms",
            "stale_after_ms",
            "freshness_timeout_ms",
        )

        for name in candidate_names:
            value = getattr(sync_settings, name, None)

            if value is None:
                continue

            try:
                parsed = float(value)
            except (TypeError, ValueError) as error:
                raise SynchronizationValidationError(
                    f"{name} must be numeric."
                ) from error

            if not math.isfinite(parsed) or parsed <= 0:
                raise SynchronizationValidationError(
                    f"{name} must be finite and greater than zero."
                )

            return parsed

        return 5000.0

    def _get_required_modalities(self) -> set[str]:
        """
        Return modalities required for a complete synchronized frame.
        """

        sync_settings = self.settings.synchronization

        candidate_names = (
            "required_modalities",
            "mandatory_modalities",
            "core_modalities",
        )

        for name in candidate_names:
            value = getattr(sync_settings, name, None)

            if value is None:
                continue

            if not isinstance(value, (list, tuple, set)):
                raise SynchronizationValidationError(
                    f"{name} must be a list, tuple, or set."
                )

            normalized = {
                str(modality).strip().lower()
                for modality in value
                if str(modality).strip()
            }

            unsupported = normalized - ALL_MODALITIES

            if unsupported:
                raise SynchronizationValidationError(
                    "Unsupported required modalities: "
                    f"{sorted(unsupported)}"
                )

            if not normalized:
                raise SynchronizationValidationError(
                    f"{name} cannot be empty."
                )

            return normalized

        return set(CORE_MODALITIES)

    # ========================================================
    # RECORD CREATION
    # ========================================================
    
    def _create_synchronization_records(
        self,
        values: Mapping[str, Any],
        *,
        anchor_timestamp: str,
    ) -> Dict[
        str,
        ModalitySynchronizationRecord,
    ]:
        records: Dict[
            str,
            ModalitySynchronizationRecord,
        ] = {}

        now = datetime.now(timezone.utc)

        synchronization_window_ms = (
            self._get_synchronization_window_ms()
        )

        stale_threshold_ms = (
            self._get_maximum_modality_age_ms()
        )

        for modality in sorted(
            ALL_MODALITIES
        ):
            value = values.get(modality)

            if value is None:
                records[modality] = (
                    ModalitySynchronizationRecord(
                        modality=modality,
                        available=False,
                        limitation_codes=[
                            "modality_missing"
                        ],
                    )
                )
                continue

            if modality == "source_device":
                records[modality] = (
                    ModalitySynchronizationRecord(
                        modality=modality,
                        available=True,
                        within_window=True,
                        selected=True,
                        metadata={
                            "non_temporal_device_context": True
                        },
                    )
                )
                continue

            source_timestamp = (
                extract_source_timestamp(value)
            )
            arrival_timestamp = (
                extract_arrival_timestamp(value)
            )

            source_dt = parse_iso_timestamp(
                source_timestamp
            )

            offset_ms = (
                timestamp_difference_ms(
                    source_timestamp,
                    anchor_timestamp,
                )
            )

            age_ms = max(
                0.0,
                (
                    now - source_dt
                ).total_seconds() * 1000.0,
            )

            within_window = (
                offset_ms
                <= synchronization_window_ms
            )

            stale = (
                age_ms > stale_threshold_ms
            )

            limitation_codes: List[str] = []

            if not within_window:
                limitation_codes.append(
                    "outside_synchronization_window"
                )

            if stale:
                limitation_codes.append(
                    "stale_modality_sample"
                )

            selected = (
                within_window
                and not stale
            )

            records[modality] = (
                ModalitySynchronizationRecord(
                    modality=modality,
                    available=True,
                    source_timestamp=source_timestamp,
                    arrival_timestamp=arrival_timestamp,
                    offset_from_anchor_ms=round(
                        offset_ms,
                        6,
                    ),
                    source_age_ms=round(
                        age_ms,
                        6,
                    ),
                    within_window=within_window,
                    stale=stale,
                    selected=selected,
                    limitation_codes=limitation_codes,
                    metadata={
                        "value_type": (
                            type(value).__name__
                        ),
                    },
                )
            )

        return records

    # ========================================================
    # FRAME BUILDING
    # ========================================================

    def _build_frame(
        self,
        *,
        anchor_timestamp: str,
        working_values: Mapping[str, Any],
        selected_values: Mapping[str, Any],
        records: Mapping[
            str,
            ModalitySynchronizationRecord,
        ],
    ) -> SynchronizedMultimodalFrame:
        self._frame_counter += 1

        frame_id = (
            f"SYNC_{self._frame_counter:06d}_"
            f"{uuid.uuid4().hex[:8].upper()}"
        )

        available_modalities = sorted(
            modality
            for modality, record
            in records.items()
            if record.available
        )

        selected_modalities = sorted(
            modality
            for modality, record
            in records.items()
            if record.selected
        )

        missing_modalities = sorted(
            modality
            for modality, record
            in records.items()
            if not record.available
        )

        stale_modalities = sorted(
            modality
            for modality, record
            in records.items()
            if record.stale
        )

        excluded_modalities = sorted(
            modality
            for modality, record
            in records.items()
            if record.available
            and not record.selected
        )

        selected_timestamps = [
            record.source_timestamp
            for record in records.values()
            if record.selected
            and record.source_timestamp
            is not None
        ]

        offsets = [
            float(
                record.offset_from_anchor_ms
            )
            for record in records.values()
            if record.selected
            and record.offset_from_anchor_ms
            is not None
        ]

        temporal_span_ms = (
            self._calculate_temporal_span(
                selected_timestamps
            )
        )

        maximum_offset_ms = (
            max(offsets)
            if offsets
            else 0.0
        )

        average_offset_ms = (
            sum(offsets) / len(offsets)
            if offsets
            else 0.0
        )

        required_modalities = (
            self._get_required_modalities()
        )

        selected_set = set(
            selected_modalities
        )

        required_selected = (
            required_modalities
            & selected_set
        )

        completeness_score = (
            len(required_selected)
            / len(required_modalities)
            if required_modalities
            else 1.0
        )

        synchronization_window_ms = (
            self._get_synchronization_window_ms()
        )

        synchronization_score = (
            1.0
            - clamp(
                maximum_offset_ms
                / max(
                    synchronization_window_ms,
                    1.0,
                ),
                0.0,
                1.0,
            )
        )

        status = self._determine_status(
            required_modalities=required_modalities,
            selected_modalities=selected_set,
            stale_modalities=set(
                stale_modalities
            ),
            excluded_modalities=set(
                excluded_modalities
            ),
        )

        warnings: List[str] = []

        if missing_modalities:
            warnings.append(
                "missing_modalities_present"
            )

        if stale_modalities:
            warnings.append(
                "stale_modalities_present"
            )

        if excluded_modalities:
            warnings.append(
                "modalities_excluded_by_time_window"
            )

        if completeness_score < 1.0:
            warnings.append(
                "required_modalities_incomplete"
            )

        return SynchronizedMultimodalFrame(
            frame_id=frame_id,
            created_at=utc_now_iso(),
            anchor_timestamp=anchor_timestamp,
            anchor_strategy=self.anchor_strategy,
            status=status,
            vision=selected_values.get("vision"),
            audio=selected_values.get("audio"),
            spatial=selected_values.get("spatial"),
            motion=selected_values.get("motion"),
            interaction=selected_values.get(
                "interaction"
            ),
            wearable=selected_values.get(
                "wearable"
            ),
            source_device=working_values.get(
                "source_device"
            ),
            environment=selected_values.get(
                "environment"
            ),
            synchronization_records=dict(
                records
            ),
            available_modalities=available_modalities,
            selected_modalities=selected_modalities,
            missing_modalities=missing_modalities,
            stale_modalities=stale_modalities,
            excluded_modalities=excluded_modalities,
            temporal_span_ms=round(
                temporal_span_ms,
                6,
            ),
            maximum_offset_ms=round(
                maximum_offset_ms,
                6,
            ),
            average_offset_ms=round(
                average_offset_ms,
                6,
            ),
            completeness_score=round(
                completeness_score,
                6,
            ),
            synchronization_score=round(
                synchronization_score,
                6,
            ),
            warnings=warnings,
            metadata={
                "synchronizer_version": (
                    SYNCHRONIZER_VERSION
                ),
                "required_modalities": sorted(
                    required_modalities
                ),
                "synchronization_window_ms": (
                    synchronization_window_ms
                ),
                "maximum_modality_age_ms": (
                    self._get_maximum_modality_age_ms()
                ),
            },
        )

    def _build_empty_frame(
        self,
        explicit_anchor_timestamp: Optional[
            str
        ],
    ) -> SynchronizedMultimodalFrame:
        self._frame_counter += 1

        anchor = (
            explicit_anchor_timestamp
            or utc_now_iso()
        )

        parse_iso_timestamp(anchor)

        records = {
            modality: ModalitySynchronizationRecord(
                modality=modality,
                available=False,
                limitation_codes=[
                    "modality_missing"
                ],
            )
            for modality in sorted(
                ALL_MODALITIES
            )
        }

        return SynchronizedMultimodalFrame(
            frame_id=(
                f"SYNC_{self._frame_counter:06d}_"
                f"{uuid.uuid4().hex[:8].upper()}"
            ),
            created_at=utc_now_iso(),
            anchor_timestamp=anchor,
            anchor_strategy=self.anchor_strategy,
            status=SynchronizationStatus.EMPTY,
            synchronization_records=records,
            missing_modalities=sorted(
                ALL_MODALITIES
            ),
            completeness_score=0.0,
            synchronization_score=0.0,
            warnings=[
                "no_modality_data_available"
            ],
            metadata={
                "synchronizer_version": (
                    SYNCHRONIZER_VERSION
                )
            },
        )

    def _calculate_temporal_span(
        self,
        timestamps: List[str],
    ) -> float:
        if len(timestamps) < 2:
            return 0.0

        parsed = [
            parse_iso_timestamp(value)
            for value in timestamps
        ]

        return (
            max(parsed) - min(parsed)
        ).total_seconds() * 1000.0

    def _determine_status(
        self,
        *,
        required_modalities: set[str],
        selected_modalities: set[str],
        stale_modalities: set[str],
        excluded_modalities: set[str],
    ) -> SynchronizationStatus:
        if not selected_modalities:
            return SynchronizationStatus.EMPTY

        missing_required = (
            required_modalities
            - selected_modalities
        )

        if (
            not missing_required
            and not stale_modalities
            and not excluded_modalities
        ):
            return SynchronizationStatus.COMPLETE

        if (
            len(missing_required)
            < len(required_modalities)
        ):
            return SynchronizationStatus.PARTIAL

        return SynchronizationStatus.DEGRADED

    # ========================================================
    # STATISTICS
    # ========================================================

    def _register_frame(
        self,
        frame: SynchronizedMultimodalFrame,
        elapsed_seconds: float,
    ) -> None:
        self.statistics.total_frames += 1
        self.statistics.cumulative_processing_seconds += (
            elapsed_seconds
        )

        self.statistics.last_frame_id = (
            frame.frame_id
        )
        self.statistics.last_status = (
            frame.status.value
        )
        self.statistics.last_error = None

        if (
            frame.status
            == SynchronizationStatus.COMPLETE
        ):
            self.statistics.complete_frames += 1

        elif (
            frame.status
            == SynchronizationStatus.PARTIAL
        ):
            self.statistics.partial_frames += 1

        elif (
            frame.status
            == SynchronizationStatus.DEGRADED
        ):
            self.statistics.degraded_frames += 1

        elif (
            frame.status
            == SynchronizationStatus.EMPTY
        ):
            self.statistics.empty_frames += 1

        self.statistics.total_stale_modalities += len(
            frame.stale_modalities
        )

        self.statistics.total_excluded_modalities += len(
            frame.excluded_modalities
        )

        self.statistics.total_missing_modalities += len(
            frame.missing_modalities
        )

        self._last_frame = frame

    # ========================================================
    # STATE AND DIAGNOSTICS
    # ========================================================

    def get_last_frame(
        self,
    ) -> Optional[SynchronizedMultimodalFrame]:
        return self._last_frame

    def get_cached_modalities(
        self,
    ) -> List[str]:
        return sorted(
            self._latest_values.keys()
        )

    def clear_cache(self) -> None:
        self._latest_values.clear()

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "anchor_strategy": (
                self.anchor_strategy.value
            ),
            "cached_modalities": (
                self.get_cached_modalities()
            ),
            "last_frame_available": (
                self._last_frame is not None
            ),
            "last_frame_id": (
                self._last_frame.frame_id
                if self._last_frame
                else None
            ),
            "last_status": (
                self._last_frame.status.value
                if self._last_frame
                else None
            ),
            "statistics": (
                self.statistics.to_dict()
            ),
        }


# ============================================================
# SELF-TEST
# ============================================================

def run_multimodal_synchronizer_self_test() -> bool:
    print("\n" + "=" * 72)
    print("NOONGIL-X | MULTIMODAL SYNCHRONIZER TEST")
    print("=" * 72)

    try:
        print("[1/7] Creating test settings...")

        settings = create_test_settings()

        synchronizer = MultimodalSynchronizer(
            settings,
            anchor_strategy=AnchorStrategy.LATEST,
        )

        print("[SUCCESS] Synchronizer initialized.")

        print("[2/7] Building processed modality outputs...")

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

        packets = simulator.generate_cycle()

        receipts = receiver.receive_batch(
            packets,
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

        if any(
            result is None
            for result in (
                vision_result,
                audio_result,
                spatial_result,
                motion_result,
                interaction_result,
            )
        ):
            raise AssertionError(
                "One or more modality results are missing."
            )

        print("[SUCCESS] Modality outputs built.")

        print("[3/7] Synchronizing modality results...")

        frame = synchronizer.synchronize_from_results(
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

        print("[SUCCESS] Multimodal frame produced.")

        print("[4/7] Validating synchronized frame...")

        frame.validate()

        expected_selected = {
            "vision",
            "audio",
            "spatial",
            "motion",
            "interaction",
            "wearable",
            "source_device",
        }

        if not expected_selected.issubset(
            set(frame.selected_modalities)
            | {"source_device"}
        ):
            raise AssertionError(
                "Expected modalities were not selected."
            )

        if frame.vision is None:
            raise AssertionError(
                "VisionData missing from frame."
            )

        if frame.audio is None:
            raise AssertionError(
                "AudioData missing from frame."
            )

        if frame.spatial is None:
            raise AssertionError(
                "SpatialData missing from frame."
            )

        if frame.motion is None:
            raise AssertionError(
                "MotionData missing from frame."
            )

        if frame.source_device is None:
            raise AssertionError(
                "SourceDevice missing from frame."
            )

        print("[SUCCESS] Synchronized frame is valid.")

        print("[5/7] Testing cached asynchronous update...")

        cached_frame = synchronizer.synchronize(
            interaction=(
                interaction_result.interaction_data
            ),
            include_cached_values=True,
            raise_on_error=True,
        )

        if cached_frame.vision is None:
            raise AssertionError(
                "Cached vision data was not reused."
            )

        print("[SUCCESS] Cached modality reuse works.")

        print("[6/7] Testing empty frame...")

        empty_synchronizer = (
            MultimodalSynchronizer(settings)
        )

        empty_frame = (
            empty_synchronizer.synchronize(
                include_cached_values=False,
                raise_on_error=True,
            )
        )

        if (
            empty_frame.status
            != SynchronizationStatus.EMPTY
        ):
            raise AssertionError(
                "Empty synchronization did not "
                "produce EMPTY status."
            )

        print("[SUCCESS] Empty frame behavior is correct.")

        print("[7/7] Checking diagnostics...")

        health = synchronizer.health_check()

        if not health["healthy"]:
            raise AssertionError(
                "Synchronizer health check failed."
            )

        if (
            health["statistics"]["total_frames"]
            != 2
        ):
            raise AssertionError(
                "Frame count is incorrect."
            )

        print("[SUCCESS] Diagnostics are correct.")

        print("\nSynchronized frame:")
        print(
            json.dumps(
                {
                    "frame_id": frame.frame_id,
                    "status": frame.status.value,
                    "anchor_timestamp": (
                        frame.anchor_timestamp
                    ),
                    "selected_modalities": (
                        frame.selected_modalities
                    ),
                    "missing_modalities": (
                        frame.missing_modalities
                    ),
                    "stale_modalities": (
                        frame.stale_modalities
                    ),
                    "excluded_modalities": (
                        frame.excluded_modalities
                    ),
                    "temporal_span_ms": (
                        frame.temporal_span_ms
                    ),
                    "maximum_offset_ms": (
                        frame.maximum_offset_ms
                    ),
                    "average_offset_ms": (
                        frame.average_offset_ms
                    ),
                    "completeness_score": (
                        frame.completeness_score
                    ),
                    "synchronization_score": (
                        frame.synchronization_score
                    ),
                    "warnings": frame.warnings,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nSynchronizer health:")
        print(
            json.dumps(
                health,
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n" + "=" * 72)
        print(
            "[PASSED] MULTIMODAL SYNCHRONIZER IS WORKING"
        )
        print("=" * 72)

        return True

    except Exception as error:
        print("\n" + "=" * 72)
        print(
            "[FAILED] MULTIMODAL SYNCHRONIZER TEST"
        )
        print("=" * 72)
        print(
            f"[ERROR] {type(error).__name__}: {error}"
        )

        return False


if __name__ == "__main__":
    if not run_multimodal_synchronizer_self_test():
        raise SystemExit(1)
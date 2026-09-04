"""
============================================================
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : Motion Input Processor
File    : layer1/modalities/motion_input.py
============================================================

Purpose
-------
Consumes normalized motion packets from MultimodalReceiver and
produces validated Layer 1 MotionData objects.

Supported sensor packets
------------------------
- accelerometer
- gyroscope
- magnetometer
- linear acceleration
- generic IMU packets

Responsibilities
----------------
1. Validate motion packet structure
2. Normalize three-axis sensor vectors
3. Combine multiple motion sensors from one acquisition cycle
4. Calculate signal-level motion intensity
5. Calculate orientation-change magnitude
6. Estimate sampling continuity
7. Detect sensor saturation
8. Track latest sensor samples
9. Build MotionData for the final sensor packet
10. Log processing and diagnostics

Architectural Boundary
----------------------
This module does NOT classify:
- walking;
- running;
- standing;
- falling;
- gestures;
- activities;
- hazards.

Those semantic interpretations belong to Layer 2 or higher.

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import json
import math
import statistics
import time

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from layer1.acquisition.multimodal_receiver import (
    MultimodalReceiver,
    ReceivedSensorPacket,
)
from layer1.config.settings import (
    Layer1Settings,
    create_default_settings,
    create_test_settings,
)
from layer1.schemas.sensor_packet import (
    ModalityMetadata,
    ModalityStatus,
    MotionData,
    Vector3,
)
from layer1.utils.logger import (
    PipelineTimer,
    get_logger,
    log_exception,
    log_sensor_event,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class MotionInputError(Exception):
    """Base exception for motion input processing."""


class MotionPacketValidationError(MotionInputError):
    """Raised when a motion packet is invalid."""


class MotionProcessingError(MotionInputError):
    """Raised when motion samples cannot be processed."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class MotionProcessingResult:
    """
    Result returned after processing motion packets.
    """

    success: bool
    motion_data: Optional[MotionData] = None
    packet_ids: List[str] = field(default_factory=list)
    sensor_types: List[str] = field(default_factory=list)
    processing_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MotionProcessorStatistics:
    """
    Runtime statistics for MotionInputProcessor.
    """

    total_packets_received: int = 0
    total_groups_processed: int = 0
    total_failed: int = 0

    accelerometer_packets: int = 0
    gyroscope_packets: int = 0
    magnetometer_packets: int = 0
    linear_acceleration_packets: int = 0
    generic_imu_packets: int = 0

    total_saturated_groups: int = 0
    total_incomplete_groups: int = 0

    cumulative_processing_seconds: float = 0.0
    last_packet_ids: List[str] = field(default_factory=list)
    last_error: Optional[str] = None

    @property
    def average_processing_seconds(self) -> float:
        if self.total_groups_processed == 0:
            return 0.0

        return (
            self.cumulative_processing_seconds
            / self.total_groups_processed
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["average_processing_seconds"] = (
            self.average_processing_seconds
        )
        return payload


@dataclass
class SensorSampleState:
    """
    Latest accepted state for one motion sensor.
    """

    sensor_type: str
    vector: Vector3
    source_timestamp: str
    arrival_timestamp: str
    sequence_number: int
    sampling_rate_hz: Optional[float]
    packet_id: str


# ============================================================
# HELPERS
# ============================================================

def parse_iso_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise MotionPacketValidationError(
            "Timestamp must be a non-empty string."
        )

    normalized = value.strip().replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise MotionPacketValidationError(
            f"Invalid ISO timestamp: {value!r}"
        ) from error

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def require_finite_float(
    value: Any,
    field_name: str,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise MotionPacketValidationError(
            f"{field_name} must be numeric."
        ) from error

    if not math.isfinite(parsed):
        raise MotionPacketValidationError(
            f"{field_name} must be finite."
        )

    return parsed


def vector_magnitude(vector: Vector3) -> float:
    return math.sqrt(
        vector.x ** 2
        + vector.y ** 2
        + vector.z ** 2
    )


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


# ============================================================
# MOTION INPUT PROCESSOR
# ============================================================

class MotionInputProcessor:
    """
    Convert receiver motion packets into validated MotionData.

    Because the receiver routes accelerometer, gyroscope, and
    magnetometer packets into one motion queue, this processor can
    combine several packets into one synchronized motion record.
    """

    SENSOR_ALIASES: Dict[str, str] = {
        "accelerometer": "accelerometer",
        "acceleration": "accelerometer",
        "accelerometer_sensor": "accelerometer",

        "gyroscope": "gyroscope",
        "gyro": "gyroscope",
        "gyroscope_sensor": "gyroscope",

        "magnetometer": "magnetometer",
        "magnetic_field": "magnetometer",
        "compass_sensor": "magnetometer",

        "linear_acceleration": "linear_acceleration",
        "linear_accelerometer": "linear_acceleration",

        "imu": "imu",
        "inertial_measurement_unit": "imu",
    }

    DEFAULT_UNITS: Dict[str, str] = {
        "accelerometer": "m/s^2",
        "gyroscope": "rad/s",
        "magnetometer": "microtesla",
        "linear_acceleration": "m/s^2",
    }

    def __init__(
        self,
        settings: Optional[Layer1Settings] = None,
    ) -> None:
        self.settings = settings or create_default_settings()
        self.settings.validate()

        self.logger = get_logger("modalities.motion_input")
        self.statistics = MotionProcessorStatistics()

        self._latest_samples: Dict[str, SensorSampleState] = {}
        self._sequence_history: Dict[str, List[int]] = {}
        self._timestamp_history: Dict[str, List[datetime]] = {}

    # ========================================================
    # PUBLIC API
    # ========================================================

    def process_packet(
        self,
        packet: ReceivedSensorPacket,
        *,
        raise_on_error: Optional[bool] = None,
    ) -> MotionProcessingResult:
        """
        Process one motion packet into MotionData.

        Previously stored latest samples may be included to build a
        fuller motion record.
        """

        return self.process_packets(
            [packet],
            include_latest_samples=True,
            raise_on_error=raise_on_error,
        )

    def process_packets(
        self,
        packets: Iterable[ReceivedSensorPacket],
        *,
        include_latest_samples: bool = True,
        raise_on_error: Optional[bool] = None,
    ) -> MotionProcessingResult:
        """
        Process and combine multiple motion packets.
        """

        packet_list = list(packets)

        should_raise = (
            self.settings.runtime.fail_fast
            if raise_on_error is None
            else raise_on_error
        )

        self.statistics.total_packets_received += len(
            packet_list
        )

        started = time.perf_counter()

        try:
            if not packet_list:
                raise MotionPacketValidationError(
                    "At least one motion packet is required."
                )

            packet_ids = [
                packet.packet_id
                for packet in packet_list
            ]

            with PipelineTimer(
                "motion_input.process_packets",
                logger=self.logger,
                metadata={
                    "packet_ids": packet_ids,
                    "packet_count": len(packet_list),
                },
            ):
                parsed_samples: Dict[
                    str,
                    SensorSampleState,
                ] = {}

                warnings: List[str] = []

                for packet in packet_list:
                    self._validate_packet(packet)

                    packet_samples = (
                        self._extract_sensor_samples(packet)
                    )

                    for sensor_type, sample in (
                        packet_samples.items()
                    ):
                        parsed_samples[sensor_type] = sample
                        self._store_sample(sample)
                        self._increment_sensor_statistic(
                            sensor_type
                        )

                combined_samples = dict(parsed_samples)

                if include_latest_samples:
                    for sensor_type, sample in (
                        self._latest_samples.items()
                    ):
                        combined_samples.setdefault(
                            sensor_type,
                            sample,
                        )

                motion_data = self._build_motion_data(
                    packet_list=packet_list,
                    samples=combined_samples,
                    warnings=warnings,
                )

                motion_data.validate()

                elapsed = time.perf_counter() - started

                if motion_data.metadata.limitations:
                    if "sensor_saturation_detected" in (
                        motion_data.metadata.limitations
                    ):
                        self.statistics.total_saturated_groups += 1

                    if "incomplete_motion_sensor_group" in (
                        motion_data.metadata.limitations
                    ):
                        self.statistics.total_incomplete_groups += 1

                self.statistics.total_groups_processed += 1
                self.statistics.cumulative_processing_seconds += (
                    elapsed
                )
                self.statistics.last_packet_ids = packet_ids
                self.statistics.last_error = None

                log_sensor_event(
                    modality="motion",
                    event="Motion sensor group processed",
                    device_id=packet_list[-1].device_id,
                    sensor_type="combined_motion",
                    packet_id=packet_list[-1].packet_id,
                    sequence_number=(
                        packet_list[-1].sequence_number
                    ),
                    details={
                        "packet_ids": packet_ids,
                        "sensor_types": sorted(
                            combined_samples.keys()
                        ),
                        "motion_intensity": (
                            motion_data.motion_intensity
                        ),
                        "orientation_change_score": (
                            motion_data
                            .orientation_change_score
                        ),
                        "sampling_continuity_score": (
                            motion_data
                            .sampling_continuity_score
                        ),
                        "sensor_saturation_score": (
                            motion_data
                            .sensor_saturation_score
                        ),
                        "limitations": (
                            motion_data.metadata.limitations
                        ),
                        "processing_seconds": round(
                            elapsed,
                            6,
                        ),
                    },
                )

                return MotionProcessingResult(
                    success=True,
                    motion_data=motion_data,
                    packet_ids=packet_ids,
                    sensor_types=sorted(
                        combined_samples.keys()
                    ),
                    processing_seconds=elapsed,
                    warnings=warnings,
                )

        except Exception as error:
            elapsed = time.perf_counter() - started

            self.statistics.total_failed += 1
            self.statistics.last_packet_ids = [
                getattr(packet, "packet_id", "")
                for packet in packet_list
            ]
            self.statistics.last_error = (
                f"{type(error).__name__}: {error}"
            )

            log_exception(
                self.logger,
                "Motion packet processing failed",
                error=error,
                details={
                    "packet_ids": self.statistics.last_packet_ids,
                },
            )

            if should_raise:
                raise

            return MotionProcessingResult(
                success=False,
                packet_ids=self.statistics.last_packet_ids,
                processing_seconds=elapsed,
                error=f"{type(error).__name__}: {error}",
            )

    def process_receiver_queue(
        self,
        receiver: MultimodalReceiver,
        *,
        maximum_items: Optional[int] = None,
        raise_on_error: Optional[bool] = None,
    ) -> Optional[MotionProcessingResult]:
        """
        Drain and combine motion packets from the receiver.
        """

        packets = receiver.drain(
            "motion",
            maximum_items=maximum_items,
        )

        if not packets:
            return None

        return self.process_packets(
            packets,
            include_latest_samples=True,
            raise_on_error=raise_on_error,
        )

    def process_latest_from_receiver(
        self,
        receiver: MultimodalReceiver,
        *,
        remove: bool = True,
        raise_on_error: Optional[bool] = None,
    ) -> Optional[MotionProcessingResult]:
        """
        Process the newest motion packet.
        """

        packet = receiver.get_latest(
            "motion",
            remove=remove,
        )

        if packet is None:
            return None

        return self.process_packet(
            packet,
            raise_on_error=raise_on_error,
        )

    # ========================================================
    # VALIDATION AND EXTRACTION
    # ========================================================

    def _validate_packet(
        self,
        packet: ReceivedSensorPacket,
    ) -> None:
        if not isinstance(packet, ReceivedSensorPacket):
            raise MotionPacketValidationError(
                "packet must be ReceivedSensorPacket."
            )

        packet.validate()

        if packet.modality != "motion":
            raise MotionPacketValidationError(
                "MotionInputProcessor accepts only "
                "modality='motion'."
            )

        if not isinstance(packet.payload, dict):
            raise MotionPacketValidationError(
                "Motion packet payload must be a dictionary."
            )

    def _normalize_sensor_type(
        self,
        sensor_type: Optional[str],
    ) -> str:
        if not sensor_type:
            return "imu"

        normalized = (
            sensor_type.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        return self.SENSOR_ALIASES.get(
            normalized,
            normalized,
        )

    def _extract_sensor_samples(
        self,
        packet: ReceivedSensorPacket,
    ) -> Dict[str, SensorSampleState]:
        """
        Extract one or more sensor vectors from a packet.
        """

        payload = packet.payload
        sensor_type = self._normalize_sensor_type(
            packet.sensor_type
        )

        samples: Dict[str, SensorSampleState] = {}

        if sensor_type == "imu":
            nested_mapping = {
                "accelerometer": payload.get("accelerometer"),
                "gyroscope": payload.get("gyroscope"),
                "magnetometer": payload.get("magnetometer"),
                "linear_acceleration": payload.get(
                    "linear_acceleration"
                ),
            }

            for nested_type, nested_payload in (
                nested_mapping.items()
            ):
                if nested_payload is None:
                    continue

                if not isinstance(
                    nested_payload,
                    Mapping,
                ):
                    raise MotionPacketValidationError(
                        f"{nested_type} must be a dictionary."
                    )

                vector = self._build_vector(
                    nested_payload,
                    sensor_type=nested_type,
                )

                samples[nested_type] = SensorSampleState(
                    sensor_type=nested_type,
                    vector=vector,
                    source_timestamp=packet.source_timestamp,
                    arrival_timestamp=packet.arrival_timestamp,
                    sequence_number=packet.sequence_number,
                    sampling_rate_hz=packet.sampling_rate_hz,
                    packet_id=packet.packet_id,
                )

            if not samples and all(
                key in payload
                for key in ("x", "y", "z")
            ):
                vector = self._build_vector(
                    payload,
                    sensor_type="accelerometer",
                )

                samples["accelerometer"] = SensorSampleState(
                    sensor_type="accelerometer",
                    vector=vector,
                    source_timestamp=packet.source_timestamp,
                    arrival_timestamp=packet.arrival_timestamp,
                    sequence_number=packet.sequence_number,
                    sampling_rate_hz=packet.sampling_rate_hz,
                    packet_id=packet.packet_id,
                )

            if not samples:
                raise MotionPacketValidationError(
                    "Generic IMU packet contains no supported "
                    "sensor vectors."
                )

            return samples

        if sensor_type not in {
            "accelerometer",
            "gyroscope",
            "magnetometer",
            "linear_acceleration",
        }:
            raise MotionPacketValidationError(
                f"Unsupported motion sensor type: "
                f"{sensor_type!r}"
            )

        vector = self._build_vector(
            payload,
            sensor_type=sensor_type,
        )

        samples[sensor_type] = SensorSampleState(
            sensor_type=sensor_type,
            vector=vector,
            source_timestamp=packet.source_timestamp,
            arrival_timestamp=packet.arrival_timestamp,
            sequence_number=packet.sequence_number,
            sampling_rate_hz=packet.sampling_rate_hz,
            packet_id=packet.packet_id,
        )

        return samples

    def _build_vector(
        self,
        payload: Mapping[str, Any],
        *,
        sensor_type: str,
    ) -> Vector3:
        for field_name in ("x", "y", "z"):
            if field_name not in payload:
                raise MotionPacketValidationError(
                    f"{sensor_type} packet is missing "
                    f"{field_name!r}."
                )

        unit = str(
            payload.get(
                "unit",
                self.DEFAULT_UNITS.get(
                    sensor_type,
                    "unknown",
                ),
            )
        )

        vector = Vector3(
            x=require_finite_float(payload["x"], "x"),
            y=require_finite_float(payload["y"], "y"),
            z=require_finite_float(payload["z"], "z"),
            unit=unit,
        )

        vector.validate()
        return vector

    # ========================================================
    # SAMPLE HISTORY
    # ========================================================

    def _store_sample(
        self,
        sample: SensorSampleState,
    ) -> None:
        self._latest_samples[sample.sensor_type] = sample

        sequences = self._sequence_history.setdefault(
            sample.sensor_type,
            [],
        )
        sequences.append(sample.sequence_number)

        timestamps = self._timestamp_history.setdefault(
            sample.sensor_type,
            [],
        )
        timestamps.append(
            parse_iso_timestamp(sample.source_timestamp)
        )

        maximum_history = max(
            5,
            int(
                self.settings.motion
                .motion_buffer_seconds
                * self.settings.motion
                .maximum_sampling_rate_hz
            ),
        )

        if len(sequences) > maximum_history:
            del sequences[:-maximum_history]

        if len(timestamps) > maximum_history:
            del timestamps[:-maximum_history]

    def _increment_sensor_statistic(
        self,
        sensor_type: str,
    ) -> None:
        if sensor_type == "accelerometer":
            self.statistics.accelerometer_packets += 1
        elif sensor_type == "gyroscope":
            self.statistics.gyroscope_packets += 1
        elif sensor_type == "magnetometer":
            self.statistics.magnetometer_packets += 1
        elif sensor_type == "linear_acceleration":
            self.statistics.linear_acceleration_packets += 1
        else:
            self.statistics.generic_imu_packets += 1

    # ========================================================
    # MOTION DATA BUILDING
    # ========================================================

    def _build_motion_data(
        self,
        *,
        packet_list: List[ReceivedSensorPacket],
        samples: Mapping[str, SensorSampleState],
        warnings: List[str],
    ) -> MotionData:
        latest_packet = max(
            packet_list,
            key=lambda packet: parse_iso_timestamp(
                packet.source_timestamp
            ),
        )

        accelerometer_sample = samples.get(
            "accelerometer"
        )
        gyroscope_sample = samples.get("gyroscope")
        magnetometer_sample = samples.get(
            "magnetometer"
        )
        linear_sample = samples.get(
            "linear_acceleration"
        )

        limitations: List[str] = []

        required_available = (
            accelerometer_sample is not None
            and gyroscope_sample is not None
        )

        if not required_available:
            limitations.append(
                "incomplete_motion_sensor_group"
            )
            warnings.append(
                "accelerometer_and_gyroscope_not_both_available"
            )

        motion_intensity = self._calculate_motion_intensity(
            accelerometer=(
                accelerometer_sample.vector
                if accelerometer_sample
                else None
            ),
            linear_acceleration=(
                linear_sample.vector
                if linear_sample
                else None
            ),
        )

        orientation_change = (
            self._calculate_orientation_change(
                gyroscope=(
                    gyroscope_sample.vector
                    if gyroscope_sample
                    else None
                ),
                sampling_rate_hz=(
                    gyroscope_sample.sampling_rate_hz
                    if gyroscope_sample
                    else None
                ),
            )
        )

        continuity = self._calculate_sampling_continuity(
            samples
        )

        saturation = self._calculate_saturation_score(
            samples
        )

        if saturation > 0.0:
            limitations.append(
                "sensor_saturation_detected"
            )

        stale_sensors = self._detect_stale_sensors(
            samples
        )

        if stale_sensors:
            limitations.append("stale_motion_data")
            warnings.append(
                "stale_sensors:" + ",".join(
                    sorted(stale_sensors)
                )
            )

        preprocessing_steps = [
            "packet_validation",
            "axis_normalization",
            "sensor_group_assembly",
            "motion_intensity_calculation",
            "orientation_change_calculation",
            "sampling_continuity_evaluation",
            "saturation_evaluation",
        ]

        source_timestamps = [
            sample.source_timestamp
            for sample in samples.values()
        ]

        metadata = ModalityMetadata(
            modality="motion",
            status=ModalityStatus.OBSERVED,
            source_timestamp=max(
                source_timestamps,
                key=lambda value: parse_iso_timestamp(value),
            ),
            arrival_timestamp=latest_packet.arrival_timestamp,
            sequence_number=latest_packet.sequence_number,
            sampling_rate_hz=self._representative_sampling_rate(
                samples
            ),
            latency_ms=max(
                packet.latency_ms
                for packet in packet_list
            ),
            source_device_id=latest_packet.device_id,
            data_reference=latest_packet.packet_id,
            preprocessing_steps=preprocessing_steps,
            limitations=limitations,
            metadata={
                "sensor_types": sorted(samples.keys()),
                "packet_ids": [
                    packet.packet_id
                    for packet in packet_list
                ],
                "simulated": any(
                    bool(
                        packet.metadata.get(
                            "simulated",
                            False,
                        )
                    )
                    for packet in packet_list
                ),
                "scenario": latest_packet.metadata.get(
                    "scenario"
                ),
                "stale_sensors": stale_sensors,
                "sample_count": len(samples),
            },
        )

        return MotionData(
            metadata=metadata,
            accelerometer=(
                accelerometer_sample.vector
                if accelerometer_sample
                else None
            ),
            gyroscope=(
                gyroscope_sample.vector
                if gyroscope_sample
                else None
            ),
            magnetometer=(
                magnetometer_sample.vector
                if magnetometer_sample
                else None
            ),
            linear_acceleration=(
                linear_sample.vector
                if linear_sample
                else None
            ),
            motion_intensity=motion_intensity,
            orientation_change_score=orientation_change,
            sampling_continuity_score=continuity,
            sensor_saturation_score=saturation,
        )

    def _calculate_motion_intensity(
        self,
        *,
        accelerometer: Optional[Vector3],
        linear_acceleration: Optional[Vector3],
    ) -> float:
        """
        Calculate bounded motion magnitude.

        Linear acceleration is preferred. When only the raw
        accelerometer exists, gravity magnitude is removed.
        """

        if linear_acceleration is not None:
            magnitude = vector_magnitude(
                linear_acceleration
            )
        elif accelerometer is not None:
            magnitude = abs(
                vector_magnitude(accelerometer) - 9.80665
            )
        else:
            return 0.0

        reference = 12.0

        return round(
            clamp(magnitude / reference, 0.0, 1.0),
            6,
        )

    def _calculate_orientation_change(
        self,
        *,
        gyroscope: Optional[Vector3],
        sampling_rate_hz: Optional[float],
    ) -> float:
        if gyroscope is None:
            return 0.0

        angular_speed = vector_magnitude(gyroscope)

        sample_interval = (
            1.0 / sampling_rate_hz
            if sampling_rate_hz
            and sampling_rate_hz > 0
            else 1.0
            / self.settings.motion.default_sampling_rate_hz
        )

        angular_change = angular_speed * sample_interval

        reference_radians = math.pi / 4.0

        return round(
            clamp(
                angular_change / reference_radians,
                0.0,
                1.0,
            ),
            6,
        )

    def _calculate_sampling_continuity(
        self,
        samples: Mapping[str, SensorSampleState],
    ) -> float:
        scores: List[float] = []

        for sensor_type in samples:
            sequences = self._sequence_history.get(
                sensor_type,
                [],
            )

            if len(sequences) < 2:
                scores.append(1.0)
                continue

            expected_transitions = len(sequences) - 1
            continuous_transitions = sum(
                1
                for previous, current in zip(
                    sequences,
                    sequences[1:],
                )
                if current - previous in {0, 1}
            )

            scores.append(
                continuous_transitions
                / expected_transitions
            )

        if not scores:
            return 0.0

        return round(
            clamp(
                statistics.fmean(scores),
                0.0,
                1.0,
            ),
            6,
        )

    def _calculate_saturation_score(
        self,
        samples: Mapping[str, SensorSampleState],
    ) -> float:
        saturation_values: List[float] = []

        limits = {
            "accelerometer": (
                self.settings.motion
                .accelerometer_maximum_absolute_value
            ),
            "linear_acceleration": (
                self.settings.motion
                .accelerometer_maximum_absolute_value
            ),
            "gyroscope": (
                self.settings.motion
                .gyroscope_maximum_absolute_value
            ),
            "magnetometer": (
                self.settings.motion
                .magnetometer_maximum_absolute_value
            ),
        }

        for sensor_type, sample in samples.items():
            limit = limits.get(sensor_type)

            if limit is None or limit <= 0:
                continue

            maximum_axis = max(
                abs(sample.vector.x),
                abs(sample.vector.y),
                abs(sample.vector.z),
            )

            if maximum_axis <= limit:
                saturation_values.append(0.0)
            else:
                saturation_values.append(
                    clamp(
                        (maximum_axis - limit) / limit,
                        0.0,
                        1.0,
                    )
                )

        if not saturation_values:
            return 0.0

        return round(
            max(saturation_values),
            6,
        )

    def _detect_stale_sensors(
        self,
        samples: Mapping[str, SensorSampleState],
    ) -> List[str]:
        now = datetime.now(timezone.utc)
        stale: List[str] = []

        for sensor_type, sample in samples.items():
            age_ms = (
                now
                - parse_iso_timestamp(
                    sample.source_timestamp
                )
            ).total_seconds() * 1000.0

            if age_ms > (
                self.settings.motion.maximum_motion_age_ms
            ):
                stale.append(sensor_type)

        return sorted(stale)

    def _representative_sampling_rate(
        self,
        samples: Mapping[str, SensorSampleState],
    ) -> Optional[float]:
        rates = [
            float(sample.sampling_rate_hz)
            for sample in samples.values()
            if sample.sampling_rate_hz is not None
            and sample.sampling_rate_hz > 0
        ]

        if not rates:
            return None

        return statistics.fmean(rates)

    # ========================================================
    # STATE AND DIAGNOSTICS
    # ========================================================

    def get_latest_samples(
        self,
    ) -> Dict[str, SensorSampleState]:
        return dict(self._latest_samples)

    def clear_history(self) -> None:
        self._latest_samples.clear()
        self._sequence_history.clear()
        self._timestamp_history.clear()

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "motion_enabled": self.settings.motion.enabled,
            "accelerometer_enabled": (
                self.settings.motion.enable_accelerometer
            ),
            "gyroscope_enabled": (
                self.settings.motion.enable_gyroscope
            ),
            "magnetometer_enabled": (
                self.settings.motion.enable_magnetometer
            ),
            "latest_sensor_types": sorted(
                self._latest_samples.keys()
            ),
            "statistics": self.statistics.to_dict(),
        }


# ============================================================
# SELF-TEST
# ============================================================

def run_motion_input_self_test() -> bool:
    print("\n" + "=" * 72)
    print("NOONGIL-X | LAYER 1 MOTION INPUT TEST")
    print("=" * 72)

    try:
        print("[1/6] Creating test settings...")

        settings = create_test_settings()
        processor = MotionInputProcessor(settings)

        print("[SUCCESS] Motion processor initialized.")

        print("[2/6] Creating receiver and simulator...")

        from layer1.acquisition.phone_sensor_simulator import (
            PhoneSensorSimulator,
            PhoneSimulatorConfig,
            SimulationScenario,
        )

        receiver = MultimodalReceiver(settings)
        receiver.start()

        simulator = PhoneSensorSimulator(
            PhoneSimulatorConfig(
                scenario=SimulationScenario.NAVIGATION,
                random_seed=42,
            )
        )

        packets = simulator.generate_cycle()

        receipts = receiver.receive_batch(
            packets,
            raise_on_error=True,
        )

        if not all(receipt.accepted for receipt in receipts):
            raise AssertionError(
                "Simulator packets were not accepted."
            )

        print("[SUCCESS] Simulator packets routed.")

        print("[3/6] Processing motion queue...")

        result = processor.process_receiver_queue(
            receiver,
            raise_on_error=True,
        )

        if result is None:
            raise AssertionError(
                "No motion packets were available."
            )

        if not result.success:
            raise AssertionError(
                f"Motion processing failed: {result.error}"
            )

        if result.motion_data is None:
            raise AssertionError(
                "MotionData was not produced."
            )

        print("[SUCCESS] Motion packets processed.")

        print("[4/6] Validating MotionData...")

        motion = result.motion_data
        motion.validate()

        if motion.metadata.modality != "motion":
            raise AssertionError(
                "MotionData modality is incorrect."
            )

        if motion.accelerometer is None:
            raise AssertionError(
                "Accelerometer data was not produced."
            )

        if motion.gyroscope is None:
            raise AssertionError(
                "Gyroscope data was not produced."
            )

        if motion.magnetometer is None:
            raise AssertionError(
                "Magnetometer data was not produced."
            )

        if set(result.sensor_types) != {
            "accelerometer",
            "gyroscope",
            "magnetometer",
        }:
            raise AssertionError(
                "Unexpected motion sensor types."
            )

        print("[SUCCESS] MotionData is valid.")

        print("[5/6] Testing invalid modality rejection...")

        vision_packet = receiver.get_latest(
            "vision",
            remove=False,
        )

        if vision_packet is None:
            raise AssertionError(
                "Vision packet missing from receiver."
            )

        invalid_result = processor.process_packet(
            vision_packet,
            raise_on_error=False,
        )

        if invalid_result.success:
            raise AssertionError(
                "Non-motion packet was incorrectly accepted."
            )

        print("[SUCCESS] Invalid modality was rejected.")

        print("[6/6] Checking diagnostics...")

        health = processor.health_check()

        if not health["healthy"]:
            raise AssertionError(
                "Motion processor health check failed."
            )

        if health["statistics"]["total_groups_processed"] != 1:
            raise AssertionError(
                "Processed group count is incorrect."
            )

        if health["statistics"]["total_failed"] != 1:
            raise AssertionError(
                "Failed count is incorrect."
            )

        print("[SUCCESS] Diagnostics are correct.")

        print("\nMotionData:")
        print(
            json.dumps(
                result.motion_data.metadata.metadata
                | {
                    "accelerometer": (
                        asdict(result.motion_data.accelerometer)
                        if result.motion_data.accelerometer
                        else None
                    ),
                    "gyroscope": (
                        asdict(result.motion_data.gyroscope)
                        if result.motion_data.gyroscope
                        else None
                    ),
                    "magnetometer": (
                        asdict(result.motion_data.magnetometer)
                        if result.motion_data.magnetometer
                        else None
                    ),
                    "motion_intensity": (
                        result.motion_data.motion_intensity
                    ),
                    "orientation_change_score": (
                        result.motion_data
                        .orientation_change_score
                    ),
                    "sampling_continuity_score": (
                        result.motion_data
                        .sampling_continuity_score
                    ),
                    "sensor_saturation_score": (
                        result.motion_data
                        .sensor_saturation_score
                    ),
                    "limitations": (
                        result.motion_data
                        .metadata.limitations
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nProcessor health:")
        print(
            json.dumps(
                health,
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n" + "=" * 72)
        print("[PASSED] LAYER 1 MOTION INPUT IS WORKING")
        print("=" * 72)

        return True

    except Exception as error:
        print("\n" + "=" * 72)
        print("[FAILED] LAYER 1 MOTION INPUT TEST")
        print("=" * 72)
        print(f"[ERROR] {type(error).__name__}: {error}")

        return False


if __name__ == "__main__":
    if not run_motion_input_self_test():
        raise SystemExit(1)
"""
============================================================
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : Spatial Input Processor
File    : layer1/modalities/spatial_input.py
============================================================

Purpose
-------
Consumes normalized spatial packets from MultimodalReceiver and
produces validated Layer 1 SpatialData objects.

Responsibilities
----------------
1. Validate GPS and compass packet structure
2. Normalize latitude, longitude, altitude, heading, and speed
3. Evaluate horizontal accuracy
4. Evaluate location and heading freshness
5. Detect degraded or incomplete spatial measurements
6. Build SpatialData for the final Multimodal Sensor Packet
7. Log processing, quality, and errors
8. Provide diagnostics and a standalone self-test

Architectural Boundary
----------------------
This module does NOT perform:
- route planning;
- indoor/outdoor classification;
- destination inference;
- navigation decisions;
- obstacle reasoning;
- semantic scene understanding;
- LLM processing.

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

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

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
    SpatialData,
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

class SpatialInputError(Exception):
    """Base exception for spatial input processing."""


class SpatialPacketValidationError(SpatialInputError):
    """Raised when a spatial packet is invalid."""


class SpatialProcessingError(SpatialInputError):
    """Raised when spatial data cannot be processed."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SpatialProcessingResult:
    """
    Result returned after processing one spatial packet.
    """

    success: bool
    spatial_data: Optional[SpatialData] = None
    packet_id: Optional[str] = None
    processing_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpatialProcessorStatistics:
    """
    Runtime statistics for SpatialInputProcessor.
    """

    total_received: int = 0
    total_processed: int = 0
    total_failed: int = 0
    total_degraded: int = 0
    total_stale: int = 0
    cumulative_processing_seconds: float = 0.0
    last_packet_id: Optional[str] = None
    last_error: Optional[str] = None

    @property
    def average_processing_seconds(self) -> float:
        if self.total_processed == 0:
            return 0.0

        return (
            self.cumulative_processing_seconds
            / self.total_processed
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

def parse_iso_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise SpatialPacketValidationError(
            "Timestamp must be a non-empty string."
        )

    normalized = value.strip().replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise SpatialPacketValidationError(
            f"Invalid ISO 8601 timestamp: {value!r}"
        ) from error

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def require_float(
    value: Any,
    field_name: str,
    *,
    allow_none: bool = False,
) -> Optional[float]:
    if value is None and allow_none:
        return None

    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise SpatialPacketValidationError(
            f"{field_name} must be numeric."
        ) from error

    if not math.isfinite(parsed):
        raise SpatialPacketValidationError(
            f"{field_name} must be finite."
        )

    return parsed


def require_non_negative(
    value: Any,
    field_name: str,
    *,
    allow_none: bool = False,
) -> Optional[float]:
    parsed = require_float(
        value,
        field_name,
        allow_none=allow_none,
    )

    if parsed is None:
        return None

    if parsed < 0:
        raise SpatialPacketValidationError(
            f"{field_name} cannot be negative."
        )

    return parsed


# ============================================================
# SPATIAL INPUT PROCESSOR
# ============================================================

class SpatialInputProcessor:
    """
    Convert receiver spatial packets into validated SpatialData.
    """

    def __init__(
        self,
        settings: Optional[Layer1Settings] = None,
    ) -> None:
        self.settings = settings or create_default_settings()
        self.settings.validate()

        self.logger = get_logger("modalities.spatial_input")
        self.statistics = SpatialProcessorStatistics()

        self._last_valid_spatial: Optional[SpatialData] = None

    # ========================================================
    # PUBLIC API
    # ========================================================

    def process_packet(
        self,
        packet: ReceivedSensorPacket,
        *,
        raise_on_error: Optional[bool] = None,
    ) -> SpatialProcessingResult:
        """
        Process one normalized spatial packet.
        """

        should_raise = (
            self.settings.runtime.fail_fast
            if raise_on_error is None
            else raise_on_error
        )

        self.statistics.total_received += 1
        started = time.perf_counter()

        try:
            with PipelineTimer(
                "spatial_input.process_packet",
                logger=self.logger,
                metadata={
                    "packet_id": packet.packet_id,
                    "device_id": packet.device_id,
                },
            ):
                self._validate_packet(packet)

                warnings: List[str] = []
                spatial_data = self._build_spatial_data(
                    packet,
                    warnings=warnings,
                )
                spatial_data.validate()

                elapsed = time.perf_counter() - started

                if spatial_data.metadata.limitations:
                    self.statistics.total_degraded += 1

                if "stale_spatial_data" in (
                    spatial_data.metadata.limitations
                ):
                    self.statistics.total_stale += 1

                self.statistics.total_processed += 1
                self.statistics.cumulative_processing_seconds += (
                    elapsed
                )
                self.statistics.last_packet_id = packet.packet_id
                self.statistics.last_error = None

                if (
                    spatial_data.latitude is not None
                    or spatial_data.longitude is not None
                    or spatial_data.heading_degrees is not None
                ):
                    self._last_valid_spatial = spatial_data

                log_sensor_event(
                    modality="spatial",
                    event="Spatial packet processed",
                    device_id=packet.device_id,
                    sensor_type=packet.sensor_type,
                    packet_id=packet.packet_id,
                    sequence_number=packet.sequence_number,
                    details={
                        "latitude": spatial_data.latitude,
                        "longitude": spatial_data.longitude,
                        "heading_degrees": (
                            spatial_data.heading_degrees
                        ),
                        "horizontal_accuracy_meters": (
                            spatial_data
                            .horizontal_accuracy_meters
                        ),
                        "speed_meters_per_second": (
                            spatial_data
                            .speed_meters_per_second
                        ),
                        "limitations": (
                            spatial_data.metadata.limitations
                        ),
                        "processing_seconds": round(
                            elapsed,
                            6,
                        ),
                    },
                )

                return SpatialProcessingResult(
                    success=True,
                    spatial_data=spatial_data,
                    packet_id=packet.packet_id,
                    processing_seconds=elapsed,
                    warnings=warnings,
                )

        except Exception as error:
            elapsed = time.perf_counter() - started

            self.statistics.total_failed += 1
            self.statistics.last_packet_id = getattr(
                packet,
                "packet_id",
                None,
            )
            self.statistics.last_error = (
                f"{type(error).__name__}: {error}"
            )

            log_exception(
                self.logger,
                "Spatial packet processing failed",
                error=error,
                details={
                    "packet_id": getattr(
                        packet,
                        "packet_id",
                        None,
                    ),
                    "device_id": getattr(
                        packet,
                        "device_id",
                        None,
                    ),
                },
            )

            if should_raise:
                raise

            return SpatialProcessingResult(
                success=False,
                packet_id=getattr(
                    packet,
                    "packet_id",
                    None,
                ),
                processing_seconds=elapsed,
                error=f"{type(error).__name__}: {error}",
            )

    def process_receiver_queue(
        self,
        receiver: MultimodalReceiver,
        *,
        maximum_items: Optional[int] = None,
        raise_on_error: Optional[bool] = None,
    ) -> List[SpatialProcessingResult]:
        """
        Drain and process spatial packets from MultimodalReceiver.
        """

        packets = receiver.drain(
            "spatial",
            maximum_items=maximum_items,
        )

        return [
            self.process_packet(
                packet,
                raise_on_error=raise_on_error,
            )
            for packet in packets
        ]

    def process_latest_from_receiver(
        self,
        receiver: MultimodalReceiver,
        *,
        remove: bool = True,
        raise_on_error: Optional[bool] = None,
    ) -> Optional[SpatialProcessingResult]:
        """
        Process the most recent spatial packet from a receiver.
        """

        packet = receiver.get_latest(
            "spatial",
            remove=remove,
        )

        if packet is None:
            return None

        return self.process_packet(
            packet,
            raise_on_error=raise_on_error,
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_packet(
        self,
        packet: ReceivedSensorPacket,
    ) -> None:
        if not isinstance(packet, ReceivedSensorPacket):
            raise SpatialPacketValidationError(
                "packet must be ReceivedSensorPacket."
            )

        packet.validate()

        if packet.modality != "spatial":
            raise SpatialPacketValidationError(
                "SpatialInputProcessor accepts only "
                "modality='spatial'."
            )

        if not isinstance(packet.payload, dict):
            raise SpatialPacketValidationError(
                "Spatial packet payload must be a dictionary."
            )

    # ========================================================
    # SPATIAL DATA BUILDING
    # ========================================================

    def _build_spatial_data(
        self,
        packet: ReceivedSensorPacket,
        *,
        warnings: List[str],
    ) -> SpatialData:
        payload = packet.payload

        latitude = require_float(
            payload.get("latitude"),
            "latitude",
            allow_none=True,
        )
        longitude = require_float(
            payload.get("longitude"),
            "longitude",
            allow_none=True,
        )
        altitude = require_float(
            payload.get("altitude_meters"),
            "altitude_meters",
            allow_none=True,
        )

        horizontal_accuracy = require_non_negative(
            payload.get("horizontal_accuracy_meters"),
            "horizontal_accuracy_meters",
            allow_none=True,
        )
        vertical_accuracy = require_non_negative(
            payload.get("vertical_accuracy_meters"),
            "vertical_accuracy_meters",
            allow_none=True,
        )

        heading = require_float(
            payload.get("heading_degrees"),
            "heading_degrees",
            allow_none=True,
        )
        heading_accuracy = require_non_negative(
            payload.get("heading_accuracy_degrees"),
            "heading_accuracy_degrees",
            allow_none=True,
        )

        speed = require_non_negative(
            payload.get("speed_meters_per_second"),
            "speed_meters_per_second",
            allow_none=True,
        )

        provider = (
            str(payload.get("provider"))
            if payload.get("provider") is not None
            else None
        )

        self._validate_ranges(
            latitude=latitude,
            longitude=longitude,
            heading=heading,
            speed=speed,
        )

        limitations: List[str] = []

        if (
            horizontal_accuracy is not None
            and horizontal_accuracy
            > self.settings.spatial
            .maximum_horizontal_accuracy_meters
        ):
            limitations.append("poor_horizontal_accuracy")

        elif (
            horizontal_accuracy is not None
            and horizontal_accuracy
            > self.settings.spatial
            .preferred_horizontal_accuracy_meters
        ):
            limitations.append("reduced_horizontal_accuracy")

        if payload.get("degraded") is True:
            limitations.append("simulated_degraded_quality")

        source_age_ms = self._calculate_age_ms(
            packet.source_timestamp
        )

        if source_age_ms > (
            self.settings.spatial.maximum_location_age_ms
        ):
            limitations.append("stale_spatial_data")

        if (
            speed is not None
            and speed
            > self.settings.spatial
            .maximum_reasonable_speed_meters_per_second
        ):
            limitations.append("unreasonable_speed_value")

        if latitude is None and longitude is None:
            if heading is None:
                raise SpatialPacketValidationError(
                    "Spatial packet must contain coordinates "
                    "or heading."
                )

            warnings.append(
                "location_missing_heading_only_packet"
            )

        if (
            latitude is None
            and longitude is not None
        ) or (
            longitude is None
            and latitude is not None
        ):
            raise SpatialPacketValidationError(
                "latitude and longitude must be provided together."
            )

        if horizontal_accuracy is None:
            warnings.append(
                "horizontal_accuracy_missing"
            )

        if heading is None:
            warnings.append("heading_missing")

        preprocessing_steps = [
            "packet_validation",
            "coordinate_normalization",
            "range_validation",
            "freshness_evaluation",
            "accuracy_evaluation",
        ]

        metadata = ModalityMetadata(
            modality="spatial",
            status=ModalityStatus.OBSERVED,
            source_timestamp=packet.source_timestamp,
            arrival_timestamp=packet.arrival_timestamp,
            sequence_number=packet.sequence_number,
            sampling_rate_hz=packet.sampling_rate_hz,
            latency_ms=packet.latency_ms,
            source_device_id=packet.device_id,
            data_reference=packet.packet_id,
            preprocessing_steps=preprocessing_steps,
            limitations=limitations,
            metadata={
                "sensor_type": packet.sensor_type,
                "payload_encoding": (
                    packet.payload_encoding.value
                ),
                "simulated": bool(
                    packet.metadata.get("simulated", False)
                ),
                "scenario": packet.metadata.get("scenario"),
                "provider": provider,
                "source_age_ms": round(
                    source_age_ms,
                    3,
                ),
                "network_strength": (
                    packet.metadata.get("network_strength")
                ),
                "network_latency_ms": (
                    packet.metadata.get(
                        "network_latency_ms"
                    )
                ),
            },
        )

        return SpatialData(
            metadata=metadata,
            latitude=latitude,
            longitude=longitude,
            altitude_meters=altitude,
            horizontal_accuracy_meters=(
                horizontal_accuracy
            ),
            vertical_accuracy_meters=vertical_accuracy,
            heading_degrees=heading,
            heading_accuracy_degrees=heading_accuracy,
            speed_meters_per_second=speed,
            provider=provider,
        )

    def _validate_ranges(
        self,
        *,
        latitude: Optional[float],
        longitude: Optional[float],
        heading: Optional[float],
        speed: Optional[float],
    ) -> None:
        if latitude is not None:
            if not -90.0 <= latitude <= 90.0:
                raise SpatialPacketValidationError(
                    "latitude must be between -90 and 90."
                )

        if longitude is not None:
            if not -180.0 <= longitude <= 180.0:
                raise SpatialPacketValidationError(
                    "longitude must be between -180 and 180."
                )

        if heading is not None:
            if not 0.0 <= heading < 360.0:
                raise SpatialPacketValidationError(
                    "heading_degrees must be in [0, 360)."
                )

        if speed is not None and speed < 0:
            raise SpatialPacketValidationError(
                "speed_meters_per_second cannot be negative."
            )

    def _calculate_age_ms(
        self,
        source_timestamp: str,
    ) -> float:
        source = parse_iso_timestamp(source_timestamp)
        now = datetime.now(timezone.utc)

        age_ms = (
            now - source
        ).total_seconds() * 1000.0

        return max(0.0, age_ms)

    # ========================================================
    # FALLBACK SUPPORT
    # ========================================================

    def get_last_valid_spatial_data(
        self,
    ) -> Optional[SpatialData]:
        """
        Return the most recent successfully processed SpatialData.
        """

        return self._last_valid_spatial

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "spatial_enabled": self.settings.spatial.enabled,
            "gps_enabled": self.settings.spatial.enable_gps,
            "compass_enabled": (
                self.settings.spatial.enable_compass
            ),
            "last_valid_spatial_available": (
                self._last_valid_spatial is not None
            ),
            "statistics": self.statistics.to_dict(),
        }


# ============================================================
# SELF-TEST
# ============================================================

def run_spatial_input_self_test() -> bool:
    print("\n" + "=" * 72)
    print("NOONGIL-X | LAYER 1 SPATIAL INPUT TEST")
    print("=" * 72)

    try:
        print("[1/6] Creating test settings...")

        settings = create_test_settings()
        processor = SpatialInputProcessor(settings)

        print("[SUCCESS] Spatial processor initialized.")

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

        print("[3/6] Processing latest spatial packet...")

        result = processor.process_latest_from_receiver(
            receiver,
            remove=True,
            raise_on_error=True,
        )

        if result is None:
            raise AssertionError(
                "No spatial packet was available."
            )

        if not result.success:
            raise AssertionError(
                f"Spatial processing failed: {result.error}"
            )

        if result.spatial_data is None:
            raise AssertionError(
                "SpatialData was not produced."
            )

        print("[SUCCESS] Spatial packet processed.")

        print("[4/6] Validating SpatialData...")

        spatial = result.spatial_data
        spatial.validate()

        if spatial.metadata.modality != "spatial":
            raise AssertionError(
                "SpatialData modality is incorrect."
            )

        if spatial.latitude is None:
            raise AssertionError(
                "Latitude was not produced."
            )

        if spatial.longitude is None:
            raise AssertionError(
                "Longitude was not produced."
            )

        if spatial.provider != "simulated_gps":
            raise AssertionError(
                "Unexpected spatial provider."
            )

        print("[SUCCESS] SpatialData is valid.")

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
                "Non-spatial packet was incorrectly accepted."
            )

        print("[SUCCESS] Invalid modality was rejected.")

        print("[6/6] Checking diagnostics...")

        health = processor.health_check()

        if not health["healthy"]:
            raise AssertionError(
                "Spatial processor health check failed."
            )

        if health["statistics"]["total_processed"] != 1:
            raise AssertionError(
                "Processed count is incorrect."
            )

        if health["statistics"]["total_failed"] != 1:
            raise AssertionError(
                "Failed count is incorrect."
            )

        if not health["last_valid_spatial_available"]:
            raise AssertionError(
                "Last valid spatial result was not stored."
            )

        print("[SUCCESS] Diagnostics are correct.")

        print("\nSpatialData:")
        print(
            json.dumps(
                result.spatial_data.metadata.metadata
                | {
                    "latitude": result.spatial_data.latitude,
                    "longitude": result.spatial_data.longitude,
                    "altitude_meters": (
                        result.spatial_data.altitude_meters
                    ),
                    "horizontal_accuracy_meters": (
                        result.spatial_data
                        .horizontal_accuracy_meters
                    ),
                    "heading_degrees": (
                        result.spatial_data.heading_degrees
                    ),
                    "heading_accuracy_degrees": (
                        result.spatial_data
                        .heading_accuracy_degrees
                    ),
                    "speed_meters_per_second": (
                        result.spatial_data
                        .speed_meters_per_second
                    ),
                    "provider": result.spatial_data.provider,
                    "limitations": (
                        result.spatial_data
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
        print("[PASSED] LAYER 1 SPATIAL INPUT IS WORKING")
        print("=" * 72)

        return True

    except Exception as error:
        print("\n" + "=" * 72)
        print("[FAILED] LAYER 1 SPATIAL INPUT TEST")
        print("=" * 72)
        print(f"[ERROR] {type(error).__name__}: {error}")

        return False


if __name__ == "__main__":
    if not run_spatial_input_self_test():
        raise SystemExit(1)
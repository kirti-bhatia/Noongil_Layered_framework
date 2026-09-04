"""
============================================================
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : Device Input Processor
File    : layer1/modalities/device_input.py
============================================================

Purpose
-------
Consumes normalized device and wearable packets from
MultimodalReceiver and produces validated Layer 1 SourceDevice
and WearableData objects.

Supported packet types
----------------------
- smartphone device status
- wireless earphones
- smartwatch
- smart cane
- haptic wearable
- generic connected device

Responsibilities
----------------
1. Validate wearable/device packet structure
2. Distinguish phone-status packets from connected-device packets
3. Normalize battery, connection, network, and capability fields
4. Track the latest smartphone state
5. Track the latest connected wearable state
6. Build SourceDevice and WearableData objects
7. Log processing and diagnostics
8. Provide a standalone self-test

Architectural Boundary
----------------------
This module does NOT:
- control Bluetooth pairing;
- access Android APIs directly;
- perform audio processing;
- perform intent reasoning;
- perform semantic interpretation;
- use an LLM.

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
from typing import Any, Dict, Iterable, List, Mapping, Optional

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
    NetworkType,
    SourceDevice,
    WearableData,
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

class DeviceInputError(Exception):
    """Base exception for device input processing."""


class DevicePacketValidationError(DeviceInputError):
    """Raised when a device packet is invalid."""


class DeviceProcessingError(DeviceInputError):
    """Raised when device metadata cannot be processed."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class DeviceProcessingResult:
    """
    Result returned after processing one device packet.
    """

    success: bool
    source_device: Optional[SourceDevice] = None
    wearable_data: Optional[WearableData] = None

    packet_id: Optional[str] = None
    sensor_type: Optional[str] = None
    processing_seconds: float = 0.0

    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeviceProcessorStatistics:
    """
    Runtime statistics for DeviceInputProcessor.
    """

    total_received: int = 0
    total_processed: int = 0
    total_failed: int = 0

    phone_status_packets: int = 0
    wearable_packets: int = 0
    connected_packets: int = 0
    disconnected_packets: int = 0
    low_battery_packets: int = 0
    degraded_network_packets: int = 0

    cumulative_processing_seconds: float = 0.0
    last_packet_id: Optional[str] = None
    last_sensor_type: Optional[str] = None
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

def normalize_text(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def require_probability(
    value: Any,
    field_name: str,
    *,
    allow_none: bool = True,
) -> Optional[float]:
    if value is None and allow_none:
        return None

    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise DevicePacketValidationError(
            f"{field_name} must be numeric."
        ) from error

    if not math.isfinite(parsed):
        raise DevicePacketValidationError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= parsed <= 1.0:
        raise DevicePacketValidationError(
            f"{field_name} must be between 0.0 and 1.0."
        )

    return parsed


def require_non_negative(
    value: Any,
    field_name: str,
    *,
    allow_none: bool = True,
) -> Optional[float]:
    if value is None and allow_none:
        return None

    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise DevicePacketValidationError(
            f"{field_name} must be numeric."
        ) from error

    if not math.isfinite(parsed) or parsed < 0:
        raise DevicePacketValidationError(
            f"{field_name} must be finite and non-negative."
        )

    return parsed


def normalize_network_type(
    value: Optional[str],
) -> NetworkType:
    if value is None:
        return NetworkType.UNKNOWN

    normalized = normalize_text(value)

    mapping = {
        "wifi": NetworkType.WIFI,
        "wi_fi": NetworkType.WIFI,
        "mobile_data": NetworkType.MOBILE_DATA,
        "cellular": NetworkType.MOBILE_DATA,
        "bluetooth": NetworkType.BLUETOOTH,
        "usb": NetworkType.USB,
        "offline": NetworkType.OFFLINE,
        "unknown": NetworkType.UNKNOWN,
    }

    return mapping.get(
        normalized,
        NetworkType.UNKNOWN,
    )


# ============================================================
# DEVICE INPUT PROCESSOR
# ============================================================

class DeviceInputProcessor:
    """
    Convert wearable-queue packets into SourceDevice or
    WearableData objects.

    The current simulator routes both:
    - phone device status;
    - connected earphone status;

    through the receiver's wearable queue. This processor
    distinguishes them using sensor_type and payload fields.
    """

    PHONE_SENSOR_TYPES = {
        "device_status",
        "phone_status",
        "smartphone_status",
        "android_device_status",
    }

    WEARABLE_SENSOR_TYPES = {
        "wireless_earphones",
        "earphones",
        "headphones",
        "smartwatch",
        "smart_cane",
        "haptic_band",
        "wearable",
        "connected_device",
    }

    def __init__(
        self,
        settings: Optional[Layer1Settings] = None,
    ) -> None:
        self.settings = settings or create_default_settings()
        self.settings.validate()

        self.logger = get_logger("modalities.device_input")
        self.statistics = DeviceProcessorStatistics()

        self._latest_source_device: Optional[
            SourceDevice
        ] = None

        self._latest_wearables: Dict[
            str,
            WearableData,
        ] = {}

    # ========================================================
    # PUBLIC API
    # ========================================================

    def process_packet(
        self,
        packet: ReceivedSensorPacket,
        *,
        raise_on_error: Optional[bool] = None,
    ) -> DeviceProcessingResult:
        """
        Process one normalized wearable/device packet.
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
                "device_input.process_packet",
                logger=self.logger,
                metadata={
                    "packet_id": packet.packet_id,
                    "device_id": packet.device_id,
                },
            ):
                self._validate_packet(packet)

                warnings: List[str] = []

                packet_kind = self._classify_packet(
                    packet
                )

                source_device: Optional[
                    SourceDevice
                ] = None

                wearable_data: Optional[
                    WearableData
                ] = None

                if packet_kind == "phone":
                    source_device = (
                        self._build_source_device(
                            packet,
                            warnings=warnings,
                        )
                    )
                    source_device.validate()

                    self._latest_source_device = (
                        source_device
                    )
                    self.statistics.phone_status_packets += 1

                elif packet_kind == "wearable":
                    wearable_data = (
                        self._build_wearable_data(
                            packet,
                            warnings=warnings,
                        )
                    )
                    wearable_data.validate()

                    wearable_key = (
                        wearable_data.device_id
                        or wearable_data.device_name
                        or packet.packet_id
                    )

                    self._latest_wearables[
                        wearable_key
                    ] = wearable_data

                    self.statistics.wearable_packets += 1

                    if wearable_data.connected:
                        self.statistics.connected_packets += 1
                    else:
                        self.statistics.disconnected_packets += 1

                else:
                    raise DevicePacketValidationError(
                        "Unable to classify device packet."
                    )

                elapsed = time.perf_counter() - started

                self.statistics.total_processed += 1
                self.statistics.cumulative_processing_seconds += (
                    elapsed
                )
                self.statistics.last_packet_id = packet.packet_id
                self.statistics.last_sensor_type = (
                    packet.sensor_type
                )
                self.statistics.last_error = None

                details: Dict[str, Any] = {
                    "packet_kind": packet_kind,
                    "processing_seconds": round(
                        elapsed,
                        6,
                    ),
                    "warnings": warnings,
                }

                if source_device is not None:
                    details.update(
                        {
                            "source_device_id": (
                                source_device.device_id
                            ),
                            "battery_level": (
                                source_device.battery_level
                            ),
                            "network_type": (
                                source_device.network_type.value
                            ),
                            "network_strength": (
                                source_device.network_strength
                            ),
                        }
                    )

                if wearable_data is not None:
                    details.update(
                        {
                            "wearable_device_id": (
                                wearable_data.device_id
                            ),
                            "wearable_device_name": (
                                wearable_data.device_name
                            ),
                            "connected": (
                                wearable_data.connected
                            ),
                            "battery_level": (
                                wearable_data.battery_level
                            ),
                        }
                    )

                log_sensor_event(
                    modality="wearable",
                    event="Device packet processed",
                    device_id=packet.device_id,
                    sensor_type=packet.sensor_type,
                    packet_id=packet.packet_id,
                    sequence_number=packet.sequence_number,
                    details=details,
                )

                return DeviceProcessingResult(
                    success=True,
                    source_device=source_device,
                    wearable_data=wearable_data,
                    packet_id=packet.packet_id,
                    sensor_type=packet.sensor_type,
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
            self.statistics.last_sensor_type = getattr(
                packet,
                "sensor_type",
                None,
            )
            self.statistics.last_error = (
                f"{type(error).__name__}: {error}"
            )

            log_exception(
                self.logger,
                "Device packet processing failed",
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
                    "sensor_type": getattr(
                        packet,
                        "sensor_type",
                        None,
                    ),
                },
            )

            if should_raise:
                raise

            return DeviceProcessingResult(
                success=False,
                packet_id=getattr(
                    packet,
                    "packet_id",
                    None,
                ),
                sensor_type=getattr(
                    packet,
                    "sensor_type",
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
    ) -> List[DeviceProcessingResult]:
        """
        Drain and process wearable/device packets.
        """

        packets = receiver.drain(
            "wearable",
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
    ) -> Optional[DeviceProcessingResult]:
        """
        Process the newest wearable/device packet.
        """

        packet = receiver.get_latest(
            "wearable",
            remove=remove,
        )

        if packet is None:
            return None

        return self.process_packet(
            packet,
            raise_on_error=raise_on_error,
        )

    # ========================================================
    # VALIDATION AND CLASSIFICATION
    # ========================================================

    def _validate_packet(
        self,
        packet: ReceivedSensorPacket,
    ) -> None:
        if not isinstance(packet, ReceivedSensorPacket):
            raise DevicePacketValidationError(
                "packet must be ReceivedSensorPacket."
            )

        packet.validate()

        if packet.modality != "wearable":
            raise DevicePacketValidationError(
                "DeviceInputProcessor accepts only "
                "modality='wearable'."
            )

        if not isinstance(packet.payload, dict):
            raise DevicePacketValidationError(
                "Device packet payload must be a dictionary."
            )

    def _classify_packet(
        self,
        packet: ReceivedSensorPacket,
    ) -> str:
        sensor_type = normalize_text(
            packet.sensor_type or ""
        )

        if sensor_type in self.PHONE_SENSOR_TYPES:
            return "phone"

        if sensor_type in self.WEARABLE_SENSOR_TYPES:
            return "wearable"

        payload = packet.payload

        device_type = normalize_text(
            str(
                payload.get(
                    "device_type",
                    "",
                )
            )
        )

        if device_type in {
            "android_smartphone",
            "smartphone",
            "phone",
        }:
            return "phone"

        if any(
            key in payload
            for key in (
                "microphone_available",
                "audio_output_available",
                "haptic_output_available",
                "capabilities",
                "connected",
            )
        ):
            return "wearable"

        if any(
            key in payload
            for key in (
                "operating_system",
                "application_version",
                "available_sensors",
                "network_type",
                "is_charging",
            )
        ):
            return "phone"

        raise DevicePacketValidationError(
            f"Unsupported device sensor type: "
            f"{packet.sensor_type!r}"
        )

    # ========================================================
    # SOURCE DEVICE
    # ========================================================

    def _build_source_device(
        self,
        packet: ReceivedSensorPacket,
        *,
        warnings: List[str],
    ) -> SourceDevice:
        payload = packet.payload

        device_id = str(
            payload.get("device_id")
            or packet.device_id
        )

        device_type = str(
            payload.get(
                "device_type",
                self.settings.device
                .expected_phone_device_type,
            )
        )

        network_strength = require_probability(
            payload.get("network_strength"),
            "network_strength",
        )

        network_latency = require_non_negative(
            payload.get("network_latency_ms"),
            "network_latency_ms",
        )

        battery_level = require_probability(
            payload.get("battery_level"),
            "battery_level",
        )

        if battery_level is None:
            warnings.append("battery_level_missing")
        elif battery_level < (
            self.settings.namara.battery_low_threshold
        ):
            self.statistics.low_battery_packets += 1
            warnings.append("low_phone_battery")

        if (
            network_strength is not None
            and network_strength
            < self.settings.namara
            .network_degraded_strength_threshold
        ):
            self.statistics.degraded_network_packets += 1
            warnings.append("degraded_network_strength")

        if (
            network_latency is not None
            and network_latency
            > self.settings.namara
            .network_high_latency_ms
        ):
            self.statistics.degraded_network_packets += 1
            warnings.append("high_network_latency")

        available_sensors = payload.get(
            "available_sensors",
            [],
        )

        if available_sensors is None:
            available_sensors = []

        if not isinstance(
            available_sensors,
            list,
        ):
            raise DevicePacketValidationError(
                "available_sensors must be a list."
            )

        return SourceDevice(
            device_id=device_id,
            device_type=device_type,
            device_name=(
                str(payload.get("device_name"))
                if payload.get("device_name")
                is not None
                else None
            ),
            operating_system=(
                str(
                    payload.get(
                        "operating_system"
                    )
                )
                if payload.get("operating_system")
                is not None
                else None
            ),
            network_type=normalize_network_type(
                payload.get("network_type")
            ),
            network_strength=network_strength,
            network_latency_ms=network_latency,
            battery_level=battery_level,
            is_charging=(
                bool(payload.get("is_charging"))
                if payload.get("is_charging")
                is not None
                else None
            ),
            application_version=(
                str(
                    payload.get(
                        "application_version"
                    )
                )
                if payload.get(
                    "application_version"
                )
                is not None
                else None
            ),
            available_sensors=[
                str(sensor)
                for sensor in available_sensors
            ],
            metadata={
                "packet_id": packet.packet_id,
                "sensor_type": packet.sensor_type,
                "source_timestamp": (
                    packet.source_timestamp
                ),
                "arrival_timestamp": (
                    packet.arrival_timestamp
                ),
                "simulated": bool(
                    packet.metadata.get(
                        "simulated",
                        False,
                    )
                ),
                "scenario": packet.metadata.get(
                    "scenario"
                ),
            },
        )

    # ========================================================
    # WEARABLE DEVICE
    # ========================================================

    def _build_wearable_data(
        self,
        packet: ReceivedSensorPacket,
        *,
        warnings: List[str],
    ) -> WearableData:
        payload = packet.payload

        connected = bool(
            payload.get("connected", False)
        )

        battery_level = require_probability(
            payload.get("battery_level"),
            "battery_level",
        )

        if (
            battery_level is not None
            and battery_level
            < self.settings.namara
            .battery_low_threshold
        ):
            self.statistics.low_battery_packets += 1
            warnings.append("low_wearable_battery")

        capabilities = payload.get(
            "capabilities",
            [],
        )

        if capabilities is None:
            capabilities = []

        if not isinstance(capabilities, list):
            raise DevicePacketValidationError(
                "capabilities must be a list."
            )

        if connected and not any(
            (
                payload.get("device_id"),
                payload.get("device_name"),
                payload.get("device_type"),
            )
        ):
            raise DevicePacketValidationError(
                "Connected wearable requires device identity."
            )

        if not connected:
            warnings.append("wearable_disconnected")

        limitations: List[str] = []

        if not connected:
            limitations.append(
                "wearable_not_connected"
            )

        if (
            connected
            and payload.get(
                "audio_output_available"
            ) is False
        ):
            limitations.append(
                "audio_output_unavailable"
            )

        if (
            connected
            and payload.get(
                "microphone_available"
            ) is False
        ):
            limitations.append(
                "microphone_unavailable"
            )

        metadata = ModalityMetadata(
            modality="wearable",
            status=(
                ModalityStatus.OBSERVED
                if connected
                else ModalityStatus.UNAVAILABLE
            ),
            source_timestamp=packet.source_timestamp,
            arrival_timestamp=packet.arrival_timestamp,
            sequence_number=packet.sequence_number,
            sampling_rate_hz=packet.sampling_rate_hz,
            latency_ms=packet.latency_ms,
            source_device_id=packet.device_id,
            data_reference=packet.packet_id,
            preprocessing_steps=[
                "packet_validation",
                "device_type_normalization",
                "connection_state_evaluation",
                "capability_normalization",
                "battery_evaluation",
            ],
            limitations=limitations,
            metadata={
                "sensor_type": packet.sensor_type,
                "payload_encoding": (
                    packet.payload_encoding.value
                ),
                "simulated": bool(
                    packet.metadata.get(
                        "simulated",
                        False,
                    )
                ),
                "scenario": packet.metadata.get(
                    "scenario"
                ),
                "network_strength": (
                    packet.metadata.get(
                        "network_strength"
                    )
                ),
                "network_latency_ms": (
                    packet.metadata.get(
                        "network_latency_ms"
                    )
                ),
            },
        )

        return WearableData(
            metadata=metadata,
            device_id=(
                str(payload.get("device_id"))
                if payload.get("device_id")
                is not None
                else None
            ),
            device_name=(
                str(payload.get("device_name"))
                if payload.get("device_name")
                is not None
                else None
            ),
            device_type=(
                str(payload.get("device_type"))
                if payload.get("device_type")
                is not None
                else None
            ),
            connected=connected,
            connection_type=(
                str(
                    payload.get(
                        "connection_type"
                    )
                )
                if payload.get("connection_type")
                is not None
                else None
            ),
            battery_level=battery_level,
            capabilities=[
                str(capability)
                for capability in capabilities
            ],
            microphone_available=(
                bool(
                    payload.get(
                        "microphone_available"
                    )
                )
                if payload.get(
                    "microphone_available"
                )
                is not None
                else None
            ),
            audio_output_available=(
                bool(
                    payload.get(
                        "audio_output_available"
                    )
                )
                if payload.get(
                    "audio_output_available"
                )
                is not None
                else None
            ),
            haptic_output_available=(
                bool(
                    payload.get(
                        "haptic_output_available"
                    )
                )
                if payload.get(
                    "haptic_output_available"
                )
                is not None
                else None
            ),
        )

    # ========================================================
    # STATE AND DIAGNOSTICS
    # ========================================================

    def get_latest_source_device(
        self,
    ) -> Optional[SourceDevice]:
        return self._latest_source_device

    def get_latest_wearables(
        self,
    ) -> Dict[str, WearableData]:
        return dict(self._latest_wearables)

    def get_preferred_wearable(
        self,
    ) -> Optional[WearableData]:
        preferred_name = (
            self.settings.device
            .preferred_earphone_name
        )

        for wearable in (
            self._latest_wearables.values()
        ):
            if (
                wearable.device_name
                == preferred_name
            ):
                return wearable

        for wearable in (
            self._latest_wearables.values()
        ):
            if wearable.connected:
                return wearable

        return None

    def clear_state(self) -> None:
        self._latest_source_device = None
        self._latest_wearables.clear()

    def health_check(self) -> Dict[str, Any]:
        preferred = self.get_preferred_wearable()

        return {
            "healthy": True,
            "phone_state_available": (
                self._latest_source_device
                is not None
            ),
            "wearable_count": len(
                self._latest_wearables
            ),
            "preferred_wearable_name": (
                preferred.device_name
                if preferred
                else None
            ),
            "preferred_wearable_connected": (
                preferred.connected
                if preferred
                else False
            ),
            "statistics": (
                self.statistics.to_dict()
            ),
        }


# ============================================================
# SELF-TEST
# ============================================================

def run_device_input_self_test() -> bool:
    print("\n" + "=" * 72)
    print("NOONGIL-X | LAYER 1 DEVICE INPUT TEST")
    print("=" * 72)

    try:
        print("[1/6] Creating test settings...")

        settings = create_test_settings()
        processor = DeviceInputProcessor(settings)

        print("[SUCCESS] Device processor initialized.")

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

        if not all(
            receipt.accepted
            for receipt in receipts
        ):
            raise AssertionError(
                "Simulator packets were not accepted."
            )

        print("[SUCCESS] Simulator packets routed.")

        print("[3/6] Processing device queue...")

        results = processor.process_receiver_queue(
            receiver,
            raise_on_error=True,
        )

        if len(results) != 2:
            raise AssertionError(
                "Expected one wearable packet and "
                "one phone-status packet."
            )

        if not all(
            result.success
            for result in results
        ):
            raise AssertionError(
                "One or more device packets failed."
            )

        print("[SUCCESS] Device packets processed.")

        print("[4/6] Validating outputs...")

        source_device = (
            processor.get_latest_source_device()
        )
        preferred_wearable = (
            processor.get_preferred_wearable()
        )

        if source_device is None:
            raise AssertionError(
                "SourceDevice was not produced."
            )

        if preferred_wearable is None:
            raise AssertionError(
                "WearableData was not produced."
            )

        source_device.validate()
        preferred_wearable.validate()

        if (
            source_device.device_id
            != "PHONE_001"
        ):
            raise AssertionError(
                "Unexpected phone device ID."
            )

        if (
            preferred_wearable.device_name
            != "realme Buds T200"
        ):
            raise AssertionError(
                "Unexpected preferred wearable."
            )

        if not preferred_wearable.connected:
            raise AssertionError(
                "Preferred wearable should be connected."
            )

        print("[SUCCESS] Device outputs are valid.")

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
                "Non-device packet was incorrectly accepted."
            )

        print("[SUCCESS] Invalid modality was rejected.")

        print("[6/6] Checking diagnostics...")

        health = processor.health_check()

        if not health["healthy"]:
            raise AssertionError(
                "Device processor health check failed."
            )

        if (
            health["statistics"]["total_processed"]
            != 2
        ):
            raise AssertionError(
                "Processed count is incorrect."
            )

        if (
            health["statistics"]["total_failed"]
            != 1
        ):
            raise AssertionError(
                "Failed count is incorrect."
            )

        if not health["phone_state_available"]:
            raise AssertionError(
                "Phone state should be available."
            )

        if health["wearable_count"] != 1:
            raise AssertionError(
                "Wearable count is incorrect."
            )

        print("[SUCCESS] Diagnostics are correct.")

        print("\nSourceDevice:")
        print(
            json.dumps(
                {
                    "device_id": (
                        source_device.device_id
                    ),
                    "device_type": (
                        source_device.device_type
                    ),
                    "device_name": (
                        source_device.device_name
                    ),
                    "operating_system": (
                        source_device.operating_system
                    ),
                    "battery_level": (
                        source_device.battery_level
                    ),
                    "is_charging": (
                        source_device.is_charging
                    ),
                    "network_type": (
                        source_device.network_type.value
                    ),
                    "network_strength": (
                        source_device.network_strength
                    ),
                    "network_latency_ms": (
                        source_device
                        .network_latency_ms
                    ),
                    "available_sensors": (
                        source_device
                        .available_sensors
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nWearableData:")
        print(
            json.dumps(
                {
                    "device_id": (
                        preferred_wearable.device_id
                    ),
                    "device_name": (
                        preferred_wearable.device_name
                    ),
                    "device_type": (
                        preferred_wearable.device_type
                    ),
                    "connected": (
                        preferred_wearable.connected
                    ),
                    "connection_type": (
                        preferred_wearable
                        .connection_type
                    ),
                    "battery_level": (
                        preferred_wearable
                        .battery_level
                    ),
                    "capabilities": (
                        preferred_wearable
                        .capabilities
                    ),
                    "microphone_available": (
                        preferred_wearable
                        .microphone_available
                    ),
                    "audio_output_available": (
                        preferred_wearable
                        .audio_output_available
                    ),
                    "haptic_output_available": (
                        preferred_wearable
                        .haptic_output_available
                    ),
                    "limitations": (
                        preferred_wearable
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
        print(
            "[PASSED] LAYER 1 DEVICE INPUT IS WORKING"
        )
        print("=" * 72)

        return True

    except Exception as error:
        print("\n" + "=" * 72)
        print(
            "[FAILED] LAYER 1 DEVICE INPUT TEST"
        )
        print("=" * 72)
        print(
            f"[ERROR] {type(error).__name__}: {error}"
        )

        return False


if __name__ == "__main__":
    if not run_device_input_self_test():
        raise SystemExit(1)
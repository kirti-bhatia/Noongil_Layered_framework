"""
============================================================
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : Interaction Input Processor
File    : layer1/modalities/interaction_input.py
============================================================

Purpose
-------
Consumes normalized interaction packets from MultimodalReceiver
and produces validated Layer 1 InteractionData objects.

Supported interaction inputs
----------------------------
- touch
- button
- voice trigger
- emergency trigger
- gesture trigger
- system event

Responsibilities
----------------
1. Validate interaction packet structure
2. Normalize interaction type and action
3. Detect explicit emergency flags
4. Enforce supported action names
5. Track recent interaction events
6. Build InteractionData for the final sensor packet
7. Log processing and diagnostics
8. Provide a standalone self-test

Architectural Boundary
----------------------
This module records explicit user or device interaction signals.

It does NOT perform:
- intent inference;
- natural-language understanding;
- command reasoning;
- decision making;
- response generation;
- LLM processing.

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import json
import time

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Mapping, Optional

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
    InteractionData,
    InteractionType,
    ModalityMetadata,
    ModalityStatus,
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

class InteractionInputError(Exception):
    """Base exception for interaction processing."""


class InteractionPacketValidationError(InteractionInputError):
    """Raised when an interaction packet is invalid."""


class InteractionProcessingError(InteractionInputError):
    """Raised when an interaction event cannot be processed."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class InteractionProcessingResult:
    """
    Result returned after processing one interaction packet.
    """

    success: bool
    interaction_data: Optional[InteractionData] = None
    packet_id: Optional[str] = None
    interaction_id: Optional[str] = None
    processing_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InteractionProcessorStatistics:
    """
    Runtime statistics for InteractionInputProcessor.
    """

    total_received: int = 0
    total_processed: int = 0
    total_failed: int = 0

    touch_events: int = 0
    button_events: int = 0
    voice_trigger_events: int = 0
    emergency_trigger_events: int = 0
    gesture_trigger_events: int = 0
    system_events: int = 0

    duplicate_events: int = 0
    unsupported_actions: int = 0

    cumulative_processing_seconds: float = 0.0
    last_packet_id: Optional[str] = None
    last_interaction_id: Optional[str] = None
    last_action: Optional[str] = None
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


@dataclass
class InteractionHistoryItem:
    """
    Compact recent interaction record.
    """

    interaction_id: str
    interaction_type: InteractionType
    action: Optional[str]
    emergency_flag: bool
    source_timestamp: str
    packet_id: str


# ============================================================
# HELPERS
# ============================================================

def parse_iso_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise InteractionPacketValidationError(
            "Timestamp must be a non-empty string."
        )

    normalized = value.strip().replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise InteractionPacketValidationError(
            f"Invalid ISO timestamp: {value!r}"
        ) from error

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def normalize_text(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


# ============================================================
# INTERACTION INPUT PROCESSOR
# ============================================================

class InteractionInputProcessor:
    """
    Convert receiver interaction packets into InteractionData.
    """

    TYPE_ALIASES: Dict[str, InteractionType] = {
        "touch": InteractionType.TOUCH,
        "tap": InteractionType.TOUCH,
        "screen_touch": InteractionType.TOUCH,

        "button": InteractionType.BUTTON,
        "hardware_button": InteractionType.BUTTON,
        "touchscreen_button": InteractionType.BUTTON,
        "volume_button": InteractionType.BUTTON,

        "voice_trigger": InteractionType.VOICE_TRIGGER,
        "wake_word": InteractionType.VOICE_TRIGGER,
        "microphone_trigger": InteractionType.VOICE_TRIGGER,

        "emergency": InteractionType.EMERGENCY_TRIGGER,
        "emergency_trigger": InteractionType.EMERGENCY_TRIGGER,
        "sos": InteractionType.EMERGENCY_TRIGGER,
        "panic_button": InteractionType.EMERGENCY_TRIGGER,

        "gesture": InteractionType.GESTURE_TRIGGER,
        "gesture_trigger": InteractionType.GESTURE_TRIGGER,

        "system_event": InteractionType.SYSTEM_EVENT,
        "system": InteractionType.SYSTEM_EVENT,
        "device_event": InteractionType.SYSTEM_EVENT,

        "none": InteractionType.NONE,
    }

    def __init__(
        self,
        settings: Optional[Layer1Settings] = None,
        *,
        history_size: int = 100,
    ) -> None:
        self.settings = settings or create_default_settings()
        self.settings.validate()

        if history_size <= 0:
            raise ValueError(
                "history_size must be greater than zero."
            )

        self.logger = get_logger(
            "modalities.interaction_input"
        )
        self.statistics = InteractionProcessorStatistics()

        self._history: Deque[InteractionHistoryItem] = deque(
            maxlen=history_size
        )

        self._recent_interaction_ids: Deque[str] = deque(
            maxlen=1000
        )
        self._recent_interaction_id_set: set[str] = set()

    # ========================================================
    # PUBLIC API
    # ========================================================

    def process_packet(
        self,
        packet: ReceivedSensorPacket,
        *,
        raise_on_error: Optional[bool] = None,
    ) -> InteractionProcessingResult:
        """
        Process one normalized interaction packet.
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
                "interaction_input.process_packet",
                logger=self.logger,
                metadata={
                    "packet_id": packet.packet_id,
                    "device_id": packet.device_id,
                },
            ):
                self._validate_packet(packet)

                warnings: List[str] = []
                interaction_data = self._build_interaction_data(
                    packet,
                    warnings=warnings,
                )

                interaction_data.validate()

                interaction_id = (
                    interaction_data.interaction_id
                    or packet.packet_id
                )

                if self._is_duplicate_interaction(
                    interaction_id
                ):
                    self.statistics.duplicate_events += 1
                    warnings.append(
                        "duplicate_interaction_id_detected"
                    )

                self._record_interaction(
                    packet=packet,
                    interaction_data=interaction_data,
                )

                elapsed = time.perf_counter() - started

                self.statistics.total_processed += 1
                self.statistics.cumulative_processing_seconds += (
                    elapsed
                )
                self.statistics.last_packet_id = packet.packet_id
                self.statistics.last_interaction_id = (
                    interaction_data.interaction_id
                )
                self.statistics.last_action = (
                    interaction_data.action
                )
                self.statistics.last_error = None

                self._increment_type_statistic(
                    interaction_data.interaction_type
                )

                log_sensor_event(
                    modality="interaction",
                    event="Interaction packet processed",
                    device_id=packet.device_id,
                    sensor_type=packet.sensor_type,
                    packet_id=packet.packet_id,
                    sequence_number=packet.sequence_number,
                    details={
                        "interaction_id": (
                            interaction_data.interaction_id
                        ),
                        "interaction_type": (
                            interaction_data
                            .interaction_type.value
                        ),
                        "action": interaction_data.action,
                        "emergency_flag": (
                            interaction_data.emergency_flag
                        ),
                        "processing_seconds": round(
                            elapsed,
                            6,
                        ),
                        "warnings": warnings,
                    },
                )

                return InteractionProcessingResult(
                    success=True,
                    interaction_data=interaction_data,
                    packet_id=packet.packet_id,
                    interaction_id=(
                        interaction_data.interaction_id
                    ),
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
                "Interaction packet processing failed",
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

            return InteractionProcessingResult(
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
    ) -> List[InteractionProcessingResult]:
        """
        Drain and process interaction packets.
        """

        packets = receiver.drain(
            "interaction",
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
    ) -> Optional[InteractionProcessingResult]:
        """
        Process the newest interaction packet.
        """

        packet = receiver.get_latest(
            "interaction",
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
            raise InteractionPacketValidationError(
                "packet must be ReceivedSensorPacket."
            )

        packet.validate()

        if packet.modality != "interaction":
            raise InteractionPacketValidationError(
                "InteractionInputProcessor accepts only "
                "modality='interaction'."
            )

        if not isinstance(packet.payload, dict):
            raise InteractionPacketValidationError(
                "Interaction payload must be a dictionary."
            )

    # ========================================================
    # INTERACTION BUILDING
    # ========================================================

    def _build_interaction_data(
        self,
        packet: ReceivedSensorPacket,
        *,
        warnings: List[str],
    ) -> InteractionData:
        payload = packet.payload

        interaction_id = str(
            payload.get("interaction_id")
            or packet.packet_id
        )

        raw_type = (
            payload.get("interaction_type")
            or packet.sensor_type
            or "none"
        )

        interaction_type = self._normalize_type(
            str(raw_type)
        )

        action = payload.get("action")

        if action is not None:
            action = normalize_text(str(action))

        emergency_flag = bool(
            payload.get("emergency_flag", False)
        )

        if (
            interaction_type
            == InteractionType.EMERGENCY_TRIGGER
        ):
            emergency_flag = True

        if emergency_flag and (
            interaction_type
            != InteractionType.EMERGENCY_TRIGGER
        ):
            warnings.append(
                "emergency_flag_overrides_interaction_type"
            )
            interaction_type = (
                InteractionType.EMERGENCY_TRIGGER
            )

        if (
            action is None
            and interaction_type
            != InteractionType.SYSTEM_EVENT
        ):
            warnings.append("interaction_action_missing")

        if action is not None:
            supported_actions = set(
                self.settings.interaction.supported_actions
            )

            if action not in supported_actions:
                self.statistics.unsupported_actions += 1

                if (
                    self.settings.runtime.fail_fast
                ):
                    raise InteractionPacketValidationError(
                        f"Unsupported interaction action: "
                        f"{action!r}"
                    )

                warnings.append(
                    "unsupported_action_recorded"
                )

        limitations: List[str] = []

        source_age_ms = self._calculate_age_ms(
            packet.source_timestamp
        )

        if source_age_ms > (
            self.settings.interaction
            .interaction_event_timeout_ms
        ):
            limitations.append(
                "stale_interaction_event"
            )

        preprocessing_steps = [
            "packet_validation",
            "interaction_type_normalization",
            "action_normalization",
            "emergency_flag_evaluation",
            "supported_action_evaluation",
            "freshness_evaluation",
        ]

        metadata = ModalityMetadata(
            modality="interaction",
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

        return InteractionData(
            metadata=metadata,
            interaction_id=interaction_id,
            interaction_type=interaction_type,
            action=action,
            value=payload.get("value"),
            emergency_flag=emergency_flag,
        )

    def _normalize_type(
        self,
        value: str,
    ) -> InteractionType:
        normalized = normalize_text(value)

        interaction_type = self.TYPE_ALIASES.get(
            normalized
        )

        if interaction_type is None:
            raise InteractionPacketValidationError(
                f"Unsupported interaction type: {value!r}"
            )

        return interaction_type

    def _calculate_age_ms(
        self,
        source_timestamp: str,
    ) -> float:
        source = parse_iso_timestamp(source_timestamp)
        now = datetime.now(timezone.utc)

        return max(
            0.0,
            (
                now - source
            ).total_seconds() * 1000.0,
        )

    # ========================================================
    # HISTORY AND DUPLICATES
    # ========================================================

    def _is_duplicate_interaction(
        self,
        interaction_id: str,
    ) -> bool:
        if interaction_id in (
            self._recent_interaction_id_set
        ):
            return True

        if (
            self._recent_interaction_ids.maxlen
            is not None
            and len(self._recent_interaction_ids)
            >= self._recent_interaction_ids.maxlen
        ):
            oldest = (
                self._recent_interaction_ids.popleft()
            )
            self._recent_interaction_id_set.discard(
                oldest
            )

        self._recent_interaction_ids.append(
            interaction_id
        )
        self._recent_interaction_id_set.add(
            interaction_id
        )

        return False

    def _record_interaction(
        self,
        *,
        packet: ReceivedSensorPacket,
        interaction_data: InteractionData,
    ) -> None:
        self._history.append(
            InteractionHistoryItem(
                interaction_id=(
                    interaction_data.interaction_id
                    or packet.packet_id
                ),
                interaction_type=(
                    interaction_data.interaction_type
                ),
                action=interaction_data.action,
                emergency_flag=(
                    interaction_data.emergency_flag
                ),
                source_timestamp=packet.source_timestamp,
                packet_id=packet.packet_id,
            )
        )

    def _increment_type_statistic(
        self,
        interaction_type: InteractionType,
    ) -> None:
        if interaction_type == InteractionType.TOUCH:
            self.statistics.touch_events += 1

        elif interaction_type == InteractionType.BUTTON:
            self.statistics.button_events += 1

        elif (
            interaction_type
            == InteractionType.VOICE_TRIGGER
        ):
            self.statistics.voice_trigger_events += 1

        elif (
            interaction_type
            == InteractionType.EMERGENCY_TRIGGER
        ):
            self.statistics.emergency_trigger_events += 1

        elif (
            interaction_type
            == InteractionType.GESTURE_TRIGGER
        ):
            self.statistics.gesture_trigger_events += 1

        elif (
            interaction_type
            == InteractionType.SYSTEM_EVENT
        ):
            self.statistics.system_events += 1

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    def get_history(
        self,
    ) -> List[InteractionHistoryItem]:
        return list(self._history)

    def latest_interaction(
        self,
    ) -> Optional[InteractionHistoryItem]:
        if not self._history:
            return None

        return self._history[-1]

    def emergency_active(self) -> bool:
        latest = self.latest_interaction()

        return bool(
            latest
            and latest.emergency_flag
        )

    def clear_history(self) -> None:
        self._history.clear()
        self._recent_interaction_ids.clear()
        self._recent_interaction_id_set.clear()

    def health_check(self) -> Dict[str, Any]:
        latest = self.latest_interaction()

        return {
            "healthy": True,
            "interaction_enabled": (
                self.settings.interaction.enabled
            ),
            "touch_enabled": (
                self.settings.interaction
                .enable_touch_input
            ),
            "button_enabled": (
                self.settings.interaction
                .enable_button_input
            ),
            "voice_trigger_enabled": (
                self.settings.interaction
                .enable_voice_trigger
            ),
            "emergency_trigger_enabled": (
                self.settings.interaction
                .enable_emergency_trigger
            ),
            "history_size": len(self._history),
            "latest_action": (
                latest.action
                if latest
                else None
            ),
            "emergency_active": self.emergency_active(),
            "statistics": self.statistics.to_dict(),
        }


# ============================================================
# SELF-TEST
# ============================================================

def run_interaction_input_self_test() -> bool:
    print("\n" + "=" * 72)
    print("NOONGIL-X | LAYER 1 INTERACTION INPUT TEST")
    print("=" * 72)

    try:
        print("[1/6] Creating test settings...")

        settings = create_test_settings()
        processor = InteractionInputProcessor(settings)

        print("[SUCCESS] Interaction processor initialized.")

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

        print("[3/6] Processing latest interaction packet...")

        result = processor.process_latest_from_receiver(
            receiver,
            remove=True,
            raise_on_error=True,
        )

        if result is None:
            raise AssertionError(
                "No interaction packet was available."
            )

        if not result.success:
            raise AssertionError(
                f"Interaction processing failed: "
                f"{result.error}"
            )

        if result.interaction_data is None:
            raise AssertionError(
                "InteractionData was not produced."
            )

        print("[SUCCESS] Interaction packet processed.")

        print("[4/6] Validating InteractionData...")

        interaction = result.interaction_data
        interaction.validate()

        if (
            interaction.metadata.modality
            != "interaction"
        ):
            raise AssertionError(
                "InteractionData modality is incorrect."
            )

        if (
            interaction.interaction_type
            != InteractionType.BUTTON
        ):
            raise AssertionError(
                "Unexpected interaction type."
            )

        if (
            interaction.action
            != "navigation_mode_requested"
        ):
            raise AssertionError(
                "Unexpected interaction action."
            )

        if interaction.emergency_flag:
            raise AssertionError(
                "Navigation interaction must not be emergency."
            )

        print("[SUCCESS] InteractionData is valid.")

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
                "Non-interaction packet was incorrectly accepted."
            )

        print("[SUCCESS] Invalid modality was rejected.")

        print("[6/6] Checking diagnostics...")

        health = processor.health_check()

        if not health["healthy"]:
            raise AssertionError(
                "Interaction processor health check failed."
            )

        if health["statistics"]["total_processed"] != 1:
            raise AssertionError(
                "Processed count is incorrect."
            )

        if health["statistics"]["total_failed"] != 1:
            raise AssertionError(
                "Failed count is incorrect."
            )

        if health["history_size"] != 1:
            raise AssertionError(
                "Interaction history count is incorrect."
            )

        print("[SUCCESS] Diagnostics are correct.")

        print("\nInteractionData:")
        print(
            json.dumps(
                result.interaction_data.metadata.metadata
                | {
                    "interaction_id": (
                        result.interaction_data
                        .interaction_id
                    ),
                    "interaction_type": (
                        result.interaction_data
                        .interaction_type.value
                    ),
                    "action": (
                        result.interaction_data.action
                    ),
                    "value": (
                        result.interaction_data.value
                    ),
                    "emergency_flag": (
                        result.interaction_data
                        .emergency_flag
                    ),
                    "limitations": (
                        result.interaction_data
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
            "[PASSED] LAYER 1 INTERACTION INPUT IS WORKING"
        )
        print("=" * 72)

        return True

    except Exception as error:
        print("\n" + "=" * 72)
        print(
            "[FAILED] LAYER 1 INTERACTION INPUT TEST"
        )
        print("=" * 72)
        print(
            f"[ERROR] {type(error).__name__}: {error}"
        )

        return False


if __name__ == "__main__":
    if not run_interaction_input_self_test():
        raise SystemExit(1)
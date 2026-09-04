"""
============================================================
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : Multimodal Receiver
File    : layer1/acquisition/multimodal_receiver.py
============================================================

Purpose
-------
The Multimodal Receiver is the entry point for smartphone and
simulated sensor data entering NOONGIL Layer 1.

It is responsible for:

1. Receiving sensor packets
2. Decoding JSON, bytes, strings, and dictionaries
3. Validating the common sensor-envelope structure
4. Validating device identity and authentication metadata
5. Detecting duplicates and invalid sequence numbers
6. Assigning arrival timestamps
7. Measuring approximate network latency
8. Normalizing modality names
9. Routing packets into modality-specific queues
10. Maintaining receiver statistics
11. Providing packets to downstream modality processors

Architectural Boundary
----------------------
This module does NOT:

- resize camera frames;
- process audio;
- calculate GPS quality;
- calculate motion intensity;
- perform multimodal synchronization;
- estimate confidence;
- perform semantic interpretation;
- run YOLO, OCR, Whisper, or an LLM.

Those operations are performed by later Layer 1 or Layer 2
modules.

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import threading
import time

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional

from layer1.config.settings import (
    ExecutionMode,
    Layer1Settings,
    create_default_settings,
    create_test_settings,
)
from layer1.schemas.sensor_packet import SUPPORTED_MODALITIES


# ============================================================
# CONSTANTS
# ============================================================

RECEIVER_SCHEMA_VERSION = "1.0"

MODALITY_ALIASES: Dict[str, str] = {
    "camera": "vision",
    "image": "vision",
    "frame": "vision",
    "rgb": "vision",
    "video": "vision",

    "microphone": "audio",
    "mic": "audio",
    "sound": "audio",
    "waveform": "audio",

    "gps": "spatial",
    "location": "spatial",
    "compass": "spatial",
    "heading": "spatial",

    "accelerometer": "motion",
    "gyroscope": "motion",
    "gyro": "motion",
    "imu": "motion",
    "magnetometer": "motion",

    "touch": "interaction",
    "button": "interaction",
    "emergency": "interaction",
    "gesture": "interaction",

    "earphone": "wearable",
    "earphones": "wearable",
    "headphones": "wearable",
    "smartwatch": "wearable",
    "smart_cane": "wearable",

    "weather": "environment",
    "traffic": "environment",
    "map": "environment",
    "transport": "environment",
}


# ============================================================
# ENUMERATIONS
# ============================================================

class ReceiverStatus(str, Enum):
    """
    Current receiver lifecycle state.
    """

    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class PacketAcceptanceStatus(str, Enum):
    """
    Result of receiving one packet.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"
    UNSUPPORTED = "unsupported"


class PayloadEncoding(str, Enum):
    """
    Supported payload encodings.
    """

    JSON = "json"
    BASE64 = "base64"
    UTF8 = "utf8"
    BINARY_REFERENCE = "binary_reference"
    UNKNOWN = "unknown"


# ============================================================
# EXCEPTIONS
# ============================================================

class ReceiverError(Exception):
    """
    Base exception for multimodal-receiver errors.
    """


class PacketDecodeError(ReceiverError):
    """
    Raised when an incoming packet cannot be decoded.
    """


class PacketValidationError(ReceiverError):
    """
    Raised when a decoded packet has an invalid structure.
    """


class PacketAuthenticationError(ReceiverError):
    """
    Raised when packet authentication fails.
    """


class PacketSizeError(ReceiverError):
    """
    Raised when an incoming packet exceeds the configured limit.
    """


class ReceiverStateError(ReceiverError):
    """
    Raised when an operation is invalid for the receiver state.
    """


# ============================================================
# GENERAL HELPERS
# ============================================================

def utc_now_iso() -> str:
    """
    Return the current UTC timestamp in ISO 8601 format.
    """

    return datetime.now(timezone.utc).isoformat(
        timespec="milliseconds"
    )


def parse_iso_timestamp(value: str) -> datetime:
    """
    Parse an ISO 8601 timestamp.

    Naive timestamps are interpreted as UTC.
    """

    if not isinstance(value, str) or not value.strip():
        raise PacketValidationError(
            "Timestamp must be a non-empty ISO 8601 string."
        )

    normalized = value.strip().replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise PacketValidationError(
            f"Invalid ISO 8601 timestamp: {value!r}"
        ) from error

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def calculate_latency_ms(
    source_timestamp: str,
    arrival_timestamp: str,
) -> float:
    """
    Calculate non-negative arrival latency in milliseconds.
    """

    source = parse_iso_timestamp(source_timestamp)
    arrival = parse_iso_timestamp(arrival_timestamp)

    latency = (
        arrival - source
    ).total_seconds() * 1000.0

    return max(0.0, latency)


def normalize_modality(value: str) -> str:
    """
    Convert a raw sensor/modality label into a Layer 1 modality.
    """

    if not isinstance(value, str) or not value.strip():
        raise PacketValidationError(
            "modality must be a non-empty string."
        )

    normalized = (
        value.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    normalized = MODALITY_ALIASES.get(
        normalized,
        normalized,
    )

    if normalized not in SUPPORTED_MODALITIES:
        raise PacketValidationError(
            f"Unsupported modality {value!r}. "
            f"Supported modalities: "
            f"{sorted(SUPPORTED_MODALITIES)}"
        )

    return normalized


def make_json_safe(value: Any) -> Any:
    """
    Recursively convert supported values into JSON-safe values.
    """

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "data": base64.b64encode(value).decode("ascii"),
        }

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


def estimate_payload_size_bytes(value: Any) -> int:
    """
    Estimate the serialized size of an incoming packet.
    """

    if isinstance(value, bytes):
        return len(value)

    if isinstance(value, str):
        return len(value.encode("utf-8"))

    try:
        serialized = json.dumps(
            make_json_safe(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise PacketDecodeError(
            f"Packet contains non-serializable values: {error}"
        ) from error

    return len(serialized.encode("utf-8"))


def stable_packet_hash(packet: Mapping[str, Any]) -> str:
    """
    Generate a stable SHA-256 fingerprint for duplicate detection.
    """

    serialized = json.dumps(
        make_json_safe(packet),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# ============================================================
# RECEIVER DATA STRUCTURES
# ============================================================

@dataclass
class ReceivedSensorPacket:
    """
    Normalized sensor envelope accepted by the receiver.

    The payload remains signal-level data. Modality-specific
    processing occurs after routing.
    """

    packet_id: str
    device_id: str
    modality: str

    source_timestamp: str
    arrival_timestamp: str

    sequence_number: int
    payload: Dict[str, Any]

    sensor_type: Optional[str] = None
    payload_encoding: PayloadEncoding = PayloadEncoding.JSON

    sampling_rate_hz: Optional[float] = None
    latency_ms: float = 0.0

    schema_version: str = RECEIVER_SCHEMA_VERSION

    authentication_verified: bool = False
    checksum: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate the normalized packet.
        """

        for name, value in (
            ("packet_id", self.packet_id),
            ("device_id", self.device_id),
            ("modality", self.modality),
            ("source_timestamp", self.source_timestamp),
            ("arrival_timestamp", self.arrival_timestamp),
            ("schema_version", self.schema_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise PacketValidationError(
                    f"{name} must be a non-empty string."
                )

        if self.modality not in SUPPORTED_MODALITIES:
            raise PacketValidationError(
                f"Unsupported normalized modality: "
                f"{self.modality!r}"
            )

        parse_iso_timestamp(self.source_timestamp)
        parse_iso_timestamp(self.arrival_timestamp)

        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            raise PacketValidationError(
                "sequence_number must be a non-negative integer."
            )

        if not isinstance(self.payload, dict):
            raise PacketValidationError(
                "payload must be a dictionary."
            )

        if self.sampling_rate_hz is not None:
            if (
                isinstance(self.sampling_rate_hz, bool)
                or not isinstance(
                    self.sampling_rate_hz,
                    (int, float),
                )
                or not math.isfinite(
                    float(self.sampling_rate_hz)
                )
                or float(self.sampling_rate_hz) <= 0
            ):
                raise PacketValidationError(
                    "sampling_rate_hz must be a positive "
                    "finite number."
                )

        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or not math.isfinite(float(self.latency_ms))
            or float(self.latency_ms) < 0
        ):
            raise PacketValidationError(
                "latency_ms must be a non-negative finite number."
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the packet to a JSON-safe dictionary.
        """

        self.validate()
        return make_json_safe(asdict(self))


@dataclass
class PacketReceipt:
    """
    Result returned for every receive operation.
    """

    status: PacketAcceptanceStatus
    accepted: bool

    packet_id: Optional[str] = None
    modality: Optional[str] = None
    device_id: Optional[str] = None
    sequence_number: Optional[int] = None

    reason: Optional[str] = None
    queue_size: Optional[int] = None
    received_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-safe receipt dictionary.
        """

        return make_json_safe(asdict(self))


@dataclass
class ReceiverStatistics:
    """
    Runtime statistics for the receiver.
    """

    total_received: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    total_duplicates: int = 0
    total_unsupported: int = 0

    accepted_by_modality: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    rejected_by_reason: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    last_packet_at: Optional[str] = None
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Return plain dictionaries instead of defaultdict values.
        """

        return {
            "total_received": self.total_received,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "total_duplicates": self.total_duplicates,
            "total_unsupported": self.total_unsupported,
            "accepted_by_modality": dict(
                self.accepted_by_modality
            ),
            "rejected_by_reason": dict(
                self.rejected_by_reason
            ),
            "last_packet_at": self.last_packet_at,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
        }


# ============================================================
# MULTIMODAL RECEIVER
# ============================================================

class MultimodalReceiver:
    """
    Thread-safe receiver and modality router for Layer 1.

    The receiver accepts raw packets from:

    - the phone-sensor simulator;
    - replayed JSON data;
    - future WebSocket/HTTP handlers;
    - unit tests;
    - local development scripts.

    Incoming packet envelope
    ------------------------
    A packet should normally contain:

    {
        "packet_id": "PHONE_PACKET_001",
        "device_id": "PHONE_001",
        "modality": "vision",
        "sensor_type": "camera",
        "timestamp": "2026-08-06T00:12:10.400+00:00",
        "sequence_number": 1,
        "sampling_rate_hz": 15,
        "payload": {...},
        "metadata": {...}
    }
    """

    def __init__(
        self,
        settings: Optional[Layer1Settings] = None,
    ) -> None:
        self.settings = settings or create_default_settings()
        self.settings.validate()

        self._status = ReceiverStatus.CREATED
        self._lock = threading.RLock()

        queue_size = max(
            1,
            self.settings.synchronization
            .maximum_buffer_items_per_modality,
        )

        self._queues: Dict[str, Deque[ReceivedSensorPacket]] = {
            modality: deque(maxlen=queue_size)
            for modality in SUPPORTED_MODALITIES
        }

        self._statistics = ReceiverStatistics()

        self._recent_hashes: Deque[str] = deque(maxlen=5000)
        self._recent_hash_set: set[str] = set()

        self._last_sequence_by_device_and_sensor: Dict[
            tuple[str, str, str],
            int,
        ] = {}

        self._registered_devices: set[str] = set()

    # ========================================================
    # LIFECYCLE
    # ========================================================

    @property
    def status(self) -> ReceiverStatus:
        """
        Return the receiver lifecycle status.
        """

        return self._status

    def start(self) -> None:
        """
        Start accepting packets.
        """

        with self._lock:
            if self._status == ReceiverStatus.RUNNING:
                return

            self._status = ReceiverStatus.RUNNING
            self._statistics.started_at = utc_now_iso()
            self._statistics.stopped_at = None

    def stop(self) -> None:
        """
        Stop accepting packets.
        """

        with self._lock:
            self._status = ReceiverStatus.STOPPED
            self._statistics.stopped_at = utc_now_iso()

    def reset(self) -> None:
        """
        Clear queues, duplicate history, sequences, and statistics.
        """

        with self._lock:
            for queue in self._queues.values():
                queue.clear()

            self._recent_hashes.clear()
            self._recent_hash_set.clear()
            self._last_sequence_by_device_and_sensor.clear()
            self._registered_devices.clear()

            self._statistics = ReceiverStatistics()
            self._status = ReceiverStatus.CREATED

    def _ensure_running(self) -> None:
        """
        Start automatically in simulation/test modes.

        Live mode should be started explicitly by the network
        transport layer.
        """

        if self._status == ReceiverStatus.RUNNING:
            return

        if self.settings.runtime.execution_mode in {
            ExecutionMode.SIMULATION,
            ExecutionMode.TEST,
            ExecutionMode.REPLAY,
        }:
            self.start()
            return

        raise ReceiverStateError(
            "Receiver is not running. Call start() before "
            "receiving live phone packets."
        )

    # ========================================================
    # DEVICE REGISTRATION
    # ========================================================

    def register_device(self, device_id: str) -> None:
        """
        Register an allowed smartphone or sensor device.
        """

        if not isinstance(device_id, str) or not device_id.strip():
            raise ValueError(
                "device_id must be a non-empty string."
            )

        with self._lock:
            self._registered_devices.add(device_id.strip())

    def unregister_device(self, device_id: str) -> None:
        """
        Remove a device from the registered-device set.
        """

        with self._lock:
            self._registered_devices.discard(device_id)

    def registered_devices(self) -> List[str]:
        """
        Return registered device identifiers.
        """

        with self._lock:
            return sorted(self._registered_devices)

    # ========================================================
    # PACKET DECODING
    # ========================================================

    def decode_packet(
        self,
        incoming: Any,
    ) -> Dict[str, Any]:
        """
        Decode an incoming packet into a dictionary.

        Supported input forms:

        - dictionary/mapping;
        - UTF-8 JSON string;
        - UTF-8 JSON bytes;
        - ReceivedSensorPacket.
        """

        packet_size = estimate_payload_size_bytes(incoming)

        maximum_size = (
            self.settings.network.maximum_packet_size_bytes
        )

        if packet_size > maximum_size:
            raise PacketSizeError(
                f"Incoming packet size {packet_size} bytes exceeds "
                f"the configured limit of {maximum_size} bytes."
            )

        if isinstance(incoming, ReceivedSensorPacket):
            return incoming.to_dict()

        if isinstance(incoming, Mapping):
            return dict(incoming)

        if isinstance(incoming, bytes):
            try:
                incoming = incoming.decode("utf-8")
            except UnicodeDecodeError as error:
                raise PacketDecodeError(
                    "Incoming bytes are not valid UTF-8 JSON."
                ) from error

        if isinstance(incoming, str):
            try:
                decoded = json.loads(incoming)
            except json.JSONDecodeError as error:
                raise PacketDecodeError(
                    f"Incoming string is not valid JSON: {error}"
                ) from error

            if not isinstance(decoded, dict):
                raise PacketDecodeError(
                    "Decoded JSON packet must be an object."
                )

            return decoded

        raise PacketDecodeError(
            "Unsupported incoming packet type: "
            f"{type(incoming).__name__}"
        )

    # ========================================================
    # AUTHENTICATION
    # ========================================================

    def _verify_authentication(
        self,
        packet: Mapping[str, Any],
    ) -> bool:
        """
        Verify an optional packet authentication token.

        The token must be supplied in either:

        packet["authentication_token"]

        or:

        packet["metadata"]["authentication_token"]
        """

        if not self.settings.network.authentication_enabled:
            return False

        expected_token = (
            self.settings.network.get_authentication_token()
        )

        provided_token = packet.get("authentication_token")

        if provided_token is None:
            metadata = packet.get("metadata", {})

            if isinstance(metadata, Mapping):
                provided_token = metadata.get(
                    "authentication_token"
                )

        if not isinstance(provided_token, str):
            raise PacketAuthenticationError(
                "Authentication token is missing."
            )

        if expected_token is None or not hmac.compare_digest(
            provided_token,
            expected_token,
        ):
            raise PacketAuthenticationError(
                "Packet authentication failed."
            )

        return True

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize_packet(
        self,
        packet: Mapping[str, Any],
        *,
        authentication_verified: bool,
    ) -> ReceivedSensorPacket:
        """
        Convert a decoded dictionary into a standard receiver packet.
        """

        arrival_timestamp = utc_now_iso()

        raw_modality = (
            packet.get("modality")
            or packet.get("sensor_group")
            or packet.get("sensor_type")
        )

        modality = normalize_modality(raw_modality)

        sensor_type = packet.get("sensor_type")

        device_id = (
            packet.get("device_id")
            or packet.get("source_device_id")
        )

        if not isinstance(device_id, str) or not device_id.strip():
            raise PacketValidationError(
                "device_id is required."
            )

        device_id = device_id.strip()

        source_timestamp = (
            packet.get("source_timestamp")
            or packet.get("timestamp")
            or packet.get("captured_at")
        )

        if source_timestamp is None:
            source_timestamp = arrival_timestamp

        parsed_source = parse_iso_timestamp(
            str(source_timestamp)
        )

        source_timestamp = parsed_source.isoformat(
            timespec="milliseconds"
        )

        sequence_number = packet.get(
            "sequence_number",
            packet.get("sequence", 0),
        )

        if isinstance(sequence_number, str):
            try:
                sequence_number = int(sequence_number)
            except ValueError as error:
                raise PacketValidationError(
                    "sequence_number must be an integer."
                ) from error

        payload = packet.get("payload")

        if payload is None:
            payload = packet.get("data")

        if payload is None:
            raise PacketValidationError(
                "Packet requires a payload or data field."
            )

        if isinstance(payload, Mapping):
            normalized_payload = dict(payload)

        elif isinstance(payload, bytes):
            normalized_payload = {
                "encoding": "base64",
                "data": base64.b64encode(payload).decode(
                    "ascii"
                ),
            }

        elif isinstance(payload, str):
            normalized_payload = {
                "value": payload,
            }

        elif isinstance(payload, (list, tuple)):
            normalized_payload = {
                "values": list(payload),
            }

        else:
            normalized_payload = {
                "value": payload,
            }

        sampling_rate_hz = packet.get("sampling_rate_hz")

        if sampling_rate_hz is not None:
            try:
                sampling_rate_hz = float(sampling_rate_hz)
            except (TypeError, ValueError) as error:
                raise PacketValidationError(
                    "sampling_rate_hz must be numeric."
                ) from error

        packet_id = packet.get("packet_id")

        if not isinstance(packet_id, str) or not packet_id.strip():
            packet_id = self._generate_packet_id(
                device_id=device_id,
                modality=modality,
                sequence_number=sequence_number,
            )

        raw_encoding = str(
            packet.get(
                "payload_encoding",
                PayloadEncoding.JSON.value,
            )
        ).lower()

        try:
            payload_encoding = PayloadEncoding(raw_encoding)
        except ValueError:
            payload_encoding = PayloadEncoding.UNKNOWN

        latency_ms = calculate_latency_ms(
            source_timestamp=source_timestamp,
            arrival_timestamp=arrival_timestamp,
        )

        metadata = packet.get("metadata", {})

        if metadata is None:
            metadata = {}

        if not isinstance(metadata, Mapping):
            raise PacketValidationError(
                "metadata must be a dictionary."
            )

        normalized = ReceivedSensorPacket(
            packet_id=packet_id.strip(),
            device_id=device_id,
            modality=modality,
            sensor_type=(
                str(sensor_type).strip()
                if sensor_type is not None
                else None
            ),
            source_timestamp=source_timestamp,
            arrival_timestamp=arrival_timestamp,
            sequence_number=sequence_number,
            sampling_rate_hz=sampling_rate_hz,
            latency_ms=latency_ms,
            payload=normalized_payload,
            payload_encoding=payload_encoding,
            schema_version=str(
                packet.get(
                    "schema_version",
                    RECEIVER_SCHEMA_VERSION,
                )
            ),
            authentication_verified=authentication_verified,
            checksum=packet.get("checksum"),
            metadata=dict(metadata),
        )

        normalized.validate()
        return normalized

    @staticmethod
    def _generate_packet_id(
        *,
        device_id: str,
        modality: str,
        sequence_number: int,
    ) -> str:
        """
        Generate a deterministic readable packet identifier.
        """

        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%f"
        )

        return (
            f"RX_{device_id}_{modality}_"
            f"{sequence_number}_{timestamp}"
        )

    # ========================================================
    # PACKET VALIDATION
    # ========================================================

    def _validate_device(
        self,
        packet: ReceivedSensorPacket,
    ) -> None:
        """
        Validate device registration requirements.
        """

        if not self.settings.network.verify_device_id:
            return

        if self.settings.device.allow_unknown_phone:
            return

        if packet.device_id not in self._registered_devices:
            raise PacketValidationError(
                f"Unregistered device: {packet.device_id!r}"
            )

    def _validate_enabled_modality(
        self,
        packet: ReceivedSensorPacket,
    ) -> None:
        """
        Ensure the modality is enabled by current settings.
        """

        enabled_modalities = set(
            self.settings.enabled_modalities()
        )

        if packet.modality not in enabled_modalities:
            raise PacketValidationError(
                f"Modality {packet.modality!r} is disabled "
                "in Layer 1 settings."
            )

    def _validate_sequence(
        self,
        packet: ReceivedSensorPacket,
    ) -> None:
        """
        Detect backward sequence numbers.

        Separate sequence tracking is used for each device,
        modality, and sensor type.
        """

        sensor_type = packet.sensor_type or packet.modality

        key = (
            packet.device_id,
            packet.modality,
            sensor_type,
        )

        previous = self._last_sequence_by_device_and_sensor.get(
            key
        )

        if previous is not None and packet.sequence_number < previous:
            raise PacketValidationError(
                "Out-of-order sequence number. "
                f"Previous={previous}, "
                f"received={packet.sequence_number}, "
                f"device={packet.device_id!r}, "
                f"sensor={sensor_type!r}."
            )

    def _validate_checksum(
        self,
        packet: ReceivedSensorPacket,
    ) -> None:
        """
        Validate an optional SHA-256 checksum of the payload.
        """

        if packet.checksum is None:
            return

        payload_serialized = json.dumps(
            make_json_safe(packet.payload),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        calculated = hashlib.sha256(
            payload_serialized
        ).hexdigest()

        if not hmac.compare_digest(
            calculated.lower(),
            str(packet.checksum).lower(),
        ):
            raise PacketValidationError(
                "Payload checksum verification failed."
            )

    # ========================================================
    # DUPLICATE DETECTION
    # ========================================================

    def _is_duplicate(
        self,
        packet: ReceivedSensorPacket,
    ) -> bool:
        """
        Detect exact duplicate packets.
        """

        fingerprint_data = {
            "packet_id": packet.packet_id,
            "device_id": packet.device_id,
            "modality": packet.modality,
            "sensor_type": packet.sensor_type,
            "source_timestamp": packet.source_timestamp,
            "sequence_number": packet.sequence_number,
            "payload": packet.payload,
        }

        fingerprint = stable_packet_hash(
            fingerprint_data
        )

        if fingerprint in self._recent_hash_set:
            return True

        if (
            self._recent_hashes.maxlen is not None
            and len(self._recent_hashes)
            >= self._recent_hashes.maxlen
        ):
            oldest = self._recent_hashes.popleft()
            self._recent_hash_set.discard(oldest)

        self._recent_hashes.append(fingerprint)
        self._recent_hash_set.add(fingerprint)

        return False

    # ========================================================
    # RECEIVING
    # ========================================================

    def receive(
        self,
        incoming: Any,
        *,
        raise_on_error: Optional[bool] = None,
    ) -> PacketReceipt:
        """
        Decode, validate, and route one sensor packet.

        Parameters
        ----------
        incoming:
            Mapping, JSON string, JSON bytes, or
            ReceivedSensorPacket.

        raise_on_error:
            Override settings.runtime.fail_fast.

        Returns
        -------
        PacketReceipt
            Acceptance or rejection details.
        """

        self._ensure_running()

        should_raise = (
            self.settings.runtime.fail_fast
            if raise_on_error is None
            else raise_on_error
        )

        with self._lock:
            self._statistics.total_received += 1
            self._statistics.last_packet_at = utc_now_iso()

            try:
                decoded = self.decode_packet(incoming)

                authentication_verified = (
                    self._verify_authentication(decoded)
                )

                normalized = self._normalize_packet(
                    decoded,
                    authentication_verified=(
                        authentication_verified
                    ),
                )

                self._validate_device(normalized)
                self._validate_enabled_modality(normalized)
                self._validate_checksum(normalized)
                self._validate_sequence(normalized)

                if self._is_duplicate(normalized):
                    self._statistics.total_duplicates += 1

                    return PacketReceipt(
                        status=PacketAcceptanceStatus.DUPLICATE,
                        accepted=False,
                        packet_id=normalized.packet_id,
                        modality=normalized.modality,
                        device_id=normalized.device_id,
                        sequence_number=(
                            normalized.sequence_number
                        ),
                        reason="duplicate_packet",
                        queue_size=len(
                            self._queues[normalized.modality]
                        ),
                    )

                queue = self._queues[normalized.modality]
                queue.append(normalized)

                sensor_type = (
                    normalized.sensor_type
                    or normalized.modality
                )

                sequence_key = (
                    normalized.device_id,
                    normalized.modality,
                    sensor_type,
                )

                self._last_sequence_by_device_and_sensor[
                    sequence_key
                ] = normalized.sequence_number

                self._registered_devices.add(
                    normalized.device_id
                )

                self._statistics.total_accepted += 1
                self._statistics.accepted_by_modality[
                    normalized.modality
                ] += 1

                return PacketReceipt(
                    status=PacketAcceptanceStatus.ACCEPTED,
                    accepted=True,
                    packet_id=normalized.packet_id,
                    modality=normalized.modality,
                    device_id=normalized.device_id,
                    sequence_number=normalized.sequence_number,
                    queue_size=len(queue),
                )

            except PacketValidationError as error:
                receipt = self._reject_packet(
                    status=PacketAcceptanceStatus.REJECTED,
                    reason=str(error),
                )

                if should_raise:
                    raise

                return receipt

            except (
                PacketDecodeError,
                PacketAuthenticationError,
                PacketSizeError,
            ) as error:
                receipt = self._reject_packet(
                    status=PacketAcceptanceStatus.REJECTED,
                    reason=str(error),
                )

                if should_raise:
                    raise

                return receipt

            except Exception as error:
                self._status = ReceiverStatus.ERROR

                receipt = self._reject_packet(
                    status=PacketAcceptanceStatus.REJECTED,
                    reason=(
                        f"Unexpected receiver error: "
                        f"{type(error).__name__}: {error}"
                    ),
                )

                if should_raise:
                    raise ReceiverError(
                        receipt.reason
                    ) from error

                return receipt

    def receive_batch(
        self,
        incoming_packets: Iterable[Any],
        *,
        raise_on_error: Optional[bool] = None,
    ) -> List[PacketReceipt]:
        """
        Receive multiple sensor packets in order.
        """

        receipts: List[PacketReceipt] = []

        for packet in incoming_packets:
            receipt = self.receive(
                packet,
                raise_on_error=raise_on_error,
            )
            receipts.append(receipt)

        return receipts

    def _reject_packet(
        self,
        *,
        status: PacketAcceptanceStatus,
        reason: str,
    ) -> PacketReceipt:
        """
        Record and return a rejected-packet result.
        """

        self._statistics.total_rejected += 1
        self._statistics.rejected_by_reason[reason] += 1

        if status == PacketAcceptanceStatus.UNSUPPORTED:
            self._statistics.total_unsupported += 1

        return PacketReceipt(
            status=status,
            accepted=False,
            reason=reason,
        )

    # ========================================================
    # QUEUE ACCESS
    # ========================================================

    def queue_size(self, modality: str) -> int:
        """
        Return the number of queued packets for a modality.
        """

        normalized = normalize_modality(modality)

        with self._lock:
            return len(self._queues[normalized])

    def total_queue_size(self) -> int:
        """
        Return the total number of queued packets.
        """

        with self._lock:
            return sum(
                len(queue)
                for queue in self._queues.values()
            )

    def get_next(
        self,
        modality: str,
        *,
        remove: bool = True,
    ) -> Optional[ReceivedSensorPacket]:
        """
        Return the oldest queued packet for a modality.
        """

        normalized = normalize_modality(modality)

        with self._lock:
            queue = self._queues[normalized]

            if not queue:
                return None

            if remove:
                return queue.popleft()

            return queue[0]

    def get_latest(
        self,
        modality: str,
        *,
        remove: bool = False,
    ) -> Optional[ReceivedSensorPacket]:
        """
        Return the most recent queued packet for a modality.
        """

        normalized = normalize_modality(modality)

        with self._lock:
            queue = self._queues[normalized]

            if not queue:
                return None

            if remove:
                return queue.pop()

            return queue[-1]

    def drain(
        self,
        modality: str,
        *,
        maximum_items: Optional[int] = None,
    ) -> List[ReceivedSensorPacket]:
        """
        Remove and return queued packets for one modality.
        """

        normalized = normalize_modality(modality)

        if maximum_items is not None:
            if (
                isinstance(maximum_items, bool)
                or not isinstance(maximum_items, int)
                or maximum_items <= 0
            ):
                raise ValueError(
                    "maximum_items must be a positive integer "
                    "or None."
                )

        items: List[ReceivedSensorPacket] = []

        with self._lock:
            queue = self._queues[normalized]

            while queue and (
                maximum_items is None
                or len(items) < maximum_items
            ):
                items.append(queue.popleft())

        return items

    def drain_all(
        self,
    ) -> Dict[str, List[ReceivedSensorPacket]]:
        """
        Drain every modality queue.
        """

        return {
            modality: self.drain(modality)
            for modality in sorted(SUPPORTED_MODALITIES)
        }

    def clear_modality(self, modality: str) -> int:
        """
        Clear one modality queue and return removed item count.
        """

        normalized = normalize_modality(modality)

        with self._lock:
            queue = self._queues[normalized]
            count = len(queue)
            queue.clear()

        return count

    def clear_all(self) -> int:
        """
        Clear all modality queues.
        """

        removed = 0

        with self._lock:
            for queue in self._queues.values():
                removed += len(queue)
                queue.clear()

        return removed

    # ========================================================
    # SNAPSHOTS AND DIAGNOSTICS
    # ========================================================

    def queue_snapshot(
        self,
        *,
        include_payload: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return a non-destructive snapshot of queued packets.
        """

        result: Dict[str, List[Dict[str, Any]]] = {}

        with self._lock:
            for modality, queue in self._queues.items():
                packet_list: List[Dict[str, Any]] = []

                for packet in queue:
                    item = {
                        "packet_id": packet.packet_id,
                        "device_id": packet.device_id,
                        "modality": packet.modality,
                        "sensor_type": packet.sensor_type,
                        "source_timestamp": (
                            packet.source_timestamp
                        ),
                        "arrival_timestamp": (
                            packet.arrival_timestamp
                        ),
                        "sequence_number": (
                            packet.sequence_number
                        ),
                        "sampling_rate_hz": (
                            packet.sampling_rate_hz
                        ),
                        "latency_ms": packet.latency_ms,
                    }

                    if include_payload:
                        item["payload"] = make_json_safe(
                            packet.payload
                        )

                    packet_list.append(item)

                result[modality] = packet_list

        return result

    def statistics(self) -> Dict[str, Any]:
        """
        Return receiver statistics and queue sizes.
        """

        with self._lock:
            data = self._statistics.to_dict()

            data.update(
                {
                    "receiver_status": self._status.value,
                    "queue_sizes": {
                        modality: len(queue)
                        for modality, queue
                        in self._queues.items()
                    },
                    "total_queue_size": (
                        self.total_queue_size()
                    ),
                    "registered_devices": (
                        sorted(self._registered_devices)
                    ),
                }
            )

            return data

    def health_check(self) -> Dict[str, Any]:
        """
        Return a lightweight receiver health report.
        """

        healthy = self._status in {
            ReceiverStatus.CREATED,
            ReceiverStatus.RUNNING,
            ReceiverStatus.STOPPED,
        }

        return {
            "healthy": healthy,
            "status": self._status.value,
            "execution_mode": (
                self.settings.runtime.execution_mode.value
            ),
            "network_enabled": self.settings.network.enabled,
            "enabled_modalities": (
                self.settings.enabled_modalities()
            ),
            "total_received": (
                self._statistics.total_received
            ),
            "total_accepted": (
                self._statistics.total_accepted
            ),
            "total_rejected": (
                self._statistics.total_rejected
            ),
            "queued_packets": self.total_queue_size(),
        }


# ============================================================
# TEST PACKET FACTORY
# ============================================================

def create_demo_input_packets() -> List[Dict[str, Any]]:
    """
    Create sample smartphone packets for receiver testing.
    """

    timestamp = utc_now_iso()

    return [
        {
            "packet_id": "PHONE_VISION_001",
            "schema_version": "1.0",
            "device_id": "PHONE_001",
            "modality": "camera",
            "sensor_type": "rgb_camera",
            "timestamp": timestamp,
            "sequence_number": 1,
            "sampling_rate_hz": 15.0,
            "payload_encoding": "json",
            "payload": {
                "frame_id": "FRAME_000001",
                "width": 640,
                "height": 480,
                "channels": 3,
                "encoding": "jpeg",
                "frame_reference": (
                    "memory://frame_000001"
                ),
            },
            "metadata": {
                "orientation": "portrait",
            },
        },
        {
            "packet_id": "PHONE_AUDIO_001",
            "schema_version": "1.0",
            "device_id": "PHONE_001",
            "modality": "microphone",
            "sensor_type": "phone_microphone",
            "timestamp": timestamp,
            "sequence_number": 1,
            "sampling_rate_hz": 16_000.0,
            "payload_encoding": "json",
            "payload": {
                "chunk_id": "AUDIO_000001",
                "sample_rate_hz": 16_000,
                "channels": 1,
                "duration_ms": 1000.0,
                "audio_reference": (
                    "memory://audio_000001"
                ),
            },
        },
        {
            "packet_id": "PHONE_GPS_001",
            "schema_version": "1.0",
            "device_id": "PHONE_001",
            "modality": "gps",
            "sensor_type": "gps",
            "timestamp": timestamp,
            "sequence_number": 1,
            "sampling_rate_hz": 1.0,
            "payload": {
                "latitude": 31.6340,
                "longitude": 74.8720,
                "horizontal_accuracy_meters": 8.4,
                "heading_degrees": 83.2,
            },
        },
        {
            "packet_id": "PHONE_ACCELEROMETER_001",
            "schema_version": "1.0",
            "device_id": "PHONE_001",
            "modality": "accelerometer",
            "sensor_type": "accelerometer",
            "timestamp": timestamp,
            "sequence_number": 1,
            "sampling_rate_hz": 50.0,
            "payload": {
                "x": 0.18,
                "y": 9.72,
                "z": 0.41,
                "unit": "m/s^2",
            },
        },
        {
            "packet_id": "PHONE_GYROSCOPE_001",
            "schema_version": "1.0",
            "device_id": "PHONE_001",
            "modality": "gyroscope",
            "sensor_type": "gyroscope",
            "timestamp": timestamp,
            "sequence_number": 1,
            "sampling_rate_hz": 50.0,
            "payload": {
                "x": 0.02,
                "y": 0.11,
                "z": -0.04,
                "unit": "rad/s",
            },
        },
        {
            "packet_id": "PHONE_INTERACTION_001",
            "schema_version": "1.0",
            "device_id": "PHONE_001",
            "modality": "button",
            "sensor_type": "touchscreen_button",
            "timestamp": timestamp,
            "sequence_number": 1,
            "payload": {
                "interaction_type": "button",
                "action": "navigation_mode_requested",
                "emergency_flag": False,
            },
        },
        {
            "packet_id": "PHONE_WEARABLE_001",
            "schema_version": "1.0",
            "device_id": "PHONE_001",
            "modality": "earphone",
            "sensor_type": "wireless_earphones",
            "timestamp": timestamp,
            "sequence_number": 1,
            "payload": {
                "device_name": "realme Buds T200",
                "connected": True,
                "connection_type": "bluetooth",
                "microphone_available": True,
                "audio_output_available": True,
            },
        },
    ]


# ============================================================
# COMMAND-LINE SELF-TEST
# ============================================================

def run_receiver_self_test() -> bool:
    """
    Test receiver decoding, validation, routing, duplicate
    detection, queue access, and statistics.
    """

    print("\n" + "=" * 72)
    print("NOONGIL-X | LAYER 1 MULTIMODAL RECEIVER TEST")
    print("=" * 72)

    try:
        print("[1/8] Creating test settings and receiver...")

        settings = create_test_settings()
        receiver = MultimodalReceiver(settings)

        receiver.start()

        if receiver.status != ReceiverStatus.RUNNING:
            raise AssertionError(
                "Receiver did not enter running state."
            )

        print("[SUCCESS] Receiver initialized.")

        print("[2/8] Creating sample phone packets...")

        packets = create_demo_input_packets()

        if len(packets) != 7:
            raise AssertionError(
                "Demo packet count is incorrect."
            )

        print("[SUCCESS] Demo packets created.")

        print("[3/8] Receiving and routing packet batch...")

        receipts = receiver.receive_batch(
            packets,
            raise_on_error=True,
        )

        if not all(
            receipt.accepted
            for receipt in receipts
        ):
            raise AssertionError(
                "One or more valid packets were rejected."
            )

        print("[SUCCESS] All packets were accepted.")

        print("[4/8] Verifying modality aliases and queues...")

        expected_queue_sizes = {
            "vision": 1,
            "audio": 1,
            "spatial": 1,
            "motion": 2,
            "interaction": 1,
            "wearable": 1,
            "environment": 0,
        }

        actual_queue_sizes = {
            modality: receiver.queue_size(modality)
            for modality in SUPPORTED_MODALITIES
        }

        if actual_queue_sizes != expected_queue_sizes:
            raise AssertionError(
                "Unexpected modality queue sizes.\n"
                f"Expected: {expected_queue_sizes}\n"
                f"Received: {actual_queue_sizes}"
            )

        print("[SUCCESS] Modality routing is correct.")

        print("[5/8] Testing duplicate detection...")

        duplicate_receipt = receiver.receive(
            packets[0],
            raise_on_error=True,
        )

        if (
            duplicate_receipt.status
            != PacketAcceptanceStatus.DUPLICATE
        ):
            raise AssertionError(
                "Duplicate packet was not detected."
            )

        print("[SUCCESS] Duplicate detection works.")

        print("[6/8] Testing invalid packet rejection...")

        invalid_receipt = receiver.receive(
            {
                "device_id": "PHONE_001",
                "modality": "unknown_sensor",
                "timestamp": utc_now_iso(),
                "sequence_number": 1,
                "payload": {"value": 1},
            },
            raise_on_error=False,
        )

        if invalid_receipt.accepted:
            raise AssertionError(
                "Invalid packet was incorrectly accepted."
            )

        print("[SUCCESS] Invalid packet was rejected.")

        print("[7/8] Reading routed packets...")

        latest_motion = receiver.get_latest(
            "motion",
            remove=False,
        )

        if latest_motion is None:
            raise AssertionError(
                "Motion queue is unexpectedly empty."
            )

        if latest_motion.sensor_type != "gyroscope":
            raise AssertionError(
                "Latest motion packet is incorrect."
            )

        drained_motion = receiver.drain("motion")

        if len(drained_motion) != 2:
            raise AssertionError(
                "Motion queue did not return two packets."
            )

        if receiver.queue_size("motion") != 0:
            raise AssertionError(
                "Motion queue was not drained."
            )

        print("[SUCCESS] Queue access functions work.")

        print("[8/8] Checking statistics and receiver health...")

        statistics = receiver.statistics()
        health = receiver.health_check()

        if statistics["total_received"] != 9:
            raise AssertionError(
                "Receiver total_received statistic is incorrect."
            )

        if statistics["total_accepted"] != 7:
            raise AssertionError(
                "Receiver total_accepted statistic is incorrect."
            )

        if statistics["total_duplicates"] != 1:
            raise AssertionError(
                "Receiver duplicate statistic is incorrect."
            )

        if statistics["total_rejected"] != 1:
            raise AssertionError(
                "Receiver rejected statistic is incorrect."
            )

        if not health["healthy"]:
            raise AssertionError(
                "Receiver health check failed."
            )

        print("[SUCCESS] Statistics and health are correct.")

        receiver.stop()

        print("\nReceiver summary:")
        print(
            json.dumps(
                receiver.statistics(),
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n" + "=" * 72)
        print("[PASSED] MULTIMODAL RECEIVER IS WORKING CORRECTLY")
        print("=" * 72)

        return True

    except Exception as error:
        print("\n" + "=" * 72)
        print("[FAILED] MULTIMODAL RECEIVER TEST")
        print("=" * 72)
        print(f"[ERROR] {type(error).__name__}: {error}")

        return False


if __name__ == "__main__":
    test_passed = run_receiver_self_test()

    if not test_passed:
        raise SystemExit(1)
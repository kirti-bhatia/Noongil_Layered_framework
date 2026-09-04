"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Layer 1 Input Schema
File    : layer2/schemas/layer1_input.py
============================================================

Purpose
-------
1. Load the Multimodal Sensor Packet produced by Layer 1.
2. Support Layer 1-style test-scenario packets.
3. Validate packet metadata, routing and modalities.
4. Resolve image and audio file paths.
5. Provide normalized input to Layer 2 modules.

This module performs input validation only. It does not perform
object detection, scene classification, OCR, speech recognition
or any other perception operation.

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

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


# ============================================================
# CONSTANTS
# ============================================================

SUPPORTED_PACKET_TYPES = {
    "noongil_layer1_sensor_packet",
    "shivi_layer1_sensor_packet",
}

SUPPORTED_MODALITIES = {
    "vision",
    "audio",
    "spatial",
    "motion",
    "interaction",
    "environment",
    "device",
    "wearable",
}

USABLE_STATUSES = {
    "observed",
    "available",
    "recovered",
    "simulated",
    "complete",
    "active",
}

ALLOWED_BUILD_STATUSES = {
    "complete",
    "partial",
    "degraded",
    "blocked",
}


# ============================================================
# EXCEPTIONS
# ============================================================

class Layer1InputError(Exception):
    """Base exception for Layer 1 input processing."""


class Layer1InputFileError(Layer1InputError):
    """Raised when the input JSON file cannot be loaded."""


class Layer1InputValidationError(Layer1InputError):
    """Raised when a Layer 1 packet fails validation."""


class Layer1MediaNotFoundError(Layer1InputValidationError):
    """Raised when a referenced media file does not exist."""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def require_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    """Validate that a value is a mapping."""

    if not isinstance(value, Mapping):
        raise Layer1InputValidationError(
            f"{field_name} must be a JSON object."
        )

    return value


def require_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    """Validate and return a non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise Layer1InputValidationError(
            f"{field_name} must be a non-empty string."
        )

    return value.strip()


def validate_iso_timestamp(
    value: Any,
    field_name: str,
) -> str:
    """Validate a basic ISO-8601 timestamp."""

    timestamp = require_non_empty_string(
        value,
        field_name,
    )

    normalized = timestamp.replace("Z", "+00:00")

    try:
        datetime.fromisoformat(normalized)
    except ValueError as error:
        raise Layer1InputValidationError(
            f"{field_name} is not a valid ISO-8601 timestamp: "
            f"{timestamp!r}"
        ) from error

    return timestamp


def validate_confidence(
    value: Any,
    field_name: str,
) -> Optional[float]:
    """Validate a confidence value when supplied."""

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise Layer1InputValidationError(
            f"{field_name} must be numeric."
        )

    number = float(value)

    if not math.isfinite(number):
        raise Layer1InputValidationError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= number <= 1.0:
        raise Layer1InputValidationError(
            f"{field_name} must be between 0 and 1."
        )

    return number


def load_json_file(
    file_path: Path,
) -> Dict[str, Any]:
    """Load a JSON object from disk."""

    if not file_path.exists():
        raise Layer1InputFileError(
            f"Layer 1 input file does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise Layer1InputFileError(
            f"Layer 1 input path is not a file: {file_path}"
        )

    try:
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)
    except json.JSONDecodeError as error:
        raise Layer1InputFileError(
            f"Invalid JSON in {file_path}: "
            f"line {error.lineno}, column {error.colno}."
        ) from error
    except OSError as error:
        raise Layer1InputFileError(
            f"Unable to read Layer 1 input: {file_path}"
        ) from error

    if not isinstance(payload, dict):
        raise Layer1InputValidationError(
            "The Layer 1 packet root must be a JSON object."
        )

    return payload


def unwrap_modality(
    modality_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Normalize a modality.

    Frozen Layer 1 format:
        {
            "available": true,
            "data": {...},
            "confidence": {...}
        }

    Test packet format:
        {
            "status": "observed",
            "frame_path": "..."
        }
    """

    if "data" in modality_payload:
        data = modality_payload.get("data")

        if data is None:
            return {}

        if not isinstance(data, Mapping):
            raise Layer1InputValidationError(
                "Wrapped modality data must be a JSON object."
            )

        normalized = dict(data)

        if "available" in modality_payload:
            normalized["_available"] = bool(
                modality_payload["available"]
            )

        if modality_payload.get("confidence") is not None:
            normalized["_layer1_confidence"] = (
                modality_payload["confidence"]
            )

        if modality_payload.get("synchronization") is not None:
            normalized["_synchronization"] = (
                modality_payload["synchronization"]
            )

        if modality_payload.get("recovery") is not None:
            normalized["_recovery"] = (
                modality_payload["recovery"]
            )

        return normalized

    return dict(modality_payload)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class Layer1Metadata:
    """Normalized Layer 1 packet metadata."""

    packet_id: str
    packet_type: str
    schema_version: str
    created_at: str
    source_frame_id: str
    build_status: str
    source_mode: str
    simulated: bool
    scenario: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MediaReference:
    """Resolved file reference for an input media modality."""

    modality: str
    declared_path: str
    resolved_path: Path
    exists: bool


@dataclass
class Layer1InputPacket:
    """
    Normalized input packet consumed by Layer 2.

    Both the frozen Layer 1 format and the test-fixture format
    are converted into this structure.
    """

    metadata: Layer1Metadata
    modalities: Dict[str, Dict[str, Any]]
    synchronization: Dict[str, Any]
    confidence: Dict[str, Any]
    recovery: Optional[Dict[str, Any]]
    acquisition_plan: Optional[Dict[str, Any]]
    layer2_contract: Dict[str, Any]
    source_device: Optional[Dict[str, Any]]
    wearable: Optional[Dict[str, Any]]

    source_file: Path
    project_root: Path

    vision_media: Optional[MediaReference] = None
    audio_media: Optional[MediaReference] = None

    warnings: List[str] = field(default_factory=list)

    @property
    def packet_id(self) -> str:
        return self.metadata.packet_id

    @property
    def scenario(self) -> Optional[str]:
        return self.metadata.scenario

    @property
    def ready_for_layer2(self) -> bool:
        return bool(
            self.layer2_contract.get(
                "ready_for_layer2",
                False,
            )
        )

    @property
    def available_modalities(self) -> List[str]:
        """Return modalities that contain usable data."""

        available = []

        for name, payload in self.modalities.items():
            if not payload:
                continue

            if payload.get("_available") is False:
                continue

            status = str(
                payload.get("status", "available")
            ).lower()

            if status in USABLE_STATUSES:
                available.append(name)

        return sorted(available)

    @property
    def missing_modalities(self) -> List[str]:
        """Return missing modalities reported by Layer 1."""

        contract_missing = self.layer2_contract.get(
            "missing_modalities"
        )

        if isinstance(contract_missing, list):
            return sorted(
                str(item)
                for item in contract_missing
            )

        synchronization_missing = (
            self.synchronization.get(
                "missing_modalities",
                [],
            )
        )

        if not isinstance(
            synchronization_missing,
            list,
        ):
            return []

        return sorted(
            str(item)
            for item in synchronization_missing
        )

    @property
    def overall_sensor_confidence(
        self,
    ) -> Optional[float]:
        """Return Layer 1 sensor confidence."""

        value = self.confidence.get(
            "overall_confidence"
        )

        if value is None:
            value = self.layer2_contract.get(
                "effective_overall_confidence"
            )

        if value is None:
            return None

        return float(value)

    @property
    def vision(self) -> Dict[str, Any]:
        return self.modalities.get("vision", {})

    @property
    def audio(self) -> Dict[str, Any]:
        return self.modalities.get("audio", {})

    @property
    def spatial(self) -> Dict[str, Any]:
        return self.modalities.get("spatial", {})

    @property
    def motion(self) -> Dict[str, Any]:
        return self.modalities.get("motion", {})

    @property
    def interaction(self) -> Dict[str, Any]:
        return self.modalities.get(
            "interaction",
            {},
        )

    @property
    def environment(self) -> Dict[str, Any]:
        return self.modalities.get(
            "environment",
            {},
        )

    @property
    def frame_path(self) -> Optional[Path]:
        if self.vision_media is None:
            return None

        return self.vision_media.resolved_path

    @property
    def audio_path(self) -> Optional[Path]:
        if self.audio_media is None:
            return None

        return self.audio_media.resolved_path

    def summary(self) -> Dict[str, Any]:
        """Return a compact packet summary."""

        return {
            "packet_id": self.packet_id,
            "scenario": self.scenario,
            "source_file": str(self.source_file),
            "ready_for_layer2": self.ready_for_layer2,
            "build_status": self.metadata.build_status,
            "available_modalities": (
                self.available_modalities
            ),
            "missing_modalities": (
                self.missing_modalities
            ),
            "overall_sensor_confidence": (
                self.overall_sensor_confidence
            ),
            "frame_path": (
                str(self.frame_path)
                if self.frame_path
                else None
            ),
            "audio_path": (
                str(self.audio_path)
                if self.audio_path
                else None
            ),
            "warnings": list(self.warnings),
        }


# ============================================================
# INPUT LOADER
# ============================================================

class Layer1InputLoader:
    """
    Load, normalize and validate Layer 1 packets.
    """

    def __init__(
        self,
        project_root: Optional[Path | str] = None,
        *,
        require_media: bool = True,
        allow_blocked_packet: bool = False,
    ) -> None:

        default_project_root = (
            Path(__file__).resolve().parents[2]
        )

        self.project_root = Path(
            project_root or default_project_root
        ).resolve()

        self.require_media = require_media
        self.allow_blocked_packet = (
            allow_blocked_packet
        )

    def load(
        self,
        file_path: Path | str,
    ) -> Layer1InputPacket:
        """Load and validate one Layer 1 packet."""

        source_file = Path(file_path).resolve()
        raw_packet = load_json_file(source_file)

        normalized = self._normalize_packet(
            raw_packet
        )

        packet = self._build_packet(
            normalized,
            source_file,
        )

        self._validate_packet(packet)
        self._resolve_media(packet)
        self._validate_media(packet)

        return packet

    def _normalize_packet(
        self,
        raw_packet: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalize frozen Layer 1 and test fixture naming.

        Frozen Layer 1:
            metadata
            confidence
            recovery
            layer2_contract

        Test fixture:
            packet_metadata
            sensor_confidence
            missing_modality_recovery
            routing
        """

        packet = dict(raw_packet)

        metadata = packet.get("metadata")

        if metadata is None:
            metadata = packet.get(
                "packet_metadata",
                {},
            )

        confidence = packet.get("confidence")

        if confidence is None:
            confidence = packet.get(
                "sensor_confidence",
                {},
            )

        recovery = packet.get("recovery")

        if recovery is None:
            recovery = packet.get(
                "missing_modality_recovery"
            )

        layer2_contract = packet.get(
            "layer2_contract"
        )

        if layer2_contract is None:
            routing = packet.get("routing", {})

            if not isinstance(routing, Mapping):
                routing = {}

            synchronization = packet.get(
                "synchronization",
                {},
            )

            if not isinstance(
                synchronization,
                Mapping,
            ):
                synchronization = {}

            layer2_contract = {
                "consumer": "NOONGIL-X Layer 2",
                "contract_version": "1.0",
                "ready_for_layer2": routing.get(
                    "ready_for_layer2",
                    True,
                ),
                "safe_to_continue": routing.get(
                    "ready_for_layer2",
                    True,
                ),
                "build_status": (
                    metadata.get(
                        "build_status",
                        "complete",
                    )
                    if isinstance(metadata, Mapping)
                    else "complete"
                ),
                "available_modalities": (
                    synchronization.get(
                        "aligned_modalities",
                        [],
                    )
                ),
                "missing_modalities": (
                    synchronization.get(
                        "missing_modalities",
                        [],
                    )
                ),
                "recommended_action": "process",
            }

        modalities = packet.get(
            "modalities",
            {},
        )

        if not isinstance(modalities, Mapping):
            modalities = {}

        normalized_modalities = {}

        for modality_name, modality_payload in (
            modalities.items()
        ):
            if modality_name not in SUPPORTED_MODALITIES:
                continue

            if modality_payload is None:
                normalized_modalities[
                    modality_name
                ] = {}
                continue

            modality_mapping = require_mapping(
                modality_payload,
                f"modalities.{modality_name}",
            )

            normalized_modalities[
                modality_name
            ] = unwrap_modality(
                modality_mapping
            )

        source_device = packet.get(
            "source_device"
        )

        if source_device is None:
            source_device = normalized_modalities.get(
                "device"
            )

        wearable = packet.get("wearable")

        if wearable is None:
            wearable = normalized_modalities.get(
                "wearable"
            )

        return {
            "metadata": metadata,
            "modalities": normalized_modalities,
            "synchronization": packet.get(
                "synchronization",
                {},
            ),
            "confidence": confidence,
            "recovery": recovery,
            "acquisition_plan": packet.get(
                "acquisition_plan"
            ),
            "layer2_contract": layer2_contract,
            "source_device": source_device,
            "wearable": wearable,
        }

    def _build_packet(
        self,
        normalized: Dict[str, Any],
        source_file: Path,
    ) -> Layer1InputPacket:

        metadata_payload = require_mapping(
            normalized["metadata"],
            "metadata",
        )

        packet_id = require_non_empty_string(
            metadata_payload.get("packet_id"),
            "metadata.packet_id",
        )

        packet_type = require_non_empty_string(
            metadata_payload.get(
                "packet_type",
                "noongil_layer1_sensor_packet",
            ),
            "metadata.packet_type",
        )

        schema_version = require_non_empty_string(
            metadata_payload.get(
                "schema_version",
                "1.0",
            ),
            "metadata.schema_version",
        )

        created_at = validate_iso_timestamp(
            metadata_payload.get("created_at"),
            "metadata.created_at",
        )

        source_frame_id = (
            metadata_payload.get(
                "source_frame_id"
            )
            or metadata_payload.get("frame_id")
            or f"FRAME_{packet_id}"
        )

        source_frame_id = require_non_empty_string(
            source_frame_id,
            "metadata.source_frame_id",
        )

        build_status = str(
            metadata_payload.get(
                "build_status",
                "complete",
            )
        ).lower()

        source_mode = str(
            metadata_payload.get(
                "source_mode",
                "unknown",
            )
        )

        warnings = metadata_payload.get(
            "warnings",
            [],
        )

        if not isinstance(warnings, list):
            warnings = [str(warnings)]

        metadata = Layer1Metadata(
            packet_id=packet_id,
            packet_type=packet_type,
            schema_version=schema_version,
            created_at=created_at,
            source_frame_id=source_frame_id,
            build_status=build_status,
            source_mode=source_mode,
            simulated=bool(
                metadata_payload.get(
                    "simulated",
                    False,
                )
            ),
            scenario=metadata_payload.get(
                "scenario"
            ),
            warnings=[
                str(item)
                for item in warnings
            ],
        )

        confidence = require_mapping(
            normalized["confidence"],
            "confidence",
        )

        layer2_contract = require_mapping(
            normalized["layer2_contract"],
            "layer2_contract",
        )

        synchronization = require_mapping(
            normalized["synchronization"],
            "synchronization",
        )

        recovery = normalized["recovery"]

        if recovery is not None:
            recovery = dict(
                require_mapping(
                    recovery,
                    "recovery",
                )
            )

        acquisition_plan = normalized[
            "acquisition_plan"
        ]

        if acquisition_plan is not None:
            acquisition_plan = dict(
                require_mapping(
                    acquisition_plan,
                    "acquisition_plan",
                )
            )

        source_device = normalized[
            "source_device"
        ]

        if source_device is not None:
            source_device = dict(
                require_mapping(
                    source_device,
                    "source_device",
                )
            )

        wearable = normalized["wearable"]

        if wearable is not None:
            wearable = dict(
                require_mapping(
                    wearable,
                    "wearable",
                )
            )

        return Layer1InputPacket(
            metadata=metadata,
            modalities=dict(
                normalized["modalities"]
            ),
            synchronization=dict(
                synchronization
            ),
            confidence=dict(confidence),
            recovery=recovery,
            acquisition_plan=acquisition_plan,
            layer2_contract=dict(
                layer2_contract
            ),
            source_device=source_device,
            wearable=wearable,
            source_file=source_file,
            project_root=self.project_root,
            warnings=list(metadata.warnings),
        )

    def _validate_packet(
        self,
        packet: Layer1InputPacket,
    ) -> None:

        if (
            packet.metadata.packet_type
            not in SUPPORTED_PACKET_TYPES
        ):
            raise Layer1InputValidationError(
                "Unsupported packet type: "
                f"{packet.metadata.packet_type!r}"
            )

        if (
            packet.metadata.build_status
            not in ALLOWED_BUILD_STATUSES
        ):
            raise Layer1InputValidationError(
                "Unsupported build status: "
                f"{packet.metadata.build_status!r}"
            )

        if (
            packet.metadata.build_status == "blocked"
            and not self.allow_blocked_packet
        ):
            raise Layer1InputValidationError(
                "Layer 1 marked this packet as blocked."
            )

        if (
            not packet.ready_for_layer2
            and not self.allow_blocked_packet
        ):
            raise Layer1InputValidationError(
                "Layer 1 packet is not ready for Layer 2."
            )

        if not packet.modalities:
            raise Layer1InputValidationError(
                "The packet contains no supported modalities."
            )

        if not packet.vision and not packet.audio:
            raise Layer1InputValidationError(
                "Layer 2 requires at least vision or audio input."
            )

        validate_confidence(
            packet.confidence.get(
                "overall_confidence"
            ),
            "confidence.overall_confidence",
        )

        validate_confidence(
            packet.layer2_contract.get(
                "effective_overall_confidence"
            ),
            (
                "layer2_contract."
                "effective_overall_confidence"
            ),
        )

    def _resolve_path(
        self,
        declared_path: str,
        source_file: Path,
    ) -> Path:
        """Resolve project-relative and scenario-relative paths."""

        path = Path(declared_path)

        if path.is_absolute():
            return path.resolve()

        candidates = [
            source_file.parent / path,
            source_file.parent.parent / path,
            self.project_root / path,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        return candidates[0].resolve()

    def _extract_media_path(
        self,
        modality: Dict[str, Any],
        keys: tuple[str, ...],
    ) -> Optional[str]:

        for key in keys:
            value = modality.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        return None

    def _resolve_media(
        self,
        packet: Layer1InputPacket,
    ) -> None:

        vision_path = self._extract_media_path(
            packet.vision,
            (
                "frame_path",
                "image_path",
                "file_path",
                "path",
                "data_reference",
            ),
        )

        if vision_path is not None:
            resolved = self._resolve_path(
                vision_path,
                packet.source_file,
            )

            packet.vision_media = MediaReference(
                modality="vision",
                declared_path=vision_path,
                resolved_path=resolved,
                exists=resolved.is_file(),
            )

        audio_path = self._extract_media_path(
            packet.audio,
            (
                "audio_path",
                "file_path",
                "path",
                "data_reference",
            ),
        )

        if audio_path is not None:
            resolved = self._resolve_path(
                audio_path,
                packet.source_file,
            )

            packet.audio_media = MediaReference(
                modality="audio",
                declared_path=audio_path,
                resolved_path=resolved,
                exists=resolved.is_file(),
            )

    def _validate_media(
        self,
        packet: Layer1InputPacket,
    ) -> None:

        if packet.vision:
            if packet.vision_media is None:
                message = (
                    "Vision modality does not contain "
                    "a frame path."
                )

                if self.require_media:
                    raise Layer1MediaNotFoundError(
                        message
                    )

                packet.warnings.append(message)

            elif not packet.vision_media.exists:
                message = (
                    "Vision frame does not exist: "
                    f"{packet.vision_media.resolved_path}"
                )

                if self.require_media:
                    raise Layer1MediaNotFoundError(
                        message
                    )

                packet.warnings.append(message)

        if packet.audio:
            if packet.audio_media is None:
                message = (
                    "Audio modality does not contain "
                    "an audio path."
                )

                if self.require_media:
                    raise Layer1MediaNotFoundError(
                        message
                    )

                packet.warnings.append(message)

            elif not packet.audio_media.exists:
                message = (
                    "Audio file does not exist: "
                    f"{packet.audio_media.resolved_path}"
                )

                if self.require_media:
                    raise Layer1MediaNotFoundError(
                        message
                    )

                packet.warnings.append(message)


# ============================================================
# PUBLIC CONVENIENCE FUNCTION
# ============================================================

def load_layer1_input(
    file_path: Path | str,
    *,
    project_root: Optional[Path | str] = None,
    require_media: bool = True,
) -> Layer1InputPacket:
    """Load one Layer 1 packet using the default loader."""

    loader = Layer1InputLoader(
        project_root=project_root,
        require_media=require_media,
    )

    return loader.load(file_path)


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test(
    scenario_name: str = "park_walking",
) -> bool:
    """Test the loader using one generated scenario."""

    project_root = (
        Path(__file__).resolve().parents[2]
    )

    packet_path = (
        project_root
        / "layer1_output_test_scenarios"
        / scenario_name
        / "layer1_sensor_packet.json"
    )

    print("=" * 72)
    print("NOONGIL-X | LAYER 1 INPUT SCHEMA SELF-TEST")
    print("=" * 72)
    print(f"Scenario : {scenario_name}")
    print(f"Packet   : {packet_path}")

    try:
        packet = load_layer1_input(
            packet_path,
            project_root=project_root,
            require_media=True,
        )

        summary = packet.summary()

        print("\n[PASS] JSON packet loaded")
        print("[PASS] Metadata validated")
        print("[PASS] Layer 2 routing validated")
        print("[PASS] Modalities normalized")
        print("[PASS] Vision frame resolved")
        print("[PASS] Audio file resolved")

        print("\nPacket summary:")

        for key, value in summary.items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 72)
        print("[PASSED] LAYER 1 INPUT SCHEMA IS WORKING")
        print("=" * 72)

        return True

    except Layer1InputError as error:
        print(f"\n[FAILED] {error}")
        print("=" * 72)

        return False


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Load and validate a NOONGIL-X Layer 1 "
            "sensor packet."
        )
    )

    parser.add_argument(
        "packet",
        nargs="?",
        help=(
            "Path to layer1_sensor_packet.json. "
            "If omitted, the self-test is executed."
        ),
    )

    parser.add_argument(
        "--scenario",
        default="park_walking",
        help=(
            "Scenario used by the self-test. "
            "Default: park_walking"
        ),
    )

    parser.add_argument(
        "--allow-missing-media",
        action="store_true",
        help=(
            "Allow missing frame or audio files and "
            "report warnings instead."
        ),
    )

    return parser


def main() -> int:

    parser = build_argument_parser()
    arguments = parser.parse_args()

    if arguments.packet is None:
        return (
            0
            if run_self_test(arguments.scenario)
            else 1
        )

    try:
        packet = load_layer1_input(
            arguments.packet,
            require_media=(
                not arguments.allow_missing_media
            ),
        )

        print(
            json.dumps(
                packet.summary(),
                indent=2,
            )
        )

        return 0

    except Layer1InputError as error:
        print(f"[ERROR] {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
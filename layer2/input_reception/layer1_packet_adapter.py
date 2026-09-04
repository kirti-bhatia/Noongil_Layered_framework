"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Layer 1 Packet Adapter
File    : layer2/input_reception/layer1_packet_adapter.py
============================================================

Purpose
-------
Adapts Layer 1 output into a consistent internal Layer 2 input.

It supports:
- Frozen Layer 1 packets
- Layer 1 test-scenario packets
- Wrapped modality structures
- Flat modality structures
- Frame and audio path resolution
- Sensor confidence preservation
- Synchronization metadata preservation
- Device and wearable information

This module does not perform perception. It only prepares input
for Layer 2 perception modules.

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
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from layer2.schemas.layer1_input import (
    Layer1InputError,
    Layer1InputLoader,
    Layer1InputPacket,
)

from layer2.utils.exceptions import (
    InputReceptionError,
    PacketValidationError,
)

from layer2.utils.logger import (
    Layer2LoggerAdapter,
    get_logger,
    log_event,
    log_exception,
)


# ============================================================
# CONSTANTS
# ============================================================

ADAPTER_VERSION = "1.0"

PERCEPTION_MODALITIES = {
    "vision",
    "audio",
    "spatial",
    "motion",
    "interaction",
    "environment",
    "device",
    "wearable",
}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class AdaptedModality:
    """Normalized representation of one modality."""

    name: str
    available: bool
    data: Dict[str, Any]

    confidence: Optional[float] = None
    timestamp: Optional[str] = None
    media_path: Optional[Path] = None

    recovered: bool = False
    synchronized: bool = False

    warnings: List[str] = field(
        default_factory=list
    )

    @property
    def usable(self) -> bool:
        """Return whether perception may use this modality."""

        if not self.available:
            return False

        if self.name in {
            "vision",
            "audio",
        }:
            return (
                self.media_path is not None
                and self.media_path.is_file()
            )

        return bool(self.data)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe summary."""

        return {
            "name": self.name,
            "available": self.available,
            "usable": self.usable,
            "data": self.data,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "media_path": (
                str(self.media_path)
                if self.media_path
                else None
            ),
            "recovered": self.recovered,
            "synchronized": self.synchronized,
            "warnings": list(self.warnings),
        }


@dataclass
class AdaptedLayer1Input:
    """
    Internal input consumed by Layer 2 modules.
    """

    packet_id: str
    timestamp: str
    source_frame_id: str

    scenario: Optional[str]
    source_mode: str
    simulated: bool

    modalities: Dict[
        str,
        AdaptedModality,
    ]

    sensor_confidence: Dict[str, Any]
    synchronization: Dict[str, Any]
    recovery: Optional[Dict[str, Any]]
    layer2_contract: Dict[str, Any]

    source_device: Optional[Dict[str, Any]]
    wearable: Optional[Dict[str, Any]]

    source_file: Path
    project_root: Path

    warnings: List[str] = field(
        default_factory=list
    )

    adapter_version: str = ADAPTER_VERSION

    @property
    def available_modalities(self) -> List[str]:
        """Return available modality names."""

        return sorted(
            name
            for name, modality
            in self.modalities.items()
            if modality.available
        )

    @property
    def usable_modalities(self) -> List[str]:
        """Return modalities ready for perception."""

        return sorted(
            name
            for name, modality
            in self.modalities.items()
            if modality.usable
        )

    @property
    def unavailable_modalities(self) -> List[str]:
        """Return unavailable modality names."""

        return sorted(
            name
            for name, modality
            in self.modalities.items()
            if not modality.available
        )

    @property
    def frame_path(self) -> Optional[Path]:

        vision = self.modalities.get(
            "vision"
        )

        if vision is None:
            return None

        return vision.media_path

    @property
    def audio_path(self) -> Optional[Path]:

        audio = self.modalities.get(
            "audio"
        )

        if audio is None:
            return None

        return audio.media_path

    @property
    def spatial_data(self) -> Dict[str, Any]:

        modality = self.modalities.get(
            "spatial"
        )

        return (
            modality.data
            if modality is not None
            else {}
        )

    @property
    def motion_data(self) -> Dict[str, Any]:

        modality = self.modalities.get(
            "motion"
        )

        return (
            modality.data
            if modality is not None
            else {}
        )

    @property
    def overall_sensor_confidence(
        self,
    ) -> Optional[float]:

        value = self.sensor_confidence.get(
            "overall_confidence"
        )

        if value is None:
            value = self.layer2_contract.get(
                "effective_overall_confidence"
            )

        if value is None:
            return None

        return float(value)

    def get_modality(
        self,
        name: str,
    ) -> Optional[AdaptedModality]:
        """Return one modality."""

        return self.modalities.get(
            name.strip().lower()
        )

    def require_modality(
        self,
        name: str,
    ) -> AdaptedModality:
        """Return a usable modality or raise an error."""

        normalized_name = (
            name.strip().lower()
        )

        modality = self.modalities.get(
            normalized_name
        )

        if modality is None:
            raise InputReceptionError(
                f"Unknown modality: "
                f"{normalized_name!r}",
                module="layer1_packet_adapter",
                details={
                    "packet_id": self.packet_id,
                    "modality": normalized_name,
                },
            )

        if not modality.usable:
            raise InputReceptionError(
                f"Modality {normalized_name!r} "
                "is not usable.",
                module="layer1_packet_adapter",
                recoverable=True,
                details={
                    "packet_id": self.packet_id,
                    "modality": normalized_name,
                    "available": (
                        modality.available
                    ),
                    "media_path": (
                        str(modality.media_path)
                        if modality.media_path
                        else None
                    ),
                },
            )

        return modality

    def summary(self) -> Dict[str, Any]:
        """Return a compact input summary."""

        return {
            "packet_id": self.packet_id,
            "scenario": self.scenario,
            "timestamp": self.timestamp,
            "source_frame_id": (
                self.source_frame_id
            ),
            "source_mode": self.source_mode,
            "simulated": self.simulated,
            "available_modalities": (
                self.available_modalities
            ),
            "usable_modalities": (
                self.usable_modalities
            ),
            "unavailable_modalities": (
                self.unavailable_modalities
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
            "overall_sensor_confidence": (
                self.overall_sensor_confidence
            ),
            "warnings": list(self.warnings),
            "adapter_version": (
                self.adapter_version
            ),
        }


# ============================================================
# ADAPTER
# ============================================================

class Layer1PacketAdapter:
    """
    Load and adapt Layer 1 packets for Layer 2.
    """

    def __init__(
        self,
        *,
        project_root: Optional[
            Path | str
        ] = None,
        scenario_directory: Optional[
            Path | str
        ] = None,
        require_media: bool = True,
        logger: Optional[
            Layer2LoggerAdapter
        ] = None,
    ) -> None:

        default_project_root = (
            Path(__file__).resolve().parents[2]
        )

        self.project_root = Path(
            project_root
            or default_project_root
        ).resolve()

        self.scenario_directory = Path(
            scenario_directory
            or (
                self.project_root
                / "layer1_output_test_scenarios"
            )
        ).resolve()

        self.require_media = require_media

        self.logger = (
            logger
            or get_logger(
                "layer1_packet_adapter"
            )
        )

        self.input_loader = (
            Layer1InputLoader(
                project_root=self.project_root,
                require_media=require_media,
            )
        )

    def adapt_file(
        self,
        file_path: Path | str,
    ) -> AdaptedLayer1Input:
        """Load and adapt a Layer 1 packet file."""

        source_path = Path(
            file_path
        ).resolve()

        log_event(
            self.logger,
            event="layer1_packet_loading",
            message=(
                "Loading Layer 1 packet."
            ),
            details={
                "file_path": str(source_path)
            },
        )

        try:
            packet = self.input_loader.load(
                source_path
            )

            adapted = self.adapt_packet(
                packet
            )

            log_event(
                self.logger,
                event="layer1_packet_adapted",
                message=(
                    "Layer 1 packet adapted "
                    "successfully."
                ),
                details={
                    "packet_id": (
                        adapted.packet_id
                    ),
                    "scenario": (
                        adapted.scenario
                    ),
                    "usable_modalities": (
                        adapted.usable_modalities
                    ),
                },
            )

            return adapted

        except Layer1InputError as error:

            log_exception(
                self.logger,
                error,
                event=(
                    "layer1_packet_adaptation_failed"
                ),
                message=(
                    "Layer 1 packet could not "
                    "be adapted."
                ),
                details={
                    "file_path": str(source_path)
                },
            )

            raise PacketValidationError(
                "Layer 1 packet validation failed.",
                module="layer1_packet_adapter",
                details={
                    "file_path": str(source_path)
                },
                cause=error,
            ) from error

    def adapt_packet(
        self,
        packet: Layer1InputPacket,
    ) -> AdaptedLayer1Input:
        """Adapt an already validated packet."""

        if not isinstance(
            packet,
            Layer1InputPacket,
        ):
            raise InputReceptionError(
                "packet must be a "
                "Layer1InputPacket.",
                module="layer1_packet_adapter",
                details={
                    "received_type": (
                        packet.__class__.__name__
                    )
                },
            )

        modalities: Dict[
            str,
            AdaptedModality,
        ] = {}

        for name in sorted(
            PERCEPTION_MODALITIES
        ):
            modality_data = (
                packet.modalities.get(
                    name,
                    {},
                )
            )

            if name == "device":
                modality_data = (
                    modality_data
                    or packet.source_device
                    or {}
                )

            elif name == "wearable":
                modality_data = (
                    modality_data
                    or packet.wearable
                    or {}
                )

            modalities[name] = (
                self._adapt_modality(
                    packet,
                    name,
                    modality_data,
                )
            )

        warnings = list(
            packet.warnings
        )

        for modality in modalities.values():
            warnings.extend(
                modality.warnings
            )

        return AdaptedLayer1Input(
            packet_id=packet.packet_id,
            timestamp=(
                packet.metadata.created_at
            ),
            source_frame_id=(
                packet.metadata.source_frame_id
            ),
            scenario=packet.scenario,
            source_mode=(
                packet.metadata.source_mode
            ),
            simulated=(
                packet.metadata.simulated
            ),
            modalities=modalities,
            sensor_confidence=dict(
                packet.confidence
            ),
            synchronization=dict(
                packet.synchronization
            ),
            recovery=(
                dict(packet.recovery)
                if packet.recovery
                is not None
                else None
            ),
            layer2_contract=dict(
                packet.layer2_contract
            ),
            source_device=(
                dict(packet.source_device)
                if packet.source_device
                is not None
                else None
            ),
            wearable=(
                dict(packet.wearable)
                if packet.wearable
                is not None
                else None
            ),
            source_file=(
                packet.source_file
            ),
            project_root=(
                packet.project_root
            ),
            warnings=sorted(
                set(warnings)
            ),
        )

    def load_scenario(
        self,
        scenario_name: str,
    ) -> AdaptedLayer1Input:
        """Load one named test scenario."""

        if (
            not isinstance(
                scenario_name,
                str,
            )
            or not scenario_name.strip()
        ):
            raise InputReceptionError(
                "scenario_name must be a "
                "non-empty string.",
                module="layer1_packet_adapter",
            )

        normalized_name = (
            scenario_name.strip()
        )

        packet_path = (
            self.scenario_directory
            / normalized_name
            / "layer1_sensor_packet.json"
        )

        if not packet_path.exists():
            raise InputReceptionError(
                f"Test scenario does not exist: "
                f"{normalized_name!r}",
                module="layer1_packet_adapter",
                details={
                    "scenario": normalized_name,
                    "expected_packet": (
                        str(packet_path)
                    ),
                },
            )

        return self.adapt_file(
            packet_path
        )

    def discover_scenarios(self) -> List[str]:
        """Return available test-scenario names."""

        manifest_path = (
            self.scenario_directory
            / "manifest.json"
        )

        if manifest_path.exists():
            try:
                manifest = json.loads(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )

                scenarios = manifest.get(
                    "scenarios",
                    [],
                )

                names = []

                for item in scenarios:
                    if not isinstance(
                        item,
                        Mapping,
                    ):
                        continue

                    name = item.get("name")

                    if (
                        isinstance(name, str)
                        and name.strip()
                    ):
                        names.append(
                            name.strip()
                        )

                if names:
                    return sorted(
                        set(names)
                    )

            except (
                OSError,
                json.JSONDecodeError,
            ):
                pass

        if not self.scenario_directory.exists():
            return []

        return sorted(
            directory.name
            for directory
            in self.scenario_directory.iterdir()
            if (
                directory.is_dir()
                and (
                    directory
                    / "layer1_sensor_packet.json"
                ).is_file()
            )
        )

    def _adapt_modality(
        self,
        packet: Layer1InputPacket,
        name: str,
        data: Mapping[str, Any],
    ) -> AdaptedModality:
        """Normalize one modality."""

        normalized_data = dict(data)

        explicit_available = (
            normalized_data.pop(
                "_available",
                None,
            )
        )

        status = str(
            normalized_data.get(
                "status",
                "",
            )
        ).lower()

        available = (
            bool(normalized_data)
            if explicit_available is None
            else bool(explicit_available)
        )

        if status in {
            "missing",
            "unavailable",
            "failed",
            "blocked",
        }:
            available = False

        confidence = self._extract_confidence(
            packet,
            name,
            normalized_data,
        )

        timestamp = self._extract_timestamp(
            packet,
            normalized_data,
        )

        media_path: Optional[Path] = None

        if (
            name == "vision"
            and packet.vision_media
            is not None
        ):
            media_path = (
                packet.vision_media
                .resolved_path
            )

        elif (
            name == "audio"
            and packet.audio_media
            is not None
        ):
            media_path = (
                packet.audio_media
                .resolved_path
            )

        recovered = self._is_recovered(
            packet,
            name,
            normalized_data,
        )

        synchronized = (
            name
            in self._synchronized_modalities(
                packet
            )
        )

        warnings = []

        if available and name in {
            "vision",
            "audio",
        }:
            if (
                media_path is None
                or not media_path.is_file()
            ):
                warnings.append(
                    f"{name} media file is "
                    "unavailable."
                )

        return AdaptedModality(
            name=name,
            available=available,
            data=normalized_data,
            confidence=confidence,
            timestamp=timestamp,
            media_path=media_path,
            recovered=recovered,
            synchronized=synchronized,
            warnings=warnings,
        )

    def _extract_confidence(
        self,
        packet: Layer1InputPacket,
        modality_name: str,
        modality_data: Mapping[str, Any],
    ) -> Optional[float]:
        """Extract one modality's sensor confidence."""

        candidates: List[Any] = []

        embedded = modality_data.get(
            "_layer1_confidence"
        )

        if isinstance(embedded, Mapping):
            candidates.extend(
                [
                    embedded.get(
                        "final_confidence"
                    ),
                    embedded.get(
                        "adjusted_confidence"
                    ),
                    embedded.get(
                        "confidence"
                    ),
                    embedded.get(
                        "score"
                    ),
                ]
            )

        elif embedded is not None:
            candidates.append(embedded)

        packet_confidence = (
            packet.confidence.get(
                modality_name
            )
        )

        if isinstance(
            packet_confidence,
            Mapping,
        ):
            candidates.extend(
                [
                    packet_confidence.get(
                        "final_confidence"
                    ),
                    packet_confidence.get(
                        "adjusted_confidence"
                    ),
                    packet_confidence.get(
                        "confidence"
                    ),
                    packet_confidence.get(
                        "score"
                    ),
                ]
            )

        elif packet_confidence is not None:
            candidates.append(
                packet_confidence
            )

        modality_confidences = (
            packet.confidence.get(
                "modality_confidences"
            )
        )

        if isinstance(
            modality_confidences,
            Mapping,
        ):
            nested = (
                modality_confidences.get(
                    modality_name
                )
            )

            if isinstance(nested, Mapping):
                candidates.extend(
                    [
                        nested.get(
                            "final_confidence"
                        ),
                        nested.get(
                            "adjusted_confidence"
                        ),
                        nested.get(
                            "confidence"
                        ),
                        nested.get("score"),
                    ]
                )

        for value in candidates:
            if (
                isinstance(
                    value,
                    (int, float),
                )
                and not isinstance(
                    value,
                    bool,
                )
            ):
                numeric_value = float(value)

                if (
                    math.isfinite(
                        numeric_value
                    )
                    and 0.0
                    <= numeric_value
                    <= 1.0
                ):
                    return numeric_value

        return None

    def _extract_timestamp(
        self,
        packet: Layer1InputPacket,
        modality_data: Mapping[str, Any],
    ) -> str:
        """Extract modality timestamp."""

        timestamp = modality_data.get(
            "timestamp"
        )

        if (
            isinstance(timestamp, str)
            and timestamp.strip()
        ):
            return timestamp.strip()

        metadata = modality_data.get(
            "metadata"
        )

        if isinstance(metadata, Mapping):
            metadata_timestamp = (
                metadata.get("timestamp")
            )

            if (
                isinstance(
                    metadata_timestamp,
                    str,
                )
                and metadata_timestamp.strip()
            ):
                return (
                    metadata_timestamp.strip()
                )

        return packet.metadata.created_at

    def _is_recovered(
        self,
        packet: Layer1InputPacket,
        modality_name: str,
        modality_data: Mapping[str, Any],
    ) -> bool:
        """Determine whether modality was recovered."""

        status = str(
            modality_data.get(
                "status",
                "",
            )
        ).lower()

        if status == "recovered":
            return True

        embedded_recovery = (
            modality_data.get("_recovery")
        )

        if isinstance(
            embedded_recovery,
            Mapping,
        ):
            if embedded_recovery.get(
                "recovered"
            ) is True:
                return True

            recovery_status = str(
                embedded_recovery.get(
                    "status",
                    "",
                )
            ).lower()

            if recovery_status == "recovered":
                return True

        if packet.recovery is None:
            return False

        recovered_modalities = (
            packet.recovery.get(
                "recovered_modalities",
                [],
            )
        )

        return (
            isinstance(
                recovered_modalities,
                list,
            )
            and modality_name
            in recovered_modalities
        )

    def _synchronized_modalities(
        self,
        packet: Layer1InputPacket,
    ) -> set[str]:
        """Return synchronized modality names."""

        candidates = (
            packet.synchronization.get(
                "selected_modalities"
            )
            or packet.synchronization.get(
                "aligned_modalities"
            )
            or packet.synchronization.get(
                "available_modalities"
            )
            or []
        )

        if not isinstance(candidates, list):
            return set()

        return {
            str(item)
            for item in candidates
        }


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def adapt_layer1_packet(
    file_path: Path | str,
    *,
    project_root: Optional[
        Path | str
    ] = None,
    require_media: bool = True,
) -> AdaptedLayer1Input:
    """Load and adapt a Layer 1 packet."""

    adapter = Layer1PacketAdapter(
        project_root=project_root,
        require_media=require_media,
    )

    return adapter.adapt_file(
        file_path
    )


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test(
    selected_scenario: Optional[str] = None,
) -> bool:

    print("=" * 72)
    print("NOONGIL-X | LAYER 1 PACKET ADAPTER SELF-TEST")
    print("=" * 72)

    try:
        adapter = Layer1PacketAdapter(
            require_media=True
        )

        scenarios = (
            adapter.discover_scenarios()
        )

        if not scenarios:
            raise AssertionError(
                "No Layer 1 test scenarios found."
            )

        print(
            f"[PASS] Discovered "
            f"{len(scenarios)} scenarios"
        )

        if selected_scenario is not None:

            if selected_scenario not in scenarios:
                raise AssertionError(
                    f"Unknown scenario: "
                    f"{selected_scenario}"
                )

            scenarios_to_test = [
                selected_scenario
            ]

        else:
            scenarios_to_test = scenarios

        tested_packets = []

        for scenario in scenarios_to_test:

            adapted = adapter.load_scenario(
                scenario
            )

            if adapted.scenario != scenario:
                raise AssertionError(
                    f"Scenario mismatch for "
                    f"{scenario}."
                )

            if not adapted.packet_id:
                raise AssertionError(
                    f"Packet ID missing for "
                    f"{scenario}."
                )

            if adapted.frame_path is None:
                raise AssertionError(
                    f"Frame path missing for "
                    f"{scenario}."
                )

            if not adapted.frame_path.is_file():
                raise AssertionError(
                    f"Frame does not exist for "
                    f"{scenario}."
                )

            if adapted.audio_path is None:
                raise AssertionError(
                    f"Audio path missing for "
                    f"{scenario}."
                )

            if not adapted.audio_path.is_file():
                raise AssertionError(
                    f"Audio does not exist for "
                    f"{scenario}."
                )

            adapted.require_modality(
                "vision"
            )

            adapted.require_modality(
                "audio"
            )

            tested_packets.append(
                adapted
            )

            print(
                f"[PASS] {scenario}: "
                f"{adapted.packet_id}"
            )

        first_packet = tested_packets[0]

        summary = first_packet.summary()

        required_summary_fields = {
            "packet_id",
            "available_modalities",
            "usable_modalities",
            "frame_path",
            "audio_path",
            "overall_sensor_confidence",
        }

        if not required_summary_fields.issubset(
            summary
        ):
            raise AssertionError(
                "Adapter summary is incomplete."
            )

        print("[PASS] Adapter summary generated")
        print("[PASS] Vision media resolved")
        print("[PASS] Audio media resolved")
        print("[PASS] Sensor confidence preserved")
        print("[PASS] Synchronization preserved")

        print("\nExample adapted packet:")

        for key, value in summary.items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 72)
        print(
            "[PASSED] LAYER 1 PACKET ADAPTER "
            "IS WORKING"
        )
        print("=" * 72)

        return True

    except (
        InputReceptionError,
        PacketValidationError,
        AssertionError,
    ) as error:
        print(f"\n[FAILED] {error}")
        print("=" * 72)

        return False


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Run the NOONGIL-X Layer 1 packet "
            "adapter self-test."
        )
    )

    parser.add_argument(
        "--scenario",
        default=None,
        help=(
            "Test one scenario. If omitted, all "
            "available scenarios are tested."
        ),
    )

    return parser


def main() -> int:

    arguments = (
        build_argument_parser()
        .parse_args()
    )

    return (
        0
        if run_self_test(
            arguments.scenario
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
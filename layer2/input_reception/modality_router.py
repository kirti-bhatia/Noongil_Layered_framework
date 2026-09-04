"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Modality Router
File    : layer2/input_reception/modality_router.py
============================================================

Purpose
-------
Routes available Layer 1 modalities to appropriate Layer 2
perception modules.

Examples:
- Vision -> scene classification, object detection, OCR, depth
- Audio -> speech recognition and sound-event detection
- Motion -> activity analysis
- Spatial -> spatial context and activity support

The router:
- Does not execute perception models
- Does not create entities, events or graphs
- Does not perform reasoning
- Does not generate final Layer 2 output

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import argparse
import copy
import uuid

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from layer2.config.settings import (
    Layer2Settings,
    create_default_settings,
    create_test_settings,
)

from layer2.input_reception.layer1_packet_adapter import (
    AdaptedLayer1Input,
    AdaptedModality,
    Layer1PacketAdapter,
)

from layer2.utils.exceptions import (
    InputReceptionError,
    MissingModalityError,
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

ROUTER_VERSION = "1.0"


# ============================================================
# ENUMERATIONS
# ============================================================

class RouteStatus(str, Enum):
    """Routing status of one Layer 2 task."""

    READY = "ready"
    DEGRADED = "degraded"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ModuleRoute:
    """Routing decision for one Layer 2 module."""

    task: str
    status: RouteStatus

    required_modalities: List[str] = field(
        default_factory=list
    )

    optional_modalities: List[str] = field(
        default_factory=list
    )

    routed_modalities: List[str] = field(
        default_factory=list
    )

    missing_modalities: List[str] = field(
        default_factory=list
    )

    dependencies: List[str] = field(
        default_factory=list
    )

    reason: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def executable(self) -> bool:
        """Return whether the module may execute."""

        return self.status in {
            RouteStatus.READY,
            RouteStatus.DEGRADED,
        }

    def to_dict(self) -> Dict[str, Any]:

        return {
            "task": self.task,
            "status": self.status.value,
            "executable": self.executable,
            "required_modalities": list(
                self.required_modalities
            ),
            "optional_modalities": list(
                self.optional_modalities
            ),
            "routed_modalities": list(
                self.routed_modalities
            ),
            "missing_modalities": list(
                self.missing_modalities
            ),
            "dependencies": list(
                self.dependencies
            ),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass
class RoutingPlan:
    """Complete routing plan for one packet."""

    packet_id: str
    plan_id: str

    routes: Dict[str, ModuleRoute]

    available_modalities: List[str]
    usable_modalities: List[str]

    warnings: List[str] = field(
        default_factory=list
    )

    router_version: str = ROUTER_VERSION

    @property
    def executable_tasks(self) -> List[str]:

        return sorted(
            task
            for task, route
            in self.routes.items()
            if route.executable
        )

    @property
    def skipped_tasks(self) -> List[str]:

        return sorted(
            task
            for task, route
            in self.routes.items()
            if route.status
            == RouteStatus.SKIPPED
        )

    @property
    def blocked_tasks(self) -> List[str]:

        return sorted(
            task
            for task, route
            in self.routes.items()
            if route.status
            == RouteStatus.BLOCKED
        )

    @property
    def degraded_tasks(self) -> List[str]:

        return sorted(
            task
            for task, route
            in self.routes.items()
            if route.status
            == RouteStatus.DEGRADED
        )

    @property
    def can_execute_pipeline(self) -> bool:
        """Return whether useful perception can run."""

        semantic_tasks = {
            "scene_classification",
            "object_detection",
            "activity_recognition",
            "ocr",
            "speech_recognition",
            "sound_event_detection",
            "depth_estimation",
            "motion_analysis",
        }

        return any(
            task in semantic_tasks
            for task in self.executable_tasks
        )

    def get_route(
        self,
        task: str,
    ) -> ModuleRoute:
        """Return one routing decision."""

        normalized_task = task.strip().lower()

        if normalized_task not in self.routes:
            raise InputReceptionError(
                f"Unknown routing task: "
                f"{normalized_task!r}",
                module="modality_router",
                details={
                    "packet_id": self.packet_id,
                    "task": normalized_task,
                },
            )

        return self.routes[normalized_task]

    def to_dict(self) -> Dict[str, Any]:

        return {
            "packet_id": self.packet_id,
            "plan_id": self.plan_id,
            "available_modalities": list(
                self.available_modalities
            ),
            "usable_modalities": list(
                self.usable_modalities
            ),
            "can_execute_pipeline": (
                self.can_execute_pipeline
            ),
            "executable_tasks": (
                self.executable_tasks
            ),
            "degraded_tasks": (
                self.degraded_tasks
            ),
            "skipped_tasks": self.skipped_tasks,
            "blocked_tasks": self.blocked_tasks,
            "routes": {
                task: route.to_dict()
                for task, route
                in self.routes.items()
            },
            "warnings": list(self.warnings),
            "router_version": (
                self.router_version
            ),
        }


# ============================================================
# MODALITY ROUTER
# ============================================================

class ModalityRouter:
    """Create routing plans for Layer 2 modules."""

    def __init__(
        self,
        settings: Optional[
            Layer2Settings
        ] = None,
        *,
        logger: Optional[
            Layer2LoggerAdapter
        ] = None,
    ) -> None:

        self.settings = (
            settings
            or create_default_settings()
        )

        self.settings.validate()

        self.logger = (
            logger
            or get_logger(
                "modality_router"
            )
        )

    def route(
        self,
        packet: AdaptedLayer1Input,
    ) -> RoutingPlan:
        """Build a routing plan for one packet."""

        if not isinstance(
            packet,
            AdaptedLayer1Input,
        ):
            raise InputReceptionError(
                "packet must be an "
                "AdaptedLayer1Input.",
                module="modality_router",
                details={
                    "received_type": (
                        packet.__class__.__name__
                    )
                },
            )

        usable = set(
            packet.usable_modalities
        )

        if not usable:
            raise MissingModalityError(
                "The packet contains no usable "
                "modalities.",
                module="modality_router",
                recoverable=True,
                details={
                    "packet_id": packet.packet_id,
                    "available_modalities": (
                        packet.available_modalities
                    ),
                },
            )

        log_event(
            self.logger,
            event="routing_started",
            message=(
                "Creating modality routing plan."
            ),
            details={
                "packet_id": packet.packet_id,
                "usable_modalities": (
                    sorted(usable)
                ),
            },
        )

        routes: Dict[str, ModuleRoute] = {}

        # Vision pipeline
        routes["vision_processing"] = (
            self._single_modality_route(
                task="vision_processing",
                modality="vision",
                enabled=(
                    self.settings.modules
                    .vision_processing
                ),
            )
        )

        routes["scene_classification"] = (
            self._single_modality_route(
                task="scene_classification",
                modality="vision",
                enabled=(
                    self.settings.modules
                    .scene_classification
                ),
                dependencies=[
                    "vision_processing"
                ],
            )
        )

        routes["object_detection"] = (
            self._single_modality_route(
                task="object_detection",
                modality="vision",
                enabled=(
                    self.settings.modules
                    .object_detection
                ),
                dependencies=[
                    "vision_processing"
                ],
            )
        )

        routes["object_tracking"] = (
            self._single_modality_route(
                task="object_tracking",
                modality="vision",
                enabled=(
                    self.settings.modules
                    .object_tracking
                ),
                dependencies=[
                    "object_detection"
                ],
            )
        )

        routes["activity_recognition"] = (
            self._alternative_route(
                task="activity_recognition",
                preferred_modality="motion",
                fallback_modalities=[
                    "vision",
                    "spatial",
                ],
                enabled=(
                    self.settings.modules
                    .activity_recognition
                ),
                dependencies=[],
            )
        )

        # Text pipeline
        routes["ocr"] = (
            self._single_modality_route(
                task="ocr",
                modality="vision",
                enabled=(
                    self.settings.modules.ocr
                ),
                dependencies=[
                    "vision_processing"
                ],
            )
        )

        routes["text_interpretation"] = (
            self._single_modality_route(
                task="text_interpretation",
                modality="vision",
                enabled=(
                    self.settings.modules
                    .text_interpretation
                ),
                dependencies=["ocr"],
            )
        )

        # Audio pipeline
        routes["audio_processing"] = (
            self._single_modality_route(
                task="audio_processing",
                modality="audio",
                enabled=(
                    self.settings.modules
                    .audio_processing
                ),
            )
        )

        routes["speech_recognition"] = (
            self._single_modality_route(
                task="speech_recognition",
                modality="audio",
                enabled=(
                    self.settings.modules
                    .speech_recognition
                ),
                dependencies=[
                    "audio_processing"
                ],
            )
        )

        routes["sound_event_detection"] = (
            self._single_modality_route(
                task="sound_event_detection",
                modality="audio",
                enabled=(
                    self.settings.modules
                    .sound_event_detection
                ),
                dependencies=[
                    "audio_processing"
                ],
            )
        )

        # Spatial pipeline
        routes["depth_estimation"] = (
            self._single_modality_route(
                task="depth_estimation",
                modality="vision",
                enabled=(
                    self.settings.modules
                    .depth_estimation
                ),
                optional_modalities=[
                    "spatial"
                ],
                dependencies=[
                    "vision_processing"
                ],
            )
        )

        routes["obstacle_detection"] = (
            self._single_modality_route(
                task="obstacle_detection",
                modality="vision",
                enabled=(
                    self.settings.modules
                    .obstacle_detection
                ),
                optional_modalities=[
                    "spatial"
                ],
                dependencies=[
                    "object_detection",
                    "depth_estimation",
                ],
            )
        )

        # Motion pipeline
        routes["motion_analysis"] = (
            self._single_modality_route(
                task="motion_analysis",
                modality="motion",
                enabled=(
                    self.settings.modules
                    .motion_analysis
                ),
                optional_modalities=[
                    "spatial",
                    "vision",
                ],
            )
        )

        # Routes above were constructed before applying packet
        # availability. Apply it now.
        for route in routes.values():
            self._apply_availability(
                route,
                usable,
            )

        # Confidence calibration depends on any usable result.
        confidence_enabled = (
            self.settings.modules
            .confidence_calibration
        )

        executable_before_confidence = [
            task
            for task, route
            in routes.items()
            if route.executable
        ]

        if not confidence_enabled:
            routes[
                "confidence_calibration"
            ] = self._disabled_route(
                "confidence_calibration"
            )

        elif executable_before_confidence:
            routes[
                "confidence_calibration"
            ] = ModuleRoute(
                task=(
                    "confidence_calibration"
                ),
                status=RouteStatus.READY,
                required_modalities=[],
                routed_modalities=sorted(
                    usable
                ),
                dependencies=sorted(
                    executable_before_confidence
                ),
                reason=(
                    "Perception results are "
                    "available for calibration."
                ),
            )

        else:
            routes[
                "confidence_calibration"
            ] = ModuleRoute(
                task=(
                    "confidence_calibration"
                ),
                status=RouteStatus.SKIPPED,
                reason=(
                    "No perception result is "
                    "available for calibration."
                ),
            )

        # Fusion route
        routes["multimodal_fusion"] = (
            self._build_fusion_route(
                usable,
                routes,
            )
        )

        # Final output building route
        perception_dependencies = [
            task
            for task, route
            in routes.items()
            if (
                route.executable
                and task
                not in {
                    "vision_processing",
                    "audio_processing",
                    "confidence_calibration",
                    "multimodal_fusion",
                }
            )
        ]

        if perception_dependencies:
            routes[
                "perception_output_builder"
            ] = ModuleRoute(
                task=(
                    "perception_output_builder"
                ),
                status=RouteStatus.READY,
                routed_modalities=sorted(
                    usable
                ),
                dependencies=sorted(
                    set(
                        perception_dependencies
                        + [
                            "confidence_calibration",
                            "multimodal_fusion",
                        ]
                    )
                ),
                reason=(
                    "At least one semantic "
                    "perception module can execute."
                ),
            )
        else:
            routes[
                "perception_output_builder"
            ] = ModuleRoute(
                task=(
                    "perception_output_builder"
                ),
                status=RouteStatus.BLOCKED,
                reason=(
                    "No semantic perception "
                    "module can execute."
                ),
            )

        warnings = []

        for task, route in routes.items():
            if route.status in {
                RouteStatus.DEGRADED,
                RouteStatus.SKIPPED,
                RouteStatus.BLOCKED,
            }:
                warnings.append(
                    f"{task}: {route.reason}"
                )

        plan = RoutingPlan(
            packet_id=packet.packet_id,
            plan_id=(
                "ROUTE_"
                f"{uuid.uuid4().hex[:12].upper()}"
            ),
            routes=routes,
            available_modalities=(
                packet.available_modalities
            ),
            usable_modalities=(
                packet.usable_modalities
            ),
            warnings=warnings,
        )

        log_event(
            self.logger,
            event="routing_completed",
            message=(
                "Modality routing plan created."
            ),
            details={
                "packet_id": packet.packet_id,
                "plan_id": plan.plan_id,
                "executable_tasks": (
                    plan.executable_tasks
                ),
                "skipped_tasks": (
                    plan.skipped_tasks
                ),
                "degraded_tasks": (
                    plan.degraded_tasks
                ),
            },
        )

        return plan

    def _single_modality_route(
        self,
        *,
        task: str,
        modality: str,
        enabled: bool,
        optional_modalities: Optional[
            List[str]
        ] = None,
        dependencies: Optional[
            List[str]
        ] = None,
    ) -> ModuleRoute:

        if not enabled:
            return self._disabled_route(
                task
            )

        return ModuleRoute(
            task=task,
            status=RouteStatus.READY,
            required_modalities=[
                modality
            ],
            optional_modalities=(
                optional_modalities or []
            ),
            dependencies=(
                dependencies or []
            ),
        )

    def _alternative_route(
        self,
        *,
        task: str,
        preferred_modality: str,
        fallback_modalities: List[str],
        enabled: bool,
        dependencies: Optional[
            List[str]
        ] = None,
    ) -> ModuleRoute:

        if not enabled:
            return self._disabled_route(
                task
            )

        return ModuleRoute(
            task=task,
            status=RouteStatus.READY,
            required_modalities=[
                preferred_modality
            ],
            optional_modalities=list(
                fallback_modalities
            ),
            dependencies=(
                dependencies or []
            ),
            metadata={
                "allow_fallback": True,
                "preferred_modality": (
                    preferred_modality
                ),
            },
        )

    def _disabled_route(
        self,
        task: str,
    ) -> ModuleRoute:

        return ModuleRoute(
            task=task,
            status=RouteStatus.SKIPPED,
            reason=(
                "Module is disabled in "
                "Layer 2 settings."
            ),
            metadata={
                "disabled_by_settings": True
            },
        )

    def _apply_availability(
        self,
        route: ModuleRoute,
        usable_modalities: Set[str],
    ) -> None:

        if route.metadata.get(
            "disabled_by_settings"
        ):
            return

        required = set(
            route.required_modalities
        )

        optional = set(
            route.optional_modalities
        )

        available_required = (
            required & usable_modalities
        )

        available_optional = (
            optional & usable_modalities
        )

        missing_required = (
            required - usable_modalities
        )

        allow_fallback = bool(
            route.metadata.get(
                "allow_fallback",
                False,
            )
        )

        if not missing_required:
            route.status = RouteStatus.READY

            route.routed_modalities = sorted(
                available_required
                | available_optional
            )

            route.missing_modalities = []

            route.reason = (
                "Required modality is available."
            )

            return

        if allow_fallback and available_optional:
            route.status = RouteStatus.DEGRADED

            route.routed_modalities = sorted(
                available_optional
            )

            route.missing_modalities = sorted(
                missing_required
            )

            route.reason = (
                "Preferred modality is unavailable; "
                "fallback modality will be used."
            )

            return

        route.status = RouteStatus.SKIPPED

        route.routed_modalities = sorted(
            available_optional
        )

        route.missing_modalities = sorted(
            missing_required
        )

        route.reason = (
            "Required modality is unavailable: "
            f"{', '.join(sorted(missing_required))}"
        )

    def _build_fusion_route(
        self,
        usable_modalities: Set[str],
        routes: Dict[str, ModuleRoute],
    ) -> ModuleRoute:

        if not (
            self.settings.modules
            .multimodal_fusion
        ):
            return self._disabled_route(
                "multimodal_fusion"
            )

        fusion_modalities = {
            modality
            for modality in usable_modalities
            if modality in {
                "vision",
                "audio",
                "spatial",
                "motion",
            }
        }

        text_route = routes.get("ocr")

        if (
            text_route is not None
            and text_route.executable
        ):
            fusion_modalities.add("text")

        minimum_modalities = (
            self.settings.fusion
            .minimum_modalities
        )

        if (
            len(fusion_modalities)
            >= minimum_modalities
        ):
            status = RouteStatus.READY
            reason = (
                "Minimum fusion modality "
                "requirement is satisfied."
            )
        elif (
            fusion_modalities
            and self.settings.fusion
            .allow_partial_fusion
        ):
            status = RouteStatus.DEGRADED
            reason = (
                "Fusion will continue with "
                "partial modality availability."
            )
        else:
            status = RouteStatus.SKIPPED
            reason = (
                "Insufficient modalities for "
                "multimodal fusion."
            )

        dependencies = [
            task
            for task, route
            in routes.items()
            if route.executable
        ]

        return ModuleRoute(
            task="multimodal_fusion",
            status=status,
            required_modalities=[],
            optional_modalities=[
                "vision",
                "audio",
                "spatial",
                "motion",
                "text",
            ],
            routed_modalities=sorted(
                fusion_modalities
            ),
            dependencies=sorted(
                dependencies
            ),
            reason=reason,
            metadata={
                "minimum_modalities": (
                    minimum_modalities
                ),
                "available_count": len(
                    fusion_modalities
                ),
            },
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def route_modalities(
    packet: AdaptedLayer1Input,
    *,
    settings: Optional[
        Layer2Settings
    ] = None,
) -> RoutingPlan:

    router = ModalityRouter(
        settings=settings
    )

    return router.route(packet)


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 72)
    print("NOONGIL-X | MODALITY ROUTER SELF-TEST")
    print("=" * 72)

    try:
        adapter = Layer1PacketAdapter(
            require_media=True
        )

        settings = create_test_settings()

        router = ModalityRouter(
            settings=settings
        )

        scenarios = (
            adapter.discover_scenarios()
        )

        if len(scenarios) != 8:
            raise AssertionError(
                "Expected eight test scenarios."
            )

        print(
            "[PASS] Eight scenarios discovered"
        )

        plans = []

        for scenario in scenarios:

            packet = adapter.load_scenario(
                scenario
            )

            plan = router.route(packet)

            if not plan.can_execute_pipeline:
                raise AssertionError(
                    f"Pipeline cannot execute for "
                    f"{scenario}."
                )

            if not plan.get_route(
                "vision_processing"
            ).executable:
                raise AssertionError(
                    f"Vision route failed for "
                    f"{scenario}."
                )

            if not plan.get_route(
                "audio_processing"
            ).executable:
                raise AssertionError(
                    f"Audio route failed for "
                    f"{scenario}."
                )

            if not plan.get_route(
                "perception_output_builder"
            ).executable:
                raise AssertionError(
                    f"Output builder route failed "
                    f"for {scenario}."
                )

            plans.append(plan)

            print(
                f"[PASS] {scenario}: "
                f"{len(plan.executable_tasks)} "
                "executable tasks"
            )

        print(
            "[PASS] All complete scenarios routed"
        )

        # Test graceful audio degradation.
        degraded_packet = (
            adapter.load_scenario(
                "park_walking"
            )
        )

        degraded_packet.modalities[
            "audio"
        ] = AdaptedModality(
            name="audio",
            available=False,
            data={},
            media_path=None,
            warnings=[
                "Simulated missing audio."
            ],
        )

        degraded_plan = router.route(
            degraded_packet
        )

        if degraded_plan.get_route(
            "speech_recognition"
        ).status != RouteStatus.SKIPPED:
            raise AssertionError(
                "Speech recognition was not "
                "skipped when audio was missing."
            )

        if not degraded_plan.get_route(
            "scene_classification"
        ).executable:
            raise AssertionError(
                "Vision processing was incorrectly "
                "blocked by missing audio."
            )

        if not (
            degraded_plan.can_execute_pipeline
        ):
            raise AssertionError(
                "Pipeline should continue without "
                "audio."
            )

        print(
            "[PASS] Missing audio handled gracefully"
        )

        # Test disabled OCR settings.
        disabled_settings = copy.deepcopy(
            settings
        )

        disabled_settings.modules.ocr = False
        disabled_settings.modules\
            .text_interpretation = False

        disabled_settings.validate()

        disabled_router = ModalityRouter(
            settings=disabled_settings
        )

        disabled_plan = (
            disabled_router.route(
                adapter.load_scenario(
                    "classroom"
                )
            )
        )

        if disabled_plan.get_route(
            "ocr"
        ).status != RouteStatus.SKIPPED:
            raise AssertionError(
                "Disabled OCR route was not "
                "skipped."
            )

        print(
            "[PASS] Disabled modules respected"
        )

        example_plan = plans[0]

        print("\nExample routing summary:")
        print(
            f"  packet_id: "
            f"{example_plan.packet_id}"
        )
        print(
            f"  plan_id: "
            f"{example_plan.plan_id}"
        )
        print(
            f"  usable_modalities: "
            f"{example_plan.usable_modalities}"
        )
        print(
            f"  executable_tasks: "
            f"{example_plan.executable_tasks}"
        )
        print(
            f"  degraded_tasks: "
            f"{example_plan.degraded_tasks}"
        )
        print(
            f"  skipped_tasks: "
            f"{example_plan.skipped_tasks}"
        )
        print(
            f"  can_execute_pipeline: "
            f"{example_plan.can_execute_pipeline}"
        )

        print("\n" + "=" * 72)
        print(
            "[PASSED] MODALITY ROUTER IS WORKING"
        )
        print("=" * 72)

        return True

    except (
        InputReceptionError,
        MissingModalityError,
        AssertionError,
    ) as error:

        log_exception(
            get_logger(
                "modality_router_self_test"
            ),
            error,
            event="router_self_test_failed",
        )

        print(f"\n[FAILED] {error}")
        print("=" * 72)

        return False


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:

    return argparse.ArgumentParser(
        description=(
            "Run the NOONGIL-X Layer 2 "
            "modality-router self-test."
        )
    )


def main() -> int:

    build_argument_parser().parse_args()

    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
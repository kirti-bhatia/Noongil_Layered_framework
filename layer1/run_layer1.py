"""
============================================================
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : Master Pipeline Orchestrator
File    : layer1/run_layer1.py
============================================================

Purpose
-------
Execute the complete NOONGIL-X Layer 1 pipeline.

Current supported source mode
-----------------------------
- simulation

Future source modes
-------------------
- live
- replay

Pipeline
--------
PhoneSensorSimulator
    -> MultimodalReceiver
    -> VisionInputProcessor
    -> AudioInputProcessor
    -> SpatialInputProcessor
    -> MotionInputProcessor
    -> InteractionInputProcessor
    -> DeviceInputProcessor
    -> NAMARAController
    -> MultimodalSynchronizer
    -> ConfidenceEstimator
    -> MissingModalityHandler
    -> SensorPacketBuilder
    -> OutputDispatcher
    -> output/layer2/layer1_sensor_packet.json

Architectural Boundary
----------------------
This file coordinates modules. It does not reimplement their
algorithms.

Compatibility
-------------
Python 3.10+
============================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from layer1.acquisition.multimodal_receiver import (
    MultimodalReceiver,
)
from layer1.acquisition.phone_sensor_simulator import (
    PhoneSensorSimulator,
    PhoneSimulatorConfig,
    SimulationScenario,
)
from layer1.config import paths as path_config

LAYER2_INPUT_PACKET_PATH = (
    path_config.LAYER2_INPUT_PACKET_PATH
)
MULTIMODAL_SENSOR_PACKET_PATH = (
    path_config.MULTIMODAL_SENSOR_PACKET_PATH
)
LAYER1_PIPELINE_SUMMARY_PATH = getattr(
    path_config,
    "LAYER1_PIPELINE_SUMMARY_PATH",
    getattr(
        path_config,
        "PIPELINE_SUMMARY_PATH",
        Path(
            "output/layer1/"
            "layer1_pipeline_summary.json"
        ),
    ),
)
ensure_parent_directory = (
    path_config.ensure_parent_directory
)


def ensure_required_directories() -> None:
    """
    Compatibility wrapper for different paths.py versions.
    """

    for function_name in (
        "ensure_required_directories",
        "create_required_directories",
        "ensure_output_directories",
    ):
        function = getattr(
            path_config,
            function_name,
            None,
        )

        if callable(function):
            function()
            return

    for required_path in (
        Path(MULTIMODAL_SENSOR_PACKET_PATH).parent,
        Path(LAYER2_INPUT_PACKET_PATH).parent,
        Path(LAYER1_PIPELINE_SUMMARY_PATH).parent,
    ):
        required_path.mkdir(
            parents=True,
            exist_ok=True,
        )
from layer1.config.settings import (
    Layer1Settings,
    create_default_settings,
    create_test_settings,
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
from layer1.output.output_dispatcher import (
    DispatchResult,
    OutputDispatcher,
)
from layer1.output.sensor_packet_builder import (
    PacketBuildResult,
    SensorPacketBuilder,
)
from layer1.processing.confidence_estimator import (
    ConfidenceEstimator,
    ConfidenceReport,
)
from layer1.processing.missing_modality_handler import (
    MissingModalityHandler,
    RecoveryResult,
)
from layer1.processing.multimodal_synchronizer import (
    MultimodalSynchronizer,
    SynchronizedMultimodalFrame,
)
from layer1.processing.namara_controller import (
    ModalityObservation,
    NAMARAContext,
    NAMARAController,
    NAMARAPlan,
)
from layer1.schemas.sensor_packet import (
    AcquisitionMode,
)
from layer1.utils.logger import (
    PipelineTimer,
    get_logger,
    log_exception,
)


# ============================================================
# CONSTANTS
# ============================================================

RUNNER_VERSION = "1.0"
DEFAULT_RANDOM_SEED = 42
DEFAULT_CYCLES = 1
DEFAULT_INTERVAL_SECONDS = 1.0


# ============================================================
# EXCEPTIONS
# ============================================================

class Layer1RunnerError(Exception):
    """Base exception for the Layer 1 master pipeline."""


class Layer1ConfigurationError(Layer1RunnerError):
    """Raised when the runner configuration is invalid."""


class Layer1StageError(Layer1RunnerError):
    """Raised when a pipeline stage fails."""


# ============================================================
# ENUMERATIONS
# ============================================================

class SourceMode(str, Enum):
    SIMULATION = "simulation"
    LIVE = "live"
    REPLAY = "replay"


class PipelineStatus(str, Enum):
    PASSED = "passed"
    PARTIAL = "partial"
    FAILED = "failed"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Layer1RunConfig:
    """
    Runtime configuration for the Layer 1 orchestrator.
    """

    source_mode: SourceMode = SourceMode.SIMULATION
    scenario: SimulationScenario = (
        SimulationScenario.NAVIGATION
    )
    acquisition_mode: AcquisitionMode = (
        AcquisitionMode.NAVIGATION
    )

    cycles: int = DEFAULT_CYCLES
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    random_seed: int = DEFAULT_RANDOM_SEED

    urgency: float = 0.55
    emergency_active: bool = False
    user_interaction_active: bool = True

    write_outputs: bool = True
    archive_previous: bool = True
    verify_dispatch: bool = True

    use_test_settings: bool = False
    stop_on_error: bool = True
    print_packet: bool = False

    def validate(self) -> None:
        if self.cycles <= 0:
            raise Layer1ConfigurationError(
                "cycles must be greater than zero."
            )

        if self.interval_seconds < 0:
            raise Layer1ConfigurationError(
                "interval_seconds cannot be negative."
            )

        if not 0.0 <= self.urgency <= 1.0:
            raise Layer1ConfigurationError(
                "urgency must be between 0.0 and 1.0."
            )

        if self.source_mode != SourceMode.SIMULATION:
            raise Layer1ConfigurationError(
                f"Source mode {self.source_mode.value!r} is not "
                "implemented yet. Use 'simulation'."
            )


@dataclass
class StageRecord:
    """
    Runtime record for one pipeline stage.
    """

    stage: str
    success: bool
    elapsed_seconds: float
    details: Dict[str, Any] = field(
        default_factory=dict
    )
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CycleResult:
    """
    Result of one complete Layer 1 cycle.
    """

    cycle_id: str
    cycle_number: int
    started_at: str
    completed_at: str

    status: PipelineStatus

    stage_records: List[StageRecord]

    generated_packet_count: int
    accepted_packet_count: int
    rejected_packet_count: int

    frame_id: Optional[str]
    confidence_report_id: Optional[str]
    recovery_report_id: Optional[str]
    namara_plan_id: Optional[str]
    layer1_packet_id: Optional[str]
    dispatch_id: Optional[str]

    overall_confidence: Optional[float]
    effective_confidence: Optional[float]

    primary_output_path: Optional[str]
    layer2_output_path: Optional[str]

    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["stage_records"] = [
            record.to_dict()
            for record in self.stage_records
        ]
        return payload


@dataclass
class Layer1PipelineSummary:
    """
    Summary of one runner invocation.
    """

    run_id: str
    runner_version: str

    started_at: str
    completed_at: str

    source_mode: str
    scenario: str
    acquisition_mode: str

    requested_cycles: int
    completed_cycles: int
    successful_cycles: int
    failed_cycles: int

    status: PipelineStatus

    official_primary_output: str
    official_layer2_output: str

    cycle_results: List[CycleResult]

    total_elapsed_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["cycle_results"] = [
            result.to_dict()
            for result in self.cycle_results
        ]
        return payload


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


def enum_from_value(
    enum_type: type[Enum],
    value: str,
) -> Enum:
    normalized = value.strip().lower()

    for item in enum_type:
        if item.value.lower() == normalized:
            return item

    valid_values = ", ".join(
        item.value
        for item in enum_type
    )

    raise argparse.ArgumentTypeError(
        f"Invalid value {value!r}. "
        f"Choose one of: {valid_values}"
    )


def write_json_atomic(
    path: Path,
    payload: Dict[str, Any],
) -> None:
    ensure_parent_directory(path)

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


# ============================================================
# LAYER 1 RUNNER
# ============================================================

class Layer1Runner:
    """
    Master orchestration class for NOONGIL-X Layer 1.
    """

    def __init__(
        self,
        config: Optional[Layer1RunConfig] = None,
        settings: Optional[Layer1Settings] = None,
    ) -> None:
        self.config = config or Layer1RunConfig()
        self.config.validate()

        if settings is not None:
            self.settings = settings
        elif self.config.use_test_settings:
            self.settings = create_test_settings()
        else:
            self.settings = create_default_settings()

        self.settings.validate()

        ensure_required_directories()

        self.logger = get_logger(
            "run_layer1"
        )

        self.receiver = MultimodalReceiver(
            self.settings
        )

        self.vision_processor = (
            VisionInputProcessor(
                self.settings
            )
        )
        self.audio_processor = (
            AudioInputProcessor(
                self.settings
            )
        )
        self.spatial_processor = (
            SpatialInputProcessor(
                self.settings
            )
        )
        self.motion_processor = (
            MotionInputProcessor(
                self.settings
            )
        )
        self.interaction_processor = (
            InteractionInputProcessor(
                self.settings
            )
        )
        self.device_processor = (
            DeviceInputProcessor(
                self.settings
            )
        )

        self.namara_controller = (
            NAMARAController(
                self.settings
            )
        )
        self.synchronizer = (
            MultimodalSynchronizer(
                self.settings
            )
        )
        self.confidence_estimator = (
            ConfidenceEstimator(
                self.settings
            )
        )
        self.recovery_handler = (
            MissingModalityHandler(
                self.settings
            )
        )

        self.packet_builder = (
            SensorPacketBuilder(
                self.settings
            )
        )
        self.output_dispatcher = (
            OutputDispatcher(
                self.settings
            )
        )

        self.simulator = self._create_simulator()

    # ========================================================
    # SOURCE CREATION
    # ========================================================

    def _create_simulator(
        self,
    ) -> PhoneSensorSimulator:
        if (
            self.config.source_mode
            != SourceMode.SIMULATION
        ):
            raise Layer1ConfigurationError(
                "Only simulation source mode is currently "
                "implemented."
            )

        simulator_config = PhoneSimulatorConfig(
            scenario=self.config.scenario,
            random_seed=self.config.random_seed,
        )

        return PhoneSensorSimulator(
            simulator_config
        )

    # ========================================================
    # PIPELINE EXECUTION
    # ========================================================

    def run(
        self,
    ) -> Layer1PipelineSummary:
        run_id = (
            "L1_RUN_"
            f"{uuid.uuid4().hex[:14].upper()}"
        )

        started_at = utc_now_iso()
        started = time.perf_counter()

        cycle_results: List[
            CycleResult
        ] = []

        self.receiver.start()

        try:
            for cycle_number in range(
                1,
                self.config.cycles + 1,
            ):
                try:
                    cycle_result = self.run_cycle(
                        cycle_number
                    )
                    cycle_results.append(
                        cycle_result
                    )

                except Exception as error:
                    failed_result = CycleResult(
                        cycle_id=(
                            "L1_CYCLE_"
                            f"{uuid.uuid4().hex[:12].upper()}"
                        ),
                        cycle_number=cycle_number,
                        started_at=utc_now_iso(),
                        completed_at=utc_now_iso(),
                        status=PipelineStatus.FAILED,
                        stage_records=[],
                        generated_packet_count=0,
                        accepted_packet_count=0,
                        rejected_packet_count=0,
                        frame_id=None,
                        confidence_report_id=None,
                        recovery_report_id=None,
                        namara_plan_id=None,
                        layer1_packet_id=None,
                        dispatch_id=None,
                        overall_confidence=None,
                        effective_confidence=None,
                        primary_output_path=None,
                        layer2_output_path=None,
                        error=(
                            f"{type(error).__name__}: {error}"
                        ),
                    )

                    cycle_results.append(
                        failed_result
                    )

                    log_exception(
                        self.logger,
                        "Layer 1 cycle failed",
                        error=error,
                        details={
                            "run_id": run_id,
                            "cycle_number": cycle_number,
                        },
                    )

                    if self.config.stop_on_error:
                        break

                if (
                    cycle_number
                    < self.config.cycles
                    and self.config.interval_seconds > 0
                ):
                    time.sleep(
                        self.config.interval_seconds
                    )

        finally:
            self.receiver.stop()

        completed_at = utc_now_iso()
        elapsed = time.perf_counter() - started

        successful_cycles = sum(
            1
            for result in cycle_results
            if result.status
            != PipelineStatus.FAILED
        )

        failed_cycles = sum(
            1
            for result in cycle_results
            if result.status
            == PipelineStatus.FAILED
        )

        if failed_cycles == 0:
            status = PipelineStatus.PASSED
        elif successful_cycles > 0:
            status = PipelineStatus.PARTIAL
        else:
            status = PipelineStatus.FAILED

        summary = Layer1PipelineSummary(
            run_id=run_id,
            runner_version=RUNNER_VERSION,
            started_at=started_at,
            completed_at=completed_at,
            source_mode=(
                self.config.source_mode.value
            ),
            scenario=(
                self.config.scenario.value
            ),
            acquisition_mode=(
                self.config
                .acquisition_mode.value
            ),
            requested_cycles=self.config.cycles,
            completed_cycles=len(
                cycle_results
            ),
            successful_cycles=successful_cycles,
            failed_cycles=failed_cycles,
            status=status,
            official_primary_output=str(
                MULTIMODAL_SENSOR_PACKET_PATH
            ),
            official_layer2_output=str(
                LAYER2_INPUT_PACKET_PATH
            ),
            cycle_results=cycle_results,
            total_elapsed_seconds=round(
                elapsed,
                6,
            ),
        )

        write_json_atomic(
            Path(
                LAYER1_PIPELINE_SUMMARY_PATH
            ),
            summary.to_dict(),
        )

        return summary

    def run_cycle(
        self,
        cycle_number: int,
    ) -> CycleResult:
        cycle_id = (
            "L1_CYCLE_"
            f"{uuid.uuid4().hex[:12].upper()}"
        )

        cycle_started_at = utc_now_iso()
        stage_records: List[
            StageRecord
        ] = []

        generated_packet_count = 0
        accepted_packet_count = 0
        rejected_packet_count = 0

        frame: Optional[
            SynchronizedMultimodalFrame
        ] = None
        confidence_report: Optional[
            ConfidenceReport
        ] = None
        recovery_result: Optional[
            RecoveryResult
        ] = None
        namara_plan: Optional[
            NAMARAPlan
        ] = None
        build_result: Optional[
            PacketBuildResult
        ] = None
        dispatch_result: Optional[
            DispatchResult
        ] = None

        try:
            # ------------------------------------------------
            # 1. Acquire packets
            # ------------------------------------------------
            stage_started = time.perf_counter()

            packets = self.simulator.generate_cycle()
            generated_packet_count = len(
                packets
            )

            receipts = self.receiver.receive_batch(
                packets,
                raise_on_error=True,
            )

            accepted_packet_count = sum(
                1
                for receipt in receipts
                if receipt.accepted
            )
            rejected_packet_count = (
                len(receipts)
                - accepted_packet_count
            )

            if rejected_packet_count:
                raise Layer1StageError(
                    f"{rejected_packet_count} packets "
                    "were rejected by the receiver."
                )

            stage_records.append(
                StageRecord(
                    stage="acquisition_and_routing",
                    success=True,
                    elapsed_seconds=round(
                        time.perf_counter()
                        - stage_started,
                        6,
                    ),
                    details={
                        "generated_packets": (
                            generated_packet_count
                        ),
                        "accepted_packets": (
                            accepted_packet_count
                        ),
                        "rejected_packets": (
                            rejected_packet_count
                        ),
                    },
                )
            )

            # ------------------------------------------------
            # 2. Process modalities
            # ------------------------------------------------
            stage_started = time.perf_counter()

            vision_result = (
                self.vision_processor
                .process_latest_from_receiver(
                    self.receiver,
                    raise_on_error=True,
                )
            )

            audio_result = (
                self.audio_processor
                .process_latest_from_receiver(
                    self.receiver,
                    raise_on_error=True,
                )
            )

            spatial_result = (
                self.spatial_processor
                .process_latest_from_receiver(
                    self.receiver,
                    raise_on_error=True,
                )
            )

            motion_result = (
                self.motion_processor
                .process_receiver_queue(
                    self.receiver,
                    raise_on_error=True,
                )
            )

            interaction_result = (
                self.interaction_processor
                .process_latest_from_receiver(
                    self.receiver,
                    raise_on_error=True,
                )
            )

            device_results = (
                self.device_processor
                .process_receiver_queue(
                    self.receiver,
                    raise_on_error=True,
                )
            )

            # Only continuously sampled core modalities are required
            # at this stage. Interaction is event-driven and may
            # legitimately be absent in a cycle. It must be passed as
            # None to the synchronizer so cached reuse or explicit
            # missing-modality handling can occur downstream.
            required_results = {
                "vision": vision_result,
                "audio": audio_result,
                "spatial": spatial_result,
                "motion": motion_result,
            }

            missing_results = [
                name
                for name, value
                in required_results.items()
                if value is None
            ]

            if missing_results:
                raise Layer1StageError(
                    "Missing required modality processing results: "
                    f"{missing_results}"
                )

            optional_results = {
                "interaction": interaction_result,
                "device": device_results,
            }

            unavailable_optional_results = [
                name
                for name, value
                in optional_results.items()
                if value is None
                or (
                    name == "device"
                    and isinstance(value, list)
                    and len(value) == 0
                )
            ]

            stage_records.append(
                StageRecord(
                    stage="modality_processing",
                    success=True,
                    elapsed_seconds=round(
                        time.perf_counter()
                        - stage_started,
                        6,
                    ),
                    details={
                        "processed_modalities": [
                            "vision",
                            "audio",
                            "spatial",
                            "motion",
                            *(
                                ["interaction"]
                                if interaction_result is not None
                                else []
                            ),
                            *(
                                ["wearable", "source_device"]
                                if device_results
                                else []
                            ),
                        ],
                        "optional_modalities_absent": (
                            unavailable_optional_results
                        ),
                        "interaction_event_present": (
                            interaction_result is not None
                        ),
                        "device_result_count": len(
                            device_results
                        ),
                    },
                )
            )

            # ------------------------------------------------
            # 3. Synchronize
            # ------------------------------------------------
            stage_started = time.perf_counter()

            frame = (
                self.synchronizer
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

            stage_records.append(
                StageRecord(
                    stage="multimodal_synchronization",
                    success=True,
                    elapsed_seconds=round(
                        time.perf_counter()
                        - stage_started,
                        6,
                    ),
                    details={
                        "frame_id": frame.frame_id,
                        "status": frame.status.value,
                        "selected_modalities": (
                            frame.selected_modalities
                        ),
                        "synchronization_score": (
                            frame
                            .synchronization_score
                        ),
                        "completeness_score": (
                            frame
                            .completeness_score
                        ),
                    },
                )
            )

            # ------------------------------------------------
            # 4. Confidence estimation
            # ------------------------------------------------
            stage_started = time.perf_counter()

            confidence_report = (
                self.confidence_estimator
                .estimate(
                    frame,
                    raise_on_error=True,
                )
            )

            stage_records.append(
                StageRecord(
                    stage="confidence_estimation",
                    success=True,
                    elapsed_seconds=round(
                        time.perf_counter()
                        - stage_started,
                        6,
                    ),
                    details={
                        "report_id": (
                            confidence_report
                            .report_id
                        ),
                        "overall_confidence": (
                            confidence_report
                            .overall_confidence
                        ),
                        "overall_level": (
                            confidence_report
                            .overall_level.value
                        ),
                        "trusted_modalities": (
                            confidence_report
                            .trusted_modalities
                        ),
                    },
                )
            )

            # ------------------------------------------------
            # 5. Missing modality handling
            # ------------------------------------------------
            stage_started = time.perf_counter()

            self.recovery_handler.update_cache(
                frame
            )

            recovery_result = (
                self.recovery_handler.handle(
                    frame,
                    confidence_report,
                    raise_on_error=True,
                )
            )

            final_frame = recovery_result.frame

            stage_records.append(
                StageRecord(
                    stage="missing_modality_recovery",
                    success=True,
                    elapsed_seconds=round(
                        time.perf_counter()
                        - stage_started,
                        6,
                    ),
                    details={
                        "report_id": (
                            recovery_result
                            .report.report_id
                        ),
                        "status": (
                            recovery_result
                            .report.status.value
                        ),
                        "safe_to_continue": (
                            recovery_result
                            .report.safe_to_continue
                        ),
                        "recovered_modalities": (
                            recovery_result
                            .report.recovered_modalities
                        ),
                        "required_unavailable": (
                            recovery_result
                            .report
                            .required_unavailable_modalities
                        ),
                    },
                )
            )

            # ------------------------------------------------
            # 6. NAMARA planning
            # ------------------------------------------------
            stage_started = time.perf_counter()

            observations = (
                self._build_namara_observations(
                    frame=final_frame,
                    confidence_report=(
                        confidence_report
                    ),
                )
            )

            namara_context = (
                self._build_namara_context(
                    frame=final_frame
                )
            )

            namara_plan = (
                self.namara_controller
                .create_plan(
                    observations,
                    namara_context,
                    raise_on_error=True,
                )
            )

            stage_records.append(
                StageRecord(
                    stage="namara_planning",
                    success=True,
                    elapsed_seconds=round(
                        time.perf_counter()
                        - stage_started,
                        6,
                    ),
                    details={
                        "plan_id": (
                            namara_plan.plan_id
                        ),
                        "effective_mode": (
                            namara_plan
                            .effective_mode.value
                        ),
                        "active_modalities": (
                            namara_plan
                            .active_modalities
                        ),
                        "forced_modalities": (
                            namara_plan
                            .forced_modalities
                        ),
                    },
                )
            )

            # ------------------------------------------------
            # 7. Build final packet
            # ------------------------------------------------
            stage_started = time.perf_counter()

            build_result = (
                self.packet_builder
                .build_from_recovery_result(
                    recovery_result=(
                        recovery_result
                    ),
                    confidence_report=(
                        confidence_report
                    ),
                    namara_plan=namara_plan,
                    source_mode=(
                        self.config
                        .source_mode.value
                    ),
                    write_outputs=False,
                    raise_on_error=True,
                )
            )

            if (
                not build_result.success
                or build_result.packet is None
            ):
                raise Layer1StageError(
                    "Sensor packet builder did not "
                    "produce a valid packet."
                )

            stage_records.append(
                StageRecord(
                    stage="sensor_packet_building",
                    success=True,
                    elapsed_seconds=round(
                        time.perf_counter()
                        - stage_started,
                        6,
                    ),
                    details={
                        "packet_id": (
                            build_result.packet
                            .metadata.packet_id
                        ),
                        "build_status": (
                            build_result.packet
                            .metadata
                            .build_status.value
                        ),
                        "ready_for_layer2": (
                            build_result.packet
                            .layer2_contract[
                                "ready_for_layer2"
                            ]
                        ),
                    },
                )
            )

            # ------------------------------------------------
            # 8. Dispatch
            # ------------------------------------------------
            stage_started = time.perf_counter()

            if self.config.write_outputs:
                dispatch_result = (
                    self.output_dispatcher
                    .dispatch_build_result(
                        build_result,
                        archive_previous=(
                            self.config
                            .archive_previous
                        ),
                        verify_after_write=(
                            self.config
                            .verify_dispatch
                        ),
                        allow_blocked=False,
                        raise_on_error=True,
                    )
                )

                if not dispatch_result.success:
                    raise Layer1StageError(
                        "Output dispatcher failed."
                    )

                dispatch_details = {
                    "dispatch_id": (
                        dispatch_result
                        .dispatch_id
                    ),
                    "primary_output_path": (
                        dispatch_result
                        .primary_output_path
                    ),
                    "layer2_output_path": (
                        dispatch_result
                        .layer2_output_path
                    ),
                    "integrity_verified": (
                        dispatch_result
                        .integrity_verified
                    ),
                }

            else:
                dispatch_details = {
                    "dispatch_skipped": True,
                }

            stage_records.append(
                StageRecord(
                    stage="output_dispatch",
                    success=True,
                    elapsed_seconds=round(
                        time.perf_counter()
                        - stage_started,
                        6,
                    ),
                    details=dispatch_details,
                )
            )

            effective_confidence = (
                recovery_result
                .report
                .adjusted_overall_confidence
            )

            build_status = (
                build_result.packet
                .metadata
                .build_status.value
            )

            cycle_status = (
                PipelineStatus.PASSED
                if build_status == "complete"
                else PipelineStatus.PARTIAL
            )

            cycle_result = CycleResult(
                cycle_id=cycle_id,
                cycle_number=cycle_number,
                started_at=cycle_started_at,
                completed_at=utc_now_iso(),
                status=cycle_status,
                stage_records=stage_records,
                generated_packet_count=(
                    generated_packet_count
                ),
                accepted_packet_count=(
                    accepted_packet_count
                ),
                rejected_packet_count=(
                    rejected_packet_count
                ),
                frame_id=final_frame.frame_id,
                confidence_report_id=(
                    confidence_report.report_id
                ),
                recovery_report_id=(
                    recovery_result
                    .report.report_id
                ),
                namara_plan_id=(
                    namara_plan.plan_id
                ),
                layer1_packet_id=(
                    build_result.packet
                    .metadata.packet_id
                ),
                dispatch_id=(
                    dispatch_result.dispatch_id
                    if dispatch_result
                    else None
                ),
                overall_confidence=(
                    confidence_report
                    .overall_confidence
                ),
                effective_confidence=(
                    effective_confidence
                ),
                primary_output_path=(
                    dispatch_result
                    .primary_output_path
                    if dispatch_result
                    else None
                ),
                layer2_output_path=(
                    dispatch_result
                    .layer2_output_path
                    if dispatch_result
                    else None
                ),
            )

            if self.config.print_packet:
                print(
                    json.dumps(
                        build_result.packet.to_dict(),
                        indent=2,
                        ensure_ascii=False,
                    )
                )

            return cycle_result

        except Exception as error:
            stage_records.append(
                StageRecord(
                    stage="pipeline_failure",
                    success=False,
                    elapsed_seconds=0.0,
                    details={},
                    error=(
                        f"{type(error).__name__}: {error}"
                    ),
                )
            )

            raise

    # ========================================================
    # NAMARA INTEGRATION
    # ========================================================

    def _build_namara_observations(
        self,
        *,
        frame: SynchronizedMultimodalFrame,
        confidence_report: ConfidenceReport,
    ) -> List[ModalityObservation]:
        observations: List[
            ModalityObservation
        ] = []

        modality_names = (
            "vision",
            "audio",
            "spatial",
            "motion",
            "interaction",
            "wearable",
            "environment",
        )

        for modality in modality_names:
            confidence_result = (
                confidence_report
                .modality_confidences.get(
                    modality
                )
            )

            record = (
                frame
                .synchronization_records.get(
                    modality
                )
            )

            value = getattr(
                frame,
                modality,
                None,
            )

            available = value is not None

            reliability = (
                confidence_result
                .confidence_score
                if confidence_result
                is not None
                else 0.0
            )

            quality_score = (
                confidence_result
                .components.quality
                if confidence_result
                is not None
                else 0.0
            )

            freshness_score = (
                confidence_result
                .components.freshness
                if confidence_result
                is not None
                else 0.0
            )

            health_score = (
                confidence_result
                .components.health
                if confidence_result
                is not None
                else 0.0
            )

            limitations = []

            if confidence_result is not None:
                limitations.extend(
                    confidence_result
                    .limitation_codes
                )

            if record is not None:
                limitations.extend(
                    record.limitation_codes
                )

            sampling_rate = None

            metadata = getattr(
                value,
                "metadata",
                None,
            )

            if metadata is not None:
                sampling_rate = getattr(
                    metadata,
                    "sampling_rate_hz",
                    None,
                )

            observations.append(
                ModalityObservation(
                    modality=modality,
                    available=available,
                    reliability=clamp(
                        reliability,
                        0.0,
                        1.0,
                    ),
                    quality_score=clamp(
                        quality_score,
                        0.0,
                        1.0,
                    ),
                    freshness_score=clamp(
                        freshness_score,
                        0.0,
                        1.0,
                    ),
                    sensor_health_score=clamp(
                        health_score,
                        0.0,
                        1.0,
                    ),
                    current_sampling_rate_hz=(
                        float(sampling_rate)
                        if sampling_rate
                        is not None
                        and float(sampling_rate) > 0
                        else None
                    ),
                    limitations=sorted(
                        set(limitations)
                    ),
                    metadata={
                        "source_frame_id": (
                            frame.frame_id
                        ),
                    },
                )
            )

        return observations

    def _build_namara_context(
        self,
        *,
        frame: SynchronizedMultimodalFrame,
    ) -> NAMARAContext:
        source_device = frame.source_device

        battery_level = 1.0
        is_charging = False
        network_strength = 1.0
        network_latency_ms = 0.0
        network_available = True

        if source_device is not None:
            battery = getattr(
                source_device,
                "battery_level",
                None,
            )
            if battery is not None:
                battery_level = clamp(
                    float(battery),
                    0.0,
                    1.0,
                )

            charging = getattr(
                source_device,
                "is_charging",
                None,
            )
            if charging is not None:
                is_charging = bool(
                    charging
                )

            strength = getattr(
                source_device,
                "network_strength",
                None,
            )
            if strength is not None:
                network_strength = clamp(
                    float(strength),
                    0.0,
                    1.0,
                )

            latency = getattr(
                source_device,
                "network_latency_ms",
                None,
            )
            if latency is not None:
                network_latency_ms = max(
                    0.0,
                    float(latency),
                )

            network_type = getattr(
                source_device,
                "network_type",
                None,
            )

            if network_type is not None:
                network_value = getattr(
                    network_type,
                    "value",
                    str(network_type),
                )
                network_available = (
                    str(network_value).lower()
                    != "offline"
                )

        emergency_active = (
            self.config.emergency_active
        )

        interaction = frame.interaction

        if interaction is not None:
            emergency_active = (
                emergency_active
                or bool(
                    getattr(
                        interaction,
                        "emergency_flag",
                        False,
                    )
                )
            )

        return NAMARAContext(
            mode=self.config.acquisition_mode,
            urgency=self.config.urgency,
            battery_level=battery_level,
            is_charging=is_charging,
            network_strength=network_strength,
            network_latency_ms=(
                network_latency_ms
            ),
            network_available=network_available,
            emergency_active=emergency_active,
            user_interaction_active=(
                self.config
                .user_interaction_active
                or interaction is not None
            ),
            metadata={
                "source_frame_id": (
                    frame.frame_id
                ),
                "source_mode": (
                    self.config
                    .source_mode.value
                ),
            },
        )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:
        return {
            "healthy": True,
            "runner_version": RUNNER_VERSION,
            "source_mode": (
                self.config.source_mode.value
            ),
            "scenario": (
                self.config.scenario.value
            ),
            "acquisition_mode": (
                self.config
                .acquisition_mode.value
            ),
            "receiver": (
                self.receiver.health_check()
            ),
            "vision": (
                self.vision_processor
                .health_check()
            ),
            "audio": (
                self.audio_processor
                .health_check()
            ),
            "spatial": (
                self.spatial_processor
                .health_check()
            ),
            "motion": (
                self.motion_processor
                .health_check()
            ),
            "interaction": (
                self.interaction_processor
                .health_check()
            ),
            "device": (
                self.device_processor
                .health_check()
            ),
            "namara": (
                self.namara_controller
                .health_check()
            ),
            "synchronizer": (
                self.synchronizer
                .health_check()
            ),
            "confidence_estimator": (
                self.confidence_estimator
                .health_check()
            ),
            "recovery_handler": (
                self.recovery_handler
                .health_check()
            ),
            "packet_builder": (
                self.packet_builder
                .health_check()
            ),
            "output_dispatcher": (
                self.output_dispatcher
                .health_check()
            ),
        }


# ============================================================
# CLI
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete NOONGIL-X Layer 1 pipeline."
        )
    )

    parser.add_argument(
        "--source",
        default=SourceMode.SIMULATION.value,
        choices=[
            item.value
            for item in SourceMode
        ],
        help=(
            "Input source mode. Only simulation is "
            "implemented currently."
        ),
    )

    parser.add_argument(
        "--scenario",
        default=(
            SimulationScenario.NAVIGATION.value
        ),
        choices=[
            item.value
            for item in SimulationScenario
        ],
        help="Simulation scenario.",
    )

    parser.add_argument(
        "--acquisition-mode",
        default=(
            AcquisitionMode.NAVIGATION.value
        ),
        choices=[
            item.value
            for item in AcquisitionMode
        ],
        help="NAMARA acquisition mode.",
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=DEFAULT_CYCLES,
        help="Number of pipeline cycles.",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help=(
            "Seconds to wait between cycles."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Simulation random seed.",
    )

    parser.add_argument(
        "--urgency",
        type=float,
        default=0.55,
        help=(
            "NAMARA urgency between 0.0 and 1.0."
        ),
    )

    parser.add_argument(
        "--emergency",
        action="store_true",
        help="Force emergency acquisition mode.",
    )

    parser.add_argument(
        "--no-write",
        action="store_true",
        help=(
            "Build the packet without dispatching "
            "official outputs."
        ),
    )

    parser.add_argument(
        "--no-archive",
        action="store_true",
        help=(
            "Do not archive previous official outputs."
        ),
    )

    parser.add_argument(
        "--no-verify",
        action="store_true",
        help=(
            "Skip post-write integrity verification."
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue later cycles after a cycle failure."
        ),
    )

    parser.add_argument(
        "--test-settings",
        action="store_true",
        help="Use create_test_settings().",
    )

    parser.add_argument(
        "--print-packet",
        action="store_true",
        help=(
            "Print the complete final packet JSON."
        ),
    )

    return parser


def config_from_args(
    args: argparse.Namespace,
) -> Layer1RunConfig:
    return Layer1RunConfig(
        source_mode=SourceMode(
            args.source
        ),
        scenario=SimulationScenario(
            args.scenario
        ),
        acquisition_mode=AcquisitionMode(
            args.acquisition_mode
        ),
        cycles=args.cycles,
        interval_seconds=args.interval,
        random_seed=args.seed,
        urgency=args.urgency,
        emergency_active=args.emergency,
        write_outputs=not args.no_write,
        archive_previous=not args.no_archive,
        verify_dispatch=not args.no_verify,
        use_test_settings=args.test_settings,
        stop_on_error=not (
            args.continue_on_error
        ),
        print_packet=args.print_packet,
    )


# ============================================================
# PRESENTATION
# ============================================================

def print_summary(
    summary: Layer1PipelineSummary,
) -> None:
    print("\n" + "=" * 72)
    print("NOONGIL-X | LAYER 1 PIPELINE SUMMARY")
    print("=" * 72)

    print(
        f"Run ID              : {summary.run_id}"
    )
    print(
        f"Status              : {summary.status.value.upper()}"
    )
    print(
        f"Source mode         : {summary.source_mode}"
    )
    print(
        f"Scenario            : {summary.scenario}"
    )
    print(
        f"Acquisition mode    : {summary.acquisition_mode}"
    )
    print(
        f"Completed cycles    : "
        f"{summary.completed_cycles}/"
        f"{summary.requested_cycles}"
    )
    print(
        f"Successful cycles   : {summary.successful_cycles}"
    )
    print(
        f"Failed cycles       : {summary.failed_cycles}"
    )
    print(
        f"Elapsed seconds     : "
        f"{summary.total_elapsed_seconds}"
    )
    print(
        f"Layer 1 output      : "
        f"{summary.official_primary_output}"
    )
    print(
        f"Layer 2 input       : "
        f"{summary.official_layer2_output}"
    )
    print(
        f"Pipeline summary    : "
        f"{LAYER1_PIPELINE_SUMMARY_PATH}"
    )

    for result in summary.cycle_results:
        print("\n" + "-" * 72)
        print(
            f"Cycle {result.cycle_number} | "
            f"{result.status.value.upper()}"
        )
        print(
            f"Packet ID           : "
            f"{result.layer1_packet_id}"
        )
        print(
            f"Dispatch ID         : "
            f"{result.dispatch_id}"
        )
        print(
            f"Overall confidence  : "
            f"{result.overall_confidence}"
        )
        print(
            f"Effective confidence: "
            f"{result.effective_confidence}"
        )
        print(
            f"Layer 2 output      : "
            f"{result.layer2_output_path}"
        )

        if result.error:
            print(
                f"Error               : "
                f"{result.error}"
            )

    print("\n" + "=" * 72)

    if summary.status == PipelineStatus.FAILED:
        print("[FAILED] LAYER 1 PIPELINE FAILED")
    elif summary.status == PipelineStatus.PARTIAL:
        print(
            "[PARTIAL] LAYER 1 PIPELINE COMPLETED "
            "WITH PARTIAL RESULTS"
        )
    else:
        print(
            "[PASSED] LAYER 1 PIPELINE IS WORKING"
        )

    print("=" * 72)


# ============================================================
# SELF-TEST
# ============================================================

def run_layer1_self_test() -> bool:
    print("\n" + "=" * 72)
    print("NOONGIL-X | COMPLETE LAYER 1 SELF-TEST")
    print("=" * 72)

    try:
        config = Layer1RunConfig(
            source_mode=SourceMode.SIMULATION,
            scenario=(
                SimulationScenario.NAVIGATION
            ),
            acquisition_mode=(
                AcquisitionMode.NAVIGATION
            ),
            cycles=1,
            interval_seconds=0.0,
            random_seed=42,
            urgency=0.55,
            write_outputs=True,
            archive_previous=True,
            verify_dispatch=True,
            use_test_settings=True,
            stop_on_error=True,
        )

        runner = Layer1Runner(
            config=config
        )

        summary = runner.run()

        if summary.failed_cycles != 0:
            raise AssertionError(
                "Layer 1 self-test contains failed cycles."
            )

        if summary.completed_cycles != 1:
            raise AssertionError(
                "Layer 1 self-test cycle count is incorrect."
            )

        cycle = summary.cycle_results[0]

        if cycle.layer1_packet_id is None:
            raise AssertionError(
                "Layer 1 packet ID is missing."
            )

        if cycle.dispatch_id is None:
            raise AssertionError(
                "Dispatch ID is missing."
            )

        if not Path(
            LAYER2_INPUT_PACKET_PATH
        ).exists():
            raise AssertionError(
                "Official Layer 2 input packet was not created."
            )

        if not Path(
            MULTIMODAL_SENSOR_PACKET_PATH
        ).exists():
            raise AssertionError(
                "Official Layer 1 packet was not created."
            )

        if not Path(
            LAYER1_PIPELINE_SUMMARY_PATH
        ).exists():
            raise AssertionError(
                "Pipeline summary was not created."
            )

        print_summary(summary)
        return True

    except Exception as error:
        print("\n" + "=" * 72)
        print("[FAILED] COMPLETE LAYER 1 SELF-TEST")
        print("=" * 72)
        print(
            f"[ERROR] {type(error).__name__}: {error}"
        )
        traceback.print_exc()
        return False


# ============================================================
# MAIN
# ============================================================

def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
        config.validate()

        runner = Layer1Runner(
            config=config
        )

        summary = runner.run()
        print_summary(summary)

        return (
            0
            if summary.status
            != PipelineStatus.FAILED
            else 1
        )

    except KeyboardInterrupt:
        print(
            "\n[STOPPED] Layer 1 pipeline "
            "interrupted by user."
        )
        return 130

    except Exception as error:
        print("\n" + "=" * 72)
        print("[FAILED] LAYER 1 PIPELINE")
        print("=" * 72)
        print(
            f"[ERROR] {type(error).__name__}: {error}"
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
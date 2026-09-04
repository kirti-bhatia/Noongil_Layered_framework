"""
============================================================
NOONGIL-X
Layer 2 : Complete Perception Pipeline
File    : layer2/run_layer2.py
============================================================

Purpose
-------
Executes the complete Layer 2 perception pipeline using real
module outputs.

Pipeline
--------
Layer 1 packet
    ↓
Vision Processor
    ├── Scene Classifier
    ├── Object Detector
    ├── Object Tracker       [temporal modes]
    └── Activity Recognizer  [temporal modes]
    ↓
OCR Engine → Text Interpreter
    ↓
Speech Recognizer
    ↓
Sound Event Detector
    ↓
Depth Estimator
    ↓
Feature Aligner
    ↓
Confidence Estimator
    ↓
Multimodal Fusion Engine
    ↓
Perception Output Builder
    ↓
Validated Layer2Output JSON
    ↓
Layer 3

Operating modes
---------------
snapshot:
    Processes one image and audio observation.

navigation:
    Enables object tracking and temporal activity buffering.

emergency:
    Enables tracking and temporal activity analysis with the
    same high-frequency perception path.

Important
---------
Static scenario fixtures contain one image, not real video.
In navigation/emergency mode, ActivityRecognizer may return
"collecting_frames" until consecutive camera frames arrive.
============================================================
"""

from __future__ import annotations

import argparse
import math

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
)

from layer2.audio.sound_event_detector import (
    SoundEventDetector,
)

from layer2.audio.speech_recognizer import (
    SpeechRecognizer,
)

from layer2.config.model_config import (
    Layer2ModelConfig,
    create_default_model_config,
)

from layer2.config.settings import (
    Layer2Settings,
    create_default_settings,
)

from layer2.input_reception.layer1_packet_adapter import (
    AdaptedLayer1Input,
    Layer1PacketAdapter,
)

from layer2.multimodal_fusion.feature_aligner import (
    FeatureAligner,
    FeatureAlignmentOutput,
)

from layer2.multimodal_fusion.fusion_engine import (
    MultimodalFusionEngine,
    MultimodalFusionOutput,
)

from layer2.output.perception_output_builder import (
    PerceptionOutputBuilder,
)

from layer2.schemas.layer2_output import (
    Layer2Output,
    utc_now_iso,
)

from layer2.schemas.module_result import (
    ModuleResult,
)

from layer2.spatial.depth_estimator import (
    DepthEstimator,
)


from layer2.spatial.object_localizer import (
    ObjectLocalizer,
)




from layer2.text.ocr_engine import (
    OCREngine,
)

from layer2.text.text_interpreter import (
    TextInterpreter,
)

from layer2.utils.logger import (
    Layer2LoggerAdapter,
    ModuleTimer,
    get_logger,
    log_event,
    log_exception,
)

from layer2.utils.model_loader import (
    ModelLoader,
)

from layer2.vision_perception.activity_recognizer import (
    ActivityRecognizer,
)

from layer2.vision_perception.object_detector import (
    ObjectDetector,
)

from layer2.vision_perception.object_tracker import (
    ObjectTracker,
)

from layer2.vision_perception.scene_classifier import (
    SceneClassifier,
)

from layer2.vision_perception.vision_processor import (
    VisionProcessor,
)


# ============================================================
# CONSTANTS
# ============================================================

LAYER2_PIPELINE_VERSION = "1.0"

SUPPORTED_PIPELINE_MODES = {
    "snapshot",
    "navigation",
    "emergency",
}

TEMPORAL_PIPELINE_MODES = {
    "navigation",
    "emergency",
}


# ============================================================
# EXCEPTION
# ============================================================

class Layer2PipelineError(Exception):
    """Raised when the Layer 2 pipeline cannot run."""


# ============================================================
# PIPELINE OUTPUT
# ============================================================

@dataclass
class Layer2PipelineRun:
    """Complete result of one Layer 2 pipeline run."""

    output: Layer2Output

    module_results: Dict[
        str,
        ModuleResult
    ]

    alignment_output: Optional[
        FeatureAlignmentOutput
    ]

    fusion_output: Optional[
        MultimodalFusionOutput
    ]

    output_path: Optional[Path]

    mode: str

    @property
    def succeeded(self) -> bool:

        return (
            self.output
            .ready_for_layer3
        )

    def summary(self) -> Dict[str, Any]:

        summary = self.output.summary()

        summary.update({
            "pipeline_mode": self.mode,
            "module_count": len(
                self.module_results
            ),
            "aligned_modalities": (
                list(
                    self.alignment_output
                    .features.keys()
                )
                if self.alignment_output
                else []
            ),
            "fusion_available": (
                self.fusion_output
                is not None
                and self.fusion_output
                .succeeded
            ),
            "output_path": (
                str(self.output_path)
                if self.output_path
                else None
            ),
        })

        return summary


# ============================================================
# LAYER 2 PIPELINE
# ============================================================

class Layer2Pipeline:
    """Complete reusable Layer 2 runtime pipeline."""

    def __init__(
        self,
        settings: Optional[
            Layer2Settings
        ] = None,
        model_config: Optional[
            Layer2ModelConfig
        ] = None,
        model_loader: Optional[
            ModelLoader
        ] = None,
        *,
        project_root: Optional[
            Path | str
        ] = None,
        mode: str = "snapshot",
        logger: Optional[
            Layer2LoggerAdapter
        ] = None,
    ) -> None:

        self.settings = (
            settings
            or create_default_settings()
        )

        self.settings.validate()

        self.model_config = (
            model_config
            or create_default_model_config()
        )

        self.model_config.validate()

        self.mode = self._validate_mode(
            mode
        )

        default_project_root = (
            Path(__file__).resolve().parents[1]
        )

        self.project_root = Path(
            project_root
            or default_project_root
        ).resolve()

        self.logger = (
            logger
            or get_logger(
                "run_layer2"
            )
        )

        # One shared loader prevents the same model from being
        # loaded repeatedly by different pipeline modules.
        self.model_loader = (
            model_loader
            or ModelLoader(
                model_config=(
                    self.model_config
                ),
                project_root=(
                    self.project_root
                ),
                logger=self.logger,
            )
        )

        self.vision_processor = (
            VisionProcessor(
                settings=self.settings,
                project_root=(
                    self.project_root
                ),
            )
        )

        self.scene_classifier = (
            SceneClassifier(
                settings=self.settings,
                model_config=(
                    self.model_config
                ),
                model_loader=(
                    self.model_loader
                ),
                project_root=(
                    self.project_root
                ),
            )
        )

        self.object_detector = (
            ObjectDetector(
                settings=self.settings,
                model_config=(
                    self.model_config
                ),
                model_loader=(
                    self.model_loader
                ),
                project_root=(
                    self.project_root
                ),
            )
        )

        self.object_tracker = (
            ObjectTracker(
                settings=self.settings,
                model_config=(
                    self.model_config
                ),
                model_loader=(
                    self.model_loader
                ),
                project_root=(
                    self.project_root
                ),
            )
        )

        self.activity_recognizer = (
            ActivityRecognizer(
                settings=self.settings,
                model_config=(
                    self.model_config
                ),
                model_loader=(
                    self.model_loader
                ),
                project_root=(
                    self.project_root
                ),
            )
        )

        self.ocr_engine = (
            OCREngine(
                settings=self.settings,
                model_config=(
                    self.model_config
                ),
                model_loader=(
                    self.model_loader
                ),
                project_root=(
                    self.project_root
                ),
            )
        )

        self.text_interpreter = (
            TextInterpreter(
                settings=self.settings
            )
        )

        self.speech_recognizer = (
            SpeechRecognizer(
                settings=self.settings,
                model_config=(
                    self.model_config
                ),
                model_loader=(
                    self.model_loader
                ),
                project_root=(
                    self.project_root
                ),
            )
        )

        self.sound_event_detector = (
            SoundEventDetector(
                settings=self.settings,
                model_config=(
                    self.model_config
                ),
                model_loader=(
                    self.model_loader
                ),
                project_root=(
                    self.project_root
                ),
            )
        )

        self.depth_estimator = (
            DepthEstimator(
                settings=self.settings,
                model_config=(
                    self.model_config
                ),
                model_loader=(
                    self.model_loader
                ),
                project_root=(
                    self.project_root
                ),
            )
        )
        self.object_localizer = ObjectLocalizer()
        self.feature_aligner = (
            FeatureAligner()
        )

        self.fusion_engine = (
            MultimodalFusionEngine(
                apply_attention=True
            )
        )

        self.output_builder = (
            PerceptionOutputBuilder()
        )

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def process_packet(
        self,
        packet: AdaptedLayer1Input,
        *,
        output_path: Optional[
            Path | str
        ] = None,
        save_output: bool = True,
    ) -> Layer2PipelineRun:
        """Process one adapted Layer 1 packet."""

        if not isinstance(
            packet,
            AdaptedLayer1Input,
        ):
            raise Layer2PipelineError(
                "packet must be an "
                "AdaptedLayer1Input."
            )

        packet_logger = self.logger.bind(
            packet_id=packet.packet_id,
            scenario=packet.scenario,
            mode=self.mode,
        )

        log_event(
            packet_logger,
            event=(
                "layer2_pipeline_started"
            ),
            message=(
                "Layer 2 perception pipeline "
                "started."
            ),
            details={
                "mode": self.mode,
                "scenario": packet.scenario,
            },
        )

        with ModuleTimer(
            "run_layer2",
            logger=packet_logger,
            packet_id=packet.packet_id,
            log_start=False,
            log_completion=False,
        ) as timer:

            module_results: Dict[
                str,
                ModuleResult
            ] = {}

            alignment_output = None
            fusion_output = None

            try:
                # ============================================
                # 1. VISION PREPROCESSING
                # ============================================

                vision_output = (
                    self.vision_processor
                    .process(
                        packet
                    )
                )

                module_results[
                    "vision_processor"
                ] = vision_output.result

                # ============================================
                # 2. VISUAL PERCEPTION
                # ============================================

                scene_output = (
                    self.scene_classifier
                    .classify(
                        packet,
                        vision_output,
                    )
                )

                module_results[
                    "scene_classifier"
                ] = scene_output.result

                object_output = (
                    self.object_detector
                    .detect(
                        packet,
                        vision_output,
                    )
                )

                module_results[
                    "object_detector"
                ] = object_output.result

                # Temporal modules are activated only in
                # navigation and emergency modes.
                if (
                    self.mode
                    in TEMPORAL_PIPELINE_MODES
                ):
                    tracker_output = (
                        self.object_tracker
                        .track(
                            packet,
                            vision_output,
                        )
                    )

                    module_results[
                        "object_tracker"
                    ] = tracker_output.result

                    activity_output = (
                        self.activity_recognizer
                        .recognize(
                            packet,
                            vision_output,
                        )
                    )

                    module_results[
                        "activity_recognizer"
                    ] = activity_output.result

                # ============================================
                # 3. TEXT PERCEPTION
                # ============================================

                ocr_output = (
                    self.ocr_engine
                    .recognize(
                        packet,
                        vision_output,
                    )
                )

                module_results[
                    "ocr_engine"
                ] = ocr_output.result

                if ocr_output.succeeded:

                    text_output = (
                        self.text_interpreter
                        .interpret(
                            ocr_output
                        )
                    )

                    module_results[
                        "text_interpreter"
                    ] = text_output.result

                # ============================================
                # 4. AUDIO PERCEPTION
                # ============================================

                speech_output = (
                    self.speech_recognizer
                    .transcribe(
                        packet
                    )
                )

                module_results[
                    "speech_recognizer"
                ] = speech_output.result

                sound_output = (
                    self.sound_event_detector
                    .detect(
                        packet
                    )
                )

                module_results[
                    "sound_event_detector"
                ] = sound_output.result

                # ============================================
                # 5. SPATIAL PERCEPTION
                # ============================================

                depth_output = (
                    self.depth_estimator
                    .estimate(
                        packet,
                        vision_output,
                    )
                )

                module_results[
                    "depth_estimator"
                ] = depth_output.result



                localization_output = (
                     self.object_localizer.localize(
                          packet,
                          object_output.objects,
                          depth_output,
                          vision_output,
                            )
                          )
                module_results[
                    "object_localizer"
                    ] = localization_output.result

                # ============================================
                # 6. FEATURE ALIGNMENT
                # ============================================

                alignment_output = (
                    self.feature_aligner
                    .align(
                        module_results,
                        source_packet_id=(
                            packet.packet_id
                        ),
                    )
                )

                if not alignment_output.succeeded:
                    raise Layer2PipelineError(
                        "Feature alignment failed: "
                        f"{alignment_output.result.errors}"
                    )

                # ============================================
                # 7. CONFIDENCE + MULTIMODAL FUSION
                # ============================================

                fusion_output = (
                    self.fusion_engine
                    .fuse(
                        alignment_output.features,
                        source_packet_id=(
                            packet.packet_id
                        ),
                    )
                )

                if not fusion_output.succeeded:
                    raise Layer2PipelineError(
                        "Multimodal fusion failed: "
                        f"{fusion_output.result.errors}"
                    )

                # ============================================
                # 8. LOCATION AND TIMESTAMP
                # ============================================

                location = (
                    self._extract_location(
                        packet
                    )
                )

                timestamp = (
                    self._extract_timestamp(
                        packet
                    )
                )

                # ============================================
                # 9. FINAL LAYER 2 OUTPUT
                # ============================================

                layer2_output = (
                    self.output_builder
                    .build(
                        module_results,
                        fusion_output=(
                            fusion_output
                        ),
                        alignment_output=(
                            alignment_output
                        ),
                        source_packet_id=(
                            packet.packet_id
                        ),
                        timestamp=timestamp,
                        location=location,
                    )
                )

                # Include complete pipeline time.
                layer2_output.processing_time_ms = (
                    max(
                        layer2_output
                        .processing_time_ms,
                        timer.elapsed_ms,
                    )
                )

                layer2_output.validate()

                # ============================================
                # 10. OUTPUT SERIALIZATION
                # ============================================

                written_path = None

                if save_output:

                    resolved_output_path = (
                        Path(output_path)
                        if output_path
                        is not None
                        else (
                            self._default_output_path(
                                packet
                            )
                        )
                    )

                    written_path = (
                        layer2_output
                        .write_json(
                            resolved_output_path
                        )
                    )

                log_event(
                    packet_logger,
                    event=(
                        "layer2_pipeline_completed"
                    ),
                    message=(
                        "Layer 2 perception pipeline "
                        "completed."
                    ),
                    details={
                        "status": (
                            layer2_output.status
                        ),
                        "ready_for_layer3": (
                            layer2_output
                            .ready_for_layer3
                        ),
                        "scene_type": (
                            layer2_output
                            .scene_type
                        ),
                        "object_count": len(
                            layer2_output.objects
                        ),
                        "processing_time_ms": (
                            timer.elapsed_ms
                        ),
                        "output_path": (
                            str(written_path)
                            if written_path
                            else None
                        ),
                    },
                )

                return Layer2PipelineRun(
                    output=layer2_output,
                    module_results=(
                        module_results
                    ),
                    alignment_output=(
                        alignment_output
                    ),
                    fusion_output=(
                        fusion_output
                    ),
                    output_path=written_path,
                    mode=self.mode,
                )

            except Exception as error:

                log_exception(
                    packet_logger,
                    error,
                    event=(
                        "layer2_pipeline_failed"
                    ),
                    message=(
                        "Layer 2 perception pipeline "
                        "failed."
                    ),
                    details={
                        "mode": self.mode,
                        "processing_time_ms": (
                            timer.elapsed_ms
                        ),
                        "completed_modules": list(
                            module_results.keys()
                        ),
                    },
                )

                if isinstance(
                    error,
                    Layer2PipelineError,
                ):
                    raise

                raise Layer2PipelineError(
                    f"{error.__class__.__name__}: "
                    f"{error}"
                ) from error

    def process_scenario(
        self,
        scenario_name: str,
        *,
        adapter: Optional[
            Layer1PacketAdapter
        ] = None,
        output_path: Optional[
            Path | str
        ] = None,
        save_output: bool = True,
        reset_temporal_state: bool = True,
    ) -> Layer2PipelineRun:
        """Load and process one test scenario."""

        if not isinstance(
            scenario_name,
            str,
        ) or not scenario_name.strip():
            raise Layer2PipelineError(
                "scenario_name must be "
                "a non-empty string."
            )

        scenario_adapter = (
            adapter
            or Layer1PacketAdapter(
                require_media=True
            )
        )

        # Scenario fixtures are independent observations.
        if reset_temporal_state:
            self.reset_temporal_state()

        packet = (
            scenario_adapter
            .load_scenario(
                scenario_name.strip()
            )
        )

        return self.process_packet(
            packet,
            output_path=output_path,
            save_output=save_output,
        )

    def reset_temporal_state(self) -> None:
        """Clear tracking and activity state."""

        self.object_tracker.reset()
        self.activity_recognizer.reset()

    def unload_models(self) -> None:
        """Release all cached AI models."""

        self.model_loader.unload_all()

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    def _extract_location(
        self,
        packet: AdaptedLayer1Input,
    ) -> Dict[str, Any]:
        """Extract optional GPS data from the packet."""

        candidates: List[Any] = []

        for modality_name in (
            "spatial",
            "gps",
            "location",
        ):
            try:
                modality = packet.get_modality(
                    modality_name
                )

            except Exception:
                modality = None

            if modality is None:
                continue

            candidates.extend([
                getattr(
                    modality,
                    "data",
                    None,
                ),
                getattr(
                    modality,
                    "payload",
                    None,
                ),
                getattr(
                    modality,
                    "metadata",
                    None,
                ),
            ])

        candidates.extend([
            getattr(
                packet,
                "location",
                None,
            ),
            getattr(
                packet,
                "spatial",
                None,
            ),
            getattr(
                packet,
                "metadata",
                None,
            ),
        ])

        for candidate in candidates:

            location = (
                self._find_location(
                    candidate
                )
            )

            if location:
                return location

        return {}

    def _find_location(
        self,
        value: Any,
    ) -> Dict[str, Any]:
        """Recursively locate latitude/longitude."""

        if isinstance(
            value,
            Mapping,
        ):
            latitude = value.get(
                "latitude"
            )

            if latitude is None:
                latitude = value.get(
                    "lat"
                )

            longitude = value.get(
                "longitude"
            )

            if longitude is None:
                longitude = value.get(
                    "lon",
                    value.get("lng"),
                )

            if (
                latitude is not None
                and longitude is not None
            ):
                location = {
                    "latitude": (
                        self._finite_float(
                            latitude
                        )
                    ),
                    "longitude": (
                        self._finite_float(
                            longitude
                        )
                    ),
                }

                if (
                    location["latitude"]
                    is None
                    or location["longitude"]
                    is None
                ):
                    return {}

                accuracy = value.get(
                    "accuracy_m",
                    value.get("accuracy"),
                )

                if accuracy is not None:

                    accuracy_value = (
                        self._finite_float(
                            accuracy
                        )
                    )

                    if (
                        accuracy_value
                        is not None
                        and accuracy_value >= 0.0
                    ):
                        location[
                            "accuracy_m"
                        ] = accuracy_value

                return location

            for nested_value in (
                value.values()
            ):
                location = (
                    self._find_location(
                        nested_value
                    )
                )

                if location:
                    return location

        elif isinstance(
            value,
            (list, tuple),
        ):
            for item in value:

                location = (
                    self._find_location(
                        item
                    )
                )

                if location:
                    return location

        return {}

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    @staticmethod
    def _extract_timestamp(
        packet: AdaptedLayer1Input,
    ) -> str:
        """Extract a valid packet timestamp."""

        for attribute_name in (
            "timestamp",
            "anchor_timestamp",
            "source_timestamp",
            "created_at",
        ):
            value = getattr(
                packet,
                attribute_name,
                None,
            )

            if isinstance(
                value,
                datetime,
            ):
                return value.isoformat()

            if isinstance(
                value,
                str,
            ) and value.strip():
                try:
                    datetime.fromisoformat(
                        value.strip().replace(
                            "Z",
                            "+00:00",
                        )
                    )

                    return value.strip()

                except ValueError:
                    continue

        return utc_now_iso()

    # --------------------------------------------------------
    # OUTPUT PATH
    # --------------------------------------------------------

    def _default_output_path(
        self,
        packet: AdaptedLayer1Input,
    ) -> Path:

        scenario = (
            str(
                packet.scenario
                or "runtime"
            )
            .strip()
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

        packet_id = (
            str(packet.packet_id)
            .strip()
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

        return (
            self.project_root
            / "output"
            / "layer2"
            / "pipeline"
            / scenario
            / f"{packet_id}_layer2_output.json"
        )

    # --------------------------------------------------------
    # GENERAL HELPERS
    # --------------------------------------------------------

    @staticmethod
    def _finite_float(
        value: Any,
    ) -> Optional[float]:

        try:
            value = float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

        if not math.isfinite(value):
            return None

        return value

    @staticmethod
    def _validate_mode(
        mode: Any,
    ) -> str:

        if not isinstance(
            mode,
            str,
        ):
            raise Layer2PipelineError(
                "Pipeline mode must be a string."
            )

        normalized = (
            mode.strip().lower()
        )

        if (
            normalized
            not in SUPPORTED_PIPELINE_MODES
        ):
            raise Layer2PipelineError(
                f"Unsupported pipeline mode: "
                f"{normalized!r}. Supported: "
                f"{sorted(SUPPORTED_PIPELINE_MODES)}"
            )

        return normalized


# ============================================================
# SELF-TEST / CLI
# ============================================================

def run_scenario_test(
    scenario_name: str,
    *,
    mode: str = "snapshot",
    save_output: bool = True,
) -> bool:
    """Run one real end-to-end scenario."""

    print("=" * 72)

    print(
        "NOONGIL-X | COMPLETE LAYER 2 "
        "PIPELINE"
    )

    print("=" * 72)

    print(f"Scenario : {scenario_name}")
    print(f"Mode     : {mode}")

    pipeline: Optional[
        Layer2Pipeline
    ] = None

    try:
        pipeline = Layer2Pipeline(
            mode=mode
        )

        run = pipeline.process_scenario(
            scenario_name,
            save_output=save_output,
        )

        if not run.succeeded:
            raise AssertionError(
                "Final output is not ready "
                "for Layer 3."
            )

        print("\nModule results:")

        for module_name, result in (
            run.module_results.items()
        ):
            status = getattr(
                result.status,
                "value",
                str(result.status),
            )

            print(
                f"  {module_name}: "
                f"{status}, "
                f"confidence="
                f"{result.confidence}"
            )

        print("\nFinal output summary:")

        for key, value in (
            run.summary().items()
        ):
            print(
                f"  {key}: {value}"
            )

        print("\n" + "=" * 72)

        print(
            "[PASSED] COMPLETE LAYER 2 "
            "PIPELINE IS WORKING"
        )

        print("=" * 72)

        return True

    except (
        Layer2PipelineError,
        AssertionError,
    ) as error:

        print(f"\n[FAILED] {error}")

        print("=" * 72)

        return False

    finally:

        if pipeline is not None:
            pipeline.unload_models()


def run_all_scenarios(
    *,
    mode: str = "snapshot",
    save_output: bool = True,
) -> bool:
    """Run all discovered test scenarios."""

    adapter = Layer1PacketAdapter(
        require_media=True
    )

    scenarios = (
        adapter.discover_scenarios()
    )

    if not scenarios:
        print("[FAILED] No scenarios discovered.")

        return False

    pipeline: Optional[
        Layer2Pipeline
    ] = None

    passed = 0
    failed = 0

    try:
        pipeline = Layer2Pipeline(
            mode=mode
        )

        print("=" * 72)

        print(
            "NOONGIL-X | ALL LAYER 2 "
            "SCENARIOS"
        )

        print("=" * 72)

        for scenario in scenarios:

            print(
                f"\n[RUN] {scenario}"
            )

            try:
                run = (
                    pipeline.process_scenario(
                        scenario,
                        adapter=adapter,
                        save_output=(
                            save_output
                        ),
                        reset_temporal_state=True,
                    )
                )

                if not run.succeeded:
                    raise Layer2PipelineError(
                        "Output is not ready "
                        "for Layer 3."
                    )

                passed += 1

                print(
                    f"[PASS] {scenario}: "
                    f"scene="
                    f"{run.output.scene_type}, "
                    f"objects="
                    f"{len(run.output.objects)}, "
                    f"confidence="
                    f"{run.output.overall_confidence}"
                )

            except Exception as error:

                failed += 1

                print(
                    f"[FAILED] {scenario}: "
                    f"{error}"
                )

        print("\n" + "=" * 72)

        print(
            f"Passed: {passed}/"
            f"{len(scenarios)}"
        )

        print(
            f"Failed: {failed}/"
            f"{len(scenarios)}"
        )

        print("=" * 72)

        return failed == 0

    finally:

        if pipeline is not None:
            pipeline.unload_models()


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete NOONGIL-X "
            "Layer 2 perception pipeline."
        )
    )

    parser.add_argument(
        "--scenario",
        default="park_walking",
        help=(
            "Scenario to process. "
            "Default: park_walking"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=sorted(
            SUPPORTED_PIPELINE_MODES
        ),
        default="snapshot",
        help=(
            "Pipeline operating mode. "
            "Default: snapshot"
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Process all discovered scenarios."
        ),
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help=(
            "Do not write Layer 2 JSON output."
        ),
    )

    return parser


def main() -> int:

    arguments = (
        build_argument_parser()
        .parse_args()
    )

    save_output = (
        not arguments.no_save
    )

    if arguments.all:

        succeeded = run_all_scenarios(
            mode=arguments.mode,
            save_output=save_output,
        )

    else:

        succeeded = run_scenario_test(
            arguments.scenario,
            mode=arguments.mode,
            save_output=save_output,
        )

    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
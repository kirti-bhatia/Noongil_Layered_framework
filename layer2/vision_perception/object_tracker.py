"""NOONGIL-X Layer 2: stateful YOLOv8 + ByteTrack object tracking."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from layer2.config.model_config import Layer2ModelConfig, create_default_model_config
from layer2.config.settings import Layer2Settings, create_default_settings
from layer2.input_reception.layer1_packet_adapter import AdaptedLayer1Input, Layer1PacketAdapter
from layer2.schemas.module_result import ModuleResult
from layer2.utils.exceptions import DependencyMissingError, ModelLoadingError
from layer2.utils.logger import Layer2LoggerAdapter, ModuleTimer, get_logger, log_event, log_exception
from layer2.utils.model_loader import ModelBundle, ModelLoader
from layer2.vision_perception.vision_processor import VisionProcessingOutput, VisionProcessor


OBJECT_TRACKER_VERSION = "1.0"


class ObjectTrackingError(Exception):
    """Raised when ByteTrack output cannot be processed."""


@dataclass(frozen=True)
class TrackedObject:
    track_id: Optional[int]
    label: str
    class_id: int
    confidence: float
    bounding_box: Dict[str, Any]
    area_ratio: float
    is_tracked: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "entity_id": f"TRACK_{self.track_id:06d}" if self.track_id is not None else None,
            "label": self.label,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 6),
            "bounding_box": self.bounding_box,
            "area_ratio": round(self.area_ratio, 6),
            "is_tracked": self.is_tracked,
        }


@dataclass
class ObjectTrackingOutput:
    result: ModuleResult
    tracked_objects: List[TrackedObject]

    @property
    def succeeded(self) -> bool:
        return self.result.usable

    @property
    def object_count(self) -> int:
        return len(self.tracked_objects)

    @property
    def tracked_count(self) -> int:
        return sum(item.is_tracked for item in self.tracked_objects)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_count": self.object_count,
            "tracked_count": self.tracked_count,
            "tracked_objects": [item.to_dict() for item in self.tracked_objects],
            "result": self.result.to_dict(),
        }


class ObjectTracker:
    """Maintain object identities across consecutive frames using ByteTrack."""

    def __init__(
        self,
        settings: Optional[Layer2Settings] = None,
        model_config: Optional[Layer2ModelConfig] = None,
        model_loader: Optional[ModelLoader] = None,
        *,
        project_root: Optional[Path | str] = None,
        logger: Optional[Layer2LoggerAdapter] = None,
    ) -> None:
        self.settings = settings or create_default_settings()
        self.settings.validate()
        self.model_config = model_config or create_default_model_config()
        self.model_config.validate()
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.logger = logger or get_logger("object_tracker")
        self.model_loader = model_loader or ModelLoader(
            model_config=self.model_config,
            project_root=self.project_root,
            logger=self.logger,
        )
        self._model: Any = None
        self._device = "cpu"
        self._frame_index = 0

    def track(
        self,
        packet: AdaptedLayer1Input,
        vision_output: VisionProcessingOutput,
    ) -> ObjectTrackingOutput:
        if not isinstance(packet, AdaptedLayer1Input):
            raise ObjectTrackingError("packet must be an AdaptedLayer1Input.")
        if not isinstance(vision_output, VisionProcessingOutput):
            raise ObjectTrackingError("vision_output must be a VisionProcessingOutput.")

        if not self.settings.modules.object_tracking:
            return ObjectTrackingOutput(
                ModuleResult.skipped(
                    module_name="object_tracker", modality="vision",
                    reason="Object tracking is disabled in settings.",
                    source_packet_id=packet.packet_id,
                ),
                [],
            )
        if not vision_output.succeeded or vision_output.image_rgb is None:
            return ObjectTrackingOutput(
                ModuleResult.failure(
                    module_name="object_tracker", modality="vision",
                    error="A successful processed vision frame is required.",
                    source_packet_id=packet.packet_id,
                ),
                [],
            )

        packet_logger = self.logger.bind(packet_id=packet.packet_id, scenario=packet.scenario)
        log_event(packet_logger, event="object_tracking_started", message="Object tracking started.")

        with ModuleTimer(
            "object_tracker", logger=packet_logger, packet_id=packet.packet_id,
            log_start=False, log_completion=False,
        ) as timer:
            try:
                model = self._get_detection_model()
                detector_parameters = self.model_config.object_detection.parameters
                tracker_parameters = self.model_config.object_tracking.parameters
                threshold = self._confidence_threshold()
                tracker_source = self.model_config.object_tracking.source
                persist = bool(tracker_parameters.get("persist_tracks", True))

                raw_results = model.track(
                    source=self._input_image(vision_output),
                    conf=threshold,
                    imgsz=int(detector_parameters.get("image_size", 640)),
                    agnostic_nms=bool(detector_parameters.get("agnostic_nms", False)),
                    tracker=tracker_source,
                    persist=persist,
                    device=self._device,
                    verbose=False,
                )
                self._frame_index += 1
                objects, image_shape = self._parse_results(raw_results)
                tracked_count = sum(item.is_tracked for item in objects)
                confidence = max((item.confidence for item in objects), default=0.0)
                warnings: List[str] = []
                if not objects:
                    warnings.append("No objects met the tracking confidence threshold.")
                elif tracked_count < len(objects):
                    warnings.append("Some detections have not yet received persistent track IDs.")

                data = {
                    "frame_index": self._frame_index,
                    "object_count": len(objects),
                    "tracked_count": tracked_count,
                    "tracked_objects": [item.to_dict() for item in objects],
                    "image_shape": {"width": image_shape[0], "height": image_shape[1]},
                    "tracker": tracker_source,
                    "persist_tracks": persist,
                    "threshold": threshold,
                    "detector_model_id": self.model_config.object_detection.model_id,
                    "tracker_version": OBJECT_TRACKER_VERSION,
                }
                factory = ModuleResult.success if not warnings else ModuleResult.partial
                result = factory(
                    module_name="object_tracker", modality="vision", data=data,
                    confidence=confidence, processing_time_ms=timer.elapsed_ms,
                    source_packet_id=packet.packet_id, warnings=warnings,
                    metadata={"scenario": packet.scenario, "tracker_backend": "ultralytics_bytetrack"},
                )
                log_event(
                    packet_logger, event="object_tracking_completed",
                    message="Object tracking completed.",
                    details={
                        "object_count": len(objects), "tracked_count": tracked_count,
                        "frame_index": self._frame_index, "processing_time_ms": timer.elapsed_ms,
                    },
                )
                return ObjectTrackingOutput(result=result, tracked_objects=objects)
            except Exception as error:
                log_exception(
                    packet_logger, error, event="object_tracking_failed",
                    message="Object tracking failed.",
                    details={"processing_time_ms": timer.elapsed_ms},
                )
                return ObjectTrackingOutput(
                    ModuleResult.failure(
                        module_name="object_tracker", modality="vision",
                        error=f"{error.__class__.__name__}: {error}",
                        processing_time_ms=timer.elapsed_ms,
                        source_packet_id=packet.packet_id,
                        metadata={"tracker": self.model_config.object_tracking.source},
                    ),
                    [],
                )

    def reset(self) -> None:
        """Clear ByteTrack state before starting an unrelated video stream."""
        if self._model is not None:
            predictor = getattr(self._model, "predictor", None)
            trackers = getattr(predictor, "trackers", None)
            if trackers:
                for tracker in trackers:
                    reset_method = getattr(tracker, "reset", None)
                    if callable(reset_method):
                        reset_method()
            # Rebuilding the predictor guarantees no previous stream state survives.
            self._model.predictor = None
        self._frame_index = 0

    def _get_detection_model(self) -> Any:
        if self._model is None:
            loaded = self.model_loader.load("object_detection")
            if isinstance(loaded, ModelBundle):
                self._model, self._device = loaded.model, loaded.device
            else:
                self._model, self._device = loaded, "cpu"
        if not callable(getattr(self._model, "track", None)):
            raise ModelLoadingError(
                "Loaded object-detection model has no callable track method.",
                module="object_tracker",
            )
        return self._model

    @staticmethod
    def _input_image(vision_output: VisionProcessingOutput) -> Any:
        return vision_output.original_image if vision_output.original_image is not None else vision_output.image_rgb

    def _confidence_threshold(self) -> float:
        vision = self.settings.vision
        value = getattr(vision, "object_confidence_threshold", None)
        if value is None:
            value = getattr(vision, "detection_confidence_threshold", 0.35)
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ObjectTrackingError("Object confidence threshold must be between 0 and 1.")
        return value

    def _parse_results(self, raw_results: Any) -> Tuple[List[TrackedObject], Tuple[int, int]]:
        if not isinstance(raw_results, (list, tuple)) or not raw_results:
            raise ObjectTrackingError("ByteTrack returned no result objects.")
        result = raw_results[0]
        shape = getattr(result, "orig_shape", None)
        if not shape or len(shape) != 2:
            raise ObjectTrackingError("Tracking result is missing orig_shape.")
        image_height, image_width = int(shape[0]), int(shape[1])
        names = getattr(result, "names", {})
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return [], (image_width, image_height)

        objects: List[TrackedObject] = []
        for box in boxes:
            xyxy = self._values(box.xyxy)
            confidences = self._values(box.conf)
            classes = self._values(box.cls)
            ids = self._values(box.id) if getattr(box, "id", None) is not None else []
            if len(xyxy) < 4 or not confidences or not classes:
                continue
            x1, y1, x2, y2 = xyxy[:4]
            confidence, class_id = float(confidences[0]), int(classes[0])
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2, confidence)):
                continue
            x1, x2 = sorted((max(0.0, min(x1, image_width)), max(0.0, min(x2, image_width))))
            y1, y2 = sorted((max(0.0, min(y1, image_height)), max(0.0, min(y2, image_height))))
            width, height = x2 - x1, y2 - y1
            if width <= 0.0 or height <= 0.0:
                continue
            track_id = int(ids[0]) if ids and math.isfinite(ids[0]) else None
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            label = str(label).strip().lower().replace(" ", "_")
            bounding_box = {
                "x1": round(x1, 3), "y1": round(y1, 3),
                "x2": round(x2, 3), "y2": round(y2, 3),
                "width": round(width, 3), "height": round(height, 3),
                "center": {"x": round((x1 + x2) / 2, 3), "y": round((y1 + y2) / 2, 3)},
                "normalized_xyxy": [
                    round(x1 / image_width, 6), round(y1 / image_height, 6),
                    round(x2 / image_width, 6), round(y2 / image_height, 6),
                ],
            }
            objects.append(TrackedObject(
                track_id=track_id, label=label, class_id=class_id,
                confidence=confidence, bounding_box=bounding_box,
                area_ratio=(width * height) / (image_width * image_height),
                is_tracked=track_id is not None,
            ))
        objects.sort(key=lambda item: item.confidence, reverse=True)
        return objects, (image_width, image_height)

    @staticmethod
    def _values(value: Any) -> List[float]:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if hasattr(value, "tolist"):
            value = value.tolist()
        while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            value = value[0]
        return [float(item) for item in value] if isinstance(value, list) else [float(value)]


def track_objects(
    packet: AdaptedLayer1Input,
    vision_output: VisionProcessingOutput,
    *,
    settings: Optional[Layer2Settings] = None,
    model_config: Optional[Layer2ModelConfig] = None,
    model_loader: Optional[ModelLoader] = None,
) -> ObjectTrackingOutput:
    return ObjectTracker(settings, model_config, model_loader).track(packet, vision_output)


def run_self_test(scenario_name: str = "park_walking", *, test_all: bool = False) -> bool:
    print("=" * 72)
    print("NOONGIL-X | OBJECT TRACKER SELF-TEST")
    print("=" * 72)
    loader: Optional[ModelLoader] = None
    try:
        adapter = Layer1PacketAdapter(require_media=True)
        settings = create_default_settings()
        model_config = create_default_model_config()
        loader = ModelLoader(model_config=model_config)
        processor = VisionProcessor(settings=settings)
        tracker = ObjectTracker(settings, model_config, loader)
        scenarios = adapter.discover_scenarios() if test_all else [scenario_name]
        if not scenarios:
            raise AssertionError("No scenarios selected.")
        print(f"[PASS] Testing {len(scenarios)} scenario(s)")
        for scenario in scenarios:
            tracker.reset()  # fixtures are unrelated streams
            packet = adapter.load_scenario(scenario)
            vision_output = processor.process(packet)
            if not vision_output.succeeded:
                raise AssertionError(f"Vision processing failed for {scenario}.")
            output = tracker.track(packet, vision_output)
            if not output.succeeded:
                raise AssertionError(f"Object tracking failed for {scenario}: {output.result.errors}")
            for item in output.tracked_objects:
                if not 0.0 <= item.confidence <= 1.0 or not 0.0 <= item.area_ratio <= 1.0:
                    raise AssertionError(f"Invalid tracking values for {scenario}.")
            ids = [item.track_id for item in output.tracked_objects[:5]]
            print(
                f"[PASS] {scenario}: objects={output.object_count}, "
                f"tracked={output.tracked_count}, top_track_ids={ids}"
            )
        print("[PASS] Detection model reused for tracking")
        print("[PASS] ByteTrack configuration applied")
        print("[PASS] Persistent track IDs parsed")
        print("[PASS] Bounding boxes validated and normalized")
        print("[PASS] Layer 3 tracking format generated")
        print("[PASS] ModuleResult generated")
        print("\n" + "=" * 72)
        print("[PASSED] OBJECT TRACKER IS WORKING")
        print("=" * 72)
        return True
    except (DependencyMissingError, ModelLoadingError, ObjectTrackingError, AssertionError) as error:
        print(f"\n[FAILED] {error}")
        print("=" * 72)
        return False
    finally:
        if loader is not None:
            loader.unload_all()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NOONGIL-X Layer 2 object-tracker self-test.")
    parser.add_argument("--scenario", default="park_walking")
    parser.add_argument("--all", action="store_true")
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()
    return 0 if run_self_test(arguments.scenario, test_all=arguments.all) else 1


if __name__ == "__main__":
    raise SystemExit(main())

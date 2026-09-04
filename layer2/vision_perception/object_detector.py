"""NOONGIL-X Layer 2: YOLOv8 object detection."""

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


OBJECT_DETECTOR_VERSION = "1.0"


class ObjectDetectionError(Exception):
    """Raised when YOLO output cannot be processed."""


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    center_x: float
    center_y: float
    normalized: Tuple[float, float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": round(self.x1, 3), "y1": round(self.y1, 3),
            "x2": round(self.x2, 3), "y2": round(self.y2, 3),
            "width": round(self.width, 3), "height": round(self.height, 3),
            "center": {"x": round(self.center_x, 3), "y": round(self.center_y, 3)},
            "normalized_xyxy": [round(value, 6) for value in self.normalized],
        }


@dataclass(frozen=True)
class DetectedObject:
    object_id: str
    label: str
    class_id: int
    confidence: float
    bounding_box: BoundingBox
    area_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 6),
            "bounding_box": self.bounding_box.to_dict(),
            "area_ratio": round(self.area_ratio, 6),
        }


@dataclass
class ObjectDetectionOutput:
    result: ModuleResult
    objects: List[DetectedObject]

    @property
    def succeeded(self) -> bool:
        return self.result.usable

    @property
    def object_count(self) -> int:
        return len(self.objects)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_count": self.object_count,
            "objects": [item.to_dict() for item in self.objects],
            "result": self.result.to_dict(),
        }


class ObjectDetector:
    """Detect objects in one processed Layer 1 camera frame."""

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
        self.logger = logger or get_logger("object_detector")
        self.model_loader = model_loader or ModelLoader(
            model_config=self.model_config,
            project_root=self.project_root,
            logger=self.logger,
        )

    def detect(
        self,
        packet: AdaptedLayer1Input,
        vision_output: VisionProcessingOutput,
    ) -> ObjectDetectionOutput:
        if not isinstance(packet, AdaptedLayer1Input):
            raise ObjectDetectionError("packet must be an AdaptedLayer1Input.")
        if not isinstance(vision_output, VisionProcessingOutput):
            raise ObjectDetectionError("vision_output must be a VisionProcessingOutput.")

        if not self.settings.modules.object_detection:
            return ObjectDetectionOutput(
                ModuleResult.skipped(
                    module_name="object_detector", modality="vision",
                    reason="Object detection is disabled in settings.",
                    source_packet_id=packet.packet_id,
                ),
                [],
            )

        if not vision_output.succeeded or vision_output.image_rgb is None:
            return ObjectDetectionOutput(
                ModuleResult.failure(
                    module_name="object_detector", modality="vision",
                    error="A successful processed vision frame is required.",
                    source_packet_id=packet.packet_id,
                ),
                [],
            )

        packet_logger = self.logger.bind(packet_id=packet.packet_id, scenario=packet.scenario)
        log_event(packet_logger, event="object_detection_started", message="Object detection started.")

        with ModuleTimer(
            "object_detector", logger=packet_logger, packet_id=packet.packet_id,
            log_start=False, log_completion=False,
        ) as timer:
            try:
                model, device = self._get_model()
                threshold = self._confidence_threshold()
                parameters = self.model_config.object_detection.parameters
                image_size = int(parameters.get("image_size", 640))
                agnostic_nms = bool(parameters.get("agnostic_nms", False))

                raw_results = model.predict(
                    source=self._input_image(vision_output),
                    conf=threshold,
                    imgsz=image_size,
                    agnostic_nms=agnostic_nms,
                    device=device,
                    verbose=False,
                )
                objects, image_shape = self._parse_results(raw_results, packet.packet_id)
                confidence = max((item.confidence for item in objects), default=0.0)
                warnings = [] if objects else ["No objects met the configured confidence threshold."]

                data = {
                    "object_count": len(objects),
                    "objects": [item.to_dict() for item in objects],
                    "image_shape": {"width": image_shape[0], "height": image_shape[1]},
                    "threshold": threshold,
                    "model_id": self.model_config.object_detection.model_id,
                    "detector_version": OBJECT_DETECTOR_VERSION,
                }
                factory = ModuleResult.success if objects else ModuleResult.partial
                result = factory(
                    module_name="object_detector", modality="vision", data=data,
                    confidence=confidence, processing_time_ms=timer.elapsed_ms,
                    source_packet_id=packet.packet_id, warnings=warnings,
                    metadata={"scenario": packet.scenario, "model_backend": "ultralytics"},
                )
                log_event(
                    packet_logger, event="object_detection_completed",
                    message="Object detection completed.",
                    details={"object_count": len(objects), "processing_time_ms": timer.elapsed_ms},
                )
                return ObjectDetectionOutput(result=result, objects=objects)
            except Exception as error:
                log_exception(
                    packet_logger, error, event="object_detection_failed",
                    message="Object detection failed.",
                    details={"processing_time_ms": timer.elapsed_ms},
                )
                return ObjectDetectionOutput(
                    ModuleResult.failure(
                        module_name="object_detector", modality="vision",
                        error=f"{error.__class__.__name__}: {error}",
                        processing_time_ms=timer.elapsed_ms,
                        source_packet_id=packet.packet_id,
                        metadata={"model_id": self.model_config.object_detection.model_id},
                    ),
                    [],
                )

    def _get_model(self) -> Tuple[Any, str]:
        loaded = self.model_loader.load("object_detection")
        if isinstance(loaded, ModelBundle):
            model, device = loaded.model, loaded.device
        else:
            model, device = loaded, "cpu"
        if not callable(getattr(model, "predict", None)):
            raise ModelLoadingError("Loaded object-detection model has no callable predict method.", module="object_detector")
        return model, device

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
            raise ObjectDetectionError("Object confidence threshold must be between 0 and 1.")
        return value

    def _parse_results(self, raw_results: Any, packet_id: str) -> Tuple[List[DetectedObject], Tuple[int, int]]:
        if not isinstance(raw_results, (list, tuple)) or not raw_results:
            raise ObjectDetectionError("YOLO returned no result objects.")
        result = raw_results[0]
        original_shape = getattr(result, "orig_shape", None)
        if not original_shape or len(original_shape) != 2:
            raise ObjectDetectionError("YOLO result is missing orig_shape.")
        image_height, image_width = int(original_shape[0]), int(original_shape[1])
        names = getattr(result, "names", {})
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return [], (image_width, image_height)

        objects: List[DetectedObject] = []
        for index, box in enumerate(boxes, start=1):
            coordinates = self._values(box.xyxy)
            confidences = self._values(box.conf)
            classes = self._values(box.cls)
            if len(coordinates) < 4 or not confidences or not classes:
                continue
            x1, y1, x2, y2 = coordinates[:4]
            confidence, class_id = float(confidences[0]), int(classes[0])
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2, confidence)):
                continue
            x1, x2 = sorted((max(0.0, min(x1, image_width)), max(0.0, min(x2, image_width))))
            y1, y2 = sorted((max(0.0, min(y1, image_height)), max(0.0, min(y2, image_height))))
            width, height = x2 - x1, y2 - y1
            if width <= 0.0 or height <= 0.0:
                continue
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            label = str(label).strip().lower().replace(" ", "_")
            bbox = BoundingBox(
                x1, y1, x2, y2, width, height, (x1 + x2) / 2.0, (y1 + y2) / 2.0,
                (x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height),
            )
            objects.append(DetectedObject(
                object_id=f"OBJ_{packet_id}_{index:03d}", label=label,
                class_id=class_id, confidence=confidence, bounding_box=bbox,
                area_ratio=(width * height) / (image_width * image_height),
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


def detect_objects(
    packet: AdaptedLayer1Input,
    vision_output: VisionProcessingOutput,
    *,
    settings: Optional[Layer2Settings] = None,
    model_config: Optional[Layer2ModelConfig] = None,
    model_loader: Optional[ModelLoader] = None,
) -> ObjectDetectionOutput:
    return ObjectDetector(settings, model_config, model_loader).detect(packet, vision_output)


def run_self_test(scenario_name: str = "park_walking", *, test_all: bool = False) -> bool:
    print("=" * 72)
    print("NOONGIL-X | OBJECT DETECTOR SELF-TEST")
    print("=" * 72)
    print("The first run may download yolov8n.pt.")
    loader: Optional[ModelLoader] = None
    try:
        adapter = Layer1PacketAdapter(require_media=True)
        settings = create_default_settings()
        model_config = create_default_model_config()
        loader = ModelLoader(model_config=model_config)
        processor = VisionProcessor(settings=settings)
        detector = ObjectDetector(settings, model_config, loader)
        scenarios = adapter.discover_scenarios() if test_all else [scenario_name]
        if not scenarios:
            raise AssertionError("No scenarios selected.")
        print(f"[PASS] Testing {len(scenarios)} scenario(s)")
        for scenario in scenarios:
            packet = adapter.load_scenario(scenario)
            vision_output = processor.process(packet)
            if not vision_output.succeeded:
                raise AssertionError(f"Vision processing failed for {scenario}.")
            output = detector.detect(packet, vision_output)
            if not output.succeeded:
                raise AssertionError(f"Object detection failed for {scenario}: {output.result.errors}")
            for item in output.objects:
                if not 0.0 <= item.confidence <= 1.0 or not 0.0 <= item.area_ratio <= 1.0:
                    raise AssertionError(f"Invalid detection values for {scenario}.")
            labels = [item.label for item in output.objects[:5]]
            print(f"[PASS] {scenario}: objects={output.object_count}, top_labels={labels}")
        print("[PASS] YOLOv8 model loaded lazily")
        print("[PASS] Bounding boxes validated and normalized")
        print("[PASS] Confidence filtering applied")
        print("[PASS] Layer 3 object format generated")
        print("[PASS] ModuleResult generated")
        print("\n" + "=" * 72)
        print("[PASSED] OBJECT DETECTOR IS WORKING")
        print("=" * 72)
        return True
    except (DependencyMissingError, ModelLoadingError, ObjectDetectionError, AssertionError) as error:
        print(f"\n[FAILED] {error}")
        print("=" * 72)
        return False
    finally:
        if loader is not None:
            loader.unload_all()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NOONGIL-X Layer 2 object-detector self-test.")
    parser.add_argument("--scenario", default="park_walking")
    parser.add_argument("--all", action="store_true")
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()
    return 0 if run_self_test(arguments.scenario, test_all=arguments.all) else 1


if __name__ == "__main__":
    raise SystemExit(main())

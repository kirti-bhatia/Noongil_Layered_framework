"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : OCR Engine
File    : layer2/text/ocr_engine.py
============================================================

Purpose
-------
Performs raw environmental-text detection and recognition using
PaddleOCR.

Responsibilities:
- load PaddleOCR lazily
- prepare the input image
- detect text regions
- recognize text
- validate confidence scores
- validate and normalize bounding boxes
- return standardized ModuleResult

Text cleaning, reading order and semantic interpretation belong
to text_interpreter.py.
============================================================
"""

from __future__ import annotations

import argparse
import math

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
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

from layer2.schemas.module_result import (
    ModuleResult,
)

from layer2.utils.exceptions import (
    DependencyMissingError,
    ModelLoadingError,
)

from layer2.utils.logger import (
    Layer2LoggerAdapter,
    ModuleTimer,
    get_logger,
    log_event,
    log_exception,
)

from layer2.utils.model_loader import (
    ModelBundle,
    ModelLoader,
)

from layer2.vision_perception.vision_processor import (
    VisionProcessingOutput,
    VisionProcessor,
)


# ============================================================
# CONSTANTS
# ============================================================

OCR_ENGINE_VERSION = "1.0"

DEFAULT_OCR_CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# EXCEPTION
# ============================================================

class OCREngineError(Exception):
    """Raised when OCR inference cannot be completed."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class OCRBoundingBox:
    """One quadrilateral OCR bounding box."""

    points: Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
    ]

    x1: float
    y1: float
    x2: float
    y2: float

    width: float
    height: float

    center_x: float
    center_y: float

    normalized_xyxy: Tuple[
        float,
        float,
        float,
        float,
    ]

    def to_dict(self) -> Dict[str, Any]:

        return {
            "points": [
                {
                    "x": round(x, 3),
                    "y": round(y, 3),
                }
                for x, y in self.points
            ],
            "x1": round(self.x1, 3),
            "y1": round(self.y1, 3),
            "x2": round(self.x2, 3),
            "y2": round(self.y2, 3),
            "width": round(
                self.width,
                3,
            ),
            "height": round(
                self.height,
                3,
            ),
            "center": {
                "x": round(
                    self.center_x,
                    3,
                ),
                "y": round(
                    self.center_y,
                    3,
                ),
            },
            "normalized_xyxy": [
                round(value, 6)
                for value
                in self.normalized_xyxy
            ],
        }


@dataclass(frozen=True)
class OCRDetection:
    """One raw OCR detection."""

    detection_id: str
    text: str
    confidence: float
    bounding_box: OCRBoundingBox

    def to_dict(self) -> Dict[str, Any]:

        return {
            "detection_id": (
                self.detection_id
            ),
            "text": self.text,
            "confidence": round(
                self.confidence,
                6,
            ),
            "bounding_box": (
                self.bounding_box.to_dict()
            ),
        }


@dataclass
class OCREngineOutput:
    """Complete raw OCR-engine output."""

    result: ModuleResult

    detections: List[OCRDetection]

    image_width: Optional[int]
    image_height: Optional[int]

    @property
    def succeeded(self) -> bool:
        return self.result.usable

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "detection_count": (
                self.detection_count
            ),
            "detections": [
                detection.to_dict()
                for detection
                in self.detections
            ],
            "image_size": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "result": self.result.to_dict(),
        }


# ============================================================
# OCR ENGINE
# ============================================================

class OCREngine:
    """PaddleOCR environmental-text engine."""

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

        default_project_root = (
            Path(__file__).resolve().parents[2]
        )

        self.project_root = Path(
            project_root
            or default_project_root
        ).resolve()

        self.logger = (
            logger
            or get_logger(
                "ocr_engine"
            )
        )

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

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def recognize(
        self,
        packet: AdaptedLayer1Input,
        vision_output: VisionProcessingOutput,
    ) -> OCREngineOutput:
        """Run raw OCR inference on one image."""

        if not isinstance(
            packet,
            AdaptedLayer1Input,
        ):
            raise OCREngineError(
                "packet must be an "
                "AdaptedLayer1Input."
            )

        if not isinstance(
            vision_output,
            VisionProcessingOutput,
        ):
            raise OCREngineError(
                "vision_output must be a "
                "VisionProcessingOutput."
            )

        if not self._ocr_enabled():

            result = ModuleResult.skipped(
                module_name="ocr_engine",
                modality="vision",
                reason=(
                    "OCR is disabled in "
                    "Layer 2 settings."
                ),
                source_packet_id=(
                    packet.packet_id
                ),
            )

            return OCREngineOutput(
                result=result,
                detections=[],
                image_width=None,
                image_height=None,
            )

        if (
            not vision_output.succeeded
            or vision_output.image_rgb is None
        ):
            result = ModuleResult.failure(
                module_name="ocr_engine",
                modality="vision",
                error=(
                    "A successful processed vision "
                    "frame is required."
                ),
                source_packet_id=(
                    packet.packet_id
                ),
            )

            return OCREngineOutput(
                result=result,
                detections=[],
                image_width=None,
                image_height=None,
            )

        packet_logger = self.logger.bind(
            packet_id=packet.packet_id,
            scenario=packet.scenario,
        )

        log_event(
            packet_logger,
            event="ocr_started",
            message="OCR inference started.",
        )

        with ModuleTimer(
            "ocr_engine",
            logger=packet_logger,
            packet_id=packet.packet_id,
            log_start=False,
            log_completion=False,
        ) as timer:

            try:
                model = self._get_model()

                image_bgr = (
                    self._prepare_image(
                        vision_output
                    )
                )

                image_height = int(
                    image_bgr.shape[0]
                )

                image_width = int(
                    image_bgr.shape[1]
                )

                raw_output = self._run_model(
                    model,
                    image_bgr,
                )

                threshold = (
                    self._confidence_threshold()
                )

                detections = (
                    self._parse_output(
                        raw_output=raw_output,
                        image_width=image_width,
                        image_height=image_height,
                        confidence_threshold=(
                            threshold
                        ),
                        packet_id=(
                            packet.packet_id
                        ),
                    )
                )

                confidence = (
                    sum(
                        detection.confidence
                        for detection
                        in detections
                    )
                    / len(detections)
                    if detections
                    else 0.0
                )

                warnings = []

                if not detections:
                    warnings.append(
                        "No text met the configured "
                        "OCR confidence threshold."
                    )

                data = {
                    "detection_count": (
                        len(detections)
                    ),
                    "detections": [
                        detection.to_dict()
                        for detection
                        in detections
                    ],
                    "image_size": {
                        "width": image_width,
                        "height": image_height,
                    },
                    "threshold": threshold,
                    "language": (
                        self.model_config
                        .ocr
                        .parameters
                        .get(
                            "language",
                            "en",
                        )
                    ),
                    "model_id": (
                        self.model_config
                        .ocr
                        .model_id
                    ),
                    "engine_version": (
                        OCR_ENGINE_VERSION
                    ),
                }

                status_factory = (
                    ModuleResult.success
                    if detections
                    else ModuleResult.partial
                )

                result = status_factory(
                    module_name="ocr_engine",
                    modality="vision",
                    data=data,
                    confidence=confidence,
                    processing_time_ms=(
                        timer.elapsed_ms
                    ),
                    source_packet_id=(
                        packet.packet_id
                    ),
                    warnings=warnings,
                    metadata={
                        "scenario": (
                            packet.scenario
                        ),
                        "model_backend": (
                            self.model_config
                            .ocr
                            .backend.value
                        ),
                    },
                )

                log_event(
                    packet_logger,
                    event="ocr_completed",
                    message=(
                        "OCR inference completed."
                    ),
                    details={
                        "detection_count": (
                            len(detections)
                        ),
                        "confidence": confidence,
                        "processing_time_ms": (
                            timer.elapsed_ms
                        ),
                    },
                )

                return OCREngineOutput(
                    result=result,
                    detections=detections,
                    image_width=image_width,
                    image_height=image_height,
                )

            except Exception as error:

                log_exception(
                    packet_logger,
                    error,
                    event="ocr_failed",
                    message=(
                        "OCR inference failed."
                    ),
                    details={
                        "processing_time_ms": (
                            timer.elapsed_ms
                        )
                    },
                )

                result = ModuleResult.failure(
                    module_name="ocr_engine",
                    modality="vision",
                    error=(
                        f"{error.__class__.__name__}: "
                        f"{error}"
                    ),
                    processing_time_ms=(
                        timer.elapsed_ms
                    ),
                    source_packet_id=(
                        packet.packet_id
                    ),
                    metadata={
                        "model_id": (
                            self.model_config
                            .ocr
                            .model_id
                        )
                    },
                )

                return OCREngineOutput(
                    result=result,
                    detections=[],
                    image_width=None,
                    image_height=None,
                )

    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    def _ocr_enabled(self) -> bool:
        """Return whether OCR is enabled."""

        modules = self.settings.modules

        if hasattr(modules, "ocr"):
            return bool(
                modules.ocr
            )

        if hasattr(
            modules,
            "text_recognition",
        ):
            return bool(
                modules.text_recognition
            )

        return True

    def _confidence_threshold(
        self,
    ) -> float:
        """Return the configured OCR threshold."""

        value = getattr(
            self.settings.vision,
            "ocr_confidence_threshold",
            DEFAULT_OCR_CONFIDENCE_THRESHOLD,
        )

        value = float(value)

        if (
            not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise OCREngineError(
                "OCR confidence threshold must "
                "be between 0 and 1."
            )

        return value

    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    def _get_model(self) -> Any:
        """Load and return PaddleOCR."""

        loaded = self.model_loader.load(
            "ocr"
        )

        if isinstance(
            loaded,
            ModelBundle,
        ):
            model = loaded.model
        else:
            model = loaded

        if not (
            callable(
                getattr(
                    model,
                    "ocr",
                    None,
                )
            )
            or callable(
                getattr(
                    model,
                    "predict",
                    None,
                )
            )
        ):
            raise ModelLoadingError(
                "Loaded OCR model has neither "
                "ocr() nor predict().",
                module="ocr_engine",
            )

        return model

    @staticmethod
    def _prepare_image(
        vision_output: VisionProcessingOutput,
    ) -> Any:
        """Convert the RGB frame to BGR."""

        try:
            import numpy as np

        except ImportError as error:
            raise DependencyMissingError(
                "NumPy is required for OCR.",
                module="ocr_engine",
            ) from error

        image_rgb = (
            vision_output.image_rgb
        )

        if (
            image_rgb is None
            or not hasattr(
                image_rgb,
                "shape",
            )
            or len(image_rgb.shape) != 3
            or image_rgb.shape[2] != 3
        ):
            raise OCREngineError(
                "Expected an RGB image array."
            )

        image_rgb = image_rgb.astype(
            np.uint8,
            copy=False,
        )

        return image_rgb[:, :, ::-1].copy()

    def _run_model(
        self,
        model: Any,
        image_bgr: Any,
    ) -> Any:
        """Run PaddleOCR 2.x or compatible API."""

        ocr_method = getattr(
            model,
            "ocr",
            None,
        )

        if callable(ocr_method):

            use_angle_cls = bool(
                self.model_config
                .ocr
                .parameters
                .get(
                    "use_angle_cls",
                    True,
                )
            )

            try:
                return ocr_method(
                    image_bgr,
                    cls=use_angle_cls,
                )

            except TypeError:
                return ocr_method(
                    image_bgr
                )

        predict_method = getattr(
            model,
            "predict",
            None,
        )

        if callable(predict_method):
            return predict_method(
                image_bgr
            )

        raise OCREngineError(
            "No supported OCR inference "
            "method is available."
        )

    # --------------------------------------------------------
    # OUTPUT PARSING
    # --------------------------------------------------------

    def _parse_output(
        self,
        *,
        raw_output: Any,
        image_width: int,
        image_height: int,
        confidence_threshold: float,
        packet_id: str,
    ) -> List[OCRDetection]:
        """Parse PaddleOCR output."""

        lines = self._extract_lines(
            raw_output
        )

        detections = []

        for line in lines:

            parsed = self._parse_line(
                line=line,
                image_width=image_width,
                image_height=image_height,
            )

            if parsed is None:
                continue

            text, confidence, box = parsed

            if confidence < confidence_threshold:
                continue

            detections.append(
                OCRDetection(
                    detection_id=(
                        f"OCR_{packet_id}_"
                        f"{len(detections) + 1:03d}"
                    ),
                    text=text,
                    confidence=confidence,
                    bounding_box=box,
                )
            )

        return detections

    @staticmethod
    def _extract_lines(
        raw_output: Any,
    ) -> List[Any]:
        """Flatten single-image PaddleOCR output."""

        if raw_output is None:
            return []

        if not isinstance(
            raw_output,
            (list, tuple),
        ):
            return []

        output = list(raw_output)

        if not output:
            return []

        if (
            len(output) == 1
            and output[0] is None
        ):
            return []

        if (
            len(output) == 1
            and isinstance(
                output[0],
                (list, tuple),
            )
        ):
            first = list(
                output[0]
            )

            if not first:
                return []

            if (
                isinstance(
                    first[0],
                    (list, tuple),
                )
                and len(first[0]) == 2
            ):
                return first

        return output

    def _parse_line(
        self,
        *,
        line: Any,
        image_width: int,
        image_height: int,
    ) -> Optional[
        Tuple[
            str,
            float,
            OCRBoundingBox,
        ]
    ]:
        """Parse one OCR result line."""

        if not isinstance(
            line,
            (list, tuple),
        ):
            return None

        if len(line) != 2:
            return None

        raw_box = line[0]
        recognition = line[1]

        if not isinstance(
            recognition,
            (list, tuple),
        ):
            return None

        if len(recognition) < 2:
            return None

        text = recognition[0]
        confidence = recognition[1]

        if (
            not isinstance(text, str)
            or not text.strip()
        ):
            return None

        if (
            not isinstance(
                confidence,
                (int, float),
            )
            or isinstance(
                confidence,
                bool,
            )
        ):
            return None

        confidence = float(
            confidence
        )

        if (
            not math.isfinite(confidence)
            or not 0.0
            <= confidence
            <= 1.0
        ):
            return None

        bounding_box = self._parse_box(
            raw_box=raw_box,
            image_width=image_width,
            image_height=image_height,
        )

        if bounding_box is None:
            return None

        return (
            text.strip(),
            confidence,
            bounding_box,
        )

    @staticmethod
    def _parse_box(
        *,
        raw_box: Any,
        image_width: int,
        image_height: int,
    ) -> Optional[OCRBoundingBox]:
        """Validate and normalize one OCR box."""

        if not isinstance(
            raw_box,
            (list, tuple),
        ):
            return None

        if len(raw_box) != 4:
            return None

        points = []

        for raw_point in raw_box:

            if not isinstance(
                raw_point,
                (list, tuple),
            ):
                return None

            if len(raw_point) < 2:
                return None

            try:
                x = float(
                    raw_point[0]
                )

                y = float(
                    raw_point[1]
                )

            except (
                TypeError,
                ValueError,
            ):
                return None

            if not (
                math.isfinite(x)
                and math.isfinite(y)
            ):
                return None

            x = max(
                0.0,
                min(
                    x,
                    float(image_width),
                ),
            )

            y = max(
                0.0,
                min(
                    y,
                    float(image_height),
                ),
            )

            points.append(
                (x, y)
            )

        x_values = [
            point[0]
            for point in points
        ]

        y_values = [
            point[1]
            for point in points
        ]

        x1 = min(x_values)
        y1 = min(y_values)

        x2 = max(x_values)
        y2 = max(y_values)

        width = x2 - x1
        height = y2 - y1

        if (
            width <= 0.0
            or height <= 0.0
            or image_width <= 0
            or image_height <= 0
        ):
            return None

        return OCRBoundingBox(
            points=(
                points[0],
                points[1],
                points[2],
                points[3],
            ),
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            width=width,
            height=height,
            center_x=(
                (x1 + x2) / 2.0
            ),
            center_y=(
                (y1 + y2) / 2.0
            ),
            normalized_xyxy=(
                x1 / image_width,
                y1 / image_height,
                x2 / image_width,
                y2 / image_height,
            ),
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def run_ocr(
    packet: AdaptedLayer1Input,
    vision_output: VisionProcessingOutput,
    *,
    settings: Optional[
        Layer2Settings
    ] = None,
    model_config: Optional[
        Layer2ModelConfig
    ] = None,
    model_loader: Optional[
        ModelLoader
    ] = None,
) -> OCREngineOutput:

    engine = OCREngine(
        settings=settings,
        model_config=model_config,
        model_loader=model_loader,
    )

    return engine.recognize(
        packet,
        vision_output,
    )


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test(
    scenario_name: str = "classroom",
    *,
    test_all: bool = False,
) -> bool:

    print("=" * 72)
    print(
        "NOONGIL-X | OCR ENGINE SELF-TEST"
    )
    print("=" * 72)

    print(
        "The first run may download "
        "PaddleOCR models."
    )

    loader: Optional[ModelLoader] = None

    try:
        adapter = Layer1PacketAdapter(
            require_media=True
        )

        settings = (
            create_default_settings()
        )

        model_config = (
            create_default_model_config()
        )

        loader = ModelLoader(
            model_config=model_config
        )

        processor = VisionProcessor(
            settings=settings
        )

        engine = OCREngine(
            settings=settings,
            model_config=model_config,
            model_loader=loader,
        )

        scenarios = (
            adapter.discover_scenarios()
            if test_all
            else [scenario_name]
        )

        if not scenarios:
            raise AssertionError(
                "No scenarios selected."
            )

        print(
            f"[PASS] Testing "
            f"{len(scenarios)} scenario(s)"
        )

        for scenario in scenarios:

            packet = adapter.load_scenario(
                scenario
            )

            vision_output = processor.process(
                packet
            )

            if not vision_output.succeeded:
                raise AssertionError(
                    "Vision processing failed "
                    f"for {scenario}."
                )

            output = engine.recognize(
                packet,
                vision_output,
            )

            if not output.succeeded:
                raise AssertionError(
                    "OCR failed "
                    f"for {scenario}: "
                    f"{output.result.errors}"
                )

            for detection in output.detections:

                if not (
                    0.0
                    <= detection.confidence
                    <= 1.0
                ):
                    raise AssertionError(
                        "Invalid OCR confidence "
                        f"for {scenario}."
                    )

                normalized_box = (
                    detection
                    .bounding_box
                    .normalized_xyxy
                )

                if not all(
                    0.0 <= value <= 1.0
                    for value
                    in normalized_box
                ):
                    raise AssertionError(
                        "Invalid normalized OCR box "
                        f"for {scenario}."
                    )

            preview = [
                detection.text
                for detection
                in output.detections[:5]
            ]

            print(
                f"[PASS] {scenario}: "
                f"detections="
                f"{output.detection_count}, "
                f"text={preview}"
            )

        print(
            "[PASS] PaddleOCR loaded lazily"
        )

        print(
            "[PASS] OCR confidence filtering applied"
        )

        print(
            "[PASS] OCR boxes validated"
        )

        print(
            "[PASS] Bounding boxes normalized"
        )

        print(
            "[PASS] Raw OCR output generated"
        )

        print(
            "[PASS] ModuleResult generated"
        )

        print("\n" + "=" * 72)

        print(
            "[PASSED] OCR ENGINE IS WORKING"
        )

        print("=" * 72)

        return True

    except (
        DependencyMissingError,
        ModelLoadingError,
        OCREngineError,
        AssertionError,
    ) as error:

        print(f"\n[FAILED] {error}")

        print("=" * 72)

        return False

    finally:
        if loader is not None:
            loader.unload_all()


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Run the NOONGIL-X Layer 2 "
            "OCR-engine self-test."
        )
    )

    parser.add_argument(
        "--scenario",
        default="classroom",
        help=(
            "Scenario tested by default. "
            "Default: classroom"
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all eight scenarios.",
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
            arguments.scenario,
            test_all=arguments.all,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
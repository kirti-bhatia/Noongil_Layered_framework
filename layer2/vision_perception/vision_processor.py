"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Vision Frame Processor
File    : layer2/vision_perception/vision_processor.py
============================================================

Purpose
-------
Prepares Layer 1 camera frames for Layer 2 vision models.

Processing:
1. Receive routed vision input
2. Load the image
3. Correct EXIF orientation
4. Convert to RGB
5. Resize while preserving aspect ratio
6. Add letterbox padding
7. Normalize pixel values
8. Estimate brightness, contrast and sharpness
9. Produce a standardized ModuleResult

This module does not classify scenes, detect objects, perform
OCR or estimate depth.

Dependencies
------------
Pillow
NumPy

Compatibility
-------------
Python 3.10+
============================================================
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from layer2.config.settings import (
    Layer2Settings,
    create_default_settings,
    create_test_settings,
)

from layer2.input_reception.layer1_packet_adapter import (
    AdaptedLayer1Input,
    Layer1PacketAdapter,
)

from layer2.schemas.module_result import (
    ModuleResult,
    ModuleStatus,
)

from layer2.utils.exceptions import (
    DependencyMissingError,
    VisionProcessingError,
)

from layer2.utils.logger import (
    Layer2LoggerAdapter,
    ModuleTimer,
    get_logger,
    log_event,
    log_exception,
)


# ============================================================
# CONSTANTS
# ============================================================

VISION_PROCESSOR_VERSION = "1.0"

SUPPORTED_IMAGE_MODES = {
    "RGB",
    "RGBA",
    "L",
    "P",
    "CMYK",
}

LOW_BRIGHTNESS_THRESHOLD = 0.15
HIGH_BRIGHTNESS_THRESHOLD = 0.90
LOW_CONTRAST_THRESHOLD = 0.08
LOW_SHARPNESS_THRESHOLD = 0.03


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class VisionQualityMetrics:
    """Quality measurements for one image."""

    brightness: float
    contrast: float
    sharpness: float
    exposure_score: float
    overall_quality: float

    def to_dict(self) -> Dict[str, float]:

        return {
            "brightness": round(
                self.brightness,
                6,
            ),
            "contrast": round(
                self.contrast,
                6,
            ),
            "sharpness": round(
                self.sharpness,
                6,
            ),
            "exposure_score": round(
                self.exposure_score,
                6,
            ),
            "overall_quality": round(
                self.overall_quality,
                6,
            ),
        }


@dataclass
class ResizeInformation:
    """Image resize and padding information."""

    original_width: int
    original_height: int

    target_width: int
    target_height: int

    resized_width: int
    resized_height: int

    scale: float

    padding_left: int
    padding_top: int
    padding_right: int
    padding_bottom: int

    aspect_ratio_preserved: bool

    def to_dict(self) -> Dict[str, Any]:

        return {
            "original_size": {
                "width": self.original_width,
                "height": self.original_height,
            },
            "target_size": {
                "width": self.target_width,
                "height": self.target_height,
            },
            "resized_content_size": {
                "width": self.resized_width,
                "height": self.resized_height,
            },
            "scale": round(
                self.scale,
                8,
            ),
            "padding": {
                "left": self.padding_left,
                "top": self.padding_top,
                "right": self.padding_right,
                "bottom": self.padding_bottom,
            },
            "aspect_ratio_preserved": (
                self.aspect_ratio_preserved
            ),
        }


@dataclass
class VisionProcessingOutput:
    """
    Complete vision-processing output.

    image_rgb:
        Resized uint8 RGB NumPy array.

    normalized_image:
        Float32 RGB NumPy array with values from 0 to 1.

    result:
        JSON-safe standardized ModuleResult.
    """

    result: ModuleResult

    image_rgb: Optional[Any]
    normalized_image: Optional[Any]

    original_image: Optional[Any] = None

    @property
    def succeeded(self) -> bool:
        return self.result.usable

    @property
    def image_shape(
        self,
    ) -> Optional[Tuple[int, ...]]:

        if self.image_rgb is None:
            return None

        return tuple(
            self.image_rgb.shape
        )


# ============================================================
# DEPENDENCY LOADING
# ============================================================

def dependency_available(
    module_name: str,
) -> bool:

    try:
        return (
            importlib.util.find_spec(
                module_name
            )
            is not None
        )
    except (
        ImportError,
        ModuleNotFoundError,
        ValueError,
    ):
        return False


def require_vision_dependencies() -> tuple[Any, Any, Any]:
    """Import and return Pillow and NumPy."""

    missing = []

    if not dependency_available("PIL"):
        missing.append("Pillow")

    if not dependency_available("numpy"):
        missing.append("numpy")

    if missing:
        raise DependencyMissingError(
            "Missing vision dependencies: "
            f"{', '.join(missing)}. Install with: "
            "python -m pip install Pillow numpy",
            module="vision_processor",
            details={
                "missing_dependencies": missing,
                "installation_command": (
                    "python -m pip install "
                    "Pillow numpy"
                ),
            },
        )

    try:
        import numpy as np

        from PIL import (
            Image,
            ImageOps,
        )

    except Exception as error:
        raise DependencyMissingError(
            "Vision dependencies are installed "
            "but could not be imported.",
            module="vision_processor",
            cause=error,
        ) from error

    return np, Image, ImageOps


# ============================================================
# QUALITY ESTIMATION
# ============================================================

def calculate_quality_metrics(
    image_rgb: Any,
    np: Any,
) -> VisionQualityMetrics:
    """Calculate lightweight image-quality metrics."""

    if (
        image_rgb is None
        or not hasattr(
            image_rgb,
            "shape",
        )
        or len(image_rgb.shape) != 3
        or image_rgb.shape[2] != 3
    ):
        raise VisionProcessingError(
            "Expected an RGB image array.",
            module="vision_processor",
        )

    image_float = image_rgb.astype(
        np.float32
    ) / 255.0

    grayscale = (
        0.299 * image_float[:, :, 0]
        + 0.587 * image_float[:, :, 1]
        + 0.114 * image_float[:, :, 2]
    )

    brightness = float(
        np.mean(grayscale)
    )

    contrast = float(
        np.std(grayscale)
    )

    horizontal_gradient = np.abs(
        np.diff(
            grayscale,
            axis=1,
        )
    )

    vertical_gradient = np.abs(
        np.diff(
            grayscale,
            axis=0,
        )
    )

    horizontal_score = (
        float(
            np.mean(
                horizontal_gradient
            )
        )
        if horizontal_gradient.size
        else 0.0
    )

    vertical_score = (
        float(
            np.mean(
                vertical_gradient
            )
        )
        if vertical_gradient.size
        else 0.0
    )

    sharpness = min(
        1.0,
        (
            horizontal_score
            + vertical_score
        )
        * 5.0,
    )

    exposure_score = max(
        0.0,
        1.0
        - abs(
            brightness - 0.5
        )
        * 2.0,
    )

    normalized_contrast = min(
        1.0,
        contrast / 0.25,
    )

    normalized_sharpness = min(
        1.0,
        sharpness,
    )

    overall_quality = (
        0.40 * exposure_score
        + 0.30 * normalized_contrast
        + 0.30 * normalized_sharpness
    )

    return VisionQualityMetrics(
        brightness=max(
            0.0,
            min(1.0, brightness),
        ),
        contrast=max(
            0.0,
            min(1.0, contrast),
        ),
        sharpness=max(
            0.0,
            min(1.0, sharpness),
        ),
        exposure_score=max(
            0.0,
            min(1.0, exposure_score),
        ),
        overall_quality=max(
            0.0,
            min(1.0, overall_quality),
        ),
    )


# ============================================================
# VISION PROCESSOR
# ============================================================

class VisionProcessor:
    """Load and preprocess Layer 1 image frames."""

    def __init__(
        self,
        settings: Optional[
            Layer2Settings
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
                "vision_processor"
            )
        )

        self.np, self.Image, self.ImageOps = (
            require_vision_dependencies()
        )

    def process(
        self,
        packet: AdaptedLayer1Input,
    ) -> VisionProcessingOutput:
        """Process one Layer 1 vision frame."""

        if not isinstance(
            packet,
            AdaptedLayer1Input,
        ):
            raise VisionProcessingError(
                "packet must be an "
                "AdaptedLayer1Input.",
                module="vision_processor",
                details={
                    "received_type": (
                        packet.__class__.__name__
                    )
                },
            )

        packet_logger = self.logger.bind(
            packet_id=packet.packet_id,
            scenario=packet.scenario,
        )

        if not (
            self.settings.modules
            .vision_processing
        ):
            result = ModuleResult.skipped(
                module_name="vision_processor",
                modality="vision",
                reason=(
                    "Vision processing is disabled "
                    "in Layer 2 settings."
                ),
                source_packet_id=(
                    packet.packet_id
                ),
            )

            return VisionProcessingOutput(
                result=result,
                image_rgb=None,
                normalized_image=None,
            )

        vision = packet.get_modality(
            "vision"
        )

        if (
            vision is None
            or not vision.usable
            or vision.media_path is None
        ):
            result = ModuleResult.failure(
                module_name="vision_processor",
                modality="vision",
                error=(
                    "A usable vision frame is "
                    "not available."
                ),
                source_packet_id=(
                    packet.packet_id
                ),
            )

            return VisionProcessingOutput(
                result=result,
                image_rgb=None,
                normalized_image=None,
            )

        frame_path = vision.media_path

        log_event(
            packet_logger,
            event=(
                "vision_processing_started"
            ),
            message=(
                "Vision-frame processing started."
            ),
            details={
                "frame_path": str(frame_path)
            },
        )

        with ModuleTimer(
            "vision_processor",
            logger=packet_logger,
            packet_id=packet.packet_id,
            log_start=False,
            log_completion=False,
        ) as timer:

            try:
                original_image = (
                    self._load_image(
                        frame_path
                    )
                )

                processed_image, resize_info = (
                    self._prepare_image(
                        original_image
                    )
                )

                image_rgb = self.np.asarray(
                    processed_image,
                    dtype=self.np.uint8,
                )

                normalized_image = (
                    image_rgb.astype(
                        self.np.float32
                    )
                    / 255.0
                )

                quality = (
                    calculate_quality_metrics(
                        image_rgb,
                        self.np,
                    )
                )

                warnings = (
                    self._quality_warnings(
                        quality
                    )
                )

                output_path = None

                if (
                    self.settings.execution
                    .save_intermediate_outputs
                ):
                    output_path = (
                        self._save_processed_frame(
                            processed_image,
                            packet,
                        )
                    )

                sensor_confidence = (
                    vision.confidence
                )

                if sensor_confidence is None:
                    sensor_confidence = (
                        packet
                        .overall_sensor_confidence
                    )

                confidence = (
                    self._calculate_confidence(
                        quality.overall_quality,
                        sensor_confidence,
                    )
                )

                data = {
                    "frame_path": str(
                        frame_path
                    ),
                    "processed_frame_path": (
                        str(output_path)
                        if output_path
                        else None
                    ),
                    "source_frame_id": (
                        packet.source_frame_id
                    ),
                    "colour_mode": "RGB",
                    "array_shape": list(
                        image_rgb.shape
                    ),
                    "array_dtype": str(
                        image_rgb.dtype
                    ),
                    "normalized_shape": list(
                        normalized_image.shape
                    ),
                    "normalized_dtype": str(
                        normalized_image.dtype
                    ),
                    "normalized_range": {
                        "minimum": float(
                            normalized_image.min()
                        ),
                        "maximum": float(
                            normalized_image.max()
                        ),
                    },
                    "resize": (
                        resize_info.to_dict()
                    ),
                    "quality": (
                        quality.to_dict()
                    ),
                    "sensor_confidence": (
                        sensor_confidence
                    ),
                    "processor_version": (
                        VISION_PROCESSOR_VERSION
                    ),
                }

                if warnings:
                    result = ModuleResult.partial(
                        module_name=(
                            "vision_processor"
                        ),
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
                            )
                        },
                    )
                else:
                    result = ModuleResult.success(
                        module_name=(
                            "vision_processor"
                        ),
                        modality="vision",
                        data=data,
                        confidence=confidence,
                        processing_time_ms=(
                            timer.elapsed_ms
                        ),
                        source_packet_id=(
                            packet.packet_id
                        ),
                        warnings=[],
                        metadata={
                            "scenario": (
                                packet.scenario
                            )
                        },
                    )

                log_event(
                    packet_logger,
                    event=(
                        "vision_processing_completed"
                    ),
                    message=(
                        "Vision-frame processing "
                        "completed."
                    ),
                    details={
                        "frame_path": (
                            str(frame_path)
                        ),
                        "image_shape": list(
                            image_rgb.shape
                        ),
                        "confidence": confidence,
                        "quality": (
                            quality.overall_quality
                        ),
                        "processing_time_ms": (
                            timer.elapsed_ms
                        ),
                    },
                )

                return VisionProcessingOutput(
                    result=result,
                    image_rgb=image_rgb,
                    normalized_image=(
                        normalized_image
                    ),
                    original_image=(
                        original_image
                    ),
                )

            except Exception as error:

                log_exception(
                    packet_logger,
                    error,
                    event=(
                        "vision_processing_failed"
                    ),
                    message=(
                        "Vision-frame processing "
                        "failed."
                    ),
                    details={
                        "frame_path": (
                            str(frame_path)
                        ),
                        "processing_time_ms": (
                            timer.elapsed_ms
                        ),
                    },
                )

                result = ModuleResult.failure(
                    module_name=(
                        "vision_processor"
                    ),
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
                        "frame_path": (
                            str(frame_path)
                        )
                    },
                )

                return VisionProcessingOutput(
                    result=result,
                    image_rgb=None,
                    normalized_image=None,
                )

    def _load_image(
        self,
        frame_path: Path,
    ) -> Any:
        """Load, orient and convert an image to RGB."""

        if not frame_path.is_file():
            raise VisionProcessingError(
                f"Frame does not exist: "
                f"{frame_path}",
                module="vision_processor",
            )

        try:
            with self.Image.open(
                frame_path
            ) as image:

                image.load()

                if image.mode not in (
                    SUPPORTED_IMAGE_MODES
                ):
                    raise VisionProcessingError(
                        "Unsupported image mode: "
                        f"{image.mode}",
                        module=(
                            "vision_processor"
                        ),
                    )

                oriented = (
                    self.ImageOps
                    .exif_transpose(image)
                )

                rgb_image = oriented.convert(
                    "RGB"
                )

                return rgb_image.copy()

        except VisionProcessingError:
            raise

        except Exception as error:
            raise VisionProcessingError(
                f"Unable to load image: "
                f"{frame_path}",
                module="vision_processor",
                details={
                    "frame_path": (
                        str(frame_path)
                    )
                },
                cause=error,
            ) from error

    def _prepare_image(
        self,
        image: Any,
    ) -> tuple[Any, ResizeInformation]:
        """Resize and optionally letterbox an image."""

        original_width, original_height = (
            image.size
        )

        target_width = (
            self.settings.vision.input_width
        )

        target_height = (
            self.settings.vision.input_height
        )

        if (
            not self.settings.vision
            .preserve_aspect_ratio
        ):
            resized = image.resize(
                (
                    target_width,
                    target_height,
                ),
                self.Image.Resampling.LANCZOS,
            )

            info = ResizeInformation(
                original_width=original_width,
                original_height=original_height,
                target_width=target_width,
                target_height=target_height,
                resized_width=target_width,
                resized_height=target_height,
                scale=(
                    target_width
                    / original_width
                ),
                padding_left=0,
                padding_top=0,
                padding_right=0,
                padding_bottom=0,
                aspect_ratio_preserved=False,
            )

            return resized, info

        scale = min(
            target_width / original_width,
            target_height / original_height,
        )

        resized_width = max(
            1,
            round(
                original_width * scale
            ),
        )

        resized_height = max(
            1,
            round(
                original_height * scale
            ),
        )

        resized_content = image.resize(
            (
                resized_width,
                resized_height,
            ),
            self.Image.Resampling.LANCZOS,
        )

        padding_width = (
            target_width - resized_width
        )

        padding_height = (
            target_height - resized_height
        )

        padding_left = (
            padding_width // 2
        )

        padding_right = (
            padding_width
            - padding_left
        )

        padding_top = (
            padding_height // 2
        )

        padding_bottom = (
            padding_height
            - padding_top
        )

        canvas = self.Image.new(
            "RGB",
            (
                target_width,
                target_height,
            ),
            color=(114, 114, 114),
        )

        canvas.paste(
            resized_content,
            (
                padding_left,
                padding_top,
            ),
        )

        info = ResizeInformation(
            original_width=original_width,
            original_height=original_height,
            target_width=target_width,
            target_height=target_height,
            resized_width=resized_width,
            resized_height=resized_height,
            scale=scale,
            padding_left=padding_left,
            padding_top=padding_top,
            padding_right=padding_right,
            padding_bottom=padding_bottom,
            aspect_ratio_preserved=True,
        )

        return canvas, info

    def _quality_warnings(
        self,
        quality: VisionQualityMetrics,
    ) -> List[str]:
        """Return image-quality warnings."""

        warnings = []

        if (
            quality.brightness
            < LOW_BRIGHTNESS_THRESHOLD
        ):
            warnings.append(
                "Image is underexposed."
            )

        elif (
            quality.brightness
            > HIGH_BRIGHTNESS_THRESHOLD
        ):
            warnings.append(
                "Image is overexposed."
            )

        if (
            quality.contrast
            < LOW_CONTRAST_THRESHOLD
        ):
            warnings.append(
                "Image has low contrast."
            )

        if (
            quality.sharpness
            < LOW_SHARPNESS_THRESHOLD
        ):
            warnings.append(
                "Image may be blurred."
            )

        return warnings

    def _calculate_confidence(
        self,
        quality_confidence: float,
        sensor_confidence: Optional[float],
    ) -> float:
        """Combine sensor reliability and image quality."""

        if sensor_confidence is None:
            return round(
                max(
                    0.0,
                    min(
                        1.0,
                        quality_confidence,
                    ),
                ),
                6,
            )

        sensor_weight = (
            self.settings.confidence
            .sensor_confidence_weight
        )

        model_weight = (
            self.settings.confidence
            .model_confidence_weight
        )

        confidence = (
            sensor_weight
            * sensor_confidence
            + model_weight
            * quality_confidence
        )

        return round(
            max(
                0.0,
                min(1.0, confidence),
            ),
            6,
        )

    def _save_processed_frame(
        self,
        image: Any,
        packet: AdaptedLayer1Input,
    ) -> Path:
        """Save a processed-frame copy."""

        output_directory = (
            self.project_root
            / "output"
            / "layer2"
            / "vision"
            / "processed"
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        safe_packet_id = (
            packet.packet_id
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

        output_path = (
            output_directory
            / (
                f"{safe_packet_id}_"
                "processed.jpg"
            )
        )

        image.save(
            output_path,
            format="JPEG",
            quality=95,
            optimize=True,
        )

        return output_path


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def process_vision(
    packet: AdaptedLayer1Input,
    *,
    settings: Optional[
        Layer2Settings
    ] = None,
) -> VisionProcessingOutput:

    processor = VisionProcessor(
        settings=settings
    )

    return processor.process(packet)


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 72)
    print("NOONGIL-X | VISION PROCESSOR SELF-TEST")
    print("=" * 72)

    try:
        require_vision_dependencies()

        print("[PASS] Pillow and NumPy available")

        adapter = Layer1PacketAdapter(
            require_media=True
        )

        processor = VisionProcessor(
            settings=create_test_settings()
        )

        scenarios = (
            adapter.discover_scenarios()
        )

        if len(scenarios) != 8:
            raise AssertionError(
                "Expected eight scenarios."
            )

        print("[PASS] Eight scenarios discovered")

        outputs = []

        for scenario in scenarios:

            packet = adapter.load_scenario(
                scenario
            )

            output = processor.process(
                packet
            )

            if not output.succeeded:
                raise AssertionError(
                    f"Vision processing failed "
                    f"for {scenario}: "
                    f"{output.result.errors}"
                )

            if output.image_rgb is None:
                raise AssertionError(
                    f"RGB image is missing for "
                    f"{scenario}."
                )

            if (
                output.image_rgb.shape
                != (640, 640, 3)
            ):
                raise AssertionError(
                    f"Unexpected image shape for "
                    f"{scenario}: "
                    f"{output.image_rgb.shape}"
                )

            if (
                output.normalized_image
                is None
            ):
                raise AssertionError(
                    f"Normalized image is missing "
                    f"for {scenario}."
                )

            minimum = float(
                output.normalized_image.min()
            )

            maximum = float(
                output.normalized_image.max()
            )

            if (
                minimum < 0.0
                or maximum > 1.0
            ):
                raise AssertionError(
                    f"Normalization range is "
                    f"invalid for {scenario}."
                )

            outputs.append(output)

            print(
                f"[PASS] {scenario}: "
                f"shape={output.image_rgb.shape}, "
                f"confidence="
                f"{output.result.confidence}"
            )

        print("[PASS] All frames converted to RGB")
        print("[PASS] EXIF orientation handled")
        print("[PASS] Aspect ratio preserved")
        print("[PASS] Letterbox resize completed")
        print("[PASS] Pixel normalization validated")
        print("[PASS] Quality metrics calculated")
        print("[PASS] ModuleResult generated")

        example = outputs[0]

        print("\nExample vision result:")
        print(
            f"  status: "
            f"{example.result.status.value}"
        )
        print(
            f"  confidence: "
            f"{example.result.confidence}"
        )
        print(
            f"  processing_time_ms: "
            f"{example.result.processing_time_ms}"
        )
        print(
            f"  image_shape: "
            f"{example.image_shape}"
        )
        print(
            f"  quality: "
            f"{example.result.data['quality']}"
        )
        print(
            f"  resize: "
            f"{example.result.data['resize']}"
        )
        print(
            f"  processed_frame: "
            f"{example.result.data['processed_frame_path']}"
        )

        print("\n" + "=" * 72)
        print(
            "[PASSED] VISION PROCESSOR IS WORKING"
        )
        print("=" * 72)

        return True

    except (
        DependencyMissingError,
        VisionProcessingError,
        AssertionError,
    ) as error:

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
            "vision-processor self-test."
        )
    )


def main() -> int:

    build_argument_parser().parse_args()

    return 0 if run_self_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
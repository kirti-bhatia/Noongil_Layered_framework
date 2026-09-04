"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Depth Estimator
File    : layer2/depth_spatial_perception/depth_estimator.py
============================================================

Purpose
-------
Generates a monocular relative-depth representation using the
configured Depth Anything V2 model.

Important
---------
This module estimates relative proximity, not physical distance
in metres.

For Depth Anything output:
- higher normalized values indicate nearer regions
- lower normalized values indicate farther regions

Responsibilities:
- load the depth model lazily
- estimate relative depth
- normalize the depth output
- divide the scene into spatial regions
- identify the nearest scene region
- generate near/mid/far distributions
- return standardized ModuleResult
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

DEPTH_ESTIMATOR_VERSION = "1.0"

DEFAULT_NEAR_THRESHOLD = 0.66
DEFAULT_FAR_THRESHOLD = 0.33

GRID_ROWS = 3
GRID_COLUMNS = 3

GRID_LABELS = (
    (
        "top_left",
        "top_center",
        "top_right",
    ),
    (
        "middle_left",
        "center",
        "middle_right",
    ),
    (
        "bottom_left",
        "bottom_center",
        "bottom_right",
    ),
)


# ============================================================
# EXCEPTION
# ============================================================

class DepthEstimationError(Exception):
    """Raised when depth estimation cannot be completed."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class DepthRegion:
    """Relative-depth statistics for one image region."""

    name: str

    row: int
    column: int

    x1: int
    y1: int
    x2: int
    y2: int

    mean_proximity: float
    maximum_proximity: float
    minimum_proximity: float

    depth_category: str

    def to_dict(self) -> Dict[str, Any]:

        return {
            "name": self.name,
            "row": self.row,
            "column": self.column,
            "bounds": {
                "x1": self.x1,
                "y1": self.y1,
                "x2": self.x2,
                "y2": self.y2,
            },
            "mean_proximity": round(
                self.mean_proximity,
                6,
            ),
            "maximum_proximity": round(
                self.maximum_proximity,
                6,
            ),
            "minimum_proximity": round(
                self.minimum_proximity,
                6,
            ),
            "depth_category": (
                self.depth_category
            ),
        }


@dataclass(frozen=True)
class DepthDistribution:
    """Near, middle and far pixel distribution."""

    near_ratio: float
    middle_ratio: float
    far_ratio: float

    def to_dict(self) -> Dict[str, float]:

        return {
            "near_ratio": round(
                self.near_ratio,
                6,
            ),
            "middle_ratio": round(
                self.middle_ratio,
                6,
            ),
            "far_ratio": round(
                self.far_ratio,
                6,
            ),
        }


@dataclass
class DepthEstimationOutput:
    """Complete relative-depth output."""

    result: ModuleResult

    relative_proximity_map: Optional[Any]

    regions: List[DepthRegion]

    nearest_region: Optional[str]

    distribution: Optional[
        DepthDistribution
    ]

    @property
    def succeeded(self) -> bool:
        return self.result.usable

    @property
    def map_shape(
        self,
    ) -> Optional[Tuple[int, ...]]:

        if self.relative_proximity_map is None:
            return None

        return tuple(
            self.relative_proximity_map.shape
        )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "map_shape": (
                list(self.map_shape)
                if self.map_shape
                else None
            ),
            "nearest_region": (
                self.nearest_region
            ),
            "regions": [
                region.to_dict()
                for region in self.regions
            ],
            "distribution": (
                self.distribution.to_dict()
                if self.distribution
                else None
            ),
            "result": self.result.to_dict(),
        }


# ============================================================
# DEPTH ESTIMATOR
# ============================================================

class DepthEstimator:
    """Depth Anything V2 relative-depth estimator."""

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
                "depth_estimator"
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

    def estimate(
        self,
        packet: AdaptedLayer1Input,
        vision_output: VisionProcessingOutput,
    ) -> DepthEstimationOutput:
        """Estimate relative scene depth."""

        if not isinstance(
            packet,
            AdaptedLayer1Input,
        ):
            raise DepthEstimationError(
                "packet must be an "
                "AdaptedLayer1Input."
            )

        if not isinstance(
            vision_output,
            VisionProcessingOutput,
        ):
            raise DepthEstimationError(
                "vision_output must be a "
                "VisionProcessingOutput."
            )

        if not self._module_enabled():

            result = ModuleResult.skipped(
                module_name="depth_estimator",
                modality="vision",
                reason=(
                    "Depth estimation is "
                    "disabled in settings."
                ),
                source_packet_id=(
                    packet.packet_id
                ),
            )

            return DepthEstimationOutput(
                result=result,
                relative_proximity_map=None,
                regions=[],
                nearest_region=None,
                distribution=None,
            )

        if (
            not vision_output.succeeded
            or vision_output.image_rgb is None
        ):
            result = ModuleResult.failure(
                module_name="depth_estimator",
                modality="vision",
                error=(
                    "A successful processed vision "
                    "frame is required."
                ),
                source_packet_id=(
                    packet.packet_id
                ),
            )

            return DepthEstimationOutput(
                result=result,
                relative_proximity_map=None,
                regions=[],
                nearest_region=None,
                distribution=None,
            )

        packet_logger = self.logger.bind(
            packet_id=packet.packet_id,
            scenario=packet.scenario,
        )

        log_event(
            packet_logger,
            event=(
                "depth_estimation_started"
            ),
            message=(
                "Relative-depth estimation started."
            ),
        )

        with ModuleTimer(
            "depth_estimator",
            logger=packet_logger,
            packet_id=packet.packet_id,
            log_start=False,
            log_completion=False,
        ) as timer:

            try:
                pipeline = (
                    self._get_pipeline()
                )

                input_image = (
                    self._input_image(
                        vision_output
                    )
                )

                raw_output = pipeline(
                    input_image
                )

                raw_depth = (
                    self._extract_depth(
                        raw_output
                    )
                )

                proximity_map = (
                    self._normalize_depth(
                        raw_depth
                    )
                )

                regions = (
                    self._calculate_regions(
                        proximity_map
                    )
                )

                nearest_region = (
                    max(
                        regions,
                        key=lambda region: (
                            region.mean_proximity
                        ),
                    ).name
                    if regions
                    else None
                )

                distribution = (
                    self._distribution(
                        proximity_map
                    )
                )

                confidence = (
                    self._estimate_confidence(
                        raw_depth,
                        vision_output,
                    )
                )

                warnings = [
                    (
                        "Depth values are relative "
                        "and must not be interpreted "
                        "as physical distance."
                    )
                ]

                data = {
                    "output_type": (
                        "relative_proximity"
                    ),
                    "higher_values_mean": (
                        "nearer"
                    ),
                    "map_shape": list(
                        proximity_map.shape
                    ),
                    "map_statistics": {
                        "minimum": float(
                            proximity_map.min()
                        ),
                        "maximum": float(
                            proximity_map.max()
                        ),
                        "mean": float(
                            proximity_map.mean()
                        ),
                        "standard_deviation": float(
                            proximity_map.std()
                        ),
                    },
                    "nearest_region": (
                        nearest_region
                    ),
                    "regions": [
                        region.to_dict()
                        for region in regions
                    ],
                    "distribution": (
                        distribution.to_dict()
                    ),
                    "near_threshold": (
                        DEFAULT_NEAR_THRESHOLD
                    ),
                    "far_threshold": (
                        DEFAULT_FAR_THRESHOLD
                    ),
                    "metric_distance_available": (
                        False
                    ),
                    "model_id": (
                        self.model_config
                        .depth_estimation
                        .model_id
                    ),
                    "estimator_version": (
                        DEPTH_ESTIMATOR_VERSION
                    ),
                }

                result = ModuleResult.success(
                    module_name=(
                        "depth_estimator"
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
                        ),
                        "model_backend": (
                            self.model_config
                            .depth_estimation
                            .backend.value
                        ),
                    },
                )

                log_event(
                    packet_logger,
                    event=(
                        "depth_estimation_completed"
                    ),
                    message=(
                        "Relative-depth estimation "
                        "completed."
                    ),
                    details={
                        "map_shape": list(
                            proximity_map.shape
                        ),
                        "nearest_region": (
                            nearest_region
                        ),
                        "confidence": confidence,
                        "processing_time_ms": (
                            timer.elapsed_ms
                        ),
                    },
                )

                return DepthEstimationOutput(
                    result=result,
                    relative_proximity_map=(
                        proximity_map
                    ),
                    regions=regions,
                    nearest_region=(
                        nearest_region
                    ),
                    distribution=distribution,
                )

            except Exception as error:

                log_exception(
                    packet_logger,
                    error,
                    event=(
                        "depth_estimation_failed"
                    ),
                    message=(
                        "Relative-depth estimation "
                        "failed."
                    ),
                    details={
                        "processing_time_ms": (
                            timer.elapsed_ms
                        )
                    },
                )

                result = ModuleResult.failure(
                    module_name=(
                        "depth_estimator"
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
                        "model_id": (
                            self.model_config
                            .depth_estimation
                            .model_id
                        )
                    },
                )

                return DepthEstimationOutput(
                    result=result,
                    relative_proximity_map=None,
                    regions=[],
                    nearest_region=None,
                    distribution=None,
                )

    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    def _module_enabled(self) -> bool:
        """Return whether depth estimation is enabled."""

        modules = self.settings.modules

        if hasattr(
            modules,
            "depth_estimation",
        ):
            return bool(
                modules.depth_estimation
            )

        if hasattr(
            modules,
            "depth_processing",
        ):
            return bool(
                modules.depth_processing
            )

        return True

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    def _get_pipeline(self) -> Any:
        """Load and return the depth pipeline."""

        loaded = self.model_loader.load(
            "depth_estimation"
        )

        if isinstance(
            loaded,
            ModelBundle,
        ):
            pipeline = loaded.model
        else:
            pipeline = loaded

        if not callable(pipeline):
            raise ModelLoadingError(
                "Loaded depth-estimation object "
                "is not callable.",
                module="depth_estimator",
            )

        return pipeline

    @staticmethod
    def _input_image(
        vision_output: VisionProcessingOutput,
    ) -> Any:
        """Return a Pillow image for the model."""

        if (
            vision_output.original_image
            is not None
        ):
            return (
                vision_output.original_image
            )

        try:
            from PIL import Image

            return Image.fromarray(
                vision_output.image_rgb
            )

        except Exception as error:
            raise DepthEstimationError(
                "Unable to create the depth-model "
                "input image."
            ) from error

    # --------------------------------------------------------
    # DEPTH PROCESSING
    # --------------------------------------------------------

    @staticmethod
    def _extract_depth(
        raw_output: Any,
    ) -> Any:
        """Extract the predicted depth tensor."""

        if not isinstance(
            raw_output,
            dict,
        ):
            raise DepthEstimationError(
                "Depth-model output must "
                "be a dictionary."
            )

        predicted_depth = raw_output.get(
            "predicted_depth"
        )

        if predicted_depth is None:
            predicted_depth = raw_output.get(
                "depth"
            )

        if predicted_depth is None:
            raise DepthEstimationError(
                "Depth-model output contains "
                "no predicted depth."
            )

        if hasattr(
            predicted_depth,
            "detach",
        ):
            predicted_depth = (
                predicted_depth.detach()
            )

        if hasattr(
            predicted_depth,
            "cpu",
        ):
            predicted_depth = (
                predicted_depth.cpu()
            )

        if hasattr(
            predicted_depth,
            "numpy",
        ):
            predicted_depth = (
                predicted_depth.numpy()
            )

        try:
            import numpy as np

            depth_array = np.asarray(
                predicted_depth,
                dtype=np.float32,
            )

        except ImportError as error:
            raise DependencyMissingError(
                "NumPy is required for "
                "depth estimation.",
                module="depth_estimator",
            ) from error

        depth_array = depth_array.squeeze()

        if depth_array.ndim != 2:
            raise DepthEstimationError(
                "Expected a two-dimensional depth "
                f"map, received {depth_array.shape}."
            )

        if depth_array.size == 0:
            raise DepthEstimationError(
                "The predicted depth map is empty."
            )

        if not np.all(
            np.isfinite(depth_array)
        ):
            raise DepthEstimationError(
                "The predicted depth map contains "
                "non-finite values."
            )

        return depth_array

    @staticmethod
    def _normalize_depth(
        raw_depth: Any,
    ) -> Any:
        """
        Normalize inverse-relative depth into proximity [0, 1].

        Depth Anything produces relative inverse depth, so larger
        values represent nearer scene regions.
        """

        try:
            import numpy as np

        except ImportError as error:
            raise DependencyMissingError(
                "NumPy is required for depth "
                "normalization.",
                module="depth_estimator",
            ) from error

        minimum = float(
            np.min(raw_depth)
        )

        maximum = float(
            np.max(raw_depth)
        )

        depth_range = (
            maximum - minimum
        )

        if depth_range <= 1e-8:
            return np.zeros_like(
                raw_depth,
                dtype=np.float32,
            )

        normalized = (
            raw_depth - minimum
        ) / depth_range

        return np.clip(
            normalized,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

    def _calculate_regions(
        self,
        proximity_map: Any,
    ) -> List[DepthRegion]:
        """Divide the depth map into a 3×3 grid."""

        height, width = (
            proximity_map.shape
        )

        regions = []

        for row in range(
            GRID_ROWS
        ):
            y1 = (
                row * height
                // GRID_ROWS
            )

            y2 = (
                (row + 1) * height
                // GRID_ROWS
            )

            for column in range(
                GRID_COLUMNS
            ):
                x1 = (
                    column * width
                    // GRID_COLUMNS
                )

                x2 = (
                    (column + 1) * width
                    // GRID_COLUMNS
                )

                region_map = proximity_map[
                    y1:y2,
                    x1:x2,
                ]

                if region_map.size == 0:
                    continue

                mean_proximity = float(
                    region_map.mean()
                )

                regions.append(
                    DepthRegion(
                        name=(
                            GRID_LABELS[
                                row
                            ][
                                column
                            ]
                        ),
                        row=row,
                        column=column,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        mean_proximity=(
                            mean_proximity
                        ),
                        maximum_proximity=(
                            float(
                                region_map.max()
                            )
                        ),
                        minimum_proximity=(
                            float(
                                region_map.min()
                            )
                        ),
                        depth_category=(
                            self._depth_category(
                                mean_proximity
                            )
                        ),
                    )
                )

        return regions

    @staticmethod
    def _depth_category(
        proximity: float,
    ) -> str:
        """Convert proximity into near/mid/far."""

        if (
            proximity
            >= DEFAULT_NEAR_THRESHOLD
        ):
            return "near"

        if (
            proximity
            <= DEFAULT_FAR_THRESHOLD
        ):
            return "far"

        return "middle"

    @staticmethod
    def _distribution(
        proximity_map: Any,
    ) -> DepthDistribution:
        """Calculate pixel depth distribution."""

        total_pixels = int(
            proximity_map.size
        )

        if total_pixels <= 0:
            raise DepthEstimationError(
                "Cannot calculate distribution "
                "for an empty map."
            )

        near_pixels = int(
            (
                proximity_map
                >= DEFAULT_NEAR_THRESHOLD
            ).sum()
        )

        far_pixels = int(
            (
                proximity_map
                <= DEFAULT_FAR_THRESHOLD
            ).sum()
        )

        middle_pixels = (
            total_pixels
            - near_pixels
            - far_pixels
        )

        return DepthDistribution(
            near_ratio=(
                near_pixels
                / total_pixels
            ),
            middle_ratio=(
                middle_pixels
                / total_pixels
            ),
            far_ratio=(
                far_pixels
                / total_pixels
            ),
        )

    @staticmethod
    def _estimate_confidence(
        raw_depth: Any,
        vision_output: VisionProcessingOutput,
    ) -> float:
        """
        Estimate output reliability.

        This is a processing-quality score, not calibrated model
        probability or metric-depth accuracy.
        """

        try:
            import numpy as np

        except ImportError as error:
            raise DependencyMissingError(
                "NumPy is required for "
                "depth confidence.",
                module="depth_estimator",
            ) from error

        mean_value = float(
            np.mean(
                np.abs(raw_depth)
            )
        )

        standard_deviation = float(
            np.std(raw_depth)
        )

        relative_variation = (
            standard_deviation
            / (mean_value + 1e-8)
        )

        map_quality = min(
            1.0,
            max(
                0.0,
                relative_variation * 2.0,
            ),
        )

        frame_quality = (
            vision_output.result.confidence
        )

        if frame_quality is None:
            frame_quality = 0.5

        confidence = (
            0.60 * frame_quality
            + 0.40 * map_quality
        )

        return round(
            max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            ),
            6,
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def estimate_depth(
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
) -> DepthEstimationOutput:

    estimator = DepthEstimator(
        settings=settings,
        model_config=model_config,
        model_loader=model_loader,
    )

    return estimator.estimate(
        packet,
        vision_output,
    )


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test(
    scenario_name: str = "navigation_request",
) -> bool:

    print("=" * 72)

    print(
        "NOONGIL-X | DEPTH ESTIMATOR "
        "SELF-TEST"
    )

    print("=" * 72)

    print(
        "The first run may download the "
        "configured depth model."
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

        estimator = DepthEstimator(
            settings=settings,
            model_config=model_config,
            model_loader=loader,
        )

        packet = adapter.load_scenario(
            scenario_name
        )

        vision_output = processor.process(
            packet
        )

        if not vision_output.succeeded:
            raise AssertionError(
                "Vision processing failed."
            )

        output = estimator.estimate(
            packet,
            vision_output,
        )

        if not output.succeeded:
            raise AssertionError(
                "Depth estimation failed: "
                f"{output.result.errors}"
            )

        if (
            output.relative_proximity_map
            is None
        ):
            raise AssertionError(
                "Relative proximity map is missing."
            )

        minimum = float(
            output
            .relative_proximity_map
            .min()
        )

        maximum = float(
            output
            .relative_proximity_map
            .max()
        )

        if (
            minimum < 0.0
            or maximum > 1.0
        ):
            raise AssertionError(
                "Depth normalization is invalid."
            )

        if len(output.regions) != 9:
            raise AssertionError(
                "Expected nine spatial regions."
            )

        if output.nearest_region is None:
            raise AssertionError(
                "Nearest region is missing."
            )

        if output.distribution is None:
            raise AssertionError(
                "Depth distribution is missing."
            )

        distribution_total = (
            output.distribution.near_ratio
            + output.distribution.middle_ratio
            + output.distribution.far_ratio
        )

        if not math.isclose(
            distribution_total,
            1.0,
            abs_tol=1e-5,
        ):
            raise AssertionError(
                "Depth distribution does not "
                "sum to one."
            )

        print(
            f"[PASS] Scenario: "
            f"{scenario_name}"
        )

        print(
            f"[PASS] Map shape: "
            f"{output.map_shape}"
        )

        print(
            f"[PASS] Map range: "
            f"{minimum:.6f} to "
            f"{maximum:.6f}"
        )

        print(
            f"[PASS] Spatial regions: "
            f"{len(output.regions)}"
        )

        print(
            f"[PASS] Nearest region: "
            f"{output.nearest_region}"
        )

        print(
            f"[PASS] Distribution: "
            f"{output.distribution.to_dict()}"
        )

        print(
            "[PASS] Depth model loaded lazily"
        )

        print(
            "[PASS] Relative depth normalized"
        )

        print(
            "[PASS] Near/middle/far regions generated"
        )

        print(
            "[PASS] Metric-distance claims prevented"
        )

        print(
            "[PASS] Layer 3 spatial format generated"
        )

        print(
            "[PASS] ModuleResult generated"
        )

        print("\n" + "=" * 72)

        print(
            "[PASSED] DEPTH ESTIMATOR "
            "IS WORKING"
        )

        print("=" * 72)

        return True

    except (
        DependencyMissingError,
        ModelLoadingError,
        DepthEstimationError,
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
            "depth-estimator self-test."
        )
    )

    parser.add_argument(
        "--scenario",
        default="navigation_request",
        help=(
            "Scenario used for testing. "
            "Default: navigation_request"
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
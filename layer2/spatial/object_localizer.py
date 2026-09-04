"""Associate detected objects with direction and per-object depth.

The current Layer 2 depth estimator produces *relative proximity*, not metres.
Accordingly, this module always computes image-relative direction, computes a
horizontal angle when camera calibration is available, and emits ``distance_m``
only when the caller explicitly supplies a genuine metric depth map.

Compatible with Python 3.10+ and the existing NOONGIL-X/SHIVI Layer 2 schemas.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from layer2.schemas.module_result import ModuleResult


OBJECT_LOCALIZER_VERSION = "1.0"


class ObjectLocalizationError(Exception):
    """Raised when object localization input is invalid."""


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole-camera intrinsics for one calibrated image resolution."""

    camera_id: str
    image_width: int
    image_height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_coefficients: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        numeric = (self.fx, self.fy, self.cx, self.cy)
        if self.image_width <= 0 or self.image_height <= 0:
            raise ObjectLocalizationError(
                "Calibration image dimensions must be positive."
            )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ObjectLocalizationError("Camera intrinsics must be finite.")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ObjectLocalizationError("fx and fy must be positive.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CameraCalibration":
        if not isinstance(payload, Mapping):
            raise ObjectLocalizationError("Calibration must be a JSON object.")
        try:
            coefficients = payload.get("distortion_coefficients", [])
            if not isinstance(coefficients, (list, tuple)):
                raise TypeError("distortion_coefficients must be a list")
            return cls(
                camera_id=str(payload.get("camera_id", "camera")),
                image_width=int(payload["image_width"]),
                image_height=int(payload["image_height"]),
                fx=float(payload["fx"]),
                fy=float(payload["fy"]),
                cx=float(payload["cx"]),
                cy=float(payload["cy"]),
                distortion_coefficients=tuple(float(x) for x in coefficients),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ObjectLocalizationError(
                f"Invalid camera calibration: {error}"
            ) from error

    @classmethod
    def from_json(cls, path: Path | str) -> "CameraCalibration":
        calibration_path = Path(path)
        try:
            payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ObjectLocalizationError(
                f"Unable to read calibration file {calibration_path}: {error}"
            ) from error
        return cls.from_dict(payload)

    def scaled_to(self, width: int, height: int) -> "CameraCalibration":
        """Scale intrinsics when inference resolution differs from calibration."""

        if width <= 0 or height <= 0:
            raise ObjectLocalizationError("Image dimensions must be positive.")
        scale_x = width / self.image_width
        scale_y = height / self.image_height
        return CameraCalibration(
            camera_id=self.camera_id,
            image_width=width,
            image_height=height,
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            distortion_coefficients=self.distortion_coefficients,
        )


@dataclass(frozen=True)
class LocalizedObject:
    """Spatial result associated with one detector object."""

    object_id: str
    label: str
    detection_confidence: float
    bounding_box: Tuple[float, float, float, float]
    direction: str
    horizontal_angle_deg: Optional[float]
    relative_proximity: Optional[float]
    proximity_category: Optional[str]
    distance_m: Optional[float]
    distance_type: str
    spatial_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "object_id": self.object_id,
            "label": self.label,
            "detection_confidence": round(self.detection_confidence, 6),
            "bounding_box": [round(value, 3) for value in self.bounding_box],
            "direction": self.direction,
            "distance_type": self.distance_type,
            "spatial_confidence": round(self.spatial_confidence, 6),
        }
        if self.horizontal_angle_deg is not None:
            payload["horizontal_angle_deg"] = round(
                self.horizontal_angle_deg, 3
            )
        if self.relative_proximity is not None:
            payload["relative_proximity"] = round(self.relative_proximity, 6)
        if self.proximity_category is not None:
            payload["proximity_category"] = self.proximity_category
        if self.distance_m is not None:
            payload["distance_m"] = round(self.distance_m, 3)
        return payload


@dataclass
class ObjectLocalizationOutput:
    """Complete object-localization output."""

    result: ModuleResult
    localized_objects: List[LocalizedObject]

    @property
    def succeeded(self) -> bool:
        return self.result.usable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_count": len(self.localized_objects),
            "objects": [item.to_dict() for item in self.localized_objects],
            "result": self.result.to_dict(),
        }


class ObjectLocalizer:
    """Fuse bounding boxes with calibration and relative/metric depth maps."""

    def __init__(
        self,
        calibration_path: Optional[Path | str] = None,
        *,
        calibration: Optional[CameraCalibration] = None,
        center_dead_zone_deg: float = 10.0,
        center_dead_zone_ratio: float = 0.12,
        minimum_depth_pixels: int = 9,
    ) -> None:
        if calibration is not None and calibration_path is not None:
            raise ObjectLocalizationError(
                "Provide calibration or calibration_path, not both."
            )
        if center_dead_zone_deg <= 0.0 or center_dead_zone_deg >= 90.0:
            raise ObjectLocalizationError(
                "center_dead_zone_deg must be between 0 and 90."
            )
        if not 0.0 < center_dead_zone_ratio < 0.5:
            raise ObjectLocalizationError(
                "center_dead_zone_ratio must be between 0 and 0.5."
            )
        if minimum_depth_pixels < 1:
            raise ObjectLocalizationError("minimum_depth_pixels must be positive.")

        self.calibration = (
            CameraCalibration.from_json(calibration_path)
            if calibration_path is not None
            else calibration
        )
        self.center_dead_zone_deg = float(center_dead_zone_deg)
        self.center_dead_zone_ratio = float(center_dead_zone_ratio)
        self.minimum_depth_pixels = int(minimum_depth_pixels)

    def localize(
        self,
        packet: Any,
        objects: Sequence[Any],
        depth_output: Any,
        vision_output: Any,
        *,
        metric_depth_map: Optional[Any] = None,
    ) -> ObjectLocalizationOutput:
        """Localize all detections.

        ``metric_depth_map`` must contain depth in metres. Never pass the current
        normalized ``relative_proximity_map`` as this argument.
        """

        started = time.perf_counter()
        packet_id = str(getattr(packet, "packet_id", "unknown"))

        try:
            if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
                raise ObjectLocalizationError("objects must be a sequence.")

            width, height = self._image_dimensions(vision_output)
            calibration = (
                self.calibration.scaled_to(width, height)
                if self.calibration is not None
                else None
            )
            relative_map = getattr(depth_output, "relative_proximity_map", None)

            localized = [
                self._localize_one(
                    item,
                    image_width=width,
                    image_height=height,
                    calibration=calibration,
                    relative_map=relative_map,
                    metric_map=metric_depth_map,
                    index=index,
                )
                for index, item in enumerate(objects, start=1)
            ]

            warnings: List[str] = []
            if calibration is None:
                warnings.append(
                    "Camera calibration unavailable; direction is image-relative "
                    "and horizontal angles are omitted."
                )
            if metric_depth_map is None:
                warnings.append(
                    "Metric depth unavailable; distance_m is omitted and relative "
                    "proximity must not be interpreted as metres."
                )

            confidence = (
                sum(item.spatial_confidence for item in localized) / len(localized)
                if localized
                else 0.0
            )
            data = {
                "object_count": len(localized),
                "objects": [item.to_dict() for item in localized],
                "camera_calibrated": calibration is not None,
                "metric_distance_available": metric_depth_map is not None,
                "localizer_version": OBJECT_LOCALIZER_VERSION,
            }
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            factory = ModuleResult.success if localized else ModuleResult.partial
            result = factory(
                module_name="object_localizer",
                modality="spatial",
                data=data,
                confidence=confidence,
                processing_time_ms=elapsed_ms,
                source_packet_id=packet_id,
                warnings=warnings or (["No detected objects to localize."] if not localized else []),
                metadata={
                    "camera_id": calibration.camera_id if calibration else None,
                    "image_width": width,
                    "image_height": height,
                },
            )
            return ObjectLocalizationOutput(result, localized)

        except Exception as error:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            result = ModuleResult.failure(
                module_name="object_localizer",
                modality="spatial",
                error=f"{error.__class__.__name__}: {error}",
                processing_time_ms=elapsed_ms,
                source_packet_id=packet_id,
                metadata={"localizer_version": OBJECT_LOCALIZER_VERSION},
            )
            return ObjectLocalizationOutput(result, [])

    def _localize_one(
        self,
        item: Any,
        *,
        image_width: int,
        image_height: int,
        calibration: Optional[CameraCalibration],
        relative_map: Optional[Any],
        metric_map: Optional[Any],
        index: int,
    ) -> LocalizedObject:
        bbox = self._bounding_box(item)
        center_x = (bbox[0] + bbox[2]) / 2.0

        if calibration is not None:
            angle = math.degrees(
                math.atan2(center_x - calibration.cx, calibration.fx)
            )
            direction = self._direction_from_angle(angle)
        else:
            angle = None
            direction = self._direction_from_ratio(center_x / image_width)

        relative = self._median_in_box(
            relative_map, bbox, image_width, image_height, metric=False
        )
        distance = self._median_in_box(
            metric_map, bbox, image_width, image_height, metric=True
        )
        detection_confidence = self._object_value(item, "confidence", 0.0)
        detection_confidence = self._clamp(float(detection_confidence), 0.0, 1.0)

        direction_quality = 1.0 if calibration is not None else 0.65
        depth_quality = 1.0 if distance is not None else (0.65 if relative is not None else 0.0)
        spatial_confidence = self._clamp(
            0.45 * detection_confidence + 0.25 * direction_quality + 0.30 * depth_quality,
            0.0,
            1.0,
        )

        return LocalizedObject(
            object_id=str(self._object_value(item, "object_id", f"OBJ_{index:03d}")),
            label=str(self._object_value(item, "label", "object")),
            detection_confidence=detection_confidence,
            bounding_box=bbox,
            direction=direction,
            horizontal_angle_deg=angle,
            relative_proximity=relative,
            proximity_category=self._proximity_category(relative),
            distance_m=distance,
            distance_type="metric" if distance is not None else "relative_only",
            spatial_confidence=spatial_confidence,
        )

    def _direction_from_angle(self, angle: float) -> str:
        if angle < -self.center_dead_zone_deg:
            return "left"
        if angle > self.center_dead_zone_deg:
            return "right"
        return "ahead"

    def _direction_from_ratio(self, ratio: float) -> str:
        if ratio < 0.5 - self.center_dead_zone_ratio:
            return "left"
        if ratio > 0.5 + self.center_dead_zone_ratio:
            return "right"
        return "ahead"

    @staticmethod
    def _image_dimensions(vision_output: Any,
                          ) -> Tuple[int, int]:
        """Return the dimensions of the image used by the object detector and depth estimator

        YOLO receives original_image, so bounding-box coordinates
        correspond to the original image dimensions.
     """
        


        original_image = getattr(
            vision_output,
            "original_image",
            None,
         )

         #PIL images provide dimensions 
         # through .sixe:
         # (width, height)
        size = getattr(
            original_image,
            "size",
            None,
         )

        if( isinstance(size, tuple) and len(size) == 2):
            width, height = int(size[0]), int(size[1])
            if width > 0 and height > 0:
                return width, height                                                                 

        #Numpy feedback
        shape = getattr(
            original_image,
            "shape",
            None,
        )

        if shape is not None and len(shape) >= 2:
            height, width = int(shape[0]), int(shape[1])
            if width > 0 and height > 0:
                return width, height

        
        # final fallback to processed images

        image_rgb = getattr(
            vision_output,
            "image_rgb",
            None
        )
        shape = getattr(
            image_rgb,
            "shape",
            None,
        )
        if shape is not None and len(shape) >= 2:
            height, width = int(shape[0]), int(shape[1])
            if width > 0 and height > 0:
                return width, height
                                                                                              
            
        raise ObjectLocalizationError(
        "vision_output does not contain an image "
        "with valid dimensions."
    )
        
        # image = getattr(vision_output, "original_image", None)
        # if image is None:
        #     image = getattr(vision_output, "image_rgb", None)
        # shape = getattr(image, "shape", None)
        # if shape is None or len(shape) < 2:
        #     raise ObjectLocalizationError(
        #         "vision_output must contain an image with a valid shape."
        #     )
        # height, width = int(shape[0]), int(shape[1])
        # if width <= 0 or height <= 0:
        #     raise ObjectLocalizationError("Image dimensions must be positive.")
        # return width, height

    @classmethod
    def _bounding_box(cls, item: Any) -> Tuple[float, float, float, float]:
        value = cls._object_value(item, "bounding_box", None)
        if value is None:
            raise ObjectLocalizationError("Detected object has no bounding_box.")

        if isinstance(value, Mapping):
            coordinates = (value.get("x1"), value.get("y1"), value.get("x2"), value.get("y2"))
        elif all(hasattr(value, name) for name in ("x1", "y1", "x2", "y2")):
            coordinates = (value.x1, value.y1, value.x2, value.y2)
        elif isinstance(value, (list, tuple)) and len(value) == 4:
            coordinates = tuple(value)
        else:
            raise ObjectLocalizationError("bounding_box must contain x1, y1, x2, y2.")

        try:
            x1, y1, x2, y2 = (float(number) for number in coordinates)
        except (TypeError, ValueError) as error:
            raise ObjectLocalizationError("Bounding-box coordinates must be numeric.") from error
        if not all(math.isfinite(number) for number in (x1, y1, x2, y2)):
            raise ObjectLocalizationError("Bounding-box coordinates must be finite.")
        if x2 <= x1 or y2 <= y1:
            raise ObjectLocalizationError("Bounding box must have positive area.")
        return x1, y1, x2, y2

    def _median_in_box(
        self,
        depth_map: Optional[Any],
        bbox: Tuple[float, float, float, float],
        image_width: int,
        image_height: int,
        *,
        metric: bool,
    ) -> Optional[float]:
        if depth_map is None:
            return None
        try:
            import numpy as np
        except ImportError as error:
            raise ObjectLocalizationError("NumPy is required for depth association.") from error

        array = np.asarray(depth_map, dtype=float).squeeze()
        if array.ndim != 2 or array.size == 0:
            raise ObjectLocalizationError("Depth map must be a non-empty 2D array.")
        map_height, map_width = array.shape
        x1 = max(0, min(map_width, int(math.floor(bbox[0] * map_width / image_width))))
        x2 = max(0, min(map_width, int(math.ceil(bbox[2] * map_width / image_width))))
        y1 = max(0, min(map_height, int(math.floor(bbox[1] * map_height / image_height))))
        y2 = max(0, min(map_height, int(math.ceil(bbox[3] * map_height / image_height))))
        crop = array[y1:y2, x1:x2]
        valid = crop[np.isfinite(crop)]
        if metric:
            valid = valid[valid > 0.0]
        else:
            valid = valid[(valid >= 0.0) & (valid <= 1.0)]
        if valid.size < self.minimum_depth_pixels:
            return None
        return float(np.median(valid))

    @staticmethod
    def _proximity_category(value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        if value >= 0.66:
            return "near"
        if value <= 0.33:
            return "far"
        return "middle"

    @staticmethod
    def _object_value(item: Any, name: str, default: Any) -> Any:
        if isinstance(item, Mapping):
            return item.get(name, default)
        return getattr(item, name, default)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))


def localize_objects(
    packet: Any,
    objects: Sequence[Any],
    depth_output: Any,
    vision_output: Any,
    *,
    calibration_path: Optional[Path | str] = None,
    metric_depth_map: Optional[Any] = None,
) -> ObjectLocalizationOutput:
    """Convenience wrapper for one localization call."""

    return ObjectLocalizer(calibration_path).localize(
        packet,
        objects,
        depth_output,
        vision_output,
        metric_depth_map=metric_depth_map,
    )


__all__ = [
    "CameraCalibration",
    "LocalizedObject",
    "ObjectLocalizationError",
    "ObjectLocalizationOutput",
    "ObjectLocalizer",
    "localize_objects",
]

"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Feature Aligner
File    : layer2/multimodal_fusion/feature_aligner.py
============================================================

Purpose
-------
Converts heterogeneous Layer 2 ModuleResult outputs into equal-
dimensional feature vectors required by the Multimodal Fusion
Engine.

The aligned feature space is interpretable rather than falsely
presented as a learned cross-modal embedding.

Feature dimensions
------------------
0   source confidence
1   information presence
2   normalized entity/event count
3   safety relevance
4   human presence
5   vehicle presence
6   readable text presence
7   speech presence
8   environmental sound presence
9   near-region evidence
10  middle-region evidence
11  far-region evidence
12  motion/activity evidence
13  navigation relevance
14  environmental-context evidence
15  source quality

Supported modality groups
-------------------------
vision:
    scene classifier, object detector and object tracker

audio:
    speech recognizer and sound-event detector

text:
    OCR engine and text interpreter

spatial:
    depth estimator and spatial processing

motion:
    activity recognizer and motion processing
============================================================
"""

from __future__ import annotations

import argparse
import math
import re

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from layer2.multimodal_fusion.fusion_engine import (
    ModalityFeature,
    MultimodalFusionEngine,
)

from layer2.schemas.module_result import (
    ModuleResult,
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

FEATURE_ALIGNER_VERSION = "1.0"

ALIGNED_FEATURE_DIMENSION = 16

FEATURE_NAMES = (
    "source_confidence",
    "information_presence",
    "normalized_item_count",
    "safety_relevance",
    "human_presence",
    "vehicle_presence",
    "text_presence",
    "speech_presence",
    "sound_presence",
    "near_region_evidence",
    "middle_region_evidence",
    "far_region_evidence",
    "motion_activity_evidence",
    "navigation_relevance",
    "environment_context",
    "source_quality",
)

SUPPORTED_FUSION_MODALITIES = {
    "vision",
    "audio",
    "text",
    "spatial",
    "motion",
}

MODALITY_MAPPING = {
    "vision": "vision",
    "objects": "vision",

    "audio": "audio",
    "speech": "audio",
    "sound": "audio",

    "text": "text",

    "depth": "spatial",
    "spatial": "spatial",

    "activity": "motion",
    "motion": "motion",
}

HUMAN_KEYWORDS = {
    "person",
    "people",
    "human",
    "pedestrian",
    "child",
    "man",
    "woman",
    "boy",
    "girl",
    "crowd",
}

VEHICLE_KEYWORDS = {
    "car",
    "vehicle",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "train",
    "ambulance",
    "taxi",
}

SAFETY_KEYWORDS = {
    "danger",
    "warning",
    "hazard",
    "emergency",
    "critical",
    "alarm",
    "siren",
    "fire",
    "collision",
    "crash",
    "gunshot",
    "explosion",
    "breaking glass",
    "help call",
    "help_call",
}

NAVIGATION_KEYWORDS = {
    "navigation",
    "navigate",
    "road",
    "street",
    "crossing",
    "traffic",
    "exit",
    "entrance",
    "gate",
    "left",
    "right",
    "straight",
    "route",
    "parking",
    "platform",
    "vehicle",
    "car",
}

MOTION_KEYWORDS = {
    "walking",
    "running",
    "moving",
    "falling",
    "standing",
    "sitting",
    "crossing",
    "approaching",
    "departing",
    "cycling",
    "driving",
}

ENVIRONMENT_KEYWORDS = {
    "park",
    "classroom",
    "shopping mall",
    "shopping_mall",
    "cafe",
    "home",
    "street",
    "road",
    "hospital",
    "office",
}


# ============================================================
# EXCEPTION
# ============================================================

class FeatureAlignmentError(Exception):
    """Raised when feature alignment cannot be completed."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class FeatureAlignmentOutput:
    """Complete feature-alignment output."""

    result: ModuleResult

    features: Dict[
        str,
        ModalityFeature
    ]

    feature_names: Tuple[
        str,
        ...
    ]

    excluded_modules: List[str]

    @property
    def succeeded(self) -> bool:
        return self.result.usable

    @property
    def modality_count(self) -> int:
        return len(self.features)

    @property
    def feature_dimension(self) -> int:
        return len(self.feature_names)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "modality_count": (
                self.modality_count
            ),
            "feature_dimension": (
                self.feature_dimension
            ),
            "feature_names": list(
                self.feature_names
            ),
            "features": {
                modality: feature.to_dict()
                for modality, feature
                in self.features.items()
            },
            "excluded_modules": list(
                self.excluded_modules
            ),
            "result": self.result.to_dict(),
        }


# ============================================================
# FEATURE ALIGNER
# ============================================================

class FeatureAligner:
    """Convert Layer 2 results into aligned vectors."""

    def __init__(
        self,
        *,
        logger: Optional[
            Layer2LoggerAdapter
        ] = None,
    ) -> None:

        self.logger = (
            logger
            or get_logger(
                "feature_aligner"
            )
        )

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def align(
        self,
        module_results: Mapping[
            str,
            ModuleResult,
        ],
        *,
        source_packet_id: Optional[
            str
        ] = None,
    ) -> FeatureAlignmentOutput:
        """Align heterogeneous module outputs."""

        if not isinstance(
            module_results,
            Mapping,
        ):
            raise FeatureAlignmentError(
                "module_results must be "
                "a mapping."
            )

        if not module_results:
            raise FeatureAlignmentError(
                "At least one module result "
                "is required."
            )

        alignment_logger = self.logger

        if source_packet_id:
            alignment_logger = (
                self.logger.bind(
                    packet_id=(
                        source_packet_id
                    )
                )
            )

        log_event(
            alignment_logger,
            event=(
                "feature_alignment_started"
            ),
            message=(
                "Multimodal feature alignment "
                "started."
            ),
            details={
                "modules": list(
                    module_results.keys()
                )
            },
        )

        with ModuleTimer(
            "feature_aligner",
            logger=alignment_logger,
            packet_id=source_packet_id,
            log_start=False,
            log_completion=False,
        ) as timer:

            try:
                grouped_results: Dict[
                    str,
                    List[
                        Tuple[
                            str,
                            ModuleResult,
                        ]
                    ],
                ] = {
                    modality: []
                    for modality
                    in SUPPORTED_FUSION_MODALITIES
                }

                excluded_modules = []

                for module_key, result in (
                    module_results.items()
                ):
                    if not isinstance(
                        module_key,
                        str,
                    ):
                        raise FeatureAlignmentError(
                            "Module-result keys must "
                            "be strings."
                        )

                    if not isinstance(
                        result,
                        ModuleResult,
                    ):
                        raise FeatureAlignmentError(
                            f"Result for "
                            f"{module_key!r} must be "
                            "a ModuleResult."
                        )

                    fusion_modality = (
                        self._fusion_modality(
                            module_key,
                            result,
                        )
                    )

                    if fusion_modality is None:
                        excluded_modules.append(
                            module_key
                        )

                        continue

                    if not result.usable:
                        excluded_modules.append(
                            module_key
                        )

                        continue

                    grouped_results[
                        fusion_modality
                    ].append(
                        (
                            module_key,
                            result,
                        )
                    )

                features = {}

                for modality in (
                    "vision",
                    "audio",
                    "text",
                    "spatial",
                    "motion",
                ):
                    group = grouped_results[
                        modality
                    ]

                    if not group:
                        continue

                    feature = (
                        self._align_group(
                            modality=modality,
                            group=group,
                            source_packet_id=(
                                source_packet_id
                            ),
                        )
                    )

                    features[
                        modality
                    ] = feature

                if not features:
                    raise FeatureAlignmentError(
                        "No usable Layer 2 results "
                        "could be aligned."
                    )

                packet_ids = (
                    self._packet_ids(
                        module_results
                    )
                )

                warnings = []

                if len(packet_ids) > 1:
                    warnings.append(
                        "Source module results "
                        "contain different packet IDs."
                    )

                if excluded_modules:
                    warnings.append(
                        "Unsupported or unusable "
                        "module results were excluded."
                    )

                average_confidence = (
                    sum(
                        feature
                        .result
                        .confidence
                        or 0.0
                        for feature
                        in features.values()
                    )
                    / len(features)
                )

                data = {
                    "modality_count": len(
                        features
                    ),
                    "modalities": list(
                        features.keys()
                    ),
                    "feature_dimension": (
                        ALIGNED_FEATURE_DIMENSION
                    ),
                    "feature_names": list(
                        FEATURE_NAMES
                    ),
                    "features": {
                        modality: (
                            feature.to_dict()
                        )
                        for modality, feature
                        in features.items()
                    },
                    "excluded_modules": (
                        excluded_modules
                    ),
                    "source_packet_ids": (
                        packet_ids
                    ),
                    "representation_type": (
                        "interpretable_aligned_features"
                    ),
                    "learned_embedding": False,
                    "aligner_version": (
                        FEATURE_ALIGNER_VERSION
                    ),
                }

                result = ModuleResult.success(
                    module_name=(
                        "feature_aligner"
                    ),
                    modality="fusion",
                    data=data,
                    confidence=(
                        self._clamp(
                            average_confidence
                        )
                    ),
                    processing_time_ms=(
                        timer.elapsed_ms
                    ),
                    source_packet_id=(
                        source_packet_id
                    ),
                    warnings=warnings,
                    metadata={
                        "feature_dimension": (
                            ALIGNED_FEATURE_DIMENSION
                        ),
                        "representation_type": (
                            "interpretable"
                        ),
                    },
                )

                log_event(
                    alignment_logger,
                    event=(
                        "feature_alignment_completed"
                    ),
                    message=(
                        "Multimodal feature alignment "
                        "completed."
                    ),
                    details={
                        "modalities": list(
                            features.keys()
                        ),
                        "feature_dimension": (
                            ALIGNED_FEATURE_DIMENSION
                        ),
                        "excluded_modules": (
                            excluded_modules
                        ),
                        "processing_time_ms": (
                            timer.elapsed_ms
                        ),
                    },
                )

                return FeatureAlignmentOutput(
                    result=result,
                    features=features,
                    feature_names=FEATURE_NAMES,
                    excluded_modules=(
                        excluded_modules
                    ),
                )

            except Exception as error:

                log_exception(
                    alignment_logger,
                    error,
                    event=(
                        "feature_alignment_failed"
                    ),
                    message=(
                        "Multimodal feature "
                        "alignment failed."
                    ),
                    details={
                        "processing_time_ms": (
                            timer.elapsed_ms
                        )
                    },
                )

                result = ModuleResult.failure(
                    module_name=(
                        "feature_aligner"
                    ),
                    modality="fusion",
                    error=(
                        f"{error.__class__.__name__}: "
                        f"{error}"
                    ),
                    processing_time_ms=(
                        timer.elapsed_ms
                    ),
                    source_packet_id=(
                        source_packet_id
                    ),
                )

                return FeatureAlignmentOutput(
                    result=result,
                    features={},
                    feature_names=FEATURE_NAMES,
                    excluded_modules=[],
                )

    # --------------------------------------------------------
    # MODALITY GROUPING
    # --------------------------------------------------------

    def _fusion_modality(
        self,
        module_key: str,
        result: ModuleResult,
    ) -> Optional[str]:
        """Map one ModuleResult to a fusion modality."""

        result_modality = getattr(
            result,
            "modality",
            "",
        )

        if hasattr(
            result_modality,
            "value",
        ):
            result_modality = (
                result_modality.value
            )

        normalized_modality = (
            self._normalize_name(
                result_modality
            )
        )

        if (
            normalized_modality
            in MODALITY_MAPPING
        ):
            return MODALITY_MAPPING[
                normalized_modality
            ]

        normalized_key = (
            self._normalize_name(
                module_key
            )
        )

        module_name = self._normalize_name(
            getattr(
                result,
                "module_name",
                "",
            )
        )

        combined_name = (
            f"{normalized_key} "
            f"{module_name}"
        )

        if any(
            keyword in combined_name
            for keyword in {
                "scene",
                "object",
                "tracker",
                "vision",
            }
        ):
            return "vision"

        if any(
            keyword in combined_name
            for keyword in {
                "speech",
                "sound",
                "audio",
            }
        ):
            return "audio"

        if any(
            keyword in combined_name
            for keyword in {
                "ocr",
                "text",
            }
        ):
            return "text"

        if any(
            keyword in combined_name
            for keyword in {
                "depth",
                "spatial",
            }
        ):
            return "spatial"

        if any(
            keyword in combined_name
            for keyword in {
                "activity",
                "motion",
            }
        ):
            return "motion"

        return None

    # --------------------------------------------------------
    # GROUP ALIGNMENT
    # --------------------------------------------------------

    def _align_group(
        self,
        *,
        modality: str,
        group: Sequence[
            Tuple[
                str,
                ModuleResult,
            ]
        ],
        source_packet_id: Optional[str],
    ) -> ModalityFeature:
        """Create one vector for a modality group."""

        individual_vectors = []

        confidences = []

        source_modules = []

        source_packet_ids = []

        for module_key, result in group:

            vector = self._extract_vector(
                modality=modality,
                result=result,
            )

            individual_vectors.append(
                vector
            )

            confidence = (
                self._result_confidence(
                    result
                )
            )

            confidences.append(
                confidence
            )

            source_modules.append(
                getattr(
                    result,
                    "module_name",
                    module_key,
                )
            )

            packet_id = getattr(
                result,
                "source_packet_id",
                None,
            )

            if packet_id is not None:
                source_packet_ids.append(
                    str(packet_id)
                )

        aligned_vector = (
            self._aggregate_vectors(
                individual_vectors
            )
        )

        group_confidence = (
            sum(confidences)
            / len(confidences)
        )

        aligned_vector[0] = (
            group_confidence
        )

        aligned_vector[1] = 1.0

        aggregate_result = (
            ModuleResult.success(
                module_name=(
                    f"aligned_{modality}"
                ),
                modality=modality,
                data={
                    "source_modules": (
                        source_modules
                    ),
                    "source_packet_ids": sorted(
                        set(
                            source_packet_ids
                        )
                    ),
                    "feature_names": list(
                        FEATURE_NAMES
                    ),
                    "feature_vector": [
                        round(
                            value,
                            6,
                        )
                        for value
                        in aligned_vector
                    ],
                    "representation_type": (
                        "interpretable_aligned_features"
                    ),
                },
                confidence=(
                    group_confidence
                ),
                processing_time_ms=0.0,
                source_packet_id=(
                    source_packet_id
                    or (
                        source_packet_ids[0]
                        if source_packet_ids
                        else None
                    )
                ),
                warnings=[],
                metadata={
                    "fusion_modality": (
                        modality
                    ),
                    "source_module_count": (
                        len(group)
                    ),
                },
            )
        )

        return ModalityFeature(
            modality=modality,
            vector=tuple(
                aligned_vector
            ),
            result=aggregate_result,
            metadata={
                "feature_names": list(
                    FEATURE_NAMES
                ),
                "source_modules": (
                    source_modules
                ),
                "representation_type": (
                    "interpretable"
                ),
            },
        )

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    def _extract_vector(
        self,
        *,
        modality: str,
        result: ModuleResult,
    ) -> List[float]:
        """Extract an interpretable vector."""

        data = result.data

        if not isinstance(
            data,
            Mapping,
        ):
            data = {}

        confidence = (
            self._result_confidence(
                result
            )
        )

        strings = self._flatten_strings(
            data
        )

        combined_text = " ".join(
            strings
        ).lower()

        count = self._extract_count(
            data
        )

        safety_relevance = (
            self._keyword_score(
                combined_text,
                SAFETY_KEYWORDS,
            )
        )

        human_presence = (
            self._keyword_score(
                combined_text,
                HUMAN_KEYWORDS,
            )
        )

        vehicle_presence = (
            self._keyword_score(
                combined_text,
                VEHICLE_KEYWORDS,
            )
        )

        navigation_relevance = (
            self._keyword_score(
                combined_text,
                NAVIGATION_KEYWORDS,
            )
        )

        motion_evidence = (
            self._keyword_score(
                combined_text,
                MOTION_KEYWORDS,
            )
        )

        environment_context = (
            self._keyword_score(
                combined_text,
                ENVIRONMENT_KEYWORDS,
            )
        )

        text_presence = 0.0
        speech_presence = 0.0
        sound_presence = 0.0

        near_evidence = 0.0
        middle_evidence = 0.0
        far_evidence = 0.0

        if modality == "text":
            text_presence = (
                1.0
                if combined_text.strip()
                else 0.0
            )

        if modality == "audio":
            speech_presence = (
                1.0
                if (
                    "transcript"
                    in combined_text
                    or "speech"
                    in combined_text
                )
                else 0.0
            )

            sound_presence = (
                1.0
                if (
                    "sound"
                    in combined_text
                    or "event"
                    in combined_text
                )
                else 0.0
            )

        if modality == "motion":
            motion_evidence = max(
                motion_evidence,
                1.0
                if data
                else 0.0,
            )

        if modality == "spatial":
            (
                near_evidence,
                middle_evidence,
                far_evidence,
            ) = self._depth_distribution(
                data
            )

        source_quality = (
            self._source_quality(
                data,
                confidence,
            )
        )

        vector = [
            confidence,
            1.0 if data else 0.0,
            min(
                1.0,
                count / 20.0,
            ),
            safety_relevance,
            human_presence,
            vehicle_presence,
            text_presence,
            speech_presence,
            sound_presence,
            near_evidence,
            middle_evidence,
            far_evidence,
            motion_evidence,
            navigation_relevance,
            environment_context,
            source_quality,
        ]

        if (
            len(vector)
            != ALIGNED_FEATURE_DIMENSION
        ):
            raise FeatureAlignmentError(
                "Internal feature dimension "
                "is incorrect."
            )

        return [
            self._clamp(value)
            for value in vector
        ]

    # --------------------------------------------------------
    # EXTRACTION HELPERS
    # --------------------------------------------------------

    @classmethod
    def _flatten_strings(
        cls,
        value: Any,
    ) -> List[str]:
        """Recursively collect textual values."""

        strings = []

        if isinstance(value, str):

            cleaned = " ".join(
                value.strip().split()
            )

            if cleaned:
                strings.append(
                    cleaned
                )

        elif isinstance(
            value,
            Mapping,
        ):
            for key, item in value.items():

                if isinstance(key, str):
                    strings.append(
                        key.replace(
                            "_",
                            " ",
                        )
                    )

                strings.extend(
                    cls._flatten_strings(
                        item
                    )
                )

        elif isinstance(
            value,
            (list, tuple, set),
        ):
            for item in value:
                strings.extend(
                    cls._flatten_strings(
                        item
                    )
                )

        return strings

    @staticmethod
    def _extract_count(
        data: Mapping[str, Any],
    ) -> float:
        """Extract a relevant item count."""

        count_keys = (
            "object_count",
            "tracked_count",
            "text_count",
            "detection_count",
            "segment_count",
            "event_count",
            "important_count",
        )

        values = []

        for key in count_keys:

            value = data.get(key)

            if isinstance(
                value,
                (int, float),
            ) and not isinstance(
                value,
                bool,
            ):
                values.append(
                    max(
                        0.0,
                        float(value),
                    )
                )

        if values:
            return max(values)

        for collection_key in (
            "objects",
            "tracked_objects",
            "texts",
            "detections",
            "segments",
            "events",
            "candidates",
        ):
            collection = data.get(
                collection_key
            )

            if isinstance(
                collection,
                (list, tuple),
            ):
                values.append(
                    float(
                        len(collection)
                    )
                )

        return (
            max(values)
            if values
            else 0.0
        )

    @staticmethod
    def _keyword_score(
        combined_text: str,
        keywords: Set[str],
    ) -> float:
        """Return normalized keyword evidence."""

        if not combined_text:
            return 0.0

        matches = sum(
            1
            for keyword in keywords
            if keyword in combined_text
        )

        if matches <= 0:
            return 0.0

        return min(
            1.0,
            0.50
            + 0.25
            * (matches - 1),
        )

    @staticmethod
    def _depth_distribution(
        data: Mapping[str, Any],
    ) -> Tuple[
        float,
        float,
        float,
    ]:
        """Extract near/middle/far evidence."""

        distribution = data.get(
            "distribution",
            {},
        )

        if not isinstance(
            distribution,
            Mapping,
        ):
            return (
                0.0,
                0.0,
                0.0,
            )

        near = FeatureAligner._safe_number(
            distribution.get(
                "near_ratio",
                0.0,
            )
        )

        middle = (
            FeatureAligner._safe_number(
                distribution.get(
                    "middle_ratio",
                    distribution.get(
                        "mid_ratio",
                        0.0,
                    ),
                )
            )
        )

        far = FeatureAligner._safe_number(
            distribution.get(
                "far_ratio",
                0.0,
            )
        )

        return (
            FeatureAligner._clamp(
                near
            ),
            FeatureAligner._clamp(
                middle
            ),
            FeatureAligner._clamp(
                far
            ),
        )

    @staticmethod
    def _source_quality(
        data: Mapping[str, Any],
        confidence: float,
    ) -> float:
        """Extract source-quality evidence."""

        quality = data.get(
            "quality"
        )

        if isinstance(
            quality,
            Mapping,
        ):
            value = quality.get(
                "overall_quality"
            )

            if value is not None:
                return FeatureAligner._clamp(
                    FeatureAligner
                    ._safe_number(
                        value
                    )
                )

        frame_quality = data.get(
            "frame_quality_confidence"
        )

        if frame_quality is not None:
            return FeatureAligner._clamp(
                FeatureAligner._safe_number(
                    frame_quality
                )
            )

        return confidence

    # --------------------------------------------------------
    # VECTOR AGGREGATION
    # --------------------------------------------------------

    @staticmethod
    def _aggregate_vectors(
        vectors: Sequence[
            Sequence[float]
        ],
    ) -> List[float]:
        """
        Aggregate module vectors within one modality.

        Maximum aggregation preserves evidence detected by any
        valid module in the same modality.
        """

        if not vectors:
            raise FeatureAlignmentError(
                "No vectors were supplied "
                "for aggregation."
            )

        expected_dimension = len(
            vectors[0]
        )

        if (
            expected_dimension
            != ALIGNED_FEATURE_DIMENSION
        ):
            raise FeatureAlignmentError(
                "Unexpected feature dimension."
            )

        for vector in vectors:

            if (
                len(vector)
                != expected_dimension
            ):
                raise FeatureAlignmentError(
                    "Feature dimensions do not match."
                )

        return [
            max(
                float(
                    vector[index]
                )
                for vector in vectors
            )
            for index in range(
                expected_dimension
            )
        ]

    # --------------------------------------------------------
    # GENERAL HELPERS
    # --------------------------------------------------------

    @staticmethod
    def _result_confidence(
        result: ModuleResult,
    ) -> float:

        confidence = result.confidence

        if confidence is None:
            return 0.0

        return FeatureAligner._clamp(
            FeatureAligner._safe_number(
                confidence
            )
        )

    @staticmethod
    def _safe_number(
        value: Any,
    ) -> float:

        try:
            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if not math.isfinite(value):
            return 0.0

        return value

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )

    @staticmethod
    def _normalize_name(
        value: Any,
    ) -> str:

        if not isinstance(
            value,
            str,
        ):
            return ""

        return (
            value.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    @staticmethod
    def _packet_ids(
        module_results: Mapping[
            str,
            ModuleResult,
        ],
    ) -> List[str]:

        packet_ids = {
            getattr(
                result,
                "source_packet_id",
                None,
            )
            for result
            in module_results.values()
        }

        packet_ids.discard(None)

        return sorted(
            str(packet_id)
            for packet_id
            in packet_ids
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def align_features(
    module_results: Mapping[
        str,
        ModuleResult,
    ],
    *,
    source_packet_id: Optional[
        str
    ] = None,
) -> FeatureAlignmentOutput:

    aligner = FeatureAligner()

    return aligner.align(
        module_results,
        source_packet_id=(
            source_packet_id
        ),
    )


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test() -> bool:

    print("=" * 72)

    print(
        "NOONGIL-X | FEATURE ALIGNER "
        "SELF-TEST"
    )

    print("=" * 72)

    try:
        packet_id = "MSP_TEST_001"

        module_results = {
            "scene_classifier": (
                ModuleResult.success(
                    module_name=(
                        "scene_classifier"
                    ),
                    modality="vision",
                    data={
                        "scene": {
                            "type": "street",
                            "confidence": 0.91,
                        }
                    },
                    confidence=0.91,
                    processing_time_ms=10.0,
                    source_packet_id=packet_id,
                    warnings=[],
                )
            ),
            "object_detector": (
                ModuleResult.success(
                    module_name=(
                        "object_detector"
                    ),
                    modality="objects",
                    data={
                        "object_count": 3,
                        "objects": [
                            {
                                "label": "person"
                            },
                            {
                                "label": "car"
                            },
                            {
                                "label": (
                                    "traffic_light"
                                )
                            },
                        ],
                    },
                    confidence=0.88,
                    processing_time_ms=10.0,
                    source_packet_id=packet_id,
                    warnings=[],
                )
            ),
            "speech_recognizer": (
                ModuleResult.success(
                    module_name=(
                        "speech_recognizer"
                    ),
                    modality="speech",
                    data={
                        "transcript": (
                            "Guide me across "
                            "the street."
                        ),
                        "speech_detected": True,
                        "segment_count": 1,
                    },
                    confidence=0.79,
                    processing_time_ms=10.0,
                    source_packet_id=packet_id,
                    warnings=[],
                )
            ),
            "sound_event_detector": (
                ModuleResult.success(
                    module_name=(
                        "sound_event_detector"
                    ),
                    modality="sound",
                    data={
                        "event_count": 2,
                        "events": [
                            {
                                "normalized_label": (
                                    "traffic_noise"
                                )
                            },
                            {
                                "normalized_label": (
                                    "car_horn"
                                )
                            },
                        ],
                    },
                    confidence=0.82,
                    processing_time_ms=10.0,
                    source_packet_id=packet_id,
                    warnings=[],
                )
            ),
            "text_interpreter": (
                ModuleResult.success(
                    module_name=(
                        "text_interpreter"
                    ),
                    modality="text",
                    data={
                        "text_count": 1,
                        "transcript": "EXIT",
                        "texts": [
                            {
                                "text": "EXIT",
                                "category": (
                                    "navigation"
                                ),
                            }
                        ],
                    },
                    confidence=0.76,
                    processing_time_ms=10.0,
                    source_packet_id=packet_id,
                    warnings=[],
                )
            ),
            "depth_estimator": (
                ModuleResult.success(
                    module_name=(
                        "depth_estimator"
                    ),
                    modality="depth",
                    data={
                        "nearest_region": (
                            "bottom_center"
                        ),
                        "distribution": {
                            "near_ratio": 0.30,
                            "middle_ratio": 0.45,
                            "far_ratio": 0.25,
                        },
                    },
                    confidence=0.83,
                    processing_time_ms=10.0,
                    source_packet_id=packet_id,
                    warnings=[],
                )
            ),
            "activity_recognizer": (
                ModuleResult.partial(
                    module_name=(
                        "activity_recognizer"
                    ),
                    modality="activity",
                    data={
                        "activity": {
                            "type": "walking"
                        },
                        "frames_collected": 16,
                        "frames_required": 16,
                    },
                    confidence=0.68,
                    processing_time_ms=10.0,
                    source_packet_id=packet_id,
                    warnings=[
                        "Activity confidence is low."
                    ],
                )
            ),
        }

        aligner = FeatureAligner()

        alignment_output = aligner.align(
            module_results,
            source_packet_id=packet_id,
        )

        if not alignment_output.succeeded:
            raise AssertionError(
                "Feature alignment failed: "
                f"{alignment_output.result.errors}"
            )

        expected_modalities = {
            "vision",
            "audio",
            "text",
            "spatial",
            "motion",
        }

        if (
            set(
                alignment_output
                .features.keys()
            )
            != expected_modalities
        ):
            raise AssertionError(
                "Expected all five aligned "
                "modalities."
            )

        for modality, feature in (
            alignment_output
            .features.items()
        ):

            if (
                feature.dimension
                != ALIGNED_FEATURE_DIMENSION
            ):
                raise AssertionError(
                    f"Invalid dimension for "
                    f"{modality}."
                )

            if not all(
                0.0 <= value <= 1.0
                for value in feature.vector
            ):
                raise AssertionError(
                    f"Invalid feature range "
                    f"for {modality}."
                )

        fusion_engine = (
            MultimodalFusionEngine(
                apply_attention=True
            )
        )

        fusion_output = fusion_engine.fuse(
            alignment_output.features,
            source_packet_id=packet_id,
        )

        if not fusion_output.succeeded:
            raise AssertionError(
                "Aligned-feature fusion failed: "
                f"{fusion_output.result.errors}"
            )

        if (
            fusion_output.vector_dimension
            != ALIGNED_FEATURE_DIMENSION
        ):
            raise AssertionError(
                "Unexpected fused dimension."
            )

        print(
            f"[PASS] Modalities aligned: "
            f"{list(alignment_output.features.keys())}"
        )

        print(
            f"[PASS] Feature dimension: "
            f"{alignment_output.feature_dimension}"
        )

        for modality, feature in (
            alignment_output
            .features.items()
        ):

            print(
                f"[PASS] {modality}: "
                f"confidence="
                f"{feature.result.confidence:.6f}, "
                f"dimension="
                f"{feature.dimension}"
            )

        print(
            f"[PASS] Fused vector dimension: "
            f"{fusion_output.vector_dimension}"
        )

        print(
            f"[PASS] Fused confidence: "
            f"{fusion_output.result.confidence:.6f}"
        )

        print(
            "[PASS] Actual ModuleResult structures accepted"
        )

        print(
            "[PASS] Related modules grouped by modality"
        )

        print(
            "[PASS] Equal-dimensional vectors generated"
        )

        print(
            "[PASS] Feature values constrained to [0, 1]"
        )

        print(
            "[PASS] Feature aligner integrated with fusion engine"
        )

        print(
            "[PASS] ModuleResult generated"
        )

        print("\n" + "=" * 72)

        print(
            "[PASSED] FEATURE ALIGNER "
            "IS WORKING"
        )

        print("=" * 72)

        return True

    except (
        FeatureAlignmentError,
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
            "feature-aligner self-test."
        )
    )


def main() -> int:

    build_argument_parser().parse_args()

    return (
        0
        if run_self_test()
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
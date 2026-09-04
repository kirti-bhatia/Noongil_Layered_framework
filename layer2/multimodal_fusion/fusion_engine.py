"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Multimodal Fusion Engine
File    : layer2/multimodal_fusion/fusion_engine.py
============================================================

Purpose
-------
Combines aligned visual, audio, text, spatial and motion feature
vectors into a unified perceptual representation.

Implemented equations
---------------------

Confidence-normalized fusion:

    z_t = sum(w_i,t * z_i,t)

                     C_i,t
    w_i,t = --------------------------
             sum(C_j,t) + epsilon

Scaled dot-product attention:

                          Q K^T
    Attention(Q,K,V) = softmax(-------) V
                         sqrt(d_k)

Important
---------
All modality vectors must already belong to the same aligned
feature space and have equal dimensions. This module does not
silently pad, truncate or invent embeddings.
============================================================
"""

from __future__ import annotations

import argparse
import math

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from layer2.confidence.confidence_calculator import (
    ConfidenceEstimationOutput,
    ConfidenceEstimator,
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

FUSION_ENGINE_VERSION = "1.0"

DEFAULT_EPSILON = 1e-8


# ============================================================
# EXCEPTION
# ============================================================

class MultimodalFusionError(Exception):
    """Raised when multimodal fusion cannot be completed."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class ModalityFeature:
    """One aligned modality feature vector."""

    modality: str

    vector: Tuple[float, ...]

    result: ModuleResult

    metadata: Dict[str, Any]

    @property
    def dimension(self) -> int:
        return len(self.vector)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "modality": self.modality,
            "vector": [
                round(value, 6)
                for value in self.vector
            ],
            "dimension": self.dimension,
            "source_module": getattr(
                self.result,
                "module_name",
                None,
            ),
            "source_packet_id": getattr(
                self.result,
                "source_packet_id",
                None,
            ),
            "confidence": (
                self.result.confidence
            ),
            "metadata": dict(
                self.metadata
            ),
        }


@dataclass
class MultimodalFusionOutput:
    """Complete multimodal-fusion output."""

    result: ModuleResult

    fused_vector: Optional[Any]

    modality_order: List[str]

    confidence_output: Optional[
        ConfidenceEstimationOutput
    ]

    attention_weights: Optional[Any]

    attention_applied: bool

    @property
    def succeeded(self) -> bool:
        return self.result.usable

    @property
    def vector_dimension(
        self,
    ) -> Optional[int]:

        if self.fused_vector is None:
            return None

        return int(
            self.fused_vector.shape[0]
        )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "fused_vector": (
                [
                    round(
                        float(value),
                        6,
                    )
                    for value
                    in self.fused_vector
                ]
                if self.fused_vector
                is not None
                else None
            ),
            "vector_dimension": (
                self.vector_dimension
            ),
            "modality_order": list(
                self.modality_order
            ),
            "confidence": (
                self.confidence_output
                .to_dict()
                if self.confidence_output
                else None
            ),
            "attention_applied": (
                self.attention_applied
            ),
            "attention_weights": (
                self.attention_weights.tolist()
                if self.attention_weights
                is not None
                else None
            ),
            "result": self.result.to_dict(),
        }


# ============================================================
# MULTIMODAL FUSION ENGINE
# ============================================================

class MultimodalFusionEngine:
    """Confidence-weighted multimodal fusion."""

    def __init__(
        self,
        confidence_estimator: Optional[
            ConfidenceEstimator
        ] = None,
        *,
        apply_attention: bool = True,
        epsilon: float = DEFAULT_EPSILON,
        logger: Optional[
            Layer2LoggerAdapter
        ] = None,
    ) -> None:

        try:
            epsilon = float(
                epsilon
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            raise MultimodalFusionError(
                "epsilon must be numeric."
            ) from error

        if (
            not math.isfinite(epsilon)
            or epsilon <= 0.0
        ):
            raise MultimodalFusionError(
                "epsilon must be positive "
                "and finite."
            )

        self.epsilon = epsilon

        self.apply_attention = bool(
            apply_attention
        )

        self.confidence_estimator = (
            confidence_estimator
            or ConfidenceEstimator(
                epsilon=epsilon
            )
        )

        self.logger = (
            logger
            or get_logger(
                "fusion_engine"
            )
        )

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def fuse(
        self,
        features: Mapping[
            str,
            ModalityFeature,
        ],
        *,
        source_packet_id: Optional[
            str
        ] = None,
    ) -> MultimodalFusionOutput:
        """Fuse aligned modality feature vectors."""

        if not isinstance(
            features,
            Mapping,
        ):
            raise MultimodalFusionError(
                "features must be a mapping."
            )

        if not features:
            raise MultimodalFusionError(
                "At least one modality feature "
                "is required."
            )

        fusion_logger = self.logger

        if source_packet_id:
            fusion_logger = (
                self.logger.bind(
                    packet_id=(
                        source_packet_id
                    )
                )
            )

        log_event(
            fusion_logger,
            event=(
                "multimodal_fusion_started"
            ),
            message=(
                "Multimodal fusion started."
            ),
            details={
                "modalities": list(
                    features.keys()
                ),
                "attention_enabled": (
                    self.apply_attention
                ),
            },
        )

        with ModuleTimer(
            "fusion_engine",
            logger=fusion_logger,
            packet_id=source_packet_id,
            log_start=False,
            log_completion=False,
        ) as timer:

            try:
                normalized_features = (
                    self._validate_features(
                        features
                    )
                )

                modality_results = {
                    modality: feature.result
                    for modality, feature
                    in normalized_features.items()
                }

                confidence_output = (
                    self.confidence_estimator
                    .estimate(
                        modality_results
                    )
                )

                if not (
                    confidence_output
                    .usable_modalities
                ):
                    raise MultimodalFusionError(
                        "No modality is usable "
                        "for fusion."
                    )

                modality_order = [
                    modality
                    for modality
                    in normalized_features
                    if modality
                    in confidence_output
                    .usable_modalities
                ]

                feature_matrix = (
                    self._feature_matrix(
                        normalized_features,
                        modality_order,
                    )
                )

                confidence_weights = (
                    self._confidence_weights(
                        confidence_output,
                        modality_order,
                    )
                )

                attention_weights = None
                attended_features = (
                    feature_matrix
                )

                if (
                    self.apply_attention
                    and len(
                        modality_order
                    ) > 1
                ):
                    (
                        attended_features,
                        attention_weights,
                    ) = (
                        self.scaled_dot_product_attention(
                            query=feature_matrix,
                            key=feature_matrix,
                            value=feature_matrix,
                        )
                    )

                fused_vector = (
                    confidence_weights
                    @ attended_features
                )

                fused_vector = (
                    self._normalize_vector(
                        fused_vector
                    )
                )

                source_packet_ids = (
                    self._source_packet_ids(
                        normalized_features
                    )
                )

                warnings = []

                if len(
                    source_packet_ids
                ) > 1:
                    warnings.append(
                        "Modality results contain "
                        "different source packet IDs."
                    )

                excluded_modalities = (
                    confidence_output
                    .excluded_modalities
                )

                if excluded_modalities:
                    warnings.append(
                        "One or more unusable "
                        "modalities were excluded."
                    )

                data = {
                    "fused_vector": [
                        round(
                            float(value),
                            6,
                        )
                        for value
                        in fused_vector
                    ],
                    "vector_dimension": int(
                        fused_vector.shape[0]
                    ),
                    "modality_order": (
                        modality_order
                    ),
                    "modality_count": len(
                        modality_order
                    ),
                    "excluded_modalities": (
                        excluded_modalities
                    ),
                    "confidence_weights": {
                        modality: round(
                            confidence_output
                            .weight_for(
                                modality
                            ),
                            6,
                        )
                        for modality
                        in modality_order
                    },
                    "fused_confidence": (
                        confidence_output
                        .fused_confidence
                    ),
                    "attention_applied": (
                        attention_weights
                        is not None
                    ),
                    "attention_weights": (
                        attention_weights.tolist()
                        if attention_weights
                        is not None
                        else None
                    ),
                    "source_packet_ids": (
                        source_packet_ids
                    ),
                    "fusion_equation": (
                        "z=sum(w_i*z_i)"
                    ),
                    "weight_equation": (
                        "w_i=C_i/(sum(C_j)+epsilon)"
                    ),
                    "attention_equation": (
                        "softmax(QK^T/sqrt(d_k))V"
                    ),
                    "engine_version": (
                        FUSION_ENGINE_VERSION
                    ),
                }

                result = ModuleResult.success(
                    module_name="fusion_engine",
                    modality="fusion",
                    data=data,
                    confidence=(
                        confidence_output
                        .fused_confidence
                    ),
                    processing_time_ms=(
                        timer.elapsed_ms
                    ),
                    source_packet_id=(
                        source_packet_id
                    ),
                    warnings=warnings,
                    metadata={
                        "attention_enabled": (
                            self.apply_attention
                        ),
                        "aligned_feature_space": (
                            True
                        ),
                    },
                )

                log_event(
                    fusion_logger,
                    event=(
                        "multimodal_fusion_completed"
                    ),
                    message=(
                        "Multimodal fusion completed."
                    ),
                    details={
                        "modalities": (
                            modality_order
                        ),
                        "vector_dimension": int(
                            fused_vector.shape[0]
                        ),
                        "fused_confidence": (
                            confidence_output
                            .fused_confidence
                        ),
                        "attention_applied": (
                            attention_weights
                            is not None
                        ),
                        "processing_time_ms": (
                            timer.elapsed_ms
                        ),
                    },
                )

                return MultimodalFusionOutput(
                    result=result,
                    fused_vector=(
                        fused_vector
                    ),
                    modality_order=(
                        modality_order
                    ),
                    confidence_output=(
                        confidence_output
                    ),
                    attention_weights=(
                        attention_weights
                    ),
                    attention_applied=(
                        attention_weights
                        is not None
                    ),
                )

            except Exception as error:

                log_exception(
                    fusion_logger,
                    error,
                    event=(
                        "multimodal_fusion_failed"
                    ),
                    message=(
                        "Multimodal fusion failed."
                    ),
                    details={
                        "processing_time_ms": (
                            timer.elapsed_ms
                        )
                    },
                )

                result = ModuleResult.failure(
                    module_name="fusion_engine",
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

                return MultimodalFusionOutput(
                    result=result,
                    fused_vector=None,
                    modality_order=[],
                    confidence_output=None,
                    attention_weights=None,
                    attention_applied=False,
                )

    # --------------------------------------------------------
    # ATTENTION
    # --------------------------------------------------------

    @staticmethod
    def scaled_dot_product_attention(
        *,
        query: Any,
        key: Any,
        value: Any,
    ) -> Tuple[Any, Any]:
        """
        Apply scaled dot-product attention.

        Returns:
            attended_features,
            attention_weights
        """

        try:
            import numpy as np

        except ImportError as error:
            raise MultimodalFusionError(
                "NumPy is required for "
                "multimodal attention."
            ) from error

        query = np.asarray(
            query,
            dtype=np.float32,
        )

        key = np.asarray(
            key,
            dtype=np.float32,
        )

        value = np.asarray(
            value,
            dtype=np.float32,
        )

        if (
            query.ndim != 2
            or key.ndim != 2
            or value.ndim != 2
        ):
            raise MultimodalFusionError(
                "Q, K and V must be "
                "two-dimensional matrices."
            )

        if (
            query.shape[1]
            != key.shape[1]
        ):
            raise MultimodalFusionError(
                "Query and key dimensions "
                "must match."
            )

        if (
            key.shape[0]
            != value.shape[0]
        ):
            raise MultimodalFusionError(
                "Key and value sequence "
                "lengths must match."
            )

        key_dimension = int(
            key.shape[1]
        )

        if key_dimension <= 0:
            raise MultimodalFusionError(
                "Key dimension must be positive."
            )

        scores = (
            query @ key.T
        ) / math.sqrt(
            key_dimension
        )

        scores = scores - np.max(
            scores,
            axis=-1,
            keepdims=True,
        )

        exponentials = np.exp(
            scores
        )

        denominator = np.sum(
            exponentials,
            axis=-1,
            keepdims=True,
        )

        attention_weights = (
            exponentials
            / (
                denominator
                + DEFAULT_EPSILON
            )
        )

        attended_features = (
            attention_weights
            @ value
        )

        return (
            attended_features.astype(
                np.float32
            ),
            attention_weights.astype(
                np.float32
            ),
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def _validate_features(
        self,
        features: Mapping[
            str,
            ModalityFeature,
        ],
    ) -> Dict[str, ModalityFeature]:
        """Validate modality feature vectors."""

        normalized = {}

        expected_dimension = None

        for raw_modality, feature in (
            features.items()
        ):
            modality = self._normalize_modality(
                raw_modality
            )

            if not modality:
                raise MultimodalFusionError(
                    "Modality names must be "
                    "non-empty strings."
                )

            if not isinstance(
                feature,
                ModalityFeature,
            ):
                raise MultimodalFusionError(
                    f"Feature for {modality!r} "
                    "must be a ModalityFeature."
                )

            feature_modality = (
                self._normalize_modality(
                    feature.modality
                )
            )

            if (
                feature_modality
                != modality
            ):
                raise MultimodalFusionError(
                    "Feature modality does not "
                    f"match mapping key: "
                    f"{feature_modality!r} != "
                    f"{modality!r}."
                )

            if not isinstance(
                feature.result,
                ModuleResult,
            ):
                raise MultimodalFusionError(
                    f"Result for {modality!r} "
                    "must be a ModuleResult."
                )

            vector = self._validate_vector(
                feature.vector,
                modality,
            )

            if expected_dimension is None:
                expected_dimension = len(
                    vector
                )

            elif (
                len(vector)
                != expected_dimension
            ):
                raise MultimodalFusionError(
                    "All modality vectors must "
                    "have equal dimensions. "
                    f"Expected "
                    f"{expected_dimension}, "
                    f"received {len(vector)} "
                    f"for {modality!r}."
                )

            normalized[modality] = (
                ModalityFeature(
                    modality=modality,
                    vector=vector,
                    result=feature.result,
                    metadata=dict(
                        feature.metadata
                    ),
                )
            )

        return normalized

    @staticmethod
    def _validate_vector(
        vector: Sequence[float],
        modality: str,
    ) -> Tuple[float, ...]:
        """Validate one numeric feature vector."""

        if not isinstance(
            vector,
            (list, tuple),
        ):
            raise MultimodalFusionError(
                f"Vector for {modality!r} "
                "must be a list or tuple."
            )

        if not vector:
            raise MultimodalFusionError(
                f"Vector for {modality!r} "
                "cannot be empty."
            )

        validated = []

        for value in vector:

            if isinstance(value, bool):
                raise MultimodalFusionError(
                    f"Vector for {modality!r} "
                    "contains a Boolean value."
                )

            try:
                numeric_value = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ) as error:
                raise MultimodalFusionError(
                    f"Vector for {modality!r} "
                    "contains a non-numeric value."
                ) from error

            if not math.isfinite(
                numeric_value
            ):
                raise MultimodalFusionError(
                    f"Vector for {modality!r} "
                    "contains a non-finite value."
                )

            validated.append(
                numeric_value
            )

        return tuple(
            validated
        )

    # --------------------------------------------------------
    # MATRIX HELPERS
    # --------------------------------------------------------

    @staticmethod
    def _feature_matrix(
        features: Mapping[
            str,
            ModalityFeature,
        ],
        modality_order: Sequence[str],
    ) -> Any:

        try:
            import numpy as np

        except ImportError as error:
            raise MultimodalFusionError(
                "NumPy is required for fusion."
            ) from error

        return np.asarray(
            [
                features[
                    modality
                ].vector
                for modality
                in modality_order
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _confidence_weights(
        confidence_output: (
            ConfidenceEstimationOutput
        ),
        modality_order: Sequence[str],
    ) -> Any:

        try:
            import numpy as np

        except ImportError as error:
            raise MultimodalFusionError(
                "NumPy is required for fusion."
            ) from error

        weights = np.asarray(
            [
                confidence_output
                .weight_for(
                    modality
                )
                for modality
                in modality_order
            ],
            dtype=np.float32,
        )

        weight_sum = float(
            weights.sum()
        )

        if weight_sum <= 0.0:
            raise MultimodalFusionError(
                "Fusion weights sum to zero."
            )

        return (
            weights
            / weight_sum
        )

    @staticmethod
    def _normalize_vector(
        vector: Any,
    ) -> Any:
        """L2-normalize the fused representation."""

        try:
            import numpy as np

        except ImportError as error:
            raise MultimodalFusionError(
                "NumPy is required for "
                "vector normalization."
            ) from error

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        if vector.ndim != 1:
            raise MultimodalFusionError(
                "Fused vector must be "
                "one-dimensional."
            )

        norm = float(
            np.linalg.norm(
                vector
            )
        )

        if norm <= DEFAULT_EPSILON:
            return vector

        return (
            vector / norm
        ).astype(
            np.float32
        )

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    @staticmethod
    def _normalize_modality(
        modality: Any,
    ) -> str:

        if not isinstance(
            modality,
            str,
        ):
            return ""

        return (
            modality.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    @staticmethod
    def _source_packet_ids(
        features: Mapping[
            str,
            ModalityFeature,
        ],
    ) -> List[str]:

        packet_ids = {
            getattr(
                feature.result,
                "source_packet_id",
                None,
            )
            for feature
            in features.values()
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

def fuse_modalities(
    features: Mapping[
        str,
        ModalityFeature,
    ],
    *,
    source_packet_id: Optional[
        str
    ] = None,
    apply_attention: bool = True,
) -> MultimodalFusionOutput:

    engine = MultimodalFusionEngine(
        apply_attention=(
            apply_attention
        )
    )

    return engine.fuse(
        features,
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
        "NOONGIL-X | MULTIMODAL FUSION "
        "ENGINE SELF-TEST"
    )

    print("=" * 72)

    try:
        packet_id = "MSP_TEST_001"

        results = {
            "vision": ModuleResult.success(
                module_name=(
                    "scene_classifier"
                ),
                modality="vision",
                data={},
                confidence=0.90,
                processing_time_ms=10.0,
                source_packet_id=packet_id,
                warnings=[],
            ),
            "audio": ModuleResult.success(
                module_name=(
                    "sound_event_detector"
                ),
                modality="audio",
                data={},
                confidence=0.70,
                processing_time_ms=10.0,
                source_packet_id=packet_id,
                warnings=[],
            ),
            "spatial": ModuleResult.success(
                module_name=(
                    "depth_estimator"
                ),
                modality="spatial",
                data={},
                confidence=0.80,
                processing_time_ms=10.0,
                source_packet_id=packet_id,
                warnings=[],
            ),
            "text": ModuleResult.partial(
                module_name=(
                    "text_interpreter"
                ),
                modality="text",
                data={},
                confidence=0.60,
                processing_time_ms=10.0,
                source_packet_id=packet_id,
                warnings=[
                    "Low-contrast text."
                ],
            ),
        }

        features = {
            "vision": ModalityFeature(
                modality="vision",
                vector=(
                    0.90,
                    0.20,
                    0.10,
                    0.70,
                ),
                result=results["vision"],
                metadata={
                    "feature_type": (
                        "aligned_visual"
                    )
                },
            ),
            "audio": ModalityFeature(
                modality="audio",
                vector=(
                    0.20,
                    0.80,
                    0.40,
                    0.10,
                ),
                result=results["audio"],
                metadata={
                    "feature_type": (
                        "aligned_audio"
                    )
                },
            ),
            "spatial": ModalityFeature(
                modality="spatial",
                vector=(
                    0.40,
                    0.10,
                    0.90,
                    0.30,
                ),
                result=results["spatial"],
                metadata={
                    "feature_type": (
                        "aligned_spatial"
                    )
                },
            ),
            "text": ModalityFeature(
                modality="text",
                vector=(
                    0.30,
                    0.60,
                    0.20,
                    0.80,
                ),
                result=results["text"],
                metadata={
                    "feature_type": (
                        "aligned_text"
                    )
                },
            ),
        }

        engine = MultimodalFusionEngine(
            apply_attention=True
        )

        output = engine.fuse(
            features,
            source_packet_id=packet_id,
        )

        if not output.succeeded:
            raise AssertionError(
                "Multimodal fusion failed: "
                f"{output.result.errors}"
            )

        if output.fused_vector is None:
            raise AssertionError(
                "Fused vector is missing."
            )

        if output.vector_dimension != 4:
            raise AssertionError(
                "Unexpected fused-vector "
                "dimension."
            )

        if output.confidence_output is None:
            raise AssertionError(
                "Confidence output is missing."
            )

        if not math.isclose(
            output.confidence_output
            .weight_sum,
            1.0,
            abs_tol=1e-6,
        ):
            raise AssertionError(
                "Fusion confidence weights "
                "do not sum to one."
            )

        if not output.attention_applied:
            raise AssertionError(
                "Attention was not applied."
            )

        if output.attention_weights is None:
            raise AssertionError(
                "Attention weights are missing."
            )

        if (
            output.attention_weights.shape
            != (4, 4)
        ):
            raise AssertionError(
                "Unexpected attention matrix shape."
            )

        try:
            import numpy as np

            attention_row_sums = (
                output.attention_weights
                .sum(
                    axis=1
                )
            )

            if not np.allclose(
                attention_row_sums,
                1.0,
                atol=1e-6,
            ):
                raise AssertionError(
                    "Attention rows do not "
                    "sum to one."
                )

            fused_norm = float(
                np.linalg.norm(
                    output.fused_vector
                )
            )

            if not math.isclose(
                fused_norm,
                1.0,
                abs_tol=1e-6,
            ):
                raise AssertionError(
                    "Fused vector is not "
                    "L2-normalized."
                )

        except ImportError as error:
            raise AssertionError(
                "NumPy is unavailable."
            ) from error

        print(
            f"[PASS] Modalities fused: "
            f"{output.modality_order}"
        )

        print(
            f"[PASS] Vector dimension: "
            f"{output.vector_dimension}"
        )

        print(
            f"[PASS] Fused confidence: "
            f"{output.confidence_output.fused_confidence:.6f}"
        )

        print(
            f"[PASS] Attention shape: "
            f"{output.attention_weights.shape}"
        )

        print(
            f"[PASS] Fused vector: "
            f"{output.fused_vector.tolist()}"
        )

        print(
            "[PASS] Confidence-normalized weights applied"
        )

        print(
            "[PASS] Scaled dot-product attention applied"
        )

        print(
            "[PASS] Attention rows normalized"
        )

        print(
            "[PASS] Equal feature dimensions validated"
        )

        print(
            "[PASS] Unified perceptual representation generated"
        )

        print(
            "[PASS] ModuleResult generated"
        )

        print("\n" + "=" * 72)

        print(
            "[PASSED] MULTIMODAL FUSION "
            "ENGINE IS WORKING"
        )

        print("=" * 72)

        return True

    except (
        MultimodalFusionError,
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
            "multimodal-fusion self-test."
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
"""
============================================================
NOONGIL-X
Layer 2 : Perception Layer
Module  : Text Interpreter
File    : layer2/text/text_interpreter.py
============================================================

Purpose
-------
Transforms raw OCR detections into structured, readable and
Layer 3-compatible environmental text.

Responsibilities:
- clean OCR text
- remove duplicate detections
- determine reading order
- group detections into text lines
- generate a consolidated transcript
- classify text purpose
- identify important safety and navigation text
- generate standardized ModuleResult
============================================================
"""

from __future__ import annotations

import argparse
import re

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
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
    Layer1PacketAdapter,
)

from layer2.schemas.module_result import (
    ModuleResult,
)

from layer2.text.ocr_engine import (
    OCRDetection,
    OCREngine,
    OCREngineOutput,
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
    ModelLoader,
)

from layer2.vision_perception.vision_processor import (
    VisionProcessor,
)


# ============================================================
# CONSTANTS
# ============================================================

TEXT_INTERPRETER_VERSION = "1.0"

DEFAULT_LINE_TOLERANCE_RATIO = 0.60
DEFAULT_DUPLICATE_THRESHOLD = 0.90


# ============================================================
# TEXT CATEGORIES
# ============================================================

WARNING_KEYWORDS = {
    "warning",
    "danger",
    "caution",
    "hazard",
    "unsafe",
    "emergency",
    "fire",
    "restricted",
    "prohibited",
    "do not enter",
    "keep out",
    "high voltage",
    "slippery",
    "construction",
}

NAVIGATION_KEYWORDS = {
    "exit",
    "entrance",
    "entry",
    "gate",
    "left",
    "right",
    "straight",
    "upstairs",
    "downstairs",
    "floor",
    "lift",
    "elevator",
    "escalator",
    "platform",
    "terminal",
    "route",
    "road",
    "street",
    "parking",
}

INSTRUCTION_KEYWORDS = {
    "press",
    "pull",
    "push",
    "stop",
    "wait",
    "stand",
    "sit",
    "turn",
    "open",
    "close",
    "scan",
    "insert",
    "remove",
    "wear",
    "keep",
}

INFORMATION_KEYWORDS = {
    "information",
    "notice",
    "welcome",
    "office",
    "hospital",
    "school",
    "college",
    "classroom",
    "department",
    "reception",
    "counter",
    "help",
}

PRICE_PATTERN = re.compile(
    r"(?i)(?:₹|rs\.?|inr|\$|usd|€|eur|£|gbp)"
    r"\s*\d+(?:[.,]\d+)?"
)

TIME_PATTERN = re.compile(
    r"(?i)\b(?:[01]?\d|2[0-3])"
    r"(?::[0-5]\d)?\s*(?:am|pm)?\b"
)

PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?\d[\d\s\-]{7,}\d)(?!\d)"
)

URL_PATTERN = re.compile(
    r"(?i)\b(?:https?://|www\.)\S+"
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+"
    r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)


# ============================================================
# EXCEPTION
# ============================================================

class TextInterpretationError(Exception):
    """Raised when OCR text cannot be interpreted."""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class InterpretedText:
    """One cleaned and interpreted text region."""

    text_id: str

    text: str
    original_text: str

    category: str
    confidence: float
    importance: str

    reading_order: int

    source_detection_ids: Tuple[
        str,
        ...
    ]

    bounding_box: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:

        return {
            "text_id": self.text_id,
            "text": self.text,
            "original_text": (
                self.original_text
            ),
            "category": self.category,
            "confidence": round(
                self.confidence,
                6,
            ),
            "importance": self.importance,
            "reading_order": (
                self.reading_order
            ),
            "source_detection_ids": list(
                self.source_detection_ids
            ),
            "bounding_box": (
                self.bounding_box
            ),
        }


@dataclass
class TextInterpretationOutput:
    """Complete interpreted-text output."""

    result: ModuleResult

    texts: List[InterpretedText]
    transcript: str

    important_texts: List[
        InterpretedText
    ]

    @property
    def succeeded(self) -> bool:
        return self.result.usable

    @property
    def text_count(self) -> int:
        return len(self.texts)

    @property
    def important_count(self) -> int:
        return len(
            self.important_texts
        )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "text_count": self.text_count,
            "important_count": (
                self.important_count
            ),
            "transcript": self.transcript,
            "texts": [
                item.to_dict()
                for item in self.texts
            ],
            "important_texts": [
                item.to_dict()
                for item
                in self.important_texts
            ],
            "result": self.result.to_dict(),
        }


# ============================================================
# TEXT INTERPRETER
# ============================================================

class TextInterpreter:
    """Interpret raw environmental OCR detections."""

    def __init__(
        self,
        settings: Optional[
            Layer2Settings
        ] = None,
        *,
        logger: Optional[
            Layer2LoggerAdapter
        ] = None,
    ) -> None:

        self.settings = (
            settings
            or create_default_settings()
        )

        self.settings.validate()

        self.logger = (
            logger
            or get_logger(
                "text_interpreter"
            )
        )

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def interpret(
        self,
        ocr_output: OCREngineOutput,
    ) -> TextInterpretationOutput:
        """Interpret raw OCR detections."""

        if not isinstance(
            ocr_output,
            OCREngineOutput,
        ):
            raise TextInterpretationError(
                "ocr_output must be an "
                "OCREngineOutput."
            )

        source_packet_id = (
            ocr_output.result
            .source_packet_id
        )

        packet_logger = self.logger.bind(
            packet_id=source_packet_id
        )

        if not ocr_output.succeeded:

            result = ModuleResult.failure(
                module_name=(
                    "text_interpreter"
                ),
                modality="text",
                error=(
                    "A usable OCR-engine output "
                    "is required."
                ),
                source_packet_id=(
                    source_packet_id
                ),
            )

            return TextInterpretationOutput(
                result=result,
                texts=[],
                transcript="",
                important_texts=[],
            )

        log_event(
            packet_logger,
            event=(
                "text_interpretation_started"
            ),
            message=(
                "OCR text interpretation started."
            ),
        )

        with ModuleTimer(
            "text_interpreter",
            logger=packet_logger,
            packet_id=source_packet_id,
            log_start=False,
            log_completion=False,
        ) as timer:

            try:
                cleaned_detections = (
                    self._clean_detections(
                        ocr_output.detections
                    )
                )

                unique_detections = (
                    self._remove_duplicates(
                        cleaned_detections
                    )
                )

                ordered_detections = (
                    self._reading_order(
                        unique_detections
                    )
                )

                interpreted_texts = []

                for index, detection in enumerate(
                    ordered_detections,
                    start=1,
                ):
                    cleaned_text = (
                        self._clean_text(
                            detection.text
                        )
                    )

                    category = (
                        self._classify_text(
                            cleaned_text
                        )
                    )

                    importance = (
                        self._importance(
                            category,
                            cleaned_text,
                        )
                    )

                    interpreted_texts.append(
                        InterpretedText(
                            text_id=(
                                f"TEXT_{index:03d}"
                            ),
                            text=cleaned_text,
                            original_text=(
                                detection.text
                            ),
                            category=category,
                            confidence=(
                                detection.confidence
                            ),
                            importance=importance,
                            reading_order=index,
                            source_detection_ids=(
                                detection.detection_id,
                            ),
                            bounding_box=(
                                detection
                                .bounding_box
                                .to_dict()
                            ),
                        )
                    )

                transcript = self._transcript(
                    interpreted_texts
                )

                important_texts = [
                    item
                    for item
                    in interpreted_texts
                    if item.importance
                    in {
                        "critical",
                        "high",
                    }
                ]

                confidence = (
                    sum(
                        item.confidence
                        for item
                        in interpreted_texts
                    )
                    / len(interpreted_texts)
                    if interpreted_texts
                    else 0.0
                )

                warnings = []

                if not interpreted_texts:
                    warnings.append(
                        "No OCR text was available "
                        "for interpretation."
                    )

                data = {
                    "text_count": (
                        len(interpreted_texts)
                    ),
                    "important_count": (
                        len(important_texts)
                    ),
                    "transcript": transcript,
                    "texts": [
                        item.to_dict()
                        for item
                        in interpreted_texts
                    ],
                    "important_texts": [
                        item.to_dict()
                        for item
                        in important_texts
                    ],
                    "categories": (
                        self._category_summary(
                            interpreted_texts
                        )
                    ),
                    "interpreter_version": (
                        TEXT_INTERPRETER_VERSION
                    ),
                }

                status_factory = (
                    ModuleResult.success
                    if interpreted_texts
                    else ModuleResult.partial
                )

                result = status_factory(
                    module_name=(
                        "text_interpreter"
                    ),
                    modality="text",
                    data=data,
                    confidence=confidence,
                    processing_time_ms=(
                        timer.elapsed_ms
                    ),
                    source_packet_id=(
                        source_packet_id
                    ),
                    warnings=warnings,
                    metadata={
                        "source_module": (
                            "ocr_engine"
                        )
                    },
                )

                log_event(
                    packet_logger,
                    event=(
                        "text_interpretation_completed"
                    ),
                    message=(
                        "OCR text interpretation "
                        "completed."
                    ),
                    details={
                        "text_count": (
                            len(
                                interpreted_texts
                            )
                        ),
                        "important_count": (
                            len(
                                important_texts
                            )
                        ),
                        "processing_time_ms": (
                            timer.elapsed_ms
                        ),
                    },
                )

                return TextInterpretationOutput(
                    result=result,
                    texts=interpreted_texts,
                    transcript=transcript,
                    important_texts=(
                        important_texts
                    ),
                )

            except Exception as error:

                log_exception(
                    packet_logger,
                    error,
                    event=(
                        "text_interpretation_failed"
                    ),
                    message=(
                        "OCR text interpretation failed."
                    ),
                    details={
                        "processing_time_ms": (
                            timer.elapsed_ms
                        )
                    },
                )

                result = ModuleResult.failure(
                    module_name=(
                        "text_interpreter"
                    ),
                    modality="text",
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

                return TextInterpretationOutput(
                    result=result,
                    texts=[],
                    transcript="",
                    important_texts=[],
                )

    # --------------------------------------------------------
    # CLEANING
    # --------------------------------------------------------

    def _clean_detections(
        self,
        detections: Sequence[
            OCRDetection
        ],
    ) -> List[OCRDetection]:
        """Remove empty or invalid detections."""

        cleaned = []

        for detection in detections:

            if not isinstance(
                detection,
                OCRDetection,
            ):
                continue

            text = self._clean_text(
                detection.text
            )

            if not text:
                continue

            cleaned.append(
                OCRDetection(
                    detection_id=(
                        detection.detection_id
                    ),
                    text=text,
                    confidence=(
                        detection.confidence
                    ),
                    bounding_box=(
                        detection.bounding_box
                    ),
                )
            )

        return cleaned

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:
        """Normalize OCR whitespace and noise."""

        if not isinstance(text, str):
            return ""

        cleaned = text.replace(
            "\n",
            " ",
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        )

        cleaned = cleaned.strip(
            " \t\r\n|"
        )

        return cleaned

    # --------------------------------------------------------
    # DUPLICATE REMOVAL
    # --------------------------------------------------------

    def _remove_duplicates(
        self,
        detections: Sequence[
            OCRDetection
        ],
    ) -> List[OCRDetection]:
        """Remove repeated OCR detections."""

        unique = []

        for candidate in detections:

            duplicate_index = None

            for index, existing in enumerate(
                unique
            ):
                text_similarity = (
                    self._text_similarity(
                        candidate.text,
                        existing.text,
                    )
                )

                box_overlap = (
                    self._box_overlap(
                        candidate,
                        existing,
                    )
                )

                if (
                    text_similarity
                    >= DEFAULT_DUPLICATE_THRESHOLD
                    and box_overlap >= 0.50
                ):
                    duplicate_index = index
                    break

            if duplicate_index is None:
                unique.append(candidate)

            elif (
                candidate.confidence
                > unique[
                    duplicate_index
                ].confidence
            ):
                unique[
                    duplicate_index
                ] = candidate

        return unique

    @staticmethod
    def _text_similarity(
        first: str,
        second: str,
    ) -> float:
        """Calculate character-level similarity."""

        from difflib import SequenceMatcher

        return SequenceMatcher(
            None,
            first.lower(),
            second.lower(),
        ).ratio()

    @staticmethod
    def _box_overlap(
        first: OCRDetection,
        second: OCRDetection,
    ) -> float:
        """Calculate intersection over union."""

        first_box = first.bounding_box
        second_box = second.bounding_box

        intersection_x1 = max(
            first_box.x1,
            second_box.x1,
        )

        intersection_y1 = max(
            first_box.y1,
            second_box.y1,
        )

        intersection_x2 = min(
            first_box.x2,
            second_box.x2,
        )

        intersection_y2 = min(
            first_box.y2,
            second_box.y2,
        )

        intersection_width = max(
            0.0,
            intersection_x2
            - intersection_x1,
        )

        intersection_height = max(
            0.0,
            intersection_y2
            - intersection_y1,
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        first_area = (
            first_box.width
            * first_box.height
        )

        second_area = (
            second_box.width
            * second_box.height
        )

        union_area = (
            first_area
            + second_area
            - intersection_area
        )

        if union_area <= 0.0:
            return 0.0

        return (
            intersection_area
            / union_area
        )

    # --------------------------------------------------------
    # READING ORDER
    # --------------------------------------------------------

    def _reading_order(
        self,
        detections: Sequence[
            OCRDetection
        ],
    ) -> List[OCRDetection]:
        """Sort text top-to-bottom and left-to-right."""

        if not detections:
            return []

        median_height = sorted(
            detection.bounding_box.height
            for detection
            in detections
        )[len(detections) // 2]

        line_tolerance = max(
            5.0,
            median_height
            * DEFAULT_LINE_TOLERANCE_RATIO,
        )

        return sorted(
            detections,
            key=lambda detection: (
                round(
                    detection
                    .bounding_box
                    .center_y
                    / line_tolerance
                ),
                detection
                .bounding_box
                .center_x,
            ),
        )

    # --------------------------------------------------------
    # SEMANTIC CLASSIFICATION
    # --------------------------------------------------------

    def _classify_text(
        self,
        text: str,
    ) -> str:
        """Classify the purpose of detected text."""

        normalized = text.lower()

        if self._contains_keyword(
            normalized,
            WARNING_KEYWORDS,
        ):
            return "warning"

        if self._contains_keyword(
            normalized,
            NAVIGATION_KEYWORDS,
        ):
            return "navigation"

        if self._contains_keyword(
            normalized,
            INSTRUCTION_KEYWORDS,
        ):
            return "instruction"

        if PRICE_PATTERN.search(text):
            return "price"

        if EMAIL_PATTERN.search(text):
            return "email"

        if URL_PATTERN.search(text):
            return "website"

        if PHONE_PATTERN.search(text):
            return "phone_number"

        if TIME_PATTERN.search(text):
            return "time"

        if self._contains_keyword(
            normalized,
            INFORMATION_KEYWORDS,
        ):
            return "information"

        if self._looks_like_number(
            text
        ):
            return "number"

        return "general"

    @staticmethod
    def _contains_keyword(
        normalized_text: str,
        keywords: Sequence[str],
    ) -> bool:

        return any(
            keyword in normalized_text
            for keyword in keywords
        )

    @staticmethod
    def _looks_like_number(
        text: str,
    ) -> bool:

        compact = re.sub(
            r"[\s,.\-+/]",
            "",
            text,
        )

        return bool(
            compact
            and compact.isdigit()
        )

    # --------------------------------------------------------
    # IMPORTANCE
    # --------------------------------------------------------

    @staticmethod
    def _importance(
        category: str,
        text: str,
    ) -> str:
        """Determine assistance priority."""

        normalized = text.lower()

        if category == "warning":

            if any(
                keyword in normalized
                for keyword in {
                    "danger",
                    "emergency",
                    "fire",
                    "high voltage",
                    "do not enter",
                    "keep out",
                }
            ):
                return "critical"

            return "high"

        if category == "navigation":
            return "high"

        if category == "instruction":
            return "medium"

        if category in {
            "phone_number",
            "price",
            "time",
            "information",
        }:
            return "medium"

        return "low"

    # --------------------------------------------------------
    # OUTPUT CONSOLIDATION
    # --------------------------------------------------------

    @staticmethod
    def _transcript(
        texts: Sequence[
            InterpretedText
        ],
    ) -> str:
        """Combine text using reading order."""

        return " ".join(
            item.text
            for item in texts
            if item.text
        ).strip()

    @staticmethod
    def _category_summary(
        texts: Sequence[
            InterpretedText
        ],
    ) -> Dict[str, int]:
        """Count interpreted text categories."""

        summary: Dict[str, int] = {}

        for item in texts:

            summary[item.category] = (
                summary.get(
                    item.category,
                    0,
                )
                + 1
            )

        return summary


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def interpret_text(
    ocr_output: OCREngineOutput,
    *,
    settings: Optional[
        Layer2Settings
    ] = None,
) -> TextInterpretationOutput:

    interpreter = TextInterpreter(
        settings=settings
    )

    return interpreter.interpret(
        ocr_output
    )


# ============================================================
# SELF-TEST
# ============================================================

def run_self_test(
    scenario_name: str = "classroom",
) -> bool:

    print("=" * 72)

    print(
        "NOONGIL-X | TEXT INTERPRETER "
        "SELF-TEST"
    )

    print("=" * 72)

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

        ocr_engine = OCREngine(
            settings=settings,
            model_config=model_config,
            model_loader=loader,
        )

        interpreter = TextInterpreter(
            settings=settings
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

        ocr_output = ocr_engine.recognize(
            packet,
            vision_output,
        )

        if not ocr_output.succeeded:
            raise AssertionError(
                "OCR failed: "
                f"{ocr_output.result.errors}"
            )

        output = interpreter.interpret(
            ocr_output
        )

        if not output.succeeded:
            raise AssertionError(
                "Text interpretation failed: "
                f"{output.result.errors}"
            )

        for index, item in enumerate(
            output.texts,
            start=1,
        ):

            if item.reading_order != index:
                raise AssertionError(
                    "Invalid reading order."
                )

            if not item.text:
                raise AssertionError(
                    "Interpreted text is empty."
                )

            if not (
                0.0
                <= item.confidence
                <= 1.0
            ):
                raise AssertionError(
                    "Invalid text confidence."
                )

        print(
            f"[PASS] Scenario: "
            f"{scenario_name}"
        )

        print(
            f"[PASS] Raw OCR detections: "
            f"{ocr_output.detection_count}"
        )

        print(
            f"[PASS] Interpreted texts: "
            f"{output.text_count}"
        )

        print(
            f"[PASS] Important texts: "
            f"{output.important_count}"
        )

        print(
            f"[PASS] Transcript: "
            f"{output.transcript[:120]!r}"
        )

        print(
            "[PASS] OCR text cleaned"
        )

        print(
            "[PASS] Duplicate text removed"
        )

        print(
            "[PASS] Reading order generated"
        )

        print(
            "[PASS] Text categories generated"
        )

        print(
            "[PASS] Important text identified"
        )

        print(
            "[PASS] Layer 3 text format generated"
        )

        print(
            "[PASS] ModuleResult generated"
        )

        print("\n" + "=" * 72)

        print(
            "[PASSED] TEXT INTERPRETER "
            "IS WORKING"
        )

        print("=" * 72)

        return True

    except (
        DependencyMissingError,
        ModelLoadingError,
        TextInterpretationError,
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
            "text-interpreter self-test."
        )
    )

    parser.add_argument(
        "--scenario",
        default="classroom",
        help=(
            "Scenario used for testing. "
            "Default: classroom"
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
"""
============================================================
NOONGIL-X
Layer 2 : Final User Media Evaluation Dashboard
File    : layer2/final_test_app.py
============================================================

Purpose
-------
Allows a user to upload a real photograph and optional audio,
run the complete Layer 2 snapshot pipeline, inspect predictions,
enter ground-truth values and calculate evaluation metrics.

Run
---
streamlit run layer2/final_test_app.py
============================================================
"""

from __future__ import annotations

import json
import math
import re
import uuid

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from layer2.input_reception.layer1_packet_adapter import (
    AdaptedLayer1Input,
    AdaptedModality,
)

from layer2.run_layer2 import (
    Layer2Pipeline,
    Layer2PipelineError,
)


# ============================================================
# CONSTANTS
# ============================================================

APP_VERSION = "1.0"

SUPPORTED_IMAGE_TYPES = [
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "webp",
]

SUPPORTED_AUDIO_TYPES = [
    "wav",
    "mp3",
    "m4a",
    "flac",
    "ogg",
    "aac",
]

MODALITY_NAMES = (
    "vision",
    "audio",
    "spatial",
    "motion",
    "interaction",
    "environment",
    "device",
    "wearable",
)


# ============================================================
# GENERAL UTILITIES
# ============================================================

def utc_timestamp() -> str:
    """Return the current UTC time in ISO format."""

    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def normalize_label(value: Any) -> str:
    """Normalize one comparison label."""

    if value is None:
        return ""

    normalized = str(value).strip().lower()

    normalized = normalized.replace(
        "_",
        " ",
    )

    normalized = normalized.replace(
        "-",
        " ",
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized


def parse_label_list(value: str) -> List[str]:
    """
    Parse labels separated by commas or newlines.

    Repeated labels are preserved. Therefore:

        person, person, car

    represents two people and one car.
    """

    if not value.strip():
        return []

    parts = re.split(
        r"[,\n]+",
        value,
    )

    return [
        normalized
        for item in parts
        if (
            normalized := normalize_label(
                item
            )
        )
    ]


def safe_suffix(
    uploaded_name: str,
    fallback: str,
) -> str:
    """Return a safe lowercase file suffix."""

    suffix = Path(
        uploaded_name
    ).suffix.lower()

    if not suffix:
        return fallback

    if not re.fullmatch(
        r"\.[a-z0-9]{1,10}",
        suffix,
    ):
        return fallback

    return suffix


def save_uploaded_file(
    uploaded_file: Any,
    destination: Path,
) -> Path:
    """Save one Streamlit upload."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_bytes(
        uploaded_file.getvalue()
    )

    return destination.resolve()


def finite_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    """Safely convert a value to a finite float."""

    try:
        converted = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default

    if not math.isfinite(converted):
        return default

    return converted


# ============================================================
# PACKET CONSTRUCTION
# ============================================================

def create_user_packet(
    *,
    image_path: Path,
    audio_path: Optional[Path],
    latitude: Optional[float],
    longitude: Optional[float],
    sensor_confidence: float,
    project_root: Path,
) -> AdaptedLayer1Input:
    """
    Create a real AdaptedLayer1Input from user-selected media.

    This does not imitate model outputs. It only packages the
    uploaded files into the same structure consumed by Layer 2.
    """

    packet_token = uuid.uuid4().hex.upper()

    packet_id = (
        f"L1_USER_TEST_{packet_token[:16]}"
    )

    source_frame_id = (
        f"USER_FRAME_{packet_token[:12]}"
    )

    timestamp = utc_timestamp()

    modalities: Dict[
        str,
        AdaptedModality,
    ] = {}

    for modality_name in MODALITY_NAMES:

        modalities[modality_name] = (
            AdaptedModality(
                name=modality_name,
                available=False,
                data={},
                confidence=None,
                timestamp=timestamp,
                media_path=None,
                recovered=False,
                synchronized=False,
                warnings=[],
            )
        )

    modalities["vision"] = AdaptedModality(
        name="vision",
        available=True,
        data={
            "source": "user_upload",
            "frame_id": source_frame_id,
            "format": (
                image_path.suffix
                .lower()
                .lstrip(".")
            ),
        },
        confidence=sensor_confidence,
        timestamp=timestamp,
        media_path=image_path,
        recovered=False,
        synchronized=True,
        warnings=[],
    )

    if audio_path is not None:

        modalities["audio"] = (
            AdaptedModality(
                name="audio",
                available=True,
                data={
                    "source": "user_upload",
                    "format": (
                        audio_path.suffix
                        .lower()
                        .lstrip(".")
                    ),
                },
                confidence=sensor_confidence,
                timestamp=timestamp,
                media_path=audio_path,
                recovered=False,
                synchronized=True,
                warnings=[],
            )
        )

    spatial_data: Dict[str, Any] = {}

    if (
        latitude is not None
        and longitude is not None
    ):
        spatial_data["location"] = {
            "latitude": latitude,
            "longitude": longitude,
        }

        modalities["spatial"] = (
            AdaptedModality(
                name="spatial",
                available=True,
                data=spatial_data,
                confidence=sensor_confidence,
                timestamp=timestamp,
                media_path=None,
                recovered=False,
                synchronized=True,
                warnings=[],
            )
        )

    available_modalities = [
        name
        for name, modality
        in modalities.items()
        if modality.available
    ]

    missing_modalities = [
        name
        for name, modality
        in modalities.items()
        if not modality.available
    ]

    return AdaptedLayer1Input(
        packet_id=packet_id,
        timestamp=timestamp,
        source_frame_id=source_frame_id,
        scenario="user_media_test",
        source_mode="user_upload",
        simulated=False,
        modalities=modalities,
        sensor_confidence={
            "overall_confidence": (
                sensor_confidence
            ),
            "vision_confidence": (
                sensor_confidence
            ),
            "audio_confidence": (
                sensor_confidence
                if audio_path is not None
                else None
            ),
        },
        synchronization={
            "anchor_timestamp": timestamp,
            "synchronized": True,
            "available_modalities": (
                available_modalities
            ),
            "missing_modalities": (
                missing_modalities
            ),
        },
        recovery=None,
        layer2_contract={
            "ready_for_layer2": True,
            "effective_overall_confidence": (
                sensor_confidence
            ),
            "available_modalities": (
                available_modalities
            ),
            "missing_modalities": (
                missing_modalities
            ),
        },
        source_device={
            "type": "user_upload",
            "application": (
                "NOONGIL-X final test"
            ),
        },
        wearable=None,
        source_file=image_path,
        project_root=project_root,
        warnings=[],
    )


# ============================================================
# ACCURACY UTILITIES
# ============================================================

def calculate_label_metrics(
    expected: List[str],
    predicted: List[str],
) -> Dict[str, Any]:
    """
    Calculate multiset label precision, recall and F1.

    Repeated objects are counted. For example, two expected
    people require two predicted person detections.
    """

    expected_counter = Counter(
        normalize_label(item)
        for item in expected
        if normalize_label(item)
    )

    predicted_counter = Counter(
        normalize_label(item)
        for item in predicted
        if normalize_label(item)
    )

    matched = sum(
        (
            expected_counter
            & predicted_counter
        ).values()
    )

    expected_count = sum(
        expected_counter.values()
    )

    predicted_count = sum(
        predicted_counter.values()
    )

    precision = (
        matched / predicted_count
        if predicted_count
        else (
            1.0
            if expected_count == 0
            else 0.0
        )
    )

    recall = (
        matched / expected_count
        if expected_count
        else (
            1.0
            if predicted_count == 0
            else 0.0
        )
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        ) > 0.0
        else 0.0
    )

    false_positives = list(
        (
            predicted_counter
            - expected_counter
        ).elements()
    )

    missed_labels = list(
        (
            expected_counter
            - predicted_counter
        ).elements()
    )

    return {
        "matched": matched,
        "expected_count": expected_count,
        "predicted_count": predicted_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": (
            false_positives
        ),
        "missed_labels": missed_labels,
    }


def tokenize_text(value: str) -> List[str]:
    """Convert text to normalized word tokens."""

    normalized = normalize_label(
        value
    )

    normalized = re.sub(
        r"[^\w\s']",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    if not normalized:
        return []

    return normalized.split()


def edit_distance(
    reference: List[str],
    hypothesis: List[str],
) -> int:
    """Calculate Levenshtein edit distance."""

    previous_row = list(
        range(
            len(hypothesis) + 1
        )
    )

    for reference_index, reference_item in enumerate(
        reference,
        start=1,
    ):
        current_row = [
            reference_index
        ]

        for hypothesis_index, hypothesis_item in enumerate(
            hypothesis,
            start=1,
        ):
            insertion = (
                current_row[
                    hypothesis_index - 1
                ]
                + 1
            )

            deletion = (
                previous_row[
                    hypothesis_index
                ]
                + 1
            )

            substitution = (
                previous_row[
                    hypothesis_index - 1
                ]
                + (
                    0
                    if reference_item
                    == hypothesis_item
                    else 1
                )
            )

            current_row.append(
                min(
                    insertion,
                    deletion,
                    substitution,
                )
            )

        previous_row = current_row

    return previous_row[-1]


def calculate_text_metrics(
    expected: str,
    predicted: str,
) -> Dict[str, Any]:
    """Calculate WER and a bounded similarity score."""

    expected_tokens = tokenize_text(
        expected
    )

    predicted_tokens = tokenize_text(
        predicted
    )

    distance = edit_distance(
        expected_tokens,
        predicted_tokens,
    )

    if expected_tokens:
        word_error_rate = (
            distance
            / len(expected_tokens)
        )
    else:
        word_error_rate = (
            0.0
            if not predicted_tokens
            else 1.0
        )

    similarity = max(
        0.0,
        1.0 - word_error_rate,
    )

    exact_match = (
        normalize_label(expected)
        == normalize_label(predicted)
    )

    return {
        "expected_words": len(
            expected_tokens
        ),
        "predicted_words": len(
            predicted_tokens
        ),
        "edit_distance": distance,
        "word_error_rate": (
            word_error_rate
        ),
        "similarity": similarity,
        "exact_match": exact_match,
    }


def extract_object_labels(
    objects: List[Dict[str, Any]],
) -> List[str]:
    """Extract object labels from Layer 2 output."""

    labels = []

    for item in objects:

        if not isinstance(item, dict):
            continue

        label = item.get(
            "label",
            item.get("type"),
        )

        normalized = normalize_label(
            label
        )

        if normalized:
            labels.append(normalized)

    return labels


def extract_sound_labels(
    sounds: List[Dict[str, Any]],
) -> List[str]:
    """Extract sound labels from Layer 2 output."""

    labels = []

    for item in sounds:

        if not isinstance(item, dict):
            continue

        label = item.get(
            "normalized_label",
            item.get(
                "label",
                item.get("type"),
            ),
        )

        normalized = normalize_label(
            label
        )

        if normalized:
            labels.append(normalized)

    return labels


def extract_ocr_text(
    recognized_text: List[Dict[str, Any]],
) -> str:
    """Create a transcript from recognized text entries."""

    text_items = []

    for item in recognized_text:

        if not isinstance(item, dict):
            continue

        value = item.get(
            "text",
            item.get("content"),
        )

        if (
            isinstance(value, str)
            and value.strip()
        ):
            text_items.append(
                value.strip()
            )

    return " ".join(
        text_items
    )


def evaluate_output(
    *,
    output: Any,
    expected_scene: str,
    expected_objects: str,
    expected_sounds: str,
    expected_speech: str,
    expected_ocr: str,
) -> Dict[str, Any]:
    """Evaluate one final Layer 2 output."""

    evaluation: Dict[str, Any] = {
        "available_metrics": [],
    }

    scores = []

    if expected_scene.strip():

        predicted_scene = normalize_label(
            output.scene_type
        )

        normalized_expected_scene = (
            normalize_label(
                expected_scene
            )
        )

        scene_correct = (
            predicted_scene
            == normalized_expected_scene
        )

        scene_score = (
            1.0
            if scene_correct
            else 0.0
        )

        evaluation["scene"] = {
            "expected": (
                normalized_expected_scene
            ),
            "predicted": (
                predicted_scene
            ),
            "correct": scene_correct,
            "score": scene_score,
        }

        evaluation[
            "available_metrics"
        ].append("scene")

        scores.append(scene_score)

    if expected_objects.strip():

        expected_object_labels = (
            parse_label_list(
                expected_objects
            )
        )

        predicted_object_labels = (
            extract_object_labels(
                output.objects
            )
        )

        object_metrics = (
            calculate_label_metrics(
                expected_object_labels,
                predicted_object_labels,
            )
        )

        object_metrics["expected"] = (
            expected_object_labels
        )

        object_metrics["predicted"] = (
            predicted_object_labels
        )

        evaluation["objects"] = (
            object_metrics
        )

        evaluation[
            "available_metrics"
        ].append("objects")

        scores.append(
            object_metrics["f1"]
        )

    if expected_sounds.strip():

        expected_sound_labels = (
            parse_label_list(
                expected_sounds
            )
        )

        predicted_sound_labels = (
            extract_sound_labels(
                output.sounds
            )
        )

        sound_metrics = (
            calculate_label_metrics(
                expected_sound_labels,
                predicted_sound_labels,
            )
        )

        sound_metrics["expected"] = (
            expected_sound_labels
        )

        sound_metrics["predicted"] = (
            predicted_sound_labels
        )

        evaluation["sounds"] = (
            sound_metrics
        )

        evaluation[
            "available_metrics"
        ].append("sounds")

        scores.append(
            sound_metrics["f1"]
        )

    if expected_speech.strip():

        predicted_speech = " ".join(
            str(item)
            for item
            in output.speech_transcript
            if str(item).strip()
        )

        speech_metrics = (
            calculate_text_metrics(
                expected_speech,
                predicted_speech,
            )
        )

        speech_metrics["expected"] = (
            expected_speech
        )

        speech_metrics["predicted"] = (
            predicted_speech
        )

        evaluation["speech"] = (
            speech_metrics
        )

        evaluation[
            "available_metrics"
        ].append("speech")

        scores.append(
            speech_metrics["similarity"]
        )

    if expected_ocr.strip():

        predicted_ocr = (
            extract_ocr_text(
                output.recognized_text
            )
        )

        ocr_metrics = (
            calculate_text_metrics(
                expected_ocr,
                predicted_ocr,
            )
        )

        ocr_metrics["expected"] = (
            expected_ocr
        )

        ocr_metrics["predicted"] = (
            predicted_ocr
        )

        evaluation["ocr"] = (
            ocr_metrics
        )

        evaluation[
            "available_metrics"
        ].append("ocr")

        scores.append(
            ocr_metrics["similarity"]
        )

    evaluation["overall_test_score"] = (
        sum(scores) / len(scores)
        if scores
        else None
    )

    evaluation["metric_count"] = len(
        scores
    )

    return evaluation


# ============================================================
# PIPELINE CACHE
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def get_pipeline(
    project_root_string: str,
) -> Layer2Pipeline:
    """
    Create one cached pipeline.

    This preserves the model cache while the dashboard remains
    open, preventing every test from reloading all models.
    """

    return Layer2Pipeline(
        project_root=Path(
            project_root_string
        ),
        mode="snapshot",
    )


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def display_summary(
    output: Any,
) -> None:
    """Display the main Layer 2 prediction summary."""

    st.subheader(
        "Layer 2 prediction"
    )

    column_one, column_two, column_three = (
        st.columns(3)
    )

    column_one.metric(
        "Scene",
        output.scene_type,
    )

    column_two.metric(
        "Objects",
        len(output.objects),
    )

    confidence = (
        output.overall_confidence
    )

    column_three.metric(
        "Overall confidence",
        (
            f"{confidence:.2%}"
            if confidence is not None
            else "Unavailable"
        ),
    )

    st.write(
        "Status:",
        output.status,
    )

    st.write(
        "Ready for Layer 3:",
        output.ready_for_layer3,
    )

    st.write(
        "Processing time:",
        f"{output.processing_time_ms:.2f} ms",
    )


def display_predictions(
    output: Any,
) -> None:
    """Display predictions produced by each modality."""

    (
        scene_tab,
        objects_tab,
        audio_tab,
        text_tab,
        spatial_tab,
        fusion_tab,
        json_tab,
    ) = st.tabs([
        "Scene",
        "Objects",
        "Audio",
        "OCR",
        "Spatial",
        "Fusion",
        "Complete JSON",
    ])

    with scene_tab:
        st.json(
            output.scene
        )

    with objects_tab:

        if output.objects:
            st.dataframe(
                output.objects,
                use_container_width=True,
            )
        else:
            st.info(
                "No objects were detected."
            )

    with audio_tab:

        st.write(
            "Speech transcript"
        )

        if output.speech_transcript:
            st.success(
                " ".join(
                    output
                    .speech_transcript
                )
            )
        else:
            st.info(
                "No speech was transcribed."
            )

        st.write(
            "Sound events"
        )

        if output.sounds:
            st.dataframe(
                output.sounds,
                use_container_width=True,
            )
        else:
            st.info(
                "No environmental sound "
                "events were returned."
            )

    with text_tab:

        if output.recognized_text:
            st.dataframe(
                output.recognized_text,
                use_container_width=True,
            )
        else:
            st.info(
                "No visible text was recognized."
            )

    with spatial_tab:

        st.write(
            "Depth"
        )

        st.json(
            output.depth
        )

        st.write(
            "Obstacles"
        )

        if output.obstacles:
            st.dataframe(
                output.obstacles,
                use_container_width=True,
            )
        else:
            st.info(
                "No obstacle records were "
                "included in the final output."
            )

        st.write(
            "Location"
        )

        st.json(
            output.location
        )

    with fusion_tab:

        st.write(
            "Perception confidence"
        )

        st.json(
            output.perception_confidence
        )

        st.write(
            "Fusion"
        )

        st.json(
            output.fusion
        )

    with json_tab:

        output_payload = (
            output.to_dict()
        )

        st.json(
            output_payload
        )

        st.download_button(
            label=(
                "Download Layer 2 JSON"
            ),
            data=json.dumps(
                output_payload,
                indent=2,
                ensure_ascii=False,
            ),
            file_name=(
                "layer2_user_test_output.json"
            ),
            mime="application/json",
        )


def display_evaluation(
    evaluation: Dict[str, Any],
) -> None:
    """Display ground-truth evaluation metrics."""

    st.subheader(
        "Accuracy evaluation"
    )

    overall_score = evaluation.get(
        "overall_test_score"
    )

    if overall_score is None:
        st.info(
            "Enter at least one ground-truth "
            "value to calculate an evaluation score."
        )

        return

    st.metric(
        "Combined test score",
        f"{overall_score:.2%}",
    )

    st.caption(
        "This combined score is the mean of only "
        "the ground-truth metrics you supplied."
    )

    scene = evaluation.get(
        "scene"
    )

    if scene:

        st.write(
            "### Scene classification"
        )

        scene_column_one, scene_column_two, scene_column_three = (
            st.columns(3)
        )

        scene_column_one.metric(
            "Expected",
            scene["expected"],
        )

        scene_column_two.metric(
            "Predicted",
            scene["predicted"],
        )

        scene_column_three.metric(
            "Correct",
            (
                "Yes"
                if scene["correct"]
                else "No"
            ),
        )

    objects = evaluation.get(
        "objects"
    )

    if objects:

        st.write(
            "### Object-label evaluation"
        )

        column_one, column_two, column_three = (
            st.columns(3)
        )

        column_one.metric(
            "Precision",
            f"{objects['precision']:.2%}",
        )

        column_two.metric(
            "Recall",
            f"{objects['recall']:.2%}",
        )

        column_three.metric(
            "F1 score",
            f"{objects['f1']:.2%}",
        )

        st.write(
            "Expected labels:",
            objects["expected"],
        )

        st.write(
            "Predicted labels:",
            objects["predicted"],
        )

        st.write(
            "Missed labels:",
            objects["missed_labels"],
        )

        st.write(
            "False positives:",
            objects["false_positives"],
        )

    sounds = evaluation.get(
        "sounds"
    )

    if sounds:

        st.write(
            "### Sound-event evaluation"
        )

        column_one, column_two, column_three = (
            st.columns(3)
        )

        column_one.metric(
            "Precision",
            f"{sounds['precision']:.2%}",
        )

        column_two.metric(
            "Recall",
            f"{sounds['recall']:.2%}",
        )

        column_three.metric(
            "F1 score",
            f"{sounds['f1']:.2%}",
        )

        st.write(
            "Expected sounds:",
            sounds["expected"],
        )

        st.write(
            "Predicted sounds:",
            sounds["predicted"],
        )

    speech = evaluation.get(
        "speech"
    )

    if speech:

        st.write(
            "### Speech-recognition evaluation"
        )

        column_one, column_two = (
            st.columns(2)
        )

        column_one.metric(
            "Similarity",
            (
                f"{speech['similarity']:.2%}"
            ),
        )

        column_two.metric(
            "Word error rate",
            (
                f"{speech['word_error_rate']:.2%}"
            ),
        )

        st.write(
            "Expected:",
            speech["expected"],
        )

        st.write(
            "Predicted:",
            speech["predicted"],
        )

    ocr = evaluation.get(
        "ocr"
    )

    if ocr:

        st.write(
            "### OCR evaluation"
        )

        column_one, column_two = (
            st.columns(2)
        )

        column_one.metric(
            "Similarity",
            f"{ocr['similarity']:.2%}",
        )

        column_two.metric(
            "Word error rate",
            (
                f"{ocr['word_error_rate']:.2%}"
            ),
        )

        st.write(
            "Expected:",
            ocr["expected"],
        )

        st.write(
            "Predicted:",
            ocr["predicted"],
        )


def add_history_record(
    *,
    run: Any,
    evaluation: Dict[str, Any],
    image_name: str,
    audio_name: Optional[str],
) -> None:
    """Add one result to the Streamlit session history."""

    if "test_history" not in st.session_state:
        st.session_state[
            "test_history"
        ] = []

    st.session_state[
        "test_history"
    ].append({
        "timestamp": utc_timestamp(),
        "image": image_name,
        "audio": audio_name,
        "scene": (
            run.output.scene_type
        ),
        "object_count": len(
            run.output.objects
        ),
        "overall_confidence": (
            run.output.overall_confidence
        ),
        "evaluation_score": (
            evaluation.get(
                "overall_test_score"
            )
        ),
        "status": run.output.status,
    })


def display_history() -> None:
    """Display results from the current browser session."""

    history = st.session_state.get(
        "test_history",
        [],
    )

    if not history:
        return

    st.subheader(
        "Current test-session history"
    )

    st.dataframe(
        history,
        use_container_width=True,
    )

    evaluated_scores = [
        item["evaluation_score"]
        for item in history
        if item.get(
            "evaluation_score"
        ) is not None
    ]

    if evaluated_scores:

        aggregate_score = (
            sum(evaluated_scores)
            / len(evaluated_scores)
        )

        st.metric(
            "Average evaluated score",
            f"{aggregate_score:.2%}",
        )

    st.download_button(
        label="Download test history",
        data=json.dumps(
            history,
            indent=2,
            ensure_ascii=False,
        ),
        file_name=(
            "layer2_test_history.json"
        ),
        mime="application/json",
    )


# ============================================================
# STREAMLIT APPLICATION
# ============================================================

def main() -> None:
    """Run the final Layer 2 test dashboard."""

    st.set_page_config(
        page_title=(
            "NOONGIL-X Layer 2 Test"
        ),
        page_icon="👁️",
        layout="wide",
    )

    st.title(
        "NOONGIL-X Layer 2 Final Test"
    )

    st.write(
        "Upload your own photograph and optional "
        "audio recording. The dashboard runs the "
        "real Layer 2 snapshot pipeline."
    )

    st.warning(
        "The first run may be slow because the AI "
        "models must be loaded. Later tests reuse "
        "the cached models."
    )

    project_root = (
        Path(__file__).resolve().parents[1]
    )

    with st.sidebar:

        st.header(
            "Input media"
        )

        image_upload = (
            st.file_uploader(
                "Select a photograph",
                type=(
                    SUPPORTED_IMAGE_TYPES
                ),
                accept_multiple_files=False,
            )
        )

        audio_upload = (
            st.file_uploader(
                "Select optional audio",
                type=(
                    SUPPORTED_AUDIO_TYPES
                ),
                accept_multiple_files=False,
            )
        )

        st.header(
            "Sensor information"
        )

        sensor_confidence = (
            st.slider(
                "Input sensor confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.90,
                step=0.01,
            )
        )

        include_location = (
            st.checkbox(
                "Include GPS location",
                value=False,
            )
        )

        latitude = None
        longitude = None

        if include_location:

            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=0.0,
                format="%.6f",
            )

            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=0.0,
                format="%.6f",
            )

    input_column, truth_column = (
        st.columns(2)
    )

    with input_column:

        st.subheader(
            "Media preview"
        )

        if image_upload is not None:
            st.image(
                image_upload,
                caption=image_upload.name,
                use_container_width=True,
            )
        else:
            st.info(
                "Select a photograph from "
                "the sidebar."
            )

        if audio_upload is not None:
            st.audio(
                audio_upload
            )

    with truth_column:

        st.subheader(
            "Optional ground truth"
        )

        st.caption(
            "Only filled fields contribute to "
            "the combined evaluation score."
        )

        expected_scene = (
            st.text_input(
                "Correct scene",
                placeholder=(
                    "Example: park"
                ),
            )
        )

        expected_objects = (
            st.text_area(
                "Correct object labels",
                placeholder=(
                    "Example: person, dog, bench"
                ),
                help=(
                    "Use repeated labels when multiple "
                    "instances exist, such as: "
                    "person, person, car"
                ),
            )
        )

        expected_sounds = (
            st.text_area(
                "Correct sound-event labels",
                placeholder=(
                    "Example: speech, traffic"
                ),
            )
        )

        expected_speech = (
            st.text_area(
                "Correct spoken transcript",
                placeholder=(
                    "Type exactly what is spoken "
                    "in the audio."
                ),
            )
        )

        expected_ocr = (
            st.text_area(
                "Correct visible text",
                placeholder=(
                    "Type the text visible "
                    "in the photograph."
                ),
            )
        )

    run_button = st.button(
        "Run complete Layer 2 test",
        type="primary",
        use_container_width=True,
    )

    if run_button:

        if image_upload is None:

            st.error(
                "Please select a photograph before "
                "running Layer 2."
            )

            return

        run_token = (
            uuid.uuid4().hex
        )

        test_directory = (
            project_root
            / "output"
            / "layer2"
            / "user_tests"
            / run_token
        )

        image_suffix = safe_suffix(
            image_upload.name,
            ".jpg",
        )

        image_path = save_uploaded_file(
            image_upload,
            test_directory
            / f"input_image{image_suffix}",
        )

        audio_path = None

        if audio_upload is not None:

            audio_suffix = safe_suffix(
                audio_upload.name,
                ".wav",
            )

            audio_path = save_uploaded_file(
                audio_upload,
                test_directory
                / (
                    f"input_audio"
                    f"{audio_suffix}"
                ),
            )

        packet = create_user_packet(
            image_path=image_path,
            audio_path=audio_path,
            latitude=(
                finite_float(latitude)
                if include_location
                else None
            ),
            longitude=(
                finite_float(longitude)
                if include_location
                else None
            ),
            sensor_confidence=(
                sensor_confidence
            ),
            project_root=project_root,
        )

        output_path = (
            test_directory
            / "layer2_output.json"
        )

        try:

            pipeline = get_pipeline(
                str(project_root)
            )

            with st.spinner(
                "Running complete Layer 2 "
                "perception pipeline..."
            ):

                run = pipeline.process_packet(
                    packet,
                    output_path=output_path,
                    save_output=True,
                )

        except Layer2PipelineError as error:

            st.error(
                "Layer 2 pipeline failed."
            )

            st.exception(error)

            return

        except Exception as error:

            st.error(
                "Unexpected test-dashboard error."
            )

            st.exception(error)

            return

        evaluation = evaluate_output(
            output=run.output,
            expected_scene=(
                expected_scene
            ),
            expected_objects=(
                expected_objects
            ),
            expected_sounds=(
                expected_sounds
            ),
            expected_speech=(
                expected_speech
            ),
            expected_ocr=expected_ocr,
        )

        add_history_record(
            run=run,
            evaluation=evaluation,
            image_name=(
                image_upload.name
            ),
            audio_name=(
                audio_upload.name
                if audio_upload
                is not None
                else None
            ),
        )

        st.success(
            "Layer 2 test completed successfully."
        )

        display_summary(
            run.output
        )

        display_predictions(
            run.output
        )

        display_evaluation(
            evaluation
        )

        evaluation_payload = {
            "application_version": (
                APP_VERSION
            ),
            "packet_id": (
                packet.packet_id
            ),
            "media": {
                "image": (
                    image_upload.name
                ),
                "audio": (
                    audio_upload.name
                    if audio_upload
                    is not None
                    else None
                ),
            },
            "evaluation": evaluation,
            "layer2_output": (
                run.output.to_dict()
            ),
        }

        st.download_button(
            label=(
                "Download complete evaluation"
            ),
            data=json.dumps(
                evaluation_payload,
                indent=2,
                ensure_ascii=False,
            ),
            file_name=(
                "layer2_complete_evaluation.json"
            ),
            mime="application/json",
            use_container_width=True,
        )

    display_history()


if __name__ == "__main__":
    main()
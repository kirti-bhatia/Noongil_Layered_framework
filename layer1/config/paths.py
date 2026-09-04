"""
============================================================
NOONGIL-X
Layer 1 : Multimodal Input Layer
Module  : Centralized Path Configuration
File    : layer1/config/paths.py
============================================================

Purpose
-------
Defines every filesystem path used by Layer 1.

All Layer 1 modules should import paths from this file instead
of manually building project-relative paths.

Compatibility
-------------
Python 3.10+
Standard library only
============================================================
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Dict, Iterable, List


# ============================================================
# PROJECT ROOT
# ============================================================

CURRENT_FILE: Path = Path(__file__).resolve()
CONFIG_DIR: Path = CURRENT_FILE.parent
LAYER1_DIR: Path = CONFIG_DIR.parent
PROJECT_ROOT: Path = LAYER1_DIR.parent


# ============================================================
# SOURCE DIRECTORIES
# ============================================================

ACQUISITION_DIR: Path = LAYER1_DIR / "acquisition"
MODALITIES_DIR: Path = LAYER1_DIR / "modalities"
PROCESSING_DIR: Path = LAYER1_DIR / "processing"
OUTPUT_MODULE_DIR: Path = LAYER1_DIR / "output"
SCHEMAS_DIR: Path = LAYER1_DIR / "schemas"
UTILS_DIR: Path = LAYER1_DIR / "utils"


# ============================================================
# PROJECT DATA DIRECTORIES
# ============================================================

DATA_DIR: Path = PROJECT_ROOT / "data"
LAYER1_DATA_DIR: Path = DATA_DIR / "layer1"

SIMULATION_DATA_DIR: Path = LAYER1_DATA_DIR / "simulation"
SCENARIO_DATA_DIR: Path = SIMULATION_DATA_DIR / "scenarios"
REPLAY_DATA_DIR: Path = LAYER1_DATA_DIR / "replay"


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR: Path = PROJECT_ROOT / "output"
LAYER1_OUTPUT_DIR: Path = OUTPUT_DIR / "layer1"
LAYER2_OUTPUT_DIR: Path = OUTPUT_DIR / "layer2"

RAW_OUTPUT_DIR: Path = LAYER1_OUTPUT_DIR / "raw"
PREPROCESSED_OUTPUT_DIR: Path = LAYER1_OUTPUT_DIR / "preprocessed"
SYNCHRONIZED_OUTPUT_DIR: Path = LAYER1_OUTPUT_DIR / "synchronized"
CONFIDENCE_OUTPUT_DIR: Path = LAYER1_OUTPUT_DIR / "confidence"
RECOVERY_OUTPUT_DIR: Path = LAYER1_OUTPUT_DIR / "recovery"
PACKET_OUTPUT_DIR: Path = LAYER1_OUTPUT_DIR / "packets"
STATE_OUTPUT_DIR: Path = LAYER1_OUTPUT_DIR / "state"
CACHE_OUTPUT_DIR: Path = LAYER1_OUTPUT_DIR / "cache"
DEBUG_OUTPUT_DIR: Path = LAYER1_OUTPUT_DIR / "debug"


# ============================================================
# MODALITY-SPECIFIC OUTPUT DIRECTORIES
# ============================================================

RAW_VISION_DIR: Path = RAW_OUTPUT_DIR / "vision"
RAW_AUDIO_DIR: Path = RAW_OUTPUT_DIR / "audio"
RAW_SPATIAL_DIR: Path = RAW_OUTPUT_DIR / "spatial"
RAW_MOTION_DIR: Path = RAW_OUTPUT_DIR / "motion"
RAW_INTERACTION_DIR: Path = RAW_OUTPUT_DIR / "interaction"
RAW_DEVICE_DIR: Path = RAW_OUTPUT_DIR / "device"
RAW_ENVIRONMENT_DIR: Path = RAW_OUTPUT_DIR / "environment"

PREPROCESSED_VISION_DIR: Path = PREPROCESSED_OUTPUT_DIR / "vision"
PREPROCESSED_AUDIO_DIR: Path = PREPROCESSED_OUTPUT_DIR / "audio"
PREPROCESSED_SPATIAL_DIR: Path = PREPROCESSED_OUTPUT_DIR / "spatial"
PREPROCESSED_MOTION_DIR: Path = PREPROCESSED_OUTPUT_DIR / "motion"
PREPROCESSED_INTERACTION_DIR: Path = PREPROCESSED_OUTPUT_DIR / "interaction"
PREPROCESSED_DEVICE_DIR: Path = PREPROCESSED_OUTPUT_DIR / "device"
PREPROCESSED_ENVIRONMENT_DIR: Path = PREPROCESSED_OUTPUT_DIR / "environment"


# ============================================================
# LOG DIRECTORIES
# ============================================================

LOGS_DIR: Path = PROJECT_ROOT / "logs"
LAYER1_LOG_DIR: Path = LOGS_DIR / "layer1"

LAYER1_LOG_PATH: Path = LAYER1_LOG_DIR / "layer1.log"
LAYER1_ERROR_LOG_PATH: Path = LAYER1_LOG_DIR / "layer1_errors.log"
LAYER1_SENSOR_LOG_PATH: Path = LAYER1_LOG_DIR / "sensor_activity.log"


# ============================================================
# TEST DIRECTORIES
# ============================================================

TESTS_DIR: Path = PROJECT_ROOT / "tests"
LAYER1_TESTS_DIR: Path = TESTS_DIR / "layer1"


# ============================================================
# OFFICIAL OUTPUT FILES
# ============================================================

LATEST_RAW_INPUT_PATH: Path = (
    RAW_OUTPUT_DIR / "latest_raw_sensor_input.json"
)

LATEST_PREPROCESSED_INPUT_PATH: Path = (
    PREPROCESSED_OUTPUT_DIR / "latest_preprocessed_input.json"
)

SYNCHRONIZED_FRAME_PATH: Path = (
    SYNCHRONIZED_OUTPUT_DIR / "synchronized_frame.json"
)

SENSOR_CONFIDENCE_PATH: Path = (
    CONFIDENCE_OUTPUT_DIR / "sensor_confidence.json"
)

SENSOR_CONFLICTS_PATH: Path = (
    CONFIDENCE_OUTPUT_DIR / "sensor_conflicts.json"
)

RECOVERY_REPORT_PATH: Path = (
    RECOVERY_OUTPUT_DIR / "recovery_report.json"
)

MULTIMODAL_SENSOR_PACKET_PATH: Path = (
    PACKET_OUTPUT_DIR / "multimodal_sensor_packet.json"
)

LAYER2_INPUT_PACKET_PATH: Path = (
    LAYER2_OUTPUT_DIR / "layer1_sensor_packet.json"
)

LAYER1_PIPELINE_SUMMARY_PATH: Path = (
    LAYER1_OUTPUT_DIR / "layer1_pipeline_summary.json"
)

RECEIVER_STATE_PATH: Path = (
    STATE_OUTPUT_DIR / "receiver_state.json"
)

SIMULATOR_STATE_PATH: Path = (
    STATE_OUTPUT_DIR / "simulator_state.json"
)

NAMARA_STATE_PATH: Path = (
    STATE_OUTPUT_DIR / "namara_state.json"
)

DEVICE_STATE_PATH: Path = (
    STATE_OUTPUT_DIR / "device_state.json"
)


# ============================================================
# REQUIRED DIRECTORY COLLECTION
# ============================================================

REQUIRED_DIRECTORIES: List[Path] = [
    ACQUISITION_DIR,
    MODALITIES_DIR,
    PROCESSING_DIR,
    OUTPUT_MODULE_DIR,
    SCHEMAS_DIR,
    UTILS_DIR,

    DATA_DIR,
    LAYER1_DATA_DIR,
    SIMULATION_DATA_DIR,
    SCENARIO_DATA_DIR,
    REPLAY_DATA_DIR,

    OUTPUT_DIR,
    LAYER1_OUTPUT_DIR,
    LAYER2_OUTPUT_DIR,

    RAW_OUTPUT_DIR,
    PREPROCESSED_OUTPUT_DIR,
    SYNCHRONIZED_OUTPUT_DIR,
    CONFIDENCE_OUTPUT_DIR,
    RECOVERY_OUTPUT_DIR,
    PACKET_OUTPUT_DIR,
    STATE_OUTPUT_DIR,
    CACHE_OUTPUT_DIR,
    DEBUG_OUTPUT_DIR,

    RAW_VISION_DIR,
    RAW_AUDIO_DIR,
    RAW_SPATIAL_DIR,
    RAW_MOTION_DIR,
    RAW_INTERACTION_DIR,
    RAW_DEVICE_DIR,
    RAW_ENVIRONMENT_DIR,

    PREPROCESSED_VISION_DIR,
    PREPROCESSED_AUDIO_DIR,
    PREPROCESSED_SPATIAL_DIR,
    PREPROCESSED_MOTION_DIR,
    PREPROCESSED_INTERACTION_DIR,
    PREPROCESSED_DEVICE_DIR,
    PREPROCESSED_ENVIRONMENT_DIR,

    LOGS_DIR,
    LAYER1_LOG_DIR,

    TESTS_DIR,
    LAYER1_TESTS_DIR,
]


# ============================================================
# DIRECTORY HELPERS
# ============================================================

def ensure_directory(path: str | Path) -> Path:
    """
    Create one directory and all missing parents.
    """

    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_parent_directory(file_path: str | Path) -> Path:
    """
    Ensure that the parent directory of a file exists.
    """

    resolved = Path(file_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_directories(
    directories: Iterable[str | Path],
) -> List[Path]:
    """
    Create multiple directories.
    """

    return [
        ensure_directory(directory)
        for directory in directories
    ]


def initialize_layer1_directories() -> Dict[str, object]:
    """
    Create and validate the full Layer 1 directory structure.
    """

    created = ensure_directories(REQUIRED_DIRECTORIES)

    missing = [
        str(path)
        for path in REQUIRED_DIRECTORIES
        if not path.exists()
    ]

    return {
        "valid": len(missing) == 0,
        "project_root": str(PROJECT_ROOT),
        "layer1_dir": str(LAYER1_DIR),
        "created_or_existing_count": len(created),
        "missing_directories": missing,
    }


# ============================================================
# PATH HELPERS
# ============================================================

def relative_to_project(path: str | Path) -> str:
    """
    Return a path relative to the project root when possible.
    """

    resolved = Path(path).expanduser().resolve()

    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def get_modality_raw_directory(modality: str) -> Path:
    """
    Return the raw-output directory for a modality.
    """

    normalized = modality.strip().lower()

    mapping = {
        "vision": RAW_VISION_DIR,
        "camera": RAW_VISION_DIR,
        "audio": RAW_AUDIO_DIR,
        "microphone": RAW_AUDIO_DIR,
        "spatial": RAW_SPATIAL_DIR,
        "gps": RAW_SPATIAL_DIR,
        "motion": RAW_MOTION_DIR,
        "accelerometer": RAW_MOTION_DIR,
        "gyroscope": RAW_MOTION_DIR,
        "magnetometer": RAW_MOTION_DIR,
        "interaction": RAW_INTERACTION_DIR,
        "button": RAW_INTERACTION_DIR,
        "touch": RAW_INTERACTION_DIR,
        "wearable": RAW_DEVICE_DIR,
        "device": RAW_DEVICE_DIR,
        "earphone": RAW_DEVICE_DIR,
        "environment": RAW_ENVIRONMENT_DIR,
    }

    if normalized not in mapping:
        raise ValueError(
            f"Unsupported modality for raw directory: {modality!r}"
        )

    return mapping[normalized]


def get_modality_preprocessed_directory(
    modality: str,
) -> Path:
    """
    Return the preprocessed-output directory for a modality.
    """

    normalized = modality.strip().lower()

    mapping = {
        "vision": PREPROCESSED_VISION_DIR,
        "camera": PREPROCESSED_VISION_DIR,
        "audio": PREPROCESSED_AUDIO_DIR,
        "microphone": PREPROCESSED_AUDIO_DIR,
        "spatial": PREPROCESSED_SPATIAL_DIR,
        "gps": PREPROCESSED_SPATIAL_DIR,
        "motion": PREPROCESSED_MOTION_DIR,
        "accelerometer": PREPROCESSED_MOTION_DIR,
        "gyroscope": PREPROCESSED_MOTION_DIR,
        "magnetometer": PREPROCESSED_MOTION_DIR,
        "interaction": PREPROCESSED_INTERACTION_DIR,
        "button": PREPROCESSED_INTERACTION_DIR,
        "touch": PREPROCESSED_INTERACTION_DIR,
        "wearable": PREPROCESSED_DEVICE_DIR,
        "device": PREPROCESSED_DEVICE_DIR,
        "earphone": PREPROCESSED_DEVICE_DIR,
        "environment": PREPROCESSED_ENVIRONMENT_DIR,
    }

    if normalized not in mapping:
        raise ValueError(
            "Unsupported modality for preprocessed directory: "
            f"{modality!r}"
        )

    return mapping[normalized]


def build_timestamped_filename(
    *,
    prefix: str,
    suffix: str,
    timestamp: str,
) -> str:
    """
    Build a filesystem-safe timestamped filename.

    Example:
        frame_20260806T062826660.jpg
    """

    if not prefix.strip():
        raise ValueError("prefix cannot be empty.")

    if not suffix.strip():
        raise ValueError("suffix cannot be empty.")

    safe_timestamp = (
        timestamp
        .replace("-", "")
        .replace(":", "")
        .replace("+", "_")
        .replace(".", "")
    )

    normalized_suffix = suffix.lstrip(".")

    return (
        f"{prefix.strip()}_"
        f"{safe_timestamp}.{normalized_suffix}"
    )


# ============================================================
# PATH REGISTRY
# ============================================================

def get_path_registry() -> Dict[str, str]:
    """
    Return important Layer 1 paths as strings.
    """

    return {
        "project_root": str(PROJECT_ROOT),
        "layer1_dir": str(LAYER1_DIR),
        "config_dir": str(CONFIG_DIR),
        "acquisition_dir": str(ACQUISITION_DIR),
        "modalities_dir": str(MODALITIES_DIR),
        "processing_dir": str(PROCESSING_DIR),
        "schemas_dir": str(SCHEMAS_DIR),
        "utils_dir": str(UTILS_DIR),

        "layer1_data_dir": str(LAYER1_DATA_DIR),
        "simulation_data_dir": str(SIMULATION_DATA_DIR),
        "scenario_data_dir": str(SCENARIO_DATA_DIR),

        "layer1_output_dir": str(LAYER1_OUTPUT_DIR),
        "raw_output_dir": str(RAW_OUTPUT_DIR),
        "preprocessed_output_dir": str(
            PREPROCESSED_OUTPUT_DIR
        ),
        "synchronized_output_dir": str(
            SYNCHRONIZED_OUTPUT_DIR
        ),
        "confidence_output_dir": str(
            CONFIDENCE_OUTPUT_DIR
        ),
        "recovery_output_dir": str(RECOVERY_OUTPUT_DIR),
        "packet_output_dir": str(PACKET_OUTPUT_DIR),
        "state_output_dir": str(STATE_OUTPUT_DIR),

        "layer1_log_dir": str(LAYER1_LOG_DIR),
        "layer1_tests_dir": str(LAYER1_TESTS_DIR),

        "multimodal_sensor_packet": str(
            MULTIMODAL_SENSOR_PACKET_PATH
        ),
        "layer2_input_packet": str(
            LAYER2_INPUT_PACKET_PATH
        ),
        "pipeline_summary": str(
            LAYER1_PIPELINE_SUMMARY_PATH
        ),
    }


# ============================================================
# SELF-TEST
# ============================================================

def run_paths_self_test() -> bool:
    """
    Create and validate all configured Layer 1 paths.
    """

    print("\n" + "=" * 72)
    print("NOONGIL-X | LAYER 1 PATH CONFIGURATION TEST")
    print("=" * 72)

    try:
        print("[1/5] Detecting project structure...")

        if LAYER1_DIR.name != "layer1":
            raise AssertionError(
                "paths.py must be located inside "
                "layer1/config/."
            )

        print(f"[SUCCESS] Project root: {PROJECT_ROOT}")
        print(f"[SUCCESS] Layer 1 dir: {LAYER1_DIR}")

        print("[2/5] Creating required directories...")

        result = initialize_layer1_directories()

        if not result["valid"]:
            raise AssertionError(
                "Some directories could not be created: "
                f"{result['missing_directories']}"
            )

        print(
            "[SUCCESS] "
            f"{result['created_or_existing_count']} "
            "directories are ready."
        )

        print("[3/5] Testing modality path routing...")

        if get_modality_raw_directory("camera") != RAW_VISION_DIR:
            raise AssertionError(
                "Camera raw-path mapping is incorrect."
            )

        if (
            get_modality_preprocessed_directory("gyroscope")
            != PREPROCESSED_MOTION_DIR
        ):
            raise AssertionError(
                "Gyroscope preprocessed-path mapping is incorrect."
            )

        print("[SUCCESS] Modality path routing is valid.")

        print("[4/5] Testing official output parents...")

        official_files = [
            MULTIMODAL_SENSOR_PACKET_PATH,
            LAYER2_INPUT_PACKET_PATH,
            LAYER1_PIPELINE_SUMMARY_PATH,
            RECEIVER_STATE_PATH,
            SIMULATOR_STATE_PATH,
            NAMARA_STATE_PATH,
        ]

        for file_path in official_files:
            ensure_parent_directory(file_path)

            if not file_path.parent.exists():
                raise AssertionError(
                    f"Missing parent directory for {file_path}"
                )

        print("[SUCCESS] Official output parents exist.")

        print("[5/5] Testing path registry serialization...")

        registry = get_path_registry()
        serialized = json.dumps(
            registry,
            indent=2,
            ensure_ascii=False,
        )

        if not serialized:
            raise AssertionError(
                "Path registry serialization failed."
            )

        print("[SUCCESS] Path registry is JSON serializable.")

        print("\nImportant paths:")
        print(serialized)

        print("\n" + "=" * 72)
        print("[PASSED] LAYER 1 PATH CONFIGURATION IS WORKING")
        print("=" * 72)

        return True

    except Exception as error:
        print("\n" + "=" * 72)
        print("[FAILED] LAYER 1 PATH CONFIGURATION TEST")
        print("=" * 72)
        print(f"[ERROR] {type(error).__name__}: {error}")

        return False


if __name__ == "__main__":
    passed = run_paths_self_test()

    if not passed:
        raise SystemExit(1)
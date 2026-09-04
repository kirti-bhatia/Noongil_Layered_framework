"""
NOONGIL-X
Combined Layer 3 and Layer 4 Test Pipeline

File:
    pipeline/noongil_reasoning_test_pipeline.py

Flow:
    Select test situation
        -> copy as Layer 2 output
        -> run Layer 3
        -> validate context graph
        -> run Layer 4
        -> validate final decision and explanation
        -> archive complete run
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# PATH CONFIGURATION
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PIPELINE_DIR = CURRENT_FILE.parent
BASE_DIR = PIPELINE_DIR.parent

TEST_DIR = BASE_DIR / "test"

OUTPUT_DIR = BASE_DIR / "output"
LAYER2_OUTPUT_DIR = OUTPUT_DIR / "layer2"
LAYER3_OUTPUT_DIR = OUTPUT_DIR / "layer3"
LAYER4_OUTPUT_DIR = OUTPUT_DIR / "layer4"

LAYER2_INPUT_PATH = (
    LAYER2_OUTPUT_DIR
    / "layer2_output.json"
)

LAYER3_CONTEXT_GRAPH_PATH = (
    LAYER3_OUTPUT_DIR
    / "context_graph.json"
)

LAYER4_DECISION_PATH = (
    LAYER4_OUTPUT_DIR
    / "decision_output.json"
)

LAYER4_EXPLANATION_PATH = (
    LAYER4_OUTPUT_DIR
    / "explanation_output.json"
)

TEST_RUNS_DIR = (
    OUTPUT_DIR
    / "test_runs"
)

for directory in (
    TEST_DIR,
    OUTPUT_DIR,
    LAYER2_OUTPUT_DIR,
    LAYER3_OUTPUT_DIR,
    LAYER4_OUTPUT_DIR,
    TEST_RUNS_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# PIPELINE COMMANDS
# ============================================================

LAYER3_PIPELINE_COMMAND = [
    sys.executable,
    "-m",
    "layer3.pipeline.layer3_pipeline",
]

LAYER4_PIPELINE_COMMAND = [
    sys.executable,
    "-m",
    "layer4.pipeline.reasoning_pipeline",
]


# ============================================================
# DISPLAY
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_error(message: str) -> None:
    print(f"[ERROR] {message}")


def print_info(message: str) -> None:
    print(f"[INFO] {message}")


def print_success(message: str) -> None:
    print(f"[SUCCESS] {message}")


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(file_path: Path) -> Optional[Dict[str, Any]]:
    if not file_path.exists():
        return None

    try:
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        return data if isinstance(data, dict) else None

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None


def save_json(
    data: Dict[str, Any],
    file_path: Path,
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# TEST CASE SELECTION
# ============================================================

def list_test_files() -> List[Path]:
    files = sorted(
        file_path
        for file_path in TEST_DIR.glob("*.json")
        if file_path.is_file()
    )

    print_header("AVAILABLE TEST SITUATIONS")

    if not files:
        print_error(
            f"No JSON files found in: {TEST_DIR}"
        )
        return []

    for index, file_path in enumerate(
        files,
        start=1,
    ):
        display_name = (
            file_path.stem
            .replace("_", " ")
            .replace("-", " ")
            .title()
        )

        print(
            f"{index}. {display_name} "
            f"({file_path.name})"
        )

    return files


def choose_test_file(
    files: List[Path],
) -> Optional[Path]:
    while True:
        choice = input(
            "\nEnter situation number "
            "(0 to cancel): "
        ).strip()

        try:
            number = int(choice)
        except ValueError:
            print_error("Enter a valid number.")
            continue

        if number == 0:
            return None

        if 1 <= number <= len(files):
            return files[number - 1]

        print_error("Choice is out of range.")


# ============================================================
# INPUT PREPARATION
# ============================================================

def prepare_layer2_input(
    selected_test: Path,
) -> bool:
    test_data = load_json(
        selected_test
    )

    if test_data is None:
        print_error(
            f"Invalid JSON test file: {selected_test}"
        )
        return False

    shutil.copy2(
        selected_test,
        LAYER2_INPUT_PATH,
    )

    print_info(
        f"Selected situation: {selected_test.name}"
    )
    print_info(
        f"Layer 2 input: {LAYER2_INPUT_PATH}"
    )

    return True


# ============================================================
# CLEAN CURRENT OUTPUTS
# ============================================================

def clear_directory_json_files(
    directory: Path,
) -> None:
    if not directory.exists():
        return

    for file_path in directory.glob("*.json"):
        try:
            file_path.unlink()
        except OSError:
            pass


def clear_previous_runtime_outputs() -> None:
    """
    Remove only generated Layer 3 and Layer 4 JSON outputs.

    The selected Layer 2 input is preserved.
    """

    clear_directory_json_files(
        LAYER3_OUTPUT_DIR
    )

    clear_directory_json_files(
        LAYER4_OUTPUT_DIR
    )


# ============================================================
# PROCESS EXECUTION
# ============================================================

def run_command(
    title: str,
    command: List[str],
) -> bool:
    print_header(title)

    result = subprocess.run(
        command,
        cwd=str(BASE_DIR),
        check=False,
    )

    if result.returncode != 0:
        print_error(
            f"{title} failed with exit code "
            f"{result.returncode}."
        )
        return False

    print_success(
        f"{title} completed."
    )

    return True


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_layer3_output() -> bool:
    graph = load_json(
        LAYER3_CONTEXT_GRAPH_PATH
    )

    if graph is None:
        print_error(
            "Layer 3 did not generate a valid "
            "context_graph.json."
        )
        return False

    if not isinstance(
        graph.get("nodes"),
        list,
    ):
        print_error(
            "Layer 3 context graph has no valid nodes list."
        )
        return False

    if not isinstance(
        graph.get("edges"),
        list,
    ):
        print_error(
            "Layer 3 context graph has no valid edges list."
        )
        return False

    print_success(
        "Layer 3 context graph validated."
    )

    return True


def validate_layer4_output() -> bool:
    decision = load_json(
        LAYER4_DECISION_PATH
    )

    explanation = load_json(
        LAYER4_EXPLANATION_PATH
    )

    if decision is None:
        print_error(
            "Layer 4 did not generate a valid "
            "decision_output.json."
        )
        return False

    if explanation is None:
        print_error(
            "Layer 4 did not generate a valid "
            "explanation_output.json."
        )
        return False

    required_decision_fields = {
        "decision_mode",
        "primary_action",
        "decision_confidence",
        "decision_status",
    }

    missing_fields = (
        required_decision_fields
        - set(decision.keys())
    )

    if missing_fields:
        print_error(
            "Decision output is missing: "
            + ", ".join(
                sorted(missing_fields)
            )
        )
        return False

    print_success(
        "Layer 4 decision and explanation validated."
    )

    return True


# ============================================================
# RUN ARCHIVING
# ============================================================

def create_run_directory(
    selected_test: Path,
) -> Path:
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    case_name = selected_test.stem

    run_directory = (
        TEST_RUNS_DIR
        / case_name
        / timestamp
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return run_directory


def copy_directory_contents(
    source: Path,
    destination: Path,
) -> None:
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not source.exists():
        return

    for item in source.iterdir():
        target = destination / item.name

        if item.is_file():
            shutil.copy2(
                item,
                target,
            )

        elif item.is_dir():
            shutil.copytree(
                item,
                target,
                dirs_exist_ok=True,
            )


def archive_run(
    selected_test: Path,
    run_directory: Path,
    report: Dict[str, Any],
) -> None:
    input_directory = (
        run_directory
        / "input"
    )

    layer3_archive = (
        run_directory
        / "layer3"
    )

    layer4_archive = (
        run_directory
        / "layer4"
    )

    input_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        selected_test,
        input_directory
        / selected_test.name,
    )

    shutil.copy2(
        LAYER2_INPUT_PATH,
        input_directory
        / "layer2_output.json",
    )

    copy_directory_contents(
        LAYER3_OUTPUT_DIR,
        layer3_archive,
    )

    copy_directory_contents(
        LAYER4_OUTPUT_DIR,
        layer4_archive,
    )

    save_json(
        report,
        run_directory
        / "combined_test_report.json",
    )


# ============================================================
# REPORT
# ============================================================

def build_report(
    selected_test: Path,
    started_at: str,
    completed_at: str,
    duration_seconds: float,
    status: str,
    failed_stage: Optional[str],
) -> Dict[str, Any]:
    context_graph = load_json(
        LAYER3_CONTEXT_GRAPH_PATH
    ) or {}

    decision = load_json(
        LAYER4_DECISION_PATH
    ) or {}

    explanation = load_json(
        LAYER4_EXPLANATION_PATH
    ) or {}

    return {
        "timestamp": completed_at,
        "pipeline": (
            "NOONGIL-X Combined Layer 3 "
            "and Layer 4 Test Pipeline"
        ),
        "status": status,
        "failed_stage": failed_stage,
        "selected_test_case": {
            "name": selected_test.stem,
            "file": str(selected_test),
        },
        "execution": {
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": duration_seconds,
        },
        "layer3_result": {
            "context_graph_generated": (
                LAYER3_CONTEXT_GRAPH_PATH.exists()
            ),
            "node_count": context_graph.get(
                "node_count",
                len(context_graph.get("nodes", [])),
            ),
            "edge_count": context_graph.get(
                "edge_count",
                len(context_graph.get("edges", [])),
            ),
        },
        "layer4_result": {
            "decision_mode": decision.get(
                "decision_mode"
            ),
            "primary_action": decision.get(
                "primary_action"
            ),
            "decision_confidence": decision.get(
                "decision_confidence"
            ),
            "decision_status": decision.get(
                "decision_status"
            ),
            "explanation_confidence": (
                explanation.get(
                    "explanation_confidence"
                )
            ),
            "summary": explanation.get(
                "summary"
            ),
        },
    }


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    report: Dict[str, Any],
    run_directory: Path,
) -> None:
    print_header("COMBINED PIPELINE RESULT")

    print(
        f"Situation: "
        f"{report['selected_test_case']['name']}"
    )
    print(
        f"Status: {report['status'].upper()}"
    )
    print(
        f"Duration: "
        f"{report['execution']['duration_seconds']}s"
    )

    if report["status"] == "success":
        layer3 = report["layer3_result"]
        layer4 = report["layer4_result"]

        print(
            f"Context Graph: "
            f"{layer3['node_count']} nodes, "
            f"{layer3['edge_count']} edges"
        )
        print(
            f"Decision Mode: "
            f"{layer4['decision_mode']}"
        )
        print(
            f"Primary Action: "
            f"{layer4['primary_action']}"
        )
        print(
            f"Decision Confidence: "
            f"{layer4['decision_confidence']}"
        )
        print(
            f"Decision Status: "
            f"{layer4['decision_status']}"
        )

    else:
        print(
            f"Failed Stage: "
            f"{report['failed_stage']}"
        )

    print(
        f"Saved Run: {run_directory}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline() -> None:
    print_header(
        "NOONGIL-X COMBINED LAYER 3 + LAYER 4 TEST PIPELINE"
    )

    test_files = list_test_files()

    if not test_files:
        return

    selected_test = choose_test_file(
        test_files
    )

    if selected_test is None:
        print_info("Pipeline cancelled.")
        return

    run_directory = create_run_directory(
        selected_test
    )

    started_at = datetime.now().isoformat(
        timespec="seconds"
    )

    start_time = time.perf_counter()

    status = "failed"
    failed_stage: Optional[str] = None

    clear_previous_runtime_outputs()

    if not prepare_layer2_input(
        selected_test
    ):
        failed_stage = "input_preparation"

    elif not run_command(
        "RUNNING LAYER 3 PIPELINE",
        LAYER3_PIPELINE_COMMAND,
    ):
        failed_stage = "layer3_pipeline"

    elif not validate_layer3_output():
        failed_stage = "layer3_validation"

    elif not run_command(
        "RUNNING LAYER 4 PIPELINE",
        LAYER4_PIPELINE_COMMAND,
    ):
        failed_stage = "layer4_pipeline"

    elif not validate_layer4_output():
        failed_stage = "layer4_validation"

    else:
        status = "success"

    completed_at = datetime.now().isoformat(
        timespec="seconds"
    )

    duration_seconds = round(
        time.perf_counter() - start_time,
        3,
    )

    report = build_report(
        selected_test=selected_test,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        status=status,
        failed_stage=failed_stage,
    )

    archive_run(
        selected_test=selected_test,
        run_directory=run_directory,
        report=report,
    )

    print_final_summary(
        report,
        run_directory,
    )


def main() -> None:
    try:
        run_pipeline()

    except KeyboardInterrupt:
        print("\n[INFO] Pipeline interrupted.")

    except Exception as error:
        print_error(
            f"Unexpected failure: "
            f"{type(error).__name__}: {error}"
        )


if __name__ == "__main__":
    main()
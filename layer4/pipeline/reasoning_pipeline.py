"""
============================================================
NOONGIL-X
Layer 4 : Reasoning & Intelligence Layer
Module  : Reasoning Pipeline
File    : layer4/pipeline/reasoning_pipeline.py
============================================================

Purpose
-------
Run the complete Layer 4 reasoning pipeline in sequence:

1. Context Analyzer
2. Cognitive State Manager
3. Situation Understanding
4. Intent Reasoner
5. Hazard Detector / Hazard Reasoner
6. Prediction Engine
7. Reasoning Fusion
8. Decision Engine
9. Explanation Engine

The pipeline:
- executes each module in the correct order
- validates generated output files
- stops safely if a required stage fails
- records execution time and stage status
- saves a final pipeline report

Output
------
output/layer4/reasoning_pipeline_report.json
============================================================
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

PIPELINE_DIR = CURRENT_FILE.parent
LAYER4_DIR = PIPELINE_DIR.parent
PROJECT_ROOT = LAYER4_DIR.parent

LAYER3_OUTPUT_DIR = PROJECT_ROOT / "output" / "layer3"
LAYER4_OUTPUT_DIR = PROJECT_ROOT / "output" / "layer4"

LAYER4_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# OPTIONAL PROJECT UTILITIES
# ============================================================

try:
    from layer4.utils.logger import (
        module_start,
        module_end,
        log_info,
        log_warning,
        log_error,
        log_success,
        log_section,
    )

except ImportError:

    def module_start(name: str) -> None:
        print("\n" + "=" * 60)
        print(f"NOONGIL-X | {name}")
        print("=" * 60)

    def module_end(name: str) -> None:
        print("\n" + "=" * 60)
        print(f"{name} Completed")
        print("=" * 60)

    def log_info(message: str) -> None:
        print(f"[INFO] {message}")

    def log_warning(message: str) -> None:
        print(f"[WARNING] {message}")

    def log_error(message: str) -> None:
        print(f"[ERROR] {message}")

    def log_success(message: str) -> None:
        print(f"[SUCCESS] {message}")

    def log_section(message: str) -> None:
        print("\n" + "-" * 60)
        print(message)
        print("-" * 60)


# ============================================================
# PIPELINE CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PipelineStage:
    """
    Configuration for one Layer 4 pipeline stage.
    """

    name: str
    module_candidates: Tuple[str, ...]
    output_file: Path
    required: bool = True


@dataclass
class StageExecutionResult:
    """
    Runtime result for one pipeline stage.
    """

    stage_name: str
    module_name: Optional[str]
    status: str
    started_at: str
    completed_at: str
    duration_seconds: float
    output_file: str
    output_exists: bool
    output_valid: bool
    output_size_bytes: int
    error: Optional[str] = None


PIPELINE_STAGES: Tuple[PipelineStage, ...] = (

    PipelineStage(
        name="Context Analyzer",
        module_candidates=(
            "layer4.context_processing.context_analyzer",
        ),
        output_file=(
            LAYER4_OUTPUT_DIR
            / "analyzed_context.json"
        ),
    ),

    PipelineStage(
        name="Cognitive State Manager",
        module_candidates=(
            "layer4.context_processing.cognitive_state_manager",
            "layer4.context_processing.cognitive_stat_manager",
        ),
        output_file=(
            LAYER4_OUTPUT_DIR
            / "cognitive_state.json"
        ),
    ),

    PipelineStage(
        name="Situation Understanding",
        module_candidates=(
            "layer4.reasoning.situation_understanding",
            "layer4.decision.situation_understanding",
        ),
        output_file=(
            LAYER4_OUTPUT_DIR
            / "situation_understanding.json"
        ),
    ),

    PipelineStage(
        name="Intent Reasoner",
        module_candidates=(
            "layer4.reasoning.intent_reasoner",
        ),
        output_file=(
            LAYER4_OUTPUT_DIR
            / "intent_reasoning.json"
        ),
    ),

    PipelineStage(
        name="Hazard Reasoning",
        module_candidates=(
            "layer4.reasoning.hazard_detector",
            "layer4.reasoning.hazard_reasoner",
        ),
        output_file=(
            LAYER4_OUTPUT_DIR
            / "hazards.json"
        ),
    ),

    PipelineStage(
        name="Prediction Engine",
        module_candidates=(
            "layer4.reasoning.prediction_engine",
        ),
        output_file=(
            LAYER4_OUTPUT_DIR
            / "predictions.json"
        ),
    ),

    PipelineStage(
        name="Reasoning Fusion",
        module_candidates=(
            "layer4.reasoning.reasoning_fusion",
        ),
        output_file=(
            LAYER4_OUTPUT_DIR
            / "reasoning_fusion.json"
        ),
    ),

    PipelineStage(
        name="Decision Engine",
        module_candidates=(
            "layer4.decision.decision_engine",
        ),
        output_file=(
            LAYER4_OUTPUT_DIR
            / "decision_output.json"
        ),
    ),

    PipelineStage(
        name="Explanation Engine",
        module_candidates=(
            "layer4.decision.explanation_engine",
        ),
        output_file=(
            LAYER4_OUTPUT_DIR
            / "explanation_output.json"
        ),
    ),
)


PIPELINE_REPORT_PATH = (
    LAYER4_OUTPUT_DIR
    / "reasoning_pipeline_report.json"
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def current_timestamp() -> str:
    return datetime.now().isoformat(
        timespec="seconds"
    )


def load_json_file(
    file_path: Path,
) -> Optional[Dict[str, Any]]:
    """
    Load and validate a JSON object.
    """

    if not file_path.exists():
        return None

    try:
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

        return None

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None


def save_json_file(
    data: Dict[str, Any],
    file_path: Path,
) -> None:
    """
    Save JSON data using UTF-8 formatting.
    """

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


def resolve_module_name(
    candidates: Tuple[str, ...],
) -> Optional[str]:
    """
    Return the first importable module name.
    """

    for module_name in candidates:

        try:
            module_spec = importlib.util.find_spec(
                module_name
            )

            if module_spec is not None:
                return module_name

        except (
            ImportError,
            ModuleNotFoundError,
            AttributeError,
        ):
            continue

    return None


def validate_output_file(
    output_file: Path,
) -> Tuple[bool, bool, int]:
    """
    Check whether a stage output exists and contains valid JSON.
    """

    exists = output_file.exists()

    if not exists:
        return False, False, 0

    try:
        size = output_file.stat().st_size
    except OSError:
        size = 0

    data = load_json_file(
        output_file
    )

    valid = (
        isinstance(data, dict)
        and len(data) > 0
    )

    return exists, valid, size


def remove_previous_output(
    output_file: Path,
) -> None:
    """
    Remove a previous stage output before execution.

    This prevents an old file from being mistaken for a newly generated
    successful result.
    """

    if not output_file.exists():
        return

    try:
        output_file.unlink()

    except OSError as error:
        log_warning(
            "Could not remove previous output "
            f"{output_file}: {error}"
        )


# ============================================================
# MODULE EXECUTION
# ============================================================

def run_module_in_process(
    module_name: str,
) -> None:
    """
    Import a module and call its main() function.

    This method is fast and keeps all stages inside the same Python process.
    """

    module = importlib.import_module(
        module_name
    )

    main_function = getattr(
        module,
        "main",
        None,
    )

    if not callable(main_function):
        raise AttributeError(
            f"Module '{module_name}' does not define a callable main() function."
        )

    main_function()


def run_module_subprocess(
    module_name: str,
) -> None:
    """
    Execute a module in an isolated Python subprocess.

    Subprocess execution prevents global state, imported modules, or logger
    configuration from one stage affecting later stages.
    """

    command = [
        sys.executable,
        "-m",
        module_name,
    ]

    completed_process = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    if completed_process.returncode != 0:
        raise RuntimeError(
            f"Module '{module_name}' exited with code "
            f"{completed_process.returncode}."
        )


def execute_stage(
    stage: PipelineStage,
    execution_mode: str,
    clean_previous_output: bool,
) -> StageExecutionResult:
    """
    Execute one Layer 4 pipeline stage.
    """

    started_at = current_timestamp()
    start_time = time.perf_counter()

    module_name = resolve_module_name(
        stage.module_candidates
    )

    if module_name is None:
        completed_at = current_timestamp()
        duration = round(
            time.perf_counter() - start_time,
            3,
        )

        return StageExecutionResult(
            stage_name=stage.name,
            module_name=None,
            status="failed",
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            output_file=str(stage.output_file),
            output_exists=False,
            output_valid=False,
            output_size_bytes=0,
            error=(
                "No importable module found. Checked: "
                + ", ".join(stage.module_candidates)
            ),
        )

    if clean_previous_output:
        remove_previous_output(
            stage.output_file
        )

    try:
        log_section(
            f"Running Stage: {stage.name}"
        )

        log_info(
            f"Module: {module_name}"
        )

        log_info(
            f"Expected Output: {stage.output_file}"
        )

        if execution_mode == "subprocess":
            run_module_subprocess(
                module_name
            )

        else:
            run_module_in_process(
                module_name
            )

        exists, valid, size = validate_output_file(
            stage.output_file
        )

        if not exists:
            raise FileNotFoundError(
                "Stage completed without generating its expected output: "
                f"{stage.output_file}"
            )

        if not valid:
            raise ValueError(
                "Stage generated an empty or invalid JSON output: "
                f"{stage.output_file}"
            )

        completed_at = current_timestamp()
        duration = round(
            time.perf_counter() - start_time,
            3,
        )

        log_success(
            f"{stage.name} completed successfully"
        )

        return StageExecutionResult(
            stage_name=stage.name,
            module_name=module_name,
            status="success",
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            output_file=str(stage.output_file),
            output_exists=exists,
            output_valid=valid,
            output_size_bytes=size,
            error=None,
        )

    except Exception as error:
        completed_at = current_timestamp()
        duration = round(
            time.perf_counter() - start_time,
            3,
        )

        exists, valid, size = validate_output_file(
            stage.output_file
        )

        error_message = (
            f"{type(error).__name__}: {error}"
        )

        log_error(
            f"{stage.name} failed: {error_message}"
        )

        return StageExecutionResult(
            stage_name=stage.name,
            module_name=module_name,
            status="failed",
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            output_file=str(stage.output_file),
            output_exists=exists,
            output_valid=valid,
            output_size_bytes=size,
            error=error_message,
        )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_pipeline_input() -> Tuple[bool, List[str]]:
    """
    Validate the main Layer 4 input before starting the pipeline.
    """

    required_input = (
        LAYER3_OUTPUT_DIR
        / "context_graph.json"
    )

    errors = []

    if not required_input.exists():
        errors.append(
            f"Missing Layer 3 input: {required_input}"
        )

    elif load_json_file(required_input) is None:
        errors.append(
            "Layer 3 context graph is empty or invalid JSON: "
            f"{required_input}"
        )

    return len(errors) == 0, errors


# ============================================================
# PIPELINE REPORT
# ============================================================

def build_pipeline_report(
    pipeline_started_at: str,
    pipeline_completed_at: str,
    total_duration: float,
    execution_mode: str,
    stop_on_failure: bool,
    results: List[StageExecutionResult],
) -> Dict[str, Any]:
    """
    Build a complete pipeline execution report.
    """

    successful_stages = [
        result
        for result in results
        if result.status == "success"
    ]

    failed_stages = [
        result
        for result in results
        if result.status == "failed"
    ]

    pipeline_status = (
        "success"
        if len(failed_stages) == 0
        and len(results) == len(PIPELINE_STAGES)
        else "failed"
    )

    final_explanation = load_json_file(
        LAYER4_OUTPUT_DIR
        / "explanation_output.json"
    )

    final_decision = load_json_file(
        LAYER4_OUTPUT_DIR
        / "decision_output.json"
    )

    return {
        "timestamp": current_timestamp(),
        "pipeline_name": (
            "NOONGIL-X Layer 4 Reasoning Pipeline"
        ),
        "pipeline_status": pipeline_status,
        "pipeline_started_at": pipeline_started_at,
        "pipeline_completed_at": pipeline_completed_at,
        "total_duration_seconds": total_duration,

        "configuration": {
            "execution_mode": execution_mode,
            "stop_on_failure": stop_on_failure,
            "clean_previous_output": True,
            "stage_count": len(
                PIPELINE_STAGES
            ),
        },

        "execution_summary": {
            "executed_stage_count": len(results),
            "successful_stage_count": len(
                successful_stages
            ),
            "failed_stage_count": len(
                failed_stages
            ),
            "all_stages_completed": (
                len(results)
                == len(PIPELINE_STAGES)
            ),
        },

        "stage_results": [
            asdict(result)
            for result in results
        ],

        "final_decision_summary": (
            final_decision.get("summary")
            if final_decision
            else None
        ),

        "final_explanation_summary": (
            final_explanation.get("summary")
            if final_explanation
            else None
        ),

        "layer4_outputs": {
            stage.name: str(
                stage.output_file
            )
            for stage in PIPELINE_STAGES
        },

        "next_layer": (
            final_decision.get("next_layer")
            if final_decision
            else None
        ),
    }


# ============================================================
# PIPELINE ORCHESTRATOR
# ============================================================

def run_reasoning_pipeline(
    execution_mode: str = "subprocess",
    stop_on_failure: bool = True,
    clean_previous_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete Layer 4 pipeline.

    Parameters
    ----------
    execution_mode:
        "subprocess" is recommended because each module runs independently.
        "in_process" imports modules and calls their main() functions directly.

    stop_on_failure:
        Stop immediately if a required stage fails.

    clean_previous_outputs:
        Remove each stage's old output before running that stage.
    """

    if execution_mode not in {
        "subprocess",
        "in_process",
    }:
        raise ValueError(
            "execution_mode must be 'subprocess' or 'in_process'."
        )

    module_start(
        "LAYER 4 REASONING PIPELINE"
    )

    pipeline_started_at = current_timestamp()
    pipeline_start_time = time.perf_counter()

    valid_input, input_errors = validate_pipeline_input()

    if not valid_input:
        for error in input_errors:
            log_error(error)

        report = {
            "timestamp": current_timestamp(),
            "pipeline_name": (
                "NOONGIL-X Layer 4 Reasoning Pipeline"
            ),
            "pipeline_status": "failed",
            "failure_stage": "input_validation",
            "errors": input_errors,
            "stage_results": [],
        }

        save_json_file(
            report,
            PIPELINE_REPORT_PATH,
        )

        module_end(
            "LAYER 4 REASONING PIPELINE"
        )

        return report

    log_info(
        "Layer 3 context graph validated"
    )

    log_info(
        f"Execution Mode: {execution_mode}"
    )

    results: List[StageExecutionResult] = []

    for stage_number, stage in enumerate(
        PIPELINE_STAGES,
        start=1,
    ):
        print("\n" + "=" * 60)
        print(
            f"STAGE {stage_number}/{len(PIPELINE_STAGES)} "
            f"| {stage.name.upper()}"
        )
        print("=" * 60)

        result = execute_stage(
            stage=stage,
            execution_mode=execution_mode,
            clean_previous_output=(
                clean_previous_outputs
            ),
        )

        results.append(result)

        if (
            result.status == "failed"
            and stage.required
            and stop_on_failure
        ):
            log_error(
                "Pipeline stopped because a required stage failed."
            )
            break

    pipeline_completed_at = current_timestamp()
    total_duration = round(
        time.perf_counter() - pipeline_start_time,
        3,
    )

    report = build_pipeline_report(
        pipeline_started_at=(
            pipeline_started_at
        ),
        pipeline_completed_at=(
            pipeline_completed_at
        ),
        total_duration=total_duration,
        execution_mode=execution_mode,
        stop_on_failure=stop_on_failure,
        results=results,
    )

    save_json_file(
        report,
        PIPELINE_REPORT_PATH,
    )

    print("\n" + "=" * 60)
    print("LAYER 4 PIPELINE SUMMARY")
    print("=" * 60)

    for result in results:
        status_text = result.status.upper()

        print(
            f"[{status_text}] "
            f"{result.stage_name} "
            f"({result.duration_seconds}s)"
        )

        if result.error:
            print(
                f"         Error: {result.error}"
            )

    print("-" * 60)
    print(
        "Successful Stages: "
        f"{report['execution_summary']['successful_stage_count']}"
    )
    print(
        "Failed Stages: "
        f"{report['execution_summary']['failed_stage_count']}"
    )
    print(
        f"Total Duration: {total_duration}s"
    )
    print(
        f"Pipeline Status: "
        f"{report['pipeline_status'].upper()}"
    )
    print(
        f"Report: {PIPELINE_REPORT_PATH}"
    )

    if report["pipeline_status"] == "success":
        log_success(
            "Complete Layer 4 reasoning pipeline executed successfully"
        )

    else:
        log_error(
            "Layer 4 reasoning pipeline did not complete successfully"
        )

    module_end(
        "LAYER 4 REASONING PIPELINE"
    )

    return report


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Default pipeline execution.

    Subprocess mode is used because it is more isolated and reliable for
    modules that manage their own imports, paths, logs, and JSON files.
    """

    try:
        run_reasoning_pipeline(
            execution_mode="subprocess",
            stop_on_failure=True,
            clean_previous_outputs=True,
        )

    except KeyboardInterrupt:
        log_warning(
            "Layer 4 pipeline interrupted by user."
        )

    except Exception as error:
        log_error(
            "Unexpected pipeline failure: "
            f"{type(error).__name__}: {error}"
        )

        traceback.print_exc()


if __name__ == "__main__":
    main()
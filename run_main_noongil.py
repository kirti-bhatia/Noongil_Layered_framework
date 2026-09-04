"""Unified SHIVI demonstration runner for Layers 2, 3 and 4.

Place this file in the project root beside ``layer1``, ``layer2``,
``layer3``, ``layer4`` and ``layer1_output_test_scenarios``.  The saved
scenario packet is the Layer 1 output, so the demo begins with Layer 2.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
SCENARIO_ROOT = PROJECT_ROOT / "layer1_output_test_scenarios"
OUTPUT_ROOT = PROJECT_ROOT / "output"
LAYER2_CANONICAL_OUTPUT = OUTPUT_ROOT / "layer2" / "layer2_output.json"
LAYER4_REPORT = OUTPUT_ROOT / "layer4" / "reasoning_pipeline_report.json"
FINAL_EXPLANATION = OUTPUT_ROOT / "layer4" / "explanation_output.json"

LAYER3_STAGES = (
    "layer3/entity extraction/entity_detector.py",
    "layer3/entity extraction/event_detector.py",
    "layer3/entity extraction/relation_detector.py",
    "layer3/graph_memory/graph_builder.py",
    "layer3/graph_memory/graph_updater.py",
    "layer3/graph_memory/graph_queries.py",
    "layer3/episodic_memory/episode_storage.py",
    "layer3/episodic_memory/retrieval.py",
    "layer3/episodic_memory/summarizer.py",
    "layer3/semantic_memory/semantic_extractor.py",
    "layer3/semantic_memory/semantic_store.py",
    "layer3/semantic_memory/semantic_updater.py",
    "layer3/semantic_memory/semantic_retrieval.py",
    "layer3/semantic_memory/semantic_queries.py",
)


class NoongilPipelineError(RuntimeError):
    """Raised when a required Noongil stage cannot complete."""


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise NoongilPipelineError(f"Required output was not created: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise NoongilPipelineError(f"Cannot read JSON output {path}: {error}") from error
    if not isinstance(data, dict):
        raise NoongilPipelineError(f"Expected a JSON object in {path}")
    return data


def run_command(command: list[str], stage: str) -> None:
    print(f"\n[RUN] {stage}")
    completed = subprocess.run(command, 
                               cwd=PROJECT_ROOT, 
                               check=False,
                               capture_output=True,
                               text = True)
    if completed.returncode != 0:
        print("\n [Error Details]")
        print(completed.stderr[-2000:])






         
        raise NoongilPipelineError(
            f"{stage} failed with exit code {completed.returncode}."
        )
    print(f"[PASS] {stage}")


def discover_scenarios() -> list[str]:
    if not SCENARIO_ROOT.is_dir():
        raise NoongilPipelineError(
            f"Scenario folder not found: {SCENARIO_ROOT}\n"
            "Copy layer1_output_test_scenarios beside run_shivi.py."
        )
    scenarios = sorted(
        path.name
        for path in SCENARIO_ROOT.iterdir()
        if path.is_dir() and (path / "layer1_sensor_packet.json").is_file()
    )
    if not scenarios:
        raise NoongilPipelineError(
            f"No folders containing layer1_sensor_packet.json found in {SCENARIO_ROOT}."
        )
    return scenarios


def choose_scenario(requested: str | None) -> str:
    scenarios = discover_scenarios()
    if requested:
        if requested not in scenarios:
            raise NoongilPipelineError(
                f"Unknown scenario {requested!r}. Available: {', '.join(scenarios)}"
            )
        return requested

    print_header("Noongil | AVAILABLE TEST SCENARIOS")
    for number, name in enumerate(scenarios, start=1):
        print(f"{number:>2}. {name}")
    while True:
        answer = input("\nChoose scenario number: ").strip()
        try:
            index = int(answer) - 1
        except ValueError:
            print("Enter a valid number.")
            continue
        if 0 <= index < len(scenarios):
            return scenarios[index]
        print(f"Enter a number from 1 to {len(scenarios)}.")


def run_layer2(scenario: str, mode: str) -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "layer2.run_layer2",
            "--scenario",
            scenario,
            "--mode",
            mode,
        ],
        "Layer 2 — Multimodal Perception",
    )

    scenario_output_dir = OUTPUT_ROOT / "layer2" / "pipeline" / scenario
    candidates = sorted(
        scenario_output_dir.glob("*_layer2_output.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise NoongilPipelineError(
            f"Layer 2 reported success but created no output in {scenario_output_dir}."
        )

    # Layer 3 uses this fixed compatibility path.
    LAYER2_CANONICAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0], LAYER2_CANONICAL_OUTPUT)
    layer2_output = read_json(LAYER2_CANONICAL_OUTPUT)

    # run_layer2.py currently returns zero whenever ready_for_layer3 is true,
    # even if essential perception models failed.  Reject that degraded result
    # here so a demo can never silently continue with an unknown scene.
    errors = layer2_output.get("errors")
    scene = layer2_output.get("scene")
    scene_type = scene.get("type") if isinstance(scene, dict) else None
    if isinstance(errors, list) and errors:
        dependency_errors = [
            str(error) for error in errors if "DEPENDENCY_MISSING" in str(error)
        ]
        if dependency_errors:
            missing = sorted(
                {
                    name
                    for message in dependency_errors
                    for name in (
                        "transformers",
                        "ultralytics",
                        "paddleocr",
                        "paddle",
                        "whisper",
                    )
                    if name in message.lower()
                }
            )
            packages = ", ".join(missing) if missing else "Layer 2 AI packages"
            raise NoongilPipelineError(
                "Layer 2 could not load required AI dependencies: " + packages
            )
    if not scene_type or str(scene_type).lower() == "unknown":
        raise NoongilPipelineError(
            "Layer 2 produced an unknown scene; refusing to send a degraded "
            "result to Layer 3."
        )
    print(f"[READY] Layer 2 output: {LAYER2_CANONICAL_OUTPUT}")


def run_layer3() -> None:
    for relative_path in LAYER3_STAGES:
        stage_file = PROJECT_ROOT / relative_path
        if not stage_file.is_file():
            raise NoongilPipelineError(f"Layer 3 stage is missing: {stage_file}")
        run_command(
            [sys.executable, str(stage_file)],
            f"Layer 3 — {stage_file.stem.replace('_', ' ').title()}",
        )

    context_graph = OUTPUT_ROOT / "layer3" / "context_graph.json"
    read_json(context_graph)
    print(f"[READY] Layer 3 context graph: {context_graph}")


def run_layer4() -> None:
    run_command(
        [sys.executable, "-m", "layer4.pipeline.reasoning_pipeline"],
        "Layer 4 — Reasoning and Decision",
    )
    report = read_json(LAYER4_REPORT)
    if report.get("pipeline_status") != "success":
        failure = report.get("failure_stage") or report.get("errors") or "unknown stage"
        raise NoongilPipelineError(f"Layer 4 pipeline reported failure: {failure}")
    read_json(FINAL_EXPLANATION)
    print(f"[READY] Layer 4 explanation: {FINAL_EXPLANATION}")


def final_message() -> str:
    result = read_json(FINAL_EXPLANATION)
    communication = result.get("communication")
    if isinstance(communication, dict):
        for key in ("user_message", "action_instruction"):
            value = communication.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    value = result.get("user_explanation")
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise NoongilPipelineError(
        "Layer 4 explanation contains no communication.user_message or user_explanation."
    )


def speak(message: str) -> None:
    try:
        import pyttsx3  # type: ignore[import-not-found]
    except ImportError:
        print("[VOICE SKIPPED] Install voice support with: pip install pyttsx3")
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)
        engine.say(message)
        engine.runAndWait()
    except Exception as error:  # Audio devices/backends vary by machine.
        print(f"[VOICE SKIPPED] Text-to-speech error: {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the unified SHIVI prototype.")
    parser.add_argument("--scenario", help="Scenario folder name; prompts if omitted.")
    parser.add_argument(
        "--mode",
        choices=("snapshot", "navigation", "emergency"),
        default="navigation",
        help="Layer 2 operating mode (default: navigation).",
    )
    parser.add_argument("--no-voice", action="store_true", help="Disable spoken output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print_header("Noongil | CONTEXT-AWARE MULTIMODAL ASSISTIVE AI")
    try:
        scenario = choose_scenario(args.scenario)
        print(f"\nSelected scenario: {scenario}")
        run_layer2(scenario, args.mode)
        run_layer3()
        run_layer4()
        message = final_message()
    except (NoongilPipelineError, KeyboardInterrupt) as error:
        print(f"\n[Noongil FAILED] {error}")
        return 1

    print_header("Noongil | FINAL ASSISTANCE RESPONSE")
    print(message)
    if not args.no_voice:
        speak(message)
    print("\n[Noongil SUCCESS] Complete perception-to-reasoning pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

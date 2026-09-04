"""NOONGIL-X: single Layer 3 + Layer 4 inspection pipeline.

Run from the NOONGIL project root:
    python pipeline/noongil_layer3_layer4_pipeline.py

It selects one test scenario, runs every Layer 3 module once, then runs the
existing Layer 4 reasoning pipeline, validates outputs, and prints a compact
summary table.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "pipeline" else Path.cwd()
TEST_DIR = PROJECT_ROOT / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
LAYER2_DIR = OUTPUT_DIR / "layer2"
LAYER3_DIR = OUTPUT_DIR / "layer3"
LAYER4_DIR = OUTPUT_DIR / "layer4"
REPORT_DIR = OUTPUT_DIR / "pipeline_reports"

for directory in (LAYER2_DIR, LAYER3_DIR, LAYER4_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Layer 3 scripts are paths because the folder "entity extraction" contains a space.
LAYER3_STAGES = [
    ("Entity Detection", PROJECT_ROOT / "layer3/entity extraction/entity_detector.py", LAYER3_DIR / "entities.json"),
    ("Event Detection", PROJECT_ROOT / "layer3/entity extraction/event_detector.py", LAYER3_DIR / "events.json"),
    ("Relation Detection", PROJECT_ROOT / "layer3/entity extraction/relation_detector.py", LAYER3_DIR / "relations.json"),
    ("Graph Builder", PROJECT_ROOT / "layer3/graph_memory/graph_builder.py", LAYER3_DIR / "context_graph.json"),
    ("Graph Updater", PROJECT_ROOT / "layer3/graph_memory/graph_updater.py", None),
    ("Graph Queries", PROJECT_ROOT / "layer3/graph_memory/graph_queries.py", None),
    ("Episode Storage", PROJECT_ROOT / "layer3/episodic_memory/episode_storage.py", LAYER3_DIR / "episodic_memory.json"),
    ("Episodic Retrieval", PROJECT_ROOT / "layer3/episodic_memory/retrieval.py", None),
    ("Episodic Summarizer", PROJECT_ROOT / "layer3/episodic_memory/summarizer.py", LAYER3_DIR / "episodic_summary.json"),
    ("Semantic Extractor", PROJECT_ROOT / "layer3/semantic_memory/semantic_extractor.py", LAYER3_DIR / "extracted_semantic_facts.json"),
    ("Semantic Store", PROJECT_ROOT / "layer3/semantic_memory/semantic_store.py", LAYER3_DIR / "semantic_memory.json"),
    ("Semantic Updater", PROJECT_ROOT / "layer3/semantic_memory/semantic_updater.py", LAYER3_DIR / "semantic_memory.json"),
    ("Semantic Retrieval", PROJECT_ROOT / "layer3/semantic_memory/semantic_retrieval.py", None),
    ("Semantic Queries", PROJECT_ROOT / "layer3/semantic_memory/semantic_queries.py", None),
]


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def clean_json_outputs(directory: Path) -> None:
    for path in directory.glob("*.json"):
        try:
            path.unlink()
        except OSError:
            pass


def copy_directory_outputs(source_dir: Path, destination_dir: Path) -> int:
    """Copy all generated files from one output directory into a run archive."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    if not source_dir.exists():
        return copied

    for source_path in source_dir.rglob("*"):
        if not source_path.is_file():
            continue

        relative_path = source_path.relative_to(source_dir)
        destination_path = destination_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        copied += 1

    return copied


def create_run_archive(
    scenario: Path,
    rows: list[dict[str, Any]],
    failed: bool,
    started_at: datetime,
    finished_at: datetime,
) -> Path:
    """Save Layer 3, Layer 4 and pipeline report in one timestamped folder."""
    timestamp = finished_at.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = REPORT_DIR / f"{timestamp}_{scenario.stem}"

    # Avoid overwriting when two runs finish in the same second.
    duplicate_index = 1
    original_run_dir = run_dir
    while run_dir.exists():
        run_dir = Path(f"{original_run_dir}_{duplicate_index}")
        duplicate_index += 1

    layer3_archive = run_dir / "layer3"
    layer4_archive = run_dir / "layer4"
    run_dir.mkdir(parents=True, exist_ok=False)

    layer3_file_count = copy_directory_outputs(LAYER3_DIR, layer3_archive)
    layer4_file_count = copy_directory_outputs(LAYER4_DIR, layer4_archive)

    graph = load_json(LAYER3_DIR / "context_graph.json") or {}
    decision = load_json(LAYER4_DIR / "decision_output.json") or {}

    report_data = {
        "scenario": scenario.name,
        "status": "FAILED" if failed else "PASSED",
        "date": finished_at.strftime("%Y-%m-%d"),
        "time": finished_at.strftime("%H:%M:%S"),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "total_execution_seconds": round(
            (finished_at - started_at).total_seconds(),
            3,
        ),
        "archive": {
            "run_directory": str(run_dir),
            "layer3_files_copied": layer3_file_count,
            "layer4_files_copied": layer4_file_count,
        },
        "summary": {
            "layer3_nodes": graph.get(
                "node_count",
                len(graph.get("nodes", [])),
            ),
            "layer3_edges": graph.get(
                "edge_count",
                len(graph.get("edges", [])),
            ),
            "layer4_decision": (
                decision.get("decision")
                or decision.get("selected_action")
                or decision.get("recommended_action")
                or "Check layer4/decision_output.json"
            ),
            "layer4_confidence": decision.get(
                "confidence",
                decision.get("decision_confidence", "Not standardized"),
            ),
        },
        "stages": rows,
    }

    report_path = run_dir / "pipeline_report.json"
    report_path.write_text(
        json.dumps(report_data, indent=4),
        encoding="utf-8",
    )

    return run_dir


def choose_scenario() -> Path | None:
    files = sorted(TEST_DIR.glob("*.json"))
    if not files:
        print(f"[ERROR] No scenarios found in {TEST_DIR}")
        return None

    print("\nAVAILABLE SCENARIOS")
    print("-" * 62)
    for index, path in enumerate(files, 1):
        print(f"{index:>2}. {path.name}")

    while True:
        raw = input("\nChoose scenario number (0 to cancel): ").strip()
        try:
            number = int(raw)
        except ValueError:
            print("[ERROR] Enter a number.")
            continue
        if number == 0:
            return None
        if 1 <= number <= len(files):
            return files[number - 1]
        print("[ERROR] Choice out of range.")


def prepare_input(scenario: Path) -> None:
    # Current Layer 3 modules read the legacy root path. Keep both copies until
    # all modules are moved to one centralized path configuration.
    shutil.copy2(scenario, OUTPUT_DIR / "layer2_output.json")
    shutil.copy2(scenario, LAYER2_DIR / "layer2_output.json")


def run_process(command: list[str]) -> tuple[bool, float, str]:
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    duration = time.perf_counter() - start
    combined = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, duration, combined.strip()


def validate_expected(path: Path | None) -> tuple[bool, str]:
    if path is None:
        return True, "No fixed output"
    data = load_json(path)
    if data is None:
        return False, "Missing/invalid JSON"
    if isinstance(data, dict):
        return True, f"{len(data)} top-level fields"
    if isinstance(data, list):
        return True, f"{len(data)} items"
    return False, "Unexpected JSON type"


def print_table(rows: list[dict[str, Any]]) -> None:
    print("\nPIPELINE STATUS")
    print("=" * 104)
    print(f"{'Layer':<8} {'Stage':<28} {'Run':<9} {'Output':<10} {'Time':>8}  Details")
    print("-" * 104)
    for row in rows:
        print(
            f"{row['layer']:<8} {row['stage']:<28} {row['run']:<9} "
            f"{row['output']:<10} {row['seconds']:>7.2f}s  {row['details'][:38]}"
        )
    print("=" * 104)


def summarize_outputs() -> None:
    graph = load_json(LAYER3_DIR / "context_graph.json") or {}
    decision = load_json(LAYER4_DIR / "decision_output.json") or {}
    print("\nFINAL OUTPUT SUMMARY")
    print("-" * 62)
    print(f"Layer 3 nodes       : {graph.get('node_count', len(graph.get('nodes', [])))}")
    print(f"Layer 3 edges       : {graph.get('edge_count', len(graph.get('edges', [])))}")
    print(f"Layer 4 decision    : {decision.get('decision') or decision.get('selected_action') or decision.get('recommended_action') or 'Check decision_output.json'}")
    print(f"Layer 4 confidence  : {decision.get('confidence', decision.get('decision_confidence', 'Not standardized'))}")


def main() -> int:
    print("\n" + "=" * 70)
    print("NOONGIL-X | LAYER 3 + LAYER 4 OUTPUT CHECK PIPELINE")
    print("=" * 70)

    scenario = choose_scenario()
    if scenario is None:
        return 0

    started_at = datetime.now()

    clean_json_outputs(LAYER3_DIR)
    clean_json_outputs(LAYER4_DIR)
    prepare_input(scenario)

    rows: list[dict[str, Any]] = []
    failed = False

    for stage_name, script_path, expected_output in LAYER3_STAGES:
        if not script_path.exists():
            rows.append({"layer": "Layer 3", "stage": stage_name, "run": "FAILED", "output": "FAILED", "seconds": 0.0, "details": "Script missing"})
            failed = True
            break

        ok, seconds, logs = run_process([sys.executable, str(script_path)])
        output_ok, details = validate_expected(expected_output)
        rows.append({
            "layer": "Layer 3",
            "stage": stage_name,
            "run": "PASS" if ok else "FAILED",
            "output": "PASS" if output_ok else "FAILED",
            "seconds": seconds,
            "details": details if ok else (logs.splitlines()[-1] if logs else "Execution failed"),
        })
        if not ok or not output_ok:
            failed = True
            break

    if not failed:
        ok, seconds, logs = run_process([sys.executable, "-m", "layer4.pipeline.reasoning_pipeline"])
        report = load_json(LAYER4_DIR / "reasoning_pipeline_report.json") or {}
        rows.append({
            "layer": "Layer 4",
            "stage": "Complete Reasoning Pipeline",
            "run": "PASS" if ok else "FAILED",
            "output": "PASS" if report else "FAILED",
            "seconds": seconds,
            "details": "9 reasoning stages" if ok else (logs.splitlines()[-1] if logs else "Execution failed"),
        })
        failed = not ok or not bool(report)

    print_table(rows)
    summarize_outputs()

    finished_at = datetime.now()
    run_dir = create_run_archive(
        scenario=scenario,
        rows=rows,
        failed=failed,
        started_at=started_at,
        finished_at=finished_at,
    )

    print(f"\nArchived run saved: {run_dir}")
    print(f"Pipeline report     : {run_dir / 'pipeline_report.json'}")
    print(f"Layer 3 outputs     : {run_dir / 'layer3'}")
    print(f"Layer 4 outputs     : {run_dir / 'layer4'}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
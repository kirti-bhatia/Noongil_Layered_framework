"""
============================================================
NOONGIL-X
Accuracy Evaluation Script
============================================================

File:
    scripts/run_accuracy_eval.py

Purpose:
    Runs every scenario in test/*.json through the real
    Layer 3 -> Layer 4 pipeline (same commands used by
    pipeline/noongil_reasoning_test_pipeline.py) and checks
    the final output against an "expected_output" block
    inside each test scenario file.

    This produces a real, defensible accuracy percentage -
    NOT a confidence score. Confidence says "how sure the
    system is." This script measures "was the system right."

------------------------------------------------------------
REQUIRED: add an "expected_output" block to each test/*.json
file. Example (add this key at the top level of e.g.
test/emergency_situation.json):

    "expected_output": {
        "expected_decision_mode": "emergency_response",
        "expected_primary_intent": "seek_emergency_help",
        "expected_hazard_type": "audio_hazard",
        "min_decision_confidence": 0.75
    }

Every key in "expected_output" is OPTIONAL. Only the keys you
include are checked for that scenario - so you can start with
just expected_decision_mode and add more later.

Recognized keys:
    expected_decision_mode      -> matched against
                                   decision_output.json / decision_mode
    expected_primary_intent     -> matched against
                                   intent_reasoning.json / primary_intent
    expected_hazard_type        -> matched against
                                   decision_output.json / highest_priority_hazard / hazard_type
                                   (use null / omit if no hazard expected)
    min_decision_confidence     -> decision_output.json / decision_confidence
                                   must be >= this value

Usage:
    python -m scripts.run_accuracy_eval
    (run from the project root, same as the existing test pipeline)
============================================================
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# PATH CONFIGURATION (mirrors noongil_reasoning_test_pipeline.py)
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
SCRIPTS_DIR = CURRENT_FILE.parent
BASE_DIR = SCRIPTS_DIR.parent

TEST_DIR = BASE_DIR / "test"

OUTPUT_DIR = BASE_DIR / "output"
LAYER2_OUTPUT_DIR = OUTPUT_DIR / "layer2"
LAYER3_OUTPUT_DIR = OUTPUT_DIR / "layer3"
LAYER4_OUTPUT_DIR = OUTPUT_DIR / "layer4"

LAYER2_INPUT_PATH = LAYER2_OUTPUT_DIR / "layer2_output.json"

INTENT_PATH = LAYER4_OUTPUT_DIR / "intent_reasoning.json"
HAZARDS_PATH = LAYER4_OUTPUT_DIR / "hazards.json"
DECISION_PATH = LAYER4_OUTPUT_DIR / "decision_output.json"

EVAL_REPORT_PATH = OUTPUT_DIR / "accuracy_eval_report.json"

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

for directory in (
    OUTPUT_DIR,
    LAYER2_OUTPUT_DIR,
    LAYER3_OUTPUT_DIR,
    LAYER4_OUTPUT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# SMALL HELPERS
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def clear_directory_json_files(directory: Path) -> None:
    if not directory.exists():
        return
    for file_path in directory.glob("*.json"):
        try:
            file_path.unlink()
        except OSError:
            pass


def run_command(title: str, command: List[str]) -> bool:
    result = subprocess.run(command, cwd=str(BASE_DIR), check=False)
    if result.returncode != 0:
        print(f"  [FAIL] {title} exited with code {result.returncode}")
        return False
    return True


# ============================================================
# SCENARIO EXECUTION
# ============================================================

def run_scenario(test_file: Path) -> Dict[str, Any]:
    """
    Runs one scenario through Layer 3 -> Layer 4 and returns
    the raw output needed for scoring.
    """

    clear_directory_json_files(LAYER3_OUTPUT_DIR)
    clear_directory_json_files(LAYER4_OUTPUT_DIR)

    shutil.copy2(test_file, LAYER2_INPUT_PATH)

    if not run_command("Layer 3 pipeline", LAYER3_PIPELINE_COMMAND):
        return {"error": "layer3_pipeline_failed"}

    if not run_command("Layer 4 pipeline", LAYER4_PIPELINE_COMMAND):
        return {"error": "layer4_pipeline_failed"}

    intent = load_json(INTENT_PATH) or {}
    hazards = load_json(HAZARDS_PATH) or {}
    decision = load_json(DECISION_PATH) or {}

    if not decision:
        return {"error": "no_decision_output_produced"}

    return {
        "intent": intent,
        "hazards": hazards,
        "decision": decision,
    }


# ============================================================
# SCORING
# ============================================================

def score_scenario(
    scenario_name: str,
    expected: Dict[str, Any],
    actual: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compares actual pipeline output to the expected_output
    block. Only checks keys present in `expected`.
    """

    checks: List[Dict[str, Any]] = []

    if "error" in actual:
        return {
            "scenario": scenario_name,
            "status": "pipeline_error",
            "error": actual["error"],
            "checks": [],
            "passed": 0,
            "total": 0,
        }

    decision = actual.get("decision", {})
    intent = actual.get("intent", {})

    if "expected_decision_mode" in expected:
        actual_value = decision.get("decision_mode")
        expected_value = expected["expected_decision_mode"]
        checks.append({
            "check": "decision_mode",
            "expected": expected_value,
            "actual": actual_value,
            "passed": actual_value == expected_value,
        })

    if "expected_primary_intent" in expected:
        actual_value = intent.get("primary_intent")
        expected_value = expected["expected_primary_intent"]
        checks.append({
            "check": "primary_intent",
            "expected": expected_value,
            "actual": actual_value,
            "passed": actual_value == expected_value,
        })

    if "expected_hazard_type" in expected:
        highest_hazard = decision.get("highest_priority_hazard") or {}
        actual_value = highest_hazard.get("hazard_type")
        expected_value = expected["expected_hazard_type"]
        checks.append({
            "check": "hazard_type",
            "expected": expected_value,
            "actual": actual_value,
            "passed": actual_value == expected_value,
        })

    if "min_decision_confidence" in expected:
        actual_value = decision.get("decision_confidence", 0.0)
        expected_value = expected["min_decision_confidence"]
        checks.append({
            "check": "min_decision_confidence",
            "expected": f">= {expected_value}",
            "actual": actual_value,
            "passed": actual_value >= expected_value,
        })

    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)

    return {
        "scenario": scenario_name,
        "status": "evaluated" if total > 0 else "no_expected_output",
        "checks": checks,
        "passed": passed,
        "total": total,
    }


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

def run_evaluation() -> Dict[str, Any]:
    print_header("NOONGIL-X ACCURACY EVALUATION")

    test_files = sorted(TEST_DIR.glob("*.json"))

    if not test_files:
        print(f"[ERROR] No test files found in: {TEST_DIR}")
        return {}

    results: List[Dict[str, Any]] = []

    for test_file in test_files:
        scenario_name = test_file.stem
        print(f"\n--- {scenario_name} ---")

        test_data = load_json(test_file)
        if test_data is None:
            print("  [SKIP] Invalid JSON, could not load.")
            results.append({
                "scenario": scenario_name,
                "status": "invalid_json",
                "checks": [],
                "passed": 0,
                "total": 0,
            })
            continue

        expected = test_data.get("expected_output")
        if not expected:
            print("  [SKIP] No 'expected_output' block found in this file yet.")
            results.append({
                "scenario": scenario_name,
                "status": "no_expected_output",
                "checks": [],
                "passed": 0,
                "total": 0,
            })
            continue

        actual = run_scenario(test_file)
        result = score_scenario(scenario_name, expected, actual)
        results.append(result)

        if result["status"] == "pipeline_error":
            print(f"  [ERROR] {result['error']}")
        else:
            for check in result["checks"]:
                mark = "PASS" if check["passed"] else "FAIL"
                print(
                    f"  [{mark}] {check['check']}: "
                    f"expected={check['expected']!r} actual={check['actual']!r}"
                )
            print(f"  -> {result['passed']}/{result['total']} checks passed")

    return summarize(results)


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [r for r in results if r["status"] in ("evaluated", "pipeline_error")]
    skipped = [r for r in results if r["status"] in ("no_expected_output", "invalid_json")]

    total_checks = sum(r["total"] for r in evaluated)
    total_passed = sum(r["passed"] for r in evaluated)

    check_level_accuracy = (
        round(100 * total_passed / total_checks, 1)
        if total_checks > 0
        else None
    )

    scenarios_fully_passed = sum(
        1 for r in evaluated
        if r["total"] > 0 and r["passed"] == r["total"]
    )
    scenario_level_accuracy = (
        round(100 * scenarios_fully_passed / len(evaluated), 1)
        if evaluated
        else None
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_scenarios": len(results),
        "evaluated_scenarios": len(evaluated),
        "skipped_scenarios": len(skipped),
        "skipped_scenario_names": [r["scenario"] for r in skipped],
        "check_level_accuracy_percent": check_level_accuracy,
        "scenario_level_accuracy_percent": scenario_level_accuracy,
        "scenarios_fully_passed": scenarios_fully_passed,
        "results": results,
    }

    print_header("SUMMARY")
    print(f"Scenarios found:              {len(results)}")
    print(f"Scenarios evaluated:          {len(evaluated)}")
    print(f"Scenarios skipped (no ground truth yet): {len(skipped)}")
    if skipped:
        print(f"  -> {', '.join(r['scenario'] for r in skipped)}")
    print()
    if check_level_accuracy is not None:
        print(f"Check-level accuracy:    {check_level_accuracy}%  ({total_passed}/{total_checks} checks)")
        print(f"Scenario-level accuracy: {scenario_level_accuracy}%  ({scenarios_fully_passed}/{len(evaluated)} scenarios fully correct)")
    else:
        print("No scenarios have an 'expected_output' block yet - nothing to score.")
        print("Add expected_output to your test/*.json files, then re-run this script.")

    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nFull report saved to: {EVAL_REPORT_PATH}")

    return report


def main() -> None:
    try:
        run_evaluation()
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted.")
    except Exception as error:
        print(f"[ERROR] Unexpected failure: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
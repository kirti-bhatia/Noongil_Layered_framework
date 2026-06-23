from pathlib import Path
import os
import subprocess
import shutil
import sys

from pathlib import Path

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TEST_DIR = BASE_DIR / "test"

OUTPUT_DIR = BASE_DIR / "output"

LAYER2_OUTPUT_DIR = OUTPUT_DIR / "layer2"

LAYER3_OUTPUT_DIR = OUTPUT_DIR / "layer3"

LAYER4_OUTPUT_DIR = OUTPUT_DIR / "layer4"

LAYER2_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LAYER3_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LAYER4_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# INPUT FILES
# ============================================================

LAYER2_INPUT_NAME = "layer2_output.json"

LAYER2_INPUT_PATH = LAYER2_OUTPUT_DIR / LAYER2_INPUT_NAME



PIPELINE_STEPS = [

    {
        "name": "Entity Detection",
        "file": os.path.join(
            BASE_DIR,
            "layer3",
            "entity extraction",
            "entity_detector.py"
        )
    },

    {
        "name": "Event Detection",
        "file": os.path.join(
            BASE_DIR,
            "layer3",
            "entity extraction",
            "event_detector.py"
        )
    },

    {
        "name": "Relation Detection",
        "file": os.path.join(
            BASE_DIR,
            "layer3",
            "entity extraction",
            "relation_detector.py"
        )
    },

    {
        "name": "Graph Builder",
        "file": os.path.join(
            BASE_DIR,
            "layer3",
            "graph_memory",
            "graph_builder.py"
        )
    },

    {
        "name": "Graph Updater",
        "file": os.path.join(
            BASE_DIR,
            "layer3",
            "graph_memory",
            "graph_updater.py"
        )
    },

    {
        "name": "Graph Queries",
        "file": os.path.join(
            BASE_DIR,
            "layer3",
            "graph_memory",
            "graph_queries.py"
        )
    },

    {
        "name": "Episode Storage",
        "file": os.path.join(
            BASE_DIR,
            "layer3",
            "episodic_memory",
            "episode_storage.py"
        )
    },

    {
        "name": "Episodic Retrieval",
        "file": os.path.join(
            BASE_DIR,
            "layer3",
            "episodic_memory",
            "retrieval.py"
        )
    },

    {
        "name": "Episodic Summarizer",
        "file": os.path.join(
            BASE_DIR,
            "layer3",
            "episodic_memory",
            "summarizer.py"
        )
    },

    {
        "name": "Semantic Extractor",
        "file": os.path.join(
            BASE_DIR,
            "layer3",
            "semantic_memory",
            "semantic_extractor.py"
        )
    },

    {
        "name": "Semantic Store",
        "file": os.path.join(
            BASE_DIR,
            "layer3",
            "semantic_memory",
            "semantic_store.py"
        )
    },

    {
        "name": "Semantic Updater",
        "file": os.path.join(
            BASE_DIR,
            "layer3",
            "semantic_memory",
            "semantic_updater.py"
        )
    },

    {
        "name": "Semantic Retrieval",
        "file": os.path.join(
            BASE_DIR,
            "layer3",
            "semantic_memory",
            "semantic_retrieval.py"
        )
    },

    {
        "name": "Semantic Queries",
        "file": os.path.join(
            BASE_DIR,
            "layer3",
            "semantic_memory",
            "semantic_queries.py"
        )
    }

]





def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def list_test_files():
    files = sorted(TEST_DIR.glob("*.json"))

    print_header("AVAILABLE TEST SITUATIONS")

    if not files:
        print("[ERROR] No JSON test files found in test folder.")
        return []

    for index, file in enumerate(files, start=1):
        print(f"{index}. {file.name}")

    return files


def choose_test_file(files):
    print("\nChoose situation number:")

    try:
        choice = int(input("Enter number: "))
    except ValueError:
        print("[ERROR] Invalid input.")
        return None

    if choice < 1 or choice > len(files):
        print("[ERROR] Choice out of range.")
        return None

    return files[choice - 1]


def prepare_input_file(test_file):
    OUTPUT_DIR.mkdir(exist_ok=True)

    destination = OUTPUT_DIR / LAYER2_INPUT_NAME

    shutil.copy(test_file, destination)

    print(f"\n[INFO] Selected Situation: {test_file.name}")
    print(f"[INFO] Copied To: {destination}")

    return destination


def run_step(step):
    name = step["name"]
    file_path = Path(step["file"])

    print("\n[DEBUG] FILE PATH TYPE:")
    print(type(file_path))

    print("\n[DEBUG] FILE PATH:")
    print(file_path)




    print_header(f"RUNNING: {name}")

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return False

    result = subprocess.run(
        [sys.executable, str(file_path)],
        cwd=BASE_DIR
    )

    if result.returncode != 0:
        print(f"\n[FAILED] {name}")
        print(f"[FAILED FILE] {file_path}")
        return False

    print(f"\n[SUCCESS] {name} completed")
    return True


def run_pipeline():
    print_header("NOONGIL-X LAYER 3 COMPLETE PIPELINE")

    files = list_test_files()

    if not files:
        return

    selected_file = choose_test_file(files)

    if selected_file is None:
        return

    prepare_input_file(selected_file)

    print_header("STARTING LAYER 3 EXECUTION")

    for step in PIPELINE_STEPS:
        success = run_step(step)

        if not success:
            print_header("PIPELINE STOPPED")
            print(f"[FIX REQUIRED] {step['name']}")
            return

    print_header("LAYER 3 PIPELINE COMPLETED SUCCESSFULLY")

    print("[INFO] Final outputs saved inside output folder.")


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
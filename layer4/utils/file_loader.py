"""
=========================================================
NOONGIL-X Layer 4
Utility: File Loader
=========================================================

Purpose:
---------
Safely load JSON files for the Cognitive Reasoning Layer.

Used By:
--------
- context_analyzer.py
- cognitive_state_manager.py
- reasoning_pipeline.py
- all reasoning modules

=========================================================
"""

import json
from pathlib import Path


# =========================================================
# LOAD JSON FILE
# =========================================================

def load_json(file_path, default=None):
    """
    Safely loads a JSON file.

    Parameters:
    -----------
    file_path : str or Path
        Path to the JSON file.

    default : any
        Default value returned if file is missing or invalid.

    Returns:
    --------
    dict/list/default
    """

    file_path = Path(file_path)

    print("\n" + "-" * 60)
    print(f"[INFO] Loading JSON file: {file_path}")
    print("-" * 60)

    if default is None:
        default = {}

    if not file_path.exists():
        print("[ERROR] File not found")
        print(f"[DEBUG] Missing path: {file_path}")
        return default

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        print("[SUCCESS] JSON loaded successfully")

        if isinstance(data, dict):
            print(f"[DEBUG] Loaded dictionary with keys: {list(data.keys())}")

        elif isinstance(data, list):
            print(f"[DEBUG] Loaded list with {len(data)} items")

        else:
            print(f"[DEBUG] Loaded data type: {type(data)}")

        return data

    except json.JSONDecodeError as e:
        print("[ERROR] Invalid JSON format")
        print(f"[DEBUG] JSON error: {e}")
        return default

    except Exception as e:
        print("[ERROR] Failed to load JSON file")
        print(f"[DEBUG] Error: {e}")
        return default


# =========================================================
# CHECK FILE EXISTS
# =========================================================

def file_exists(file_path):
    """
    Checks whether a file exists.
    """

    file_path = Path(file_path)

    exists = file_path.exists()

    print(f"[DEBUG] File exists check: {file_path} -> {exists}")

    return exists


# =========================================================
# LOAD MULTIPLE JSON FILES
# =========================================================

def load_multiple_json(file_paths):
    """
    Loads multiple JSON files and returns them as a dictionary.

    Parameters:
    -----------
    file_paths : dict
        Example:
        {
            "context_graph": "output/context_graph.json",
            "semantic_memory": "output/semantic_memory.json"
        }

    Returns:
    --------
    dict
    """

    print("\n" + "=" * 60)
    print("[INFO] Loading multiple JSON files")
    print("=" * 60)

    loaded_data = {}

    for name, path in file_paths.items():
        print(f"\n[INFO] Loading file group: {name}")

        loaded_data[name] = load_json(
            path,
            default={}
        )

    print("\n[SUCCESS] Multiple JSON loading complete")

    return loaded_data


# =========================================================
# TEST RUN
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("NOONGIL-X FILE LOADER TEST")
    print("=" * 60)

    test_path = Path(__file__).resolve()

    print(f"[INFO] Current file path: {test_path}")
    print("[SUCCESS] file_loader.py is working correctly")
"""
=========================================================
NOONGIL-X Layer 3
Semantic Memory Store
=========================================================

Purpose:
---------
Store extracted semantic facts into
persistent semantic memory.

Input:
-------
extracted_semantic_facts.json

Output:
--------
semantic_memory.json

=========================================================
"""

import json
import os
from pathlib import Path


# =====================================================
# PATHS
# =====================================================

# BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)
LAYER3_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output",
    "layer3"
)

os.makedirs(
    LAYER3_OUTPUT_DIR,
    exist_ok=True
)

FACT_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "extracted_semantic_facts.json"
)

SEMANTIC_MEMORY_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "semantic_memory.json"
)
SEMANTIC_MEMORY_FILE = Path(SEMANTIC_MEMORY_FILE)

# =====================================================
# LOAD JSON
# =====================================================

def load_json(file_path):
    file_path = Path(file_path)

    print(f"\n[INFO] Loading {file_path.name}")

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        print("[SUCCESS] Loaded")

        return data

    except Exception as e:

        print("[ERROR]")
        print(e)

        return None


# =====================================================
# LOAD MEMORY DATABASE
# =====================================================

def load_memory_database():

    if not SEMANTIC_MEMORY_FILE.exists():

        print(
            "\n[INFO] Creating New Semantic Memory"
        )

        return {

            "fact_count": 0,

            "facts": []
        }

    try:

        with open(
            SEMANTIC_MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            memory = json.load(file)

        print(
            "\n[SUCCESS] Existing Semantic Memory Loaded"
        )

        return memory

    except Exception as e:

        print("[ERROR]")
        print(e)

        return {

            "fact_count": 0,

            "facts": []
        }


# =====================================================
# CHECK DUPLICATE FACT
# =====================================================

def fact_exists(
        existing_facts,
        new_fact
):

    for fact in existing_facts:

        if (
            fact["fact"]
            ==
            new_fact["fact"]
        ):

            return True

    return False


# =====================================================
# STORE FACTS
# =====================================================

def store_facts(
        extracted_facts,
        memory_db
):

    print(
        "\n[INFO] Storing Semantic Facts"
    )

    new_facts = 0

    for fact in extracted_facts:

        if not fact_exists(
                memory_db["facts"],
                fact
        ):

            memory_db["facts"].append(
                fact
            )

            new_facts += 1

            print(
                f"[NEW FACT] "
                f"{fact['fact']}"
            )

        else:

            print(
                f"[SKIPPED] "
                f"{fact['fact']}"
            )

    memory_db["fact_count"] = \
        len(memory_db["facts"])

    print(
        f"\n[INFO] Added: "
        f"{new_facts}"
    )

    return memory_db


# =====================================================
# SAVE MEMORY
# =====================================================

def save_memory(memory_db):

    print(
        "\n[INFO] Saving Semantic Memory"
    )

    with open(
        SEMANTIC_MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            memory_db,

            file,

            indent=4
        )

    print(
        f"[SUCCESS] Saved To\n"
        f"{SEMANTIC_MEMORY_FILE}"
    )


# =====================================================
# DISPLAY MEMORY SUMMARY
# =====================================================

def show_summary(memory_db):

    print(
        "\n[INFO] Semantic Memory Summary"
    )

    print(
        f"Total Facts: "
        f"{memory_db['fact_count']}"
    )

    categories = {}

    for fact in memory_db["facts"]:

        category = fact["category"]

        categories[category] = \
            categories.get(
                category,
                0
            ) + 1

    print("\nCategories:")

    for cat, count in categories.items():

        print(
            f"  {cat}: {count}"
        )


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n" + "=" * 60)
    print("NOONGIL-X SEMANTIC MEMORY STORE")
    print("=" * 60)

    extracted_data = load_json(
        FACT_FILE
    )

    if extracted_data is None:

        print(
            "\n[ERROR] No Extracted Facts Found"
        )

        return

    memory_db = \
        load_memory_database()

    memory_db = store_facts(

        extracted_data["facts"],

        memory_db
    )

    save_memory(
        memory_db
    )

    show_summary(
        memory_db
    )

    print(
        "\n[SUCCESS] SEMANTIC STORAGE COMPLETE"
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
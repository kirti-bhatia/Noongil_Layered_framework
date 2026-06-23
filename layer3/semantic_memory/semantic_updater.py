"""
=========================================================
NOONGIL-X Layer 3
Semantic Memory Updater
=========================================================

Purpose:
---------
Update confidence and frequency of
existing semantic facts.

Input:
-------
semantic_memory.json

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
SEMANTIC_MEMORY_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "semantic_memory.json"
)


# =====================================================
# LOAD MEMORY
# =====================================================

def load_memory():

    print("\n[INFO] Loading Semantic Memory")

    try:

        with open(
            SEMANTIC_MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            memory = json.load(file)

        print("[SUCCESS] Loaded")

        return memory

    except Exception as e:

        print("[ERROR]")
        print(e)

        return None


# =====================================================
# UPDATE CONFIDENCE
# =====================================================

def update_confidence(memory):

    print(
        "\n[INFO] Updating Confidence Scores"
    )

    total_facts = memory["fact_count"]

    if total_facts == 0:

        print(
            "[INFO] No Facts Available"
        )

        return memory

    for fact in memory["facts"]:

        frequency = fact.get(
            "frequency",
            1
        )

        reinforcement = fact.get(
            "reinforcement",
            0
        )

        old_confidence = fact.get(
            "confidence",
            0
        )

        # --------------------------------
        # Better Confidence Formula
        # --------------------------------

        confidence = (
            0.3
            +
            (frequency * 0.1)
            +
            (reinforcement * 0.05)
        )

        confidence = min(
            confidence,
            1.0
        )

        confidence = round(
            confidence,
            2
        )

        fact["confidence"] = confidence

        print(
            f"[UPDATED] "
            f"{fact['fact_id']} | "
            f"{old_confidence} -> "
            f"{confidence}"
        )

    return memory

# =====================================================
# REMOVE WEAK FACTS
# =====================================================

def remove_weak_facts(
        memory,
        threshold=0.10
):

    print(
        "\n[INFO] Removing Weak Facts"
    )

    original_count = len(
        memory["facts"]
    )

    filtered_facts = []

    for fact in memory["facts"]:

        if fact["confidence"] >= threshold:

            filtered_facts.append(
                fact
            )

        else:

            print(
                f"[REMOVED] "
                f"{fact['fact_id']}"
            )

    memory["facts"] = filtered_facts

    memory["fact_count"] = len(
        filtered_facts
    )

    removed = (

        original_count

        -

        len(filtered_facts)
    )

    print(
        f"[INFO] Removed: "
        f"{removed}"
    )

    return memory


# =====================================================
# FACT SUMMARY
# =====================================================

def display_summary(memory):

    print(
        "\n[INFO] Semantic Memory Summary"
    )

    print(
        f"Total Facts: "
        f"{memory['fact_count']}"
    )

    for fact in memory["facts"]:

        print(
            f"{fact['fact_id']} | "
            f"{fact['confidence']}"
        )


# =====================================================
# SAVE MEMORY
# =====================================================

def save_memory(memory):

    print(
        "\n[INFO] Saving Memory"
    )

    with open(
        SEMANTIC_MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            memory,
            file,
            indent=4
        )

    print(
        f"[SUCCESS] Saved To\n"
        f"{SEMANTIC_MEMORY_FILE}"
    )


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n" + "=" * 60)
    print("NOONGIL-X SEMANTIC MEMORY UPDATER")
    print("=" * 60)

    memory = load_memory()

    if memory is None:

        return

    memory = update_confidence(
        memory
    )

    memory = remove_weak_facts(
        memory,
        threshold=0.10
    )

    display_summary(
        memory
    )

    save_memory(
        memory
    )

    print(
        "\n[SUCCESS] SEMANTIC UPDATE COMPLETE"
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
"""
=========================================================
NOONGIL-X Layer 3
Semantic Memory Retrieval
=========================================================

Purpose:
---------
Retrieve learned knowledge from semantic memory.

Input:
-------
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

        print("[SUCCESS] Semantic Memory Loaded")

        return memory

    except Exception as e:

        print("[ERROR]")
        print(e)

        return None


# =====================================================
# DISPLAY FACT
# =====================================================

def print_fact(fact):

    print("\n" + "-" * 50)

    print(
        f"Fact ID     : "
        f"{fact['fact_id']}"
    )

    print(
        f"Category    : "
        f"{fact['category']}"
    )

    print(
        f"Fact        : "
        f"{fact['fact']}"
    )

    print(
        f"Frequency   : "
        f"{fact['frequency']}"
    )

    print(
        f"Confidence  : "
        f"{fact['confidence']}"
    )

    print("-" * 50)


# =====================================================
# ALL FACTS
# =====================================================

def get_all_facts(memory):

    print("\n[QUERY] All Facts")

    facts = memory["facts"]

    print(
        f"[INFO] Total Facts: "
        f"{len(facts)}"
    )

    for fact in facts:

        print_fact(fact)

    return facts


# =====================================================
# CATEGORY SEARCH
# =====================================================

def search_by_category(
        memory,
        category
):

    print(
        f"\n[QUERY] Category = "
        f"{category}"
    )

    results = []

    for fact in memory["facts"]:

        if (
            fact["category"]
            .lower()
            ==
            category.lower()
        ):

            results.append(fact)

    print(
        f"[INFO] Matches: "
        f"{len(results)}"
    )

    return results


# =====================================================
# KEYWORD SEARCH
# =====================================================

def search_by_keyword(
        memory,
        keyword
):

    print(
        f"\n[QUERY] Keyword = "
        f"{keyword}"
    )

    results = []

    for fact in memory["facts"]:

        if (
            keyword.lower()
            in
            fact["fact"].lower()
        ):

            results.append(fact)

    print(
        f"[INFO] Matches: "
        f"{len(results)}"
    )

    return results


# =====================================================
# HIGH CONFIDENCE FACTS
# =====================================================

def get_high_confidence_facts(
        memory,
        threshold=0.7
):

    print(
        f"\n[QUERY] Confidence >= "
        f"{threshold}"
    )

    results = []

    for fact in memory["facts"]:

        if (
            fact["confidence"]
            >= threshold
        ):

            results.append(fact)

    print(
        f"[INFO] Matches: "
        f"{len(results)}"
    )

    return results


# =====================================================
# FACT BY ID
# =====================================================

def get_fact_by_id(
        memory,
        fact_id
):

    print(
        f"\n[QUERY] Fact ID = "
        f"{fact_id}"
    )

    for fact in memory["facts"]:

        if fact["fact_id"] == fact_id:

            print_fact(fact)

            return fact

    print("[INFO] Fact Not Found")

    return None


# =====================================================
# MEMORY SUMMARY
# =====================================================

def memory_summary(memory):

    print("\n[INFO] Semantic Memory Summary")

    print(
        f"Total Facts: "
        f"{memory['fact_count']}"
    )

    categories = {}

    for fact in memory["facts"]:

        category = fact["category"]

        categories[category] = (
            categories.get(
                category,
                0
            ) + 1
        )

    print("\nCategories:")

    for category, count in categories.items():

        print(
            f"{category}: {count}"
        )


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n" + "=" * 60)
    print("NOONGIL-X SEMANTIC MEMORY RETRIEVAL")
    print("=" * 60)

    memory = load_memory()

    if memory is None:

        return

    memory_summary(memory)

    get_all_facts(memory)

    search_by_category(
        memory,
        "behavior"
    )

    search_by_keyword(
        memory,
        "navigation"
    )

    get_high_confidence_facts(
        memory,
        threshold=0.7
    )

    print(
        "\n[SUCCESS] RETRIEVAL COMPLETE"
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
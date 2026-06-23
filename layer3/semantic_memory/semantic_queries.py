"""
=========================================================
NOONGIL-X Layer 3
Semantic Memory Query Engine
=========================================================

Purpose:
---------
Advanced querying of semantic memory.

Supported Queries:
------------------
1. Search by Category
2. Search by Keyword
3. Search by Confidence
4. Search by Frequency
5. Search by Fact ID
6. Top-K Facts
7. Statistics

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

        print("[SUCCESS] Memory Loaded")

        return memory

    except Exception as e:

        print("[ERROR]")
        print(e)

        return None


# =====================================================
# DISPLAY FACT
# =====================================================

def display_fact(fact):

    print("\n" + "-" * 50)

    print(
        f"Fact ID      : "
        f"{fact['fact_id']}"
    )

    print(
        f"Category     : "
        f"{fact['category']}"
    )

    print(
        f"Fact         : "
        f"{fact['fact']}"
    )

    print(
        f"Frequency    : "
        f"{fact.get('frequency',0)}"
    )

    print(
        f"Confidence   : "
        f"{fact.get('confidence',0)}"
    )

    print(
        f"Reinforcement: "
        f"{fact.get('reinforcement_score',0)}"
    )

    print("-" * 50)


# =====================================================
# QUERY BY CATEGORY
# =====================================================

def query_by_category(
        memory,
        category
):

    print(
        f"\n[QUERY] Category = "
        f"{category}"
    )

    matches = []

    for fact in memory["facts"]:

        if (
            fact["category"]
            .lower()
            ==
            category.lower()
        ):

            matches.append(
                fact
            )

    print(
        f"[INFO] Matches: "
        f"{len(matches)}"
    )

    return matches


# =====================================================
# QUERY BY KEYWORD
# =====================================================

def query_by_keyword(
        memory,
        keyword
):

    print(
        f"\n[QUERY] Keyword = "
        f"{keyword}"
    )

    matches = []

    for fact in memory["facts"]:

        if (
            keyword.lower()
            in
            fact["fact"].lower()
        ):

            matches.append(
                fact
            )

    print(
        f"[INFO] Matches: "
        f"{len(matches)}"
    )

    return matches


# =====================================================
# QUERY BY CONFIDENCE
# =====================================================

def query_by_confidence(
        memory,
        threshold
):

    print(
        f"\n[QUERY] Confidence >= "
        f"{threshold}"
    )

    matches = []

    for fact in memory["facts"]:

        if (
            fact.get(
                "confidence",
                0
            )
            >= threshold
        ):

            matches.append(
                fact
            )

    print(
        f"[INFO] Matches: "
        f"{len(matches)}"
    )

    return matches


# =====================================================
# QUERY BY FREQUENCY
# =====================================================

def query_by_frequency(
        memory,
        threshold
):

    print(
        f"\n[QUERY] Frequency >= "
        f"{threshold}"
    )

    matches = []

    for fact in memory["facts"]:

        if (
            fact.get(
                "frequency",
                0
            )
            >= threshold
        ):

            matches.append(
                fact
            )

    print(
        f"[INFO] Matches: "
        f"{len(matches)}"
    )

    return matches


# =====================================================
# QUERY BY FACT ID
# =====================================================

def query_by_fact_id(
        memory,
        fact_id
):

    print(
        f"\n[QUERY] Fact ID = "
        f"{fact_id}"
    )

    for fact in memory["facts"]:

        if (
            fact["fact_id"]
            ==
            fact_id
        ):

            return fact

    return None


# =====================================================
# TOP K FACTS
# =====================================================

def get_top_facts(
        memory,
        k=5
):

    print(
        f"\n[QUERY] Top {k} Facts"
    )

    facts = sorted(

        memory["facts"],

        key=lambda x:
        x.get(
            "confidence",
            0
        ),

        reverse=True
    )

    return facts[:k]


# =====================================================
# MEMORY STATISTICS
# =====================================================

def memory_statistics(memory):

    print("\n" + "=" * 50)

    print(
        "SEMANTIC MEMORY STATISTICS"
    )

    print("=" * 50)

    print(
        f"Total Facts : "
        f"{memory['fact_count']}"
    )

    categories = {}

    for fact in memory["facts"]:

        category = fact["category"]

        categories[category] = \
            categories.get(
                category,
                0
            ) + 1

    print("\nCategories")

    for cat, count in categories.items():

        print(
            f"{cat} : {count}"
        )


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n" + "=" * 60)

    print(
        "NOONGIL-X "
        "SEMANTIC QUERY ENGINE"
    )

    print("=" * 60)

    memory = load_memory()

    if memory is None:

        return

    memory_statistics(memory)

    # ----------------------------------
    # Example Queries
    # ----------------------------------

    results = query_by_keyword(
        memory,
        "navigation"
    )

    for fact in results:

        display_fact(fact)

    top_facts = get_top_facts(
        memory,
        k=3
    )

    print(
        f"\n[INFO] Top Facts Found: "
        f"{len(top_facts)}"
    )

    for fact in top_facts:

        display_fact(fact)

    print(
        "\n[SUCCESS] QUERY ENGINE COMPLETE"
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
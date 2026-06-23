"""
=========================================================
NOONGIL-X Layer 3
Semantic Memory Extractor
=========================================================

Purpose:
---------
Extract semantic facts from episodic memory.

Input:
-------
episodic_memory.json

Output:
--------
extracted_semantic_facts.json

=========================================================
"""

import json
import os
import uuid

from pathlib import Path
from collections import Counter


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

MEMORY_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "episodic_memory.json"
)

OUTPUT_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "extracted_semantic_facts.json"
)

# =====================================================
# LOAD MEMORY
# =====================================================

def load_memory():

    print("\n[INFO] Loading Episodic Memory")

    try:

        with open(
            MEMORY_FILE,
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
# FACT ID
# =====================================================

def generate_fact_id():

    return "SEM_" + str(uuid.uuid4())[:8]


# =====================================================
# CONFIDENCE
# =====================================================

def calculate_confidence(
        frequency,
        total_episodes
):

    if total_episodes == 0:

        return 0.0

    confidence = (
        frequency /
        total_episodes
    )

    return round(
        confidence,
        2
    )


# =====================================================
# EVENT PATTERNS
# =====================================================

def extract_event_patterns(
        episodes
):

    print(
        "\n[INFO] Extracting Event Patterns"
    )

    facts = []

    event_counter = Counter()

    for episode in episodes:

        for event in episode["events"]:

            event_counter[event] += 1

    total_episodes = len(episodes)

    for event, count in event_counter.items():

        if count >= 2:

            confidence = calculate_confidence(

                count,

                total_episodes
            )

            fact_text = (

                f"User frequently engages in "
                f"{event} activities."
            )

            fact = {

                "fact_id":
                generate_fact_id(),

                "category":
                "behavior",

                "fact":
                fact_text,

                "frequency":
                count,

                "confidence":
                confidence
            }

            facts.append(fact)

            print(
                f"[FACT] {fact_text}"
            )

    return facts


# =====================================================
# LOCATION PATTERNS
# =====================================================

def extract_location_patterns(
        episodes
):

    print(
        "\n[INFO] Extracting Location Patterns"
    )

    facts = []

    location_counter = Counter()

    for episode in episodes:

        location_counter[
            episode["scene"]
        ] += 1

    total_episodes = len(episodes)

    for location, count in location_counter.items():

        if count >= 2:

            confidence = calculate_confidence(

                count,

                total_episodes
            )

            fact_text = (

                f"{location} is a frequently "
                f"observed environment."
            )

            fact = {

                "fact_id":
                generate_fact_id(),

                "category":
                "location",

                "fact":
                fact_text,

                "frequency":
                count,

                "confidence":
                confidence
            }

            facts.append(fact)

            print(
                f"[FACT] {fact_text}"
            )

    return facts


# =====================================================
# ENTITY PATTERNS
# =====================================================

def extract_entity_patterns(
        episodes
):

    print(
        "\n[INFO] Extracting Entity Patterns"
    )

    facts = []

    entity_counter = Counter()

    for episode in episodes:

        for entity in episode["entities"]:

            entity_counter[entity] += 1

    total_episodes = len(episodes)

    for entity, count in entity_counter.items():

        if count >= 2:

            confidence = calculate_confidence(

                count,

                total_episodes
            )

            fact_text = (

                f"{entity} frequently appears "
                f"in user experiences."
            )

            fact = {

                "fact_id":
                generate_fact_id(),

                "category":
                "entity",

                "fact":
                fact_text,

                "frequency":
                count,

                "confidence":
                confidence
            }

            facts.append(fact)

            print(
                f"[FACT] {fact_text}"
            )

    return facts


# =====================================================
# SAVE FACTS
# =====================================================

def save_facts(facts):

    print("\n[INFO] Saving Facts")

    output = {

        "fact_count":
        len(facts),

        "facts":
        facts
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print(
        f"[SUCCESS] Saved To\n"
        f"{OUTPUT_FILE}"
    )


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n" + "=" * 60)
    print("NOONGIL-X SEMANTIC EXTRACTOR")
    print("=" * 60)

    memory = load_memory()

    if memory is None:

        return

    episodes = memory["episodes"]

    semantic_facts = []

    semantic_facts.extend(

        extract_event_patterns(
            episodes
        )
    )

    semantic_facts.extend(

        extract_location_patterns(
            episodes
        )
    )

    semantic_facts.extend(

        extract_entity_patterns(
            episodes
        )
    )

    print(
        f"\n[INFO] Total Facts: "
        f"{len(semantic_facts)}"
    )

    save_facts(
        semantic_facts
    )

    print(
        "\n[SUCCESS] SEMANTIC EXTRACTION COMPLETE"
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
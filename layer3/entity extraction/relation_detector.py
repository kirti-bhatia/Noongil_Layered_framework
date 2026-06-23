"""
=====================================================
NOONGIL-X Layer 3
Relation Detector
=====================================================

Purpose:
---------
Creates graph relationships between entities.

Input:
-------
detected_entities.json

Output:
--------
detected_relations.json
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

# ============================================================
# INPUT FILE
# ============================================================

INPUT_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "entities.json"
)

# ============================================================
# OUTPUT FILE
# ============================================================

OUTPUT_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "relations.json"
)

# =====================================================
# LOAD ENTITY FILE
# =====================================================

def load_entities():

    print("\n[INFO] Loading Entity File")

    try:

        with open(INPUT_FILE, "r", encoding="utf-8") as file:

            data = json.load(file)

        print("[SUCCESS] Entity File Loaded")

        return data

    except Exception as e:

        print("[ERROR] Failed Loading Entity File")
        print(e)

        return None


# =====================================================
# CREATE RELATION
# =====================================================

def create_relation(source, relation, target):

    return {
        "source": source,
        "relation": relation,
        "target": target
    }


# =====================================================
# RELATION EXTRACTION
# =====================================================

def extract_relations(entity_data):

    print("\n[INFO] Extracting Relations")

    relations = []

    entities = entity_data["entities"]

    locations = []
    objects = []
    activities = []
    nouns = []

    # ------------------------------------------
    # Categorize entities
    # ------------------------------------------

    for entity in entities:

        entity_type = entity["type"]

        if entity_type == "location":
            locations.append(entity["name"])

        elif entity_type == "object":
            objects.append(entity["name"])

        elif entity_type == "activity":
            activities.append(entity["name"])

        elif entity_type == "noun":
            nouns.append(entity["name"])

    print(f"[INFO] Locations Found: {locations}")
    print(f"[INFO] Objects Found: {objects}")
    print(f"[INFO] Activities Found: {activities}")
    print(f"[INFO] Nouns Found: {nouns}")

    # ------------------------------------------
    # USER LOCATED IN LOCATION
    # ------------------------------------------

    for location in locations:

        relation = create_relation(
            "user",
            "located_in",
            location
        )

        relations.append(relation)

        print(
            f"[RELATION] user -> located_in -> {location}"
        )

    # ------------------------------------------
    # USER PERFORMING ACTIVITY
    # ------------------------------------------

    for activity in activities:

        relation = create_relation(
            "user",
            "performing",
            activity
        )

        relations.append(relation)

        print(
            f"[RELATION] user -> performing -> {activity}"
        )

    # ------------------------------------------
    # OBJECTS INSIDE LOCATION
    # ------------------------------------------

    if len(locations) > 0:

        location = locations[0]

        for obj in objects:

            relation = create_relation(
                obj,
                "inside",
                location
            )

            relations.append(relation)

            print(
                f"[RELATION] {obj} -> inside -> {location}"
            )

    # ------------------------------------------
    # USER REQUESTING NOUN
    # ------------------------------------------

    for noun in nouns:

        relation = create_relation(
            "user",
            "requesting",
            noun
        )

        relations.append(relation)

        print(
            f"[RELATION] user -> requesting -> {noun}"
        )

    return relations


# =====================================================
# SAVE OUTPUT
# =====================================================

def save_relations(relations):

    print("\n[INFO] Saving Relation File")

    output = {
        "relation_count": len(relations),
        "relations": relations
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
        f"[SUCCESS] Saved To {OUTPUT_FILE}"
    )


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n" + "=" * 60)
    print("NOONGIL-X RELATION DETECTOR")
    print("=" * 60)

    entity_data = load_entities()

    if entity_data is None:
        return

    relations = extract_relations(entity_data)

    print(
        f"\n[INFO] Total Relations Created: {len(relations)}"
    )

    save_relations(relations)

    print(
        "\n[SUCCESS] RELATION DETECTION COMPLETE"
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
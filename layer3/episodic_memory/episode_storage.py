"""
=========================================================
NOONGIL-X Layer 3
Episodic Memory Store
=========================================================

Purpose:
---------
Convert current perception into an episodic memory.

Input:
-------
detected_entities.json
detected_relations.json
detected_events.json

Output:
--------
episodic_memory.json

=========================================================
"""

import json
import os
# import os
import uuid

from pathlib import Path
from datetime import datetime


# =====================================================
# PATHS
# =====================================================

import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

LAYER3_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "layer3"
)

os.makedirs(
    LAYER3_OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# INPUT FILES
# ============================================================

ENTITY_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "entities.json"
)

RELATION_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "relations.json"
)

EVENT_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "events.json"
)

# ============================================================
# MEMORY FILE
# ============================================================

# MEMORY_FILE = os.path.join(
#     LAYER3_OUTPUT_DIR,
#     "episodic_memory.json"
# )
MEMORY_FILE = Path(
    os.path.join(
        LAYER3_OUTPUT_DIR,
        "episodic_memory.json"
    )
)
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
# EVENT WEIGHTS
# =====================================================

EVENT_WEIGHTS = {

    "emergency": 1.0,

    "navigation_request": 0.8,

    "meeting": 0.8,

    "learning": 0.7,

    "shopping": 0.5,

    "movement": 0.3,
    "conversation_event": 0.4,

    "unknown": 0.1
}


# =====================================================
# IMPORTANCE SCORE
# =====================================================

def calculate_importance_score(
        entities,
        relations,
        events
):

    print(
        "\n[INFO] Calculating Importance Score"
    )

    # -------------------------------------
    # Event Score
    # -------------------------------------

    if len(events) > 0:

        scores = []

        for event in events:

            event_type = event["event_type"]

            score = EVENT_WEIGHTS.get(
                event_type,
                0.1
            )

            scores.append(score)

        # event_score = (
        #     sum(scores)
        #     /
        #     len(scores)
        # )
        event_score = max(scores)

    else:

        event_score = 0

    # -------------------------------------
    # Entity Score
    # -------------------------------------

    entity_score = min(

        len(entities) / 10,

        1.0
    )

    # -------------------------------------
    # Relation Score
    # -------------------------------------

    relation_score = min(

        len(relations) / 10,

        1.0
    )

    # -------------------------------------
    # Interaction Score
    # -------------------------------------

    interaction_score = 0

    # for event in events:

    #     if event["event_type"] == \
    #             "navigation_request":

    #         interaction_score = 0.5

    #         break
    for event in events:
        event_type = event["event_type"]

        if event_type == "emergency":
            interaction_score = 1.0
            break
        elif event_type == "navigation_request":
            interaction_score = 0.5

    # -------------------------------------
    # Final Formula
    # -------------------------------------

    importance = (

        (event_score * 0.5)

        +

        (entity_score * 0.15)

        +

        (relation_score * 0.15)

        +

        (interaction_score * 0.2)
    )

    importance = round(
        importance,
        2
    )

    print(
        f"[INFO] Importance Score: "
        f"{importance}"
    )

    return importance


# =====================================================
# SCENE EXTRACTION
# =====================================================

def extract_scene(entities):

    print(
        "\n[INFO] Determining Scene"
    )

    for entity in entities:

        if entity["type"] == \
                "location":

            print(
                f"[INFO] Scene: "
                f"{entity['name']}"
            )

            return entity["name"]

    return "unknown"


# =====================================================
# CREATE EPISODE
# =====================================================

def create_episode(
        entity_data,
        relation_data,
        event_data
):

    print(
        "\n[INFO] Creating Episode"
    )

    entities = entity_data["entities"]

    relations = relation_data["relations"]

    events = event_data["events"]

    episode = {

        "episode_id":

        "EPI_" +

        str(uuid.uuid4())[:8],

        "timestamp":

        datetime.now().isoformat(),

        "scene":

        extract_scene(entities),

        "entities":

        [

            entity["name"]

            for entity in entities
        ],

        "relations":

        [

            f"{r['source']} "

            f"{r['relation']} "

            f"{r['target']}"

            for r in relations
        ],

        "events":

        [

            event["event_type"]

            for event in events
        ],

        "importance_score":

        calculate_importance_score(

            entities,

            relations,

            events
        )
    }

    print(
        f"[SUCCESS] Episode Created: "
        f"{episode['episode_id']}"
    )

    return episode


# =====================================================
# LOAD MEMORY DATABASE
# =====================================================

def load_memory_database():

    if not MEMORY_FILE.exists():

        print(
            "\n[INFO] Creating New "
            "Memory Database"
        )

        return {

            "episode_count": 0,

            "episodes": []
        }

    try:

        with open(
            MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except:

        return {

            "episode_count": 0,

            "episodes": []
        }


# =====================================================
# SAVE MEMORY
# =====================================================

def save_episode(episode):

    memory_db = \
        load_memory_database()

    memory_db["episodes"].append(
        episode
    )

    memory_db["episode_count"] = \
        len(memory_db["episodes"])

    with open(
        MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            memory_db,

            file,

            indent=4
        )

    print(
        f"[SUCCESS] Episode Stored"
    )

    print(
        f"[INFO] Total Episodes: "
        f"{memory_db['episode_count']}"
    )


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n" + "=" * 60)
    print("NOONGIL-X EPISODIC MEMORY STORE")
    print("=" * 60)

    entity_data = \
        load_json(ENTITY_FILE)

    relation_data = \
        load_json(RELATION_FILE)

    event_data = \
        load_json(EVENT_FILE)

    if (
        entity_data is None
        or relation_data is None
        or event_data is None
    ):

        print(
            "\n[ERROR] Missing Input Files"
        )

        return

    episode = create_episode(

        entity_data,

        relation_data,

        event_data
    )

    save_episode(
        episode
    )

    print(
        "\n[SUCCESS] EPISODIC MEMORY STORED"
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
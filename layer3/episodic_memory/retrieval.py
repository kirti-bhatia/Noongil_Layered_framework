"""
=========================================================
NOONGIL-X Layer 3
Episodic Memory Retriever
=========================================================

Purpose:
---------
Retrieve memories from episodic memory storage.

Queries:
---------
1. Retrieve all episodes
2. Retrieve latest episode
3. Retrieve by scene
4. Retrieve by event
5. Retrieve important episodes
6. Retrieve by entity

=========================================================
"""

import json

import os
from pathlib import Path


# =====================================================
# PATHS
# =====================================================

import os
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
# MEMORY FILE
# ============================================================

MEMORY_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "episodic_memory.json"
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

        print("[ERROR] Failed Loading Memory")

        print(e)

        return None


# =====================================================
# PRINT EPISODE
# =====================================================

def print_episode(episode):

    print("\n" + "-" * 50)

    print(
        f"Episode ID : "
        f"{episode['episode_id']}"
    )

    print(
        f"Timestamp  : "
        f"{episode['timestamp']}"
    )

    print(
        f"Scene      : "
        f"{episode['scene']}"
    )

    print(
        f"Importance : "
        f"{episode['importance_score']}"
    )

    print(
        f"Events     : "
        f"{episode['events']}"
    )

    print(
        f"Entities   : "
        f"{episode['entities']}"
    )

    print("-" * 50)


# =====================================================
# ALL EPISODES
# =====================================================

def get_all_episodes(memory):

    print("\n[QUERY] All Episodes")

    episodes = memory["episodes"]

    print(
        f"[INFO] Total Episodes: "
        f"{len(episodes)}"
    )

    return episodes


# =====================================================
# LATEST EPISODE
# =====================================================

def get_latest_episode(memory):

    print("\n[QUERY] Latest Episode")

    if not memory["episodes"]:

        print("[INFO] No Episodes Found")

        return None

    latest = memory["episodes"][-1]

    print_episode(latest)

    return latest


# =====================================================
# SEARCH BY SCENE
# =====================================================

def get_episodes_by_scene(
        memory,
        scene_name
):

    print(
        f"\n[QUERY] Scene = "
        f"{scene_name}"
    )

    results = []

    for episode in memory["episodes"]:

        if (
            episode["scene"]
            .lower()
            ==
            scene_name.lower()
        ):

            results.append(
                episode
            )

    print(
        f"[INFO] Matches: "
        f"{len(results)}"
    )

    return results


# =====================================================
# SEARCH BY EVENT
# =====================================================

def get_episodes_by_event(
        memory,
        event_type
):

    print(
        f"\n[QUERY] Event = "
        f"{event_type}"
    )

    results = []

    for episode in memory["episodes"]:

        if event_type in \
                episode["events"]:

            results.append(
                episode
            )

    print(
        f"[INFO] Matches: "
        f"{len(results)}"
    )

    return results


# =====================================================
# SEARCH BY ENTITY
# =====================================================

def get_episodes_by_entity(
        memory,
        entity_name
):

    print(
        f"\n[QUERY] Entity = "
        f"{entity_name}"
    )

    results = []

    for episode in memory["episodes"]:

        entities = [

            entity.lower()

            for entity in
            episode["entities"]
        ]

        if (
            entity_name.lower()
            in entities
        ):

            results.append(
                episode
            )

    print(
        f"[INFO] Matches: "
        f"{len(results)}"
    )

    return results


# =====================================================
# IMPORTANT MEMORIES
# =====================================================

def get_important_episodes(
        memory,
        threshold=0.5
):

    print(
        f"\n[QUERY] Importance > "
        f"{threshold}"
    )

    results = []

    for episode in memory["episodes"]:

        if (
            episode[
                "importance_score"
            ]
            >= threshold
        ):

            results.append(
                episode
            )

    print(
        f"[INFO] Matches: "
        f"{len(results)}"
    )

    return results


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n" + "=" * 60)

    print(
        "NOONGIL-X "
        "EPISODIC MEMORY RETRIEVER"
    )

    print("=" * 60)

    memory = load_memory()

    if memory is None:

        return

    # -----------------------------------------
    # Example Queries
    # -----------------------------------------

    latest = get_latest_episode(
        memory
    )

    scene_results = \
        get_episodes_by_scene(
            memory,
            "park"
        )

    event_results = \
        get_episodes_by_event(
            memory,
            "navigation_request"
        )

    entity_results = \
        get_episodes_by_entity(
            memory,
            "gate"
        )

    important_results = \
        get_important_episodes(
            memory,
            threshold=0.5
        )

    print(
        "\n[SUCCESS] "
        "MEMORY RETRIEVAL COMPLETE"
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
"""
=========================================================
NOONGIL-X Layer 3
Episodic Memory Summarizer
=========================================================

Purpose:
---------
Generate high-level summaries from episodic memory.

Input:
-------
episodic_memory.json

Output:
--------
episodic_summary.json

=========================================================
"""

import json

import os
from pathlib import Path
from collections import Counter


# =====================================================
# PATHS
# =====================================================
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
# FILES
# ============================================================

MEMORY_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "episodic_memory.json"
)

SUMMARY_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "episodic_summary.json"
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
# SCENE STATISTICS
# =====================================================

def analyze_scenes(episodes):

    print("\n[INFO] Analyzing Scenes")

    scenes = []

    for episode in episodes:

        scenes.append(
            episode["scene"]
        )

    counter = Counter(scenes)

    print(
        f"[INFO] Unique Scenes: "
        f"{len(counter)}"
    )

    return dict(counter)


# =====================================================
# EVENT STATISTICS
# =====================================================

def analyze_events(episodes):

    print("\n[INFO] Analyzing Events")

    events = []

    for episode in episodes:

        for event in episode["events"]:

            events.append(event)

    counter = Counter(events)

    return dict(counter)


# =====================================================
# ENTITY STATISTICS
# =====================================================

def analyze_entities(episodes):

    print("\n[INFO] Analyzing Entities")

    entities = []

    for episode in episodes:

        for entity in episode["entities"]:

            entities.append(entity)

    counter = Counter(entities)

    return dict(counter)


# =====================================================
# IMPORTANT MEMORIES
# =====================================================

def get_top_memories(episodes):

    print(
        "\n[INFO] Finding Important Memories"
    )

    sorted_episodes = sorted(

        episodes,

        key=lambda x:
        x["importance_score"],

        reverse=True
    )

    top_memories = sorted_episodes[:5]

    print(
        f"[INFO] Top Memories: "
        f"{len(top_memories)}"
    )

    return top_memories


# =====================================================
# GENERATE TEXT SUMMARY
# =====================================================

def generate_summary_text(
        scene_stats,
        event_stats,
        entity_stats,
        total_episodes
):

    print(
        "\n[INFO] Generating Summary Text"
    )

    most_common_scene = "unknown"

    if scene_stats:

        most_common_scene = max(
            scene_stats,
            key=scene_stats.get
        )

    most_common_event = "unknown"

    if event_stats:

        most_common_event = max(
            event_stats,
            key=event_stats.get
        )

    most_common_entity = "unknown"

    if entity_stats:

        most_common_entity = max(
            entity_stats,
            key=entity_stats.get
        )

    summary = (

        f"NOONGIL has stored "

        f"{total_episodes} episodic memories. "

        f"The most common scene is "

        f"'{most_common_scene}'. "

        f"The most frequent event is "

        f"'{most_common_event}'. "

        f"The most observed entity is "

        f"'{most_common_entity}'."
    )

    return summary


# =====================================================
# SAVE SUMMARY
# =====================================================

def save_summary(summary_data):

    print("\n[INFO] Saving Summary")

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            summary_data,

            file,

            indent=4
        )

    print(
        f"[SUCCESS] Saved To\n"
        f"{SUMMARY_FILE}"
    )


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n" + "=" * 60)
    print("NOONGIL-X EPISODIC MEMORY SUMMARIZER")
    print("=" * 60)

    memory = load_memory()

    if memory is None:

        return

    episodes = memory["episodes"]

    scene_stats = analyze_scenes(
        episodes
    )

    event_stats = analyze_events(
        episodes
    )

    entity_stats = analyze_entities(
        episodes
    )

    top_memories = get_top_memories(
        episodes
    )

    summary_text = generate_summary_text(

        scene_stats,

        event_stats,

        entity_stats,

        len(episodes)
    )

    summary_data = {

        "total_episodes":

        len(episodes),

        "scene_statistics":

        scene_stats,

        "event_statistics":

        event_stats,

        "entity_statistics":

        entity_stats,

        "top_memories":

        top_memories,

        "summary_text":

        summary_text
    }

    save_summary(
        summary_data
    )

    print("\nSUMMARY")
    print("-" * 50)
    print(summary_text)
    print("-" * 50)

    print(
        "\n[SUCCESS] MEMORY SUMMARY COMPLETE"
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
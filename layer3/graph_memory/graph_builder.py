"""
=========================================================
NOONGIL-X Layer 3
Graph Builder
=========================================================

Purpose:
---------
Creates Context Graph Memory from:

1. detected_entities.json
2. detected_relations.json
3. detected_events.json

Output:
--------
context_graph.json

=========================================================
"""

import json
import os
from pathlib import Path

import networkx as nx


# =========================================================
# PATH CONFIGURATION
# =========================================================

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
# OUTPUT FILE
# ============================================================

OUTPUT_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "context_graph.json"
)

# =========================================================
# LOAD JSON
# =========================================================

def load_json(file_path):
    file_path = Path(file_path)
    print(f"\n[INFO] Loading: {file_path.name}")

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

        print("[ERROR] Failed Loading File")
        print(e)

        return None


# =========================================================
# GRAPH CREATION
# =========================================================

def build_graph(
        entity_data,
        relation_data,
        event_data
):

    print("\n[INFO] Creating Graph")

    graph = nx.DiGraph()

    # -----------------------------------------------------
    # ENTITY NODES
    # -----------------------------------------------------

    print("\n[INFO] Adding Entity Nodes")

    for entity in entity_data["entities"]:

        node_name = entity["name"]
        node_type = entity["type"]

        graph.add_node(

            node_name,

            category="entity",

            entity_type=node_type
        )

        print(
            f"[NODE] {node_name}"
            f" ({node_type})"
        )

    # -----------------------------------------------------
    # USER NODE
    # -----------------------------------------------------

    if not graph.has_node("user"):

        graph.add_node(

            "user",

            category="agent"
        )

        print("[NODE] user")

    # -----------------------------------------------------
    # RELATION EDGES
    # -----------------------------------------------------

    print("\n[INFO] Adding Relations")

    for relation in relation_data["relations"]:

        source = relation["source"]

        target = relation["target"]

        relation_type = relation["relation"]

        graph.add_edge(

            source,

            target,

            relation=relation_type
        )

        print(
            f"[EDGE] "
            f"{source} -> "
            f"{relation_type} -> "
            f"{target}"
        )

    # -----------------------------------------------------
    # EVENT NODES
    # -----------------------------------------------------

    print("\n[INFO] Adding Event Nodes")

    for event in event_data["events"]:

        event_id = event["event_id"]

        event_type = event["event_type"]

        graph.add_node(

            event_id,

            category="event",

            event_type=event_type
        )

        print(
            f"[EVENT NODE] "
            f"{event_id}"
        )

        actor = event.get("actor")

        if actor:

            graph.add_edge(

                actor,

                event_id,

                relation="participated_in"
            )

            print(
                f"[EDGE] "
                f"{actor} -> "
                f"participated_in -> "
                f"{event_id}"
            )

    return graph


# =========================================================
# EXPORT GRAPH
# =========================================================

def save_graph(graph):

    print("\n[INFO] Exporting Graph")

    nodes = []

    edges = []

    # -----------------------------------------------------
    # NODES
    # -----------------------------------------------------

    for node, attrs in graph.nodes(data=True):

        nodes.append({

            "id": node,

            **attrs
        })

    # -----------------------------------------------------
    # EDGES
    # -----------------------------------------------------

    for source, target, attrs in graph.edges(data=True):

        edges.append({

            "source": source,

            "target": target,

            **attrs
        })

    graph_json = {

        "node_count":
        len(nodes),

        "edge_count":
        len(edges),

        "nodes":
        nodes,

        "edges":
        edges
    }

    with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
    ) as file:

        json.dump(
            graph_json,
            file,
            indent=4
        )

    print(
        f"[SUCCESS] Saved Graph To:\n"
        f"{OUTPUT_FILE}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n" + "=" * 60)
    print("NOONGIL-X GRAPH BUILDER")
    print("=" * 60)

    entity_data = load_json(
        ENTITY_FILE
    )

    relation_data = load_json(
        RELATION_FILE
    )

    event_data = load_json(
        EVENT_FILE
    )

    if (
        entity_data is None
        or relation_data is None
        or event_data is None
    ):

        print(
            "\n[ERROR] Missing Input Files"
        )

        return

    graph = build_graph(

        entity_data,

        relation_data,

        event_data
    )

    print(
        f"\n[INFO] Total Nodes: "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"[INFO] Total Edges: "
        f"{graph.number_of_edges()}"
    )

    save_graph(graph)

    print(
        "\n[SUCCESS] GRAPH BUILD COMPLETE"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
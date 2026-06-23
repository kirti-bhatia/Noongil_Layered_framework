"""
=========================================================
NOONGIL-X Layer 3
Graph Updater
=========================================================

Purpose:
---------
Updates existing Context Graph Memory.

Input:
-------
context_graph.json
detected_entities.json
detected_relations.json
detected_events.json

Output:
--------
updated_context_graph.json

=========================================================
"""

import json
import os
from pathlib import Path

import networkx as nx


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

GRAPH_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "context_graph.json"
)

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

UPDATED_GRAPH_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "context_graph.json"
)

# =====================================================
# LOAD JSON
# =====================================================

def load_json(path):
    path = Path(path)
    print(f"\n[INFO] Loading {path.name}")

    try:

        with open(
            path,
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
# REBUILD GRAPH FROM JSON
# =====================================================

def rebuild_graph(graph_data):

    print("\n[INFO] Rebuilding Graph")

    graph = nx.DiGraph()

    # -----------------------------
    # Load Nodes
    # -----------------------------

    for node in graph_data["nodes"]:

        node_id = node["id"]

        attributes = node.copy()

        del attributes["id"]

        graph.add_node(
            node_id,
            **attributes
        )

    # -----------------------------
    # Load Edges
    # -----------------------------

    for edge in graph_data["edges"]:

        source = edge["source"]
        target = edge["target"]

        attributes = edge.copy()

        del attributes["source"]
        del attributes["target"]

        graph.add_edge(
            source,
            target,
            **attributes
        )

    print(
        f"[INFO] Nodes Loaded: "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"[INFO] Edges Loaded: "
        f"{graph.number_of_edges()}"
    )

    return graph


# =====================================================
# UPDATE ENTITY NODES
# =====================================================

def update_entities(
        graph,
        entity_data
):

    print("\n[INFO] Updating Entities")

    added = 0

    for entity in entity_data["entities"]:

        name = entity["name"]

        if not graph.has_node(name):

            graph.add_node(

                name,

                category="entity",

                entity_type=
                entity["type"]
            )

            added += 1

            print(
                f"[NEW NODE] {name}"
            )

    print(
        f"[INFO] New Nodes Added: "
        f"{added}"
    )


# =====================================================
# UPDATE RELATIONS
# =====================================================

def update_relations(
        graph,
        relation_data
):

    print("\n[INFO] Updating Relations")

    added = 0

    for relation in relation_data["relations"]:

        source = relation["source"]

        target = relation["target"]

        rel_type = relation["relation"]

        if not graph.has_edge(
                source,
                target
        ):

            graph.add_edge(

                source,

                target,

                relation=rel_type
            )

            added += 1

            print(

                f"[NEW EDGE] "

                f"{source} -> "

                f"{rel_type} -> "

                f"{target}"
            )

    print(
        f"[INFO] New Edges Added: "
        f"{added}"
    )


# =====================================================
# UPDATE EVENTS
# =====================================================

def update_events(
        graph,
        event_data
):

    print("\n[INFO] Updating Events")

    added = 0

    for event in event_data["events"]:

        event_id = event["event_id"]

        if not graph.has_node(event_id):

            graph.add_node(

                event_id,

                category="event",

                event_type=
                event["event_type"]
            )

            added += 1

            print(
                f"[NEW EVENT] "
                f"{event_id}"
            )

            actor = event.get(
                "actor"
            )

            if actor:

                graph.add_edge(

                    actor,

                    event_id,

                    relation=
                    "participated_in"
                )

    print(
        f"[INFO] New Events Added: "
        f"{added}"
    )


# =====================================================
# SAVE GRAPH
# =====================================================

def save_graph(graph):

    print("\n[INFO] Saving Updated Graph")

    nodes = []

    edges = []

    # -----------------------------
    # Export Nodes
    # -----------------------------

    for node, attrs in graph.nodes(data=True):

        nodes.append({

            "id": node,

            **attrs
        })

    # -----------------------------
    # Export Edges
    # -----------------------------

    for source, target, attrs in graph.edges(data=True):

        edges.append({

            "source": source,

            "target": target,

            **attrs
        })

    output = {

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
        UPDATED_GRAPH_FILE,
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
        f"{UPDATED_GRAPH_FILE}"
    )


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n" + "=" * 60)
    print("NOONGIL-X GRAPH UPDATER")
    print("=" * 60)

    graph_data = load_json(
        GRAPH_FILE
    )

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
        graph_data is None
        or entity_data is None
        or relation_data is None
        or event_data is None
    ):

        print(
            "\n[ERROR] Missing Files"
        )

        return

    graph = rebuild_graph(
        graph_data
    )

    update_entities(
        graph,
        entity_data
    )

    update_relations(
        graph,
        relation_data
    )

    update_events(
        graph,
        event_data
    )

    print(
        f"\n[INFO] Final Nodes: "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"[INFO] Final Edges: "
        f"{graph.number_of_edges()}"
    )

    save_graph(graph)

    print(
        "\n[SUCCESS] GRAPH UPDATE COMPLETE"
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
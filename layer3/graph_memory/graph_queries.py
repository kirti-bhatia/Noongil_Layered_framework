"""
=========================================================
NOONGIL-X Layer 3
Graph Queries
=========================================================

Purpose:
---------
Query the Context Graph Memory

Input:
--------
output/context_graph.json

=========================================================
"""

import json
import os
from pathlib import Path

import networkx as nx


# ============================================================
# PATH CONFIGURATION
# ============================================================

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
# FILES
# ============================================================

GRAPH_FILE = os.path.join(
    LAYER3_OUTPUT_DIR,
    "context_graph.json"
)


# =====================================================
# LOAD GRAPH
# =====================================================

def load_graph():
    
    print("\n[INFO] Loading Context Graph")

    try:

        with open(
            GRAPH_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            graph_data = json.load(file)

        print("[SUCCESS] Graph Loaded")

    except Exception as e:

        print("[ERROR] Failed Loading Graph")
        print(e)

        return None

    graph = nx.DiGraph()

    # --------------------------------------------
    # Load Nodes
    # --------------------------------------------

    for node in graph_data["nodes"]:

        node_id = node["id"]

        attrs = node.copy()

        del attrs["id"]

        graph.add_node(
            node_id,
            **attrs
        )

    # --------------------------------------------
    # Load Edges
    # --------------------------------------------

    for edge in graph_data["edges"]:

        source = edge["source"]
        target = edge["target"]

        attrs = edge.copy()

        del attrs["source"]
        del attrs["target"]

        graph.add_edge(
            source,
            target,
            **attrs
        )

    print(
        f"[INFO] Nodes: "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"[INFO] Edges: "
        f"{graph.number_of_edges()}"
    )

    return graph


# =====================================================
# QUERY 1
# USER LOCATION
# =====================================================

def get_user_location(graph):

    print("\n[QUERY] User Location")

    for _, target, attrs in graph.out_edges(
            "user",
            data=True
    ):

        if attrs.get(
                "relation"
        ) == "located_in":

            print(
                f"[RESULT] "
                f"User is in: {target}"
            )

            return target

    print(
        "[RESULT] Location Not Found"
    )

    return None


# =====================================================
# QUERY 2
# USER ACTIVITIES
# =====================================================

def get_user_activities(graph):

    print("\n[QUERY] User Activities")

    activities = []

    for _, target, attrs in graph.out_edges(
            "user",
            data=True
    ):

        if attrs.get(
                "relation"
        ) == "performing":

            activities.append(target)

    print(
        f"[RESULT] {activities}"
    )

    return activities


# =====================================================
# QUERY 3
# EVENTS
# =====================================================

def get_all_events(graph):

    print("\n[QUERY] Events")

    events = []

    for node, attrs in graph.nodes(
            data=True
    ):

        if attrs.get(
                "category"
        ) == "event":

            events.append({

                "event_id": node,

                "event_type":
                attrs.get(
                    "event_type"
                )
            })

    print(
        f"[RESULT] "
        f"{len(events)} Events Found"
    )

    return events


# =====================================================
# QUERY 4
# ENTITY CONNECTIONS
# =====================================================

def get_connected_entities(
        graph,
        entity_name
):

    print(
        f"\n[QUERY] "
        f"Connections For {entity_name}"
    )

    if not graph.has_node(
            entity_name
    ):

        print(
            "[RESULT] Entity Not Found"
        )

        return []

    connections = []

    for neighbor in graph.neighbors(
            entity_name
    ):

        connections.append(
            neighbor
        )

    print(
        f"[RESULT] "
        f"{connections}"
    )

    return connections


# =====================================================
# QUERY 5
# NAVIGATION REQUESTS
# =====================================================

def get_navigation_events(graph):

    print(
        "\n[QUERY] Navigation Events"
    )

    navigation_events = []

    for node, attrs in graph.nodes(
            data=True
    ):

        if (
            attrs.get("category")
            == "event"
            and
            attrs.get("event_type")
            == "navigation_request"
        ):

            navigation_events.append(
                node
            )

    print(
        f"[RESULT] "
        f"{navigation_events}"
    )

    return navigation_events


# =====================================================
# QUERY 6
# ALL NODES
# =====================================================

def get_all_nodes(graph):

    print("\n[QUERY] All Nodes")

    nodes = list(
        graph.nodes()
    )

    print(nodes)

    return nodes


# =====================================================
# QUERY 7
# ALL RELATIONS
# =====================================================

def get_all_relations(graph):

    print("\n[QUERY] All Relations")

    relations = []

    for source, target, attrs in graph.edges(
            data=True
    ):

        relations.append({

            "source": source,

            "relation":
            attrs.get(
                "relation"
            ),

            "target": target
        })

    print(
        f"[RESULT] "
        f"{len(relations)} Relations"
    )

    return relations


# =====================================================
# MAIN
# =====================================================

def main():

    print("\n" + "=" * 60)
    print("NOONGIL-X GRAPH QUERY ENGINE")
    print("=" * 60)

    graph = load_graph()

    if graph is None:

        return

    # ----------------------------------------
    # Run Example Queries
    # ----------------------------------------

    get_user_location(graph)

    get_user_activities(graph)

    get_all_events(graph)

    # get_connected_entities(
    #     graph,
    #     "park"
    # )
    location = get_user_location(graph)

    if location:
        get_connected_entities(
        graph,
        location
    )

    get_navigation_events(graph)

    get_all_nodes(graph)

    get_all_relations(graph)

    print(
        "\n[SUCCESS] QUERY TEST COMPLETE"
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
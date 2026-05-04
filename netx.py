import math

import networkx as nx


def build_radial_ring_network(
    spoke_count: int = 8,
    ring_radii: tuple[float, ...] = (5, 10, 15),
) -> nx.Graph:
    """Build an abstract radial-ring network with weighted spoke and ring edges."""
    if spoke_count < 3:
        raise ValueError("spoke_count must be at least 3")
    if not ring_radii:
        raise ValueError("ring_radii must contain at least one radius")

    radii = tuple(float(radius) for radius in ring_radii)
    if any(radius <= 0 for radius in radii):
        raise ValueError("all ring radii must be positive")
    if tuple(sorted(radii)) != radii or len(set(radii)) != len(radii):
        raise ValueError("ring_radii must be sorted in strictly increasing order")

    graph = nx.Graph()
    graph.add_node(
        "hub",
        radius=0.0,
        spoke_index=None,
        angle=None,
        pos=(0.0, 0.0),
    )

    for spoke_index in range(spoke_count):
        angle = 2 * math.pi * spoke_index / spoke_count
        previous_node = "hub"
        previous_radius = 0.0

        for radius in radii:
            node = (radius, spoke_index)
            graph.add_node(
                node,
                radius=radius,
                spoke_index=spoke_index,
                angle=angle,
                pos=(radius * math.cos(angle), radius * math.sin(angle)),
            )
            graph.add_edge(
                previous_node,
                node,
                kind="spoke",
                weight=radius - previous_radius,
            )
            previous_node = node
            previous_radius = radius

    for radius in radii:
        ring_weight = 2 * math.pi * radius / spoke_count
        for spoke_index in range(spoke_count):
            graph.add_edge(
                (radius, spoke_index),
                (radius, (spoke_index + 1) % spoke_count),
                kind="ring",
                weight=ring_weight,
            )

    return graph


if __name__ == "__main__":
    radial_ring = build_radial_ring_network()
    hub_to_outer = nx.shortest_path_length(
        radial_ring,
        "hub",
        (15.0, 3),
        weight="weight",
    )

    print(f"nodes: {radial_ring.number_of_nodes()}")
    print(f"edges: {radial_ring.number_of_edges()}")
    print(f"connected: {nx.is_connected(radial_ring)}")
    print(f"hub_to_outer_distance: {hub_to_outer:g}")

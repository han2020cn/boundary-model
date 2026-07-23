import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


import networkx as nx
import helpers.cal_route as cr
from helpers.types import NetworkNode


@dataclass(frozen=True, slots=True)
class NetworkRoute:
    route_id: str
    stops: tuple[NetworkNode, ...]


@dataclass(frozen=True, slots=True)
class NetworkContext:
    network_type: str
    graph: nx.Graph
    hub: NetworkNode
    routes: tuple[NetworkRoute, ...]
    request_nodes: tuple[NetworkNode, ...]


def build_grid_graph(net) -> nx.Graph:
    graph = nx.grid_2d_graph(net.grid, net.grid)
    nx.set_edge_attributes(graph, net.grid_len, "weight")
    nx.set_node_attributes(
        graph,
        {
            node: (float(node[0]), float(node[1]))
            for node in graph.nodes
        },
        "pos",
    )
    return graph

"""Build an abstract radial-ring network with weighted spoke and ring edges."""
def build_radial_ring_graph(
    spoke_count: int = 8,
    ring_radial: tuple[float, ...] = (5, 10, 15),
) -> nx.Graph:
    if spoke_count < 3:
        raise ValueError("spoke_count must be at least 3")
    if not ring_radial:
        raise ValueError("ring_radii must contain at least one radius")

    radii = tuple(float(radius) for radius in ring_radial)
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

"""Build graph and route configuration from the user-editable Nets class."""
def build_network_context(nets) -> NetworkContext:
    network_type = getattr(nets, "_type")
    if network_type == "grid":
        graph = build_grid_graph(nets)
        route_specs = cr.routes(nets)
        if route_specs is None:
            route_specs = getattr(nets, "routes", None)
        if route_specs is None:
            route_specs = (getattr(nets, "fixed_stops"),)
        request_nodes = tuple(graph.nodes)
    elif network_type == "hub_spoke":
        graph = build_radial_ring_graph(nets.spoke_count, nets.ring_radial)
        route_specs = getattr(nets, "hub_spoke_routes", None)
        if route_specs is None:
            route_specs = _legacy_hub_spoke_routes(nets)
        allow_hub_requests = bool(getattr(nets, "allow_hub_requests", False))
        request_nodes = tuple(
            node for node in graph.nodes if allow_hub_requests or node != nets.hub
        )
    else:
        raise ValueError("network_type must be 'grid' or 'hub_spoke'")

    routes = _build_routes(route_specs)
    context = NetworkContext(
        network_type=network_type,
        graph=graph,
        hub=nets.hub,
        routes=routes,
        request_nodes=tuple(request_nodes),
    )
    _validate_network_context(context)
    return context


def _legacy_hub_spoke_routes(nets) -> tuple[tuple[NetworkNode, ...], ...]:
    routes = []
    index = 1
    while hasattr(nets, f"fixed_stops_{index}"):
        routes.append(tuple(getattr(nets, f"fixed_stops_{index}")))
        index += 1
    if not routes:
        raise ValueError("hub_spoke network requires hub_spoke_routes")
    return tuple(routes)


def _build_routes(route_specs) -> tuple[NetworkRoute, ...]:
    routes = []
    for index, route_spec in enumerate(route_specs, start=1):
        if isinstance(route_spec, NetworkRoute):
            route = route_spec
        else:
            route = NetworkRoute(
                route_id=f"route_{index}",
                stops=tuple(route_spec),
            )
        if len(route.stops) < 2:
            raise ValueError(f"{route.route_id} must contain at least two stops")
        routes.append(route)
    if not routes:
        raise ValueError("network context requires at least one route")
    return tuple(routes)


def _validate_network_context(context: NetworkContext) -> None:
    missing = []
    if context.hub not in context.graph:
        missing.append(("hub", context.hub))

    for route in context.routes:
        for stop in route.stops:
            if stop not in context.graph:
                missing.append((route.route_id, stop))

    for node in context.request_nodes:
        if node not in context.graph:
            missing.append(("request_nodes", node))

    if missing:
        details = ", ".join(f"{owner}: {node!r}" for owner, node in missing[:10])
        if len(missing) > 10:
            details += f", ... {len(missing) - 10} more"
        raise ValueError(
            f"{context.network_type} network contains nodes that are not in graph: "
            f"{details}"
        )


def print_network_summary(graph: nx.Graph) -> None:
    """Print a readable summary of a radial-ring network."""
    edge_kind_counts = Counter(
        data.get("kind", "unknown")
        for _, _, data in graph.edges(data=True)
    )

    print(f"nodes: {graph.number_of_nodes()}")
    print(f"edges: {graph.number_of_edges()}")
    print(f"connected: {nx.is_connected(graph)}")
    print("edge kinds:")
    for kind, count in sorted(edge_kind_counts.items()):
        print(f"  {kind}: {count}")

    print("nodes:")
    for node, data in sorted(graph.nodes(data=True), key=_node_sort_key):
        radius = data.get("radius")
        spoke_index = data.get("spoke_index")
        pos = data.get("pos")
        print(
            f"  {node}: "
            f"radius={_format_number(radius)}, "
            f"spoke_index={spoke_index}, "
            f"pos={_format_position(pos)}"
        )

    print("edges:")
    for source, target, data in sorted(graph.edges(data=True), key=_edge_sort_key):
        kind = data.get("kind", "unknown")
        weight = data.get("weight")
        print(
            f"  {source} -- {target}: "
            f"kind={kind}, "
            f"weight={_format_number(weight)}"
        )


def print_graph_contents(
    graph: nx.Graph,
    name: str = "graph",
    max_items: int | None = None,
) -> None:
    """Print graph type, nodes, edges, and stored attributes."""
    print(f"{name}:")
    print(f"  type: {type(graph).__name__}")
    print(f"  nodes: {graph.number_of_nodes()}")
    print(f"  edges: {graph.number_of_edges()}")
    print(f"  connected: {nx.is_connected(graph)}")

    node_items = list(graph.nodes(data=True))
    edge_items = list(graph.edges(data=True))
    if max_items is not None:
        node_items = node_items[:max_items]
        edge_items = edge_items[:max_items]

    print("  node data:")
    for node, data in node_items:
        print(f"    {node}: {data}")
    if max_items is not None and graph.number_of_nodes() > max_items:
        print(f"    ... {graph.number_of_nodes() - max_items} more nodes")

    print("  edge data:")
    for source, target, data in edge_items:
        print(f"    {source} -- {target}: {data}")
    if max_items is not None and graph.number_of_edges() > max_items:
        print(f"    ... {graph.number_of_edges() - max_items} more edges")


def print_grid_and_radial_ring_graphs(net, choice

) -> None:
    """Build and print both supported graph structures."""
    if choice == 'radial' :
        radial_ring_graph = build_radial_ring_graph(net.spoke_count, net.ring_radial)
        print_graph_contents(radial_ring_graph, "radial-ring graph")
        print()
    else:
        grid_graph = build_grid_graph(net)
        radial_ring_graph = build_radial_ring_graph(net.spoke_count, net.ring_radial)
        print_graph_contents(grid_graph, "grid graph")
        print()
    




def plot_radial_ring_network(
    graph: nx.Graph,
    output_path: str | Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot a radial-ring network using node positions stored on the graph."""
    import matplotlib.pyplot as plt

    pos = nx.get_node_attributes(graph, "pos")
    if len(pos) != graph.number_of_nodes():
        missing_nodes = [node for node in graph.nodes if node not in pos]
        raise ValueError(f"all nodes must have a 'pos' attribute; missing: {missing_nodes}")

    spoke_edges = [
        (source, target)
        for source, target, data in graph.edges(data=True)
        if data.get("kind") == "spoke"
    ]
    ring_edges = [
        (source, target)
        for source, target, data in graph.edges(data=True)
        if data.get("kind") == "ring"
    ]
    other_edges = [
        (source, target)
        for source, target, data in graph.edges(data=True)
        if data.get("kind") not in {"spoke", "ring"}
    ]

    fig, ax = plt.subplots(figsize=(9, 9))
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=spoke_edges,
        edge_color="tab:blue",
        width=2.2,
        alpha=0.85,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=ring_edges,
        edge_color="tab:orange",
        width=1.8,
        alpha=0.85,
        ax=ax,
    )
    if other_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=other_edges,
            edge_color="tab:gray",
            width=1.4,
            alpha=0.75,
            ax=ax,
        )

    non_hub_nodes = [node for node in graph.nodes if node != "hub"]
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=non_hub_nodes,
        node_color="white",
        edgecolors="tab:blue",
        linewidths=1.4,
        node_size=280,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=["hub"] if "hub" in graph else [],
        node_color="tab:red",
        edgecolors="black",
        linewidths=1.5,
        node_size=520,
        ax=ax,
    )

    labels = {node: _node_label(node) for node in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)

    ax.set_title("Radial-Ring Network")
    ax.set_aspect("equal")
    ax.axis("off")

    saved_path = None
    if output_path is not None:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


def _format_number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_position(pos: object) -> str:
    if not isinstance(pos, tuple) or len(pos) != 2:
        return str(pos)
    return f"({_format_number(pos[0])}, {_format_number(pos[1])})"


def _node_label(node: object) -> str:
    if node == "hub":
        return "hub"
    if isinstance(node, tuple) and len(node) == 2:
        return f"({_format_number(node[0])}, {node[1]})"
    return str(node)


def _node_sort_key(item: tuple[object, dict]) -> tuple[float, int]:
    node, data = item
    radius = data.get("radius", math.inf)
    spoke_index = data.get("spoke_index")
    return (float(radius), -1 if spoke_index is None else int(spoke_index))


def _edge_sort_key(item: tuple[object, object, dict]) -> tuple[str, tuple[float, int], tuple[float, int]]:
    source, target, data = item
    return (
        str(data.get("kind", "")),
        _single_node_sort_key(source),
        _single_node_sort_key(target),
    )


def _single_node_sort_key(node: object) -> tuple[float, int]:
    if node == "hub":
        return (0.0, -1)
    if isinstance(node, tuple) and len(node) == 2:
        return (float(node[0]), int(node[1]))
    return (math.inf, math.inf)


if __name__ == "__main__":
    radial_ring = build_radial_ring_graph()
    hub_to_outer = nx.shortest_path_length(
        radial_ring,
        "hub",
        (15.0, 3),
        weight="weight",
    )

    print_network_summary(radial_ring)
    print(f"hub_to_outer_distance: {hub_to_outer:g}")
    plot_radial_ring_network(radial_ring, "radial_ring_network.png")

from __future__ import annotations

import math
from typing import Any

import networkx as nx


def required_grid_fleet_num(nets: Any, network_context: Any, fleet: Any) -> int:
    """Return the fixed-route fleet size implied by grid edge length and headway."""
    _validate_grid_inputs(nets, fleet)

    return sum(
        required_grid_route_vehicle_count(
            nets,
            _route_edge_count(route, network_context.graph),
            fleet,
        )
        for route in network_context.routes
    )


def required_grid_route_vehicle_count(
    nets: Any,
    route_edge_count: float,
    fleet: Any,
) -> int:
    _validate_grid_inputs(nets, fleet)
    route_distance_km = float(route_edge_count) * _grid_edge_km(nets)
    route_cycle_time_min = route_distance_km / float(fleet.speed)
    return max(1, int(math.ceil(route_cycle_time_min / float(fleet.freq))))


def _route_edge_count(route: Any, graph: nx.Graph) -> int:
    stops = tuple(route.stops)
    edge_count = 0
    for index, stop in enumerate(stops):
        next_stop = stops[(index + 1) % len(stops)]
        segment = nx.shortest_path(graph, stop, next_stop, weight="weight")
        edge_count += len(segment) - 1
    return edge_count


def _grid_edge_km(nets: Any) -> float:
    return float(nets.grid_len) / 1000.0


def _validate_grid_inputs(nets: Any, fleet: Any) -> None:
    if getattr(nets, "_type", None) != "grid":
        raise ValueError("grid fleet sizing only supports Grid networks")
    if float(nets.grid_len) <= 0:
        raise ValueError("nets.grid_len must be positive")
    if float(fleet.speed) <= 0:
        raise ValueError("fleet.speed must be positive")
    if float(fleet.freq) <= 0:
        raise ValueError("fleet.freq must be positive")

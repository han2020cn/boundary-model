from __future__ import annotations

from collections import defaultdict
from typing import Any
import math

import networkx as nx

from helpers.config import LoopContext, TripRequest
from helpers.types import CandidateConstraint, GridNode


DISTANCE_CACHE_KEY = "_boundary_model_shortest_path_lengths"
PhysicalTripKey = tuple[str, int, int]
LoopLoadKey = tuple[PhysicalTripKey, int]


def distance_shortpath(
    a: GridNode,
    b: GridNode,
    net_graph: nx.Graph,
) -> float:
    distance_cache = net_graph.graph.get(DISTANCE_CACHE_KEY)
    if distance_cache is None:
        distance_cache = {
            source: dict(lengths)
            for source, lengths in nx.all_pairs_dijkstra_path_length(
                net_graph,
                weight="weight",
            )
        }
        net_graph.graph[DISTANCE_CACHE_KEY] = distance_cache

    return float(distance_cache[a][b])


def _build_loop_capacity_constraint(
    runtime_fleet: Any,
    loads: defaultdict[LoopLoadKey, int],
    route_length: int,
) -> CandidateConstraint:
    def constraint(candidate: dict[str, Any]) -> str | None:
        trip_key = (
            str(candidate["loop_id"]),
            int(candidate["vehicle_id"]),
            int(candidate["departure_index"]),
        )
        if _check_loop_capacity(
            runtime_fleet,
            loads,
            trip_key,
            int(candidate["boarding_anchor"]),
            int(candidate["alighting_anchor"]),
            route_length,
        ):
            return None
        return "capacity_limit"

    return constraint


def _calculate_travel_weighted(
    start_offset: float,
    end_offset: float,
    route_length: float,
) -> float:
    delta = (float(end_offset) - float(start_offset)) % float(route_length)
    return delta if delta > 0.0 else float(route_length)


def _scheduled_pass_candidates(
    config,
    fleet,
    loop: LoopContext,
    stop_offset_d: float,
    earliest_time: float,
    vehicle_delay: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    bus_v = fleet.speed
    headway = float(fleet.freq)
    span = float(config.span)
    departure_count = int(math.floor(span / headway)) + 1

    vehicle_ids = tuple(sorted(loop.vehicle_offsets))
    candidates: list[dict[str, Any]] = []
    if not vehicle_ids:
        return candidates

    for departure_index in range(departure_count):
        vehicle_id = vehicle_ids[departure_index % len(vehicle_ids)]
        delay = 0.0 if vehicle_delay is None else float(vehicle_delay[vehicle_id])
        route_departure_time = float(departure_index) * fleet.freq
        pass_time = route_departure_time + float(stop_offset_d) / bus_v + delay / bus_v
        if pass_time + 1e-9 >= float(earliest_time):
            candidates.append(
                {
                    "vehicle_id": vehicle_id,
                    "departure_index": departure_index,
                    "route_departure_time": route_departure_time,
                    "pass_time": float(pass_time),
                }
            )
    return candidates


def next_assigned_departure_time(
    config,
    fleet,
    loop: LoopContext,
    departure_index: int,
) -> float:
    vehicle_count = len(loop.vehicle_offsets)
    if vehicle_count <= 0:
        return float(config.span)

    headway = float(fleet.freq)
    departure_count = int(math.ceil(float(config.span) / headway)) + 1
    next_departure_index = int(departure_index) + vehicle_count
    if next_departure_index < departure_count:
        return float(next_departure_index) * float(fleet.freq)
    return float(config.span)


def _node_sort_key(node: GridNode) -> tuple[str, str]:
    return (type(node).__name__, repr(node))


def _nearest_loop_for_request(
    request: TripRequest,
    loops: tuple[LoopContext, ...],
    graph: nx.Graph,
) -> LoopContext:
    def route_distance(loop: LoopContext) -> tuple[float, str]:
        stops = tuple(loop.fixed_stop_indices)
        origin_distance = min(distance_shortpath(request.origin, stop, graph) for stop in stops)
        destination_distance = min(
            distance_shortpath(request.destination, stop, graph)
            for stop in stops
        )
        return (float(origin_distance + destination_distance), loop.id)

    return min(loops, key=route_distance)


def _calculate_travel_index(start_index: int, end_index: int, cycle_length: int) -> int:
    delta = (end_index - start_index) % cycle_length
    return delta if delta > 0 else cycle_length


def _check_loop_capacity(
    fleet,
    loads: defaultdict[LoopLoadKey, int],
    trip_key: PhysicalTripKey,
    boarding_index: int,
    alighting_index: int,
    route_length: int,
) -> bool:
    travel_time = _calculate_travel_index(boarding_index, alighting_index, route_length)
    for step in range(travel_time):
        edge_index = (boarding_index + step) % route_length
        if loads[(trip_key, edge_index)] >= fleet.cap:
            return False
    return True


def _reserve_loop_capacity(
    loads: defaultdict[LoopLoadKey, int],
    trip_key: PhysicalTripKey,
    boarding_index: int,
    alighting_index: int,
    route_length: int,
) -> None:
    travel_time = _calculate_travel_index(boarding_index, alighting_index, route_length)
    for step in range(travel_time):
        edge_index = (boarding_index + step) % route_length
        loads[(trip_key, edge_index)] += 1

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any
import math

import networkx as nx

from demand_generation import TripRequest

DISTANCE_CACHE_KEY = "_boundary_model_shortest_path_lengths"
GridNode = tuple[int, int]
Scenario = dict[str, Any]

FLEET_SIZE = 7
VEHICLE_CAPACITY = 30
HUB: GridNode = (4, 4)
FIXED_STOPS: tuple[GridNode, ...] = (
    (1, 1),
    (1, 4),
    (1, 7),
    (4, 7),
    (7, 7),
    (7, 4),
    (7, 1),
    (4, 1),
)
MODE_NAMES = {
    1: "fixed_route",
    2: "deviated_route",
    3: "drt_rolling_horizon",
    4: "hub_and_spoke",
}
SPOKE_ORDER = ("north", "east", "south", "west")
RESULT_COLUMNS = [
    "scenario_id",
    "lambda",
    "hs",
    "ht",
    "seed",
    "mode_id",
    "mode_name",
    "feasible",
    "feasibility_reason",
    "total_requests",
    "served_requests",
    "unserved_requests",
    "benchmark_expenditure",
    "net_expenditure",
    "total_wait",
    "total_walk",
    "total_onboard",
    "total_service_time",
    "avg_wait",
    "avg_walk",
    "avg_onboard",
    "avg_service_time",
]


@dataclass(frozen=True, slots=True)
class LoopContext:
    route_nodes: tuple[GridNode, ...]
    route_length: int
    fixed_stop_indices: dict[GridNode, int]
    optional_stops: tuple[GridNode, ...]
    optional_anchor_indices: dict[GridNode, int]
    vehicle_offsets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SpokeVehicle:
    vehicle_id: int
    spoke_name: str
    first_departure: int


def build_grid_graph(grid_size: int = 9) -> nx.Graph:
    graph = nx.grid_2d_graph(grid_size, grid_size)
    nx.set_edge_attributes(graph, 1, "weight")
    return graph


def manhattan_distance(
    a: GridNode,
    b: GridNode,
    graph: nx.Graph | None = None,
) -> int:
    if graph is None:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    distance_cache = graph.graph.get(DISTANCE_CACHE_KEY)
    if distance_cache is None:
        distance_cache = {
            source: dict(lengths)
            for source, lengths in nx.all_pairs_dijkstra_path_length(
                graph,
                weight="weight",
            )
        }
        graph.graph[DISTANCE_CACHE_KEY] = distance_cache

    return int(distance_cache[a][b])


def evaluate_mode_1( # 评估固定路线模式
    requests: list[TripRequest],
    scenario: Scenario,
    graph: nx.Graph,
) -> dict[str, Any]:
    loop = _build_loop_context(graph)
    loads: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    vehicle_completion = {vehicle_id: 0.0 for vehicle_id in range(FLEET_SIZE)}
    served_requests = 0
    total_wait = 0.0
    total_walk = 0.0
    total_onboard = 0.0

    for request in _sorted_requests(requests):
        best_choice: dict[str, Any] | None = None

        for boarding_stop, boarding_index in loop.fixed_stop_indices.items():
            origin_walk = manhattan_distance(request.origin, boarding_stop, graph)
            for alighting_stop, alighting_index in loop.fixed_stop_indices.items():
                destination_walk = manhattan_distance(request.destination, alighting_stop, graph)
                walk_time = float(origin_walk + destination_walk)
                onboard_time = float(
                    _circular_travel_time(
                        boarding_index,
                        alighting_index,
                        loop.route_length,
                    )
                )

                for vehicle_id, offset in enumerate(loop.vehicle_offsets):
                    first_pass = (boarding_index - offset) % loop.route_length
                    boarding_time = _next_cyclic_pass(
                        request.departure_time,
                        first_pass,
                        loop.route_length,
                    )
                    wait_time = float(boarding_time - request.departure_time)
                    if not _check_loop_capacity(
                        loads,
                        vehicle_id,
                        boarding_time,
                        boarding_index,
                        alighting_index,
                        loop.route_length,
                    ):
                        continue

                    operator_finish = float(boarding_time + onboard_time)
                    ranking = (
                        wait_time + walk_time + onboard_time,
                        wait_time,
                        walk_time,
                        onboard_time,
                        vehicle_id,
                        boarding_index,
                        alighting_index,
                    )
                    if best_choice is None or ranking < best_choice["ranking"]:
                        best_choice = {
                            "vehicle_id": vehicle_id,
                            "boarding_index": boarding_index,
                            "alighting_index": alighting_index,
                            "boarding_time": boarding_time,
                            "operator_finish": operator_finish,
                            "wait_time": wait_time,
                            "walk_time": walk_time,
                            "onboard_time": onboard_time,
                            "ranking": ranking,
                        }

        if best_choice is None:
            result = _finalize_result(
                mode_id=1,
                scenario=scenario,
                total_requests=len(requests),
                served_requests=served_requests,
                benchmark_expenditure=None,
                net_expenditure=sum(vehicle_completion.values()),
                total_wait=total_wait,
                total_walk=total_walk,
                total_onboard=total_onboard,
                feasible=False,
                feasibility_reason="capacity_limit",
            )
            result["benchmark_expenditure"] = None
            return result

        _reserve_loop_capacity(
            loads,
            best_choice["vehicle_id"],
            best_choice["boarding_time"],
            best_choice["boarding_index"],
            best_choice["alighting_index"],
            loop.route_length,
        )
        vehicle_id = int(best_choice["vehicle_id"])
        vehicle_completion[vehicle_id] = max(
            vehicle_completion[vehicle_id],
            float(best_choice["operator_finish"]),
        )
        served_requests += 1
        total_wait += float(best_choice["wait_time"])
        total_walk += float(best_choice["walk_time"])
        total_onboard += float(best_choice["onboard_time"])

    net_expenditure = sum(vehicle_completion.values())
    result = _finalize_result(
        mode_id=1,
        scenario=scenario,
        total_requests=len(requests),
        served_requests=served_requests,
        benchmark_expenditure=net_expenditure,
        net_expenditure=net_expenditure,
        total_wait=total_wait,
        total_walk=total_walk,
        total_onboard=total_onboard,
        feasible=True,
        feasibility_reason="feasible",
    )
    result["benchmark_expenditure"] = result["net_expenditure"]
    return result


def evaluate_mode_2( # 评估偏离路线模式 deviated route
    requests: list[TripRequest],
    scenario: Scenario,
    graph: nx.Graph,
    benchmark_expenditure: float | None,
) -> dict[str, Any]:
    if benchmark_expenditure is None:
        return _finalize_result(
            mode_id=2,
            scenario=scenario,
            total_requests=len(requests),
            served_requests=0,
            benchmark_expenditure=None,
            net_expenditure=0.0,
            total_wait=0.0,
            total_walk=0.0,
            total_onboard=0.0,
            feasible=False,
            feasibility_reason="benchmark_mode_infeasible",
        )

    loop = _build_loop_context(graph)
    candidate_locations = _build_mode_2_locations(loop)
    loads: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    vehicle_completion = {vehicle_id: 0.0 for vehicle_id in range(FLEET_SIZE)}
    vehicle_delay = {vehicle_id: 0 for vehicle_id in range(FLEET_SIZE)}
    completion_sum = 0.0
    served_requests = 0
    total_wait = 0.0
    total_walk = 0.0
    total_onboard = 0.0

    for request in _sorted_requests(requests):
        best_choice: dict[str, Any] | None = None
        benchmark_blocked = False

        for boarding_location in candidate_locations:
            boarding_node = boarding_location["node"]
            boarding_anchor = int(boarding_location["anchor_index"])
            boarding_optional = bool(boarding_location["optional"])
            origin_walk = manhattan_distance(request.origin, boarding_node, graph)

            for alighting_location in candidate_locations:
                alighting_node = alighting_location["node"]
                alighting_anchor = int(alighting_location["anchor_index"])
                alighting_optional = bool(alighting_location["optional"])
                destination_walk = manhattan_distance(request.destination, alighting_node, graph)
                walk_time = float(origin_walk + destination_walk)
                base_route_time = _circular_travel_time(
                    boarding_anchor,
                    alighting_anchor,
                    loop.route_length,
                )
                passenger_onboard = float(
                    base_route_time
                    + int(boarding_optional)
                    + int(alighting_optional)
                )
                optional_count = int(boarding_optional) + int(alighting_optional)
                route_entry_offset = 2 if boarding_optional else 0
                passenger_board_offset = 1 if boarding_optional else 0
                operator_finish_offset = route_entry_offset + base_route_time + (
                    2 if alighting_optional else 0
                )

                for vehicle_id, offset in enumerate(loop.vehicle_offsets):
                    delayed_first_pass = (
                        (boarding_anchor - offset) % loop.route_length
                    ) + vehicle_delay[vehicle_id]
                    anchor_time = _next_cyclic_pass(
                        request.departure_time,
                        delayed_first_pass,
                        loop.route_length,
                    )
                    passenger_board_time = anchor_time + passenger_board_offset
                    route_start_time = anchor_time + route_entry_offset
                    wait_time = float(passenger_board_time - request.departure_time)

                    if not _check_loop_capacity(
                        loads,
                        vehicle_id,
                        route_start_time,
                        boarding_anchor,
                        alighting_anchor,
                        loop.route_length,
                    ):
                        continue

                    candidate_completion = max(
                        vehicle_completion[vehicle_id],
                        float(anchor_time + operator_finish_offset),
                    )
                    candidate_expenditure = (
                        completion_sum
                        - vehicle_completion[vehicle_id]
                        + candidate_completion
                    )
                    if candidate_expenditure > benchmark_expenditure + 1e-9:
                        benchmark_blocked = True
                        continue

                    ranking = (
                        wait_time + walk_time + passenger_onboard,
                        wait_time,
                        walk_time,
                        passenger_onboard,
                        optional_count,
                        vehicle_id,
                        boarding_anchor,
                        alighting_anchor,
                        boarding_node,
                        alighting_node,
                    )
                    if best_choice is None or ranking < best_choice["ranking"]:
                        best_choice = {
                            "vehicle_id": vehicle_id,
                            "boarding_anchor": boarding_anchor,
                            "alighting_anchor": alighting_anchor,
                            "route_start_time": route_start_time,
                            "candidate_completion": candidate_completion,
                            "candidate_expenditure": candidate_expenditure,
                            "added_delay": 2 * optional_count,
                            "wait_time": wait_time,
                            "walk_time": walk_time,
                            "onboard_time": passenger_onboard,
                            "ranking": ranking,
                        }

        if best_choice is None:
            return _finalize_result(
                mode_id=2,
                scenario=scenario,
                total_requests=len(requests),
                served_requests=served_requests,
                benchmark_expenditure=benchmark_expenditure,
                net_expenditure=completion_sum,
                total_wait=total_wait,
                total_walk=total_walk,
                total_onboard=total_onboard,
                feasible=False,
                feasibility_reason="benchmark_exceeded" if benchmark_blocked else "capacity_limit",
            )

        _reserve_loop_capacity(
            loads,
            int(best_choice["vehicle_id"]),
            int(best_choice["route_start_time"]),
            int(best_choice["boarding_anchor"]),
            int(best_choice["alighting_anchor"]),
            loop.route_length,
        )
        vehicle_id = int(best_choice["vehicle_id"])
        vehicle_completion[vehicle_id] = float(best_choice["candidate_completion"])
        vehicle_delay[vehicle_id] += int(best_choice["added_delay"])
        completion_sum = float(best_choice["candidate_expenditure"])
        served_requests += 1
        total_wait += float(best_choice["wait_time"])
        total_walk += float(best_choice["walk_time"])
        total_onboard += float(best_choice["onboard_time"])

    return _finalize_result(
        mode_id=2,
        scenario=scenario,
        total_requests=len(requests),
        served_requests=served_requests,
        benchmark_expenditure=benchmark_expenditure,
        net_expenditure=completion_sum,
        total_wait=total_wait,
        total_walk=total_walk,
        total_onboard=total_onboard,
        feasible=True,
        feasibility_reason="feasible",
    )


def evaluate_mode_3( # 评估动态路线模式 DRT rolling horizon
    requests: list[TripRequest],
    scenario: Scenario,
    graph: nx.Graph,
    benchmark_expenditure: float | None,
) -> dict[str, Any]:
    if benchmark_expenditure is None:
        return _finalize_result(
            mode_id=3,
            scenario=scenario,
            total_requests=len(requests),
            served_requests=0,
            benchmark_expenditure=None,
            net_expenditure=0.0,
            total_wait=0.0,
            total_walk=0.0,
            total_onboard=0.0,
            feasible=False,
            feasibility_reason="benchmark_mode_infeasible",
        )

    vehicle_schedules: dict[int, list[TripRequest]] = {
        vehicle_id: [] for vehicle_id in range(FLEET_SIZE)
    }
    vehicle_evaluations = {
        vehicle_id: _evaluate_drt_schedule([], graph) for vehicle_id in range(FLEET_SIZE)
    }
    assigned_request_ids: set[int] = set()
    current_active_travel = 0.0
    lookahead = 20
    step = 10
    max_departure = max((request.departure_time for request in requests), default=0)
    planning_time = 0

    while planning_time <= max_departure + lookahead and len(assigned_request_ids) < len(requests):
        visible_requests = [
            request
            for request in _sorted_requests(requests)
            if request.request_id not in assigned_request_ids
            and request.departure_time <= planning_time + lookahead
        ]

        for request in visible_requests:
            if request.request_id in assigned_request_ids:
                continue

            best_insertion: dict[str, Any] | None = None
            for vehicle_id in range(FLEET_SIZE):
                schedule = vehicle_schedules[vehicle_id]
                for position in range(len(schedule) + 1):
                    candidate_schedule = schedule[:position] + [request] + schedule[position:]
                    candidate_evaluation = _evaluate_drt_schedule(candidate_schedule, graph)
                    candidate_active_travel = (
                        current_active_travel
                        - vehicle_evaluations[vehicle_id]["active_travel"]
                        + candidate_evaluation["active_travel"]
                    )
                    if candidate_active_travel > benchmark_expenditure + 1e-9:
                        continue

                    inserted_assignment = candidate_evaluation["assignments"][position]
                    ranking = (
                        inserted_assignment["wait"],
                        inserted_assignment["onboard"],
                        candidate_active_travel - current_active_travel,
                        vehicle_id,
                        position,
                    )
                    if best_insertion is None or ranking < best_insertion["ranking"]:
                        best_insertion = {
                            "vehicle_id": vehicle_id,
                            "schedule": candidate_schedule,
                            "evaluation": candidate_evaluation,
                            "candidate_active_travel": candidate_active_travel,
                            "ranking": ranking,
                        }

            if best_insertion is None:
                return _finalize_result(
                    mode_id=3,
                    scenario=scenario,
                    total_requests=len(requests),
                    served_requests=len(assigned_request_ids),
                    benchmark_expenditure=benchmark_expenditure,
                    net_expenditure=current_active_travel,
                    total_wait=_sum_assignment_metric(vehicle_evaluations, "wait"),
                    total_walk=0.0,
                    total_onboard=_sum_assignment_metric(vehicle_evaluations, "onboard"),
                    feasible=False,
                    feasibility_reason="insertion_failed",
                )

            vehicle_id = int(best_insertion["vehicle_id"])
            vehicle_schedules[vehicle_id] = list(best_insertion["schedule"])
            vehicle_evaluations[vehicle_id] = dict(best_insertion["evaluation"])
            current_active_travel = float(best_insertion["candidate_active_travel"])
            assigned_request_ids.add(request.request_id)

        planning_time += step

    if len(assigned_request_ids) != len(requests):
        return _finalize_result(
            mode_id=3,
            scenario=scenario,
            total_requests=len(requests),
            served_requests=len(assigned_request_ids),
            benchmark_expenditure=benchmark_expenditure,
            net_expenditure=current_active_travel,
            total_wait=_sum_assignment_metric(vehicle_evaluations, "wait"),
            total_walk=0.0,
            total_onboard=_sum_assignment_metric(vehicle_evaluations, "onboard"),
            feasible=False,
            feasibility_reason="unassigned_requests",
        )

    total_wait = _sum_assignment_metric(vehicle_evaluations, "wait")
    total_onboard = _sum_assignment_metric(vehicle_evaluations, "onboard")
    return _finalize_result(
        mode_id=3,
        scenario=scenario,
        total_requests=len(requests),
        served_requests=len(requests),
        benchmark_expenditure=benchmark_expenditure,
        net_expenditure=current_active_travel,
        total_wait=total_wait,
        total_walk=0.0,
        total_onboard=total_onboard,
        feasible=True,
        feasibility_reason="feasible",
    )


def evaluate_mode_4( # 评估枢纽辐射模式 hub-and-spoke
    requests: list[TripRequest],
    scenario: Scenario,
    graph: nx.Graph,
    benchmark_expenditure: float | None,
) -> dict[str, Any]:
    if benchmark_expenditure is None:
        return _finalize_result(
            mode_id=4,
            scenario=scenario,
            total_requests=len(requests),
            served_requests=0,
            benchmark_expenditure=None,
            net_expenditure=0.0,
            total_wait=0.0,
            total_walk=0.0,
            total_onboard=0.0,
            feasible=False,
            feasibility_reason="benchmark_mode_infeasible",
        )

    spoke_paths = _build_spoke_paths(graph)
    spoke_stops = _build_spoke_stop_list(spoke_paths)
    dispatches = _build_spoke_dispatches()
    loads: defaultdict[tuple[int, GridNode, GridNode, int], int] = defaultdict(int)
    vehicle_completion = {vehicle_id: 0.0 for vehicle_id in range(FLEET_SIZE)}
    completion_sum = 0.0
    served_requests = 0
    total_wait = 0.0
    total_walk = 0.0
    total_onboard = 0.0

    for request in _sorted_requests(requests):
        origin_stop = _nearest_spoke_stop(request.origin, spoke_stops, graph)
        destination_stop = _nearest_spoke_stop(request.destination, spoke_stops, graph)
        walk_time = float(
            manhattan_distance(request.origin, origin_stop, graph)
            + manhattan_distance(request.destination, destination_stop, graph)
        )

        inbound_leg = _select_inbound_leg(
            origin_stop,
            request.departure_time,
            graph,
            dispatches,
            loads,
        )
        if inbound_leg is None:
            return _finalize_result(
                mode_id=4,
                scenario=scenario,
                total_requests=len(requests),
                served_requests=served_requests,
                benchmark_expenditure=benchmark_expenditure,
                net_expenditure=completion_sum,
                total_wait=total_wait,
                total_walk=total_walk,
                total_onboard=total_onboard,
                feasible=False,
                feasibility_reason="capacity_limit",
            )

        outbound_leg = _select_outbound_leg(
            destination_stop,
            int(inbound_leg["arrival_time"]),
            graph,
            dispatches,
            loads,
        )
        if outbound_leg is None:
            return _finalize_result(
                mode_id=4,
                scenario=scenario,
                total_requests=len(requests),
                served_requests=served_requests,
                benchmark_expenditure=benchmark_expenditure,
                net_expenditure=completion_sum,
                total_wait=total_wait,
                total_walk=total_walk,
                total_onboard=total_onboard,
                feasible=False,
                feasibility_reason="capacity_limit",
            )

        updated_completion = dict(vehicle_completion)
        candidate_completion_sum = completion_sum
        for leg in (inbound_leg, outbound_leg):
            vehicle_id = leg.get("vehicle_id")
            if vehicle_id is None:
                continue
            cycle_finish = float(leg["cycle_finish"])
            previous = updated_completion[int(vehicle_id)]
            if cycle_finish > previous:
                candidate_completion_sum += cycle_finish - previous
                updated_completion[int(vehicle_id)] = cycle_finish

        if candidate_completion_sum > benchmark_expenditure + 1e-9:
            return _finalize_result(
                mode_id=4,
                scenario=scenario,
                total_requests=len(requests),
                served_requests=served_requests,
                benchmark_expenditure=benchmark_expenditure,
                net_expenditure=completion_sum,
                total_wait=total_wait,
                total_walk=total_walk,
                total_onboard=total_onboard,
                feasible=False,
                feasibility_reason="benchmark_exceeded",
            )

        for leg in (inbound_leg, outbound_leg):
            if leg.get("vehicle_id") is None:
                continue
            _reserve_path_capacity(
                loads,
                int(leg["vehicle_id"]),
                list(leg["path"]),
                int(leg["start_time"]),
            )

        vehicle_completion = updated_completion
        completion_sum = candidate_completion_sum
        served_requests += 1
        total_wait += float(inbound_leg["wait_time"]) + float(outbound_leg["wait_time"])
        total_walk += walk_time
        total_onboard += float(inbound_leg["onboard_time"]) + float(outbound_leg["onboard_time"])

    return _finalize_result(
        mode_id=4,
        scenario=scenario,
        total_requests=len(requests),
        served_requests=served_requests,
        benchmark_expenditure=benchmark_expenditure,
        net_expenditure=completion_sum,
        total_wait=total_wait,
        total_walk=total_walk,
        total_onboard=total_onboard,
        feasible=True,
        feasibility_reason="feasible",
    )


def _sorted_requests(requests: list[TripRequest]) -> list[TripRequest]:
    return sorted(requests, key=lambda request: (request.departure_time, request.request_id))


def _expand_route(
    graph: nx.Graph,
    ordered_stops: tuple[GridNode, ...],
) -> tuple[GridNode, ...]:
    route_nodes: list[GridNode] = []
    for index, stop in enumerate(ordered_stops):
        next_stop = ordered_stops[(index + 1) % len(ordered_stops)]
        segment = nx.shortest_path(graph, stop, next_stop, weight="weight")
        if not route_nodes:
            route_nodes.extend(segment)
        else:
            route_nodes.extend(segment[1:])
    return tuple(route_nodes)


def _build_loop_context(graph: nx.Graph) -> LoopContext:
    route_nodes = _expand_route(graph, FIXED_STOPS)
    route_positions = {node: index for index, node in enumerate(route_nodes[:-1])}
    optional_anchor_indices: dict[GridNode, int] = {}
    route_set = set(route_nodes[:-1])

    for node in route_nodes[:-1]:
        anchor_index = route_positions[node]
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor in route_set:
                continue
            current_index = optional_anchor_indices.get(neighbor)
            if current_index is None or anchor_index < current_index:
                optional_anchor_indices[neighbor] = anchor_index

    route_length = len(route_nodes) - 1
    return LoopContext(
        route_nodes=route_nodes,
        route_length=route_length,
        fixed_stop_indices={stop: route_positions[stop] for stop in FIXED_STOPS},
        optional_stops=tuple(sorted(optional_anchor_indices)),
        optional_anchor_indices=optional_anchor_indices,
        vehicle_offsets=tuple((vehicle_id * route_length) // FLEET_SIZE for vehicle_id in range(FLEET_SIZE)),
    )


def _build_mode_2_locations(loop: LoopContext) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    for stop, anchor_index in loop.fixed_stop_indices.items():
        locations.append(
            {
                "node": stop,
                "anchor_index": anchor_index,
                "optional": False,
            }
        )
    for stop in loop.optional_stops:
        locations.append(
            {
                "node": stop,
                "anchor_index": loop.optional_anchor_indices[stop],
                "optional": True,
            }
        )
    return locations


def _next_cyclic_pass(earliest_time: int, first_pass: int, cycle_length: int) -> int:
    if earliest_time <= first_pass:
        return first_pass
    cycles_needed = math.ceil((earliest_time - first_pass) / cycle_length)
    return first_pass + cycles_needed * cycle_length


def _circular_travel_time(start_index: int, end_index: int, cycle_length: int) -> int:
    delta = (end_index - start_index) % cycle_length
    return delta if delta > 0 else cycle_length


def _check_loop_capacity(
    loads: defaultdict[tuple[int, int, int], int],
    vehicle_id: int,
    route_start_time: int,
    boarding_index: int,
    alighting_index: int,
    route_length: int,
) -> bool:
    travel_time = _circular_travel_time(boarding_index, alighting_index, route_length)
    for step in range(travel_time):
        edge_index = (boarding_index + step) % route_length
        edge_time = route_start_time + step
        if loads[(vehicle_id, edge_index, edge_time)] >= VEHICLE_CAPACITY:
            return False
    return True


def _reserve_loop_capacity(
    loads: defaultdict[tuple[int, int, int], int],
    vehicle_id: int,
    route_start_time: int,
    boarding_index: int,
    alighting_index: int,
    route_length: int,
) -> None:
    travel_time = _circular_travel_time(boarding_index, alighting_index, route_length)
    for step in range(travel_time):
        edge_index = (boarding_index + step) % route_length
        edge_time = route_start_time + step
        loads[(vehicle_id, edge_index, edge_time)] += 1


def _evaluate_drt_schedule(
    scheduled_requests: list[TripRequest],
    graph: nx.Graph,
) -> dict[str, Any]:
    assignments: list[dict[str, Any]] = []
    current_time = 0
    current_location = HUB
    active_travel = 0.0

    for request in scheduled_requests:
        deadhead = manhattan_distance(current_location, request.origin, graph)
        arrival_at_origin = current_time + deadhead
        pickup_time = max(request.departure_time, arrival_at_origin)
        onboard_time = manhattan_distance(request.origin, request.destination, graph)
        dropoff_time = pickup_time + onboard_time
        assignments.append(
            {
                "request_id": request.request_id,
                "pickup_time": float(pickup_time),
                "dropoff_time": float(dropoff_time),
                "wait": float(pickup_time - request.departure_time),
                "onboard": float(onboard_time),
            }
        )
        active_travel += float(deadhead + onboard_time)
        current_time = dropoff_time
        current_location = request.destination

    return {
        "assignments": assignments,
        "completion_time": float(current_time),
        "active_travel": float(active_travel),
    }


def _sum_assignment_metric(
    vehicle_evaluations: dict[int, dict[str, Any]],
    metric: str,
) -> float:
    total = 0.0
    for evaluation in vehicle_evaluations.values():
        total += sum(float(assignment[metric]) for assignment in evaluation["assignments"])
    return total


def _build_spoke_paths(graph: nx.Graph) -> dict[str, tuple[GridNode, ...]]:
    edge_nodes = {
        "north": (HUB[0], 8),
        "east": (8, HUB[1]),
        "south": (HUB[0], 0),
        "west": (0, HUB[1]),
    }
    return {
        name: tuple(nx.shortest_path(graph, HUB, edge_node, weight="weight"))
        for name, edge_node in edge_nodes.items()
    }


def _build_spoke_stop_list(
    spoke_paths: dict[str, tuple[GridNode, ...]],
) -> tuple[GridNode, ...]:
    all_stops = {HUB}
    for path in spoke_paths.values():
        all_stops.update(path)
    return tuple(sorted(all_stops))


def _build_spoke_dispatches() -> dict[str, list[SpokeVehicle]]:
    dispatches = {name: [] for name in SPOKE_ORDER}
    for vehicle_id in range(FLEET_SIZE):
        dispatch = SpokeVehicle(
            vehicle_id=vehicle_id,
            spoke_name=SPOKE_ORDER[vehicle_id % len(SPOKE_ORDER)],
            first_departure=vehicle_id,
        )
        dispatches[dispatch.spoke_name].append(dispatch)
    return dispatches


def _nearest_spoke_stop(
    point: GridNode,
    spoke_stops: tuple[GridNode, ...],
    graph: nx.Graph,
) -> GridNode:
    return min(
        spoke_stops,
        key=lambda stop: (
            manhattan_distance(point, stop, graph),
            abs(stop[0] - HUB[0]) + abs(stop[1] - HUB[1]),
            stop[0],
            stop[1],
        ),
    )


def _spoke_name_for_stop(stop: GridNode) -> str:
    if stop == HUB:
        return "hub"
    if stop[0] == HUB[0]:
        return "north" if stop[1] > HUB[1] else "south"
    return "east" if stop[0] > HUB[0] else "west"


def _select_inbound_leg(
    stop: GridNode,
    earliest_time: int,
    graph: nx.Graph,
    dispatches: dict[str, list[SpokeVehicle]],
    loads: defaultdict[tuple[int, GridNode, GridNode, int], int],
) -> dict[str, Any] | None:
    if stop == HUB:
        return {
            "vehicle_id": None,
            "path": [HUB],
            "start_time": float(earliest_time),
            "arrival_time": float(earliest_time),
            "wait_time": 0.0,
            "onboard_time": 0.0,
            "cycle_finish": float(earliest_time),
        }

    spoke_name = _spoke_name_for_stop(stop)
    path_to_hub = nx.shortest_path(graph, stop, HUB, weight="weight")
    distance_to_hub = len(path_to_hub) - 1
    best_leg: dict[str, Any] | None = None

    for dispatch in dispatches[spoke_name]:
        first_inbound_pass = dispatch.first_departure + 8 - distance_to_hub
        boarding_time = _next_cyclic_pass(earliest_time, first_inbound_pass, 8)
        if not _check_path_capacity(loads, dispatch.vehicle_id, path_to_hub, boarding_time):
            continue

        arrival_time = boarding_time + distance_to_hub
        ranking = (
            boarding_time - earliest_time,
            arrival_time,
            dispatch.vehicle_id,
        )
        if best_leg is None or ranking < best_leg["ranking"]:
            best_leg = {
                "vehicle_id": dispatch.vehicle_id,
                "path": path_to_hub,
                "start_time": float(boarding_time),
                "arrival_time": float(arrival_time),
                "wait_time": float(boarding_time - earliest_time),
                "onboard_time": float(distance_to_hub),
                "cycle_finish": float(arrival_time),
                "ranking": ranking,
            }

    return best_leg


def _select_outbound_leg(
    stop: GridNode,
    earliest_hub_departure: int,
    graph: nx.Graph,
    dispatches: dict[str, list[SpokeVehicle]],
    loads: defaultdict[tuple[int, GridNode, GridNode, int], int],
) -> dict[str, Any] | None:
    if stop == HUB:
        return {
            "vehicle_id": None,
            "path": [HUB],
            "start_time": float(earliest_hub_departure),
            "arrival_time": float(earliest_hub_departure),
            "wait_time": 0.0,
            "onboard_time": 0.0,
            "cycle_finish": float(earliest_hub_departure),
        }

    spoke_name = _spoke_name_for_stop(stop)
    path_from_hub = nx.shortest_path(graph, HUB, stop, weight="weight")
    distance_from_hub = len(path_from_hub) - 1
    best_leg: dict[str, Any] | None = None

    for dispatch in dispatches[spoke_name]:
        departure_time = _next_cyclic_pass(earliest_hub_departure, dispatch.first_departure, 8)
        if not _check_path_capacity(loads, dispatch.vehicle_id, path_from_hub, departure_time):
            continue

        arrival_time = departure_time + distance_from_hub
        cycle_finish = departure_time + 8
        ranking = (
            departure_time - earliest_hub_departure,
            arrival_time,
            dispatch.vehicle_id,
        )
        if best_leg is None or ranking < best_leg["ranking"]:
            best_leg = {
                "vehicle_id": dispatch.vehicle_id,
                "path": path_from_hub,
                "start_time": float(departure_time),
                "arrival_time": float(arrival_time),
                "wait_time": float(departure_time - earliest_hub_departure),
                "onboard_time": float(distance_from_hub),
                "cycle_finish": float(cycle_finish),
                "ranking": ranking,
            }

    return best_leg


def _check_path_capacity(
    loads: defaultdict[tuple[int, GridNode, GridNode, int], int],
    vehicle_id: int,
    path: list[GridNode],
    start_time: int,
) -> bool:
    for step in range(len(path) - 1):
        key = (vehicle_id, path[step], path[step + 1], start_time + step)
        if loads[key] >= VEHICLE_CAPACITY:
            return False
    return True


def _reserve_path_capacity(
    loads: defaultdict[tuple[int, GridNode, GridNode, int], int],
    vehicle_id: int,
    path: list[GridNode],
    start_time: int,
) -> None:
    for step in range(len(path) - 1):
        key = (vehicle_id, path[step], path[step + 1], start_time + step)
        loads[key] += 1


def _finalize_result(
    mode_id: int,
    scenario: Scenario,
    total_requests: int,
    served_requests: int,
    benchmark_expenditure: float | None,
    net_expenditure: float,
    total_wait: float,
    total_walk: float,
    total_onboard: float,
    feasible: bool, #布尔值，表示方案是否可行
    feasibility_reason: str,
) -> dict[str, Any]:
    denominator = served_requests if served_requests > 0 else 0
    total_service_time = total_wait + total_walk + total_onboard
    avg_wait = total_wait / denominator if denominator else 0.0
    avg_walk = total_walk / denominator if denominator else 0.0
    avg_onboard = total_onboard / denominator if denominator else 0.0
    avg_service_time = total_service_time / denominator if denominator else 0.0

    return {
        "scenario_id": scenario["scenario_id"],
        "lambda": scenario["lambda"],
        "hs": scenario["hs"],
        "ht": scenario["ht"],
        "seed": scenario["seed"],
        "mode_id": mode_id,
        "mode_name": MODE_NAMES[mode_id],
        "feasible": bool(feasible and served_requests == total_requests),
        "feasibility_reason": feasibility_reason,
        "total_requests": int(total_requests),
        "served_requests": int(served_requests),
        "unserved_requests": int(total_requests - served_requests),
        "benchmark_expenditure": _round_metric(benchmark_expenditure),
        "net_expenditure": _round_metric(net_expenditure),
        "total_wait": _round_metric(total_wait),
        "total_walk": _round_metric(total_walk),
        "total_onboard": _round_metric(total_onboard),
        "total_service_time": _round_metric(total_service_time),
        "avg_wait": _round_metric(avg_wait),
        "avg_walk": _round_metric(avg_walk),
        "avg_onboard": _round_metric(avg_onboard),
        "avg_service_time": _round_metric(avg_service_time),
    }


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)
